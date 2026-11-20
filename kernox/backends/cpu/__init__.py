"""``cpu`` backend: native host code (C++ compiled by g++, OpenMP threads).

The kernel is generated per plan (``codegen.py``), compiled for the host CPU
(``-march=native``, ``toolchain.py``) and loaded through ``ctypes``. A launch
runs synchronously on the calling thread plus ``torch.get_num_threads() - 1``
OpenMP threads (the OpenMP runtime is the one Torch already loaded), and
releases the GIL during the computation.

Determinism: the work decomposition (j tiles, j slices) only depends on the
sizes, never on the number of threads; two launches with the same sizes give
bitwise identical results whatever ``torch.set_num_threads``.
"""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import platform
import threading
from functools import lru_cache

from ...plan import Plan, ReductionKind, output_shapes
from ..base import CompileInfo, Sizes, Support, UnsupportedError, Variant
from . import codegen, toolchain

_KMIN_KINDS = codegen.K_KINDS
_ERRLEN = 1024
VARIANT_NAME = "tiled_lanes_jsplit"

# j split: aim for TARGET_TASKS tasks (rows blocks × j slices), each slice
# holding at least MIN_TILES_PER_SLICE tiles; temporary bounded by TEMP_CAP_BYTES.
TARGET_TASKS = 1024
MIN_TILES_PER_SLICE = 8
TEMP_CAP_BYTES = 256 << 20
INT32_CHUNK = (1 << 31) - 1


def _variant(tj: int, bi: int = codegen.ROWS, tasks: int = TARGET_TASKS) -> Variant:
    """``tj`` j tile (multiple of 16), ``bi`` rows per task, ``tasks`` target
    number of tasks for the j split (launch parameter)."""
    return Variant(VARIANT_NAME, tuple(sorted({"tj": tj, "bi": bi, "tasks": tasks}.items())))


def split_j(m: int, n: int, layout, target_tasks: int = TARGET_TASKS, nb: int = 1) -> tuple[int, int]:
    """(S, chunk): number of j slices and slice length (multiple of TJ).

    A function of the sizes only (not of the thread count), so that results
    are reproducible whatever the number of threads."""
    tj = layout.tile_j
    ngi = -(-(nb * m) // layout.rows)
    s = min(max(1, -(-target_tasks // ngi)), max(1, n // (MIN_TILES_PER_SLICE * tj)))
    if layout.partial_bytes:
        s = min(s, max(1, TEMP_CAP_BYTES // max(1, nb * m * layout.partial_bytes)))
    s = max(s, -(-n // (INT32_CHUNK - tj)))  # int32 indices relative to the slice
    if s <= 1:
        return 1, max(n, 1)
    chunk = -(-n // s)
    chunk = -(-chunk // tj) * tj
    s = -(-n // chunk)
    return s, chunk


def _scheme_options(plan: Plan, layout) -> tuple[str, ...]:
    """Contract v0.7: effective summation scheme; ``direct`` and ``block`` share
    the implementation (sum per j tile of ``TJ`` points, published)."""
    if plan.reduction.kind not in codegen.FLOAT_ACC_KINDS:
        return ()
    out = (f"sum_scheme={plan.sum_scheme}",)
    if plan.sum_scheme in ("direct", "block"):
        out += (f"sum_block={layout.tile_j}",)
    return out


def host_platform() -> str | None:
    """None on a supported host (Linux x86-64), otherwise the reason. Other ISAs
    (aarch64 NEON/SVE, ppc64le VSX, RISC-V V) are neither generated for nor tested."""
    import sys

    machine = platform.machine().lower()
    if not sys.platform.startswith("linux") or machine not in ("x86_64", "amd64"):
        return f"cpu backend: Linux x86-64 only (host: {sys.platform}/{machine or '?'})"
    return None


@lru_cache(maxsize=1)
def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _check_device(device):
    import torch

    d = torch.device(device)
    if d.type != "cpu":
        raise UnsupportedError(f"cpu backend: expected a CPU device, got {d}")
    return d


class CompiledCpuReduction:
    def __init__(self, plan: Plan, lib, layout, info: CompileInfo, target_tasks: int = TARGET_TASKS):
        self.plan = plan
        self.layout = layout
        self.target_tasks = target_tasks
        # amendment v0.6 §2: height of work-list tiles = rows of one task
        self.ranges_tile_m = layout.rows
        self.info = info
        self._lib = lib  # never unloaded
        self._fn = lib.kp_launch
        self._fn.restype = ctypes.c_int
        self._fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                             ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                             ctypes.c_char_p, ctypes.c_size_t]
        self._nin = len(layout.in_names)
        self._nout = len(layout.out_names)
        self.last_split = (1, 0, 0)

    def _check(self, inputs: dict, outputs: dict, batch=None) -> tuple[int, int, int]:
        """Cheap revalidation of everything native memory safety depends on
        (same rules as the GPU backends, device ``cpu``)."""
        import torch

        plan = self.plan
        tdt = getattr(torch, plan.dtype)
        dev = torch.device("cpu")
        r = plan.batch_rank
        if bool(r) != (batch is not None):
            raise ValueError(f"plan batch_rank={r}: BatchLayout {'required' if r else 'unexpected'}")
        nb = 1 if batch is None else int(batch.nb)
        m = n = None
        for arg in plan.args:
            t = inputs[arg.name]
            if t.device != dev or t.dtype != tdt:
                raise ValueError(f"input {arg.name}: invalid device/dtype")
            tail = 1 if arg.role == "param" else 2
            if batch is None:
                if not t.is_contiguous():
                    raise ValueError(f"input {arg.name}: not contiguous")
            else:
                if t.dim() not in (tail, tail + r) or (arg.role != "param" and t.dim() != tail + r):
                    raise ValueError(f"input {arg.name}: batch rank {t.dim() - tail} != {r}")
                st = t.stride()
                if (arg.dim > 1 and st[-1] != 1) or (tail == 2 and t.shape[-2] > 1 and st[-2] != arg.dim):
                    raise ValueError(f"input {arg.name}: last two dimensions not contiguous")
                off = batch.host_offsets[arg.name]
                if off.dtype != torch.int64 or tuple(off.shape) != (nb,) or off.device != dev \
                        or not off.is_contiguous():
                    raise ValueError(f"offsets of {arg.name}: expected contiguous int64 ({nb},) on cpu")
                rows = 1 if arg.role == "param" else t.shape[-2]
                if nb and t.numel():
                    lo, hi = int(off.min()), int(off.max())
                    room = t.untyped_storage().nbytes() // t.element_size() - t.storage_offset()
                    if lo < -t.storage_offset() or hi + rows * arg.dim > room:
                        raise ValueError(f"offsets of {arg.name} out of storage bounds")
            shape = tuple(t.shape[-tail:])
            if arg.role == "param":
                if shape != (arg.dim,):
                    raise ValueError(f"param {arg.name}: shape {tuple(t.shape)}, expected ({arg.dim},)")
                continue
            if shape[1] != arg.dim:
                raise ValueError(f"input {arg.name}: shape {tuple(t.shape)}, expected dim {arg.dim}")
            if arg.role == "i":
                if m is not None and shape[0] != m:
                    raise ValueError("i arguments have different lengths")
                m = shape[0]
            else:
                if n is not None and shape[0] != n:
                    raise ValueError("j arguments have different lengths")
                n = shape[0]
        if m is None:
            m = plan.meta.get("m")
            if m is None:
                raise UnsupportedError("plan without i argument: M must be provided through plan.meta['m']")
        n = 0 if n is None else n
        expected = output_shapes(plan, m, () if batch is None else (nb,))
        for name, (shape, dt) in expected.items():
            t = outputs[name]
            if (tuple(t.shape) != tuple(shape) or t.dtype != getattr(torch, dt)
                    or t.device != dev or not t.is_contiguous()):
                raise ValueError(f"output {name}: expected contiguous {shape} {dt} on cpu")
        return m, n, nb

    def _check_ranges(self, ranges, m: int, n: int) -> int:
        import torch

        if not self.plan.ranges:
            raise ValueError("plan without ranges: unexpected RangesLayout")
        if int(ranges.tile_m) != self.ranges_tile_m:
            raise ValueError(f"ranges: tile_m {ranges.tile_m} != {self.ranges_tile_m}")
        if n >= INT32_CHUNK:
            raise UnsupportedError("ranges: N >= 2^31 not supported (int32 j indices)")
        for name, d, w in (("work", ranges.work, 4), ("redranges", ranges.redranges, 2)):
            if (d.device.type != "cpu" or d.dtype != torch.int64 or not d.is_contiguous() or d.dim() != 2
                    or d.shape[1] != w):
                raise ValueError(f"ranges: expected contiguous {name} int64 (·, {w}) on cpu")
        s = ranges.summary
        if s.nw == 0:
            return 0
        if (s.i0_min < 0 or s.i1_max > m or s.h_min < 0 or s.h_max > self.ranges_tile_m
                or s.k0_min < 0 or s.k1_max > s.nr or s.dk_min < 0):
            raise ValueError("ranges: work list out of bounds")
        if s.nr and (s.c_min < 0 or s.d_max > n):
            raise ValueError("ranges: redranges outside [0, N]")
        return s.covered

    def _fill_neutral(self, outputs: dict) -> None:
        kind = self.plan.reduction.kind
        inf = float("inf")
        for name, t in outputs.items():
            if name == "index":
                t.fill_(-1)
            elif kind in (ReductionKind.SUM, ReductionKind.COUNT):
                t.zero_()
            elif kind in (ReductionKind.MIN, ReductionKind.KMIN, ReductionKind.KMIN_ARGKMIN,
                          ReductionKind.MIN_ARGMIN):
                t.fill_(inf)
            elif kind in (ReductionKind.MAX, ReductionKind.LOGSUMEXP, ReductionKind.MAX_ARGMAX,
                          ReductionKind.LOGSUMEXPWEIGHT):
                t.fill_(-inf)
            else:  # SUMSOFTMAXWEIGHT: 0/0
                t.fill_(float("nan"))

    def launch(self, inputs: dict, outputs: dict, batch=None, ranges=None) -> None:
        import torch

        if batch is not None and ranges is not None:
            raise UnsupportedError("ranges combined with batch: rejected (amendment v0.6)")
        if self.plan.ranges and ranges is None:
            raise ValueError("plan ranges=True: RangesLayout required")
        m, n, nb = self._check(inputs, outputs, batch)
        if nb == 0 or m == 0 or (self.plan.reduction.kind in _KMIN_KINDS and self.plan.reduction.k == 0):
            return
        nw = 0
        if ranges is not None:
            covered = self._check_ranges(ranges, m, n)
            if covered < m:
                self._fill_neutral(outputs)
            nw = int(ranges.work.shape[0])
            if nw == 0 or ranges.redranges.shape[0] == 0:
                if covered and ranges.redranges.shape[0] == 0:
                    self._fill_neutral(outputs)
                return
            nsplit, chunk = 1, max(n, 1)
        else:
            nsplit, chunk = split_j(m, n, self.layout, self.target_tasks, nb=max(nb, 1))
        if nsplit > 1 and nb * m * nsplit * self.layout.partial_bytes > TEMP_CAP_BYTES:
            raise UnsupportedError("N >= 2^31 with NB·M too large: temporary exceeds the published bound")
        temp = None
        if nsplit > 1:
            temp = torch.empty(nb * m * nsplit * self.layout.partial_bytes, dtype=torch.uint8)
        ins = (ctypes.c_void_p * self._nin)(*[inputs[k].data_ptr() for k in self.layout.in_names])
        outs = (ctypes.c_void_p * self._nout)(*[outputs[k].data_ptr() for k in self.layout.out_names])
        boff = None
        if batch is not None:
            boff = (ctypes.c_void_p * self._nin)(*[batch.host_offsets[k].data_ptr() for k in self.layout.in_names])
        self.last_split = (nsplit, chunk, 0 if temp is None else temp.numel())
        err = ctypes.create_string_buffer(_ERRLEN)
        rc = self._fn(m, n, ins, outs, nsplit, chunk, None if temp is None else temp.data_ptr(), nb, boff,
                      nw, None if ranges is None else ranges.work.data_ptr(),
                      None if ranges is None else ranges.redranges.data_ptr(),
                      max(1, torch.get_num_threads()), err, _ERRLEN)
        if rc != 0:
            raise RuntimeError(f"kp_launch (cpu) failed [{rc}]: {err.value.decode(errors='replace')}")


class CpuBackend:
    name = "cpu"
    fp_contract_off = True
    flush_denormals = True  # amendment v0.11: FTZ + DAZ (MXCSR) inside the kernel only
    batch = True
    ranges = True
    sum_schemes = ("direct", "block", "kahan")

    def __init__(self):
        self._mem: dict = {}
        self._lock = threading.Lock()

    def supports(self, plan: Plan, device) -> Support:
        try:
            import torch  # noqa: F401
        except ImportError:
            return Support(False, "torch unavailable")
        try:
            _check_device(device)
        except (UnsupportedError, RuntimeError) as e:
            return Support(False, str(e))
        host = host_platform()
        if host is not None:
            return Support(False, host)
        try:
            toolchain.compiler()
        except toolchain.ToolchainError as e:
            return Support(False, str(e))
        if plan.sum_scheme not in self.sum_schemes:
            return Support(False, f"unknown sum_scheme: {plan.sum_scheme!r}")
        if plan.ranges and plan.batch_rank:
            return Support(False, "ranges combined with batch dimensions: rejected (amendment v0.6)")
        rank = {"float16": 0, "float32": 1, "float64": 2}
        if plan.dtype not in rank or plan.acc_dtype not in rank:
            return Support(False, f"dtype {plan.dtype}/{plan.acc_dtype} not supported")
        if rank[plan.acc_dtype] < rank[plan.dtype]:
            return Support(False, "acc_dtype narrower than dtype is not allowed")
        if plan.dtype == "float16":
            if plan.acc_dtype == "float64":
                return Support(False, "dtype float16: acc_dtype float16 or float32 only (amendment v0.9)")
            if not toolchain.float16_supported():
                return Support(False, "dtype float16: the C++ compiler has no _Float16 for this target")
        if plan.acc_dtype != plan.dtype and plan.reduction.kind not in codegen.FLOAT_ACC_KINDS:
            return Support(False, "acc_dtype different from dtype: SUM and LOGSUMEXP family only")
        kind, k = plan.reduction.kind, plan.reduction.k
        if kind in (ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT) and plan.out_dim < 2:
            return Support(False, f"{kind.value}: outputs = (f, g_1..g_E) with E >= 1")
        if kind in _KMIN_KINDS:
            if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                return Support(False, f"invalid K: {k!r}")
            if plan.out_dim != 1:
                return Support(False, "KMIN/ARGKMIN: scalar formula only")
            if k > 1024:
                return Support(False, "K > 1024 not supported")
        elif k is not None:
            return Support(False, "K given for a reduction without K")
        if plan.out_dim == 0:
            return Support(False, "formula without output")
        if plan.index_dtype != "int64":
            return Support(False, "int64 indices only")
        return Support(True)

    def device_identity(self, device) -> dict:
        """CPU model, resolved ``-march=native`` target and compiler (wisdom key)."""
        arch, flags = toolchain.native_target()
        return {"device_name": cpu_name(), "arch": arch, "target_flags": flags,
                "compiler": toolchain.compiler_version()}

    # ------------------------------------------------------------ variants
    def default_variant(self, plan: Plan, device, sizes: Sizes | None = None) -> Variant:
        """Deterministic, size-independent: j tile from ``codegen.default_tile_j``
        (256 points, less when the j operands of a tile would not fit in L1),
        32 rows per task, 1024 target tasks."""
        return _variant(codegen.default_tile_j(plan))

    def variants(self, plan: Plan, device, sizes: Sizes | None, effort: str) -> list[Variant]:
        if effort not in ("measure", "patient"):
            raise ValueError(f"unknown effort: {effort!r}")
        out = [self.default_variant(plan, device, sizes)]
        if effort == "measure":
            tj = codegen.default_tile_j(plan)
            cands = [_variant(tj=max(tj // 2, 32), bi=16), _variant(tj=tj * 2, bi=32), _variant(tj=tj, bi=64)]
        else:
            cands = [_variant(tj=tj, bi=bi, tasks=ta) for tj in (64, 128, 256, 512) for bi in (8, 16, 32, 64)
                     for ta in (1024, 4096)]
        for v in cands:
            if v not in out:
                out.append(v)
        return out[: 4 if effort == "measure" else 32]

    def compile(self, plan: Plan, device, variant: Variant | None = None) -> CompiledCpuReduction:
        import time

        from ...codegen.pair import pair_function

        sup = self.supports(plan, device)
        if not sup.ok:
            raise UnsupportedError(sup.reason)
        if variant is None:
            variant = self.default_variant(plan, device, None)
        if variant.name != VARIANT_NAME:
            raise UnsupportedError(f"unknown cpu variant: {variant.name!r}")
        prm = dict(variant.params)
        unknown = set(prm) - {"tj", "bi", "tasks"}
        if unknown:
            raise UnsupportedError(f"unknown variant parameters: {sorted(unknown)}")
        tj, bi, tasks = int(prm.get("tj", codegen.default_tile_j(plan))), int(prm.get("bi", codegen.ROWS)), int(
            prm.get("tasks", TARGET_TASKS))
        if tj <= 0 or tj % codegen.LN or tj > 4096 or not 1 <= bi <= 1024 or tasks < 1:
            raise UnsupportedError(f"invalid variant parameters: {prm}")
        t0 = time.perf_counter()
        # fast_math: same IEEE pair function (approximation allowed, not required)
        pair = pair_function(dataclasses.replace(plan, fast_math=False) if plan.fast_math else plan, "hip")
        note = json.dumps(variant.as_json(), sort_keys=True)
        vmath = toolchain.libmvec_functions()
        source, layout = codegen.generate(plan, pair.source, tj=tj, bi=bi, variant_note=note,
                                          simd_math=toolchain.libmvec_declarations(vmath) if vmath else "")
        sha = hashlib.sha256(source.encode()).hexdigest()
        mem_key = (sha, bool(plan.fp_contract))
        with self._lock:
            hit = self._mem.get(mem_key)
        if hit is not None:
            lib, build = hit
            cache_hit, secs = True, 0.0
        else:
            try:
                build = toolchain.build_shared_object(source, bool(plan.fp_contract),
                                                      libs=("-lmvec",) if vmath else ())
            except toolchain.ToolchainError as exc:
                raise UnsupportedError(f"cpu backend: {exc}") from exc
            lib = ctypes.CDLL(str(build.path), mode=ctypes.RTLD_LOCAL)
            with self._lock:
                self._mem[mem_key] = (lib, build)
            cache_hit, secs = build.cache_hit, time.perf_counter() - t0
        arch, _ = toolchain.native_target()
        info = CompileInfo(
            backend="cpu",
            device_name=cpu_name(),
            arch=arch,
            compiler=build.compiler,
            options=build.options + (f"libmvec={'on' if vmath else 'off'}",)
            # fast_math=True allows approximate exp/log without requiring them: the cpu
            # backend keeps the IEEE code (libmvec, <= 4 ulp is already vectorized).
            + (("fast_math=ieee",) if plan.fast_math else ())
            + (("flush_denormals=ftz+daz",) if plan.flush_denormals else ()) + _scheme_options(plan, layout),
            cache_hit=cache_hit,
            compile_seconds=secs,
            temp_bytes_per_launch=TEMP_CAP_BYTES,  # bound; exact value: last_split[2]
            source_sha256=sha,
            variant=variant,
        )
        return CompiledCpuReduction(plan, lib, layout, info, target_tasks=tasks)


_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


def get_backend() -> CpuBackend:
    """Entry point for the ``kernox.runtime`` registry (shared instance)."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = CpuBackend()
        return _INSTANCE


__all__ = ["CpuBackend", "CompiledCpuReduction", "get_backend"]
