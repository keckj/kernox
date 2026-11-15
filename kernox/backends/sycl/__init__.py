"""``sycl`` backend (Intel XPU through Torch XPU), contract v0.

Interop: the kernel is submitted on the ``sycl::queue`` of the current Torch
XPU stream (``torch.xpu.current_stream(device).sycl_queue``), an in-order
queue, with the USM device pointers of the Torch tensors. No host
synchronization, no copy, no global temporary (``temp_bytes_per_launch == 0``).

The native module is loaded through ``ctypes`` after ``import torch``: it shares
the process's single ``libsycl.so.9`` (the one from Torch XPU).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import threading
from ...plan import Plan, ReductionKind, output_shapes
from ..base import CompileInfo, Sizes, Support, UnsupportedError, Variant
from . import codegen, toolchain

_KMIN_KINDS = (ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN)
_ERRLEN = 1024
VARIANT_NAME = "tiled1d_jsplit"


# ``CompileInfo.variant`` now lives in base.py; alias kept.
SyclCompileInfo = CompileInfo


def _variant(wg=None, tj=0, sg=0, groups=None, grf=0, unroll=1, coop=0) -> Variant:
    """Parameters: ``wg`` work-group size, ``tj`` requested j tile (0 =
    heuristic, 32 KiB local budget), ``sg`` forced sub-group size
    (0 = compiler's choice; 16 or 32 on the G31), ``groups`` target number of
    work-groups for splitting the j axis (launch parameter),
    ``grf`` register file size (0 = default, 128, 256), ``unroll``
    unrolling of the loops over the top-K state, ``coop`` cooperative top-K: G
    lanes of a sub-group per row i (0 = one work-item per row)."""
    from .codegen import WORK_GROUP

    params = {"wg": wg or WORK_GROUP, "tj": tj, "sg": sg, "groups": groups or TARGET_GROUPS, "grf": grf,
              "unroll": unroll, "coop": coop}
    return Variant(VARIANT_NAME, tuple(sorted(params.items())))

# j-axis split: aim for at least TARGET_GROUPS work-groups, each slice holding
# at least MIN_TILES_PER_SLICE tiles; temporary bounded by TEMP_CAP_BYTES.
TARGET_GROUPS = 512
MIN_TILES_PER_SLICE = 4
TEMP_CAP_BYTES = 256 << 20
BATCH_MIN_CHUNK = 1024  # minimum j per slice to follow the S of a single element (batch)
INT32_CHUNK = (1 << 31) - 1

# Cooperative top-K (``coop`` variant). G31 measurements:
# G such that ceil(K/G) <= COOP_TARGET_SLOTS (K=32 -> 8, 64 -> 16, 128 -> 16) is
# the best or within 10% of the best in all 22 cases; GRF 128 is enough
# as long as L <= COOP_SLOTS (0 spills measured), including at D = 64.
COOP_K_MIN = 32          # FP32 (measured: at K = 16, one work-item per row is still better)
COOP_K_MIN_FP64 = 16     # FP64 (measured: coop G = 4 at K = 16, ×0.42 to ×0.76)
COOP_TARGET_SLOTS = 4
COOP_SLOTS = 8   # beyond: GRF 256 (spills measured with GRF 128)


def coop_group(k: int) -> int:
    """G (lanes per row, sub-group 16): smallest G such that ceil(K/G) <= COOP_TARGET_SLOTS, G <= 16."""
    g = 2
    while g < 16 and -(-k // g) > COOP_TARGET_SLOTS:
        g *= 2
    return g


def coop_grf(plan: Plan, g: int) -> int:
    """GRF 256 if the per-lane state exceeds COOP_SLOTS ranks, otherwise default (128)."""
    k = plan.reduction.k or 1
    return 256 if -(-k // g) > COOP_SLOTS else 0


def split_j(m: int, n: int, layout, target_groups: int = TARGET_GROUPS, nb: int = 1) -> tuple[int, int]:
    """(S, chunk): number of j slices and length of one slice (multiple of TJ).

    Batch (``nb`` > 1): the group target applies to NB·M rows, but S does not
    drop below that of a single element as long as slices keep at least
    ``BATCH_MIN_CHUNK`` j. Based on NB·M alone, S = 1 as soon as NB·M fills the
    target: j slices too long, native batch 8 to 12% slower than a loop of
    calls when each element fills the card (G31); at the S of a single
    element, small problems pay for the merge (short slices).
    """
    tj = layout.tile_j

    def target_s(rows):
        ngi = -(-rows // layout.rows)
        return min(max(1, -(-target_groups // ngi)), max(1, n // (MIN_TILES_PER_SLICE * tj)))

    s = target_s(nb * m)
    if nb > 1:
        s = max(s, min(target_s(m), max(1, n // BATCH_MIN_CHUNK)))
    if layout.partial_bytes:
        s = min(s, max(1, TEMP_CAP_BYTES // (nb * m * layout.partial_bytes)))
    # the main kernel's state stores indices as int32 relative to the slice
    s = max(s, -(-n // (INT32_CHUNK - tj)))
    if s <= 1:
        return 1, max(n, 1)
    chunk = -(-n // s)
    chunk = -(-chunk // tj) * tj
    s = -(-n // chunk)
    return s, chunk


def _pair(plan: Plan):
    """Core pair function if shipped, otherwise the temporary local adapter."""
    try:
        from ...codegen.pair import pair_function  # core
    except ImportError:
        from ._pair_local import pair_function
    return pair_function(plan, "sycl")


def _scheme_options(plan: Plan, layout) -> tuple[str, ...]:
    """Contract v0.7: publishes the effective summation scheme and block size T.
    ``direct`` and ``block`` share the implementation (sum per j tile)."""
    if plan.reduction.kind not in codegen.FLOAT_ACC_KINDS:
        return ()
    scheme = plan.sum_scheme
    out = (f"sum_scheme={scheme}",)
    if scheme in ("direct", "block"):  # direct: same per-tile sum, T published too
        out += (f"sum_block={layout.tile_j}",)
    return out


def _probe(lib, device_index: int) -> tuple[int, int]:
    """Builds the kernels (driver JIT) before any launch; a driver rejection
    (e.g. "total scratch space exceeds HW supported limit") becomes an
    ``UnsupportedError`` at compile time. Returns the private memory per
    work-item (main, merge), in bytes."""
    import torch

    fn = lib.kp_probe
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    priv = (ctypes.c_int64 * 2)()
    err = ctypes.create_string_buffer(_ERRLEN)
    rc = fn(torch.xpu.current_stream(device_index).sycl_queue, priv, err, _ERRLEN)
    if rc != 0:
        raise UnsupportedError("sycl backend: kernel rejected by the driver: "
                               + err.value.decode(errors="replace"))
    return priv[0], priv[1]


def _device_index(device) -> int:
    import torch

    d = torch.device(device)
    if d.type != "xpu":
        raise UnsupportedError(f"sycl backend: expected an XPU device, got {d}")
    return d.index if d.index is not None else torch.xpu.current_device()


class CompiledSyclReduction:
    def __init__(self, plan: Plan, device_index: int, lib, layout, info: CompileInfo,
                 target_groups: int = TARGET_GROUPS):
        self.target_groups = target_groups
        self.plan = plan
        self.device_index = device_index
        self.layout = layout
        # amendment v0.6 §2: height of work-list tiles = i rows of one work-group
        self.ranges_tile_m = layout.rows
        self.info = info
        self._lib = lib  # never unloaded: kernels may still be in flight
        self._fn = lib.kp_launch
        self._fn.restype = ctypes.c_int
        self._fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                             ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                             ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_char_p, ctypes.c_size_t]
        self._nin = len(layout.in_names)
        self._nout = len(layout.out_names)

    def _check(self, inputs: dict, outputs: dict, batch=None) -> tuple[int, int, int]:
        """Cheap revalidation of everything native memory safety depends on.

        ``batch`` (amendment v0.6): inputs ``(*B, L, d)`` of which only the last
        two dimensions must be contiguous; the extent reached through the
        offset tables (read on the host) is bounded by the storage."""
        import torch

        plan = self.plan
        tdt = getattr(torch, plan.dtype)
        dev = torch.device("xpu", self.device_index)
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
                if off.dtype != torch.int64 or tuple(off.shape) != (nb,):
                    raise ValueError(f"offsets of {arg.name}: expected int64 ({nb},)")
                dv = batch.offsets[arg.name]
                if dv.device != dev or dv.dtype != torch.int64 or not dv.is_contiguous() or tuple(dv.shape) != (nb,):
                    raise ValueError(f"device offsets of {arg.name}: expected contiguous int64 ({nb},) on {dev}")
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
                raise ValueError(f"output {name}: expected contiguous {shape} {dt} on {dev}")
        return m, n, nb

    def _check_ranges(self, ranges, m: int, n: int) -> int:
        """Work list (amendment v0.6 §2): shapes, dtypes, device and bounds
        checked on the host copies (memory safety). Returns the number of
        covered rows (disjoint tiles)."""
        import torch

        if not self.plan.ranges:
            raise ValueError("plan without ranges: unexpected RangesLayout")
        if int(ranges.tile_m) != self.ranges_tile_m:
            raise ValueError(f"ranges: tile_m {ranges.tile_m} != {self.ranges_tile_m}")
        if n >= INT32_CHUNK:
            raise UnsupportedError("ranges: N >= 2^31 not supported (int32 j indices)")
        dev = torch.device("xpu", self.device_index)
        hw, hr = ranges.host_work, ranges.host_redranges
        for name, d, h, w in (("work", ranges.work, hw, 4), ("redranges", ranges.redranges, hr, 2)):
            if (d.device != dev or d.dtype != torch.int64 or not d.is_contiguous() or d.dim() != 2
                    or d.shape[1] != w or tuple(h.shape) != tuple(d.shape) or h.dtype != torch.int64):
                raise ValueError(f"ranges: expected contiguous {name} int64 (·, {w}) on {dev}")
        s = ranges.summary  # computed once per layout (host copies): O(1) per call
        if s.nw == 0:
            return 0
        if (s.i0_min < 0 or s.i1_max > m or s.h_min < 0 or s.h_max > self.ranges_tile_m
                or s.k0_min < 0 or s.k1_max > s.nr or s.dk_min < 0):
            raise ValueError("ranges: work list out of bounds")
        if s.nr and (s.c_min < 0 or s.d_max > n):
            raise ValueError("ranges: redranges outside [0, N]")
        return s.covered

    def _fill_neutral(self, outputs: dict) -> None:
        """Reduction neutral values (rows covered by no tile)."""
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
        if ranges is not None:
            covered = self._check_ranges(ranges, m, n)
            if covered < m:
                self._fill_neutral(outputs)
            nw = int(ranges.work.shape[0])
            if nw == 0 or ranges.redranges.shape[0] == 0:
                if covered and ranges.redranges.shape[0] == 0:
                    self._fill_neutral(outputs)  # no interval: neutral everywhere
                return
        queue = torch.xpu.current_stream(self.device_index).sycl_queue
        ins = (ctypes.c_void_p * self._nin)(*[inputs[k].data_ptr() for k in self.layout.in_names])
        outs = (ctypes.c_void_p * self._nout)(*[outputs[k].data_ptr() for k in self.layout.out_names])
        # batch: j split of a single element, NB·M·S temporary (see split_j);
        # ranges: one work-group per tile, no j split
        if ranges is not None:
            nsplit, chunk = 1, max(n, 1)
        else:
            nsplit, chunk = split_j(m, n, self.layout, self.target_groups, nb=max(nb, 1))
        if nsplit > 1 and nb * m * nsplit * self.layout.partial_bytes > TEMP_CAP_BYTES:
            raise UnsupportedError("N >= 2^31 with NB·M too large: temporary exceeds the published bound")
        temp = None
        if nsplit > 1:
            # temporary allocated by the Torch allocator on the current stream: its
            # release at the end of the function is ordered after the kernels.
            temp = torch.empty(nb * m * nsplit * self.layout.partial_bytes, dtype=torch.uint8,
                               device=torch.device("xpu", self.device_index))
        boff = None
        if batch is not None:
            boff = (ctypes.c_void_p * self._nin)(*[batch.offsets[k].data_ptr() for k in self.layout.in_names])
        self.last_split = (nsplit, chunk, 0 if temp is None else temp.numel())
        err = ctypes.create_string_buffer(_ERRLEN)
        rc = self._fn(queue, m, n, ins, outs, nsplit, chunk,
                      None if temp is None else temp.data_ptr(), nb, boff,
                      0 if ranges is None else nw,
                      None if ranges is None else ranges.work.data_ptr(),
                      None if ranges is None else ranges.redranges.data_ptr(), err, _ERRLEN)
        if rc != 0:
            raise RuntimeError(f"kp_launch (sycl) failed [{rc}]: {err.value.decode(errors='replace')}")


class SyclBackend:
    name = "sycl"
    fp_contract_off = True  # contract v0.3: fp_contract=False supported (-ffp-contract=off)
    batch = True            # amendment v0.6 §1: native batch dimensions (element = grid group)
    ranges = True           # amendment v0.6 §2: work list, one work-group per tile (i0, i1, k0, k1)
    sum_schemes = ("direct", "block", "kahan")  # amendment v0.7 (read by the runtime)

    def __init__(self):
        self._mem: dict = {}
        self._lock = threading.Lock()

    def supports(self, plan: Plan, device) -> Support:
        try:
            import torch
        except ImportError:
            return Support(False, "torch unavailable")
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return Support(False, "torch.xpu unavailable")
        try:
            idx = _device_index(device)
        except UnsupportedError as e:
            return Support(False, str(e))
        if plan.fast_math:
            return Support(False, "fast_math not supported (never enabled by the sycl backend)")
        if plan.sum_scheme not in ("direct", "block", "kahan"):
            return Support(False, f"unknown sum_scheme: {plan.sum_scheme!r}")
        if plan.ranges and plan.batch_rank:
            return Support(False, "ranges combined with batch dimensions: rejected (amendment v0.6)")
        rank = {"float16": 0, "float32": 1, "float64": 2}
        if plan.dtype not in rank or plan.acc_dtype not in rank:
            return Support(False, f"dtype {plan.dtype}/{plan.acc_dtype} not supported")
        if rank[plan.acc_dtype] < rank[plan.dtype]:
            return Support(False, "acc_dtype narrower than dtype is not allowed")
        if plan.dtype == "float16":
            # amendment v0.9: formula in sycl::half (each operation rounded to
            # binary16), float16 or float32 accumulator (float64 rejected)
            if plan.acc_dtype == "float64":
                return Support(False, "dtype float16: acc_dtype float16 or float32 only (amendment v0.9)")
            if not getattr(torch.xpu.get_device_properties(idx), "has_fp16", True):
                return Support(False, "float16 rejected: the device does not declare aspect::fp16")
            from ... import _symox_compat

            if not _symox_compat.sycl_half():
                return Support(False, "dtype float16: symox has no sycl::half printer (SyclDtype::Float16 required)")
        if plan.acc_dtype != plan.dtype and plan.reduction.kind not in codegen.FLOAT_ACC_KINDS:
            return Support(False, "acc_dtype different from dtype: SUM and LOGSUMEXP family only")
        if "float64" in (plan.dtype, plan.acc_dtype):
            if not torch.xpu.get_device_properties(idx).has_fp64:
                return Support(False, "float64 rejected: the device does not declare aspect::fp64")
        kind, k = plan.reduction.kind, plan.reduction.k
        if kind in (ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT) and plan.out_dim < 2:
            return Support(False, f"{kind.value}: outputs = (f, g_1..g_E) with E >= 1")
        if kind in _KMIN_KINDS:
            if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                return Support(False, f"invalid K: {k!r}")
            if plan.out_dim != 1:
                return Support(False, "KMIN/ARGKMIN: scalar formula only (v0)")
            if k > 1024:
                return Support(False, "K > 1024 not supported by the v0 register strategy")
        elif k is not None:
            return Support(False, "K given for a reduction without K")
        if plan.out_dim == 0:
            return Support(False, "formula without output")
        if plan.index_dtype != "int64":
            return Support(False, "int64 indices only")
        return Support(True)

    # ------------------------------------------------------------ variants (v0.2)

    def default_variant(self, plan: Plan, device, sizes: Sizes | None = None) -> Variant:
        """Deterministic, size-independent heuristic (ESTIMATE effort):
        WG 256, j tile bounded by 32 KiB of local memory, sub-group left to the
        compiler, j split aiming at 512 work-groups.

        Top-K with K >= COOP_K_MIN (32; 16 in FP64, COOP_K_MIN_FP64): cooperative variant (sub-group 16, G lanes
        per row, at most COOP_TARGET_SLOTS ranks per lane up to K = 64, without register
        spills). Threshold, G and GRF measured on G31.
        Top-K with 16 < K < COOP_K_MIN: state unrolled in registers + GRF 256 (chosen
        by A/B measurement on G31).
        """
        k = plan.reduction.k or 0
        kmin = COOP_K_MIN_FP64 if plan.dtype == "float64" else COOP_K_MIN
        if plan.reduction.kind in _KMIN_KINDS and k >= kmin:
            if plan.dtype == "float32" and 256 < k <= 512 and self._has_sg32(device):
                # measured: coop G = 32 (sub-group 32, L = 16) ×1.8 to ×2.9 faster than
                # G = 16 at K = 512 in FP32; 200-call stress test without hangs. Not in FP64 (×3
                # slower) nor at K = 1024 in FP32 (massive spills, ×38).
                return _variant(sg=32, coop=32, grf=256)
            return _variant(sg=16, coop=(g := coop_group(k)), grf=coop_grf(plan, g))
        if plan.reduction.kind in _KMIN_KINDS and k > 16:  # 16 < K < COOP_K_MIN
            di = sum(a.dim for a in plan.args_of("i"))
            if (plan.dtype == "float32" and di <= 16) or (plan.dtype == "float64" and di <= 3):
                return _variant(grf=256, unroll=1)
            return _variant(grf=0, unroll=0)
        return _variant()

    @staticmethod
    def _has_sg32(device) -> bool:
        try:
            import torch

            props = torch.xpu.get_device_properties(_device_index(device))
            return 32 in set(int(x) for x in props.sub_group_sizes)
        except Exception:
            return False

    def variants(self, plan: Plan, device, sizes: Sizes | None, effort: str) -> list[Variant]:
        import torch

        idx = _device_index(device)
        props = torch.xpu.get_device_properties(idx)
        max_wg = int(props.max_work_group_size)
        sgs = set(int(x) for x in props.sub_group_sizes)
        size = {"float16": 2, "float32": 4, "float64": 8}[plan.dtype]
        dj = sum(a.dim for a in plan.args_of("j"))
        k = plan.reduction.k or 0
        is_k = plan.reduction.kind in _KMIN_KINDS
        heavy_k = is_k and k > 8
        out = [self.default_variant(plan, device, sizes)]
        if effort not in ("measure", "patient"):
            raise ValueError(f"unknown effort: {effort!r}")
        cands: list[Variant] = []
        if is_k and k >= 16:  # coop candidates from K = 16 (default threshold: 16 in FP64, 32 in FP32)
            g0 = coop_group(k)
            r0 = coop_grf(plan, g0)
            g2 = min(16, g0 * 2)
            gh = max(2, g0 // 2)
            if effort == "measure":  # ≤ 4: coop G/2, 2G, G with the other GRF; registers
                cands += [_variant(sg=16, coop=gh, grf=coop_grf(plan, gh)),
                          _variant(sg=16, coop=g2, grf=coop_grf(plan, g2)),
                          _variant(sg=16, coop=g0, grf=0 if r0 else 256),
                          _variant(grf=256, unroll=1)]
            else:  # ≤ 32
                cands += [_variant(wg=wg, sg=16, coop=g, grf=gr) for g in (2, 4, 8, 16)
                          for wg in (128, 256) for gr in (0, 256)]
                cands += [_variant(sg=16, coop=g, tj=128) for g in (4, 8, 16)]
                # coop G = 32 (sub-group 32): candidate in FP32 with K <= 512 only
                # (measured: K = 1024 FP32 in 11.9 s, ×38; FP64 K = 512 ×3 slower)
                if 32 in sgs and plan.dtype == "float32" and k <= 512:
                    cands += [_variant(wg=256, sg=32, coop=32, grf=256)]
                cands += [_variant(grf=g, unroll=u) for g in (0, 256) for u in (0, 1)]
        else:
            if effort == "measure":
                cands += [_variant(128, 0, 0, 512), _variant(256, 0, 0, 2048)]
                if heavy_k:
                    cands += [_variant(grf=256, unroll=1)]
            else:
                cands += [_variant(wg, tj, sg, 512) for wg in (128, 256, 512) for tj in (128, 256)
                          for sg in (0, 16, 32)]
                cands += [_variant(wg, tj, 0, 2048) for wg in (128, 256, 512) for tj in (128, 256)]
        cap = 4 if effort == "measure" else 32
        for v in cands:
            prm = dict(v.params)
            wg, tj, sg = prm["wg"], prm["tj"], prm["sg"]
            if wg > max_wg or (sg and sg not in sgs):
                continue
            if sg == 32 and heavy_k and not prm["coop"]:  # large top-K state: SIMD32 spills
                continue
            if tj and dj * size * tj > codegen.LOCAL_MAX_BYTES:
                continue
            if v not in out:
                out.append(v)
        return out[:cap]

    def compile(self, plan: Plan, device, variant: Variant | None = None) -> CompiledSyclReduction:
        import time

        import torch

        sup = self.supports(plan, device)
        if not sup.ok:
            raise UnsupportedError(sup.reason)
        if variant is None:
            variant = self.default_variant(plan, device, None)
        if variant.name != VARIANT_NAME:
            raise UnsupportedError(f"unknown sycl variant: {variant.name!r}")
        prm = dict(variant.params)
        unknown = set(prm) - {"wg", "tj", "sg", "groups", "grf", "unroll", "coop"}
        if unknown:
            raise UnsupportedError(f"unknown variant parameters: {sorted(unknown)}")
        wg, tj, sg, groups = int(prm.get("wg", 256)), int(prm.get("tj", 0)), int(prm.get("sg", 0)), int(
            prm.get("groups", TARGET_GROUPS))
        idx = _device_index(device)
        props = torch.xpu.get_device_properties(idx)
        if wg < 16 or wg > int(props.max_work_group_size) or wg % 16:
            raise UnsupportedError(f"invalid work-group size: {wg}")
        if sg and sg not in set(int(x) for x in props.sub_group_sizes):
            raise UnsupportedError(f"sub-group size {sg} not supported by the device")
        grf = int(prm.get("grf", 0))
        if grf not in (0, 128, 256):
            raise UnsupportedError(f"invalid GRF size: {grf}")
        if groups < 1 or tj < 0:
            raise UnsupportedError("invalid variant parameters")
        target = toolchain.target_for_device_id(props.device_id)
        t0 = time.perf_counter()
        pair = _pair(plan)
        note = json.dumps(variant.as_json(), sort_keys=True)
        # The SYCL runtime indexes device images by kernel NAME process-wide:
        # two loaded modules with the same kernel name get mixed up (observed
        # on the G31). The name is therefore derived from the hash of the full
        # kernel source (reduction, K, tiles and variant included).
        unroll = int(prm.get("unroll", 1))
        if unroll not in (0, 1):
            raise UnsupportedError(f"invalid unroll: {unroll}")
        coop = int(prm.get("coop", 0))
        if coop:
            if plan.reduction.kind not in _KMIN_KINDS:
                raise UnsupportedError("cooperative top-K: KMIN/ARGKMIN only")
            if not sg:
                raise UnsupportedError("cooperative top-K: a forced sub-group size (sg) is required")
            if coop not in codegen.COOP_GROUPS or sg % coop or wg % sg:
                raise UnsupportedError(f"invalid cooperative group: G={coop}, sg={sg}, wg={wg}")
        kw = dict(wg=wg, tj=tj or None, sg=sg, grf=grf, unroll=bool(unroll), coop=coop, variant_note=note)
        draft, _ = codegen.generate(plan, pair.source, "TAG", **kw)
        tag = hashlib.sha256(draft.encode()).hexdigest()[:24]
        source, layout = codegen.generate(plan, pair.source, tag, **kw)
        mem_key = (hashlib.sha256(source.encode()).hexdigest(), target, bool(plan.fp_contract))
        with self._lock:
            hit = self._mem.get(mem_key)
        if hit is not None:
            lib, build = hit
            cache_hit, secs = True, 0.0
        else:
            build = toolchain.build_shared_object(source, target, bool(plan.fp_contract))
            lib = ctypes.CDLL(str(build.path), mode=ctypes.RTLD_LOCAL)
            _probe(lib, idx)  # UnsupportedError if the driver rejects the kernels
            with self._lock:
                self._mem[mem_key] = (lib, build)
            cache_hit, secs = build.cache_hit, time.perf_counter() - t0
        info = CompileInfo(
            backend="sycl",
            device_name=f"{props.name} (0x{props.device_id:x})",
            arch=target,
            compiler=build.compiler,
            options=build.options + _scheme_options(plan, layout),
            cache_hit=cache_hit,
            compile_seconds=secs,
            temp_bytes_per_launch=TEMP_CAP_BYTES,  # bound; exact value: last_split[2]
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            variant=variant,
        )
        return CompiledSyclReduction(plan, idx, lib, layout, info, target_groups=groups)


_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


def get_backend() -> SyclBackend:
    """Entry point for the ``kernox.runtime`` registry (shared instance)."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = SyclBackend()
        return _INSTANCE


__all__ = ["SyclBackend", "CompiledSyclReduction", "get_backend"]
