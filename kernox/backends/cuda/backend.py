"""CUDA backend: NVRTC (CUBIN) + CUDA Driver API, Torch interop.

Status: implemented; tested state: ``docs/backends.md``.

Interop:
- primary context of the device (shared with the Torch CUDA runtime);
- launch on ``torch.cuda.current_stream(device)``; no host synchronization;
- temporaries (j partition) allocated with ``torch.empty`` on the current device
  and stream: the Torch caching allocator only reuses them after the work
  already queued on that stream;
- no CPU copy, no CPU fallback.
"""

from __future__ import annotations

import ctypes as ct
import fcntl
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from ..._env import getenv
from ...plan import Plan, ReductionKind as R, output_shapes
from dataclasses import dataclass, replace

from ..base import CompileInfo, Sizes, Support, UnsupportedError, Variant
from . import _driver, _kernel

MAX_K = 256                      # per-thread list possible up to this K (registers)
MAX_K_COOP = 1024                # beyond: UnsupportedError (32 ranks per lane at G = 32)
COOP_K_MIN = 16                  # cooperative top-K from K = 16 (measured)


# Cooperative threshold per architecture (cc major): on Maxwell (sm_52, 980 Ti), K = 16
# costs x1.07 versus the per-thread list; the cooperative path starts at K = 32 there.
COOP_K_MIN_BY_MAJOR = {5: 32}


def coop_group(k: int, cc_major: int | None = None) -> int:
    """G (lanes per row) of the cooperative top-K; 0: per-thread list (K < 16).

    Top-K measurements (2080 Ti, KNN D = 3 20000²): the per-thread list loses from
    K = 32 (4.33 ms versus 2.05 with G = 16, 2.14 with G = 32); K = 64: 10.6 versus 2.8
    (G = 32); K = 256 FP32: 704 versus 10.3 (register spilling); for K >= 64, G = 32
    is the best everywhere (K = 300-1024: x0.48-0.77 versus G = 16).
    K = 16 FP32 1.71 ms (G = 16) versus 2.28 (per thread); K = 8 FP32: the
    per-thread list remains better (1.59 versus 2.26)."""
    if k < COOP_K_MIN_BY_MAJOR.get(cc_major, COOP_K_MIN):
        return 0
    g = 16 if k <= 32 else 32
    if -(-k // g) <= 32:
        return g
    raise UnsupportedError(f"K = {k} > {MAX_K_COOP}")


GRID_Z_MAX = 65535               # batch elements per launch (z dimension of the grid)
INT32_MAX_N = 2**31 - 1         # beyond: variant with int64 local indices
TEMP_CAP_BYTES = 64 << 20        # published bound on temporaries per launch
SMEM_LIMIT = 48 << 10            # dynamic shared memory without opt-in
BLOCK_CANDIDATES = (256, 192, 128, 96, 64, 32)
TILE_CANDIDATES = (512, 384, 256, 192, 128, 96, 64, 32, 16)
# --fmad follows Plan.fp_contract (contract v0.3): added by _options(plan)
BASE_OPTIONS = ("--std=c++17", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")


def _options(plan, arch: str) -> tuple:
    """NVRTC options: FMA contraction according to ``plan.fp_contract`` (default True)."""
    fmad = "--fmad=true" if getattr(plan, "fp_contract", True) else "--fmad=false"
    return BASE_OPTIONS + (fmad, f"--gpu-architecture={arch}")

_DT_BYTES = {"float16": 2, "float32": 4, "float64": 8}
# j partition: target = factor x resident blocks ("waves"); off: S = 1.
# Policy 2 (after measurements): partition as long as the i grid has fewer than
# ``factor`` waves, to limit quantization (e.g. 79 blocks on 68 SMs at
# 1 block/SM = 2 waves, the second one at 16 %).
# j partition: target in waves of resident blocks. "auto": 4 waves, except
# 2 waves for K >= 8 lists when the pair is cheap (FP32, j <= 16 components)
# or when M is very small relative to the GPU (see CudaCompiledReduction.auto_factor).
SPLIT_MODES = {"auto": None, "x2": 2, "x4": 4, "x8": 8, "off": 0}
SPLIT_POLICY = "split-policy-5"
TILE_MIN_SPLIT = 16  # minimum tile when the j partition shrinks the tile


@dataclass(frozen=True)
class CudaCompileInfo(CompileInfo):
    """``CompileInfo`` + effective variant (contract v0.2: ``info.variant``)."""

    variant: Variant | None = None


def _pack(vals):
    """Kernel parameters: each value in an 8-byte slot (little-endian: an ``int``
    parameter reads the first 4 bytes), with the matching pointer array.
    The value array is kept alive by the ``_keep`` reference."""
    k = len(vals)
    buf = (ct.c_uint64 * k)(*vals)
    base = ct.addressof(buf)
    ptrs = (ct.c_void_p * k)(*range(base, base + 8 * k, 8))
    ptrs._keep = buf
    return ptrs


def row_bytes(plan) -> int:
    """Bytes of one j tile row (all j variables)."""
    return sum(a.dim for a in plan.args_of("j")) * _DT_BYTES[plan.dtype]


def staging_for(plan) -> str:
    """``"smem"`` if a tile of TILE_CANDIDATES[-1] rows fits in SMEM_LIMIT,
    otherwise ``"global"`` (direct read of the j rows: large D, e.g. D = 990)."""
    return "smem" if TILE_CANDIDATES[-1] * row_bytes(plan) <= SMEM_LIMIT else "global"


def _make_variant(block, tile, split):
    return Variant("tiled", (("block", int(block)), ("split", str(split)), ("tile", int(tile))))


def _cache_dir() -> Path:
    # KERNOX_CACHE_DIR: directory shared by the core (wisdom); KERNOX_CUDA_CACHE_DIR: CUDA-specific
    d = getenv("KERNOX_CUDA_CACHE_DIR") or getenv("KERNOX_CACHE_DIR")
    root = Path(d) if d else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "kernox"
    return root / "cuda"


def _default_pair_function():
    from ...codegen.pair import pair_function  # provided by the core

    return pair_function, "core"


# Ranges (amendment v0.6 §2): smaller blocks than the dense default (finer work tiles,
# fewer inactive rows at the edge of an i block). Sweep on 2080 Ti, M = N = 10^5,
# 64 clusters, ρ = 0.25: gauss FP32 10.7 -> 8.0 ms (B = 128); KNN FP32 9.5 -> 8.5;
# KNN FP64 143 -> 127 (B = 96); gauss FP64 396 -> 361 (B = 96).
RANGES_BLOCK = {"float16": 128, "float32": 128, "float64": 96}
RANGES_TILE = 128


def _ranges_default(comp):
    b = RANGES_BLOCK[comp.plan.dtype]
    try:
        return comp.with_variant(_make_variant(b, min(RANGES_TILE, 2 * b), "auto"))
    except UnsupportedError:
        return comp


class CudaBackend:
    name = "cuda"
    #: contract v0.3: plan.fp_contract == False -> NVRTC --fmad=false (_options)
    fp_contract_off = True
    #: amendment v0.7: implemented schemes; "block" publishes ``sum_block=T`` (j tile of the variant)
    sum_schemes = ("direct", "block", "kahan")
    #: amendment v0.6 §1: native batch (batch index = blockIdx.z, offset table per argument)
    batch = True
    #: amendment v0.6 §2: native ranges (one block per work tile of height ranges_tile_m)
    ranges = True

    def __init__(self, pair_function=None, coop_group=None):
        #: G forced for the cooperative top-K (measurements); None: ``coop_group(K)`` policy
        self._coop_override = coop_group
        if pair_function is None:
            self._pair, self._pair_origin = _default_pair_function()
        else:
            self._pair, self._pair_origin = pair_function, "custom"
        self._lock = threading.Lock()
        self._memo: dict = {}
        # Fast path: (id(plan), device, local indices) -> (plan, compiled). The runtime
        # calls default_variant/compile on every launch; without this cache, the symox
        # pair and the kernel source were regenerated on every call (~1.5 ms).
        self._fast: dict = {}

    # ------------------------------------------------------------------ support
    def supports(self, plan: Plan, device) -> Support:
        import torch

        dev = torch.device(device)
        if dev.type != "cuda" or torch.version.hip is not None:
            return Support(False, "not an NVIDIA CUDA device")
        if plan.dtype not in _DT_BYTES:
            return Support(False, f"dtype {plan.dtype} is not supported")
        if plan.acc_dtype not in ((plan.dtype, "float32") if plan.dtype == "float16" else (plan.dtype, "float64")):
            return Support(False, f"acc_dtype {plan.acc_dtype} is not supported for dtype {plan.dtype}")
        if plan.acc_dtype != plan.dtype and plan.reduction.kind not in (
                R.SUM, R.LOGSUMEXP, R.SUMSOFTMAXWEIGHT, R.LOGSUMEXPWEIGHT):
            return Support(False, "acc_dtype different from dtype: SUM and the LOGSUMEXP family only")
        if plan.fast_math and plan.dtype == "float16":
            return Support(False, "fast_math in FP16 is not supported by the CUDA backend (amendment v0.10)")
        if plan.ranges and plan.batch_rank:
            return Support(False, "ranges combined with batch: rejected (amendment v0.6)")
        if plan.index_dtype != "int64":
            return Support(False, "index_dtype int64 only")
        if not plan.args_of("j"):
            return Support(False, "at least one j argument is required in v0")
        if plan.out_dim < 1:
            return Support(False, "empty formula")
        k = plan.reduction.kind
        if _kernel.sum_scheme(plan) not in _kernel.SUM_SCHEMES:
            return Support(False, f"sum_scheme {_kernel.sum_scheme(plan)!r} is not supported")
        if k in (R.SUMSOFTMAXWEIGHT, R.LOGSUMEXPWEIGHT) and plan.out_dim < 2:
            return Support(False, f"{k.value}: formula (f, g_1..g_E) with E >= 1 required")
        if k in (R.KMIN, R.ARGKMIN, R.KMIN_ARGKMIN):
            K = plan.reduction.k
            if not isinstance(K, int) or isinstance(K, bool) or K < 0:
                return Support(False, "K must be an integer >= 0")
            if plan.out_dim != 1:
                return Support(False, "KMIN/ARGKMIN: scalar formula only (v0)")
            if K > MAX_K_COOP:
                return Support(False, f"K > {MAX_K_COOP} is not supported")
        return Support(True)

    # ------------------------------------------------------------------ compile
    # ------------------------------------------------------------------ variants (v0.2)
    def default_variant(self, plan: Plan, device, sizes: Sizes) -> Variant:
        """Heuristic (estimate): block/tile maximizing resident threads, automatic
        j partition. Deterministic for a given device."""
        return self.compile(plan, device).variant

    def variants(self, plan: Plan, device, sizes: Sizes, effort: str) -> list:
        """Valid candidates, ``default_variant`` first; <= 4 (measure), <= 32 (patient)."""
        base = self.compile(plan, device)
        if isinstance(base, _EmptyReduction):
            return [base.variant]
        cands = base.feasible()  # [(resident threads, B, tj)] in decreasing order
        out = [base.variant]
        af = base.auto_factor(*base.waves_args(sizes.m, getattr(sizes, "nb", 1) or 1)) if sizes is not None else base.auto_factor()
        if effort == "measure":  # default + the other partition policies (same block/tile)
            other = "x4" if af == 2 else "x2"
            pool = [(base.block, base.tile, other), (base.block, base.tile, "off"), (base.block, base.tile, "x8")]
            pool += [(b, t, "auto") for _, b, t in cands[:3]]
            limit = 4
        elif effort == "patient":
            # two passes: (auto, off) for the 12 best occupancies, then the other
            # partitions for the 4 best (24 + 8 = 32); a product pool of
            # occupancies x 4 modes runs out before reaching small blocks (measured:
            # B = 128 lost for KNN K = 32 FP64, 36.7 -> 46.9 ms)
            pool = ([(b, t, sp) for _, b, t in cands[:12] for sp in ("auto", "off")]
                    + [(b, t, sp) for _, b, t in cands[:4] for sp in ("x2" if af == 4 else "x4", "x8")])
            limit = 32
        else:
            raise ValueError(f"unknown effort {effort!r}")
        for b, t, sp in pool:
            v = _make_variant(b, t, sp)
            if v not in out:
                out.append(v)
            if len(out) >= limit:
                break
        return out

    def compile(self, plan: Plan, device, variant: Variant | None = None, _local_index: str = "int"):
        """Compile (or reload) the kernel; ``variant`` only sets launch parameters
        (block, tile, j partition): the CUBIN is shared between variants, and the
        variant is part of the in-memory cache key.

        int32 local indices if N < 2^31; int64 variant compiled on demand at the
        first launch with N >= 2^31."""
        import torch

        dev = torch.device(device)
        ordinal = dev.index if dev.index is not None else torch.cuda.current_device()
        fast = self._fast.get((id(plan), ordinal, _local_index))
        if fast is not None and fast[0] is plan:
            base = fast[1]
            return base if variant is None else self._with_variant(base, base._memo_key, variant)
        sup = self.supports(plan, dev)
        if not sup.ok:
            raise UnsupportedError(sup.reason)
        K = plan.reduction.k
        if plan.reduction.kind in (R.KMIN, R.ARGKMIN, R.KMIN_ARGKMIN) and K == 0:
            return _EmptyReduction(plan, self._info_stub(ordinal))
        if plan.dtype == "float16":
            # amendment v0.9: pair generated in FP32 then rewritten in kp_h (FP16 rounding per operation)
            p32 = replace(plan, dtype="float32", acc_dtype="float32", sum_scheme="direct")
            pair = self._pair(p32, "cuda")
            pair = replace(pair, source=_kernel.half_pair_source(pair.source)) if hasattr(pair, "__dataclass_fields__") \
                else type("P", (), {"source": _kernel.half_pair_source(pair.source), "name": pair.name})()
        else:
            # amendment v0.10: IEEE pair generated, then FP32 exp/log rewritten as approximate
            pair = self._pair(replace(plan, fast_math=False) if plan.fast_math else plan, "cuda")
            if plan.fast_math and plan.dtype == "float32":
                src = _kernel.fast_pair_source(pair.source)
                pair = replace(pair, source=src) if hasattr(pair, "__dataclass_fields__") \
                    else type("P", (), {"source": src, "name": pair.name})()
        staging = staging_for(plan)
        drv = _driver.driver()
        cdev = drv.device(ordinal)
        cc_major = drv.attribute(cdev, _driver.ATTR_CC_MAJOR)
        coop = 0
        if plan.reduction.kind in (R.KMIN, R.ARGKMIN, R.KMIN_ARGKMIN):
            # forced G (measurements, any K >= 1) or coop_group(K, architecture) policy
            coop = coop_group(K, cc_major) if self._coop_override is None or K < 1 else self._coop_override
        source = _kernel.generate(plan, pair.source, pair.name, local_index=_local_index, staging=staging,
                                  coop=coop)
        source_sha = hashlib.sha256(source.encode()).hexdigest()

        cc = f"{drv.attribute(cdev, _driver.ATTR_CC_MAJOR)}{drv.attribute(cdev, _driver.ATTR_CC_MINOR)}"
        arch = f"sm_{cc}"
        nv = _driver.nvrtc()
        options = _options(plan, arch)
        compiler = f"NVRTC {nv.version[0]}.{nv.version[1]}"
        key = hashlib.sha256(json.dumps(
            [_kernel.GENERATOR_VERSION, source_sha, list(options), compiler, arch]).encode()).hexdigest()
        memo_key = (key, ordinal)
        with self._lock:
            hit = self._memo.get(memo_key)
        if hit is not None:
            with self._lock:
                self._fast[(id(plan), ordinal, _local_index)] = (plan, hit)
            return hit if variant is None else self._with_variant(hit, memo_key, variant)
        t0 = time.perf_counter()
        cubin, cache_hit = self._cubin(nv, source, options, key)
        seconds = time.perf_counter() - t0
        ctx = _driver.primary_context(ordinal)
        module = _driver.Module(ctx, cubin)
        info_opts = options + (f"pair={self._pair_origin}", f"generator={_kernel.GENERATOR_VERSION}",
                               f"local_index={_local_index}", f"staging={staging}", SPLIT_POLICY,
                               f"coop={coop}",
                               # contract v0.7: effective scheme; "block": T = launch j tile
                               # (the variant's "tile" parameter, shrunk under partition: launch_config)
                               f"sum_scheme={_kernel.sum_scheme(plan)}")
        info = CudaCompileInfo(
            backend="cuda", device_name=drv.name(cdev), arch=arch,
            compiler=f"{compiler} ({nv.path}); driver {drv.version}",
            options=info_opts, cache_hit=cache_hit, compile_seconds=seconds,
            temp_bytes_per_launch=TEMP_CAP_BYTES, source_sha256=source_sha,
        )
        wide = None
        if _local_index == "int":
            wide = lambda v: self.compile(plan, dev, v, _local_index="long long")  # noqa: E731
        comp = CudaCompiledReduction(plan, module, info, ordinal, drv, cdev, source, wide, staging, coop)
        if getattr(plan, "ranges", False) and not coop:
            comp = _ranges_default(comp)
        comp._memo_key = memo_key
        with self._lock:
            comp = self._memo.setdefault(memo_key, comp)
            self._fast[(id(plan), ordinal, _local_index)] = (plan, comp)
        return comp if variant is None else self._with_variant(comp, memo_key, variant)

    def _with_variant(self, base, memo_key, variant):
        if isinstance(base, _EmptyReduction) or variant == base.variant:
            return base
        key = memo_key + (variant,)
        with self._lock:
            hit = self._memo.get(key)
        if hit is None:
            hit = base.with_variant(variant)
            with self._lock:
                hit = self._memo.setdefault(key, hit)
        return hit

    def _info_stub(self, ordinal):
        return CudaCompileInfo("cuda", "", "", "", (), True, 0.0, 0, "", Variant("empty"))

    @staticmethod
    def _cubin(nv, source, options, key):
        """Atomic disk cache (tmp + rename), file lock around compilation."""
        folder = _cache_dir()
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{key}.cubin"
        if target.exists():
            return target.read_bytes(), True
        with open(folder / f"{key}.lock", "w") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                if target.exists():  # compiled by another process while waiting
                    return target.read_bytes(), True
                cubin, _log = nv.compile_cubin(source, options)
                fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-", suffix=".cubin")
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(cubin)
                    (folder / f"{key}.cu").write_text(source)
                    os.replace(tmp, target)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                return cubin, False
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)


class _EmptyReduction:
    """K = 0: outputs (M, 0), no launch."""

    variant = Variant("empty")

    def __init__(self, plan, info):
        self.plan, self.info = plan, info

    ranges_tile_m = 1

    def launch(self, inputs, outputs, batch=None, ranges=None):
        return None


class CudaCompiledReduction:
    def __init__(self, plan: Plan, module, info, ordinal, drv, cdev, source, wide_factory=None,
                 staging="smem", coop=0):
        self.plan, self.info, self.ordinal, self.source = plan, info, ordinal, source
        # lanes per row i (cooperative top-K); 1: one thread per row
        self.lanes = coop or 1
        self._block_scheme = _kernel.sum_scheme(plan) == "block"
        self._wide_factory, self._wide = wide_factory, None
        self._reduce = module.function("kp_reduce")
        self._merge = module.function("kp_merge")
        self._franges = module.function("kp_reduce_ranges") if getattr(plan, "ranges", False) else None
        # variant without tie-breaking (tiles with sorted, disjoint intervals)
        self._franges_mono = (module.function("kp_reduce_ranges_mono")
                              if self._franges is not None and _kernel.ranges_tie_break(plan.reduction.kind, coop)
                              else None)
        self._mono_cache = (None, False)
        self._module = module
        self._sms = drv.attribute(cdev, _driver.ATTR_MULTIPROCESSOR_COUNT)
        self._tbytes = _DT_BYTES[plan.dtype]
        self.staging = staging
        # shared memory bytes per j tile row (0 for direct global reads)
        self._row_bytes = row_bytes(plan) if staging == "smem" else 0
        st = _kernel.state_layout(plan)
        self._state_bytes = st.bytes(self._tbytes, _DT_BYTES[plan.acc_dtype])
        self._state = st
        self._outs = _kernel.output_names(plan)
        self.attrs = dict(self._reduce.attrs)
        self.block, self.tile, self.blocks_per_sm = self._choose_block()
        self.split_mode = "auto"
        self.variant = _make_variant(self.block, self.tile, self.split_mode)
        self.info = self._info_for(info, self.variant)

    @property
    def ranges_tile_m(self) -> int:
        """Amendment v0.6 §2: work tile height = i rows per block (B / G)."""
        return self.block // self.lanes

    def feasible(self):
        """Admissible [(resident threads/SM, B, tj)], sorted by decreasing occupancy."""
        out = []
        for b in BLOCK_CANDIDATES:
            if b > self._reduce.attrs["max_threads"]:
                continue
            for tj in TILE_CANDIDATES:
                if tj > 2 * b or tj * self._row_bytes > SMEM_LIMIT:
                    continue
                occ = self._reduce.occupancy(b, tj * self._row_bytes)
                if occ:
                    out.append((occ * b, b, tj))
        return sorted(out, reverse=True)

    def _info_for(self, info, variant):
        """CompileInfo of the variant; "block" scheme: ``sum_block=T`` (amendment v0.7),
        T = j tile of the variant (blocks of T terms, the last tile of a slice may be
        shorter)."""
        opts = tuple(o for o in info.options if not o.startswith("sum_block="))
        if self._block_scheme:
            opts += (f"sum_block={dict(variant.params)['tile']}",)
        return replace(info, variant=variant, options=opts)

    def with_variant(self, variant: Variant):
        """Copy sharing the module, configured according to ``variant`` (validated)."""
        import copy

        p = dict(variant.params)
        if variant.name != "tiled" or set(p) != {"block", "tile", "split"}:
            raise UnsupportedError(f"unknown CUDA variant {variant}")
        b, tj, sp = p["block"], p["tile"], p["split"]
        # tj a multiple of 4: 16-byte alignment of the tile regions (see _kernel.generate)
        if sp not in SPLIT_MODES or b > self._reduce.attrs["max_threads"] or b % 32 or tj < 4 or tj % 4:
            raise UnsupportedError(f"invalid CUDA variant {variant}")
        smem = tj * self._row_bytes
        occ = self._reduce.occupancy(b, smem) if smem <= SMEM_LIMIT else 0
        if not occ:
            raise UnsupportedError(f"CUDA variant cannot be resident on this device {variant}")
        new = copy.copy(self)
        new.block, new.tile, new.blocks_per_sm, new.split_mode = b, tj, occ, sp
        new.variant = variant
        new.info = self._info_for(self.info, variant)
        new._wide = None
        new.__dict__.pop("_lc", None)  # launch constants (configurations per (M, N)) specific to the variant
        return new

    def _choose_block(self):
        """(B, tj, blocks/SM): maximizes resident threads per SM (registers and shared
        memory), then the tile size, then the block size."""
        best = None
        for b in BLOCK_CANDIDATES:
            if b > self._reduce.attrs["max_threads"]:
                continue
            for tj in TILE_CANDIDATES:
                if tj > 2 * b:
                    continue
                smem = tj * self._row_bytes
                if smem > SMEM_LIMIT:
                    continue
                occ = self._reduce.occupancy(b, smem)
                key = (occ * b, tj, b)
                if best is None or key > best[0]:
                    best = (key, b, tj, occ)
        if best is None or best[0][0] == 0:
            raise UnsupportedError("no admissible block size (shared memory/registers)")
        return best[1], best[2], best[3]

    def split(self, m: int, n: int, nb: int = 1) -> tuple[int, int]:
        """(S, jchunk): j partition when the i rows do not fill the GPU."""
        S, jchunk, _ = self.launch_config(m, n, nb)
        return S, jchunk

    def waves_args(self, m: int, nb: int = 1) -> tuple[int, int]:
        """(i blocks, resident blocks on the GPU) for M rows and ``nb`` batch elements."""
        return -(-m * self.lanes // self.block) * max(nb, 1), max(1, self.blocks_per_sm * self._sms)

    def auto_factor(self, blocks_i: int | None = None, resident: int | None = None) -> int:
        """Target of the "auto" mode (split-policy-4), measured on an idle GPU:
        4 waves help expensive pairs (FP64, large D); for K >= 8 lists, the cost
        of the partials and of the merge (proportional to S * K) dominates if the
        pair is cheap (FP32, <= 16 j components) or if M is very small (N >> M:
        S around 100) -> 2 waves."""
        k = self.plan.reduction.k or 0
        if k < 8:
            return 4
        cheap = self.plan.dtype == "float32" and row_bytes(self.plan) // self._tbytes <= 16
        tiny_m = blocks_i is not None and resident is not None and 8 * blocks_i < resident
        return 2 if cheap or tiny_m else 4

    def launch_config(self, m: int, n: int, nb: int = 1) -> tuple[int, int, int]:
        """(S, jchunk, effective tile).

        split-policy-5: when partitioning is requested but bounded by
        ``n // (4 * tile)`` (K >= 8 lists: to fewer than min(S, 8) slices), the launch tile is shrunk (down to TILE_MIN_SPLIT) so
        that each slice keeps at least 4 tiles; it is only a launch argument (dynamic shared
        memory), so occupancy can only increase."""
        blocks_i, resident = self.waves_args(m, nb)
        S, T = 1, self.tile
        factor = SPLIT_MODES[self.split_mode]
        if factor is None:
            factor = self.auto_factor(blocks_i, resident)
        if n > 0 and factor and blocks_i < factor * resident:
            S = -(-factor * resident // blocks_i)
            # Shrink the tile while the n // (4 T) bound caps S. K >= 8 lists: only if
            # it leaves fewer than min(S, 8) slices, since increasing S there costs
            # more (partials + merge) than it gains (measured).
            want = S if (self.plan.reduction.k or 0) < 8 else min(S, 8)
            # "block" scheme: published T (sum_block) = tile of the variant, never shrunk
            while T > TILE_MIN_SPLIT and n // (4 * T) < want and not self._block_scheme:
                T = max(TILE_MIN_SPLIT, (T // 2) // 4 * 4)
            S = min(S, max(1, n // (4 * T)), 65535)
            S = min(S, max(1, TEMP_CAP_BYTES // max(1, m * max(nb, 1) * self._state_bytes)))
        if S <= 1:
            return 1, max(n, 1), self.tile
        jchunk = -(-(-(-n // S)) // T) * T
        S = -(-n // jchunk)
        return S, jchunk, T

    def _launch_cache(self):
        """Launch constants computed once (host cost, contract v0.3)."""
        c = self.__dict__.get("_lc")
        if c is None:
            import torch

            plan = self.plan
            outs = {}
            for name, (shape, dt) in output_shapes(plan, 1).items():
                assert shape[0] == 1 and output_shapes(plan, 2)[name][0] == (2,) + tuple(shape[1:])
                outs[name] = (tuple(shape[1:]), getattr(torch, dt))
            raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
            st = self._state
            vdt = {"T": getattr(torch, plan.dtype), "ACC": getattr(torch, plan.acc_dtype),
                   "kp_idx": torch.int64}[st.val_type]
            c = self._lc = {
                "args": [(a.name, a.role, a.dim) for a in plan.args],
                "tdt": getattr(torch, plan.dtype), "outs": outs, "raw_stream": raw,
                "vdt": vdt, "configs": {},
            }
        return c

    def _config(self, c, m, n, nb=1):
        key = (m, n) if nb == 1 else (m, n, nb)
        cfg = c["configs"].get(key)
        if cfg is None:
            if len(c["configs"]) >= 256:
                c["configs"].clear()
            cfg = c["configs"][key] = self.launch_config(m, n, nb)
        return cfg

    def _stream(self, c, dev):
        import torch

        raw = c["raw_stream"]
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        return raw(idx) if raw is not None else torch.cuda.current_stream(dev).cuda_stream

    def _launch_batched(self, inputs: dict, outputs: dict, batch) -> None:
        """Amendment v0.6 §1: grid (i blocks, S, batch elements), input offsets read per
        block from the ``(nargs, nb)`` table; outputs ``(nb, M, ·)``; partials
        ``(nb, M, S, ·)``; launches in chunks of ``GRID_Z_MAX`` elements."""
        import torch

        if not self.plan.batch_rank:
            raise UnsupportedError("non-batched plan launched with a BatchLayout")
        c = self._launch_cache()
        nb = int(batch.nb)
        first = outputs[self._outs[0]]
        dev = first.device
        tdt = c["tdt"]
        m = n = None
        ins = []
        hit = self.__dict__.get("_boff")
        known = hit is not None and hit[0] is batch  # tables already checked (cached BatchLayout)
        # Revalidation of what memory safety depends on: only the last two dimensions
        # are contiguous, batch offsets come from the runtime table.
        for name, role, dim in c["args"]:
            t = inputs[name]
            if t.device != dev or t.dtype != tdt:
                raise ValueError(f"argument {name}: invalid device/dtype")
            if not known:
                off = batch.offsets.get(name)
                if (off is None or tuple(off.shape) != (nb,) or off.dtype != torch.int64
                        or off.device != dev):
                    raise ValueError(f"argument {name}: batch offset table missing or malformed")
            if t.dim() < 1 or t.shape[-1] != dim or (dim > 1 and t.stride(-1) != 1):
                raise ValueError(f"argument {name}: expected a contiguous last dimension of size {dim}")
            if role != "param":
                if t.dim() < 2 or (t.shape[-2] > 1 and t.stride(-2) != dim):
                    raise ValueError(f"argument {name}: expected contiguous last two dimensions")
                ln = int(t.shape[-2])
                if role == "i":
                    if m is not None and ln != m:
                        raise ValueError("i arguments have different lengths")
                    m = ln
                else:
                    if n is not None and ln != n:
                        raise ValueError("j arguments have different lengths")
                    n = ln
            ins.append(t.data_ptr())
        if m is None:
            m = int(first.shape[1]) if first.dim() >= 2 else 0
        for name, (rest, odt) in c["outs"].items():
            t = outputs[name]
            if (t.shape != (nb, m, *rest) or t.dtype != odt or t.device != dev
                    or not t.is_contiguous()):
                raise ValueError(f"invalid output {name}")
        if m == 0 or nb == 0:
            return
        if n is None:
            raise UnsupportedError("plan without a j argument: N is undetermined")
        if n > INT32_MAX_N and self._wide_factory is not None:
            if self._wide is None:
                self._wide = self._wide_factory(self.variant)
            return self._wide.launch(inputs, outputs, batch=batch)
        if known:
            table = hit[1]
        else:
            table = torch.stack([batch.offsets[name] for name, _, _ in c["args"]]).contiguous()
            self._boff = (batch, table)
        stream = self._stream(c, dev)
        S, jchunk, tj = self._config(c, m, n, nb)
        B = self.block
        grid_x = -(-m * self.lanes // B)
        smem = tj * self._row_bytes
        outs = [outputs[o].data_ptr() for o in self._outs]
        pv = pi = None
        pptr = [0, 0]
        if S > 1:
            st = self._state
            pv = torch.empty((nb * m * S * st.n_val,), dtype=c["vdt"], device=dev)
            pi = torch.empty((max(1, nb * m * S * st.n_idx),), dtype=torch.int64, device=dev)
            pptr = [pv.data_ptr(), pi.data_ptr()]
        for b0 in range(0, nb, GRID_Z_MAX):
            nz = min(GRID_Z_MAX, nb - b0)
            tail = [table.data_ptr(), nb, b0]
            self._reduce.launch_params(_pack([m, n, jchunk, tj] + ins + outs + pptr + tail),
                                       (grid_x, S, nz), B, smem, stream)
            if S > 1:
                self._merge.launch_params(_pack([m, S] + outs + pptr + tail),
                                          (-(-m * self.lanes // 128), 1, nz), 128, 0, stream)
        # table (cached), pv/pi: the Torch allocator respects stream order.

    def _launch_ranges(self, c, dev, m, n, vals, ranges, stream) -> None:
        """Amendment v0.6 §2: neutral elements on every row (``kp_reduce`` with N = 0), then
        one block per work tile ``(i0, i1, k0, k1)`` (``kp_reduce_ranges``), without
        j partition; ARG*/K* ties go to the smallest j of J(i)."""
        import torch

        if self._franges is None:
            raise UnsupportedError("plan without ranges launched with a RangesLayout")
        if int(ranges.tile_m) != self.ranges_tile_m:
            raise ValueError(f"tile_m={ranges.tile_m}, expected {self.ranges_tile_m} (ranges_tile_m)")
        work, red = ranges.work, ranges.redranges
        for nm, t in (("work", work), ("redranges", red)):
            if t.dtype != torch.int64 or not t.is_contiguous() or t.device != dev or t.dim() != 2:
                raise ValueError(f"ranges.{nm}: expected a contiguous int64 (·, ·) tensor on {dev}")
        if work.shape[1] != 4 or red.shape[1] != 2:
            raise ValueError("ranges: expected work (W, 4) and redranges (R, 2)")
        B = self.block
        if B > self._franges.attrs["max_threads"]:
            raise UnsupportedError(f"kp_reduce_ranges: block {B} > max_threads")
        tj = self.tile
        smem = tj * self._row_bytes
        fr = self._franges
        if self._franges_mono is not None and self._j_sorted(ranges):
            fr = self._franges_mono
        vals[0], vals[1], vals[2], vals[3] = m, 0, 1, tj
        self._reduce.launch_params(_pack(vals + [0, 0]), (-(-m * self.lanes // B), 1, 1), B, smem, stream)
        nw = int(work.shape[0])
        if nw:
            vals[1], vals[2] = n, 0
            fr.launch_params(_pack(vals + [0, 0, work.data_ptr(), red.data_ptr()]),
                                        (nw, 1, 1), B, smem, stream)

    def _j_sorted(self, ranges) -> bool:
        """Does every tile traverse sorted, disjoint intervals (increasing j)?

        Computed on the device (a single boolean transferred), cached by identity of the
        ``RangesLayout`` (the runtime reuses the same object from one call to the next)."""
        hit, ok = self._mono_cache
        if hit is ranges:
            return ok
        import torch

        work, red = ranges.work, ranges.redranges  # on the device: a single boolean transferred
        if work.shape[0] == 0 or red.shape[0] < 2:
            ok = True
        else:
            # bad[k] = 1 if red[k+1] starts before the end of red[k]; a tile [k0, k1)
            # is sorted if there is no bad[k] for k in [k0, k1 - 1).
            bad = (red[1:, 0] < red[:-1, 1]).to(torch.int64)
            cb = torch.cat([torch.zeros(1, dtype=torch.int64, device=bad.device), torch.cumsum(bad, 0)])
            top = red.shape[0] - 1
            k0, k1 = work[:, 2].clamp(max=top), work[:, 3]
            hi = torch.maximum(k1 - 1, k0).clamp(max=top)
            ok = bool(((cb[hi] - cb[k0]) == 0).all())
        self._mono_cache = (ranges, ok)
        return ok

    def launch(self, inputs: dict, outputs: dict, batch=None, ranges=None) -> None:
        import torch

        if batch is not None and ranges is not None:
            raise UnsupportedError("ranges combined with batch: rejected (amendment v0.6)")
        if batch is not None:
            return self._launch_batched(inputs, outputs, batch)
        c = self._launch_cache()
        first = outputs[self._outs[0]]
        m = int(first.shape[0])
        if m == 0:
            return
        dev = first.device
        tdt = c["tdt"]
        n = None
        vals = [0, 0, 0, 0]  # M, N, jchunk, tj: filled in below
        # Cheap revalidation of what memory safety depends on.
        for name, role, dim in c["args"]:
            t = inputs[name]
            if t.device != dev or t.dtype != tdt or not t.is_contiguous():
                raise ValueError(f"argument {name}: invalid device/dtype/contiguity")
            if role == "i":
                ok = t.shape == (m, dim)
            elif role == "j":
                if n is None:
                    n = int(t.shape[0])
                ok = t.shape == (n, dim)
            else:
                ok = t.numel() == dim
            if not ok:
                raise ValueError(f"argument {name}: invalid shape {tuple(t.shape)}")
            vals.append(t.data_ptr())
        for name, (rest, odt) in c["outs"].items():
            t = outputs[name]
            if (t.shape != (m, *rest) or t.dtype != odt or t.device != dev
                    or not t.is_contiguous()):
                raise ValueError(f"invalid output {name}")
        vals += [outputs[o].data_ptr() for o in self._outs]
        if n > INT32_MAX_N and self._wide_factory is not None:
            if self._wide is None:
                self._wide = self._wide_factory(self.variant)
            return self._wide.launch(inputs, outputs, ranges=ranges)
        stream = self._stream(c, dev)
        if ranges is not None:
            return self._launch_ranges(c, dev, m, n, vals, ranges, stream)
        S, jchunk, tj = self._config(c, m, n)
        vals[0], vals[1], vals[2], vals[3] = m, n, jchunk, tj
        B = self.block
        grid_x = -(-m * self.lanes // B)
        smem = tj * self._row_bytes
        if S == 1:
            vals += [0, 0]
            self._reduce.launch_params(_pack(vals), (grid_x, 1, 1), B, smem, stream)
            return
        st = self._state
        pv = torch.empty((m * S * st.n_val,), dtype=c["vdt"], device=dev)
        pi = torch.empty((max(1, m * S * st.n_idx),), dtype=torch.int64, device=dev)
        vals += [pv.data_ptr(), pi.data_ptr()]
        self._reduce.launch_params(_pack(vals), (grid_x, S, 1), B, smem, stream)
        mvals = [m, S] + vals[4 + len(c["args"]):]
        self._merge.launch_params(_pack(mvals), (-(-m * self.lanes // 128), 1, 1), 128, 0, stream)
        # pv/pi: freed on exit; the Torch allocator respects stream order.
