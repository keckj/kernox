"""``hip`` backend (AMD ROCm), compliant with contract v0.

Toolchain: hipRTC **from the same ROCm package as Torch** (in-process compilation),
loading through ``hipModuleLoadData`` and launching through ``hipModuleLaunchKernel``
on ``torch.cuda.current_stream(device)``. See ``docs/backend-hip.md``.
"""

from __future__ import annotations

import ctypes
import hashlib
import threading
import time
from dataclasses import dataclass, replace

from ...plan import Plan, ReductionKind, output_shapes
from ..base import CompileInfo, Sizes, Support, UnsupportedError, Variant
from . import _cache, _hip
from ._kernel import (COOP_K_MIN, GENERATOR_VERSION, KMIN_KINDS, S_MAX, TOPK_SPLIT_BELOW_BLOCKS_PER_CU,
                      TOPK_SPLIT_BLOCKS_PER_CU, KernelConfig, choose_config,
                      coop_group, kernel_source, partial_bytes, partial_layout)

KERNEL_NAME = "kp_reduce"
MERGE_NAME = "kp_merge"
RANGES_NAME = "kp_reduce_ranges"
GRID_Z_MAX = 65535  # batch elements per launch (z dimension of the grid)
K_MAX = 1024  # coop G=32, 32 ranks/lane; beyond that: UnsupportedError
_DT_RANK = {"float16": -1, "float32": 0, "float64": 1}

MEASURE_MAX, PATIENT_MAX = 4, 32  # contract v0.2


@dataclass(frozen=True)
class HipCompileInfo(CompileInfo):
    """``CompileInfo`` + effective variant (contract v0.2; field missing from base.py)."""

    variant: Variant | None = None


_modules_lock = threading.Lock()
# (key, device) -> (module, kp_reduce, kp_merge); never unloaded (see docs: lifetime)
_modules: dict = {}


def _default_pair_provider(plan: Plan):
    from ...codegen.pair import pair_function  # pair function from the core (symox)

    return pair_function(plan, "hip")


def _torch():
    import torch

    return torch


def _device_index(device) -> int:
    torch = _torch()
    dev = torch.device(device)
    if dev.type != "cuda":
        raise UnsupportedError(f"hip backend: device {dev} is not a ROCm device")
    return dev.index if dev.index is not None else torch.cuda.current_device()


def _plan_key_or_none(plan: Plan):
    try:
        return plan.key()
    except ImportError:  # _plan_key not yet provided by the core
        return None


@dataclass(frozen=True)
class HipOptions:
    """Compilation options; all of them are part of the cache key.

    ``fp_contract`` (contract v0.3): ``None`` = follow ``plan.fp_contract``
    (``True`` by default → ``-ffp-contract=on``: FMA contraction within a single
    expression only, never ``fast``); ``False`` → ``-ffp-contract=off``.
    ``"on"``/``"off"``: explicit override (diagnostics). The effective flags are
    part of the cache key.
    """

    fp_contract: str | None = None
    block: int = 256
    opt_level: str = "-O3"
    # Traversal variant: "auto" | "thread" | "coop" (KMIN/ARGKMIN); coop group (None = auto).
    strategy: str = "auto"
    group: int | None = None
    tile_j: int | None = None  # None: bounded by the LDS (32 KiB) and the block size
    # j partition (decided at launch time, no recompilation):
    # "auto" (small M), "off", "force" (tests/benchmarks); jchunk sets the range size.
    split: str = "auto"
    jchunk: int | None = None
    # Target blocks per CU before partitioning j: 32 beats 8 in every gauss/count case
    # measured on R9700 (−7 to −15 %); to be confirmed on other cards.
    split_blocks_per_cu: int = 32
    temp_max_bytes: int = 256 << 20  # published bound on the partition temporaries
    # LSE/SSMW: 1 = direct insertion into the row state, 2 = local state per tile
    # then merge; None = depends on the state size (``soft_levels_auto``).
    soft_levels: int | None = None

    def flags(self, arch: str, plan: Plan | None = None) -> list[str]:
        fc = self.fp_contract
        if fc is None:
            fc = "on" if getattr(plan, "fp_contract", True) else "off"
        if fc not in ("off", "on"):
            raise ValueError("fp_contract must be None, 'off' or 'on'")
        return [f"--offload-arch={arch}", self.opt_level, "-std=c++17",
                f"-ffp-contract={fc}", "-fno-fast-math",
                "-fno-gpu-flush-denormals-to-zero"]


_ACC_KINDS = (ReductionKind.SUM, ReductionKind.LOGSUMEXP, ReductionKind.SUMSOFTMAXWEIGHT,
              ReductionKind.LOGSUMEXPWEIGHT)


def _scheme_options(plan: Plan, cfg: KernelConfig) -> tuple:
    """Effective summation scheme (amendment v0.7); ``sum_block=T`` for block."""
    if plan.reduction.kind not in _ACC_KINDS:
        return ()
    return (f"sum_scheme={plan.sum_scheme}",) + (
        (f"sum_block={cfg.tile_j}",) if plan.sum_scheme == "block" else ())


class HipCompiledReduction:
    def __init__(self, plan: Plan, device_index: int, fns, info: CompileInfo, cfg: KernelConfig,
                 options: "HipOptions | None" = None, cu_count: int = 1):
        self.plan = plan
        self.device_index = device_index
        fns = tuple(fns) if fns is not None else (None, None)
        self._fn, self._merge = fns[0], fns[1]
        self._franges = fns[2] if len(fns) > 2 else None
        # amendment v0.6 §2: work tile height = i rows per block
        self.ranges_tile_m = cfg.rows_per_block
        self._boff = None  # (layout, (nargs, nb) int64 table) of the last batched call
        self.options = options or HipOptions()
        self.cu_count = cu_count
        self.last_launch: dict = {}
        self.info = info
        self.config = cfg
        self._out_names = list(output_shapes(plan, 0).keys())
        self._i_args = plan.args_of("i")
        self._j_args = plan.args_of("j")
        self.source: str = ""
        self.cache_key: str = ""

    def resources(self, kernel: str = "reduce") -> dict:
        rt = _hip.hip_runtime()
        fn = {"reduce": self._fn, "merge": self._merge, "ranges": self._franges}[kernel]
        return {
            "num_regs": rt.func_attribute(fn, _hip.FUNC_ATTR_NUM_REGS),
            "local_bytes_per_thread": rt.func_attribute(fn, _hip.FUNC_ATTR_LOCAL_SIZE_BYTES),
            "shared_bytes": rt.func_attribute(fn, _hip.FUNC_ATTR_SHARED_SIZE_BYTES),
            "max_threads_per_block": rt.func_attribute(fn, _hip.FUNC_ATTR_MAX_THREADS_PER_BLOCK),
        }

    def split_plan(self, M: int, N: int, nb: int = 1) -> tuple[int, int]:
        """(S, jchunk) for (M, N) and ``nb`` batch elements; S = 1: no j partition."""
        o, cfg = self.options, self.config
        TJ = cfg.tile_j
        if N == 0 or o.split == "off":
            return 1, max(N, 1)
        if o.jchunk is not None:
            jchunk = max(1, int(o.jchunk))
            S = -(-N // jchunk)
            if S > S_MAX:
                raise UnsupportedError(f"jchunk={jchunk} gives S={S} > S_MAX={S_MAX}")
        else:
            gx = -(-M // cfg.rows_per_block) * max(nb, 1)
            if o.split == "force":
                S = min(S_MAX, -(-N // TJ))
            else:
                target = o.split_blocks_per_cu * self.cu_count
                S = 1 if gx >= target else min(S_MAX, -(-target // gx), -(-N // TJ))
            while S > 1 and partial_bytes(self.plan, S, M) * nb > o.temp_max_bytes:
                S //= 2
            if S <= 1:
                return 1, N
            jchunk = -(-(-(-N // S)) // TJ) * TJ
            S = -(-N // jchunk)
        if S > 1 and partial_bytes(self.plan, S, M) * nb > o.temp_max_bytes:
            raise UnsupportedError("partition temporaries exceed temp_max_bytes")
        return S, jchunk

    def temp_bytes(self, M: int, N: int) -> int:
        S, _ = self.split_plan(M, N)
        return partial_bytes(self.plan, S, M) if S > 1 else 0

    def _check(self, inputs: dict, outputs: dict, batch=None) -> tuple[int, int]:
        """Cheap revalidation of everything memory safety depends on.

        ``batch`` (amendment v0.6 §1): inputs ``(*B, L, d)`` of which only the last two
        dimensions are contiguous (the batch offsets come from the runtime table),
        params ``(*Bp, d)`` or ``(d,)``; contiguous ``(nb, M, ·)`` outputs."""
        torch = _torch()
        plan = self.plan
        dt = getattr(torch, plan.dtype)
        M = N = None
        for a in plan.args:
            t = inputs.get(a.name)
            if t is None:
                raise ValueError(f"missing input: {a.name}")
            if t.device.type != "cuda" or t.device.index != self.device_index:
                raise ValueError(f"{a.name}: expected device cuda:{self.device_index}, got {t.device}")
            if batch is not None:
                if t.dtype != dt:
                    raise ValueError(f"{a.name}: invalid dtype {t.dtype}")
                if a.name not in batch.offsets or tuple(batch.offsets[a.name].shape) != (batch.nb,):
                    raise ValueError(f"{a.name}: batch offset table missing or malformed")
                if t.shape[-1] != a.dim or (a.dim > 1 and t.stride(-1) != 1):
                    raise ValueError(f"{a.name}: expected a contiguous last dimension of size {a.dim}")
                if a.role == "param":
                    continue
                if t.dim() < 2 or (t.shape[-2] > 1 and t.stride(-2) != a.dim):
                    raise ValueError(f"{a.name}: expected the last two dimensions to be contiguous")
                L_ = t.shape[-2]
                if a.role == "i":
                    if M is not None and L_ != M:
                        raise ValueError("i arguments have different lengths")
                    M = L_
                else:
                    if N is not None and L_ != N:
                        raise ValueError("j arguments have different lengths")
                    N = L_
                continue
            if t.dtype != dt or not t.is_contiguous():
                raise ValueError(f"{a.name}: invalid dtype {t.dtype} or non-contiguous tensor")
            if a.role == "param":
                if tuple(t.shape) != (a.dim,):
                    raise ValueError(f"{a.name}: expected shape ({a.dim},), got {tuple(t.shape)}")
                continue
            if t.dim() != 2 or t.shape[1] != a.dim:
                raise ValueError(f"{a.name}: expected shape (·, {a.dim}), got {tuple(t.shape)}")
            if a.role == "i":
                if M is not None and t.shape[0] != M:
                    raise ValueError("i arguments have different lengths")
                M = t.shape[0]
            else:
                if N is not None and t.shape[0] != N:
                    raise ValueError("j arguments have different lengths")
                N = t.shape[0]
        first_out = outputs[self._out_names[0]]
        if M is None:
            M = first_out.shape[1 if batch is not None else 0]
        expected = output_shapes(plan, M, (batch.nb,)) if batch is not None else output_shapes(plan, M)
        for name, (shape, odt) in expected.items():
            t = outputs.get(name)
            if t is None:
                raise ValueError(f"missing output: {name}")
            if (tuple(t.shape) != shape or t.dtype != getattr(torch, odt) or not t.is_contiguous()
                    or t.device.type != "cuda" or t.device.index != self.device_index):
                raise ValueError(f"output {name}: expected a contiguous {shape} {odt} tensor on cuda:{self.device_index}")
        if N is None:  # ruled out by supports()
            raise UnsupportedError("plan without a j argument: N is undetermined")
        return M, N

    def launch(self, inputs: dict, outputs: dict, batch=None, ranges=None) -> None:
        if batch is not None and ranges is not None:
            raise UnsupportedError("ranges combined with batch: rejected (amendment v0.6)")
        if batch is not None:
            self._launch_batched(inputs, outputs, batch)
            return
        torch = _torch()
        M, N = self._check(inputs, outputs)
        if M == 0:
            return
        if self.plan.reduction.kind in KMIN_KINDS and self.plan.reduction.k == 0:
            return
        if ranges is not None:
            self._launch_ranges(inputs, outputs, M, ranges)
            return
        S, jchunk = self.split_plan(M, N)
        cfg = self.config
        gx = -(-M // cfg.rows_per_block)
        if gx > 2**31 - 1:
            raise UnsupportedError("M too large for the grid")
        ins = [ctypes.c_void_p(inputs[a.name].data_ptr()) for a in self.plan.args]
        outs = [ctypes.c_void_p(outputs[n].data_ptr()) for n in self._out_names]
        rt = _hip.hip_runtime()
        with torch.cuda.device(self.device_index):
            stream = torch.cuda.current_stream(self.device_index).cuda_stream
            if S == 1:
                args = ins + [ctypes.c_longlong(M), ctypes.c_longlong(N), ctypes.c_longlong(jchunk),
                              ctypes.c_int(0)] + outs + [ctypes.c_void_p(0), ctypes.c_void_p(0)]
                rt.launch(self._fn, (gx, 1, 1), (cfg.block, 1, 1), 0, stream, args)
                self.last_launch = {"S": 1, "jchunk": jchunk, "temp_bytes": 0, "grid": (gx, 1)}
                return
            # j partition: [S, M, W] partials allocated by Torch on this device and this
            # stream (reuse ordered by the stream, see docs), then merge.
            pv, pi = self._partials(S * M)
            pptr = [ctypes.c_void_p(pv.data_ptr()), ctypes.c_void_p(pi.data_ptr() if pi is not None else 0)]
            args = ins + [ctypes.c_longlong(M), ctypes.c_longlong(N), ctypes.c_longlong(jchunk),
                          ctypes.c_int(1)] + outs + pptr
            rt.launch(self._fn, (gx, S, 1), (cfg.block, 1, 1), 0, stream, args)
            margs = [ctypes.c_longlong(M), ctypes.c_int(S)] + pptr + outs
            rt.launch(self._merge, (-(-M // cfg.block), 1, 1), (cfg.block, 1, 1), 0, stream, margs)
            self.last_launch = {"S": S, "jchunk": jchunk, "grid": (gx, S),
                                "temp_bytes": pv.numel() * pv.element_size()
                                + (pi.numel() * 8 if pi is not None else 0)}

    def _partials(self, rows: int):
        torch = _torch()
        W, pvt, has_pi = partial_layout(self.plan)
        pdt = {"kp_acc": getattr(torch, self.plan.acc_dtype), "kp_i64": torch.int64,
               "kp_t": getattr(torch, self.plan.dtype)}[pvt]
        dev = torch.device("cuda", self.device_index)
        pv = torch.empty(rows * W, dtype=pdt, device=dev)
        pi = torch.empty(rows * W, dtype=torch.int64, device=dev) if has_pi else None
        return pv, pi

    def _launch_batched(self, inputs: dict, outputs: dict, batch) -> None:
        """Addendum v0.6 §1: grid (gx, S, batch elements), offsets read per block from the
        ``(nargs, nb)`` table; partials ``(nb, S, M, W)``; launches in chunks of
        ``GRID_Z_MAX`` elements."""
        torch = _torch()
        if not self.plan.batch_rank:
            raise UnsupportedError("non-batched plan launched with a BatchLayout")
        M, N = self._check(inputs, outputs, batch)
        nb = int(batch.nb)
        if M == 0 or nb == 0:
            return
        if self.plan.reduction.kind in KMIN_KINDS and self.plan.reduction.k == 0:
            return
        S, jchunk = self.split_plan(M, N, nb)
        cfg = self.config
        gx = -(-M // cfg.rows_per_block)
        if gx > 2**31 - 1:
            raise UnsupportedError("M too large for the grid")
        hit = self._boff
        if hit is not None and hit[0] is batch:
            table = hit[1]
        else:
            table = torch.stack([batch.offsets[a.name].to(torch.int64) for a in self.plan.args]).contiguous()
            self._boff = (batch, table)
        ins = [ctypes.c_void_p(inputs[a.name].data_ptr()) for a in self.plan.args]
        outs = [ctypes.c_void_p(outputs[n].data_ptr()) for n in self._out_names]
        rt = _hip.hip_runtime()
        with torch.cuda.device(self.device_index):
            stream = torch.cuda.current_stream(self.device_index).cuda_stream
            pv = pi = None
            if S > 1:
                pv, pi = self._partials(nb * S * M)
            pptr = [ctypes.c_void_p(pv.data_ptr() if pv is not None else 0),
                    ctypes.c_void_p(pi.data_ptr() if pi is not None else 0)]
            for b0 in range(0, nb, GRID_Z_MAX):
                nz = min(GRID_Z_MAX, nb - b0)
                tail = [ctypes.c_void_p(table.data_ptr()), ctypes.c_longlong(nb), ctypes.c_longlong(b0)]
                args = ins + [ctypes.c_longlong(M), ctypes.c_longlong(N), ctypes.c_longlong(jchunk),
                              ctypes.c_int(1 if S > 1 else 0)] + outs + pptr + tail
                rt.launch(self._fn, (gx, S, nz), (cfg.block, 1, 1), 0, stream, args)
                if S > 1:
                    margs = [ctypes.c_longlong(M), ctypes.c_int(S)] + pptr + outs + tail
                    rt.launch(self._merge, (-(-M // cfg.block), 1, nz), (cfg.block, 1, 1), 0, stream, margs)
        self.last_launch = {"S": S, "jchunk": jchunk, "grid": (gx, S, nb), "nb": nb,
                            "temp_bytes": (pv.numel() * pv.element_size() + (pi.numel() * 8 if pi is not None else 0))
                            if pv is not None else 0}

    def _launch_ranges(self, inputs: dict, outputs: dict, M: int, ranges) -> None:
        """Addendum v0.6 §2: neutral elements on every row (N = 0 launch), then one block per
        work tile ``(i0, i1, k0, k1)``; ties go to the smallest j (broken by j)."""
        torch = _torch()
        if self._franges is None:
            raise UnsupportedError("plan without ranges launched with a RangesLayout")
        if int(ranges.tile_m) != self.ranges_tile_m:
            raise ValueError(f"expected tile_m={self.ranges_tile_m} (ranges_tile_m), got {ranges.tile_m}")
        work, red = ranges.work, ranges.redranges
        for nm, t in (("work", work), ("redranges", red)):
            if (t.dtype != torch.int64 or not t.is_contiguous() or t.device.type != "cuda"
                    or t.device.index != self.device_index or t.dim() != 2):
                raise ValueError(f"ranges.{nm}: expected a contiguous int64 (·, ·) tensor on cuda:{self.device_index}")
        if work.shape[1] != 4 or red.shape[1] != 2:
            raise ValueError("ranges: expected work of shape (W, 4) and redranges of shape (R, 2)")
        cfg = self.config
        gx = -(-M // cfg.rows_per_block)
        ins = [ctypes.c_void_p(inputs[a.name].data_ptr()) for a in self.plan.args]
        outs = [ctypes.c_void_p(outputs[n].data_ptr()) for n in self._out_names]
        rt = _hip.hip_runtime()
        nw = int(work.shape[0])
        with torch.cuda.device(self.device_index):
            stream = torch.cuda.current_stream(self.device_index).cuda_stream
            args = ins + [ctypes.c_longlong(M), ctypes.c_longlong(0), ctypes.c_longlong(1),
                          ctypes.c_int(0)] + outs + [ctypes.c_void_p(0), ctypes.c_void_p(0)]
            rt.launch(self._fn, (gx, 1, 1), (cfg.block, 1, 1), 0, stream, args)   # neutral elements
            if nw:
                rargs = ins + [ctypes.c_longlong(M), ctypes.c_void_p(work.data_ptr()),
                               ctypes.c_void_p(red.data_ptr())] + outs
                rt.launch(self._franges, (nw, 1, 1), (cfg.block, 1, 1), 0, stream, rargs)
        self.last_launch = {"S": 1, "work_tiles": nw, "grid": (nw, 1), "temp_bytes": 0}


class HipBackend:
    name = "hip"
    fp_contract_off = True  # contract v0.3: plan.fp_contract=False → -ffp-contract=off
    # amendment v0.7 (SUM, LSE, SSMW): block = j tile (T = tile_j, published as "sum_block=T"
    # in CompileInfo.options) summed into a fresh accumulator; kahan = Neumaier.
    sum_schemes = ("direct", "block", "kahan")
    batch = True    # amendment v0.6 §1: per-block offsets (grid z), (nb, M, ·) outputs
    ranges = True   # amendment v0.6 §2: work list in tiles of height ranges_tile_m

    def __init__(self, options: HipOptions | None = None, pair_provider=None):
        self.options = options or HipOptions()
        self._pair_provider = pair_provider or _default_pair_provider
        self._mem: dict = {}
        self._lock = threading.Lock()

    # -- capabilities --------------------------------------------------------
    @staticmethod
    def _lds_limit(dev: int) -> int:
        torch = _torch()
        p = torch.cuda.get_device_properties(dev)
        return int(getattr(p, "shared_memory_per_block", 65536))

    def supports(self, plan: Plan, device) -> Support:
        torch = _torch()
        try:
            dev = _device_index(device)
        except UnsupportedError as e:
            return Support(False, str(e))
        if torch.version.hip is None or not torch.cuda.is_available():
            return Support(False, "Torch ROCm unavailable")
        if plan.dtype not in _DT_RANK or plan.acc_dtype not in _DT_RANK:
            return Support(False, f"unsupported dtype: {plan.dtype}/{plan.acc_dtype}")
        if _DT_RANK[plan.acc_dtype] < _DT_RANK[plan.dtype]:
            return Support(False, "acc_dtype is narrower than dtype")
        if plan.fast_math:
            return Support(False, "fast_math is not supported by the hip backend")
        if plan.sum_scheme not in self.sum_schemes:
            return Support(False, f"unknown sum_scheme={plan.sum_scheme!r} (amendment v0.7)")
        if plan.batch_rank and plan.ranges:
            return Support(False, "ranges combined with batch: rejected (amendment v0.6)")
        if plan.index_dtype != "int64":
            return Support(False, "index_dtype must be int64")
        if not plan.args_of("j"):
            return Support(False, "plan without a j argument: N is undetermined in launch (see notes)")
        if plan.out_dim < 1:
            return Support(False, "formula has no output")
        kind, k = plan.reduction.kind, plan.reduction.k
        if kind in KMIN_KINDS:
            if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                return Support(False, f"invalid K: {k!r}")
            if plan.out_dim != 1:
                return Support(False, "KMIN/ARGKMIN: scalar formulas only (v0)")
            if k > K_MAX:
                return Support(False, f"K={k} > K_MAX={K_MAX} for the hip backend v0")
        try:
            cfg = self._config(plan, dev)
        except ValueError as e:
            return Support(False, str(e))
        if cfg is None:
            return Support(False, "j row too wide for shared memory")
        return Support(True)

    def _config(self, plan: Plan, dev: int, o: HipOptions | None = None):
        o = o or self.options
        return choose_config(plan, self._lds_limit(dev), o.block, o.strategy, o.group, o.tile_j,
                             o.soft_levels)

    # -- variants (contract v0.2) ---------------------------------------------
    # name: "row" (one thread per i, no j partition), "jsplit" (j partition decided
    # at launch time, blocks_per_cu threshold), "coop_topk" (KMIN, list distributed
    # over `group` lanes; split "off" or "auto"). Parameters: block, tile_j, group,
    # split, blocks_per_cu. All concrete: a variant does not depend on the defaults.
    @staticmethod
    def _variant_of(cfg: KernelConfig, o: HipOptions) -> Variant:
        params = {"block": cfg.block, "tile_j": cfg.tile_j, "split": o.split}
        if o.split == "auto":
            params["blocks_per_cu"] = o.split_blocks_per_cu
        if cfg.levels:  # LSE/SSMW only
            params["levels"] = cfg.levels
        if cfg.strategy == "coop":
            params["group"] = cfg.group
            name = "coop_topk"
        else:
            name = "jsplit" if o.split == "auto" else "row"
        return Variant(name, tuple(sorted(params.items())))

    def _options_for(self, variant: Variant) -> HipOptions:
        p = dict(variant.params)
        unknown = set(p) - {"block", "tile_j", "split", "blocks_per_cu", "group", "levels"}
        if unknown or variant.name not in ("row", "jsplit", "coop_topk"):
            raise UnsupportedError(f"unknown hip variant: {variant}")
        split = p.get("split", "off" if variant.name == "row" else "auto")
        if (variant.name == "row") != (split == "off") and variant.name != "coop_topk":
            raise UnsupportedError(f"inconsistent variant: {variant}")
        return replace(self.options, block=int(p.get("block", self.options.block)),
                       tile_j=int(p["tile_j"]) if "tile_j" in p else None,
                       strategy="coop" if variant.name == "coop_topk" else "thread",
                       group=int(p["group"]) if "group" in p else None, split=split,
                       soft_levels=int(p["levels"]) if "levels" in p else None,
                       split_blocks_per_cu=int(p.get("blocks_per_cu", self.options.split_blocks_per_cu)),
                       jchunk=None)

    def _valid(self, plan: Plan, dev: int, o: HipOptions) -> KernelConfig | None:
        if o.block % 64 or not 64 <= o.block <= 1024:
            return None
        try:
            cfg = self._config(plan, dev, o)
        except ValueError:
            return None
        if cfg is None or (cfg.strategy == "coop" and cfg.slots(plan.reduction.k or 1) > 32):
            return None
        return cfg

    def default_variant(self, plan: Plan, device, sizes: Sizes) -> Variant:
        """ESTIMATE heuristic (deterministic; thresholds from a single measurement series,
        to be confirmed):

        - KMIN/ARGKMIN, K >= ``coop_k_min`` (32; 48 if the j row has >= 64 dimensions):
          ``coop_topk``, with G such that there are <= 24 ranks per lane (G=16 above K=64);
        - j partition if the grid has fewer than ``32 × CU`` blocks and N > tile, with S
          sized for ``32 × CU`` blocks; top-K with K >= 32: fewer than ``4 × CU`` blocks,
          S sized for ``8 × CU`` (merge cost S·M·K);
        - block 256, tile = min(256, 32 KiB / j row);
        - LSE/SSMW: one-level insertion (``levels`` 1) with FP64 accumulation or if the
          per-row state exceeds 256 B, otherwise two levels (measured on R9700; block 128 not
          kept: SSMW FP64 E64 one level 100.9 ms with block 256 vs 103.6 with block 128).
        """
        dev = _device_index(device)
        cu = _torch().cuda.get_device_properties(dev).multi_processor_count
        o = replace(self.options, strategy="auto", group=None, tile_j=None, jchunk=None)
        cfg = self._config(plan, dev, o)
        if cfg is None:
            raise UnsupportedError("j row too wide for shared memory")
        gx = -(-max(sizes.m, 1) // cfg.rows_per_block) * max(getattr(sizes, "nb", 1), 1)
        k = plan.reduction.k
        if plan.reduction.kind in KMIN_KINDS and k is not None and k >= COOP_K_MIN:
            below, bpc = TOPK_SPLIT_BELOW_BLOCKS_PER_CU, TOPK_SPLIT_BLOCKS_PER_CU
        else:
            below = bpc = o.split_blocks_per_cu
        split = gx < below * cu and sizes.n > cfg.tile_j
        return self._variant_of(cfg, replace(o, split="auto" if split else "off", split_blocks_per_cu=bpc))

    def variants(self, plan: Plan, device, sizes: Sizes, effort: str) -> list[Variant]:
        """``measure``: <= 4 discriminating variants; ``patient``: <= 32; default first."""
        if effort not in ("measure", "patient"):
            raise ValueError(f"unknown effort: {effort!r}")
        dev = _device_index(device)
        base = self.default_variant(plan, device, sizes)
        bp = dict(base.params)
        kmin = plan.reduction.kind in KMIN_KINDS
        k = plan.reduction.k or 1
        cands: list[Variant] = [base]

        def add(name, **params):
            params = {kk: v for kk, v in params.items() if v is not None}
            if params.get("split") != "auto":
                params.pop("blocks_per_cu", None)
            if name != "coop_topk":
                name = "jsplit" if params.get("split") == "auto" else "row"
            params.pop("group", None) if name != "coop_topk" else None
            v = Variant(name, tuple(sorted(params.items())))
            o = self._options_for(v)
            cfg = self._valid(plan, dev, o)
            if cfg is not None:
                v = self._variant_of(cfg, o)  # canonical form (effective tile)
                if v not in cands:
                    cands.append(v)

        split_now = bp["split"]
        other_split = "off" if split_now == "auto" else "auto"
        jname = lambda sp: "jsplit" if sp == "auto" else "row"  # noqa: E731
        bpc = self.options.split_blocks_per_cu
        alt_bpc = 8 if bpc != 8 else 32
        if effort == "measure":
            # Choices for gfx1201 (wave32, 64 KiB LDS/WGP, 32 CUs reported):
            if kmin:
                g = coop_group(k, plan.dtype)
                if base.name == "coop_topk":
                    if k <= 128:  # thread: spills measured (FP64 from 64, FP32 from 128)
                        add(jname(split_now), block=256, split=split_now, blocks_per_cu=bpc)
                    else:         # more ranks per lane (32), fewer shuffles
                        add("coop_topk", block=256, group=max(2, bp["group"] // 2), split=split_now,
                            blocks_per_cu=bpc)
                    add("coop_topk", block=256, group=min(32, bp["group"] * 2), split=split_now,
                        blocks_per_cu=bpc)                                                   # fewer ranks/lane
                elif k >= 16:
                    add("coop_topk", block=256, group=g, split=split_now, blocks_per_cu=bpc)  # coop from medium K
                    add(jname(split_now), block=128, split=split_now, blocks_per_cu=bpc)
                else:
                    add(jname(split_now), block=128, split=split_now, blocks_per_cu=bpc)
                    add(jname(split_now), block=512, split=split_now, blocks_per_cu=bpc)
                inv_bpc = TOPK_SPLIT_BLOCKS_PER_CU if k >= COOP_K_MIN else bpc
                add(base.name, **{**bp, "split": other_split, "blocks_per_cu": inv_bpc})   # opposite partition
            else:
                if "levels" in bp:  # LSE/SSMW: other insertion level (registers)
                    add(jname(split_now), block=bp["block"], split=split_now, blocks_per_cu=bpc,
                        levels=3 - bp["levels"])
                add(jname(split_now), block=128, split=split_now, blocks_per_cu=bpc)        # occupancy
                add(jname(other_split), block=256, split=other_split, blocks_per_cu=bpc)    # opposite partition
                if split_now == "auto":
                    add("jsplit", block=256, split="auto", blocks_per_cu=alt_bpc)         # other partition granularity
                else:
                    add("row", block=256, tile_j=max(1, bp["tile_j"] // 4), split="off")  # small tile (large D)
            return cands[:MEASURE_MAX]
        blocks = (128, 256, 512)
        splits = (("off", None), ("auto", 8), ("auto", 32))
        if kmin:
            groups = [gg for gg in (2, 4, 8, 16, 32) if 4 <= -(-k // gg) <= 32]
            for b in (128, 256):
                for sp, bpc in splits[:2]:
                    for gg in groups:
                        add("coop_topk", block=b, group=gg, split=sp, blocks_per_cu=bpc)
            if k <= 128:
                for b in blocks:
                    for sp, bpc in splits:
                        add(jname(sp), block=b, split=sp, blocks_per_cu=bpc)
        else:
            for b in blocks:
                for sp, bpc in splits:
                    add(jname(sp), block=b, split=sp, blocks_per_cu=bpc)
                    add(jname(sp), block=b, split=sp, blocks_per_cu=bpc, tile_j=max(1, b // 4))
            for sp, bpc in splits:
                add(jname(sp), block=64, split=sp, blocks_per_cu=bpc)
            if "levels" in bp:  # LSE/SSMW: the other insertion level, default tile
                for b in blocks:
                    for sp, bpc in splits:
                        add(jname(sp), block=b, split=sp, blocks_per_cu=bpc, levels=3 - bp["levels"])
        return cands[:PATIENT_MAX]

    # -- compilation ---------------------------------------------------------
    def compile(self, plan: Plan, device, variant: Variant | None = None) -> HipCompiledReduction:
        sup = self.supports(plan, device)
        if not sup.ok:
            raise UnsupportedError(sup.reason)
        torch = _torch()
        dev = _device_index(device)
        props = torch.cuda.get_device_properties(dev)
        arch = props.gcnArchName
        opts = self.options if variant is None else self._options_for(variant)
        cfg = self._valid(plan, dev, opts) if variant is not None else self._config(plan, dev, opts)
        if cfg is None:
            raise UnsupportedError(f"invalid variant for {arch}: {variant}")
        eff_variant = self._variant_of(cfg, opts)
        cu = props.multi_processor_count
        temp_bound = 0 if opts.split == "off" else opts.temp_max_bytes
        if plan.reduction.kind in KMIN_KINDS and plan.reduction.k == 0:
            # K = 0: no launch, hence nothing to compile (object without a native function).
            info = HipCompileInfo("hip", props.name, arch, self._compiler_id(), tuple(opts.flags(arch, plan)),
                                   True, 0.0, 0, "", eff_variant)
            return HipCompiledReduction(plan, dev, None, info, cfg, opts, cu)

        pair = self._pair_provider(plan)
        src = kernel_source(plan, pair.source, pair.name, cfg)
        src_sha = hashlib.sha256(src.encode()).hexdigest()
        flags = opts.flags(arch, plan)
        compiler = self._compiler_id()
        key = _cache.make_key({
            "plan_key": _plan_key_or_none(plan),
            "generator": GENERATOR_VERSION,
            "source_sha256": src_sha,
            "compiler": compiler,
            "options": flags,
            "arch": arch,
            "variant": eff_variant.as_json(),
        })
        with self._lock:
            hit = self._mem.get((key, dev))
        if hit is not None:
            fns, info0 = hit
            info = HipCompileInfo(**{**info0.__dict__, "cache_hit": True, "compile_seconds": 0.0})
            cr = HipCompiledReduction(plan, dev, fns, info, cfg, opts, cu)
            cr.source, cr.cache_key = src, key
            return cr

        t0 = time.perf_counter()

        def build() -> bytes:
            return _hip.hiprtc().compile(src, "kp_reduce.hip", flags).code

        code, disk_hit = _cache.get_or_build(key, build)
        fns = self._load(key, dev, code, build, bool(plan.ranges))
        dt = time.perf_counter() - t0
        info = HipCompileInfo(
            variant=eff_variant, backend="hip", device_name=props.name, arch=arch, compiler=compiler,
            options=tuple(flags) + _scheme_options(plan, cfg), cache_hit=disk_hit, compile_seconds=dt,
            temp_bytes_per_launch=temp_bound, source_sha256=src_sha,
        )
        with self._lock:
            self._mem[(key, dev)] = (fns, info)
        cr = HipCompiledReduction(plan, dev, fns, info, cfg, opts, cu)
        cr.source, cr.cache_key = src, key
        return cr

    @staticmethod
    def _load(key: str, dev: int, code: bytes, build, ranges: bool = False):
        torch = _torch()
        rt = _hip.hip_runtime()
        with _modules_lock:
            got = _modules.get((key, dev))
            if got is not None:
                return got[1:]
            with torch.cuda.device(dev):
                try:
                    mod = rt.module_load(code)
                except _hip.HipError:
                    # corrupted artifact: invalidate, then recompile
                    _cache.invalidate(key)
                    code, _ = _cache.get_or_build(key, build)
                    mod = rt.module_load(code)
                fns = (rt.get_function(mod, KERNEL_NAME), rt.get_function(mod, MERGE_NAME))
                if ranges:
                    fns += (rt.get_function(mod, RANGES_NAME),)
            _modules[(key, dev)] = (mod,) + fns
            return fns

    @staticmethod
    def _compiler_id() -> str:
        rtc = _hip.hiprtc()
        rt = _hip.hip_runtime()
        maj, mi = rtc.version()
        return f"hiprtc {maj}.{mi} (HIP runtime {rt.runtime_version()}, {rtc.path})"
