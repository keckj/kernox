"""Generation of the tiled fused-reduction HIP kernels.

A module contains two kernels:

``kp_reduce``: grid (gx, S). Block (bx, s) handles rows ``i`` over the range
``j ∈ [s·jchunk, min(N, (s+1)·jchunk))``; it cooperatively loads tiles of
``TJ`` rows ``j`` into LDS and walks through each tile. If ``S == 1``
(``partial == 0``) it writes the outputs directly; otherwise it writes its
partial state to a temporary ``[S, M, W]`` buffer (the "j partition" variant).

``kp_merge``: one thread per row ``i``; merges the S partials in increasing
order of ``s`` (hence of ``j``) and writes the outputs.

Traversal strategies (chosen at compile time, ``KernelConfig.strategy``):

- ``thread``: one thread per row ``i``; accumulators in registers; ``j`` is
  traversed in strictly increasing order, so strict comparisons give "smallest
  index on ties". KMIN: sorted list of K (value, index) pairs per thread,
  insertion through unrolled swaps (registers; spills for large K).
- ``coop`` (KMIN/ARGKMIN only): G consecutive lanes (G divides the wavefront
  size, G <= 32) per row ``i``. The sorted list of ``G·L >= K`` elements is
  **distributed**: lane ``t`` holds ranks ``[t·L, (t+1)·L)`` in registers
  (L ≈ 16 instead of K). Each lane evaluates a different ``j``, then the G
  candidates are broadcast (``__shfl``) and inserted one after the other; an
  insertion shifts the list by one rank, and the element leaving one lane
  enters at the head of the next one (``__shfl_up``). Full lexicographic
  comparison (value, j): the insertion order does not matter.

No assumption on the wavefront size (32 on gfx12, 64 on CDNA):
synchronization through ``__syncthreads``, shuffles restricted to aligned
groups of G <= 32 lanes. Blocks of 256 threads.

LOGSUMEXP / SUMSOFTMAXWEIGHT (contract v0.5, ``thread`` strategy): state
``(m, s[, t_e])`` in ``kp_acc`` per component, shifted by the running maximum;
the tile state is merged into the row state at the end of each tile (two-level
summation, as for SUM), then the partial states are merged by ``kp_merge``. A
term ``f = −inf`` is ignored (zero weight); ``m`` is never NaN (NaNs live in
``s``/``t``); when the maxima are equal the factor is exactly 1, which gives
``exp(−inf − (−inf)) = 0`` for an empty state and no spurious NaN at +inf.
Exact libm ``exp``/``log`` (``expf``/``exp``, ``logf``/``log``).

Semantics: NaN is filtered out (``v == v``) or ignored through strict
comparison; an empty slot (index −1) is beaten by any candidate; ±inf are valid.
"Sorted list" KMIN algorithm: same idea as KeOps (``KMin_ArgKMin_Reduction``),
rewritten; no KeOps code reused.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...plan import Plan, ReductionKind

GENERATOR_VERSION = "hip-v7"

_CTYPE = {"float16": "kp_h", "float32": "float", "float64": "double"}   # kp_h: amendment v0.9
_SIZEOF = {"float16": 2, "float32": 4, "float64": 8}

KMIN_KINDS = (ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN)
SOFT_KINDS = (ReductionKind.LOGSUMEXP, ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT)
# LSE/SSMW: direct (one-level) insertion with FP64 accumulation or beyond this state size
# per row i (the local tile state doubles the register count). R9700 measurements:
# SSMW FP64 E64 2·10⁴² 189.7 → 100.9 ms (spills 852 → 140 B), E16 57.4 → 47.9,
# E1 36.8 → 31.8, LSE FP64 D16 73.5 → 69.0; FP32 neutral (±3 %) → two levels kept
# below the threshold (better accumulation), one level above it (FP32 E64: 212 → 146 VGPRs).
SOFT_ONE_LEVEL_BYTES = 256
ELEM_KINDS = (ReductionKind.SUM, ReductionKind.COUNT, ReductionKind.MIN, ReductionKind.MAX,
              ReductionKind.ARGMIN, ReductionKind.ARGMAX, ReductionKind.MIN_ARGMIN,
              ReductionKind.MAX_ARGMAX)
# amendment v0.8: MIN_ARGMIN / MAX_ARGMAX = ARGMIN / ARGMAX that also return the value
ARGLIKE = (ReductionKind.ARGMIN, ReductionKind.MIN_ARGMIN)
ARGLIKE_MAX = (ReductionKind.ARGMAX, ReductionKind.MAX_ARGMAX)
S_MAX = 64  # maximum number of partials (pointer array of the top-K merge)

# Thresholds measured on an idle GPU (R9700 only),
# provisional, to be confirmed on other cards:
COOP_K_MIN = 32      # KMIN: coop from K = 32 (D <= 48 FP32, FP64 D3: ×0.68 to ×0.95 vs thread)
# Heavy j row (sum of the dimensions of the j arguments >= COOP_HEAVY_D): the pair computation
# dominates and coop G=2 loses at K=32 (D64: thread ×0.92) but wins from K=48 on (D64: ×0.53).
COOP_HEAVY_D = 64
COOP_K_MIN_HEAVY = 48
COOP_SLOTS = 24      # target ranks per lane: K=48 → G2 (24 ranks) beats G4 (D64 ×0.61, FP64 D3 ×0.85)
# R9700: K=128 → G16 (8 ranks/lane) beats G8 in all 4 measured cases (×0.79 to ×0.84);
# K=256 → G16 beats G8 and G32. Above K=64 we therefore use G=16 (provisional).
COOP_G_LARGE_K = 16
COOP_UNROLL_MAX = 256  # G·ranks/lane above which the candidate loop is not unrolled
COOP_MAX_SLOTS = 32   # maximum ranks per lane (registers); G <= 32 (wave32) → K <= 1024
THREAD_K_MAX = 256    # thread strategy: whole list in a single thread
# j partition of top-K reductions with K >= COOP_K_MIN (coop or thread): the merge costs S·M·K,
# so only partition below TOPK_SPLIT_BELOW_BLOCKS_PER_CU blocks/CU, with S sized for
# TOPK_SPLIT_BLOCKS_PER_CU blocks/CU. Measurements (G16, K=128, N=2·10⁵): M=1024 (2 blocks/CU)
# S=4 beats S=1 (×0.80/×0.89) and S=16 (×0.76); M=2048 (4 blocks/CU) S=2 only ×0.97;
# from 8 blocks/CU on, S=1 is best. Thread D64 K=32 (6 blocks/CU): S=3 ×1.15, S=6 ×1.5.
# G2/G4 at M=50,000 (12 to 24 blocks/CU) lose ×1.1 to ×1.4 with a partition.
TOPK_SPLIT_BELOW_BLOCKS_PER_CU = 4
TOPK_SPLIT_BLOCKS_PER_CU = 8


def coop_k_min(plan: Plan) -> int:
    """K from which coop is the default, depending on the width of the j row."""
    d = sum(a.dim for a in plan.args_of("j"))
    return COOP_K_MIN_HEAVY if d >= COOP_HEAVY_D else COOP_K_MIN


@dataclass(frozen=True)
class KernelConfig:
    block: int        # threads per block (multiple of 64)
    tile_j: int       # j rows per LDS tile
    strategy: str = "thread"  # "thread" | "coop"
    group: int = 1    # lanes per row i (coop)
    levels: int = 0   # LSE/SSMW: 2 = local state per tile then merge, 1 = direct insertion; 0 otherwise

    @property
    def rows_per_block(self) -> int:
        return self.block // self.group

    def slots(self, k: int) -> int:
        return -(-k // self.group) if self.strategy == "coop" else k

    def lds_bytes(self, plan: Plan) -> int:
        dj = sum(a.dim for a in plan.args_of("j"))
        return self.tile_j * dj * _SIZEOF[plan.dtype]


def coop_group(k: int, dtype: str = "float32") -> int:
    if k > THREAD_K_MAX:
        # Measurements (d² D3, 2·10⁴²): K=512 FP32 G32 136.7 ms vs G16 217.7;
        # FP64 G16 106.8 vs G32 252.7 (G32: 249 VGPRs, occupancy 5). At most
        # COOP_MAX_SLOTS ranks per lane, hence G=32 above K=512 (provisional).
        if dtype != "float64" or -(-k // COOP_G_LARGE_K) > COOP_MAX_SLOTS:
            return 32
        return COOP_G_LARGE_K
    if k > 64:
        return COOP_G_LARGE_K
    g = 2
    while g < 32 and -(-k // g) > COOP_SLOTS:
        g *= 2
    return g


def soft_state_bytes(plan: Plan) -> int:
    """Size of the per-row-i state of LSE/SSMW reductions (0 for the others)."""
    kind, d = plan.reduction.kind, plan.out_dim
    if kind is ReductionKind.LOGSUMEXP:
        n = 2 * d
    elif kind in (ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT):
        n = 1 + d
    else:
        return 0
    return n * _SIZEOF[plan.acc_dtype]


def soft_levels_auto(plan: Plan) -> int:
    """Direct insertion (one level) with FP64 accumulation or when the state exceeds
    ``SOFT_ONE_LEVEL_BYTES``; otherwise a local tile state then a merge (two levels)."""
    if plan.acc_dtype == "float64" or soft_state_bytes(plan) > SOFT_ONE_LEVEL_BYTES:
        return 1
    return 2


def choose_config(plan: Plan, lds_limit: int, block: int = 256, strategy: str = "auto",
                  group: int | None = None, tile_j: int | None = None,
                  levels: int | None = None) -> KernelConfig | None:
    """Tile bounded by the LDS; ``None`` if a single j row does not fit."""
    kind, k = plan.reduction.kind, plan.reduction.k
    if kind in SOFT_KINDS:
        # sum_scheme (amendment v0.7): block = fresh tile state, then merged (2 levels);
        # kahan = single compensated state (1 level); direct = state size rule.
        forced = {"block": 2, "kahan": 1}.get(plan.sum_scheme)
        if levels is None:
            levels = forced or soft_levels_auto(plan)
        if levels not in (1, 2):
            raise ValueError("LSE/SSMW levels: 1 or 2")
        if forced is not None and levels != forced:
            raise ValueError(f"sum_scheme={plan.sum_scheme} requires levels={forced}")
    else:
        levels = 0
    if strategy == "auto":
        strategy = "coop" if kind in KMIN_KINDS and k is not None and k >= coop_k_min(plan) else "thread"
    if strategy == "thread" and kind in KMIN_KINDS and (k or 0) > THREAD_K_MAX:
        raise ValueError(f"thread strategy: K <= {THREAD_K_MAX} required")
    if strategy == "coop":
        if kind not in KMIN_KINDS:
            raise ValueError("coop strategy is restricted to KMIN/ARGKMIN")
        g = group or coop_group(k or 1, plan.dtype)
        if g not in (2, 4, 8, 16, 32):
            raise ValueError("coop group: 2, 4, 8, 16 or 32")
    else:
        g = 1
    row = sum(a.dim for a in plan.args_of("j")) * _SIZEOF[plan.dtype]
    budget = min(lds_limit, 32 * 1024)
    if row == 0:
        tj = block
    else:
        tj = min(block, budget // row)
        if tj < 1:
            tj = 1 if row <= lds_limit else 0
        if tj < 1:
            return None
    if tile_j is not None:
        if tile_j < 1 or tile_j * row > lds_limit:
            raise ValueError("invalid tile_j")
        tj = tile_j
    return KernelConfig(block, tj, strategy, g, levels)


def partial_layout(plan: Plan):
    """(width W, C type of the partial values or None, partial indices?)."""
    kind, d = plan.reduction.kind, plan.out_dim
    kahan = plan.sum_scheme == "kahan"
    if kind is ReductionKind.SUM:
        return (2 * d if kahan else d), "kp_acc", False   # kahan: (s_c..., c_c...)
    if kind is ReductionKind.COUNT:
        return d, "kp_i64", False
    if kind in (ReductionKind.MIN, ReductionKind.MAX):
        return d, "kp_t", False
    if kind in ARGLIKE + ARGLIKE_MAX:
        return d, "kp_t", True
    if kind is ReductionKind.LOGSUMEXP:      # (m_c, s_c); kahan (m_c, s_c, cs_c)
        return (3 if kahan else 2) * d, "kp_acc", False
    if kind in (ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT):  # (m, s, t_1..t_E); kahan (m, s, cs, t.., ct..)
        return (1 + 2 * d if kahan else 1 + d), "kp_acc", False
    return plan.reduction.k, "kp_t", True


def partial_bytes(plan: Plan, S: int, M: int) -> int:
    W, vt, has_i = partial_layout(plan)
    vs = {"kp_acc": _SIZEOF[plan.acc_dtype], "kp_i64": 8, "kp_t": _SIZEOF[plan.dtype]}[vt]
    return S * M * W * (vs + (8 if has_i else 0))


def _inf(ct: str) -> str:
    return "__builtin_inf()" if ct == "double" else "__builtin_inff()"


_SOFT_HELPERS = """__device__ __forceinline__ float kp_exp(float x) { return expf(x); }
__device__ __forceinline__ double kp_exp(double x) { return exp(x); }
__device__ __forceinline__ float kp_log(float x) { return logf(x); }
__device__ __forceinline__ double kp_log(double x) { return log(x); }
__device__ __forceinline__ float kp_nan(float) { return __builtin_nanf(""); }
__device__ __forceinline__ double kp_nan(double) { return __builtin_nan(""); }
"""


def _soft_names(kind, d):
    """State components: list of (m, s, [t...], score, [weights]) as C names."""
    if kind is ReductionKind.LOGSUMEXP:
        return [(f"m{c}", f"s{c}", [], f"r.o{c}", []) for c in range(d)]
    return [("m0", "s0", [f"t{e}" for e in range(d - 1)], "r.o0", [f"r.o{1 + e}" for e in range(d - 1)])]


def _soft_insert(w, ind, m, s, ts, v, gs, kahan=False):
    """Insert a term v (weight g) into the state (m, s, t); v == −inf is ignored.
    ``kahan``: compensations ``c<s>``, ``c<t>`` (Neumaier), rescaled along with s and t."""
    w(f"{ind}{{ const kp_acc v = (kp_acc)({v});")
    for e, g in enumerate(gs):
        w(f"{ind}  const kp_acc g{e} = (kp_acc)({g});")
    w(f"{ind}  if (v != -KP_AINF) {{")
    if not kahan:
        w(f"{ind}    if (v > {m}) {{ const kp_acc c = kp_exp({m} - v); {s} = {s} * c + (kp_acc)1;"
          + "".join(f" {t} = {t} * c + g{e};" for e, t in enumerate(ts)) + f" {m} = v; }}")
        w(f"{ind}    else {{ const kp_acc q = (v == {m}) ? (kp_acc)1 : kp_exp(v - {m}); {s} += q;"
          + "".join(f" {t} += q * g{e};" for e, t in enumerate(ts)) + " }")
    else:
        sc = " ".join(f"{x} *= c; c{x} *= c;" for x in [s] + ts)
        w(f"{ind}    if (v > {m}) {{ const kp_acc c = kp_exp({m} - v); {sc}"
          f" kp_kadd({s}, c{s}, (kp_acc)1);" + "".join(f" kp_kadd({t}, c{t}, g{e});" for e, t in enumerate(ts))
          + f" {m} = v; }}")
        w(f"{ind}    else {{ const kp_acc q = (v == {m}) ? (kp_acc)1 : kp_exp(v - {m}); kp_kadd({s}, c{s}, q);"
          + "".join(f" kp_kadd({t}, c{t}, q * g{e});" for e, t in enumerate(ts)) + " }")
    w(f"{ind}  }}")
    w(f"{ind}}}")


def _soft_merge(w, ind, m, s, ts, mb, sb, tbs, cb=None):
    """Merge state b into (m, s, t); the factor is exactly 1 when the maxima are equal.
    ``cb`` (kahan): compensations of b, aligned with [sb] + tbs; TwoSum merge."""
    if cb is None:
        w(f"{ind}if ({mb} > {m}) {{ const kp_acc c = kp_exp({m} - {mb}); {s} = {s} * c + {sb};"
          + "".join(f" {t} = {t} * c + {tb};" for t, tb in zip(ts, tbs)) + f" {m} = {mb}; }}")
        w(f"{ind}else {{ const kp_acc q = ({mb} == {m}) ? (kp_acc)1 : kp_exp({mb} - {m}); {s} += {sb} * q;"
          + "".join(f" {t} += {tb} * q;" for t, tb in zip(ts, tbs)) + " }")
        return
    xs, bs = [s] + ts, [sb] + list(tbs)
    w(f"{ind}if ({mb} > {m}) {{ const kp_acc c = kp_exp({m} - {mb});"
      + "".join(f" {x} *= c; c{x} *= c; kp_kmerge({x}, c{x}, {b}, {cbx});" for x, b, cbx in zip(xs, bs, cb))
      + f" {m} = {mb}; }}")
    w(f"{ind}else {{ const kp_acc q = ({mb} == {m}) ? (kp_acc)1 : kp_exp({mb} - {m});"
      + "".join(f" kp_kmerge({x}, c{x}, {b} * q, {cbx} * q);" for x, b, cbx in zip(xs, bs, cb)) + " }")


def _soft_finish(w, ind, kind, m, s, ts, dst, kahan=False):
    """Write the result: LSE ``s NaN → NaN; m = +inf → +inf; else m + log s``;
    SSMW ``m = +inf → NaN; else t / s`` (0/0 = NaN if empty); LSEW (amendment v0.8)
    ``m + log t`` with no special case. ``kahan``: s and t are compensated (``kp_kfin``) first."""
    if kahan:
        for x in [s] + ts:
            w(f"{ind}{x} = kp_kfin({x}, c{x});")
    if kind is ReductionKind.LOGSUMEXP:
        w(f"{ind}{dst(0)} = ({s} != {s}) ? {s} : (({m} == KP_AINF) ? {m} : {m} + kp_log({s}));")
    elif kind is ReductionKind.LOGSUMEXPWEIGHT:
        for e, t in enumerate(ts):
            w(f"{ind}{dst(e)} = {m} + kp_log({t});")
    else:
        for e, t in enumerate(ts):
            w(f"{ind}{dst(e)} = ({m} == KP_AINF) ? kp_nan((kp_acc)0) : {t} / {s};")


_KAHAN_HELPERS = """// Kahan-Babuska (Neumaier), amendment v0.7: additions/subtractions only
__device__ __forceinline__ kp_acc kp_kabs(kp_acc x) { return x < (kp_acc)0 ? -x : x; }
__device__ __forceinline__ void kp_kadd(kp_acc& s, kp_acc& c, const kp_acc x) {
  const kp_acc t = s + x;
  c += (kp_kabs(s) >= kp_kabs(x)) ? ((s - t) + x) : ((x - t) + s);
  s = t;
}
__device__ __forceinline__ void kp_kmerge(kp_acc& s, kp_acc& c, const kp_acc s2, const kp_acc c2) {
  const kp_acc t = s + s2; const kp_acc bp = t - s;            // TwoSum(s, s2)
  const kp_acc e = (s - (t - bp)) + (s2 - bp);
  c = (c + c2) + e; s = t;
}
// non-finite sum (±inf, NaN): s as is (c = inf − inf would be NaN)
__device__ __forceinline__ kp_acc kp_kfin(kp_acc s, kp_acc c) { return (s - s == (kp_acc)0) ? s + c : s; }
"""


_BETTER = """__device__ __forceinline__ bool kp_better(kp_t v, kp_i64 j, kp_t bv, kp_i64 bj) {
  // lexicographic order (value, j); an empty slot (bj < 0) is beaten by any candidate
  return j >= 0 && (bj < 0 || v < bv || (v == bv && j < bj));
}
"""


_HALF_KERNEL = """// FP16 (amendment v0.9): lane exchanges through float (exact conversion)
__device__ __forceinline__ kp_h __shfl(kp_h v, int src, int w) { return kp_h(__shfl((float)v, src, w)); }
__device__ __forceinline__ kp_h __shfl_up(kp_h v, unsigned int d, int w) { return kp_h(__shfl_up((float)v, d, w)); }
"""
_HALF_SOFT = """__device__ __forceinline__ kp_h kp_exp(kp_h x) { return kp_h(expf((float)x)); }
__device__ __forceinline__ kp_h kp_log(kp_h x) { return kp_h(logf((float)x)); }
__device__ __forceinline__ kp_h kp_nan(kp_h) { return kp_h(__builtin_nanf("")); }
"""


def kernel_source(plan: Plan, pair_source: str, pair_name: str, cfg: KernelConfig) -> str:
    from ...codegen.pair import HALF_PRELUDE
    from ...plan import accumulated_output_dtype

    T = _CTYPE[plan.dtype]
    A = _CTYPE[plan.acc_dtype]
    OA = _CTYPE[accumulated_output_dtype(plan)]   # FP16: accumulated outputs in FP16 (v0.9 b)
    kind = plan.reduction.kind
    d = plan.out_dim
    K = plan.reduction.k
    BS, TJ, G = cfg.block, cfg.tile_j, cfg.group
    R = cfg.rows_per_block
    coop = cfg.strategy == "coop"
    inf = _inf(T)
    W, pvt, has_pi = partial_layout(plan)
    names_out = {
        ReductionKind.SUM: [("value", "kp_oacc")], ReductionKind.COUNT: [("value", "kp_i64")],
        ReductionKind.MIN: [("value", "kp_t")], ReductionKind.MAX: [("value", "kp_t")],
        ReductionKind.ARGMIN: [("index", "kp_i64")], ReductionKind.ARGMAX: [("index", "kp_i64")],
        ReductionKind.KMIN: [("value", "kp_t")], ReductionKind.ARGKMIN: [("index", "kp_i64")],
        ReductionKind.KMIN_ARGKMIN: [("value", "kp_t"), ("index", "kp_i64")],
        ReductionKind.LOGSUMEXP: [("value", "kp_oacc")],
        ReductionKind.SUMSOFTMAXWEIGHT: [("value", "kp_oacc")],
        ReductionKind.LOGSUMEXPWEIGHT: [("value", "kp_oacc")],
        ReductionKind.MIN_ARGMIN: [("value", "kp_t"), ("index", "kp_i64")],
        ReductionKind.MAX_ARGMAX: [("value", "kp_t"), ("index", "kp_i64")],
    }[kind]
    soft = kind in SOFT_KINDS
    kahan = plan.sum_scheme == "kahan" and (soft or kind is ReductionKind.SUM)
    lp = "l" if cfg.levels == 2 else ""  # prefix of the local tile state (two levels)
    if soft and coop:
        raise ValueError("coop strategy is restricted to KMIN/ARGKMIN")
    out_params = [f"{ct}* __restrict__ out_{n}" for n, ct in names_out]
    part_params = [f"{pvt}* __restrict__ pv", "kp_i64* __restrict__ pi"]

    L = []
    w = L.append
    w("// Generated by kernox.backends.hip (" + GENERATOR_VERSION + f", strategy={cfg.strategy}, G={G})")
    if plan.dtype == "float16":
        w(HALF_PRELUDE)
        w(_HALF_KERNEL)
    w(f"typedef {T} kp_t;")
    w(f"typedef {A} kp_acc;")
    w(f"typedef {OA} kp_oacc;")
    w("typedef long long kp_i64;")
    w("")
    w(pair_source)
    w("")
    w(_BETTER)
    if soft:
        w(f"#define KP_AINF {_inf(A)}")
        w(_SOFT_HELPERS)
        if plan.acc_dtype == "float16":
            w(_HALF_SOFT)
    if kahan:
        w(_KAHAN_HELPERS)

    # ======================================================== kp_reduce
    params = [f"const kp_t* __restrict__ a{n}" for n, _a in enumerate(plan.args)]
    params += ["const kp_i64 M", "const kp_i64 N", "const kp_i64 jchunk", "const int partial"]
    params += out_params + part_params
    w(f'extern "C" __global__ __launch_bounds__({BS}) void kp_reduce(')
    w("    " + ",\n    ".join(params) + ")")
    w("{")
    for n, a in enumerate(plan.args):
        if a.role == "j":
            w(f"  __shared__ kp_t sh{n}[{TJ * a.dim}];")
    if coop:
        w(f"  const int lane = threadIdx.x % {G};")
        w(f"  const kp_i64 i = (kp_i64)blockIdx.x * {R} + (kp_i64)(threadIdx.x / {G});")
    else:
        w(f"  const kp_i64 i = (kp_i64)blockIdx.x * {R} + (kp_i64)threadIdx.x;")
    w("  const bool active = i < M;")
    w("  const kp_i64 s = blockIdx.y;")
    w("  const kp_i64 jb = s * jchunk;")
    w("  const kp_i64 je = (jb + jchunk < N) ? (jb + jchunk) : N;")
    for n, a in enumerate(plan.args):
        if a.role == "i":
            w(f"  kp_t x{n}[{a.dim}];")
            w("  #pragma unroll")
            w(f"  for (int c = 0; c < {a.dim}; ++c) x{n}[c] = active ? a{n}[i * {a.dim} + c] : (kp_t)0;")

    # accumulators
    if kind is ReductionKind.SUM:
        for c in range(d):
            w(f"  kp_acc acc{c} = (kp_acc)0;" + (f" kp_acc cacc{c} = (kp_acc)0;" if kahan else ""))
    elif kind is ReductionKind.COUNT:
        for c in range(d):
            w(f"  kp_i64 acc{c} = 0;")
    elif kind in (ReductionKind.MIN,) + ARGLIKE:
        for c in range(d):
            w(f"  kp_t acc{c} = {inf}; kp_i64 idx{c} = -1;")
    elif kind in (ReductionKind.MAX,) + ARGLIKE_MAX:
        for c in range(d):
            w(f"  kp_t acc{c} = -{inf}; kp_i64 idx{c} = -1;")
    elif soft:
        for m, s, ts, _v, _g in _soft_names(kind, d):
            w(f"  kp_acc {m} = -KP_AINF, {s} = (kp_acc)0;" + "".join(f" kp_acc {t} = (kp_acc)0;" for t in ts)
              + ("".join(f" kp_acc c{x} = (kp_acc)0;" for x in [s] + ts) if kahan else ""))
    else:
        S_ = cfg.slots(K)
        w(f"  kp_t kv[{S_}]; kp_i64 ki[{S_}];")
        w("  #pragma unroll")
        w(f"  for (int k = 0; k < {S_}; ++k) {{ kv[k] = {inf}; ki[k] = -1; }}")

    def call(jj):
        args = []
        for n, a in enumerate(plan.args):
            if a.role == "i":
                args.append(f"x{n}")
            elif a.role == "j":
                args.append(f"sh{n} + ({jj}) * {a.dim}")
            else:
                args.append(f"a{n}")
        return f"{pair_name}({', '.join(args)})"

    w(f"  for (kp_i64 j0 = jb; j0 < je; j0 += {TJ}) {{")
    w(f"    const int nt = (int)((je - j0) < {TJ} ? (je - j0) : {TJ});")
    w("    __syncthreads();")
    for n, a in enumerate(plan.args):
        if a.role == "j":
            w(f"    for (int t = threadIdx.x; t < nt * {a.dim}; t += {BS}) sh{n}[t] = a{n}[j0 * {a.dim} + t];")
    w("    __syncthreads();")
    if not coop:
        w("    if (active) {")
        if kind is ReductionKind.SUM and not kahan:
            for c in range(d):
                w(f"      kp_acc t{c} = (kp_acc)0;")
        elif soft:
            for m, s, ts, _v, _g in _soft_names(kind, d) if lp else ():
                w(f"      kp_acc l{m} = -KP_AINF, l{s} = (kp_acc)0;" + "".join(f" kp_acc l{t} = (kp_acc)0;" for t in ts))
        w("      for (int jj = 0; jj < nt; ++jj) {")
        w(f"        const auto r = {call('jj')};")
        w("        const kp_i64 j = j0 + jj; (void)j;")
        if kind is ReductionKind.SUM and kahan:
            for c in range(d):
                w(f"        kp_kadd(acc{c}, cacc{c}, (kp_acc)r.o{c});")
        elif kind is ReductionKind.SUM:
            for c in range(d):
                w(f"        t{c} += (kp_acc)r.o{c};")
        elif kind is ReductionKind.COUNT:
            for c in range(d):
                w(f"        acc{c} += r.o{c} ? 1 : 0;")
        elif soft:
            for m, s, ts, v, gs in _soft_names(kind, d):
                _soft_insert(w, "        ", lp + m, lp + s, [lp + t for t in ts], v, gs, kahan)
        elif kind is ReductionKind.MIN:
            for c in range(d):
                w(f"        {{ const kp_t v = r.o{c}; if (v < acc{c}) acc{c} = v; }}")
        elif kind is ReductionKind.MAX:
            for c in range(d):
                w(f"        {{ const kp_t v = r.o{c}; if (v > acc{c}) acc{c} = v; }}")
        elif kind in ARGLIKE:
            for c in range(d):
                w(f"        {{ const kp_t v = r.o{c}; if (v < acc{c} || (v == acc{c} && (idx{c} < 0 || j < idx{c}))) {{ acc{c} = v; idx{c} = j; }} }}")
        elif kind in ARGLIKE_MAX:
            for c in range(d):
                w(f"        {{ const kp_t v = r.o{c}; if (v > acc{c} || (v == acc{c} && (idx{c} < 0 || j < idx{c}))) {{ acc{c} = v; idx{c} = j; }} }}")
        else:
            w("        const kp_t v = r.o0;")
            # lexicographic (value, j): a tie with a smaller j seen later (out-of-order ranges)
            # must also be inserted
            w(f"        if (v == v && kp_better(v, j, kv[{K - 1}], ki[{K - 1}])) {{")
            w("          kp_t cv = v; kp_i64 cj = j;")
            w("          #pragma unroll")
            w(f"          for (int k = 0; k < {K}; ++k) {{")
            w("            if (kp_better(cv, cj, kv[k], ki[k])) { const kp_t tv = kv[k]; const kp_i64 tj = ki[k];"
              " kv[k] = cv; ki[k] = cj; cv = tv; cj = tj; }")
            w("          }")
            w("        }")
        w("      }")
        if kind is ReductionKind.SUM and not kahan:
            for c in range(d):
                w(f"      acc{c} += t{c};")
        elif soft:
            for m, s, ts, _v, _g in _soft_names(kind, d) if lp else ():
                _soft_merge(w, "      ", m, s, ts, "l" + m, "l" + s, ["l" + t for t in ts])
        w("    }")
    else:
        Ls = cfg.slots(K)
        w("    if (active) {")
        w(f"      for (int base = 0; base < nt; base += {G}) {{")
        w("        const int jj = base + lane;")
        w(f"        kp_t v = {inf}; kp_i64 jv = -1;")
        w(f"        if (jj < nt) {{ const auto r = {call('jj')}; v = r.o0; jv = (v == v) ? (j0 + jj) : -1; }}")
        # Loop over the G candidates of the group: unrolled only while G·L stays small. FP32
        # K=1024 (G32, L32) unrolled: 28,515 instructions, 542 branches (instruction cache
        # overflow, ×2 slower than FP64, which the compiler did not unroll: 13,387).
        w("        #pragma unroll" + ("" if G * Ls <= COOP_UNROLL_MAX else " 1"))
        w(f"        for (int q = 0; q < {G}; ++q) {{")
        w(f"          const kp_t cv = __shfl(v, q, {G}); const kp_i64 cj = __shfl(jv, q, {G});")
        w(f"          const kp_t lv = __shfl(kv[{Ls - 1}], {G - 1}, {G}); const kp_i64 lj = __shfl(ki[{Ls - 1}], {G - 1}, {G});")
        w("          if (kp_better(cv, cj, lv, lj)) {  // uniform within the group")
        w(f"            const bool aff = kp_better(cv, cj, kv[{Ls - 1}], ki[{Ls - 1}]);")
        w(f"            const kp_t pv_ = __shfl_up(kv[{Ls - 1}], 1, {G}); const kp_i64 pj_ = __shfl_up(ki[{Ls - 1}], 1, {G});")
        w(f"            const int paff_ = __shfl_up((int)aff, 1, {G}); const int paff = lane > 0 ? paff_ : 0;")
        w("            if (aff) {")
        w("              kp_t xv = paff ? pv_ : cv; kp_i64 xj = paff ? pj_ : cj;")
        w("              #pragma unroll")
        w(f"              for (int k = 0; k < {Ls}; ++k) {{")
        w("                if (kp_better(xv, xj, kv[k], ki[k])) { const kp_t tv = kv[k]; const kp_i64 tj = ki[k];"
          " kv[k] = xv; ki[k] = xj; xv = tv; xj = tj; }")
        w("              }")
        w("            }")
        w("          }")
        w("        }")
        w("      }")
        w("    }")
    w("  }")

    # write-back
    w("  if (!active) return;")
    if soft:
        W_ = W
        w("  if (partial) {")
        w(f"    kp_acc* p = pv + (s * M + i) * {W_};")
        if kind is ReductionKind.LOGSUMEXP and kahan:
            for c in range(d):
                w(f"    p[{3 * c}] = m{c}; p[{3 * c + 1}] = s{c}; p[{3 * c + 2}] = cs{c};")
        elif kind is ReductionKind.LOGSUMEXP:
            for c in range(d):
                w(f"    p[{2 * c}] = m{c}; p[{2 * c + 1}] = s{c};")
        elif kahan:
            E = d - 1
            w("    p[0] = m0; p[1] = s0; p[2] = cs0;" + "".join(f" p[{3 + e}] = t{e};" for e in range(E))
              + "".join(f" p[{3 + E + e}] = ct{e};" for e in range(E)))
        else:
            w("    p[0] = m0; p[1] = s0;" + "".join(f" p[{2 + e}] = t{e};" for e in range(d - 1)))
        w("    return;")
        w("  }")
        for c, (m, s, ts, _v, _g) in enumerate(_soft_names(kind, d)):
            if kind is ReductionKind.LOGSUMEXP:
                _soft_finish(w, "  ", kind, m, s, ts, lambda e, c=c: f"out_value[i * {d} + {c}]", kahan)
            else:
                _soft_finish(w, "  ", kind, m, s, ts, lambda e: f"out_value[i * {d - 1} + {e}]", kahan)
    elif kind is ReductionKind.SUM and kahan:
        w("  if (partial) {")
        for c in range(d):
            w(f"    pv[(s * M + i) * {2 * d} + {c}] = acc{c}; pv[(s * M + i) * {2 * d} + {d + c}] = cacc{c};")
        w("    return;")
        w("  }")
        for c in range(d):
            w(f"  out_value[i * {d} + {c}] = kp_kfin(acc{c}, cacc{c});")
    elif kind in ELEM_KINDS:
        w("  if (partial) {")
        for c in range(d):
            w(f"    pv[(s * M + i) * {d} + {c}] = acc{c};")
            if has_pi:
                w(f"    pi[(s * M + i) * {d} + {c}] = idx{c};")
        w("    return;")
        w("  }")
        for c in range(d):
            if kind in ARGLIKE + ARGLIKE_MAX:
                w(f"  out_index[i * {d} + {c}] = idx{c};")
            if kind not in (ReductionKind.ARGMIN, ReductionKind.ARGMAX):
                w(f"  out_value[i * {d} + {c}] = acc{c};")
    else:
        Ls = cfg.slots(K)
        off = "lane * %d + k" % Ls if coop else "k"
        w("  #pragma unroll")
        w(f"  for (int k = 0; k < {Ls}; ++k) {{")
        w(f"    const int rk = {off};")
        w(f"    if (rk >= {K}) break;")
        w(f"    if (partial) {{ pv[(s * M + i) * {K} + rk] = kv[k]; pi[(s * M + i) * {K} + rk] = ki[k]; continue; }}")
        if kind in (ReductionKind.KMIN, ReductionKind.KMIN_ARGKMIN):
            w(f"    out_value[i * {K} + rk] = kv[k];")
        if kind in (ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN):
            w(f"    out_index[i * {K} + rk] = ki[k];")
        w("  }")
    w("}")
    w("")

    # ======================================================== kp_merge
    mparams = ["const kp_i64 M", "const int S"] + [f"const {pvt}* __restrict__ pv",
                                                 "const kp_i64* __restrict__ pi"] + out_params
    w(f'extern "C" __global__ __launch_bounds__({BS}) void kp_merge(')
    w("    " + ",\n    ".join(mparams) + ")")
    w("{")
    w(f"  const kp_i64 i = (kp_i64)blockIdx.x * {BS} + (kp_i64)threadIdx.x;")
    w("  if (i >= M) return;")
    if soft:
        names = _soft_names(kind, d)
        for m, s, ts, _v, _g in names:
            w(f"  kp_acc {m} = -KP_AINF, {s} = (kp_acc)0;" + "".join(f" kp_acc {t} = (kp_acc)0;" for t in ts)
              + ("".join(f" kp_acc c{x} = (kp_acc)0;" for x in [s] + ts) if kahan else ""))
        w("  for (int s = 0; s < S; ++s) {")
        w(f"    const kp_acc* p = pv + ((kp_i64)s * M + i) * {W};")
        if kind is ReductionKind.LOGSUMEXP and kahan:
            for c, (m, s_, ts, _v, _g) in enumerate(names):
                _soft_merge(w, "    ", m, s_, ts, f"p[{3 * c}]", f"p[{3 * c + 1}]", [], [f"p[{3 * c + 2}]"])
        elif kind is ReductionKind.LOGSUMEXP:
            for c, (m, s_, ts, _v, _g) in enumerate(names):
                _soft_merge(w, "    ", m, s_, ts, f"p[{2 * c}]", f"p[{2 * c + 1}]", [])
        elif kahan:
            m, s_, ts, _v, _g = names[0]
            E = d - 1
            _soft_merge(w, "    ", m, s_, ts, "p[0]", "p[1]", [f"p[{3 + e}]" for e in range(E)],
                        ["p[2]"] + [f"p[{3 + E + e}]" for e in range(E)])
        else:
            m, s_, ts, _v, _g = names[0]
            _soft_merge(w, "    ", m, s_, ts, "p[0]", "p[1]", [f"p[{2 + e}]" for e in range(d - 1)])
        w("  }")
        for c, (m, s_, ts, _v, _g) in enumerate(names):
            if kind is ReductionKind.LOGSUMEXP:
                _soft_finish(w, "  ", kind, m, s_, ts, lambda e, c=c: f"out_value[i * {d} + {c}]", kahan)
            else:
                _soft_finish(w, "  ", kind, m, s_, ts, lambda e: f"out_value[i * {d - 1} + {e}]", kahan)
    elif kind is ReductionKind.SUM and kahan:
        for c in range(d):
            w(f"  {{ kp_acc a = (kp_acc)0, ca = (kp_acc)0;")
            w(f"    for (int s = 0; s < S; ++s) {{ const kp_acc* p = pv + ((kp_i64)s * M + i) * {2 * d};"
              f" kp_kmerge(a, ca, p[{c}], p[{d + c}]); }}")
            w(f"    out_value[i * {d} + {c}] = kp_kfin(a, ca); }}")
    elif kind in ELEM_KINDS:
        w(f"  for (int c = 0; c < {d}; ++c) {{")
        if kind is ReductionKind.SUM:
            w("    kp_acc a = (kp_acc)0;")
            w(f"    for (int s = 0; s < S; ++s) a += pv[((kp_i64)s * M + i) * {d} + c];")
            w(f"    out_value[i * {d} + c] = a;")
        elif kind is ReductionKind.COUNT:
            w("    kp_i64 a = 0;")
            w(f"    for (int s = 0; s < S; ++s) a += pv[((kp_i64)s * M + i) * {d} + c];")
            w(f"    out_value[i * {d} + c] = a;")
        elif kind in (ReductionKind.MIN, ReductionKind.MAX):
            op = "<" if kind is ReductionKind.MIN else ">"
            sign = "" if kind is ReductionKind.MIN else "-"
            w(f"    kp_t a = {sign}{inf};")
            w(f"    for (int s = 0; s < S; ++s) {{ const kp_t v = pv[((kp_i64)s * M + i) * {d} + c]; if (v {op} a) a = v; }}")
            w(f"    out_value[i * {d} + c] = a;")
        else:
            op = "<" if kind in ARGLIKE else ">"
            sign = "" if kind in ARGLIKE else "-"
            w(f"    kp_t a = {sign}{inf}; kp_i64 b = -1;")
            w("    for (int s = 0; s < S; ++s) {")
            w(f"      const kp_i64 o = ((kp_i64)s * M + i) * {d} + c;")
            w("      const kp_t v = pv[o]; const kp_i64 j = pi[o];")
            w("      // increasing s = increasing j: on ties, the earlier partial (smaller j) is kept")
            w(f"      if (j >= 0 && (b < 0 || v {op} a)) {{ a = v; b = j; }}")
            w("    }")
            w(f"    out_index[i * {d} + c] = b;")
            if kind not in (ReductionKind.ARGMIN, ReductionKind.ARGMAX):
                w(f"    out_value[i * {d} + c] = a;")
        w("  }")
    else:
        w(f"  int ptr[{S_MAX}];")
        w("  for (int s = 0; s < S; ++s) ptr[s] = 0;")
        w(f"  for (int r = 0; r < {K}; ++r) {{")
        w(f"    kp_t bv = {inf}; kp_i64 bj = -1; int bs = -1;")
        w("    for (int s = 0; s < S; ++s) {")
        w(f"      if (ptr[s] >= {K}) continue;")
        w(f"      const kp_i64 o = ((kp_i64)s * M + i) * {K} + ptr[s];")
        w("      if (kp_better(pv[o], pi[o], bv, bj)) { bv = pv[o]; bj = pi[o]; bs = s; }")
        w("    }")
        w("    if (bs >= 0) ptr[bs] += 1;")
        if kind in (ReductionKind.KMIN, ReductionKind.KMIN_ARGKMIN):
            w(f"    out_value[i * {K} + r] = bv;")
        if kind in (ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN):
            w(f"    out_index[i * {K} + r] = bj;")
        w("  }")
    w("}")
    if plan.batch_rank:
        L = _batched(L, plan, names_out, W)
    if plan.ranges:
        L = _with_ranges_kernel(L, cfg, pvt)
    return "\n".join(L) + "\n"


def _out_widths(plan: Plan) -> dict:
    """Width (elements per row i) of each output."""
    from ...plan import output_shapes
    import dataclasses
    p0 = dataclasses.replace(plan, batch_rank=0, ranges=False)
    out = {}
    for name, (shape, _dt) in output_shapes(p0, 1).items():
        wdt = 1
        for x in shape[1:]:
            wdt *= x
        out[name] = wdt
    return out


def _kernel_span(L, name):
    start = next(n for n, x in enumerate(L) if x.startswith('extern "C" __global__') and f"void {name}(" in x)
    end = next(n for n in range(start, len(L)) if L[n] == "}")
    return start, end


def _batched(L, plan: Plan, names_out, W):
    """Addendum v0.6 §1: batch element ``kp_bz = b0 + blockIdx.z``; inputs shifted by the
    offset table ``boff[arg * nbt + kp_bz]`` (in elements); contiguous ``(nb, M, ·)`` outputs;
    partials ``(nb, S, M, W)``."""
    widths = _out_widths(plan)
    L = list(L)
    extra = ["const kp_i64* __restrict__ boff", "const kp_i64 nbt", "const kp_i64 b0"]
    for kname in ("kp_reduce", "kp_merge"):
        st, _ = _kernel_span(L, kname)
        sig = L[st + 1]
        for n, _a in enumerate(plan.args):
            sig = sig.replace(f"const kp_t* __restrict__ a{n},", f"const kp_t* __restrict__ a{n}_,")
        for nm, ct in names_out:
            sig = sig.replace(f"{ct}* __restrict__ out_{nm}", f"{ct}* __restrict__ out_{nm}_")
        sig = sig.replace("__restrict__ pv,", "__restrict__ pv_,").replace("__restrict__ pi)", "__restrict__ pi_)")
        sig = sig.replace("__restrict__ pi,", "__restrict__ pi_,")
        assert sig.endswith(")")
        L[st + 1] = sig[:-1] + ",\n    " + ",\n    ".join(extra) + ")"
        pro = ["  const kp_i64 kp_bz = b0 + (kp_i64)blockIdx.z;"]
        if kname == "kp_reduce":
            for n, _a in enumerate(plan.args):
                pro.append(f"  const kp_t* __restrict__ a{n} = a{n}_ + boff[{n} * nbt + kp_bz];")
            pro.append("  const kp_i64 kp_S = gridDim.y;")
        else:
            pro.append("  const kp_i64 kp_S = S;")
        for nm, ct in names_out:
            pro.append(f"  {ct}* __restrict__ out_{nm} = out_{nm}_ + kp_bz * M * {widths[nm]};")
        # partials: the original type is kept from the parameter declaration
        decl_pv = next(x for x in sig.split(",") if "pv_" in x).strip().replace("pv_", "pv")
        decl_pi = next(x for x in sig.split(",") if "pi_" in x).strip().rstrip(")").replace("pi_", "pi")
        pro.append(f"  {decl_pv} = pv_ + kp_bz * kp_S * M * {W};")
        pro.append(f"  {decl_pi} = pi_ ? pi_ + kp_bz * kp_S * M * {W} : pi_;")
        L[st + 3:st + 3] = pro
    return L


def _with_ranges_kernel(L, cfg: KernelConfig, pvt: str):
    """Addendum v0.6 §2: add ``kp_reduce_ranges``, derived from ``kp_reduce``: one block per
    work tile ``(i0, i1, k0, k1)`` (height <= rows_per_block), reducing over
    ``∪ red[k0:k1]`` in interval order (ties broken by j), without a j partition;
    rows that are not covered receive the neutral element through an N = 0 launch."""
    st, en = _kernel_span(L, "kp_reduce")
    K_ = list(L[st:en + 1])
    R, G = cfg.rows_per_block, cfg.group
    K_[0] = K_[0].replace("void kp_reduce(", "void kp_reduce_ranges(")
    K_[1] = K_[1].replace("const kp_i64 N,\n    const kp_i64 jchunk,\n    const int partial",
                          "const kp_i64* __restrict__ work,\n    const kp_i64* __restrict__ red")
    assert "work" in K_[1]
    out = []
    for x in K_:
        if x.startswith("  const kp_i64 i = (kp_i64)blockIdx.x * "):
            out.append("  const kp_i64* wk = work + 4 * (kp_i64)blockIdx.x;")
            out.append("  const kp_i64 wi1 = wk[1], wk0 = wk[2], wk1 = wk[3];")
            x = x.replace(f"(kp_i64)blockIdx.x * {R}", "wk[0]")
        elif x == "  const bool active = i < M;":
            x = "  const bool active = i < wi1; (void)M;"
        elif x == "  const kp_i64 s = blockIdx.y;":
            x = (f"  const kp_i64 s = 0; const int partial = 0; {pvt}* pv = nullptr; kp_i64* pi = nullptr;"
                 " (void)s; (void)pv; (void)pi;")
        elif x.startswith("  const kp_i64 jb = ") or x.startswith("  const kp_i64 je = "):
            continue
        elif x.startswith("  for (kp_i64 j0 = jb; j0 < je; j0 += "):
            out.append("  for (kp_i64 kr = wk0; kr < wk1; ++kr) {")
            out.append("  const kp_i64 jb = red[2 * kr], je = red[2 * kr + 1];")
        elif x == "  if (!active) return;":
            out.append("  }")
        out.append(x)
    # pv/pi are no longer parameters
    out[1] = out[1].replace(",\n    kp_t* __restrict__ pv", "").replace(",\n    kp_acc* __restrict__ pv", "")
    out[1] = out[1].replace(",\n    kp_i64* __restrict__ pv", "").replace(",\n    kp_i64* __restrict__ pi", "")
    assert "pv" not in out[1] and "pi" not in out[1].split("work")[1], out[1]
    return list(L) + [""] + out
