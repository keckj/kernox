"""Generation of the fused reduction for the ``cpu`` backend (host C++ + OpenMP).

Strategy:

- work unit ("task"): a block of at most ``BI`` rows i, against one slice of
  the j axis (or, with ranges, against the intervals of one work-list tile).
  Tasks are handed out dynamically to the threads of one OpenMP parallel
  region through an atomic counter: no OpenMP loop scheduling symbol is used,
  only ``GOMP_parallel`` and ``omp_get_thread_num``;
- the j axis of a task is traversed in tiles of ``TJ`` points. Each tile of
  the j arguments is transposed into a thread-local SoA buffer
  (``yb[c * TJ + t]``) and reused by the ``BI`` rows of the block;
- per row and per tile, the pair function is evaluated over the tile in an
  ``omp simd`` loop (one SIMD lane per j point; the j operands are read from
  the SoA buffer, the i row and the parameters are uniform), the values are
  stored in a tile buffer, then reduced with ``LN`` explicit lane accumulators
  folded in a fixed pairwise tree at the end of the tile. The tile result is
  merged into the row state. Padding points (end of an interval) evaluate valid
  data and are then replaced by the identity of the reduction;
- summation: the per-tile sum is a fresh accumulator added to the row total
  (block summation with block ``TJ``), for ``direct`` and ``block`` alike;
  ``kahan``: Neumaier per lane, lanes and tiles merged with TwoSum;
- LOGSUMEXP family: per tile, maximum of the scores first, then sums of
  ``exp(v - max)`` (one exponential per term), merged into the row state with
  the rescaling rule of the contract;
- top-K: sorted list (value, j) per row; a vectorized prefilter skips the
  groups of ``LN`` points that cannot enter the list;
- j split ("split-j"): ``S`` slices processed as independent tasks; the partial
  states go to a temporary (indices int64 first, then values), then a second
  parallel region merges them per row in increasing slice order.

Results only depend on (TJ, slice boundaries), never on the number of threads:
the j split is a function of the sizes only (see ``__init__.split_j``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...plan import Plan, ReductionKind, accumulated_output_dtype, output_shapes
from ..sycl import codegen as _sycl

_CTYPE = {"float16": "kp_h", "float32": "float", "float64": "double"}
_SIZE = {"kp_h": 2, "float": 4, "double": 8, "int64_t": 8}

LN = 16          # SIMD lanes of the explicit accumulators
TILE_J = 256     # default j tile (largest)
TILE_J_MIN = 64  # smallest default j tile
L1_TILE_BYTES = 24 * 1024  # budget of the SoA j tile: stays in a 32 KiB L1 data cache
ROWS = 32        # default rows per task
SOA_MAX_DIM = 64     # beyond: j operands read in place (no transposition, no simd)
SOA_MAX_TOTAL = 128
K_STATE_BUDGET = 256 * 1024  # bytes of top-K row states per thread

SOFT_KINDS = _sycl.SOFT_KINDS
ARG_KINDS = _sycl.ARG_KINDS
FLOAT_ACC_KINDS = _sycl.FLOAT_ACC_KINDS
K_KINDS = (ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN)


@dataclass(frozen=True)
class KernelLayout:
    tile_j: int
    rows: int                      # rows per task (= ranges_tile_m)
    in_names: tuple[str, ...]
    out_names: tuple[str, ...]
    partial_bytes: int             # partial state per (row i, j slice)
    soa: bool


def _port(lines):
    """sycl helper lines -> host C++ (``sycl::exp`` -> ``kp_exp``...)."""
    return [ln.replace("sycl::", "kp_") for ln in lines]


def state_spec(plan: Plan) -> tuple[int, str, bool]:
    w, vt, has_idx = _sycl.state_spec(plan)
    return w, {"sycl::half": "kp_h"}.get(vt, vt), has_idx


def partial_bytes_per_row(plan: Plan) -> int:
    w, vt, has_idx = state_spec(plan)
    return w * (_SIZE[vt] + (8 if has_idx else 0))


def out_names(plan: Plan) -> tuple[str, ...]:
    return tuple(output_shapes(plan, 0).keys())


def rows_for(plan: Plan, bi: int) -> int:
    """Rows per task, reduced for large top-K states."""
    if plan.reduction.kind in K_KINDS:
        w, vt, _ = state_spec(plan)
        bi = min(bi, max(1, K_STATE_BUDGET // (w * (_SIZE[vt] + 4))))
    return bi


def default_tile_j(plan: Plan) -> int:
    """Default j tile: ``TILE_J``, halved (down to ``TILE_J_MIN``) while the SoA
    copy of the j operands of one tile exceeds ``L1_TILE_BYTES``. Each tile is
    reread by the ``BI`` rows of a task, so it must stay in L1: measured on
    EPYC 9654, D = 16 in float64 (136 B per j point) runs 14 % faster with a
    tile of 128 points (17 KiB) than with 256 (35 KiB), and 28 % slower with 512.
    A function of the plan only, so results stay independent of the sizes."""
    per_point = sum(a.dim for a in plan.args_of("j")) * _SIZE[_CTYPE[plan.dtype]]
    tj = TILE_J
    while tj > TILE_J_MIN and tj * per_point > L1_TILE_BYTES:
        tj //= 2
    return tj


def use_soa(plan: Plan) -> bool:
    dims = [a.dim for a in plan.args_of("j")]
    return all(d <= SOA_MAX_DIM for d in dims) and sum(dims) <= SOA_MAX_TOTAL


PRELUDE = r"""
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <omp.h>
#include <xmmintrin.h>
using std::isnan;
#define __device__
#define __forceinline__ inline __attribute__((always_inline))
"""

HELPERS = r"""
#ifdef KP_HALF_DEFINED
namespace std {
template <> struct numeric_limits<kp_h> {
    static constexpr bool is_specialized = true;
    static kp_h infinity() { return kp_h(numeric_limits<float>::infinity()); }
    static kp_h quiet_NaN() { return kp_h(numeric_limits<float>::quiet_NaN()); }
};
}
#endif
namespace {
template <class X> __forceinline__ X kp_inf() { return X(std::numeric_limits<float>::infinity()); }
template <class X> __forceinline__ bool kp_isnan(X x) { return !(x == x); }
__forceinline__ float kp_exp(float x) { return kp_vexpf(x); }
__forceinline__ double kp_exp(double x) { return kp_vexp(x); }
__forceinline__ float kp_log(float x) { return std::log(x); }
__forceinline__ double kp_log(double x) { return std::log(x); }
__forceinline__ float kp_fabs(float x) { return std::fabs(x); }
__forceinline__ double kp_fabs(double x) { return std::fabs(x); }
__forceinline__ bool kp_isfinite(float x) { return std::isfinite(x); }
__forceinline__ bool kp_isfinite(double x) { return std::isfinite(x); }
// fmin / fmax without a libm call (vectorizable): the other operand when one is NaN
__forceinline__ float kp_fminf(float a, float b) { return (a < b || b != b) ? a : b; }
__forceinline__ float kp_fmaxf(float a, float b) { return (a > b || b != b) ? a : b; }
__forceinline__ double kp_fmin(double a, double b) { return (a < b || b != b) ? a : b; }
__forceinline__ double kp_fmax(double a, double b) { return (a > b || b != b) ? a : b; }
#ifdef KP_HALF_DEFINED
__forceinline__ kp_h kp_fminf(kp_h a, kp_h b) { return (a < b || b != b) ? a : b; }
__forceinline__ kp_h kp_fmaxf(kp_h a, kp_h b) { return (a > b || b != b) ? a : b; }
__forceinline__ kp_h kp_exp(kp_h x) { return kp_h(kp_vexpf((float)x)); }
__forceinline__ kp_h kp_log(kp_h x) { return logf(x); }
__forceinline__ kp_h kp_fabs(kp_h x) { return fabsf(x); }
__forceinline__ bool kp_isfinite(kp_h x) { return std::isfinite((float)x); }
#endif
}  // namespace
"""


# flush_denormals (amendment v0.11): FTZ (bit 15) and DAZ (bit 6) of MXCSR, which
# also governs AVX and AVX-512, set by each thread of a parallel region and
# restored when it leaves it. The OpenMP pool is shared with PyTorch, and the
# calling thread takes part in the region: neither may keep the flags.
FTZ_GUARD = r"""
namespace {
struct KpFtz {
    unsigned csr;
    KpFtz() : csr(_mm_getcsr()) { _mm_setcsr(csr | 0x8040u); }
    ~KpFtz() { _mm_setcsr(csr); }
    KpFtz(const KpFtz&) = delete;
    KpFtz& operator=(const KpFtz&) = delete;
};
}  // namespace
"""


# exp through libmvec: its vector exp sends every lane below about -87.3 (float)
# or -708.4 (double) to a scalar fallback, 5 to 8 times slower when many terms
# underflow (log-sum-exp with a small temperature, narrow Gaussian kernels).
# Below the fast range, exp(x) = exp(x + 40) * exp(-40): the argument stays in
# the fast range, the product is the subnormal result (one extra rounding) and
# `live` zeroes the lanes whose result rounds to 0 anyway (no subnormal assist).
# Written with 0/1 factors and std::min/max only, no ternary: GCC distributes a
# select over the call (exp of the raw argument, or two calls), which brings
# the slow path back. NaN and +-inf propagate as with exp. Without libmvec:
# the plain libm calls.
VEXP = r"""
namespace {
__forceinline__ float kp_vexpf(float x) {
    const float lo = (float)(x < -87.0f);
    const float e = expf(std::max(x + 40.0f * lo, -87.0f));
    const float live = std::min(std::max(x + 105.0f, 0.0f), 1.0f);  // exp(-104) rounds to 0
    return e * (live * ((1.0f - lo) + lo * 0x1.397924p-58f));       // exp(-40)
}
__forceinline__ double kp_vexp(double x) {
    const double lo = (double)(x < -708.0);
    const double e = exp(std::max(x + 40.0 * lo, -708.0));
    const double live = std::min(std::max(x + 747.0, 0.0), 1.0);  // exp(-746) rounds to 0
    return e * (live * ((1.0 - lo) + lo * 0x1.39792499b1a24p-58));
}
// Terms of a log-sum-exp sum (x - max <= 0): the sum holds a term equal to 1,
// so a term below the fast range only needs to stay below half an ulp of it;
// exp(-87) (float) or exp(-708) (double) does, at the cost of one max.
__forceinline__ float kp_exp_sum(float x) { return expf(std::max(x, -87.0f)); }
__forceinline__ double kp_exp_sum(double x) { return exp(std::max(x, -708.0)); }
}  // namespace
"""

NO_VEXP = r"""
namespace {
__forceinline__ float kp_vexpf(float x) { return expf(x); }
__forceinline__ double kp_vexp(double x) { return exp(x); }
__forceinline__ float kp_exp_sum(float x) { return expf(x); }
__forceinline__ double kp_exp_sum(double x) { return exp(x); }
}  // namespace
"""

# exp / expf calls of the pair function (not kp_exp, std::exp, x.exp)
_EXP_CALL = re.compile(r"(?<![\w:.])exp(f?)\(")


def _soft_groups(plan: Plan):
    """sycl groups with the score / weight component indices parsed."""
    out = []
    for m, vexpr, sums in _sycl._soft_groups(plan):
        sc = int(re.fullmatch(r"o\.o(\d+)", vexpr).group(1))
        s2 = [(si, ci, None if w is None else int(re.fullmatch(r"o\.o(\d+)", w).group(1))) for si, ci, w in sums]
        out.append((m, sc, s2))
    return out


def soa_pair(pair_source: str) -> str | None:
    """``kp_pair_soa``: copy of ``kp_pair`` whose j operands are read with stride
    ``TJ`` (``yj_n[c]`` -> ``yj_n[c * TJ]``), called with ``yb + off * TJ + t``.

    Every load of the t loop is then affine in t (base + t), so the loop
    vectorizes; with a local ``T y[d]`` array, GCC turns the array into an
    "omp simd array" whose addressing it cannot analyze. None if the source does
    not have the expected form (the backend then uses the generic path)."""
    start = pair_source.find("kp_out kp_pair(")
    if start < 0:
        return None
    line0 = pair_source.rfind("\n", 0, start) + 1
    depth, k, opened = 0, pair_source.index("{", start), False
    for k in range(pair_source.index("{", start), len(pair_source)):
        ch = pair_source[k]
        if ch == "{":
            depth += 1
            opened = True
        elif ch == "}":
            depth -= 1
            if opened and depth == 0:
                break
    else:
        return None
    fn = pair_source[line0:k + 1].replace("kp_out kp_pair(", "kp_out kp_pair_soa(", 1)
    head, body = fn.split("{", 1)
    body2 = re.sub(r"\b(yj_\d+)\[(\d+)\]", r"\1[\2 * TJ]", body)
    # same overload sets as fminf/fmin (float, kp_h / double), without libm call
    body2 = re.sub(r"(?<![\w.:])(fmin|fmax)(f?)\(", r"kp_\1\2(", body2)
    if re.search(r"\byj_\d+\b(?!\[\d+ \* TJ\])", body2):
        return None  # j pointer used other than through a literal index
    return head + "{" + body2


def _fold(n, body):
    """Fixed pairwise tree over LN lanes: ``body(dst, src)`` lines."""
    return [f"for (int w_ = LN / 2; w_ >= 1; w_ /= 2) for (int l = 0; l < w_; ++l) {{ {body('l', 'l + w_')} }}"]


def generate(plan: Plan, pair_source: str, *, tj: int = TILE_J, bi: int = ROWS,
             variant_note: str = "", simd_math: str = "") -> tuple[str, KernelLayout]:
    """``simd_math``: declarations of the vector variants of libm functions
    (``toolchain.libmvec_declarations``), inserted before the pair function."""
    vexp = re.search(r"\bfloat expf\(float\)", simd_math) is not None
    if vexp:  # before any use of pair_source (SoA copy included)
        pair_source = _EXP_CALL.sub(lambda m: "kp_vexpf(" if m.group(1) else "kp_vexp(", pair_source)
    kind = plan.reduction.kind
    T = _CTYPE[plan.dtype]
    ACC = _CTYPE[plan.acc_dtype]
    D = plan.out_dim
    W, VT, has_idx = state_spec(plan)
    is_k = kind in K_KINDS
    neg = kind in (ReductionKind.MAX, ReductionKind.ARGMAX, ReductionKind.MAX_ARGMAX)
    soft = kind in SOFT_KINDS
    kahan = _sycl._kahan(plan)
    rng = bool(plan.ranges)
    if rng and plan.batch_rank:
        raise ValueError("ranges combined with batch: rejected (amendment v0.6)")
    if tj % LN or tj <= 0:
        raise ValueError(f"tile j must be a positive multiple of {LN}")
    bi = rows_for(plan, bi)
    outs = out_names(plan)
    soa_fn = soa_pair(pair_source) if use_soa(plan) else None
    soa = soa_fn is not None
    count = kind is ReductionKind.COUNT
    nvb = 1 if is_k else D  # components stored in the tile buffer
    VB = "int32_t" if count else "T"
    OUTV = "int64_t" if count else ("OUT" if kind in FLOAT_ACC_KINDS else "T")
    op = ">" if neg else "<"
    jargs = [(n, a) for n, a in enumerate(plan.args) if a.role == "j"]
    joff, off = {}, 0
    for n, a in jargs:
        joff[n] = off
        off += a.dim
    DJ = max(off, 1)

    L: list[str] = []
    a = L.append
    a("// Generated by kernox.backends.cpu.codegen, do not edit.")
    a(f"// variant: {variant_note}")
    a(PRELUDE)
    if simd_math:
        a(simd_math)
    a(VEXP if vexp else NO_VEXP)
    if plan.flush_denormals:
        a(FTZ_GUARD)
    a(pair_source)
    a(HELPERS)
    a("namespace {")
    a(f"using T = {T};")
    a(f"using ACC = {ACC};")
    a(f"using OUT = {_CTYPE[accumulated_output_dtype(plan)]};")
    a(f"using VT = {VT};")
    a(f"using VB = {VB};")
    a(f"using OUTV = {OUTV};")
    a(f"constexpr int TJ = {tj};")
    a(f"constexpr int BI = {bi};")
    a(f"constexpr int D = {D};")
    a(f"constexpr int W = {W};")
    a(f"constexpr int LN = {LN};")
    a(f"constexpr int DJ = {DJ};")
    a(f"constexpr int NVB = {nvb};")
    if soa:
        a(soa_fn)
    if kind in FLOAT_ACC_KINDS:
        a("\n".join(_port([_sycl.KAHAN_HELPERS])).replace("inline", "__forceinline__"))
    if is_k:
        a("template <class IT>")
        a("__forceinline__ bool kp_less(T x, IT ix, T y, IT iy) {")
        a("    // lexicographic order (value, j); an empty slot (iy < 0) is the largest")
        a("    return iy < 0 || x < y || (x == y && ix < iy);")
        a("}")
        a("template <class IT>")
        a("inline void kp_insert(VT* v, IT* ix, T cv, IT ci) {")
        a("    if (!kp_less(cv, ci, v[W - 1], ix[W - 1])) return;")
        a("    int k = W - 1;")
        a("    while (k > 0 && kp_less(cv, ci, v[k - 1], ix[k - 1])) { v[k] = v[k - 1]; ix[k] = ix[k - 1]; --k; }")
        a("    v[k] = cv; ix[k] = ci;")
        a("}")

    # ---------------------------------------------------------------- init
    a("template <class IT>")
    a("__forceinline__ void kp_init(VT* v, IT* ix) {")
    a("    (void)ix;")
    a("    for (int c = 0; c < W; ++c) {")
    if kind in (ReductionKind.SUM, ReductionKind.COUNT) or soft:
        a("        v[c] = VT(0);")
    else:
        a(f"        v[c] = {'-' if neg else ''}kp_inf<T>();")
    if has_idx:
        a("        ix[c] = -1;")
    a("    }")
    if soft:
        for m, _sc, _s in _soft_groups(plan):
            a(f"    v[{m}] = -kp_inf<VT>();  // m: empty state")
    a("}")

    # ---------------------------------------------------------------- merge (partial state, int64 indices)
    a("template <class IT>")
    a("inline void kp_merge(VT* v, IT* ix, const VT* pv, const int64_t* pi) {")
    a("    (void)ix; (void)pi;")
    if is_k:
        a("    for (int k = 0; k < W; ++k) { if (pi[k] < 0) break; kp_insert<IT>(v, ix, pv[k], static_cast<IT>(pi[k])); }")
    elif soft:
        for m, sc, sums in _soft_groups(plan):
            grp = (m, None, [(si, ci, None) for si, ci, _w in sums])
            for line in _port(_sycl._soft_merge_lines("v", "pv", grp)):
                a("    " + line)
    elif kind is ReductionKind.SUM and kahan:
        a("    for (int c = 0; c < D; ++c) kp_kmerge(v[c], v[D + c], pv[c], pv[D + c]);")
    elif kind in (ReductionKind.SUM, ReductionKind.COUNT):
        a("    for (int c = 0; c < W; ++c) v[c] += pv[c];")
    elif kind in (ReductionKind.MIN, ReductionKind.MAX):
        a(f"    for (int c = 0; c < W; ++c) if (pv[c] {op} v[c]) v[c] = pv[c];")
    else:
        a(f"    for (int c = 0; c < W; ++c) if (pi[c] >= 0 && (ix[c] < 0 || pv[c] {op} v[c] || (pv[c] == v[c] && pi[c] < ix[c]))) "
          "{ v[c] = pv[c]; ix[c] = static_cast<IT>(pi[c]); }")
    a("}")

    # ---------------------------------------------------------------- final write
    a("template <class IT>")
    a("inline void kp_write(const VT* v, const IT* ix, int64_t jbeg, OUTV* out_value, int64_t* out_index, int64_t i) {")
    a("    (void)ix; (void)jbeg; (void)out_value; (void)out_index;")

    def gidx(c):
        return f"(ix[{c}] < 0 ? int64_t(-1) : jbeg + int64_t(ix[{c}]))"

    if soft:
        for line in _port(_sycl._soft_finish_lines(plan, "v")):
            a("    " + line)
    elif kind is ReductionKind.SUM and kahan:
        a("    for (int c = 0; c < D; ++c) out_value[i * D + c] = static_cast<OUTV>(kp_kfin(v[c], v[D + c]));")
    elif kind in (ReductionKind.SUM, ReductionKind.COUNT, ReductionKind.MIN, ReductionKind.MAX):
        a("    for (int c = 0; c < D; ++c) out_value[i * D + c] = static_cast<OUTV>(v[c]);")
    elif kind in ARG_KINDS:
        a("    for (int c = 0; c < D; ++c) {")
        if "value" in outs:
            a("        out_value[i * D + c] = v[c];")
        a(f"        out_index[i * D + c] = {gidx('c')};")
        a("    }")
    else:
        a("    for (int k = 0; k < W; ++k) {")
        if "value" in outs:
            a("        out_value[i * W + k] = v[k];")
        if "index" in outs:
            a(f"        out_index[i * W + k] = {gidx('k')};")
        a("    }")
    a("}")

    # ---------------------------------------------------------------- task
    IT = "int32_t"
    a("// Rows [i0, i1) against the j intervals iv[2k], iv[2k+1] (k < niv); indices")
    a("// in the state are int32 relative to jbeg.")
    a("inline void kp_task(const T* const* in, int64_t i0, int64_t i1, const int64_t* iv, int64_t niv,")
    a("                    int64_t jbeg, VT* sv, int32_t* sx, T* yb, VB* vb) {")
    a("    (void)yb; (void)sx;")
    a("    const int nr = static_cast<int>(i1 - i0);")
    a(f"    for (int r = 0; r < nr; ++r) kp_init<{IT}>(sv + r * W, sx + r * W);")
    a("    for (int64_t kr = 0; kr < niv; ++kr) {")
    a("    const int64_t jend = iv[2 * kr + 1];")
    a("    for (int64_t jb = iv[2 * kr]; jb < jend; jb += TJ) {")
    a("        const int nt = (jend - jb < TJ) ? static_cast<int>(jend - jb) : TJ;")
    a("        const int ntp = (nt + LN - 1) / LN * LN;")
    a("        const int32_t jl0 = static_cast<int32_t>(jb - jbeg);")
    a("        (void)jl0;")
    if soa:
        for n, arg in jargs:
            a("        {")
            a(f"            const T* __restrict__ src = in[{n}] + jb * {arg.dim};")
            a(f"            T* __restrict__ dst = yb + {joff[n]} * TJ;")
            a("            for (int t = 0; t < ntp; ++t) {")
            a("                const int tt = t < nt ? t : nt - 1;  // padding: valid data")
            a(f"                for (int c = 0; c < {arg.dim}; ++c) dst[c * TJ + t] = src[tt * {arg.dim} + c];")
            a("            }")
            a("        }")
    a("        for (int r = 0; r < nr; ++r) {")
    a("            const int64_t i = i0 + r;")
    a("            (void)i;")
    a("            VT* __restrict__ v = sv + r * W;")
    a(f"            {IT}* __restrict__ ix = sx + r * W;")
    a("            (void)ix;")
    # row / parameter pointers
    call = []
    for n, arg in enumerate(plan.args):
        if arg.role == "i":
            a(f"            const T* __restrict__ x{n} = in[{n}] + i * {arg.dim};")
            call.append(f"x{n}")
        elif arg.role == "param":
            a(f"            const T* __restrict__ x{n} = in[{n}];")
            call.append(f"x{n}")
        else:
            call.append(f"y{n}")
    if is_k:
        stores = ["vb[t] = o.o0;"]
    elif count:
        stores = [f"vb[{c} * TJ + t] = o.o{c} ? 1 : 0;" for c in range(D)]
    else:
        stores = [f"vb[{c} * TJ + t] = o.o{c};" for c in range(D)]
    if soa:
        for n, arg in jargs:
            a(f"            const T* __restrict__ yb{n} = yb + {joff[n]} * TJ;")
        call = [f"yb{c[1:]} + t" if c.startswith("y") else c for c in call]
        a("            #pragma omp simd")
        a("            for (int t = 0; t < ntp; ++t) {")
    else:
        a("            for (int t = 0; t < ntp; ++t) {")
        a("                const int64_t jj = jb + (t < nt ? t : nt - 1);")
        for n, arg in jargs:
            a(f"                const T* __restrict__ y{n} = in[{n}] + jj * {arg.dim};")
    a(f"                const kp_out o = {'kp_pair_soa' if soa else 'kp_pair'}({', '.join(call)});")
    for s in stores:
        a("                " + s)
    a("            }")
    # padding: identity
    if is_k or kind in ARG_KINDS:
        pad = "std::numeric_limits<T>::quiet_NaN()"
    elif count or kind is ReductionKind.SUM:
        pad = "VB(0)"
    elif kind is ReductionKind.MIN:
        pad = "kp_inf<T>()"
    elif kind is ReductionKind.MAX:
        pad = "-kp_inf<T>()"
    else:
        pad = None
    if soft:
        groups = _soft_groups(plan)
        a("            for (int t = nt; t < ntp; ++t) {")
        for _m, sc, _s in groups:
            a(f"                vb[{sc} * TJ + t] = -kp_inf<T>();")
        a("            }")
    else:
        a(f"            for (int t = nt; t < ntp; ++t) for (int c = 0; c < NVB; ++c) vb[c * TJ + t] = {pad};")

    P = "            "
    # --- reductions of the tile buffer
    if kind is ReductionKind.SUM and not kahan:
        a(P + "for (int c = 0; c < D; ++c) {")
        a(P + "    ACC acc[LN];")
        a(P + "    for (int l = 0; l < LN; ++l) acc[l] = ACC(0);")
        a(P + "    const VB* __restrict__ b = vb + c * TJ;")
        a(P + "    for (int t0 = 0; t0 < ntp; t0 += LN) {")
        a(P + "        #pragma omp simd")
        a(P + "        for (int l = 0; l < LN; ++l) acc[l] += static_cast<ACC>(b[t0 + l]);")
        a(P + "    }")
        for line in _fold(LN, lambda d, s: f"acc[{d}] += acc[{s}];"):
            a(P + "    " + line)
        a(P + "    v[c] += acc[0];  // block summation: tile sum added to the total")
        a(P + "}")
    elif kind is ReductionKind.SUM:
        a(P + "for (int c = 0; c < D; ++c) {")
        a(P + "    ACC acc[LN], cmp[LN];")
        a(P + "    for (int l = 0; l < LN; ++l) { acc[l] = ACC(0); cmp[l] = ACC(0); }")
        a(P + "    const VB* __restrict__ b = vb + c * TJ;")
        a(P + "    for (int t0 = 0; t0 < ntp; t0 += LN) {")
        a(P + "        #pragma omp simd")
        a(P + "        for (int l = 0; l < LN; ++l) kp_add(acc[l], cmp[l], static_cast<ACC>(b[t0 + l]));")
        a(P + "    }")
        for line in _fold(LN, lambda d, s: f"kp_kmerge(acc[{d}], cmp[{d}], acc[{s}], cmp[{s}]);"):
            a(P + "    " + line)
        a(P + "    kp_kmerge(v[c], v[D + c], acc[0], cmp[0]);")
        a(P + "}")
    elif count:
        a(P + "for (int c = 0; c < D; ++c) {")
        a(P + "    int32_t acc[LN];")
        a(P + "    for (int l = 0; l < LN; ++l) acc[l] = 0;")
        a(P + "    const VB* __restrict__ b = vb + c * TJ;")
        a(P + "    for (int t0 = 0; t0 < ntp; t0 += LN) {")
        a(P + "        #pragma omp simd")
        a(P + "        for (int l = 0; l < LN; ++l) acc[l] += b[t0 + l];")
        a(P + "    }")
        a(P + "    int64_t tot = 0;")
        a(P + "    for (int l = 0; l < LN; ++l) tot += acc[l];")
        a(P + "    v[c] += tot;")
        a(P + "}")
    elif kind in (ReductionKind.MIN, ReductionKind.MAX):
        a(P + "for (int c = 0; c < D; ++c) {")
        a(P + "    T acc[LN];")
        a(P + f"    for (int l = 0; l < LN; ++l) acc[l] = {'-' if neg else ''}kp_inf<T>();")
        a(P + "    const VB* __restrict__ b = vb + c * TJ;")
        a(P + "    for (int t0 = 0; t0 < ntp; t0 += LN) {")
        a(P + "        #pragma omp simd")
        a(P + "        for (int l = 0; l < LN; ++l) { const T x = b[t0 + l]; acc[l] = (x " + op + " acc[l]) ? x : acc[l]; }  // NaN ignored")
        a(P + "    }")
        for line in _fold(LN, lambda d, s: f"acc[{d}] = (acc[{s}] {op} acc[{d}]) ? acc[{s}] : acc[{d}];"):
            a(P + "    " + line)
        a(P + f"    if (acc[0] {op} v[c]) v[c] = acc[0];")
        a(P + "}")
    elif kind in ARG_KINDS:
        a(P + "for (int c = 0; c < D; ++c) {")
        a(P + "    T acc[LN]; int32_t aix[LN];")
        a(P + f"    for (int l = 0; l < LN; ++l) {{ acc[l] = {'-' if neg else ''}kp_inf<T>(); aix[l] = -1; }}")
        a(P + "    const VB* __restrict__ b = vb + c * TJ;")
        a(P + "    for (int t0 = 0; t0 < ntp; t0 += LN) {")
        a(P + "        #pragma omp simd")
        a(P + "        for (int l = 0; l < LN; ++l) {")
        a(P + "            const T x = b[t0 + l];")
        a(P + f"            const bool tk = !kp_isnan(x) && (aix[l] < 0 || x {op} acc[l]);  // j increasing in a lane")
        a(P + "            acc[l] = tk ? x : acc[l];")
        a(P + "            aix[l] = tk ? jl0 + t0 + l : aix[l];")
        a(P + "        }")
        a(P + "    }")
        take = (lambda d, s: f"aix[{s}] >= 0 && (aix[{d}] < 0 || acc[{s}] {op} acc[{d}] || "
                f"(acc[{s}] == acc[{d}] && aix[{s}] < aix[{d}]))")
        for line in _fold(LN, lambda d, s: f"if ({take(d, s)}) {{ acc[{d}] = acc[{s}]; aix[{d}] = aix[{s}]; }}"):
            a(P + "    " + line)
        a(P + f"    if (aix[0] >= 0 && (ix[c] < 0 || acc[0] {op} v[c] || (acc[0] == v[c] && aix[0] < ix[c]))) "
          "{ v[c] = acc[0]; ix[c] = aix[0]; }")
        a(P + "}")
    elif is_k:
        a(P + "for (int t0 = 0; t0 < ntp; t0 += LN) {")
        a(P + "    const T thr = v[W - 1];")
        a(P + "    int any = ix[W - 1] < 0;")
        a(P + "    if (!any) {")
        a(P + "        #pragma omp simd reduction(|:any)")
        a(P + "        for (int l = 0; l < LN; ++l) any |= (vb[t0 + l] <= thr) ? 1 : 0;  // NaN: false")
        a(P + "    }")
        a(P + "    if (!any) continue;")
        a(P + "    for (int l = 0; l < LN; ++l) {")
        a(P + "        const T x = vb[t0 + l];")
        a(P + f"        if (!kp_isnan(x)) kp_insert<{IT}>(v, ix, x, jl0 + t0 + l);")
        a(P + "    }")
        a(P + "}")
    else:  # soft
        ninf = "-kp_inf<ACC>()"
        for m, sc, sums in groups:
            a(P + "{")
            a(P + f"    const VB* __restrict__ bs = vb + {sc} * TJ;")
            a(P + "    ACC mx[LN];")
            a(P + f"    for (int l = 0; l < LN; ++l) mx[l] = {ninf};")
            a(P + "    for (int t0 = 0; t0 < ntp; t0 += LN) {")
            a(P + "        #pragma omp simd")
            a(P + "        for (int l = 0; l < LN; ++l) { const ACC x = static_cast<ACC>(bs[t0 + l]); mx[l] = (x > mx[l]) ? x : mx[l]; }")
            a(P + "    }")
            for line in _fold(LN, lambda d, s: f"mx[{d}] = (mx[{s}] > mx[{d}]) ? mx[{s}] : mx[{d}];"):
                a(P + "    " + line)
            a(P + "    const ACC mt = mx[0];")
            for k, (_si, _ci, w) in enumerate(sums):
                a(P + f"    ACC s{k}[LN];" + (f" ACC c{k}[LN];" if kahan else ""))
                a(P + f"    for (int l = 0; l < LN; ++l) {{ s{k}[l] = ACC(0);" + (f" c{k}[l] = ACC(0);" if kahan else "") + " }")
            a(P + "    for (int t0 = 0; t0 < ntp; t0 += LN) {")
            a(P + "        #pragma omp simd")
            a(P + "        for (int l = 0; l < LN; ++l) {")
            a(P + "            const int t = t0 + l;")
            a(P + "            const ACC x = static_cast<ACC>(bs[t]);")
            a(P + f"            const bool skip = (x == {ninf});  // v = -inf: ignored, even if g is NaN")
            # weighted sums (SSMW) keep the exact exp: a weight can lift a tiny term
            ex = "kp_exp_sum" if all(w is None for _s, _c, w in sums) else "kp_exp"
            a(P + f"            const ACC q = skip ? ACC(0) : ((x == mt) ? ACC(1) : {ex}(x - mt));")
            for k, (_si, _ci, w) in enumerate(sums):
                term = "q" if w is None else f"(skip ? ACC(0) : q * static_cast<ACC>(vb[{w} * TJ + t]))"
                if kahan:
                    a(P + f"            kp_add(s{k}[l], c{k}[l], {term});")
                else:
                    a(P + f"            s{k}[l] += {term};")
            a(P + "        }")
            a(P + "    }")
            for k, _ in enumerate(sums):
                if kahan:
                    for line in _fold(LN, lambda d, s, k=k: f"kp_kmerge(s{k}[{d}], c{k}[{d}], s{k}[{s}], c{k}[{s}]);"):
                        a(P + "    " + line)
                else:
                    for line in _fold(LN, lambda d, s, k=k: f"s{k}[{d}] += s{k}[{s}];"):
                        a(P + "    " + line)
            a(P + "    VT tp[W];")
            a(P + f"    tp[{m}] = mt;")
            for k, (si, ci, _w) in enumerate(sums):
                a(P + f"    tp[{si}] = s{k}[0];" + (f" tp[{ci}] = c{k}[0];" if ci is not None else ""))
            grp = (m, None, [(si, ci, None) for si, ci, _w in sums])
            for line in _port(_sycl._soft_merge_lines("v", "tp", grp)):
                a(P + "    " + line)
            a(P + "}")
    a("        }  // rows")
    a("    }  // tiles")
    a("    }  // intervals")
    a("}")
    a("}  // namespace")
    a("")

    # ---------------------------------------------------------------- entry point
    nin = len(plan.args)
    a('extern "C" int kp_launch(int64_t M, int64_t N, void* const* kp_in, void* const* kp_outp,')
    a("                         int64_t S, int64_t chunk, void* kp_temp, int64_t NB, const int64_t* const* kp_boff,")
    a("                         int64_t NW, const int64_t* kp_work, const int64_t* kp_red, int nthreads,")
    a("                         char* kp_err, size_t kp_errlen)")
    a("{")
    a("    (void)N;")
    if plan.batch_rank:
        a("    if (NB < 1 || kp_boff == nullptr) {")
        a('        std::snprintf(kp_err, kp_errlen, "invalid batch (NB, offset tables)");')
        a("        return 5;")
        a("    }")
    else:
        a("    if (NB != 1) {")
        a('        std::snprintf(kp_err, kp_errlen, "plan without batch: expected NB = 1");')
        a("        return 5;")
        a("    }")
    a("    if (S < 1 || (S > 1 && kp_temp == nullptr) || chunk < 1) {")
    a('        std::snprintf(kp_err, kp_errlen, "invalid j split");')
    a("        return 4;")
    a("    }")
    if rng:
        a("    if (S != 1 || NW < 0 || (NW > 0 && (kp_work == nullptr || kp_red == nullptr))) {")
        a('        std::snprintf(kp_err, kp_errlen, "ranges: invalid work list");')
        a("        return 6;")
        a("    }")
    else:
        a("    (void)NW; (void)kp_work; (void)kp_red;")
    a("    if (M <= 0) return 0;")
    a(f"    OUTV* out_value_g = {'static_cast<OUTV*>(kp_outp[%d])' % outs.index('value') if 'value' in outs else 'nullptr'};")
    a(f"    int64_t* out_index_g = {'static_cast<int64_t*>(kp_outp[%d])' % outs.index('index') if 'index' in outs else 'nullptr'};")
    a("    const size_t nstate = static_cast<size_t>(M) * static_cast<size_t>(S) * W;  // per batch element")
    if has_idx:
        a("    int64_t* tix_g = static_cast<int64_t*>(kp_temp);")
        a("    VT* tv_g = reinterpret_cast<VT*>(tix_g + (kp_temp ? NB * nstate : 0));")
    else:
        a("    int64_t* tix_g = nullptr;")
        a("    VT* tv_g = static_cast<VT*>(kp_temp);")
    a("    const int64_t ngi = (M + BI - 1) / BI;")
    a(f"    const int64_t ntask = {'NW' if rng else 'NB * S * ngi'};")
    a("    if (ntask <= 0) return 0;")
    a("    int nth = nthreads < 1 ? 1 : nthreads;")
    a("    if (nth > ntask) nth = static_cast<int>(ntask);")
    a("    std::atomic<int64_t> next{0};")
    a("    std::atomic<int> failed{0};")
    a("    const size_t ybytes = (static_cast<size_t>(DJ) * TJ * sizeof(T) + 63) / 64 * 64;")
    a("    const size_t vbytes = (static_cast<size_t>(NVB) * TJ * sizeof(VB) + 63) / 64 * 64;")
    a("    const size_t sbytes = (static_cast<size_t>(BI) * W * sizeof(VT) + 63) / 64 * 64;")
    a("    const size_t xbytes = (static_cast<size_t>(BI) * W * sizeof(int32_t) + 63) / 64 * 64;")
    a("    #pragma omp parallel num_threads(nth) if (nth > 1)")
    a("    {")
    if plan.flush_denormals:
        a("        const KpFtz kp_ftz;")
    a("        unsigned char* buf = static_cast<unsigned char*>(std::aligned_alloc(64, ybytes + vbytes + sbytes + xbytes));")
    a("        if (buf == nullptr) failed.store(1);")
    a("        else {")
    a("        T* yb = reinterpret_cast<T*>(buf);")
    a("        VB* vb = reinterpret_cast<VB*>(buf + ybytes);")
    a("        VT* sv = reinterpret_cast<VT*>(buf + ybytes + vbytes);")
    a("        int32_t* sx = reinterpret_cast<int32_t*>(buf + ybytes + vbytes + sbytes);")
    a(f"        const T* in[{max(nin, 1)}];")
    a("        for (int64_t task = next.fetch_add(1, std::memory_order_relaxed); task < ntask;")
    a("             task = next.fetch_add(1, std::memory_order_relaxed)) {")
    if rng:
        a("            const int64_t* wk = kp_work + 4 * task;  // (i0, i1, k0, k1)")
        a("            const int64_t i0 = wk[0], i1 = wk[1];")
        for n in range(nin):
            a(f"            in[{n}] = static_cast<const T*>(kp_in[{n}]);")
        a("            OUTV* out_value = out_value_g; int64_t* out_index = out_index_g;")
        a("            (void)out_value; (void)out_index;")
        a("            const int64_t jbeg = 0;  // ranges: global j indices (N < 2^31)")
        a("            kp_task(in, i0, i1, kp_red + 2 * wk[2], wk[3] - wk[2], jbeg, sv, sx, yb, vb);")
        a(f"            for (int64_t i = i0; i < i1; ++i) kp_write<int32_t>(sv + (i - i0) * W, sx + (i - i0) * W, jbeg, out_value, out_index, i);")
    else:
        a("            const int64_t kb = task / (S * ngi);")
        a("            const int64_t s = (task % (S * ngi)) / ngi;")
        a("            const int64_t gi = task % ngi;")
        a("            const int64_t i0 = gi * BI, i1 = (M - i0 < BI) ? M : i0 + BI;")
        for n in range(nin):
            shift = f" + kp_boff[{n}][kb]" if plan.batch_rank else ""
            a(f"            in[{n}] = static_cast<const T*>(kp_in[{n}]){shift};")
        a(f"            OUTV* out_value = out_value_g ? out_value_g + kb * M * {_sycl._out_width(plan, 'value') if 'value' in outs else 0} : nullptr;")
        a(f"            int64_t* out_index = out_index_g ? out_index_g + kb * M * {_sycl._out_width(plan, 'index') if 'index' in outs else 0} : nullptr;")
        a("            const int64_t jbeg = s * chunk;")
        a("            const int64_t iv[2] = {jbeg, (N - jbeg < chunk) ? N : jbeg + chunk};")
        a("            kp_task(in, i0, i1, iv, iv[0] < iv[1] ? 1 : 0, jbeg, sv, sx, yb, vb);")
        a("            if (S == 1) {")
        a("                for (int64_t i = i0; i < i1; ++i) kp_write<int32_t>(sv + (i - i0) * W, sx + (i - i0) * W, jbeg, out_value, out_index, i);")
        a("            } else {")
        a("                for (int64_t i = i0; i < i1; ++i) {")
        a("                    const size_t base = kb * nstate + (static_cast<size_t>(i) * S + s) * W;")
        a("                    const VT* sr = sv + (i - i0) * W;")
        a("                    for (int c = 0; c < W; ++c) tv_g[base + c] = sr[c];")
        if has_idx:
            a("                    const int32_t* xr = sx + (i - i0) * W;")
            a("                    for (int c = 0; c < W; ++c) tix_g[base + c] = xr[c] < 0 ? int64_t(-1) : jbeg + int64_t(xr[c]);")
        a("                }")
        a("            }")
    a("        }")
    a("        std::free(buf);")
    a("        }")
    a("    }")
    a("    if (failed.load()) {")
    a('        std::snprintf(kp_err, kp_errlen, "out of memory (thread buffers)");')
    a("        return 7;")
    a("    }")
    if not rng:
        a("    if (S > 1) {")
        a("        const int64_t nm = NB * ngi;")
        a("        int nth2 = nthreads < 1 ? 1 : nthreads;")
        a("        if (nth2 > nm) nth2 = static_cast<int>(nm);")
        a("        std::atomic<int64_t> next2{0};")
        a("        #pragma omp parallel num_threads(nth2) if (nth2 > 1)")
        a("        {")
        if plan.flush_denormals:
            a("            const KpFtz kp_ftz;")
        a("            unsigned char* buf = static_cast<unsigned char*>(std::aligned_alloc(64, (static_cast<size_t>(W) * (sizeof(VT) + 8) + 63) / 64 * 64));")
        a("            if (buf == nullptr) failed.store(1);")
        a("            else {")
        a("            VT* st = reinterpret_cast<VT*>(buf);")
        a("            int64_t* sx2 = reinterpret_cast<int64_t*>(buf + (static_cast<size_t>(W) * sizeof(VT) + 7) / 8 * 8);")
        a("            for (int64_t task = next2.fetch_add(1, std::memory_order_relaxed); task < nm;")
        a("                 task = next2.fetch_add(1, std::memory_order_relaxed)) {")
        a("                const int64_t kb = task / ngi, gi = task % ngi;")
        a("                const int64_t i0 = gi * BI, i1 = (M - i0 < BI) ? M : i0 + BI;")
        a(f"                OUTV* out_value = out_value_g ? out_value_g + kb * M * {_sycl._out_width(plan, 'value') if 'value' in outs else 0} : nullptr;")
        a(f"                int64_t* out_index = out_index_g ? out_index_g + kb * M * {_sycl._out_width(plan, 'index') if 'index' in outs else 0} : nullptr;")
        a("                for (int64_t i = i0; i < i1; ++i) {")
        a("                    kp_init<int64_t>(st, sx2);")
        a("                    for (int64_t s = 0; s < S; ++s) {")
        a("                        const size_t base = kb * nstate + (static_cast<size_t>(i) * S + s) * W;")
        a(f"                        kp_merge<int64_t>(st, sx2, tv_g + base, {'tix_g + base' if has_idx else 'nullptr'});")
        a("                    }")
        a("                    kp_write<int64_t>(st, sx2, 0, out_value, out_index, i);")
        a("                }")
        a("            }")
        a("            std::free(buf);")
        a("            }")
        a("        }")
        a("        if (failed.load()) {")
        a('            std::snprintf(kp_err, kp_errlen, "out of memory (merge buffers)");')
        a("            return 7;")
        a("        }")
        a("    }")
    a("    return 0;")
    a("}")
    layout = KernelLayout(tj, bi, tuple(x.name for x in plan.args), outs, partial_bytes_per_row(plan), soa)
    return "\n".join(L) + "\n", layout
