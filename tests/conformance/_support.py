"""Shared tools for the conformance suite (all GPU backends).

Reference: the formula is evaluated term by term in FP64 by
``kernox.oracle.evaluate_numpy`` (symox NumPy lambdify, a path independent
of the backends and of the Torch oracle) on inputs **already rounded to the
plan dtype**, then reduced according to the v0 contract (NaNs ignored, ties
go to the smallest j, padding with (+inf, -1)).

Tolerances (see ``docs/conformance.md``): each term gets an evaluation error
bound ``T_ij``; it is zero when the FP32 evaluation of the formula reproduces
the FP64 one exactly (integer data: exact comparison, ties included).
Otherwise ``T_ij = 4 (E_ij + C_ij) + 4 u |F_ij| + floor``, where ``E_ij`` is
the measured gap between an independent FP32 evaluation (Torch CPU, oracle)
and the FP64 reference, scaled by ``u64/u32`` for an FP64 plan, and ``C_ij``
is the FP64 sensitivity of the term to a relative perturbation ``u`` of the
inputs (conditioning: cancellations, functions near a zero).
A term whose finiteness differs between the two evaluations (FP32 overflow,
NaN on one side only) is "undetermined" and its row is not judged.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pytest

torch = pytest.importorskip("torch")
symox = pytest.importorskip("symox")

import kernox as kp  # noqa: E402
from kernox.errors import BackendUnavailableError  # noqa: E402
from kernox.oracle import OracleReduction, evaluate_numpy  # noqa: E402
from kernox.plan import ReductionKind as R, output_shapes  # noqa: E402

U = {"float32": 2.0**-24, "float64": 2.0**-53}
TINY = {"float32": float(np.finfo(np.float32).tiny), "float64": float(np.finfo(np.float64).tiny)}
NP = {"float32": np.float32, "float64": np.float64}
TORCH = {"float32": torch.float32, "float64": torch.float64}
K_KINDS = (R.KMIN, R.ARGKMIN, R.KMIN_ARGKMIN)
SOFT_KINDS = (R.LOGSUMEXP, R.SUMSOFTMAXWEIGHT)
BACKENDS = ("cuda", "hip", "sycl")


# --------------------------------------------------------------- backends
def device_for(name: str):
    """Torch device of backend ``name``, or None (never simulated)."""
    if name == "cuda":
        ok = torch.cuda.is_available() and torch.version.hip is None
        return "cuda:0" if ok else None
    if name == "hip":
        ok = torch.cuda.is_available() and torch.version.hip is not None
        return "cuda:0" if ok else None
    if name == "sycl":
        ok = hasattr(torch, "xpu") and torch.xpu.is_available()
        return "xpu:0" if ok else None
    raise ValueError(name)


def backend_status(name: str):
    """``(device, None)`` if the backend is usable, ``(None, reason)`` otherwise."""
    wanted = os.environ.get("KP_CONFORMANCE_BACKENDS")
    if wanted and name not in wanted.split(","):
        return None, f"excluded by KP_CONFORMANCE_BACKENDS={wanted}"
    dev = device_for(name)
    if dev is None:
        return None, f"no Torch device for backend {name}"
    try:
        kp.get_backend(name)
    except BackendUnavailableError as exc:
        return None, str(exc)
    return dev, None


def synchronize(device):
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "xpu":
        torch.xpu.synchronize(dev)


# ------------------------------------------------------------------ inputs
def to_inputs(plan, arrays: dict, device):
    """Rounds the FP64 arrays to the plan dtype. Returns (device tensors, rounded FP64)."""
    dt = plan.dtype
    dev_t, ref = {}, {}
    for a in plan.args:
        v = np.asarray(arrays[a.name], dtype=np.float64)
        with np.errstate(over="ignore"):
            r = v.astype(NP[dt])
        ref[a.name] = r.astype(np.float64)
        dev_t[a.name] = torch.from_numpy(np.ascontiguousarray(r)).to(device)
    return dev_t, ref


def random_arrays(plan, m, n, rng, *, kind="normal", scale=1.0):
    """FP64 arrays per plan argument (roles after any i/j swap)."""
    out = {}
    for a in plan.args:
        rows = {"i": m, "j": n, "param": None}[a.role]
        shape = (a.dim,) if rows is None else (rows, a.dim)
        if kind == "normal":
            v = rng.standard_normal(shape) * scale
        elif kind == "uniform":
            v = rng.uniform(-scale, scale, shape)
        elif kind == "int":
            v = rng.integers(-int(scale), int(scale) + 1, shape).astype(np.float64)
        else:
            raise ValueError(kind)
        if a.role == "param":
            v = np.abs(v) + 0.5  # strictly positive parameters (scales, thresholds)
        out[a.name] = v
    return out


# --------------------------------------------------------------- reference
@dataclass
class Terms:
    """Term-by-term FP64 values and evaluation error bounds."""

    F: np.ndarray  # (M, N, D) float64 (bool for COUNT)
    T: np.ndarray  # (M, N, D) error bound (inf: undetermined); COUNT: ambiguity (bool)
    dtype: str
    notes: list = field(default_factory=list)


def _torch_terms(plan32, ref):
    """Independent FP32 evaluation (Torch CPU oracle) of the FP32 plan, (M, N, D)."""
    red = OracleReduction(plan32, "cpu")
    inputs = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in ref.items()}
    m = next(inputs[a.name].shape[0] for a in plan32.args if a.role == "i")
    n = next(inputs[a.name].shape[0] for a in plan32.args if a.role == "j")
    with torch.no_grad():
        vals = red._evaluate(inputs, 0, m, 0, n)
    return vals.numpy()


def _conditioning(plan, ref, F, u, patterns=2):
    """FP64 sensitivity of each term to a relative perturbation ``u`` of the inputs.

    A correct backend computes the exact formula of inputs perturbed by a few
    ``u`` (FMA contraction, different libm, operation order); a single FP32
    evaluation (Torch) may be right by chance on an ill-conditioned term
    (cancellation, ``cos`` near a zero). We take the largest gap over
    ``patterns`` draws of fixed ± signs.
    """
    rng = np.random.default_rng(0x5EED)
    C = np.zeros(F.shape)
    for _ in range(patterns):
        sgn = {k: rng.choice((-1.0, 1.0), size=np.shape(v)) for k, v in ref.items()}
        for s in (1.0, -1.0):
            pert = {k: v * (1.0 + s * u * sgn[k]) for k, v in ref.items()}
            with np.errstate(all="ignore"):
                P = evaluate_numpy(plan, pert)
                d = np.where(np.isfinite(F) & np.isfinite(P), np.abs(P - F), 0.0)
            C = np.maximum(C, d)
    return C


def _rounding_samples(plan, plan32, ref, samples=4):
    """FP32/FP64 gap on neighboring inputs (relative perturbations ``k·u32``).

    A single FP32 evaluation may by chance round an intermediate almost
    exactly (``atan2(x, c) + 1`` after cancellation, ``exp`` of an amplified
    argument) and underestimate the intermediate rounding error that a
    correct backend (libm within 1-2 ulp) legitimately makes. We take the
    largest gap over a few neighbors; non-finite classes are ignored.
    """
    rng = np.random.default_rng(0xACC)
    E = 0.0
    u32 = U["float32"]
    for r in range(samples):
        pert = {k: np.asarray(v, dtype=np.float64)
                * (1.0 + (1 + r % 2) * u32 * rng.choice((-1.0, 1.0), size=np.shape(v)))
                for k, v in ref.items()}
        try:
            with np.errstate(all="ignore"):
                Fp = evaluate_numpy(plan, pert)
                Gp = _torch_terms(plan32, pert).astype(np.float64)
                Ep = np.where(np.isfinite(Fp) & np.isfinite(Gp), np.abs(Gp - Fp), 0.0)
        except Exception:  # pragma: no cover - neighbor evaluation failed
            continue
        E = np.maximum(E, Ep)
    return E


def terms(plan, ref, plan32=None) -> Terms:
    """FP64 reference and ``T`` bounds (see the module docstring)."""
    with np.errstate(all="ignore"):
        F = evaluate_numpy(plan, ref)
    dt = plan.dtype
    m, n, d = F.shape
    if m == 0 or n == 0:
        z = np.zeros(F.shape, dtype=bool if F.dtype == bool else np.float64)
        return Terms(F, z, dt)
    if plan32 is None:
        plan32 = plan
    try:
        with np.errstate(all="ignore"):
            G = _torch_terms(plan32, ref).astype(np.float64 if F.dtype != bool else bool)
    except Exception as exc:  # pragma: no cover - FP32 plan cannot be built
        G = None
        note = f"no FP32 evaluation ({type(exc).__name__}: {exc})"
    if F.dtype == bool:
        amb = np.zeros(F.shape, bool) if G is None else (G != F)
        return Terms(F, amb, dt)
    if G is None:
        T = 2.0**-20 * np.abs(F)
        return Terms(F, T, dt, [note])
    same_class = (np.isnan(F) == np.isnan(G)) & (np.isinf(F) == np.isinf(G))
    same_class &= ~(np.isinf(F) & (np.sign(F) != np.sign(G)))
    with np.errstate(all="ignore"):
        E = np.where(np.isfinite(F) & np.isfinite(G), np.abs(G - F), 0.0)
    exact = E == 0.0
    E = np.maximum(E, _rounding_samples(plan, plan32, ref))
    u = U[dt]
    scale = 1.0 if dt == "float32" else U["float64"] / U["float32"]
    C = _conditioning(plan, ref, F, u)
    with np.errstate(all="ignore"):
        T = np.where(exact, 0.0, 4.0 * (E * scale + C) + 4.0 * u * np.abs(F) + 4.0 * TINY[dt])
    T = np.where(same_class, T, np.inf)
    T = np.where(np.isfinite(F), T, np.where(same_class, 0.0, np.inf))
    return Terms(F, T, dt)


def reduce_ref(kind, k, F):
    """Contract reduction of F (M, N, D) in FP64, vectorized. dict output -> ndarray."""
    m, n, d = F.shape
    if kind is R.SUM:
        with np.errstate(all="ignore"):
            return {"value": F.sum(axis=1) if n else np.zeros((m, d))}
    if kind is R.COUNT:
        return {"value": F.sum(axis=1, dtype=np.int64) if n else np.zeros((m, d), np.int64)}
    if kind in SOFT_KINDS:
        return {"value": _soft_ref(kind, F)[0]}
    nan = np.isnan(F)
    if kind in (R.MIN, R.MAX, R.ARGMIN, R.ARGMAX):
        is_min = kind in (R.MIN, R.ARGMIN)
        fill = math.inf if is_min else -math.inf
        G = np.where(nan, fill, F)
        if n == 0:
            val = np.full((m, d), fill)
            idx = np.full((m, d), -1, np.int64)
        else:
            val = G.min(axis=1) if is_min else G.max(axis=1)
            valid = ~nan & (G == val[:, None, :])
            first = np.argmax(valid, axis=1)
            idx = np.where(valid.any(axis=1), first, -1).astype(np.int64)
        return {"value": val} if kind in (R.MIN, R.MAX) else {"index": idx}
    assert d == 1
    vals = np.full((m, k), math.inf)
    idx = np.full((m, k), -1, np.int64)
    if n and k:
        f = F[:, :, 0]
        isn = np.isnan(f)
        key = np.where(isn, math.inf, f)
        js = np.broadcast_to(np.arange(n), (m, n))
        order = np.lexsort((js, key, isn), axis=1)[:, :k]
        kk = order.shape[1]
        sel_nan = np.take_along_axis(isn, order, axis=1)
        vals[:, :kk] = np.where(sel_nan, math.inf, np.take_along_axis(f, order, axis=1))
        idx[:, :kk] = np.where(sel_nan, -1, order)
    out = {"value": vals, "index": idx}
    if kind is R.KMIN:
        return {"value": vals}
    if kind is R.ARGKMIN:
        return {"index": idx}
    return out


def _soft_ref(kind, F):
    """LOGSUMEXP / SUMSOFTMAXWEIGHT in FP64 (amendment v0.5), vectorized.

    Returns (value, softmax weights p (M, N, Ds), class); class: 0 finite,
    1 forced NaN, 2 +inf (LSE) / NaN (SSMW), 3 empty (-inf / NaN).
    """
    m, n, d = F.shape
    f = F if kind is R.LOGSUMEXP else F[:, :, :1]
    live = f != -math.inf
    nan = np.isnan(f).any(axis=1)
    pinf = (f == math.inf).any(axis=1)
    empty = ~live.any(axis=1)
    with np.errstate(all="ignore"):
        fm = np.where(live & ~np.isnan(f) & (f != math.inf), f, -math.inf)
        mx = fm.max(axis=1) if n else np.full((m, f.shape[2]), -math.inf)
        w = np.where(live, np.exp(np.where(live, f, 0.0) - mx[:, None, :]), 0.0)
        s = w.sum(axis=1)
        p = w / s[:, None, :]
        if kind is R.LOGSUMEXP:
            val = mx + np.log(s)
            val = np.where(empty, -math.inf, np.where(pinf, math.inf, val))
            val = np.where(nan, math.nan, val)
        else:
            g = F[:, :, 1:]
            val = np.where(live, w * g, 0.0).sum(axis=1) / s
            bad = (nan | pinf | empty)
            val = np.where(bad, math.nan, val)
    cls = np.where(nan, 1, np.where(pinf, 2, np.where(empty, 3, 0)))
    return val, p, cls


# ----------------------------------------------------------------- judging
class Mismatch(AssertionError):
    pass


def _fail(msg, **ctx):
    detail = ", ".join(f"{k}={v!r}" for k, v in ctx.items())
    raise Mismatch(f"{msg} ({detail})")


def check(plan, got: dict, t: Terms, *, acc_dtype=None):
    """Compares the backend outputs (NumPy dict) with the reference; raises Mismatch.

    Returns the number of rows not judged (undetermined).
    """
    kind, k = plan.reduction.kind, plan.reduction.k
    F, T = t.F, t.T
    m, n, d = F.shape
    ref = reduce_ref(kind, k, F)
    for key, r in ref.items():
        g = got[key]
        if g.shape != r.shape:
            _fail("shape", key=key, got=g.shape, want=r.shape)
    skipped = 0
    if m == 0:
        return 0
    if kind is R.COUNT:
        g, r = got["value"], ref["value"]
        amb = T.sum(axis=1)
        bad = np.abs(g - r) > amb
        if bad.any():
            i, c = np.argwhere(bad)[0]
            _fail("COUNT", i=int(i), c=int(c), got=int(g[i, c]), want=int(r[i, c]), ambiguous=int(amb[i, c]))
        return 0
    row_undet = np.isinf(T).any(axis=1) if n else np.zeros((m, d), bool)  # (M, D)
    if kind in SOFT_KINDS:
        return _check_soft(plan, got["value"].astype(np.float64), F, T, row_undet, acc_dtype)
    if kind is R.SUM:
        acc = acc_dtype or plan.acc_dtype
        ua = U[acc]
        g, r = got["value"].astype(np.float64), ref["value"]
        with np.errstate(all="ignore"):
            finite_terms = np.where(np.isfinite(F), np.abs(F), 0.0)
            L1 = finite_terms.sum(axis=1)
            gamma = n * ua / (1 - n * ua) if n * ua < 1 else math.inf
            tol = np.where(np.isfinite(T), T, 0.0).sum(axis=1) + gamma * L1 + 4 * TINY[acc]
            # exact terms that are multiples of 1/2 and L1 < 2**-2/u: every partial sum is
            # representable, the summation order does not matter -> exact equality
            halves = np.where(np.isfinite(F), F * 2.0, 0.0)
            exact_rows = ((T == 0.0) & (halves == np.round(halves))).all(axis=1) & (2.0 * L1 < 0.25 / ua)
            tol = np.where(exact_rows, 0.0, tol)
        maxacc = float(np.finfo(NP[acc]).max)
        for i in range(m):
            for c in range(d):
                if row_undet[i, c] or L1[i, c] * (1 + 1e-3) >= maxacc:
                    skipped += 1
                    continue
                want, have = r[i, c], g[i, c]
                if math.isnan(want) or math.isinf(want):
                    if not (have == want or (math.isnan(want) and math.isnan(have))):
                        _fail("SUM non-finite", i=i, c=c, got=have, want=want)
                    continue
                if not abs(have - want) <= tol[i, c]:
                    _fail("SUM", i=i, c=c, got=have, want=want, tol=float(tol[i, c]), L1=float(L1[i, c]))
        return skipped
    if kind in (R.MIN, R.MAX, R.ARGMIN, R.ARGMAX):
        is_min = kind in (R.MIN, R.ARGMIN)
        val = reduce_ref(R.MIN if is_min else R.MAX, None, F)["value"]
        idx = reduce_ref(R.ARGMIN if is_min else R.ARGMAX, None, F)["index"]
        for i in range(m):
            for c in range(d):
                if row_undet[i, c]:
                    skipped += 1
                    continue
                Tr = T[i, :, c]
                tmax = float(Tr.max()) if n else 0.0
                if kind in (R.MIN, R.MAX):
                    have, want = float(got["value"][i, c]), float(val[i, c])
                    if math.isinf(want) or tmax == 0.0:
                        ok = have == want
                    else:
                        ok = abs(have - want) <= tmax
                    if not ok:
                        _fail(kind.value, i=i, c=c, got=have, want=want, tol=tmax)
                    continue
                have, want = int(got["index"][i, c]), int(idx[i, c])
                _check_arg(F[i, :, c], Tr, have, want, is_min, kind.value, i, c)
        return skipped
    # KMIN / ARGKMIN / KMIN_ARGKMIN
    f = F[:, :, 0]
    Tm = T[:, :, 0]
    for i in range(m):
        if n and np.isinf(Tm[i]).any():
            skipped += 1
            continue
        full = reduce_ref(R.KMIN_ARGKMIN, k, F[i:i + 1])
        rv, ri = full["value"][0], full["index"][0]
        tmax = float(Tm[i].max()) if n else 0.0
        nvalid = int((ri >= 0).sum())
        if "value" in got:
            gv = got["value"][i].astype(np.float64)
            if not (gv[nvalid:] == math.inf).all():
                _fail("KMIN padding", i=i, got=gv.tolist(), want=rv.tolist())
            for r in range(nvalid):
                ok = gv[r] == rv[r] if (tmax == 0.0 or math.isinf(rv[r])) else abs(gv[r] - rv[r]) <= tmax
                if not ok:
                    _fail("KMIN", i=i, rank=r, got=float(gv[r]), want=float(rv[r]), tol=tmax)
            if nvalid > 1 and not (gv[1:nvalid] >= gv[:nvalid - 1]).all():
                _fail("KMIN not sorted", i=i, got=gv.tolist())
        if "index" in got:
            gi = got["index"][i].astype(np.int64)
            if not (gi[nvalid:] == -1).all():
                _fail("ARGKMIN padding", i=i, got=gi.tolist(), want=ri.tolist())
            sel = gi[:nvalid]
            if ((sel < 0) | (sel >= n)).any() or len(set(sel.tolist())) != nvalid:
                _fail("ARGKMIN invalid indices", i=i, got=sel.tolist())
            if tmax == 0.0:
                if not np.array_equal(sel, ri[:nvalid]):
                    _fail("ARGKMIN (exact)", i=i, got=sel.tolist(), want=ri[:nvalid].tolist())
            else:
                fv = f[i, sel]
                if np.isnan(fv).any():
                    _fail("ARGKMIN on NaN", i=i, got=sel.tolist())
                if not (np.abs(fv - rv[:nvalid]) <= Tm[i, sel] + tmax).all():
                    _fail("ARGKMIN", i=i, got=sel.tolist(), want=ri[:nvalid].tolist())
                if "value" in got and not (np.abs(got["value"][i][:nvalid] - fv) <= Tm[i, sel] + tmax).all():
                    _fail("KMIN/ARGKMIN inconsistent", i=i)
    return skipped


def _check_soft(plan, g, F, T, row_undet, acc_dtype):
    """Error bound: term perturbation (T), rounding of the weights, of the sum and of the log.

    LOGSUMEXP is 1-Lipschitz in the max norm of the terms; a relative error δ
    on the sum of the weights gives δ on the log. Weights: relative error
    ``δ_j = T_j + 2 u_acc |f_j - m| + 8 u_acc`` (term, shift, exp). SUMSOFTMAXWEIGHT:
    weighted mean, error ``Σ p_j (δ_j |g_j - out| + T_g,j) + (γ + 4u) Σ p_j |g_j|``.
    """
    kind = plan.reduction.kind
    acc = acc_dtype or plan.acc_dtype
    ua = U[acc]
    m, n, d = F.shape
    val, p, cls = _soft_ref(kind, F)
    gamma = (n + 4) * ua / (1 - (n + 4) * ua)
    lse = kind is R.LOGSUMEXP
    skipped = 0
    with np.errstate(all="ignore"):
        Tf = np.where(np.isfinite(T), T, 0.0)
        fs = F if lse else F[:, :, :1]
    for i in range(m):
        for c in range(d if lse else 1):
            undet = row_undet[i, c] if lse else row_undet[i].any()
            if undet:
                skipped += 1
                continue
            cc = c if lse else 0
            outs = range(c, c + 1) if lse else range(d - 1)
            if cls[i, cc] != 0:
                for e in outs:
                    have, want = g[i, e], val[i, e]
                    if not (have == want or (math.isnan(have) and math.isnan(want))):
                        _fail(f"{kind.value} special", i=i, e=e, got=have, want=want, cls=int(cls[i, cc]))
                continue
            pj = p[i, :, cc]
            live = pj > 0
            with np.errstate(all="ignore"):
                shift = np.where(live, np.abs(fs[i, :, cc] - fs[i, :, cc][live].max()), 0.0) if live.any() \
                    else np.zeros(n)
            delta = Tf[i, :, cc] + 2 * ua * shift + 8 * ua
            for e in outs:
                have, want = g[i, e], val[i, e]
                if not math.isfinite(want):  # NaN from a live weight g (propagated)
                    if not (have == want or (math.isnan(have) and math.isnan(want))):
                        _fail(f"{kind.value} non-finite", i=i, e=e, got=have, want=want)
                    continue
                if lse:
                    tol = float((pj * delta).sum()) + 4 * gamma + 4 * ua * abs(want) + 4 * TINY[acc]
                else:
                    gj, Tg = F[i, :, 1 + e], Tf[i, :, 1 + e]
                    with np.errstate(all="ignore"):
                        gl = np.where(live, np.abs(gj), 0.0)
                        tol = (4 * float((pj * (delta * (gl + abs(want)) + Tg)).sum())
                               + (4 * gamma + 8 * ua) * float((pj * gl).sum()) + 4 * ua * abs(want)
                               + 4 * TINY[acc])
                if not math.isfinite(have) or not abs(have - want) <= tol:
                    _fail(kind.value, i=i, e=e, got=have, want=want, tol=tol)
    return skipped


def _check_arg(f, Tr, have, want, is_min, what, i, c):
    n = f.shape[0]
    if want < 0:
        if have != -1:
            _fail(what, i=i, c=c, got=have, want=-1)
        return
    if not 0 <= have < n or math.isnan(f[have]):
        _fail(what + " out of bounds or NaN", i=i, c=c, got=have, want=want)
    best = f[want]
    if Tr.max() == 0.0 or math.isinf(best):
        # exact evaluation: exact semantics (smallest j reaching the extremum)
        if have != want:
            _fail(what + " (exact)", i=i, c=c, got=have, want=want, fgot=float(f[have]), fwant=float(best))
        return
    s = 1.0 if is_min else -1.0
    near = s * (f - best) <= Tr + Tr[want]
    near &= ~np.isnan(f)
    if not near[have]:
        _fail(what, i=i, c=c, got=have, want=want, fgot=float(f[have]), fwant=float(best))
    if near.sum() == 1 and have != want:
        _fail(what + " (separated)", i=i, c=c, got=have, want=want)


# ------------------------------------------------------------------ execution
def run_case(backend, device, expr, arrays, dtype, *, acc_dtype=None):
    """Compiles, runs on the backend and judges against the reference. Returns the number of rows not judged."""
    plan = expr.to_plan(dtype, acc_dtype=acc_dtype)
    try:
        plan32 = expr.to_plan("float32") if dtype == "float64" else plan
    except kp.ConstantConversionError:
        plan32 = None
    run = kp.compile(expr, dtype=dtype, backend=backend, acc_dtype=acc_dtype)
    dev_t, ref = to_inputs(plan, arrays, device)
    out = run(**dev_t)
    synchronize(device)
    m = next(dev_t[a.name].shape[0] for a in plan.args if a.role == "i")
    keys = list(output_shapes(plan, m))
    out = out if isinstance(out, tuple) else (out,)
    got = {key: o.cpu().numpy() for key, o in zip(keys, out)}
    for key, o in zip(keys, out):
        want_dt = {"value": TORCH.get(plan.acc_dtype if plan.reduction.kind in (R.SUM, *SOFT_KINDS)
                                      else plan.dtype),
                   "index": torch.int64}[key]
        if plan.reduction.kind is R.COUNT:
            want_dt = torch.int64
        assert o.dtype == want_dt, (key, o.dtype, want_dt)
        assert o.device == torch.device(device), (o.device, device)
    t = terms(plan, ref, plan32)
    return check(plan, got, t, acc_dtype=acc_dtype)
