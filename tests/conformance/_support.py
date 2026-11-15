"""Outils communs de la suite de conformance (tous backends GPU).

Référence : la formule est évaluée terme à terme en FP64 par
``kernox.oracle.evaluate_numpy`` (lambdify NumPy de symox, chemin
indépendant des backends et de l'oracle Torch) sur les entrées **déjà
arrondies au dtype** du plan, puis réduite selon le contrat v0 (NaN ignorés,
égalités au plus petit j, complétion (+inf, −1)).

Tolérances (voir ``docs/conformance.md``) : chaque terme reçoit une borne
d'erreur d'évaluation ``T_ij`` ; elle est nulle quand l'évaluation FP32 de la
formule reproduit exactement le FP64 (données entières : comparaison exacte,
égalités comprises). Sinon ``T_ij = 4 (E_ij + C_ij) + 4 u |F_ij| + plancher``, où
``E_ij`` est l'écart mesuré entre une évaluation FP32 indépendante (Torch CPU,
oracle) et la référence FP64, mis à l'échelle ``u64/u32`` pour un plan FP64, et
``C_ij`` la sensibilité FP64 du terme à une perturbation relative ``u`` des
entrées (conditionnement : annulations, fonctions près d'un zéro).
Un terme dont la finitude diffère entre les deux évaluations (dépassement en
FP32, NaN d'un seul côté) est « indéterminé » et sa ligne n'est pas jugée.
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
    """Device Torch du backend ``name`` ou None (jamais simulé)."""
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
    """``(device, None)`` si le backend est utilisable, ``(None, raison)`` sinon."""
    wanted = os.environ.get("KP_CONFORMANCE_BACKENDS")
    if wanted and name not in wanted.split(","):
        return None, f"exclu par KP_CONFORMANCE_BACKENDS={wanted}"
    dev = device_for(name)
    if dev is None:
        return None, f"aucun device Torch pour le backend {name}"
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


# ----------------------------------------------------------------- entrées
def to_inputs(plan, arrays: dict, device):
    """Arrondit les tableaux FP64 au dtype du plan. Retourne (tenseurs device, FP64 arrondis)."""
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
    """Tableaux FP64 par argument du plan (rôles après échange éventuel i/j)."""
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
            v = np.abs(v) + 0.5  # paramètres strictement positifs (échelles, seuils)
        out[a.name] = v
    return out


# --------------------------------------------------------------- référence
@dataclass
class Terms:
    """Valeurs FP64 terme à terme et bornes d'erreur d'évaluation."""

    F: np.ndarray  # (M, N, D) float64 (bool pour COUNT)
    T: np.ndarray  # (M, N, D) borne d'erreur (inf : indéterminé) ; COUNT : ambiguïté (bool)
    dtype: str
    notes: list = field(default_factory=list)


def _torch_terms(plan32, ref):
    """Évaluation FP32 indépendante (oracle Torch CPU) du plan FP32, (M, N, D)."""
    red = OracleReduction(plan32, "cpu")
    inputs = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in ref.items()}
    m = next(inputs[a.name].shape[0] for a in plan32.args if a.role == "i")
    n = next(inputs[a.name].shape[0] for a in plan32.args if a.role == "j")
    with torch.no_grad():
        vals = red._evaluate(inputs, 0, m, 0, n)
    return vals.numpy()


def _conditioning(plan, ref, F, u, patterns=2):
    """Sensibilité FP64 de chaque terme à une perturbation relative ``u`` des entrées.

    Un backend correct calcule la formule exacte d'entrées perturbées de
    quelques ``u`` (contraction FMA, libm différente, ordre des opérations) ;
    une seule évaluation FP32 (Torch) peut tomber juste par chance sur un
    terme mal conditionné (annulation, ``cos`` près d'un zéro). On prend le
    plus grand écart sur ``patterns`` tirages de signes ± fixés.
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
    """Écart FP32/FP64 sur des entrées voisines (perturbations relatives ``k·u32``).

    Une seule évaluation FP32 peut arrondir par chance un intermédiaire
    presque exactement (``atan2(x, c) + 1`` après annulation, ``exp`` d'un
    argument amplifié) et sous-estimer l'erreur d'arrondi intermédiaire
    qu'un backend correct (libm à 1-2 ulp) commet légitimement. On prend le
    plus grand écart sur quelques voisins ; les classes non finies sont ignorées.
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
        except Exception:  # pragma: no cover - évaluation voisine impossible
            continue
        E = np.maximum(E, Ep)
    return E


def terms(plan, ref, plan32=None) -> Terms:
    """Référence FP64 et bornes ``T`` (voir la docstring du module)."""
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
    except Exception as exc:  # pragma: no cover - plan FP32 non constructible
        G = None
        note = f"pas d'évaluation FP32 ({type(exc).__name__}: {exc})"
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
    """Réduction du contrat sur F (M, N, D) FP64, vectorisée. dict sortie -> ndarray."""
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
    """LOGSUMEXP / SUMSOFTMAXWEIGHT FP64 (avenant v0.5), vectorisé.

    Retourne (valeur, poids softmax p (M, N, Ds), classe) ; classe : 0 fini,
    1 NaN imposé, 2 +inf (LSE) / NaN (SSMW), 3 vide (−inf / NaN).
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


# ---------------------------------------------------------------- jugement
class Mismatch(AssertionError):
    pass


def _fail(msg, **ctx):
    detail = ", ".join(f"{k}={v!r}" for k, v in ctx.items())
    raise Mismatch(f"{msg} ({detail})")


def check(plan, got: dict, t: Terms, *, acc_dtype=None):
    """Compare les sorties du backend (dict NumPy) à la référence ; lève Mismatch.

    Retourne le nombre de lignes non jugées (indéterminées).
    """
    kind, k = plan.reduction.kind, plan.reduction.k
    F, T = t.F, t.T
    m, n, d = F.shape
    ref = reduce_ref(kind, k, F)
    for key, r in ref.items():
        g = got[key]
        if g.shape != r.shape:
            _fail("forme", key=key, got=g.shape, want=r.shape)
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
            # termes exacts multiples de 1/2 et L1 < 2**-2/u : toute somme partielle est
            # représentable, l'ordre de sommation est indifférent -> égalité exacte
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
                        _fail("SUM non fini", i=i, c=c, got=have, want=want)
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
                _fail("KMIN complétion", i=i, got=gv.tolist(), want=rv.tolist())
            for r in range(nvalid):
                ok = gv[r] == rv[r] if (tmax == 0.0 or math.isinf(rv[r])) else abs(gv[r] - rv[r]) <= tmax
                if not ok:
                    _fail("KMIN", i=i, rank=r, got=float(gv[r]), want=float(rv[r]), tol=tmax)
            if nvalid > 1 and not (gv[1:nvalid] >= gv[:nvalid - 1]).all():
                _fail("KMIN non trié", i=i, got=gv.tolist())
        if "index" in got:
            gi = got["index"][i].astype(np.int64)
            if not (gi[nvalid:] == -1).all():
                _fail("ARGKMIN complétion", i=i, got=gi.tolist(), want=ri.tolist())
            sel = gi[:nvalid]
            if ((sel < 0) | (sel >= n)).any() or len(set(sel.tolist())) != nvalid:
                _fail("ARGKMIN indices invalides", i=i, got=sel.tolist())
            if tmax == 0.0:
                if not np.array_equal(sel, ri[:nvalid]):
                    _fail("ARGKMIN (exact)", i=i, got=sel.tolist(), want=ri[:nvalid].tolist())
            else:
                fv = f[i, sel]
                if np.isnan(fv).any():
                    _fail("ARGKMIN sur NaN", i=i, got=sel.tolist())
                if not (np.abs(fv - rv[:nvalid]) <= Tm[i, sel] + tmax).all():
                    _fail("ARGKMIN", i=i, got=sel.tolist(), want=ri[:nvalid].tolist())
                if "value" in got and not (np.abs(got["value"][i][:nvalid] - fv) <= Tm[i, sel] + tmax).all():
                    _fail("KMIN/ARGKMIN incohérents", i=i)
    return skipped


def _check_soft(plan, g, F, T, row_undet, acc_dtype):
    """Borne d'erreur : perturbation des termes (T), arrondi des poids, de la somme et du log.

    LOGSUMEXP est 1-lipschitzienne en norme max des termes ; une erreur relative
    δ sur la somme des poids donne δ sur le log. Poids : erreur relative
    ``δ_j = T_j + 2 u_acc |f_j − m| + 8 u_acc`` (terme, décalage, exp). SUMSOFTMAXWEIGHT :
    moyenne pondérée, erreur ``Σ p_j (δ_j |g_j − out| + T_g,j) + (γ + 4u) Σ p_j |g_j|``.
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
                        _fail(f"{kind.value} spécial", i=i, e=e, got=have, want=want, cls=int(cls[i, cc]))
                continue
            pj = p[i, :, cc]
            live = pj > 0
            with np.errstate(all="ignore"):
                shift = np.where(live, np.abs(fs[i, :, cc] - fs[i, :, cc][live].max()), 0.0) if live.any() \
                    else np.zeros(n)
            delta = Tf[i, :, cc] + 2 * ua * shift + 8 * ua
            for e in outs:
                have, want = g[i, e], val[i, e]
                if not math.isfinite(want):  # NaN d'un poids g vivant (propagé)
                    if not (have == want or (math.isnan(have) and math.isnan(want))):
                        _fail(f"{kind.value} non fini", i=i, e=e, got=have, want=want)
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
        _fail(what + " hors bornes ou NaN", i=i, c=c, got=have, want=want)
    best = f[want]
    if Tr.max() == 0.0 or math.isinf(best):
        # évaluation exacte : sémantique exacte (plus petit j atteignant l'extremum)
        if have != want:
            _fail(what + " (exact)", i=i, c=c, got=have, want=want, fgot=float(f[have]), fwant=float(best))
        return
    s = 1.0 if is_min else -1.0
    near = s * (f - best) <= Tr + Tr[want]
    near &= ~np.isnan(f)
    if not near[have]:
        _fail(what, i=i, c=c, got=have, want=want, fgot=float(f[have]), fwant=float(best))
    if near.sum() == 1 and have != want:
        _fail(what + " (séparé)", i=i, c=c, got=have, want=want)


# ------------------------------------------------------------------ exécution
def run_case(backend, device, expr, arrays, dtype, *, acc_dtype=None):
    """Compile, exécute sur le backend et juge contre la référence. Retourne le nombre de lignes non jugées."""
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
