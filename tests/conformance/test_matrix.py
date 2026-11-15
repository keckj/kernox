"""Matrice systématique : réductions × dtypes × dimensions × K × tailles.

Données sur grille entière : l'évaluation est exacte en FP32 comme en FP64
(sommes bornées par 2**23), donc la comparaison est **exacte**, égalités
massives comprises (plus petit j, ordre lexicographique (valeur, j)).
Profil rapide : un sous-ensemble couvrant (chaque dimension, chaque K et
chaque famille de tailles apparaît) ; profil long (``slow``) : produit complet.
"""

from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pytest

from ._support import K_KINDS, R, run_case

import kernox as kp

DIMS = (1, 2, 3, 7, 16, 64, 65)
K_SPECS = (0, 1, 2, 31, 32, 33, 64, 128, 256, "N-1", "N", "N+1")
# bords de tuile/bloc (31, 32, 33, 255, 256, 257), multiples, premiers, M << N, M >> N, vides
SIZES_QUICK = ((1, 1), (31, 33), (256, 255), (257, 32), (3, 1031), (1031, 3), (0, 7), (7, 0))
SIZES_SLOW = SIZES_QUICK + ((32, 32), (33, 257), (255, 256), (512, 1024), (97, 4099), (4099, 97),
                            (1, 8191), (8191, 1), (2, 65537), (12289, 2))
NONK = (R.SUM, R.MIN, R.MAX, R.ARGMIN, R.ARGMAX, R.COUNT)
DTYPES = ("float32", "float64")
MAX_TERMS = 8_000_000  # M*N*D de la référence FP64 (mémoire hôte)


def _sizes(slow, d):
    return [(m, n) for m, n in (SIZES_SLOW if slow else SIZES_QUICK) if m * n * d <= MAX_TERMS]


def _amplitude(d, n):
    # |x - y| <= 2a : somme sur j de d termes (2a)^2 < 2**23 (exacte en FP32)
    return max(1, int(math.sqrt(2.0**23 / (4 * d * max(n, 1)))))


def _grid(rng, rows, d, a):
    return rng.integers(-a, a + 1, (rows, d)).astype(np.float64)


def _formula(name, d):
    """(expression réduite par l'appelant, générateur d'entrées)."""
    x, y = kp.Vi("x", d), kp.Vj("y", d)
    if name == "sqdist":
        return ((x - y) ** 2).sum(), lambda rng, m, n, a: {"x": _grid(rng, m, d, a), "y": _grid(rng, n, d, a)}
    if name == "multi":
        # plusieurs arguments i et j, paramètre scalaire (puissance de 2 : exact)
        ai, bj, s = kp.Vi("ai", 1), kp.Vj("bj", 1), kp.Param("s", 1)
        f = ((x - y) ** 2).sum() * ai + bj * s
        return f, lambda rng, m, n, a: {"x": _grid(rng, m, d, a), "y": _grid(rng, n, d, a),
                                        "ai": _grid(rng, m, 1, 2), "bj": _grid(rng, n, 1, 8),
                                        "s": np.array([0.5])}
    if name == "vecout":
        # sortie vectorielle (D composantes), paramètre vectoriel
        v, w = kp.Param("v", d), kp.Vj("w", d)
        f = (x - y) * v + w
        return f, lambda rng, m, n, a: {"x": _grid(rng, m, d, a), "y": _grid(rng, n, d, a),
                                        "v": rng.integers(-3, 4, d).astype(np.float64),
                                        "w": _grid(rng, n, d, a)}
    raise ValueError(name)


def _reduce(f, kind, k, over="j"):
    if kind is R.SUM:
        return f.sum_over(over)
    if kind is R.MIN:
        return f.min(over=over)
    if kind is R.MAX:
        return f.max(over=over)
    if kind is R.ARGMIN:
        return f.argmin(over=over)
    if kind is R.ARGMAX:
        return f.argmax(over=over)
    if kind is R.KMIN:
        return f.kmin(k, over=over)
    if kind is R.ARGKMIN:
        return f.argkmin(k, over=over)
    if kind is R.KMIN_ARGKMIN:
        return f.kmin_argkmin(k, over=over)
    raise ValueError(kind)


# K au-delà de cette borne : un backend peut refuser explicitement (UnsupportedError
# avant lancement, contrat v0) ; le refus est accepté et signalé, jamais un résultat faux.
K_GUARANTEED = 256


def _run_sizes(backend, expr, gen, dtype, sizes, d, *, seed, over="j", acc_dtype=None):
    name, dev = backend
    for m, n in sizes:
        rng = np.random.default_rng([seed, m, n, d])
        rows_i, rows_j = (m, n) if over == "j" else (n, m)
        arrays = gen(rng, rows_i, rows_j, _amplitude(d, n))
        try:
            run_case(name, dev, expr, arrays, dtype, acc_dtype=acc_dtype)
        except kp.UnsupportedError as exc:
            k = expr.to_plan(dtype).reduction.k
            if k is None or k <= K_GUARANTEED:
                raise
            warnings.warn(f"[{name}] K={k} non pris en charge : {exc}", stacklevel=2)
        except AssertionError as exc:
            raise AssertionError(f"[{name} {dtype} d={d} M={m} N={n} over={over} seed={seed}] {exc}") from exc


# ------------------------------------------------------------- non-K
def _nonk_cases(slow):
    cases = []
    for kind, dtype in itertools.product(NONK, DTYPES):
        dims = DIMS if slow else DIMS[(len(cases) % 3)::3]
        for d in dims:
            cases.append((kind, dtype, d))
    return cases


def _nonk(backend, kind, dtype, d, slow):
    sizes = _sizes(slow, d)
    if kind is R.COUNT:
        x, y, r = kp.Vi("x", d), kp.Vj("y", d), kp.Param("r", 1)
        f = (((x - y) ** 2).sum() <= r)
        g0 = _formula("sqdist", d)[1]

        def gen(rng, m, n, a):
            out = g0(rng, m, n, a)
            out["r"] = np.array([float(d * a * a) + 0.5])
            return out
        _run_sizes(backend, f.sum_over("j"), gen, dtype, sizes, d, seed=11)
        return
    for fname in ("sqdist", "multi", "vecout"):
        f, gen = _formula(fname, d)
        _run_sizes(backend, _reduce(f, kind, None), gen, dtype, sizes, d, seed={"sqdist": 1, "multi": 2, "vecout": 3}[fname])


@pytest.mark.parametrize("kind,dtype,d", _nonk_cases(False), ids=lambda v: getattr(v, "value", str(v)))
def test_matrix_quick(backend, kind, dtype, d):
    _nonk(backend, kind, dtype, d, slow=False)


@pytest.mark.slow
@pytest.mark.parametrize("kind,dtype,d", _nonk_cases(True), ids=lambda v: getattr(v, "value", str(v)))
def test_matrix_slow(backend, kind, dtype, d):
    _nonk(backend, kind, dtype, d, slow=True)


# --------------------------------------------------------------- K
def _k_cases(slow):
    cases = []
    for idx, (kind, dtype, k) in enumerate(itertools.product(K_KINDS, DTYPES, K_SPECS)):
        if slow:
            cases += [(kind, dtype, k, d) for d in DIMS]
        else:
            cases.append((kind, dtype, k, DIMS[idx % len(DIMS)]))
    return cases


def _kvalue(spec, n):
    if isinstance(spec, int):
        return spec
    return {"N-1": n - 1, "N": n, "N+1": n + 1}[spec]


def _k(backend, kind, dtype, spec, d, slow):
    f, gen = _formula("sqdist", d)
    sizes = _sizes(slow, d)
    if isinstance(spec, str):
        sizes = [(m, n) for m, n in sizes if n >= 1]
        for m, n in sizes:
            k = _kvalue(spec, n)
            _run_sizes(backend, _reduce(f, kind, k), gen, dtype, [(m, n)], d, seed=k)
        return
    _run_sizes(backend, _reduce(f, kind, spec), gen, dtype, sizes, d, seed=spec)


@pytest.mark.parametrize("kind,dtype,k,d", _k_cases(False), ids=lambda v: getattr(v, "value", str(v)))
def test_k_quick(backend, kind, dtype, k, d):
    _k(backend, kind, dtype, k, d, slow=False)


@pytest.mark.slow
@pytest.mark.parametrize("kind,dtype,k,d", _k_cases(True), ids=lambda v: getattr(v, "value", str(v)))
def test_k_slow(backend, kind, dtype, k, d):
    _k(backend, kind, dtype, k, d, slow=True)


# --------------------------------------------------------------- over="i", acc_dtype
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", (R.SUM, R.ARGMIN, R.MAX, R.COUNT, R.KMIN_ARGKMIN), ids=lambda v: v.value)
def test_over_i(backend, kind, dtype):
    d = 3
    if kind is R.COUNT:
        x, y, r = kp.Vi("x", d), kp.Vj("y", d), kp.Param("r", 1)
        expr = (((x - y) ** 2).sum() <= r).sum_over("i")

        def gen(rng, m, n, a):
            return {"x": _grid(rng, m, d, a), "y": _grid(rng, n, d, a), "r": np.array([4.5])}
    else:
        f, gen = _formula("vecout" if kind is not R.KMIN_ARGKMIN else "sqdist", d)
        expr = _reduce(f, kind, 5, over="i")
    # over="i" : M (lignes de sortie) est le nombre de lignes de y ; gen reçoit (lignes x, lignes y)
    _run_sizes(backend, expr, gen, dtype, [(1, 1), (33, 257), (1031, 3), (5, 0), (0, 5)], d, seed=3, over="i")


def test_acc_dtype_float64(backend):
    # SUM FP32 accumulée en FP64 : sortie float64, données non entières
    name, dev = backend
    x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 1)
    expr = (kp.exp(-((x - y) ** 2).sum()) * b).sum_over("j")
    for m, n in ((1, 1), (33, 4099), (257, 255)):
        rng = np.random.default_rng([m, n])
        arrays = {"x": rng.standard_normal((m, 3)), "y": rng.standard_normal((n, 3)), "b": rng.standard_normal((n, 1))}
        run_case(name, dev, expr, arrays, "float32", acc_dtype="float64")
