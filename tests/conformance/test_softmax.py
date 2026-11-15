"""LOGSUMEXP et SUMSOFTMAXWEIGHT (avenant v0.5) sur chaque backend GPU.

Formules réelles (noyau gaussien en log, attention softmax E = 1, 4, 16) et
cas adverses de l'avenant : grands f (1e4), f très négatifs (−1e4), mélanges
±inf, NaN, N = 0, N = 1, ligne dominante, partition de j avec tranches vides.
Un backend qui ne met pas encore ces réductions en œuvre répond
``Support(False)`` : le test est alors sauté (jamais simulé).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import kernox as kp
from kernox.errors import UnsupportedError

from ._support import run_case

DTYPES = ("float32", "float64")
INF, NAN = math.inf, math.nan


def _run(backend, expr, arrays, dtype, **kw):
    name, dev = backend
    try:
        return run_case(name, dev, expr, arrays, dtype, **kw)
    except UnsupportedError as exc:
        pytest.skip(f"{name} : {exc}")


def _gauss(D):
    x, y, s = kp.Vi("x", D), kp.Vj("y", D), kp.Param("s")
    return (-((x - y) ** 2).sum() / (2.0 * s ** 2)).logsumexp(over="j")


def _attn(E, dq=8):
    q, k, v = kp.Vi("q", dq), kp.Vj("k", dq), kp.Vj("v", E)
    return (q.dot(k) / math.sqrt(dq)).sumsoftmaxweight(v, over="j")


SIZES = ((1, 1), (7, 33), (64, 300), (3, 20000))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("D", [3, 16])
def test_gaussian_log_kernel(backend, dtype, D):
    for m, n in SIZES:
        rng = np.random.default_rng([D, m, n])
        arrays = {"x": rng.standard_normal((m, D)), "y": rng.standard_normal((n, D)), "s": np.array([0.3])}
        _run(backend, _gauss(D), arrays, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("E", [1, 4, 16])
def test_attention(backend, dtype, E):
    for m, n in SIZES:
        rng = np.random.default_rng([E, m, n])
        arrays = {"q": 2 * rng.standard_normal((m, 8)), "k": 2 * rng.standard_normal((n, 8)),
                  "v": rng.standard_normal((n, E))}
        _run(backend, _attn(E), arrays, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_vector_lse_over_i_and_acc64(backend, dtype):
    x, y = kp.Vi("x", 2), kp.Vj("y", 2)
    d2 = ((x - y) ** 2).sum()
    rng = np.random.default_rng(3)
    arrays = {"x": rng.standard_normal((50, 2)), "y": rng.standard_normal((400, 2))}
    _run(backend, kp.cat(-d2, -4 * d2, x[0] * y[1]).logsumexp(over="j"), arrays, dtype)
    _run(backend, (-d2).logsumexp(over="i"), arrays, dtype)
    _run(backend, (-d2).sumsoftmaxweight(kp.cat(x, y), over="i"), arrays, dtype)
    if dtype == "float32":
        _run(backend, (-d2).logsumexp(over="j"), arrays, dtype, acc_dtype="float64")
        _run(backend, (-d2).sumsoftmaxweight(y, over="j"), arrays, dtype, acc_dtype="float64")


POOLS = {
    "big": [-1e4, -1e4 + 1.0, -9990.0, -9999.5],     # f = -y = 1e4 : décalage indispensable
    "very_negative": [1e4, 1e4 - 2.0, 1e4 + 3.0],
    "neg_inf_mix": [INF, 1.0, -2.0, 0.5],             # f = -inf : poids nul
    "inf_mix": [INF, -INF, 1.0, -1.0, 0.0],
    "nan": [NAN, 1.0, -2.0, 0.0],
    "mixed": [NAN, INF, -INF, -0.0, 0.0, 1e4, -1e4, 1.0],
    "dominant": [-800.0, 0.0, 1.0, 2.0],
}


def _special(kind, E=3):
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    f = x - y  # x = 0 : f_ij = -y_j exactement
    if kind == "lse":
        return f.logsumexp(over="j")
    return f.sumsoftmaxweight(kp.Vj("w", E) * 1.0, over="j")


@pytest.mark.parametrize("kind", ["lse", "ssmw"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("family", list(POOLS))
def test_special_values(backend, kind, dtype, family):
    for m, n in ((1, 1), (5, 33), (4, 1031), (3, 20000)):
        rng = np.random.default_rng([m, n, len(family)])
        y = rng.choice(np.array(POOLS[family]), (n, 1))
        arrays = {"x": np.zeros((m, 1)), "y": y, "w": rng.standard_normal((n, 3))}
        if kind == "lse":
            arrays.pop("w")
        try:
            _run(backend, _special(kind), arrays, dtype)
        except AssertionError as exc:
            raise AssertionError(f"[{family} M={m} N={n}] {exc}") from exc


@pytest.mark.parametrize("kind", ["lse", "ssmw"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_and_single(backend, kind, dtype):
    for n, ys in ((0, []), (1, [0.25]), (1, [INF]), (1, [-INF]), (1, [NAN]), (3, [INF, INF, INF]),
                  (2, [-INF, -INF]), (2, [-INF, INF])):
        arrays = {"x": np.zeros((4, 1)), "y": np.array(ys, dtype=np.float64).reshape(n, 1),
                  "w": np.arange(3.0 * n).reshape(n, 3)}
        if kind == "lse":
            arrays.pop("w")
        try:
            _run(backend, _special(kind), arrays, dtype)
        except AssertionError as exc:
            raise AssertionError(f"[y={ys}] {exc}") from exc


@pytest.mark.parametrize("kind", ["lse", "ssmw"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_partition_with_empty_slices(backend, kind, dtype):
    """M petit, N grand (partition de j) : longues plages f = −inf, une valeur vivante isolée."""
    n = 50000
    y = np.full((n, 1), INF)
    y[[123, 31000, 49999]] = [[1.0], [-2.0], [0.5]]
    rng = np.random.default_rng(9)
    arrays = {"x": np.zeros((5, 1)), "y": y, "w": rng.standard_normal((n, 3))}
    if kind == "lse":
        arrays.pop("w")
    _run(backend, _special(kind), arrays, dtype)
    y2 = np.full((n, 1), INF)  # toutes tranches vides : −inf / NaN
    arrays["y"] = y2
    _run(backend, _special(kind), arrays, dtype)
