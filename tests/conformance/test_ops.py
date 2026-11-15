"""Conformance of the PyKeOps parity operations (``kernox/ops.py``).

The symox DAG of each operation is printed by the backend and judged against
the shared FP64 reference (``run_case``). The keopscore semantics themselves
are checked on the oracle in ``tests/test_ops.py``. For discontinuous
operations (round, mod, step, one_hot, order-0 bspline), the data are taken
on a dyadic grid where every intermediate operation is exact in float32 as in
float64: the error-bound judgment cannot depend on a jump at a discontinuity.
"""

from __future__ import annotations

import numpy as np
import pytest

import kernox as kp

from ._support import run_case

DTYPES = ("float32", "float64")


def _grid(rng, shape, step=1 / 16, span=64):
    return rng.integers(-span, span + 1, size=shape) * step


@pytest.mark.parametrize("dtype", DTYPES)
def test_discontinuous_elementwise(backend, dtype):
    name, dev = backend
    x, y = kp.Vi("x", 3), kp.Vj("y", 3)
    f = (x.round() + y.round(1) * 2 + (x - y).relu() + (x * y).step() - x.mod(2.0, 0.5) + y.mod(0.75)
         + (x + y).clampint(-1, 2) + x.ifelse(y, -y))
    expr = (f * kp.exp(-0.01 * y.sqnorm2())).sum().sum_over("j")
    rng = np.random.default_rng(3)
    arrays = {"x": _grid(rng, (200, 3)), "y": _grid(rng, (150, 3))}
    run_case(name, dev, expr, arrays, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_smooth_elementwise(backend, dtype):
    name, dev = backend
    x, y = kp.Vi("x", 2), kp.Vj("y", 2)
    d = x - y
    f = d.sinxdivx() + d.sinc() * 0.5 + (d * d + 0.1).xlogx() + (d * d + 1).inv()
    expr = f.sum().sum_over("j")
    rng = np.random.default_rng(4)
    arrays = {"x": rng.standard_normal((300, 2)), "y": rng.standard_normal((250, 2))}
    arrays["y"][:3] = arrays["x"][:3]  # d = 0: sinxdivx(0) = 1 branches
    run_case(name, dev, expr, arrays, dtype)


@pytest.mark.parametrize("ds", (1, 3, 9))
@pytest.mark.parametrize("dtype", DTYPES)
def test_norms_and_distances(backend, dtype, ds):
    name, dev = backend
    x, y, s, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("s", ds), kp.Vj("b", 3)
    f = kp.exp(-x.weightedsqdist(y, s)) * x.normalize().dot(b) + x.sqdist(y) * 0.01 + y.weightedsqnorm(s) * 0.01
    expr = f.sum_over("j")
    rng = np.random.default_rng(5)
    S = rng.random((200, ds)) + 0.5 if ds != 9 else np.stack(
        [(lambda a: (a @ a.T + np.eye(3)).ravel())(rng.standard_normal((3, 3)) * 0.3) for _ in range(200)])
    X = rng.standard_normal((180, 3))
    X[0] = 0.0  # normalize(0) = 0
    arrays = {"x": X, "y": rng.standard_normal((200, 3)), "s": S, "b": rng.standard_normal((200, 3))}
    run_case(name, dev, expr, arrays, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_one_hot_elem_extract(backend, dtype):
    name, dev = backend
    f, x, b = kp.Vi("f", 1), kp.Vi("x", 4), kp.Vj("b", 5)
    expr = (f.one_hot(5) * b + x.elem(2) * b.extract(1, 3).sum() + x.extract(1, 2).sum()).sum_over("j")
    rng = np.random.default_rng(6)
    F = rng.integers(0, 9, size=(160, 1)) * 0.5  # half-integers: exact one_hot boundaries
    F[F > 4.0] -= 4.0
    arrays = {"f": F, "x": rng.standard_normal((160, 4)), "b": rng.standard_normal((140, 5))}
    run_case(name, dev, expr, arrays, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_linear_algebra(backend, dtype):
    name, dev = backend
    x, y, A = kp.Vi("x", 3), kp.Vj("y", 2), kp.Vj("A", 6)
    v = A.matvecmult(x)                     # (2,)
    w = x.vecmatmult(A)                     # (2,) with A viewed as (3, 2)
    t = x.tensorprod(y)                     # (6,)
    k = x.keops_kron(y, [3], [2])           # (6,)
    td = A.keops_tensordot(x, [2, 3], [3], [1], [0])  # (2,)
    tp = x.keops_tensordot(A, [3], [3, 2], [], [], [2, 0, 1])  # (18,), permuted
    f = kp.exp(-(v - y).sqnorm2() * 0.1) * (w.dot(y) + t.dot(k) * 0.1 + td.sum() + tp.sum() * 0.01)
    expr = f.sum_over("j")
    rng = np.random.default_rng(7)
    arrays = {"x": rng.standard_normal((170, 3)), "y": rng.standard_normal((130, 2)),
              "A": rng.standard_normal((130, 6)) * 0.5}
    run_case(name, dev, expr, arrays, dtype)


@pytest.mark.parametrize("k", (0, 1, 3))
@pytest.mark.parametrize("dtype", DTYPES)
def test_bspline(backend, dtype, k):
    name, dev = backend
    t, x, b = kp.Vj("t", 8), kp.Vi("x", 1), kp.Vj("b", 8 - k - 1)
    expr = t.bspline(x, k).dot(b).sum_over("j")
    rng = np.random.default_rng(8)
    T = np.sort(_grid(rng, (120, 8), step=1 / 8, span=16), axis=1)  # repeated knots possible
    arrays = {"t": T, "x": _grid(rng, (150, 1), step=1 / 32, span=72), "b": rng.standard_normal((120, 8 - k - 1))}
    run_case(name, dev, expr, arrays, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_softdtw_sqdist(backend, dtype):
    name, dev = backend
    x, y, g, b = kp.Vi("x", 6), kp.Vj("y", 5), kp.Param("g", 1), kp.Vj("b", 1)
    expr = (x.softdtw_sqdist(y, g) * b).sum_over("j")
    rng = np.random.default_rng(9)
    arrays = {"x": rng.standard_normal((90, 6)), "y": rng.standard_normal((70, 5)),
              "g": np.array([0.5]), "b": rng.standard_normal((70, 1))}
    run_case(name, dev, expr, arrays, dtype)
