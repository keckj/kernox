"""Targeted port of PyKeOps tests (getkeops/keops @ ef72bc907df78cd18bff866d3dc7cb6375f11a4d,
MIT license; detailed provenance in THIRD_PARTY.md).

Taken over: formulas, dimensions and data distributions of the forward tests
(no autograd, no batch, no LogSumExp or block-sparse, outside the v0 scope).
The KeOps tolerances (fixed ``atol``) are replaced by the suite's shared
judgment (per-term error bound, see ``_support.py``), which is stricter.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ._support import run_case

import kernox as kp

DTYPES = ("float32", "float64")


def _rand(rng, *shape):
    return rng.random(shape)  # torch.rand: uniform [0, 1)


@pytest.mark.parametrize("dtype", DTYPES)
def test_generic_syntax(backend, dtype):
    # pykeops/test/test_torch.py::test_generic_syntax_float / _double
    # formula "Square(p-a)*Exp(x+y)", aliases p=Pm(1), a=Vj(1), x=Vi(3), y=Vj(3), axis=1
    name, dev = backend
    p, a, x, y = kp.Param("p", 1), kp.Vj("a", 1), kp.Vi("x", 3), kp.Vj("y", 3)
    expr = (kp.square(p - a) * kp.exp(x + y)).sum_over("j")
    rng = np.random.default_rng(42)
    arrays = {"p": np.array([0.4]), "a": _rand(rng, 6, 1), "x": _rand(rng, 10, 3), "y": _rand(rng, 6, 3)}
    run_case(name, dev, expr, arrays, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_chunks_large_dimension(backend, dtype):
    # pykeops/test/test_chunks.py: M, N, D, DV = 200, 300, 300, 1; exp(-|x-y|^2) @ b
    name, dev = backend
    M, N, D, DV = 200, 300, 300, 1
    x, y, b = kp.Vi("x", D), kp.Vj("y", D), kp.Vj("b", DV)
    expr = (kp.exp(-((x - y) ** 2).sum()) * b).sum_over("j")
    rng = np.random.default_rng(0)
    arrays = {"x": _rand(rng, M, D) / math.sqrt(D), "y": _rand(rng, N, D) / math.sqrt(D),
              "b": rng.standard_normal((N, DV))}
    run_case(name, dev, expr, arrays, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_finalchunks_large_output(backend, dtype):
    # pykeops/test/test_finalchunks.py: M, N, D, DV = 200, 300, 3, 300 (output of dimension 300)
    name, dev = backend
    M, N, D, DV = 200, 300, 3, 300
    x, y, b = kp.Vi("x", D), kp.Vj("y", D), kp.Vj("b", DV)
    expr = (kp.exp(-((x - y) ** 2).sum()) * b).sum_over("j")
    rng = np.random.default_rng(0)
    arrays = {"x": _rand(rng, M, D) / math.sqrt(D), "y": _rand(rng, N, D) / math.sqrt(D),
              "b": rng.standard_normal((N, DV))}
    run_case(name, dev, expr, arrays, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_clamp(backend, dtype):
    # pykeops/test/test_lazytensor_clamp.py (forward): M = N = 1000, D = 3, a = -1.23, b = 1.54
    name, dev = backend
    x, y = kp.Vi("x", 3), kp.Vj("y", 3)
    dxy = kp.clamp(x * y, -1.23, 1.54).sum()
    expr = kp.exp(-(dxy ** 2)).sum_over("j")
    rng = np.random.default_rng(0)
    run_case(name, dev, expr, {"x": rng.standard_normal((1000, 3)), "y": rng.standard_normal((1000, 3))}, dtype)


@pytest.mark.parametrize("over", ("i", "j"))
@pytest.mark.parametrize("dtype", DTYPES)
def test_min_argmin(backend, dtype, over):
    # pykeops/test/test_torch.py::test_LazyTensor_min (no batch):
    # D_ij = log(1 + (relu(l_i x_i + y_j) s)^2).sum; K_ij = cos(D^1.5 + 1) sin(D (3.2 + s))
    name, dev = backend
    x, l, y, s = kp.Vi("x", 3), kp.Vi("l", 1), kp.Vj("y", 3), kp.Param("s", 1)
    d = kp.log(1.0 + (kp.maximum(l * x + y, 0.0) * s) ** 2).sum()
    k = kp.cos(d ** 1.5 + 1.0) * kp.sin(d * (3.2 + s))
    rng = np.random.default_rng(42)
    arrays = {"x": _rand(rng, 10, 3), "l": _rand(rng, 10, 1), "y": _rand(rng, 6, 3), "s": _rand(rng, 1) + 0.1}
    run_case(name, dev, k.min(over=over), arrays, dtype)
    run_case(name, dev, k.argmin(over=over), arrays, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_argmin_chunked(backend, dtype):
    # pykeops/test/test_torch.py::test_LazyTensor_min_chunked: D = 990, min_argmin over i (dim=0)
    name, dev = backend
    x, y = kp.Vi("x", 990), kp.Vj("y", 990)
    kxy = kp.exp(-((x + y) ** 2)).sum()
    rng = np.random.default_rng(42)
    arrays = {"x": _rand(rng, 10, 990), "y": _rand(rng, 6, 990)}
    run_case(name, dev, kxy.min(over="i"), arrays, dtype)
    run_case(name, dev, kxy.argmin(over="i"), arrays, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_argkmin(backend, dtype):
    # pykeops/test/test_numpy.py::test_argkmin: SqDist(x, y), ArgKMin K = 3, M = 10, N = 6, D = 3
    name, dev = backend
    x, y = kp.Vi("x", 3), kp.Vj("y", 3)
    rng = np.random.default_rng(42)
    arrays = {"x": _rand(rng, 10, 3), "y": _rand(rng, 6, 3)}
    run_case(name, dev, ((x - y) ** 2).sum().argkmin(3, over="j"), arrays, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gaussian_kernel_sum(backend, dtype):
    # pykeops/test/test_torch.py::test_LazyTensor_sum (no batch, forward):
    # D_ij = (0.5 (l_i x_i - y_j)^2 / s).sum; a_i = sum_j exp(-D_ij)
    name, dev = backend
    x, l, y, s = kp.Vi("x", 3), kp.Vi("l", 1), kp.Vj("y", 3), kp.Param("s", 3)
    expr = kp.exp(-(0.5 * (l * x - y) ** 2 / s).sum()).sum_over("j")
    rng = np.random.default_rng(42)
    arrays = {"x": _rand(rng, 10, 3), "l": _rand(rng, 10, 1), "y": _rand(rng, 6, 3), "s": _rand(rng, 3) + 0.1}
    run_case(name, dev, expr, arrays, dtype)
