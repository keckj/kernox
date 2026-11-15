"""Avenant v0.8 (MIN_ARGMIN, MAX_ARGMAX, LOGSUMEXPWEIGHT) sur chaque backend GPU.

Un backend qui répond ``Support(False)`` doit lever ``UnsupportedError``
avant tout lancement : le test est alors **sauté** avec la raison. Sinon :
MIN_ARGMIN / MAX_ARGMAX doivent coïncider bit à bit avec MIN et ARGMIN
(resp. MAX, ARGMAX) du même backend, eux-mêmes jugés par la conformance ;
LOGSUMEXPWEIGHT est comparé à la référence FP64 dense.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import kernox as kp
from kernox.errors import UnsupportedError

from ._support import synchronize

DTYPES = ("float32", "float64")


def _compile_or_skip(expr, dtype, backend, x):
    run = kp.compile(expr, dtype=dtype, backend=backend)
    try:
        return run(**x)
    except UnsupportedError as e:
        pytest.skip(f"{backend} : {e}")


@pytest.mark.parametrize("is_min", (True, False))
@pytest.mark.parametrize("dtype", DTYPES)
def test_min_argmin_matches_separate(backend, dtype, is_min):
    name, dev = backend
    rng = np.random.default_rng(0)
    td = getattr(torch, dtype)
    X = torch.tensor(rng.integers(-3, 4, size=(300, 2)), dtype=td, device=dev)
    Y = torch.tensor(rng.integers(-3, 4, size=(2000, 2)), dtype=td, device=dev)
    Y[7, 0] = float("nan")
    x, y = kp.Vi("x", 2), kp.Vj("y", 2)
    f = x * y
    feed = {"x": X, "y": Y}
    v, i = _compile_or_skip((f.min_argmin if is_min else f.max_argmax)(over="j"), dtype, name, feed)
    synchronize(dev)
    val = kp.compile((f.min if is_min else f.max)(over="j"), dtype=dtype, backend=name)(**feed)
    idx = kp.compile((f.argmin if is_min else f.argmax)(over="j"), dtype=dtype, backend=name)(**feed)
    assert v.dtype == td and i.dtype == torch.int64 and v.device == X.device
    assert torch.equal(v, val) and torch.equal(i, idx)


@pytest.mark.parametrize("dtype", DTYPES)
def test_logsumexpweight(backend, dtype):
    name, dev = backend
    rng = np.random.default_rng(1)
    td = getattr(torch, dtype)
    X, Y, G = rng.standard_normal((200, 3)), rng.standard_normal((1500, 3)), rng.random((1500, 4)) + 0.1
    x, y, g = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("g", 4)
    expr = (-(x - y).sqnorm2() * 10).logsumexp(over="j", weight=g)
    feed = {k: torch.tensor(v, dtype=td, device=dev) for k, v in (("x", X), ("y", Y), ("g", G))}
    out = _compile_or_skip(expr, dtype, name, feed)
    synchronize(dev)
    Xr, Yr, Gr = (torch.tensor(v.astype(dtype), dtype=torch.float64) for v in (X, Y, G))
    F = -((Xr[:, None] - Yr[None]) ** 2).sum(-1) * 10
    ref = torch.logsumexp(F[..., None] + torch.log(Gr)[None], dim=1)
    tol = 1e-4 if dtype == "float32" else 1e-12
    torch.testing.assert_close(out.cpu().double(), ref, rtol=tol, atol=tol)
