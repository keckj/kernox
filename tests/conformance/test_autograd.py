"""Autograd of SUM reductions on each available GPU backend.

FP64 ``gradcheck`` (first order, and second order on one case) and comparison
with dense Torch (FP32 and FP64); the gradients are run by the backend itself
(fused gradient reductions, see ``kernox.autograd``).
"""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox import Param, Vi, Vj, where

TOL = {torch.float32: dict(rtol=2e-4, atol=2e-5), torch.float64: dict(rtol=1e-10, atol=1e-11)}


def _cases():
    x, y, w, b, s = Vi("x", 3), Vj("y", 3), Vj("w", 2), Vj("b", 16), Param("s")
    d2 = ((x - y) ** 2).sum()

    def dd2(X, Y):
        return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)

    return {
        "gauss": ((kp.exp(-d2 * s[0]) * w).sum_over("j"), ("x", "y", "w", "s"),
                  lambda t: torch.exp(-dd2(t["x"], t["y"]) * t["s"][0]) @ t["w"]),
        "cauchy_over_i": ((1.0 / (1.0 + d2) * Vi("v", 2)).sum_over("i"), ("x", "y", "v"),
                          lambda t: (1.0 / (1.0 + dd2(t["x"], t["y"]))).T @ t["v"]),
        "laplace": ((kp.exp(-kp.sqrt(d2 + 0.1)) * w).sum_over("j"), ("x", "y", "w"),
                    lambda t: torch.exp(-torch.sqrt(dd2(t["x"], t["y"]) + 0.1)) @ t["w"]),
        "trunc_vec": ((where(d2 <= 2.0, kp.exp(-d2), 0.0) * b).sum_over("j"), ("x", "y", "b"),
                      lambda t: torch.where(dd2(t["x"], t["y"]) <= 2.0, torch.exp(-dd2(t["x"], t["y"])),
                                            torch.zeros(())) @ t["b"]),
    }


def _inputs(names, dtype, device, m=37, n=53, seed=0):
    g = torch.Generator().manual_seed(seed)
    dims = {"x": (m, 3), "y": (n, 3), "w": (n, 2), "b": (n, 16), "v": (m, 2), "s": (1,)}
    out = {}
    for k in names:
        v = torch.randn(*dims[k], dtype=torch.float64, generator=g)
        if k == "s":
            v = v.abs() + 0.3
        out[k] = v.to(dtype=dtype, device=device).requires_grad_(True)
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", list(_cases()))
def test_gradients_match_dense_torch(backend, name, dtype):
    bname, dev = backend
    red, names, dense = _cases()[name]
    run = kp.compile(red, dtype=str(dtype).removeprefix("torch."), backend=bname)
    t = _inputs(names, dtype, dev)
    G = torch.randn(*run(**t).shape, dtype=dtype, generator=torch.Generator().manual_seed(1)).to(dev)
    got = torch.autograd.grad((run(**t) * G).sum(), [t[k] for k in names])
    r = {k: v.detach().double().cpu().requires_grad_(True) for k, v in t.items()}
    want = torch.autograd.grad((dense(r) * G.double().cpu()).sum(), [r[k] for k in names])
    for k, g_, w_ in zip(names, got, want):
        assert g_.dtype == dtype and g_.device == t[k].device, k
        scale = w_.abs().max().item() or 1.0
        torch.testing.assert_close(g_.double().cpu() / scale, w_ / scale, **TOL[dtype], msg=f"{name}:{k}")


@pytest.mark.parametrize("name", ["gauss", "cauchy_over_i"])
def test_gradcheck_float64(backend, name):
    bname, dev = backend
    red, names, _ = _cases()[name]
    run = kp.compile(red, dtype="float64", backend=bname)
    t = _inputs(names, torch.float64, dev, m=6, n=7)
    f = lambda *a: run(**dict(zip(names, a)))  # noqa: E731
    assert torch.autograd.gradcheck(f, tuple(t[k] for k in names))


def test_gradgradcheck_float64(backend):
    bname, dev = backend
    red, names, _ = _cases()["gauss"]
    run = kp.compile(red, dtype="float64", backend=bname)
    t = _inputs(names, torch.float64, dev, m=4, n=5)
    f = lambda *a: run(**dict(zip(names, a)))  # noqa: E731
    assert torch.autograd.gradgradcheck(f, tuple(t[k] for k in names))
