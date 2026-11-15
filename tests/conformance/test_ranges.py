"""Ranges block-sparse (avenant v0.6 §2), sur l'oracle et sur chaque backend GPU.

Un backend qui ne prend pas encore les ranges en charge répond
``Support(False)`` : ses cas sont **sautés explicitement** (motif affiché).
L'oracle exécute toujours la suite. Critères : ``Ranges.dense`` ≡ appel sans
ranges (bit à bit hors SUM) ; masques par étiquettes contre une référence
dense masquée FP64 calculée sur l'hôte ; gradients contre Torch dense masqué.
"""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox import Ranges, Vi, Vj, ranges_from_labels
from kernox.errors import UnsupportedError

from ._support import BACKENDS, backend_status, synchronize


@pytest.fixture(params=("oracle",) + BACKENDS)
def target(request):
    name = request.param
    if name == "oracle":
        return name, "cpu"
    dev, reason = backend_status(name)
    if dev is None:
        pytest.skip(f"backend {name} absent : {reason}")
    return name, dev


def _call_or_skip(run, r, name, **kw):
    try:
        return run.with_ranges(r)(**kw)
    except UnsupportedError as exc:
        pytest.skip(f"{name} : {exc}")


def _setup(dev, dtype, density, m=257, n=1031, seed=0):
    g = torch.Generator().manual_seed(seed)
    li = torch.sort(torch.randint(0, 5, (m,), generator=g)).values
    lj = torch.sort(torch.randint(0, 7, (n,), generator=g)).values
    keep = torch.rand(5, 7, generator=g) < density
    X = torch.randint(-3, 4, (m, 3), generator=g).double()
    Y = torch.randint(-3, 4, (n, 3), generator=g).double()
    B = torch.randn(n, 2, dtype=torch.float64, generator=g)
    r = ranges_from_labels(li.to(dev), lj.to(dev), keep.to(dev))
    t = dict(x=X.to(dev, dtype), y=Y.to(dev, dtype), b=B.to(dev, dtype))
    return r, t, keep[li][:, lj], (X, Y, B)


def _formulas():
    x, y, b = Vi("x", 3), Vj("y", 3), Vj("b", 2)
    d2 = ((x - y) ** 2).sum()
    return {
        "sum": ((kp.exp(-0.1 * d2) * b).sum_over("j"), True),
        "count": ((d2 < 6.0).sum_over("j"), False),
        "argmin": (d2.argmin(over="j"), False),
        "kmin_argkmin": (d2.kmin_argkmin(5, over="j"), False),
    }


def _reference(kind, X, Y, B, mask):
    D = ((X[:, None] - Y[None]) ** 2).sum(-1)
    inf = float("inf")
    if kind == "sum":
        return (torch.exp(-0.1 * D) * mask) @ B
    if kind == "count":
        return ((D < 6.0) & mask).sum(1, keepdim=True)
    Dm = torch.where(mask, D, torch.full_like(D, inf))
    order = torch.argsort(Dm, dim=1, stable=True)
    if kind == "argmin":
        idx = order[:, :1]
        return torch.where(mask.any(1, keepdim=True), idx, torch.full_like(idx, -1))
    vals = Dm.gather(1, order[:, :5])
    return vals, torch.where(torch.isinf(vals), torch.full_like(order[:, :5], -1), order[:, :5])


def _pick(run, t):
    return {k: v for k, v in t.items() if k in run.input_names}


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("density", [1.0, 0.5, 0.05, 0.0])
@pytest.mark.parametrize("kind", list(_formulas()))
def test_ranges_match_masked_reference(target, kind, density, dtype):
    name, dev = target
    red, is_sum = _formulas()[kind]
    r, t, mask, host = _setup(dev, getattr(torch, dtype), density)
    run = kp.compile(red, dtype=dtype, backend=name)
    got = _call_or_skip(run, r, name, **_pick(run, t))
    if dev != "cpu":
        synchronize(dev)
    want = _reference(kind, *host, mask)
    for o, w in zip(got if isinstance(got, tuple) else (got,), want if isinstance(want, tuple) else (want,)):
        assert o.device == torch.device(dev)
        if is_sum:
            tol = dict(rtol=1e-4, atol=1e-4) if dtype == "float32" else dict(rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(o.double().cpu(), w, **tol)
        else:
            assert torch.equal(o.cpu(), w.to(o.dtype)), kind


@pytest.mark.parametrize("kind", list(_formulas()))
def test_dense_ranges_equal_plain(target, kind):
    name, dev = target
    red, is_sum = _formulas()[kind]
    _, t, _, _ = _setup(dev, torch.float32, 1.0, seed=1)
    run = kp.compile(red, dtype="float32", backend=name)
    m, n = t["x"].shape[0], t["y"].shape[0]
    got = _call_or_skip(run, Ranges.dense(m, n, device=dev), name, **_pick(run, t))
    plain = run(**_pick(run, t))
    for o, p in zip(got if isinstance(got, tuple) else (got,), plain if isinstance(plain, tuple) else (plain,)):
        if is_sum:
            torch.testing.assert_close(o, p, rtol=1e-5, atol=1e-5)
        else:
            assert torch.equal(o, p)


def test_ranges_gradients(target):
    name, dev = target
    red, _ = _formulas()["sum"]
    r, t, mask, host = _setup(dev, torch.float64, 0.5, m=61, n=83, seed=2)
    run = kp.compile(red, dtype="float64", backend=name)
    t = {k: v.clone().requires_grad_(True) for k, v in t.items()}
    out = _call_or_skip(run, r, name, **t)
    G = torch.randn(*out.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(3)).to(dev)
    got = torch.autograd.grad((out * G).sum(), list(t.values()))
    X, Y, B = (h.clone().requires_grad_(True) for h in host)
    D = ((X[:, None] - Y[None]) ** 2).sum(-1)
    want = torch.autograd.grad((((torch.exp(-0.1 * D) * mask) @ B) * G.cpu()).sum(), [X, Y, B])
    for k, g_, w_ in zip(t, got, want):
        torch.testing.assert_close(g_.cpu(), w_, rtol=1e-10, atol=1e-10, msg=k)
