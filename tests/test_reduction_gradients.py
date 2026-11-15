"""Gradients of reductions other than SUM (PyKeOps parity).

- LOGSUMEXP, LOGSUMEXPWEIGHT, SUMSOFTMAXWEIGHT: surrogate SUM plan, fused
  reductions (``autograd.surrogate``);
- MIN, MAX, KMIN and the values of MIN_ARGMIN, MAX_ARGMAX, KMIN_ARGKMIN: derivative
  at the selected term (``autograd.SelectGradient``).

Reference: the same function in dense Torch (small sizes), ``torch.autograd``.
"""

from __future__ import annotations

import pytest
import torch

import kernox as kp

F64 = torch.float64


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _inputs(m=7, n=11, seed=0):
    g = _gen(seed)
    return dict(x=torch.randn(m, 2, dtype=F64, generator=g), y=torch.randn(n, 2, dtype=F64, generator=g),
                b=torch.rand(n, 3, dtype=F64, generator=g) + 0.1, p=torch.tensor([0.7, -0.3], dtype=F64))


def _vars():
    return kp.Vi("x", 2), kp.Vj("y", 2), kp.Vj("b", 3), kp.Param("p", 2)


def _dense(X, Y, P):
    """f[i, j] (M, N, 2): exp(-|x - y|²)·p + sin(x·y)."""
    d2 = ((X[:, None] - Y[None]) ** 2).sum(-1, keepdim=True)
    return torch.exp(-d2) * P + torch.sin((X[:, None] * Y[None]).sum(-1, keepdim=True))


def _kp_f(x, y, p):
    return kp.exp(-(x - y).sqnorm2()) * p + kp.sin((x * y).sum())


def _check(red, dense_fn, t, names=("x", "y", "b", "p"), seed=5):
    leaves = {k: v.clone().requires_grad_(k in names) for k, v in t.items()}
    out = red(**{k: v for k, v in leaves.items() if k in red.plan_names})
    G = torch.randn(out.shape, dtype=F64, generator=_gen(seed))
    got = torch.autograd.grad((out * G).sum(), [leaves[k] for k in names if k in red.plan_names])
    ref_leaves = {k: v.clone().requires_grad_(k in names) for k, v in t.items()}
    ref = dense_fn(**ref_leaves)
    torch.testing.assert_close(out, ref.detach(), rtol=1e-12, atol=1e-12)
    want = torch.autograd.grad((ref * G).sum(), [ref_leaves[k] for k in names if k in red.plan_names])
    for k, a, w in zip([k for k in names if k in red.plan_names], got, want):
        torch.testing.assert_close(a, w, rtol=1e-10, atol=1e-12, msg=k)


class _Run:
    def __init__(self, expr):
        self.run = expr.compile("float64", backend="oracle")
        self.plan_names = {a.name for a in self.run.plan.args}

    def __call__(self, **kw):
        return self.run(**kw)


@pytest.mark.parametrize("over", ("j", "i"))
def test_logsumexp_gradient(over):
    x, y, b, p = _vars()
    t = _inputs()
    red = _Run(_kp_f(x, y, p).logsumexp(over=over))
    dim = 1 if over == "j" else 0
    _check(red, lambda x, y, b, p: torch.logsumexp(_dense(x, y, p), dim=dim), t)


def test_logsumexpweight_gradient():
    x, y, b, p = _vars()
    t = _inputs()
    red = _Run(_kp_f(x, y, p)[0].logsumexp(over="j", weight=b))
    _check(red, lambda x, y, b, p: torch.logsumexp(_dense(x, y, p)[..., :1] + torch.log(b)[None], dim=1), t)


def test_sumsoftmaxweight_gradient():
    x, y, b, p = _vars()
    t = _inputs()
    red = _Run(_kp_f(x, y, p)[1].sumsoftmaxweight(b * y[0], over="j"))
    _check(red, lambda x, y, b, p: (torch.softmax(_dense(x, y, p)[..., 1], dim=1)[..., None]
                                    * (b * y[:, :1])[None]).sum(1), t)


def test_logsumexp_gradient_empty_rows_are_zero():
    """Row without any contribution (ranges): L = −∞, gradient is zero, not NaN."""
    x, y, b, p = _vars()
    t = _inputs(m=6, n=8)
    li, lj = torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    keep = torch.tensor([[True, False], [False, False]])  # group i=1: no j
    r = kp.ranges_from_labels(li, lj, keep)
    run = _kp_f(x, y, p).logsumexp(over="j").compile("float64", backend="oracle").with_ranges(r)
    X = t["x"].clone().requires_grad_(True)
    out = run(x=X, y=t["y"], p=t["p"])
    assert torch.isinf(out[3:]).all()
    (g,) = torch.autograd.grad(out[:3].sum(), X)
    assert torch.isfinite(g).all() and (g[3:] == 0).all()


@pytest.mark.parametrize("kind", ("min", "max", "min_argmin", "max_argmax"))
@pytest.mark.parametrize("over", ("j", "i"))
def test_min_max_gradient(kind, over):
    x, y, b, p = _vars()
    t = _inputs()
    f = _kp_f(x, y, p)
    red = _Run(getattr(f, kind)(over=over))
    dim = 1 if over == "j" else 0
    op = torch.amin if kind.startswith("min") else torch.amax

    def dense(x, y, b, p):
        D = _dense(x, y, p)
        idx = (D.argmin if kind.startswith("min") else D.argmax)(dim=dim, keepdim=True)
        return torch.gather(D, dim, idx).squeeze(dim)  # gradient only at the selected index

    if kind in ("min", "max"):
        _check(red, dense, t)
    else:
        run = red.run
        leaves = {k: v.clone().requires_grad_(True) for k, v in t.items() if k != "b"}
        v, i = run(**leaves)
        assert not i.requires_grad
        G = torch.randn(v.shape, dtype=F64, generator=_gen(9))
        got = torch.autograd.grad((v * G).sum(), list(leaves.values()))
        ref_leaves = {k: v_.clone().requires_grad_(True) for k, v_ in t.items() if k != "b"}
        ref = dense(b=None, **ref_leaves)
        torch.testing.assert_close(v, ref.detach(), rtol=0, atol=0)
        assert torch.equal(v, op(_dense(t["x"], t["y"], t["p"]), dim=dim))
        want = torch.autograd.grad((ref * G).sum(), list(ref_leaves.values()))
        for a, w in zip(got, want):
            torch.testing.assert_close(a, w, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kind", ("kmin", "kmin_argkmin"))
def test_kmin_gradient(kind):
    x, y = kp.Vi("x", 2), kp.Vj("y", 2)
    t = _inputs()
    run = getattr((x - y).sqnorm2(), kind)(3, over="j").compile("float64", backend="oracle")
    X, Y = t["x"].clone().requires_grad_(True), t["y"].clone().requires_grad_(True)
    out = run(x=X, y=Y)
    v = out[0] if isinstance(out, tuple) else out
    G = torch.randn(v.shape, dtype=F64, generator=_gen(3))
    got = torch.autograd.grad((v * G).sum(), (X, Y))
    Xr, Yr = t["x"].clone().requires_grad_(True), t["y"].clone().requires_grad_(True)
    D = ((Xr[:, None] - Yr[None]) ** 2).sum(-1)
    ref = torch.topk(D, 3, dim=1, largest=False).values
    want = torch.autograd.grad((ref * G).sum(), (Xr, Yr))
    torch.testing.assert_close(v, ref.detach(), rtol=1e-14, atol=1e-14)
    for a, w in zip(got, want):
        torch.testing.assert_close(a, w, rtol=1e-10, atol=1e-12)


def test_min_gradient_batch_and_no_candidate():
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    run = ((x - y) ** 2).min(over="j").compile("float64", backend="oracle")
    g = _gen(4)
    X = torch.randn(2, 5, 1, dtype=F64, generator=g).requires_grad_(True)
    Y = torch.randn(1, 6, 1, dtype=F64, generator=g).requires_grad_(True)  # broadcast over the batch
    out = run(x=X, y=Y)
    gx, gy = torch.autograd.grad(out.sum(), (X, Y))
    Xr, Yr = X.detach().clone().requires_grad_(True), Y.detach().clone().requires_grad_(True)
    ref = ((Xr - Yr.transpose(-1, -2)) ** 2).amin(-1, keepdim=True)
    wx, wy = torch.autograd.grad(ref.sum(), (Xr, Yr))
    torch.testing.assert_close(gx, wx, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gy, wy, rtol=1e-12, atol=1e-12)
    # row without candidates (NaN everywhere): index −1, zero gradient
    Yn = torch.full((4, 1), float("nan"), dtype=F64)
    X1 = torch.randn(3, 1, dtype=F64).requires_grad_(True)
    (g1,) = torch.autograd.grad(run(x=X1, y=Yn).sum(), X1)
    assert (g1 == 0).all()


def test_min_gradient_with_ranges():
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    run = ((x - y) ** 2).min(over="j").compile("float64", backend="oracle")
    lx, ly = torch.repeat_interleave(torch.tensor([0, 1]), 5), torch.repeat_interleave(torch.tensor([0, 1]), 4)
    r = kp.ranges_from_labels(lx, ly, torch.eye(2, dtype=torch.bool))
    g = _gen(6)
    X = torch.randn(10, 1, dtype=F64, generator=g).requires_grad_(True)
    Y = torch.randn(8, 1, dtype=F64, generator=g).requires_grad_(True)
    out = run.with_ranges(r)(x=X, y=Y)
    gx, gy = torch.autograd.grad(out.sum(), (X, Y))
    Xr, Yr = X.detach().clone().requires_grad_(True), Y.detach().clone().requires_grad_(True)
    D = torch.where(lx[:, None] == ly[None], (Xr - Yr.T) ** 2, torch.full((), float("inf"), dtype=F64))
    wx, wy = torch.autograd.grad(D.amin(1).sum(), (Xr, Yr))
    torch.testing.assert_close(gx, wx, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gy, wy, rtol=1e-12, atol=1e-12)
