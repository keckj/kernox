"""Autograd of SUM reductions (CPU oracle): FP64 gradcheck, dense Torch, rules."""

import pytest

torch = pytest.importorskip("torch")

import kernox as kp  # noqa: E402
from kernox import Param, Vi, Vj, where  # noqa: E402
from kernox.autograd import gradient_plans  # noqa: E402
from kernox.errors import UnsupportedError  # noqa: E402

F64 = torch.float64


def _t(*shape, dtype=F64, seed=0, grad=True):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, generator=g).requires_grad_(grad)


def _gauss():
    x, y, w, s = Vi("x", 3), Vj("y", 3), Vj("w", 2), Param("s")
    d2 = ((x - y) ** 2).sum()
    return (kp.exp(-d2 * s[0]) * w).sum_over("j")


def _gauss_dense(X, Y, W, S):
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return torch.exp(-d2 * S[0]) @ W


def test_gradcheck_order1_and_2():
    run = kp.compile(_gauss(), dtype="float64", backend="oracle")
    args = (_t(5, 3, seed=1), _t(7, 3, seed=2), _t(7, 2, seed=3), torch.tensor([0.7], dtype=F64, requires_grad=True))
    f = lambda X, Y, W, S: run(x=X, y=Y, w=W, s=S)  # noqa: E731
    assert torch.autograd.gradcheck(f, args)
    assert torch.autograd.gradgradcheck(f, args)


def test_matches_dense_torch():
    run = kp.compile(_gauss(), dtype="float64", backend="oracle")
    X, Y, W = _t(6, 3, seed=4), _t(9, 3, seed=5), _t(9, 2, seed=6)
    S = torch.tensor([0.3], dtype=F64, requires_grad=True)
    G = _t(6, 2, seed=7, grad=False)
    got = torch.autograd.grad((run(x=X, y=Y, w=W, s=S) * G).sum(), (X, Y, W, S))
    want = torch.autograd.grad((_gauss_dense(X, Y, W, S) * G).sum(), (X, Y, W, S))
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)


def test_over_i_and_partial_requires_grad():
    x, y = Vi("x", 2), Vj("y", 2)
    red = (kp.sin(x - y) * kp.cos(y)).sum().sum_over("i")  # output indexed by j
    run = kp.compile(red, dtype="float64", backend="oracle")
    X, Y = _t(4, 2, seed=8), _t(5, 2, seed=9, grad=False)
    out = run(x=X, y=Y)
    assert out.shape == (5, 1)
    (gx,) = torch.autograd.grad(out.sum(), (X,))
    Xd = X.detach().clone().requires_grad_(True)
    dense = (torch.sin(Xd[:, None, :] - Y[None]) * torch.cos(Y[None])).sum(-1).sum(0)
    (want,) = torch.autograd.grad(dense.sum(), (Xd,))
    torch.testing.assert_close(gx, want)
    assert torch.autograd.gradcheck(lambda X: run(x=X, y=Y), (X,))


def test_where_gradient_of_active_branch():
    x, y = Vi("x", 1), Vj("y", 1)
    d = x[0] - y[0]
    red = where(d > 0, kp.sqrt(d), d * d).sum_over("j")  # sqrt(d < 0) never selected
    run = kp.compile(red, dtype="float64", backend="oracle")
    X = torch.tensor([[0.5], [-1.0], [2.0]], dtype=F64, requires_grad=True)
    Y = torch.tensor([[0.0], [1.0], [3.0]], dtype=F64, requires_grad=True)
    gx, gy = torch.autograd.grad(run(x=X, y=Y).sum(), (X, Y))
    assert torch.isfinite(gx).all() and torch.isfinite(gy).all()
    Xd, Yd = X.detach().clone().requires_grad_(), Y.detach().clone().requires_grad_()
    D = Xd - Yd.T
    dense = torch.where(D > 0, torch.sqrt(D.clamp_min(1e-300)), D * D).sum()
    want = torch.autograd.grad(dense, (Xd, Yd))
    torch.testing.assert_close(gx, want[0])
    torch.testing.assert_close(gy, want[1])


def test_max_tie_splits_gradient_and_abs_sign():
    x, y = Vi("x", 1), Vj("y", 1)
    run = kp.compile((kp.maximum(x[0], y[0]) + kp.abs(x[0] - y[0])).sum_over("j"), dtype="float64",
                     backend="oracle")
    X = torch.tensor([[1.0]], dtype=F64, requires_grad=True)
    Y = torch.tensor([[1.0], [0.0]], dtype=F64, requires_grad=True)
    gx, gy = torch.autograd.grad(run(x=X, y=Y).sum(), (X, Y))
    # tie: max -> 1/2 each, abs -> sign(0) = 0; y = 0: max -> x, abs -> +1 for x
    assert gx.item() == 0.5 + 1.0 + 1.0
    assert gy[:, 0].tolist() == [0.5, -1.0]


def test_float32_with_float64_accumulation():
    x, y = Vi("x", 3), Vj("y", 3)
    red = kp.exp(-((x - y) ** 2).sum()).sum_over("j")
    run = kp.compile(red, dtype="float32", acc_dtype="float64", backend="oracle")
    X, Y = _t(4, 3, dtype=torch.float32, seed=10), _t(6, 3, dtype=torch.float32, seed=11)
    out = run(x=X, y=Y)
    assert out.dtype == F64
    gx, gy = torch.autograd.grad(out.sum(), (X, Y))
    assert gx.dtype == torch.float32 and gy.dtype == torch.float32
    Xd, Yd = X.detach().double().requires_grad_(), Y.detach().double().requires_grad_()
    want = torch.autograd.grad(torch.exp(-((Xd[:, None] - Yd[None]) ** 2).sum(-1)).sum(), (Xd, Yd))
    torch.testing.assert_close(gx.double(), want[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gy.double(), want[1], rtol=1e-5, atol=1e-6)


def test_scalar_param_tensor_and_python_param():
    x, y, s = Vi("x", 1), Vj("y", 1), Param("s")
    run = kp.compile((s[0] * x[0] * y[0]).sum_over("j"), dtype="float64", backend="oracle")
    X, Y = _t(3, 1, seed=12), _t(4, 1, seed=13)
    S = torch.tensor(2.0, dtype=F64, requires_grad=True)  # 0-d
    (gs,) = torch.autograd.grad(run(x=X, y=Y, s=S).sum(), (S,))
    assert gs.shape == () and torch.allclose(gs, (X.sum() * Y.sum()).detach())
    gx, = torch.autograd.grad(run(x=X, y=Y, s=2.0).sum(), (X,))  # Python parameter: no gradient
    torch.testing.assert_close(gx, 2.0 * Y.sum().detach().expand(3, 1))


def test_empty_sizes():
    run = kp.compile(_gauss(), dtype="float64", backend="oracle")
    X, Y, W = _t(0, 3), _t(5, 3, seed=1), _t(5, 2, seed=2)
    gx, gy, gw = torch.autograd.grad(run(x=X, y=Y, w=W, s=0.5).sum(), (X, Y, W), allow_unused=True)
    assert gx.shape == (0, 3)
    assert gy is None or torch.count_nonzero(gy) == 0


@pytest.mark.parametrize("method", ["min", "max", "kmin"])
def test_select_outputs_are_differentiable_once(method):
    """MIN/MAX/KMIN: gradient flows to the selected term (details: test_reduction_gradients.py);
    no second derivative (once_differentiable)."""
    x, y = Vi("x", 1), Vj("y", 1)
    f = (x[0] - y[0]) ** 2
    red = f.kmin(2, over="j") if method == "kmin" else getattr(f, method)(over="j")
    run = kp.compile(red, dtype="float64", backend="oracle")
    X, Y = _t(3, 1), _t(4, 1, seed=1)
    out = run(x=X, y=Y)
    assert out.requires_grad
    (g,) = torch.autograd.grad(out.sum(), X, create_graph=True)
    assert torch.isfinite(g).all()
    with pytest.raises(RuntimeError):
        g.sum().backward()
    with torch.no_grad():
        assert not run(x=X, y=Y).requires_grad


def test_integer_outputs_have_no_gradient():
    x, y = Vi("x", 1), Vj("y", 1)
    f = (x[0] - y[0]) ** 2
    X, Y = _t(3, 1), _t(4, 1, seed=1)
    for red in (f.argmin(over="j"), (f < 1.0).sum_over("j")):
        out = kp.compile(red, dtype="float64", backend="oracle")(x=X, y=Y)
        assert out.dtype == torch.int64 and not out.requires_grad
    vals, idx = kp.compile(f.kmin_argkmin(2, over="j"), dtype="float64", backend="oracle")(x=X, y=Y)
    assert vals.requires_grad and not idx.requires_grad


def test_gradient_plans_are_fused_sums():
    plan = _gauss().to_plan("float64")
    gp = gradient_plans(plan, {"x", "y", "w", "s"})
    assert gp.i_plan.reduction.kind.value == "sum" and gp.j_plan.reduction.kind.value == "sum"
    assert [lay[0] for lay in gp.i_layout] == ["x", "s"] and [lay[0] for lay in gp.j_layout] == ["w", "y"]
    assert gp.i_plan.out_dim == 3 + 1 and gp.j_plan.out_dim == 2 + 3
    assert {a.name: a.role for a in gp.j_plan.args}["x"] == "j"
    with pytest.raises(UnsupportedError):
        gradient_plans(((Vi("x") - Vj("y")) ** 2).sum().min(over="j").to_plan("float64"), {"x"})


def _corpus_sums():
    from corpus import reductions

    return {k: r for k, r in reductions().items() if r.kind.value == "sum"}


@pytest.mark.parametrize("name", list(_corpus_sums()))
def test_corpus_gradient_plans_generate(name):
    from kernox.codegen.pair import DIALECTS, pair_function

    red = _corpus_sums()[name]
    for dtype in ("float32", "float64"):
        plan = red.to_plan(dtype)
        gp = gradient_plans(plan, {a.name for a in plan.args})
        for p in (gp.i_plan, gp.j_plan):
            if p is not None:
                for dialect in DIALECTS:
                    pair_function(p, dialect)
