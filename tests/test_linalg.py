"""Opérateur linéaire et ``solve`` (parité PyKeOps : ``matvec``, ``rmatvec``, ``T``,
``LazyTensor.solve`` / ``KernelSolve``), sur l'oracle, contre l'algèbre dense de Torch."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import kernox as kp
from kernox.errors import UnsupportedError, ValidationError

F64 = torch.float64


def _gauss(dim_b=2):
    x, y, b, s = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", dim_b), kp.Param("s")
    k = kp.exp(-(x - y).sqnorm2() / (2.0 * s**2))
    return kp.compile((k * b).sum_over("j"), dtype="float64", backend="oracle")


def _dense(x, y, s):
    return torch.exp(-((x[:, None] - y[None]) ** 2).sum(-1) / (2 * s**2))


def _data(m=40, n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, 3, dtype=F64, generator=g), torch.randn(n, 3, dtype=F64, generator=g)


def test_matvec_rmatvec_against_dense():
    x, y = _data(30, 45)
    op = _gauss().linear_operator("b", x=x, y=y, s=0.9)
    k = _dense(x, y, 0.9)
    assert op.shape == (60, 90) and op.T.shape == (90, 60) and op.dtype == np.float64
    v, u = torch.randn(45, 2, dtype=F64), torch.randn(30, 2, dtype=F64)
    torch.testing.assert_close(op @ v, k @ v)
    torch.testing.assert_close(op.matvec(v), k @ v)
    torch.testing.assert_close(op.rmatvec(u), k.T @ u)
    torch.testing.assert_close(op.T @ u, k.T @ u)
    torch.testing.assert_close(op.T.rmatvec(v), k @ v)
    torch.testing.assert_close(op.t().T @ v, k @ v)
    # formes SciPy : vecteur aplati, matrice de colonnes aplaties, NumPy
    torch.testing.assert_close(op.matvec(v.reshape(-1)), (k @ v).reshape(-1))
    cols = torch.randn(90, 3, dtype=F64)
    want = torch.stack([(k @ cols[:, c].reshape(45, 2)).reshape(-1) for c in range(3)], 1)
    torch.testing.assert_close(op.matmat(cols), want)
    got = op.matvec(v.reshape(-1).numpy())
    assert isinstance(got, np.ndarray)
    np.testing.assert_allclose(got, (k @ v).reshape(-1).numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(op.rmatmat(u.reshape(-1, 1).numpy())[:, 0], (k.T @ u).reshape(-1).numpy(),
                               rtol=1e-12, atol=1e-12)


def test_non_square_matrix_valued_operator():
    """Sortie Do=3 depuis Dv=2 : J(x, y) non diagonal, Kᵀ par le produit vecteur-jacobien."""
    x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 2)
    k = kp.exp(-(x - y).sqnorm2())
    f = kp.cat(k * b[0], (x * y).sum() * b[1] + k * b[0], 2.0 * b[1])
    run = kp.compile(f.sum_over("j"), dtype="float64", backend="oracle")
    xt, yt = _data(20, 25, seed=3)
    op = run.linear_operator("b", x=xt, y=yt)
    kd = torch.exp(-((xt[:, None] - yt[None]) ** 2).sum(-1))
    dot = xt @ yt.T
    # matrice dense (M·3, N·2), ordre row-major (i, composante)
    a = torch.zeros(20, 3, 25, 2, dtype=F64)
    a[:, 0, :, 0] = kd
    a[:, 1, :, 0] = kd
    a[:, 1, :, 1] = dot
    a[:, 2, :, 1] = 2.0
    a = a.reshape(60, 50)
    v, u = torch.randn(25, 2, dtype=F64), torch.randn(20, 3, dtype=F64)
    assert op.shape == (60, 50)
    torch.testing.assert_close(op @ v, (a @ v.reshape(-1)).reshape(20, 3))
    torch.testing.assert_close(op.rmatvec(u), (a.T @ u.reshape(-1)).reshape(25, 2))
    with pytest.raises(ValidationError, match="carré"):
        op.solve(torch.randn(20, 3, dtype=F64))


def test_scipy_linear_operator():
    sla = pytest.importorskip("scipy.sparse.linalg")
    x, _ = _data(60, 60, seed=1)
    op = _gauss(1).linear_operator("b", x=x, y=x, s=0.7)
    vals = sla.eigsh(sla.aslinearoperator(op), k=4, return_eigenvectors=False)
    ref = np.linalg.eigvalsh(_dense(x, x, 0.7).numpy())[-4:]
    np.testing.assert_allclose(np.sort(vals), ref, rtol=1e-8)
    rhs = np.random.default_rng(0).standard_normal(60)
    sol, info = sla.cg(sla.aslinearoperator(op) + 0.5 * sla.aslinearoperator(np.eye(60)), rhs, rtol=1e-12)
    assert info == 0
    np.testing.assert_allclose(sol, np.linalg.solve(_dense(x, x, 0.7).numpy() + 0.5 * np.eye(60), rhs),
                               rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("alpha", [1e-10, 0.1, 2.0])
def test_solve_against_dense(alpha):
    x, _ = _data(50, 50, seed=2)
    rhs = torch.randn(50, 2, dtype=F64, generator=torch.Generator().manual_seed(4))
    run = _gauss()
    a, info = run.linear_operator("b", x=x, y=x, s=0.5).solve(rhs, alpha=alpha, eps=1e-12, return_info=True)
    assert info["converged"] and info["niter"] >= 1
    ref = torch.linalg.solve(_dense(x, x, 0.5) + alpha * torch.eye(50, dtype=F64), rhs)
    torch.testing.assert_close(a, ref, rtol=1e-8, atol=1e-8)
    # même résultat par Reduction.solve et le critère d'arrêt de PyKeOps (eps)
    b = run.solve("b", rhs, alpha=alpha, eps=1e-12, x=x, y=x, s=0.5)
    torch.testing.assert_close(b, a, rtol=0, atol=0)


def test_solve_default_tolerance_and_x0():
    x, _ = _data(40, 40, seed=5)
    rhs = torch.randn(40, 1, dtype=F64, generator=torch.Generator().manual_seed(6))
    op = _gauss(1).linear_operator("b", x=x, y=x, s=0.6)
    a, info = op.solve(rhs, alpha=0.3, return_info=True)
    # arrêt de PyKeOps : ‖r‖ < max(eps‖b‖, eps), eps = 1e-6
    assert info["atol"] == pytest.approx(max(1e-6, 1e-6 * float(rhs.norm())))
    res = op @ a + 0.3 * a - rhs
    assert float(res.norm()) < info["atol"]
    # démarrage à la solution : zéro itération
    _, info0 = op.solve(rhs, alpha=0.3, x0=a, return_info=True)
    assert info0["niter"] <= 1


def test_solve_maxiter_warns():
    x, _ = _data(40, 40, seed=7)
    op = _gauss(1).linear_operator("b", x=x, y=x, s=0.6)
    with pytest.warns(RuntimeWarning, match="sans convergence"):
        _, info = op.solve(torch.randn(40, 1, dtype=F64), alpha=1e-10, eps=1e-14, maxiter=2, return_info=True)
    assert not info["converged"] and info["niter"] == 2


def _dense_solve(x, rhs, s, alpha):
    return torch.linalg.solve(_dense(x, x, s) + alpha * torch.eye(x.shape[0], dtype=F64), rhs)


def test_solve_gradients_against_dense():
    x0, _ = _data(30, 30, seed=8)
    rhs0 = torch.randn(30, 2, dtype=F64, generator=torch.Generator().manual_seed(9))
    w = torch.randn(30, 2, dtype=F64, generator=torch.Generator().manual_seed(10))
    run = _gauss()

    def leaves():
        return (x0.clone().requires_grad_(), rhs0.clone().requires_grad_(),
                torch.tensor(0.8, dtype=F64, requires_grad=True))

    x, rhs, s = leaves()
    a = run.solve("b", rhs, alpha=0.2, eps=1e-13, x=x, y=x, s=s)
    got = torch.autograd.grad((a * w).sum(), [x, rhs, s])
    xr, rr, sr = leaves()
    want = torch.autograd.grad((_dense_solve(xr, rr, sr, 0.2) * w).sum(), [xr, rr, sr])
    for g, r in zip(got, want):
        torch.testing.assert_close(g, r, rtol=1e-8, atol=1e-9)


def test_solve_second_order():
    x0, _ = _data(20, 20, seed=11)
    rhs0 = torch.randn(20, 1, dtype=F64, generator=torch.Generator().manual_seed(12))
    run = _gauss(1)
    s = torch.tensor(0.9, dtype=F64, requires_grad=True)
    sr = torch.tensor(0.9, dtype=F64, requires_grad=True)
    x = x0.clone().requires_grad_()
    xr = x0.clone().requires_grad_()
    a = run.solve("b", rhs0, alpha=0.5, eps=1e-14, x=x, y=x, s=s)
    ar = _dense_solve(xr, rhs0, sr, 0.5)
    (gs,) = torch.autograd.grad(a.square().sum(), s, create_graph=True)
    (gsr,) = torch.autograd.grad(ar.square().sum(), sr, create_graph=True)
    torch.testing.assert_close(gs, gsr, rtol=1e-8, atol=1e-10)
    got = torch.autograd.grad(gs, [x, s])
    want = torch.autograd.grad(gsr, [xr, sr])
    for g, r in zip(got, want):
        torch.testing.assert_close(g, r, rtol=1e-7, atol=1e-9)


def test_operator_products_are_differentiable():
    x0, y0 = _data(15, 18, seed=13)
    v0, u0 = torch.randn(18, 2, dtype=F64), torch.randn(15, 2, dtype=F64)
    run = _gauss()
    x, v, u = x0.clone().requires_grad_(), v0.clone().requires_grad_(), u0.clone().requires_grad_()
    xr, vr, ur = x0.clone().requires_grad_(), v0.clone().requires_grad_(), u0.clone().requires_grad_()
    op = run.linear_operator("b", x=x, y=y0, s=0.7)
    k = _dense(xr, y0, 0.7)
    got = torch.autograd.grad((op @ v).sin().sum() + op.rmatvec(u).cos().sum(), [x, v, u])
    want = torch.autograd.grad((k @ vr).sin().sum() + (k.T @ ur).cos().sum(), [xr, vr, ur])
    for g, r in zip(got, want):
        torch.testing.assert_close(g, r, rtol=1e-10, atol=1e-11)


def test_conjugate_gradient_function():
    g = torch.Generator().manual_seed(14)
    a = torch.randn(25, 25, dtype=F64, generator=g)
    spd = a @ a.T + 25 * torch.eye(25, dtype=F64)
    b = torch.randn(25, 3, dtype=F64, generator=g)
    x, info = kp.conjugate_gradient(lambda v: spd @ v, b, eps=1e-13)
    assert info["converged"] and info["maxiter"] == 250
    torch.testing.assert_close(x, torch.linalg.solve(spd, b), rtol=1e-10, atol=1e-10)
    x, info = kp.conjugate_gradient(lambda v: spd @ v, torch.zeros_like(b))
    assert info["niter"] == 0 and not x.any()


def test_linearity_is_checked():
    x, y, b, s = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 1), kp.Param("s")
    k = kp.exp(-(x - y).sqnorm2())
    xt, yt = _data(10, 12)
    bt = torch.randn(12, 1, dtype=F64)

    def op(expr, var="b", **kw):
        run = kp.compile(expr, dtype="float64", backend="oracle")
        args = {"x": xt, "y": yt, "b": bt, "s": 0.5}
        args = {k_: v for k_, v in args.items() if k_ in run.input_names and k_ != var}
        return run.linear_operator(var, **{**args, **kw})

    with pytest.raises(ValidationError, match="non linéaire"):
        op((k * b * b).sum_over("j"))
    with pytest.raises(ValidationError, match="non linéaire"):
        op((k * kp.sin(b)).sum_over("j"))
    # zéro non reconnu symboliquement (produit d'exponentielles contre exponentielle
    # de la somme) : refus prudent, jamais d'hypothèse de linéarité
    with pytest.raises(ValidationError, match="non démontrée"):
        op((k * kp.exp(b) - k).sum_over("j"))
    with pytest.raises(ValidationError, match="en b=0"):
        op((k * b + k).sum_over("j"))
    with pytest.raises(ValidationError, match="SUM"):
        op((k * b).max(over="j"))
    with pytest.raises(ValidationError, match="variable j"):
        op((k * x[0]).sum_over("j"), var="x", b=bt)
    with pytest.raises(ValidationError, match="n'est pas une entrée"):
        op((k * b).sum_over("j"), var="z")
    with pytest.raises(ValidationError, match="ne pas la fournir"):
        kp.compile((k * b).sum_over("j"), dtype="float64", backend="oracle").linear_operator(
            "b", x=xt, y=yt, b=bt)
    # where linéaire par morceaux : accepté (nul en b = 0, dérivée sans b)
    op(kp.where(x[0] < y[0], k * b, -b).sum_over("j"))


def test_operand_validation():
    x, _ = _data(10, 10)
    run = _gauss(1)
    op = run.linear_operator("b", x=x, y=x, s=0.5)
    with pytest.raises(ValidationError, match="dtype"):
        op @ torch.randn(10, 1, dtype=torch.float32)
    with pytest.raises(ValidationError, match="forme"):
        op @ torch.randn(9, 1, dtype=F64)
    with pytest.raises(ValidationError, match="alpha"):
        op.solve(torch.randn(10, 1, dtype=F64), alpha=-1.0)
    with pytest.raises(ValidationError, match="forme"):
        op.solve(torch.randn(10, 2, dtype=F64))
    with pytest.raises(UnsupportedError, match="transposé"):
        op.T.solve(torch.randn(10, 1, dtype=F64))
    with pytest.raises(ValidationError, match="deux fois"):
        run.solve("b", torch.randn(10, 1, dtype=F64), inputs={"x": x}, x=x, y=x, s=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = run.solve("b", torch.randn(10, 1, dtype=F64), alpha=1.0, inputs={"x": x, "y": x, "s": 0.5})
    assert out.shape == (10, 1)


def test_ranged_reduction_operator():
    x, _ = _data(24, 24, seed=15)
    labels = torch.arange(24) // 8
    run = _gauss(1)
    ranges = kp.ranges_from_labels(labels, labels, torch.eye(3, dtype=torch.bool))
    ranged = run.with_ranges(ranges)
    mask = (labels[:, None] == labels[None]).to(F64)
    k = _dense(x, x, 0.6) * mask
    op = ranged.linear_operator("b", x=x, y=x, s=0.6)
    v = torch.randn(24, 1, dtype=F64)
    torch.testing.assert_close(op @ v, k @ v)
    torch.testing.assert_close(op.rmatvec(v), k.T @ v)
    rhs = torch.randn(24, 1, dtype=F64)
    torch.testing.assert_close(ranged.solve("b", rhs, alpha=0.1, eps=1e-13, x=x, y=x, s=0.6),
                               torch.linalg.solve(k + 0.1 * torch.eye(24, dtype=F64), rhs), rtol=1e-9, atol=1e-10)


def _dense_b(x, y, s):
    return torch.exp(-((x[..., :, None, :] - y[..., None, :, :]) ** 2).sum(-1) / (2 * s**2))


def test_batched_operator_products():
    g = torch.Generator().manual_seed(20)
    x = torch.randn(3, 12, 3, dtype=F64, generator=g)
    y = torch.randn(3, 15, 3, dtype=F64, generator=g)
    op = _gauss().linear_operator("b", x=x, y=y, s=0.8)
    assert op.batch == (3,) and op.shape == (3, 24, 30)
    k = _dense_b(x, y, 0.8)
    v = torch.randn(3, 15, 2, dtype=F64, generator=g)
    u = torch.randn(3, 12, 2, dtype=F64, generator=g)
    torch.testing.assert_close(op @ v, k @ v)
    torch.testing.assert_close(op.rmatvec(u), k.transpose(-1, -2) @ u)
    # v non batché diffusé, v de batch (2, 1) contre entrées (3,) -> (2, 3)
    torch.testing.assert_close(op @ v[0], k @ v[0])
    v2 = torch.randn(2, 1, 15, 2, dtype=F64, generator=g)
    torch.testing.assert_close(op @ v2, k @ v2)
    torch.testing.assert_close(op.T @ u[:, None].expand(3, 2, 12, 2).transpose(0, 1).contiguous(),
                               k.transpose(-1, -2) @ u[:, None].expand(3, 2, 12, 2).transpose(0, 1))
    with pytest.raises(ValidationError, match="aplaties"):
        op @ v[0].reshape(-1)
    # opérateur non batché appliqué à un v batché
    op1 = _gauss().linear_operator("b", x=x[0], y=y[0], s=0.8)
    assert op1.batch == ()
    torch.testing.assert_close(op1 @ v, k[0] @ v)
    torch.testing.assert_close(op1.rmatvec(u), k[0].T @ u)


def test_batched_solve_and_gradients():
    g = torch.Generator().manual_seed(21)
    x0 = torch.randn(2, 20, 3, dtype=F64, generator=g)
    rhs0 = torch.randn(20, 1, dtype=F64, generator=g)        # diffusé sur le batch des entrées
    w = torch.randn(2, 20, 1, dtype=F64, generator=g)
    run = _gauss(1)
    eye = torch.eye(20, dtype=F64)

    def leaves():
        return (x0.clone().requires_grad_(), rhs0.clone().requires_grad_(),
                torch.tensor(0.7, dtype=F64, requires_grad=True))

    x, rhs, s = leaves()
    a, info = run.linear_operator("b", x=x, y=x, s=s).solve(rhs, alpha=0.3, eps=1e-13, return_info=True)
    assert a.shape == (2, 20, 1) and info["converged"] and info["maxiter"] == 200
    got = torch.autograd.grad((a * w).sum(), [x, rhs, s])
    xr, rr, sr = leaves()
    ar = torch.linalg.solve(_dense_b(xr, xr, sr) + 0.3 * eye, rr)
    torch.testing.assert_close(a.detach(), ar.detach(), rtol=1e-9, atol=1e-10)
    want = torch.autograd.grad((ar * w).sum(), [xr, rr, sr])
    for gg, r in zip(got, want):
        torch.testing.assert_close(gg, r, rtol=1e-8, atol=1e-9)
    # entrées non batchées, plusieurs seconds membres (4, M, 1)
    rhs4 = torch.randn(4, 20, 1, dtype=F64, generator=g)
    a4 = run.solve("b", rhs4, alpha=0.3, eps=1e-13, x=x0[0], y=x0[0], s=0.7)
    torch.testing.assert_close(a4, torch.linalg.solve(_dense_b(x0[0], x0[0], 0.7) + 0.3 * eye, rhs4),
                               rtol=1e-9, atol=1e-10)
    with pytest.raises(ValidationError, match="x0"):
        run.solve("b", rhs4, x0=torch.zeros(20, 1, dtype=F64), x=x0[0], y=x0[0], s=0.7)
    with pytest.raises(ValidationError, match="diffusable"):
        run.solve("b", torch.randn(3, 20, 1, dtype=F64), x=x0, y=x0, s=0.7)
