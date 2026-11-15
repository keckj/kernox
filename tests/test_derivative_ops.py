"""Opérateurs de dérivation dans la formule (parité PyKeOps : Grad, Diff,
GradMatrix, Divergence, Laplacian, TraceOperator), contre ``torch.func``."""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox.errors import ValidationError

F64 = torch.float64


def _pairs(formula, **tensors):
    """Valeur de ``formula`` sur les paires (i, j=0) : y a une seule ligne."""
    run = formula.sum_over("j").compile("float64", backend="oracle")
    return run(**{k: v for k, v in tensors.items() if k in formula.variables})


def _f_torch(x, y, b):
    """f(x, y) = exp(-|x - y|²) · b + sin(x[0] · y)  (dimension 3)."""
    return torch.exp(-((x - y) ** 2).sum()) * b + torch.sin(x[0] * y)


def _f_kp(x, y, b):
    return kp.exp(-(x - y).sqnorm2()) * b + kp.sin(x[0] * y)


@pytest.fixture
def data():
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, dtype=F64, generator=g)  # noqa: E731
    return dict(x=r(6, 3), y=r(1, 3), b=r(1, 3), e=r(6, 3), u=r(6, 3))


def test_grad_diff_grad_matrix(data):
    x, y, b, e, u = (kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 3), kp.Vi("e", 3), kp.Vi("u", 3))
    f = _f_kp(x, y, b)
    J = torch.stack([torch.func.jacrev(_f_torch)(xi, data["y"][0], data["b"][0]) for xi in data["x"]])  # (M, 3, 3)
    got = _pairs(f.grad(x, e), **data)
    torch.testing.assert_close(got, torch.einsum("mkc,mk->mc", J, data["e"]), rtol=1e-12, atol=1e-12)
    got = _pairs(kp.diff(f, x, u), **data)
    torch.testing.assert_close(got, torch.einsum("mkc,mc->mk", J, data["u"]), rtol=1e-12, atol=1e-12)
    got = _pairs(f.grad_matrix(x), **data)
    torch.testing.assert_close(got, J.reshape(6, 9), rtol=1e-12, atol=1e-12)
    # par rapport à une variable j et à un paramètre
    got = _pairs(f.grad(y, e), **data)
    Jy = torch.stack([torch.func.jacrev(_f_torch, argnums=1)(xi, data["y"][0], data["b"][0]) for xi in data["x"]])
    torch.testing.assert_close(got, torch.einsum("mkc,mk->mc", Jy, data["e"]), rtol=1e-12, atol=1e-12)


def test_divergence_laplacian(data):
    x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 3)
    f = _f_kp(x, y, b)
    J = torch.stack([torch.func.jacrev(_f_torch)(xi, data["y"][0], data["b"][0]) for xi in data["x"]])
    got = _pairs(f.divergence(x), **data)
    torch.testing.assert_close(got[:, 0], torch.einsum("mcc->m", J), rtol=1e-12, atol=1e-12)
    s = lambda xi: _f_torch(xi, data["y"][0], data["b"][0]).sum()  # noqa: E731
    H = torch.stack([torch.func.hessian(s)(xi) for xi in data["x"]])
    got = _pairs(f.sum().laplacian(x), **data)
    torch.testing.assert_close(got[:, 0], torch.einsum("mcc->m", H), rtol=1e-12, atol=1e-12)


def test_trace_operator_of_linear_map(data):
    """f(v) = A v (A variable j, 3×3) : trace de l'application = trace(A)."""
    x, A = kp.Vi("x", 3), kp.Vj("A", 9)
    f = A.matvecmult(x)
    Am = torch.randn(1, 9, dtype=F64)
    got = _pairs(f.trace_operator(x), x=data["x"], A=Am)
    torch.testing.assert_close(got[:, 0], Am.view(3, 3).trace().expand(6), rtol=1e-14, atol=1e-14)


def test_derivative_ops_are_differentiable_formulas(data):
    """Le résultat est une formule ordinaire : réduite et dérivée à nouveau (ordre 2)."""
    x, y, b, e = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 3), kp.Vi("e", 3)
    X = data["x"].clone().requires_grad_(True)
    out = _pairs(_f_kp(x, y, b).grad(x, e), x=X, y=data["y"], b=data["b"], e=data["e"])
    (g,) = torch.autograd.grad(out.sum(), X)

    def ref(xi, ei):
        return torch.func.vjp(lambda z: _f_torch(z, data["y"][0], data["b"][0]), xi)[1](ei)[0].sum()

    want = torch.stack([torch.func.grad(ref)(xi, ei) for xi, ei in zip(data["x"], data["e"])])
    torch.testing.assert_close(g, want, rtol=1e-11, atol=1e-11)


def test_validation():
    x, y = kp.Vi("x", 3), kp.Vj("y", 3)
    f = (x * y).sum()
    with pytest.raises(ValidationError):
        f.grad(x * 2, kp.Vi("e", 1))  # pas une variable
    with pytest.raises(ValidationError):
        f.grad(x, kp.Vi("e", 2))  # e de la dimension de f
    with pytest.raises(ValidationError):
        (x * y).divergence(kp.Vi("z", 2))
    with pytest.raises(ValidationError):
        (x * y).laplacian(x)  # f non scalaire
    with pytest.raises(ValidationError):
        kp.diff(f, x, kp.Vi("u", 2))
