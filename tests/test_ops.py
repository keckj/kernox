"""PyKeOps-parity formula operations (``kernox/ops.py``) on the oracle.

Each operation is compared with an **independent** NumPy transcription of the
keopscore C code (``keopscore/utils/math_functions.py`` and
``keopscore/formulas/maths/*.py``), evaluated in float64. Backend conformance
(the same DAG printed as CUDA/HIP/SYCL) lives in
``tests/conformance/test_ops.py``.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import numpy as np
import pytest
import torch

import kernox as kp
from kernox.errors import ValidationError


def evaluate(formula, **arrays):
    """Value of ``formula`` at each row i (a single row j): array (M, dim).

    ``Vj`` variables are passed with a single row; ``z`` (Vj, zero)
    guarantees that a j axis exists. ``+ 0`` leaves every nonzero value unchanged.
    """
    red = (formula + kp.Vj("z", 1)).sum_over("j")
    fn = red.compile("float64", backend="oracle")
    feed = {k: torch.as_tensor(np.asarray(v, dtype=np.float64)) for k, v in arrays.items()
            if k in formula.variables}
    feed["z"] = torch.zeros(1, 1, dtype=torch.float64)
    return fn(**feed).numpy()


# ------------------------------------------------- keopscore references
def c_round(v):
    """C ``round``: half away from zero (exact through Decimal)."""
    return np.array([float(Decimal(float(a)).to_integral_value(ROUND_HALF_UP)) if np.isfinite(a) else a
                     for a in np.ravel(v)]).reshape(np.shape(v))


def keops_round(x, d):
    if d == 0:
        return c_round(x)
    return c_round(x * 10.0 ** d) / 10.0 ** d


def keops_one_hot(f, D):
    """``out[(int)(f + .5)] = 1`` with ``f + .5`` **exact** (the keopscore float32 case,
    where the sum in double is exact; in float64 keopscore rounds the sum)."""
    out = np.zeros((len(f), D))
    for r, v in enumerate(f):
        k = int(Fraction(float(v)) + Fraction(1, 2))  # truncation toward zero
        out[r, k] = 1.0
    return out


def keops_bspline(t, x, k):
    """Transcription of ``keopscore/formulas/maths/BSpline.py`` (in-place De Boor)."""
    nk = len(t)
    B = [1.0 if (t[j] <= x and x < t[j + 1]) else 0.0 for j in range(nk - 1)]
    for kk in range(1, k + 1):
        r1 = (x - t[0]) / (t[kk] - t[0]) if t[0] < t[kk] else 0.0
        for i in range(nk - 1 - kk):
            r2 = (x - t[i + 1]) / (t[i + 1 + kk] - t[i + 1]) if t[i + 1] < t[i + 1 + kk] else 0.0
            B[i] = r1 * B[i] + (1.0 - r2) * B[i + 1]
            r1 = r2
    return np.array(B[:nk - k - 1])


def bspline_cox_de_boor(t, x, k):
    """Classic Cox-de Boor recursion (independent mathematical reference)."""
    def b(i, kk):
        if kk == 0:
            return 1.0 if t[i] <= x < t[i + 1] else 0.0
        left = (x - t[i]) / (t[i + kk] - t[i]) * b(i, kk - 1) if t[i + kk] > t[i] else 0.0
        right = (t[i + kk + 1] - x) / (t[i + kk + 1] - t[i + 1]) * b(i + 1, kk - 1) if t[i + kk + 1] > t[i + 1] else 0.0
        return left + right
    return np.array([b(i, k) for i in range(len(t) - k - 1)])


RNG = np.random.default_rng(1234)
X = np.concatenate([RNG.standard_normal((40, 3)) * 3,
                    np.array([[0.0, -0.0, 0.5], [-0.5, 1.5, -2.5], [2.5, 1e-300, -1e-300],
                              [np.inf, -np.inf, 7.0], [0.49999999999999994, -0.49999999999999994, 3.0]])])
xi = kp.Vi("x", 3)


def same(got, want, rtol=0.0, atol=0.0):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, equal_nan=True)


def keops_softdtw(x, y, gamma):
    """keopscore ``SoftDTW_SqDist::Op`` transcribed line by line (one x, y pair)."""
    n, m = len(x), len(y)
    rjm1 = np.zeros(n)
    rij = (x[0] - y[0]) ** 2
    rim1j = rij
    for i in range(1, n):
        rij = (x[i] - y[0]) ** 2 + rim1j
        rjm1[i - 1] = rim1j
        rim1j = rij
    rjm1[n - 1] = rij
    for j in range(1, m):
        rij = (x[0] - y[j]) ** 2 + rjm1[0]
        rim1j = rij
        for i in range(1, n):
            rij = (x[i] - y[j]) ** 2
            mn = np.fmin(np.fmin(rjm1[i - 1], rjm1[i]), rim1j)
            rij += mn - gamma * np.log(np.exp((mn - rjm1[i - 1]) / gamma) + np.exp((mn - rim1j) / gamma)
                                       + np.exp((mn - rjm1[i]) / gamma))
            rjm1[i - 1] = rim1j
            rim1j = rij
        rjm1[n - 1] = rij
    return rij


def torch_softdtw(x, y, gamma):
    """Soft-DTW (Cuturi & Blondel 2017) in Torch, full R table, for autograd."""
    n, m = x.shape[-1], y.shape[-1]
    D = (x[:, None, :, None] - y[None, :, None, :]) ** 2  # (M, N, n, m)
    R = [[None] * m for _ in range(n)]
    for j in range(m):
        for i in range(n):
            if i == 0 and j == 0:
                R[i][j] = D[..., 0, 0]
            elif j == 0:
                R[i][j] = D[..., i, 0] + R[i - 1][0]
            elif i == 0:
                R[i][j] = D[..., 0, j] + R[0][j - 1]
            else:
                r = torch.stack([R[i - 1][j - 1], R[i - 1][j], R[i][j - 1]])
                R[i][j] = D[..., i, j] - gamma * torch.logsumexp(-r / gamma, dim=0)
    return R[n - 1][m - 1]


# ------------------------------------------------------------- scalars
@pytest.mark.parametrize("name, ref", [
    ("relu", lambda x: np.where(x < 0, 0.0, x)),
    ("step", lambda x: np.where(x < 0, 0.0, 1.0)),
    ("inv", lambda x: 1.0 / x),
])
def test_unary_exact(name, ref):
    with np.errstate(divide="ignore"):
        same(evaluate(getattr(xi, name)(), x=X), ref(X))
    assert getattr(kp, name)(xi).dim == 3


def test_relu_step_nan():
    nan = np.array([[np.nan, 1.0, -1.0]])
    same(evaluate(xi.relu(), x=nan), [[np.nan, 1.0, 0.0]])
    same(evaluate(xi.step(), x=nan), [[1.0, 1.0, 0.0]])  # keopscore: NaN < 0 is false → 1


def test_sinxdivx_sinc_xlogx():
    with np.errstate(all="ignore"):
        same(evaluate(xi.sinxdivx(), x=X), np.where(X != 0, np.sin(X) / X, 1.0), rtol=1e-15)
        px = np.pi * X
        same(evaluate(xi.sinc(), x=X), np.where(px != 0, np.sin(px) / px, 1.0), rtol=1e-15)
        P = np.abs(X)
        same(evaluate(xi.xlogx(), x=P), np.where(P != 0, P * np.log(P), 0.0), rtol=1e-15)
    assert np.isnan(evaluate(xi.xlogx(), x=[[-1.0, 1.0, 0.0]])[0, 0])


@pytest.mark.parametrize("d", (0, 1, 2, -1))
def test_round(d):
    halves = np.array([[0.5, 1.5, 2.5], [-0.5, -1.5, -2.5], [0.05, 0.15, -0.25], [123.456, -9.995, 1e300]])
    data = np.concatenate([X, halves])
    # d ≠ 0: symox rewrites "/ 10^d" as "× 10^-d" (rounded constant);
    # at most 1 ulp of difference on the final division (recorded in the parity notes).
    with np.errstate(invalid="ignore", over="ignore"):
        same(evaluate(xi.round(d), x=data), keops_round(data, d), rtol=0 if d == 0 else 2.3e-16)


def test_round_signed_zero():
    with np.errstate(divide="ignore"):
        got = evaluate(1 / xi.round(), x=[[-0.3, 0.3, -0.0]])  # 1/±0 = ±inf (evaluate adds +0)
    assert list(got[0]) == [-np.inf, np.inf, -np.inf]  # C round: −0 for (−1/2, 0]


def test_round_rejects_non_integer():
    with pytest.raises(ValidationError):
        xi.round(0.5)


@pytest.mark.parametrize("n, off", [(2.0, 0.0), (0.7, 0.3), (-1.5, 0.0), (3.0, -1.0)])
def test_mod(n, off):
    with np.errstate(invalid="ignore"):
        same(evaluate(xi.mod(n, off), x=X), X - n * np.floor((X - off) / n))


def test_mod_vector_modulus():
    n = kp.Vj("n", 3)
    N = np.array([[1.0, 2.5, -0.75]])
    with np.errstate(invalid="ignore"):
        same(evaluate(xi.mod(n), x=X, n=N), X - N * np.floor(X / N))


def test_clampint_ifelse():
    same(evaluate(xi.clampint(-1, 2), x=X), np.clip(X, -1, 2))
    with pytest.raises(ValidationError):
        xi.clampint(-1.5, 2)
    y = kp.Vj("y", 3)
    Y = np.array([[10.0, 20.0, 30.0]])
    same(evaluate(xi.ifelse(y, -y), x=X, y=Y), np.where(X >= 0, Y, -Y))


# --------------------------------------------------------------- vectors
def test_normalize_sqdist():
    Xf = X[np.isfinite(X).all(1)]
    Z = np.concatenate([Xf, np.zeros((1, 3))])
    sq = (Z * Z).sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        want = np.where(sq == 0, 0.0, Z / np.sqrt(sq))
    same(evaluate(xi.normalize(), x=Z), want, rtol=4e-16, atol=0)
    assert np.all(evaluate(xi.normalize(), x=np.zeros((1, 3))) == 0)
    y = kp.Vj("y", 3)
    Y = np.array([[0.5, -1.0, 2.0]])
    same(evaluate(xi.sqdist(y), x=Xf, y=Y), ((Xf - Y) ** 2).sum(1, keepdims=True), rtol=1e-15)


@pytest.mark.parametrize("ds", (1, 3, 9))
def test_weightedsqnorm_weightedsqdist(ds):
    Xf = X[np.isfinite(X).all(1)]
    s, y = kp.Vj("s", ds), kp.Vj("y", 3)
    S = RNG.standard_normal((1, ds))
    Y = RNG.standard_normal((1, 3))
    for f, V in ((xi.weightedsqnorm(s), Xf), (xi.weightedsqdist(y, s), Xf - Y)):
        if ds == 1:
            want = S[0, 0] * (V * V).sum(1)
        elif ds == 3:
            want = (S * V * V).sum(1)
        else:
            want = np.einsum("kl,mk,ml->m", S.reshape(3, 3), V, V)
        same(evaluate(f, x=Xf, s=S, y=Y)[:, 0], want, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValidationError):
        xi.weightedsqnorm(kp.Vj("s", 4))


def test_one_hot():
    D = 5
    f = np.concatenate([RNG.uniform(-0.49, D - 0.51, 60), [0.5, 1.5, 3.5, -0.5, 4.49999, 0.49999999999999994]])
    xs = kp.Vi("f", 1)
    same(evaluate(xs.one_hot(D), f=f[:, None]), keops_one_hot(f, D))
    # outside [0, D) and NaN: undefined behavior in keopscore, zero vector here
    out = evaluate(xs.one_hot(D), f=np.array([[-2.0], [5.5], [np.nan], [np.inf]]))
    assert np.all(out == 0)
    with pytest.raises(ValidationError):
        xi.one_hot(3)


def test_elem_extract():
    same(evaluate(xi.elem(1), x=X), X[:, 1:2])
    same(evaluate(xi.extract(1, 2), x=X), X[:, 1:3])
    for bad in (lambda: xi.elem(3), lambda: xi.extract(2, 2), lambda: xi.extract(0, 0)):
        with pytest.raises(ValidationError):
            bad()


@pytest.mark.parametrize("n, m", [(2, 3), (1, 3), (4, 3)])
def test_matvecmult_vecmatmult(n, m):
    Xf = X[np.isfinite(X).all(1)]
    A = kp.Vj("A", n * m)
    Am = RNG.standard_normal((1, n * m))
    same(evaluate(A.matvecmult(xi), x=Xf, A=Am), Xf @ Am.reshape(n, m).T, rtol=1e-13, atol=1e-13)
    same(evaluate(xi.vecmatmult(A), x=Xf, A=Am), Xf @ Am.reshape(m, n), rtol=1e-13, atol=1e-13)
    with pytest.raises(ValidationError):
        kp.Vj("B", 4).matvecmult(xi)


def test_tensorprod():
    y = kp.Vj("y", 2)
    Y = np.array([[2.0, -3.0]])
    Xf = X[np.isfinite(X).all(1)]
    same(evaluate(xi.tensorprod(y), x=Xf, y=Y), np.einsum("mk,l->mkl", Xf, Y[0]).reshape(-1, 6))


@pytest.mark.parametrize("dimfa, dimfb, contfa, contfb, permute", [
    ((2, 3), (3, 4), (1,), (0,), None),
    ((2, 3), (3, 4), (1,), (0,), (1, 0)),
    ((2, 3, 2), (3, 2), (1, 2), (0, 1), None),
    ((2, 2), (2, 3), (), (), (3, 1, 0, 2)),
    ((6,), (6,), (0,), (0,), None),
])
def test_tensordot(dimfa, dimfb, contfa, contfb, permute):
    da, db = int(np.prod(dimfa)), int(np.prod(dimfb))
    a, b = kp.Vi("a", da), kp.Vj("b", db)
    Aa, Bb = RNG.standard_normal((7, da)), RNG.standard_normal((1, db))
    got = evaluate(a.keops_tensordot(b, dimfa, dimfb, contfa, contfb, *(() if permute is None else (permute,))),
                   a=Aa, b=Bb)
    for r in range(7):
        T = np.tensordot(Aa[r].reshape(dimfa), Bb[0].reshape(dimfb), axes=(contfa, contfb))
        if permute is not None:
            T = T.transpose(np.argsort(permute))
        np.testing.assert_allclose(got[r], np.ravel(T), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("dimfa, dimfb", [((3,), (2,)), ((2, 2), (3, 1)), ((2, 3), (2, 2))])
def test_kron(dimfa, dimfb):
    da, db = int(np.prod(dimfa)), int(np.prod(dimfb))
    a, b = kp.Vi("a", da), kp.Vj("b", db)
    Aa, Bb = RNG.standard_normal((5, da)), RNG.standard_normal((1, db))
    got = evaluate(a.keops_kron(b, dimfa, dimfb), a=Aa, b=Bb)
    for r in range(5):
        np.testing.assert_array_equal(got[r], np.ravel(np.kron(Aa[r].reshape(dimfa), Bb[0].reshape(dimfb))))


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_bspline(k):
    t = np.array([0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 3.5, 4.0, 4.0])  # includes repeated knots
    knots = kp.Vj("t", len(t))
    xs = np.concatenate([RNG.uniform(-0.5, 4.5, 80), t, [np.nextafter(1.0, 0.0)]])
    got = evaluate(knots.bspline(kp.Vi("x", 1), k), x=xs[:, None], t=t[None])
    for r, x in enumerate(xs):
        np.testing.assert_allclose(got[r], keops_bspline(t, x, k), rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(got[r], bspline_cox_de_boor(t, x, k), rtol=1e-12, atol=1e-14)
    with pytest.raises(ValidationError):
        kp.Vj("t2", k + 1).bspline(kp.Vi("x", 1), k)


def test_bspline_partition_of_unity():
    t = np.linspace(-1.0, 1.0, 12)
    k = 3
    xs = RNG.uniform(t[k], t[-k - 1], 50)
    got = evaluate(kp.Vj("t", 12).bspline(kp.Vi("x", 1), k), x=xs[:, None], t=t[None])
    np.testing.assert_allclose(got.sum(1), 1.0, rtol=1e-14)


@pytest.mark.parametrize(("n", "m", "gamma"), [(1, 1, 0.5), (1, 4, 0.5), (3, 1, 0.5), (5, 7, 0.1), (6, 6, 1.0),
                                                (8, 5, 2.0)])
def test_softdtw_sqdist(n, m, gamma):
    rng = np.random.default_rng(n * 10 + m)
    X, Y = rng.standard_normal((7, n)), rng.standard_normal((1, m))
    f = kp.Vi("x", n).softdtw_sqdist(kp.Vj("y", m), kp.Param("g", 1))
    assert f.dim == 1
    got = evaluate(f, x=X, y=Y, g=[gamma])[:, 0]
    want = np.array([keops_softdtw(X[r], Y[0], gamma) for r in range(len(X))])
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13)


def test_softdtw_sqdist_gradients_including_gamma():
    """x, y **and** γ (keopscore refuses γ) against torch.autograd on the full recursion."""
    n, m = 5, 4
    x = torch.randn(6, n, dtype=torch.float64, requires_grad=True)
    y = torch.randn(3, m, dtype=torch.float64, requires_grad=True)
    g = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, 1, dtype=torch.float64)
    f = kp.Vi("x", n).softdtw_sqdist(kp.Vj("y", m), kp.Param("g", 1)) * kp.Vj("b", 1)
    out = f.sum_over("j").compile("float64", backend="oracle")(x=x, y=y, g=g, b=b)
    grads = torch.autograd.grad(out.sum(), (x, y, g))
    xr, yr, gr = (t.detach().clone().requires_grad_(True) for t in (x, y, g))
    ref = (torch_softdtw(xr, yr, gr[0]) * b[None, :, 0]).sum()
    for a, w in zip(grads, torch.autograd.grad(ref, (xr, yr, gr))):
        torch.testing.assert_close(a, w, rtol=1e-12, atol=1e-12)


def test_softdtw_sqdist_validation():
    with pytest.raises(ValidationError):
        kp.Vi("x", 3).softdtw_sqdist(kp.Vj("y", 3), kp.Param("g", 2))


# ------------------------------------------------------------- autograd
def test_gradients_through_new_ops():
    """Symbolic gradients (differentiated DAG) against torch.autograd on the same function."""
    x = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(6, 3, dtype=torch.float64)
    s = torch.randn(6, 9, dtype=torch.float64)
    xv, yv, sv = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("s", 9)
    f = (xv.normalize().dot(yv) + xv.weightedsqdist(yv, sv) * 0.1 + xv.relu().sum() + xv.sinxdivx().sum()
         + kp.Vj("A", 6).matvecmult(xv).sum())
    A = torch.randn(6, 6, dtype=torch.float64)
    out = f.sum_over("j").compile("float64", backend="oracle")(x=x, y=y, s=s, A=A)
    (g,) = torch.autograd.grad(out.sum(), x)

    xr = x.detach().clone().requires_grad_(True)
    n = xr / xr.norm(dim=1, keepdim=True)
    d = xr[:, None, :] - y[None]
    w = torch.einsum("jkl,ijk,ijl->ij", s.view(6, 3, 3), d, d)
    ref = ((n[:, None, :] * y[None]).sum(-1) + 0.1 * w + torch.relu(xr).sum(1, keepdim=True)
           + (torch.sin(xr) / xr).sum(1, keepdim=True)
           + torch.einsum("jnm,im->ijn", A.view(6, 2, 3), xr).sum(-1)).sum()
    (gr,) = torch.autograd.grad(ref, xr)
    torch.testing.assert_close(g, gr, rtol=1e-12, atol=1e-12)


def test_module_functions_match_methods():
    y = kp.Vj("y", 3)
    assert kp.relu(xi).components == xi.relu().components
    assert kp.sqdist(xi, y).components == xi.sqdist(y).components
    assert kp.tensordot(xi, y, [3], [3], [0], [0]).components == xi.keops_tensordot(y, [3], [3], [0], [0]).components
    assert kp.kron(xi, y, [3], [3]).dim == 9


def test_factorize_and_auto_factorize_are_semantic_identities():
    """keopscore ``Factorize``/``AutoFactorize``: the generator's CSE already factorizes."""
    import re

    from kernox.codegen.pair import pair_function

    x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 1)
    g = kp.exp(-(x - y).sqnorm2())
    f = kp.sin(g) + kp.cos(g) * g * b
    assert f.factorize(g) is f and f.auto_factorize() is f
    with pytest.raises(TypeError):
        f.factorize(1.0)
    plan = f.sum_over("j").to_plan("float64")
    for dialect in ("cuda", "hip", "sycl"):
        src = pair_function(plan, dialect).source
        assert len(re.findall(r"\bexp\(", src)) == 1, src  # g evaluated only once
    z = kp.Vi("z", 1, complex=True)
    assert z.factorize(g).auto_factorize() is z
