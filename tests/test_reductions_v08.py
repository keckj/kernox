"""Avenant v0.8 : MIN_ARGMIN, MAX_ARGMAX, LOGSUMEXPWEIGHT (frontend, plan, oracles)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import kernox as kp
from kernox.errors import ValidationError
from kernox.oracle import reduce_numpy, reference_numpy
from kernox.plan import ReductionKind as R

F64 = torch.float64


def _t(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


# ------------------------------------------------------------- frontend
def test_plans_and_shapes():
    x, y, g = kp.Vi("x", 2), kp.Vj("y", 2), kp.Vj("g", 3)
    p = (x * y).min_argmin(over="j").to_plan("float32")
    assert p.reduction.kind is R.MIN_ARGMIN and p.out_dim == 2
    assert {k: v for k, v in kp.plan.output_shapes(p, 5).items()} == {
        "value": ((5, 2), "float32"), "index": ((5, 2), "int64")}
    p = (x * y).max_argmax(over="i").to_plan("float64")
    assert p.reduction.kind is R.MAX_ARGMAX and p.meta["swapped_ij"]
    f = -(x - y).sqnorm2()
    p = f.logsumexp(over="j", weight=g).to_plan("float32", acc_dtype="float64")
    assert p.reduction.kind is R.LOGSUMEXPWEIGHT and p.out_dim == 4 and p.acc_dtype == "float64"
    assert kp.plan.output_shapes(p, 7) == {"value": ((7, 3), "float64")}
    assert f.logsumexp(over="j").to_plan("float32").reduction.kind is R.LOGSUMEXP
    keys = {e.to_plan("float32").key() for e in (
        f.min_argmin(over="j"), f.min(over="j"), f.argmin(over="j"), f.max_argmax(over="j"),
        f.logsumexp(over="j", weight=g), f.sumsoftmaxweight(g, over="j"))}
    assert len(keys) == 6


def test_validation():
    x, y = kp.Vi("x", 2), kp.Vj("y", 2)
    with pytest.raises(ValidationError):
        (x * y).logsumexp(over="j", weight=y)  # score non scalaire
    with pytest.raises(TypeError):
        (x * y).sum().logsumexp(over="j", weight=y[0] > 0)
    with pytest.raises(ValidationError):
        kp.builder.make_plan([(x * y).sum().components[0]], R.LOGSUMEXPWEIGHT, "float32",
                             args=[kp.builder.make_arg("x", "i", 2), kp.builder.make_arg("y", "j", 2)])
    with pytest.raises(ValidationError):
        (x * y).min_argmin(over="j").to_plan("float32", acc_dtype="float64")


# ---------------------------------------------------- MIN_ARGMIN / MAX
def _data(rng, m, n, d):
    X = rng.integers(-3, 4, size=(m, d)).astype(np.float64)  # égalités fréquentes
    Y = rng.integers(-3, 4, size=(n, d)).astype(np.float64)
    return X, Y


@pytest.mark.parametrize("over", ("i", "j"))
@pytest.mark.parametrize("is_min", (True, False))
def test_min_argmin_equals_separate_reductions(over, is_min):
    rng = np.random.default_rng(0)
    X, Y = _data(rng, 37, 1500, 3)
    Y[5, 1] = np.nan
    Y[:, 2] = np.nan  # composante sans candidat : (±inf, −1)
    x, y = kp.Vi("x", 3), kp.Vj("y", 3)
    f = x * y
    both = (f.min_argmin if is_min else f.max_argmax)(over=over).compile("float64", backend="oracle")
    val = (f.min if is_min else f.max)(over=over).compile("float64", backend="oracle")
    idx = (f.argmin if is_min else f.argmax)(over=over).compile("float64", backend="oracle")
    v, i = both(x=_t(X), y=_t(Y))
    assert v.dtype == F64 and i.dtype == torch.int64
    assert torch.equal(v, val(x=_t(X), y=_t(Y))) and torch.equal(i, idx(x=_t(X), y=_t(Y)))
    if over == "j":
        assert torch.all(i[:, 2] == -1) and torch.all(v[:, 2] == (math.inf if is_min else -math.inf))
        ref = reference_numpy((f.min_argmin if is_min else f.max_argmax)(over="j").to_plan("float64"),
                              {"x": X, "y": Y})
        np.testing.assert_array_equal(v.numpy(), ref["value"])
        np.testing.assert_array_equal(i.numpy(), ref["index"])
        # cohérence : la valeur est celle de l'indice retenu
        sel = np.take_along_axis((X[:, None, :] * Y[None]), np.maximum(i.numpy(), 0)[:, None, :], 1)[:, 0]
        ok = i.numpy() >= 0
        np.testing.assert_array_equal(v.numpy()[ok], sel[ok])


def test_min_argmin_batch_and_ranges():
    rng = np.random.default_rng(1)
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    f = (x - y) ** 2
    run = f.min_argmin(over="j").compile("float64", backend="oracle")
    X, Y = rng.standard_normal((2, 3, 20, 1)), rng.standard_normal((2, 3, 30, 1))
    v, i = run(x=_t(X), y=_t(Y))
    assert v.shape == (2, 3, 20, 1) and i.shape == (2, 3, 20, 1)
    D = (X[..., :, None, 0] - Y[..., None, :, 0]) ** 2
    np.testing.assert_array_equal(i.numpy()[..., 0], D.argmin(-1))
    np.testing.assert_array_equal(v.numpy()[..., 0], D.min(-1))
    # ranges : deux groupes diagonaux
    lx, ly = np.repeat([0, 1], 10), np.repeat([0, 1], 15)
    r = kp.ranges_from_labels(torch.as_tensor(lx), torch.as_tensor(ly), torch.eye(2, dtype=torch.bool))
    v, i = run.with_ranges(r)(x=_t(X[0, 0]), y=_t(Y[0, 0]))
    Dm = np.where(lx[:, None] == ly[None], D[0, 0], np.inf)
    np.testing.assert_array_equal(i.numpy()[:, 0], Dm.argmin(-1))


def test_min_argmin_indices_not_differentiable_values_are():
    """Indices non différentiables ; valeurs : gradient au terme retenu
    (``tests/test_reduction_gradients.py``)."""
    x = torch.randn(5, 1, dtype=F64, requires_grad=True)
    y = torch.randn(4, 1, dtype=F64)
    v, i = (kp.Vi("x", 1) * kp.Vj("y", 1)).min_argmin(over="j").compile("float64", backend="oracle")(x=x, y=y)
    assert not i.requires_grad
    v.sum().backward()
    torch.testing.assert_close(x.grad, y[i[:, 0]], rtol=0, atol=0)


# ---------------------------------------------------- LOGSUMEXPWEIGHT
def _lsew_dense(F, G):
    """log Σ_j exp(F_ij) G_je en FP64 dense, poids positifs."""
    return torch.logsumexp(F[..., None] + torch.log(G)[None], dim=1)


@pytest.mark.parametrize("scheme", ("direct", "block", "kahan"))
def test_lsew_against_dense(scheme):
    rng = np.random.default_rng(2)
    X, Y, G = rng.standard_normal((40, 3)), rng.standard_normal((700, 3)), rng.random((700, 4)) + 0.1
    x, y, g = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("g", 4)
    f = -(x - y).sqnorm2() * 30  # grands écarts : le décalage est nécessaire
    out = kp.compile(f.logsumexp(over="j", weight=g), dtype="float64", backend="oracle",
                     sum_scheme=scheme)(x=_t(X), y=_t(Y), g=_t(G))
    F = -((_t(X)[:, None] - _t(Y)[None]) ** 2).sum(-1) * 30
    torch.testing.assert_close(out, _lsew_dense(F, _t(G)), rtol=1e-13, atol=1e-13)


def test_lsew_numpy_reference_and_over_i():
    rng = np.random.default_rng(3)
    X, Y, G = rng.standard_normal((9, 2)), rng.standard_normal((13, 2)), rng.standard_normal((13, 2))
    x, y, g = kp.Vi("x", 2), kp.Vj("y", 2), kp.Vj("g", 2)
    e = (x * y).sum().logsumexp(over="j", weight=g)
    got = e.compile("float64", backend="oracle")(x=_t(X), y=_t(Y), g=_t(G)).numpy()
    ref = reference_numpy(e.to_plan("float64"), {"x": X, "y": Y, "g": G})["value"]
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13, equal_nan=True)
    assert np.isnan(got).any()  # poids négatifs : t_e < 0 → NaN
    # over="i" : poids côté i
    gi = kp.Vi("h", 2)
    e = (x * y).sum().logsumexp(over="i", weight=gi)
    H = rng.random((9, 2))
    got = e.compile("float64", backend="oracle")(x=_t(X), y=_t(Y), h=_t(H))
    F = _t(X) @ _t(Y).T  # (M, N)
    torch.testing.assert_close(got, _lsew_dense(F.T, _t(H)), rtol=1e-13, atol=1e-13)


def test_lsew_special_values():
    """Valeurs spéciales de l'avenant v0.8 (oracle Torch et référence NumPy)."""
    inf, nan = math.inf, math.nan
    # une ligne i par cas ; f(i, j) = a_i · b_j choisi par les données, poids g_j
    rows = {
        "vide (-inf)": ([-inf], [[1.0, 2.0]], [-inf, -inf]),
        "+inf, poids > 0 et nul": ([inf], [[2.0, 0.0]], [inf, nan]),
        "+inf, poids < 0": ([inf], [[-1.0, 3.0]], [nan, inf]),
        "f NaN": ([nan], [[1.0, 1.0]], [nan, nan]),
        "g NaN de poids nul": ([-inf, 0.0], [[nan, nan], [1.0, 2.0]], [0.0, math.log(2.0)]),
    }
    x, y, g = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("g", 2)
    e = (x + y).logsumexp(over="j", weight=g)
    run = e.compile("float64", backend="oracle")
    plan = e.to_plan("float64")
    for name, (fy, gs, want) in rows.items():
        X = np.zeros((1, 1))
        Y = np.array(fy)[:, None]
        G = np.array(gs)
        got = run(x=_t(X), y=_t(Y), g=_t(G)).numpy()[0]
        vals = np.concatenate([X[:, None, :] + Y[None], G[None]], axis=-1)  # (1, N, 1 + E)
        ref = reduce_numpy(plan, vals)["value"][0]
        np.testing.assert_array_equal(got, np.array(want), err_msg=name)
        np.testing.assert_array_equal(ref, np.array(want), err_msg=name)


def test_lsew_batch():
    rng = np.random.default_rng(4)
    x, y, g = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("g", 2)
    run = (x * y).logsumexp(over="j", weight=g).compile("float64", backend="oracle")
    X, Y, G = rng.standard_normal((3, 6, 1)), rng.standard_normal((3, 8, 1)), rng.random((3, 8, 2))
    out = run(x=_t(X), y=_t(Y), g=_t(G))
    assert out.shape == (3, 6, 2)
    for b in range(3):
        F = _t(X[b]) @ _t(Y[b]).T
        torch.testing.assert_close(out[b], _lsew_dense(F, _t(G[b])), rtol=1e-13, atol=1e-13)
