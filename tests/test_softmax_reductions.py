"""LOGSUMEXP et SUMSOFTMAXWEIGHT (contrat v0.5) : frontend, plan, oracles.

Oracle Torch par blocs (fusion d'états par décalage) contre l'oracle NumPy
FP64 indépendant (``math.fsum``) et contre ``torch.logsumexp`` / softmax dense.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import kernox as kp
from kernox.errors import ValidationError
from kernox.oracle import OracleReduction, reference_numpy
from kernox.plan import ReductionKind as R
from kernox.plan import output_shapes

INF, NAN = math.inf, math.nan


def run_oracle(plan, inputs, block_pairs=7):
    m = next(inputs[a.name].shape[0] for a in plan.args if a.role == "i")
    outs = {k: torch.empty(s, dtype=getattr(torch, d)) for k, (s, d) in output_shapes(plan, m).items()}
    OracleReduction(plan, "cpu", block_pairs).launch(inputs, outs)
    return outs["value"].numpy()


def _lse(f):
    return f.logsumexp(over="j")


def _ssmw(f, g):
    return f.sumsoftmaxweight(g, over="j")


# ------------------------------------------------------------------ frontend / plan
def test_plan_shapes_and_kinds():
    x, y, v = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("v", 5)
    d2 = ((x - y) ** 2).sum()
    p = _lse(-d2).to_plan("float32")
    assert p.reduction.kind is R.LOGSUMEXP and output_shapes(p, 4) == {"value": ((4, 1), "float32")}
    p = _lse(kp.cat(-d2, -2 * d2)).to_plan("float32", acc_dtype="float64")
    assert output_shapes(p, 4) == {"value": ((4, 2), "float64")}
    p = _ssmw(-d2, v).to_plan("float64")
    assert p.reduction.kind is R.SUMSOFTMAXWEIGHT and p.out_dim == 6
    assert output_shapes(p, 4) == {"value": ((4, 5), "float64")}
    p = (-d2).logsumexp(over="i").to_plan("float32")
    assert p.meta["swapped_ij"] and [a.role for a in p.args] == ["i", "j"]


def test_frontend_validation():
    x, y = kp.Vi("x", 2), kp.Vj("y", 2)
    with pytest.raises(ValidationError):
        (x - y).sumsoftmaxweight(y, over="j")          # score non scalaire
    with pytest.raises(TypeError):
        (x - y).sum().sumsoftmaxweight(x < y, over="j")
    with pytest.raises(TypeError):
        (x < y).logsumexp(over="j")
    with pytest.raises(ValidationError):
        (x - y).sum().logsumexp(over="k")
    with pytest.raises(ValidationError):
        kp.builder.make_plan([x.components[0]], R.SUMSOFTMAXWEIGHT, "float32",
                             args=_lse((x - y).sum()).to_plan("float32").args)


def test_backward_first_order_only():
    """Gradient d'ordre 1 (plan SUM substitut, cf. test_reduction_gradients.py) ;
    la dérivée seconde est refusée (once_differentiable)."""
    x = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    y = torch.randn(7, 2, dtype=torch.float64)
    f = -((kp.Vi("x", 2) - kp.Vj("y", 2)) ** 2).sum()
    for expr in (_lse(f), _ssmw(f, kp.Vj("y", 2))):
        out = kp.compile(expr, dtype="float64", backend="oracle")(x=x, y=y)
        (g,) = torch.autograd.grad(out.sum(), x, create_graph=True)
        assert torch.isfinite(g).all()
        with pytest.raises(RuntimeError):
            g.sum().backward()


# ------------------------------------------------------------------ oracles
def _inputs(plan, m, n, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return {a.name: torch.randn((m if a.role == "i" else n, a.dim), generator=g, dtype=torch.float64) * scale
            for a in plan.args}


def formulas():
    x, y, s = kp.Vi("x", 3), kp.Vj("y", 3), kp.Param("s")
    d2 = ((x - y) ** 2).sum()
    out = {"lse_gauss": _lse(-d2 / (2.0 * s ** 2)), "lse_vec": _lse(kp.cat(-d2, -3 * d2, x[0] * y[1]))}
    for E in (1, 4, 16):
        q, k_, v = kp.Vi("q", 8), kp.Vj("k", 8), kp.Vj("v", E)
        out[f"attn_E{E}"] = _ssmw(q.dot(k_) / math.sqrt(8.0), v)
    return out


@pytest.mark.parametrize("name", list(formulas()))
@pytest.mark.parametrize("mn", [(1, 1), (6, 29), (17, 300)])
def test_oracle_torch_vs_numpy(name, mn):
    expr = formulas()[name]
    plan = expr.to_plan("float64")
    inp = _inputs(plan, *mn, seed=mn[1])
    if "s" in inp:
        inp["s"] = torch.tensor([0.7], dtype=torch.float64)
    for bp in (3, 50, 1 << 20):
        np.testing.assert_allclose(run_oracle(plan, inp, bp), reference_numpy(plan, inp)["value"],
                                   rtol=1e-13, atol=1e-13)


def test_oracle_vs_torch_dense():
    x, y, v = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("v", 4)
    X, Y, V = (torch.randn(s, dtype=torch.float64) for s in ((40, 3), (333, 3), (333, 4)))
    D = -((X[:, None] - Y[None]) ** 2).sum(-1) * 30.0
    f = -((x - y) ** 2).sum() * 30.0
    lse = run_oracle(_lse(f).to_plan("float64"), {"x": X, "y": Y}, 1000)
    # LSE proche de 0 possible : l'erreur d'arrondi suit l'échelle des termes |f|, pas |LSE|
    np.testing.assert_allclose(lse[:, 0], torch.logsumexp(D, 1).numpy(), rtol=1e-14,
                               atol=1e-15 * float(D.abs().max()))
    sw = run_oracle(_ssmw(f, v).to_plan("float64"), {"x": X, "y": Y, "v": V}, 1000)
    np.testing.assert_allclose(sw, (torch.softmax(D, 1) @ V).numpy(), rtol=1e-12, atol=1e-14)


def _special_plan(kind, E=2):
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    f = x - y                                    # x = 0 : f_ij = -y_j exactement
    if kind is R.LOGSUMEXP:
        return _lse(f).to_plan("float64")
    return _ssmw(f, kp.Vj("w", E) * 1.0).to_plan("float64")


SPECIAL = {
    "big": [-1e4, -1e4 + 1, -9990.0],            # f = 1e4 : sans débordement
    "very_negative": [1e4, 1e4 - 2, 1e4 + 3],
    "neg_inf_mix": [INF, 1.0, INF, -2.0],        # f = -inf : poids nul
    "all_neg_inf": [INF, INF, INF],
    "pos_inf": [-INF, 0.0, 3.0],
    "two_pos_inf": [-INF, 0.0, -INF],
    "pos_and_neg_inf": [-INF, INF],
    "nan": [0.0, NAN, 1.0],
    "nan_and_pos_inf": [NAN, -INF],
    "single": [0.5],
    "empty": [],
    "dominant": [-800.0, 0.0, 1.0, 2.0],
}


def _expected(kind, f, w):
    f = np.asarray(f, dtype=np.float64)
    if np.isnan(f).any():
        return np.full(1 if kind is R.LOGSUMEXP else w.shape[1], NAN)
    if (f == INF).any():
        return np.full(1 if kind is R.LOGSUMEXP else w.shape[1], INF if kind is R.LOGSUMEXP else NAN)
    live = f != -INF
    if not live.any():
        return np.full(1 if kind is R.LOGSUMEXP else w.shape[1], -INF if kind is R.LOGSUMEXP else NAN)
    mx = f[live].max()
    e = np.where(live, np.exp(np.where(live, f - mx, 0.0)), 0.0)
    if kind is R.LOGSUMEXP:
        return np.array([mx + math.log(e.sum())])
    return (e[:, None] * w).sum(0) / e.sum()


@pytest.mark.parametrize("kind", [R.LOGSUMEXP, R.SUMSOFTMAXWEIGHT], ids=lambda k: k.value)
@pytest.mark.parametrize("case", list(SPECIAL))
def test_special_values(kind, case):
    yv = SPECIAL[case]
    n = len(yv)
    plan = _special_plan(kind)
    w = np.arange(1.0, 2 * n + 1).reshape(n, 2)
    inp = {"x": torch.zeros(2, 1, dtype=torch.float64), "y": torch.tensor(yv, dtype=torch.float64).reshape(n, 1)}
    if kind is R.SUMSOFTMAXWEIGHT:
        inp["w"] = torch.tensor(w)
    want = _expected(kind, -np.asarray(yv, dtype=np.float64), w)
    ref = reference_numpy(plan, inp)["value"]
    for row in ref:
        np.testing.assert_allclose(row, want, rtol=1e-15, equal_nan=True)
    for bp in (2, 3, 1 << 20):  # tranches de j : vides (tout −inf), à +inf, à NaN
        np.testing.assert_allclose(run_oracle(plan, inp, bp), ref, rtol=1e-15, equal_nan=True)


def test_empty_slices_in_partition():
    """Tranches de j entièrement à −inf entre deux tranches vivantes : neutres."""
    plan = _special_plan(R.SUMSOFTMAXWEIGHT, E=1)
    yv = [INF] * 6 + [0.0, 1.0] + [INF] * 6 + [2.0]
    n = len(yv)
    inp = {"x": torch.zeros(3, 1, dtype=torch.float64), "y": torch.tensor(yv, dtype=torch.float64).reshape(n, 1),
           "w": torch.arange(n, dtype=torch.float64).reshape(n, 1)}
    ref = reference_numpy(plan, inp)["value"]
    assert np.isfinite(ref).all()
    for bp in (3, 6, 9):
        np.testing.assert_allclose(run_oracle(plan, inp, bp), ref, rtol=1e-15)


def test_nan_weight_with_zero_weight_term_ignored():
    """g NaN sur un terme f = −inf (poids nul) : ignoré ; sur un terme vivant : NaN."""
    plan = _special_plan(R.SUMSOFTMAXWEIGHT, E=1)
    inp = {"x": torch.zeros(1, 1, dtype=torch.float64), "y": torch.tensor([[INF], [0.0]], dtype=torch.float64),
           "w": torch.tensor([[NAN], [3.0]], dtype=torch.float64)}
    assert reference_numpy(plan, inp)["value"][0, 0] == 3.0
    assert run_oracle(plan, inp, 1)[0, 0] == 3.0
    inp["y"] = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    assert math.isnan(reference_numpy(plan, inp)["value"][0, 0])
    assert math.isnan(run_oracle(plan, inp, 1)[0, 0])
