"""Sémantique des réductions : oracle Torch par blocs vs oracle NumPy FP64 indépendant."""
import math

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from kernox import Param, Vi, Vj, exp, sqrt, where
from kernox.oracle import OracleReduction, reference_numpy
from kernox.plan import output_shapes

from conftest import make_inputs

INF = math.inf


def run_oracle(plan, inputs, block_pairs=7):
    m = next(inputs[a.name].shape[0] for a in plan.args if a.role == "i")
    outs = {k: torch.empty(s, dtype=getattr(torch, d)) for k, (s, d) in output_shapes(plan, m).items()}
    OracleReduction(plan, "cpu", block_pairs).launch(inputs, outs)
    return {k: v.numpy() for k, v in outs.items()}


def assert_same(plan, got, ref, rtol):
    assert got.keys() == ref.keys()
    for key in got:
        g, r = got[key], np.asarray(ref[key])
        assert g.shape == r.shape, key
        if key == "index" or plan.reduction.kind.value == "count":
            np.testing.assert_array_equal(g, r)
        else:
            np.testing.assert_allclose(g, r, rtol=rtol, atol=rtol, equal_nan=True)


def formulas():
    x, y, w, s = Vi("x", 3), Vj("y", 3), Vj("w", 2), Param("s")
    d2 = ((x - y) ** 2).sum()
    g = exp(-d2 / (2.0 * s ** 2))
    return {
        "sum_vec": (g * w).sum_over("j"),
        "sum_over_i": (g * w).sum_over("i"),
        "count": (d2 <= 2.5).sum_over("j"),
        "min": (x - y).min(over="j"),
        "max": (x - y).max(over="j"),
        "argmin": (x - y).argmin(over="j"),
        "argmax": (x - y).argmax(over="i"),
        "kmin": d2.kmin(4, over="j"),
        "argkmin": d2.argkmin(4, over="j"),
        "kmin_argkmin": d2.kmin_argkmin(5, over="i"),
        "where_sqrt": where(d2 > 1.0, 1 / sqrt(d2 - 1.0), 0.0).sum_over("j"),
    }


@pytest.mark.parametrize("name", list(formulas()))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mn", [(5, 9), (13, 4), (1, 1)])
def test_random_matches_numpy(name, dtype, mn):
    plan = formulas()[name].to_plan(dtype)
    inputs = make_inputs(plan, *mn, seed=hash((name, mn)) % 1000)
    assert_same(plan, run_oracle(plan, inputs), reference_numpy(plan, inputs),
                1e-5 if dtype == "float32" else 1e-12)


def column_plan(kind, dtype="float64", k=None):
    """Valeurs v_j exactement contrôlées : f(i, j) = x_i * v_j avec x = 1."""
    x, v = Vi("x"), Vj("v")
    f = x * v
    r = getattr(f, kind)(k, over="j") if k is not None else getattr(f, kind)(over="j")
    return r.to_plan(dtype)


def col_inputs(values, m=2, dtype=torch.float64):
    return {"x": torch.ones((m, 1), dtype=dtype), "v": torch.tensor(values, dtype=dtype).reshape(-1, 1)}


VALUES = [3.0, 1.0, float("nan"), 1.0, -0.0, 0.0, INF, -INF, float("nan"), -INF]


@pytest.mark.parametrize("block", [1, 3, 1000])
def test_ties_nan_inf(block):
    inp = col_inputs(VALUES)
    out = run_oracle(column_plan("min"), inp, block)
    assert out["value"][0, 0] == -INF
    out = run_oracle(column_plan("argmin"), inp, block)
    assert out["index"][0, 0] == 7  # premier -inf, NaN ignorés
    out = run_oracle(column_plan("argmax"), inp, block)
    assert out["index"][0, 0] == 6
    plan = column_plan("kmin_argkmin", k=8)
    out = run_oracle(plan, inp, block)
    np.testing.assert_array_equal(out["index"][0], [7, 9, 4, 5, 1, 3, 0, 6])
    np.testing.assert_array_equal(out["value"][0], [-INF, -INF, -0.0, 0.0, 1.0, 1.0, 3.0, INF])
    assert_same(plan, out, reference_numpy(plan, inp), 0)


@pytest.mark.parametrize("block", [1, 4, 1000])
def test_k_larger_than_valid_candidates(block):
    inp = col_inputs([float("nan"), 2.0, float("nan"), INF])
    plan = column_plan("kmin_argkmin", k=6)
    out = run_oracle(plan, inp, block)
    np.testing.assert_array_equal(out["index"][0], [1, 3, -1, -1, -1, -1])
    np.testing.assert_array_equal(out["value"][0], [2.0, INF, INF, INF, INF, INF])
    assert_same(plan, out, reference_numpy(plan, inp), 0)


def test_all_nan_row():
    inp = col_inputs([float("nan")] * 3)
    assert run_oracle(column_plan("min"), inp)["value"][0, 0] == INF
    assert run_oracle(column_plan("max"), inp)["value"][0, 0] == -INF
    assert run_oracle(column_plan("argmin"), inp)["index"][0, 0] == -1
    assert run_oracle(column_plan("argmax"), inp)["index"][0, 0] == -1
    s = run_oracle((Vi("x") * Vj("v")).sum_over("j").to_plan("float64"), inp)["value"]
    assert np.isnan(s).all()  # SUM propage les NaN


def test_count_nan_comparison_false():
    x, v = Vi("x"), Vj("v")
    plan = (x * v < 2.0).sum_over("j").to_plan("float32")
    inp = col_inputs([1.0, float("nan"), 5.0, -INF], dtype=torch.float32)
    out = run_oracle(plan, inp)
    assert out["value"].dtype == np.int64 and (out["value"] == 2).all()


def test_empty_axes_and_k0():
    plan = column_plan("kmin_argkmin", k=3)
    out = run_oracle(plan, col_inputs([], m=2))  # N = 0
    np.testing.assert_array_equal(out["index"], -np.ones((2, 3)))
    assert (out["value"] == INF).all()
    assert_same(plan, out, reference_numpy(plan, col_inputs([], m=2)), 0)
    for kind, neutral in (("min", INF), ("max", -INF), ("argmin", -1)):
        o = run_oracle(column_plan(kind), col_inputs([], m=2))
        assert (next(iter(o.values())) == neutral).all()
    s = run_oracle((Vi("x") * Vj("v")).sum_over("j").to_plan("float64"), col_inputs([], m=2))
    assert (s["value"] == 0).all()
    out = run_oracle(column_plan("argkmin", k=0), col_inputs([1.0, 2.0]))
    assert out["index"].shape == (2, 0)
    out = run_oracle(column_plan("argkmin", k=2), col_inputs([1.0], m=0))
    assert out["index"].shape == (0, 2)


def test_deep_shared_formula_with_where_is_not_exponential():
    """DAG partagé profond avec ``where`` et ``min`` : construction du plan
    (``realify``, ``fold_constants``), gradient et oracle restent rapides.

    ``e_{k+1} = exp(min(e_k, y) − e_k) + sin(e_k) + x`` réutilise ``e_k``
    trois fois : l'arbre déplié a ~3^20 nœuds. Les marches non mémoïsées
    (et l'oracle sans CSE en présence de ``Piecewise``) ne finissaient pas.
    """
    import time

    import kernox as kp

    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    e = x
    for _ in range(20):
        e = kp.exp(kp.minimum(e, y) - e) * 0.5 + kp.sin(e) + x
    f = kp.where(e > 0, e, -e)
    start = time.perf_counter()
    run = f.sum_over("j").compile("float64", backend="oracle")
    X = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)
    out = run(x=X, y=torch.randn(3, 1, dtype=torch.float64))
    (g,) = torch.autograd.grad(out.sum(), X)
    assert torch.isfinite(out).all() and torch.isfinite(g).all()
    assert time.perf_counter() - start < 120
