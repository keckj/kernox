"""Schémas de sommation (avenant v0.7) : plan, clé, compile, oracle, gradients (CPU)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import kernox as kp  # noqa: E402
from kernox import Vi, Vj  # noqa: E402
from kernox.backends.base import Support  # noqa: E402
from kernox.errors import UnsupportedError, ValidationError  # noqa: E402
from kernox.oracle import SUM_BLOCK, _Accumulator  # noqa: E402
from kernox.runtime import register_backend  # noqa: E402


def _red():
    return (Vj("w", 1) * Vi("one", 1)).sum_over("j")


def _call(run, values, dtype=torch.float32):
    w = torch.tensor(values, dtype=dtype)[:, None]
    return run(one=torch.ones(1, 1, dtype=dtype), w=w)[0, 0].item()


def test_plan_field_key_and_validation():
    base = _red().to_plan("float32")
    assert base.sum_scheme == "direct"
    keys = {s: _red().to_plan("float32", sum_scheme=s).key() for s in ("direct", "block", "kahan")}
    assert keys["direct"] == base.key() and len(set(keys.values())) == 3
    with pytest.raises(ValidationError, match="sum_scheme invalide"):
        kp.compile(_red(), dtype="float32", backend="oracle", sum_scheme="pairwise")
    plan = _red().to_plan("float32", sum_scheme="kahan")
    assert kp.compile(plan, dtype="float32", backend="oracle").plan.sum_scheme == "kahan"
    with pytest.raises(ValidationError, match="sum_scheme incohérent"):
        kp.compile(plan, dtype="float32", backend="oracle", sum_scheme="direct")


def test_oracle_simulates_each_scheme():
    # 1e8 + 1 + 1 + 1 + 1 − 1e8 en FP32 : direct séquentiel perd les 1, kahan les garde
    vals = [1e8, 1.0, 1.0, 1.0, 1.0, -1e8]
    got = {s: _call(kp.compile(_red(), dtype="float32", backend="oracle", sum_scheme=s), vals)
           for s in ("direct", "block", "kahan")}
    assert got == {"direct": 0.0, "block": 0.0, "kahan": 4.0}
    # block : tuile neuve tous les SUM_BLOCK termes. 1e8 seul dans la première
    # tuile, puis SUM_BLOCK uns : direct les perd un à un (ulp(1e8) = 8), block
    # somme la tuile exactement (256) avant de l'ajouter au total.
    vals = [1e8] + [0.0] * (SUM_BLOCK - 1) + [1.0] * SUM_BLOCK
    got = {s: _call(kp.compile(_red(), dtype="float32", backend="oracle", sum_scheme=s), vals)
           for s in ("direct", "block", "kahan")}
    assert got == {"direct": 1e8, "block": 1e8 + SUM_BLOCK, "kahan": 1e8 + SUM_BLOCK}
    run = kp.compile(_red(), dtype="float32", backend="oracle", sum_scheme="block")
    assert f"sum_block={SUM_BLOCK}" in run.compiled(torch.device("cpu")).info.options


def test_acc_float64_and_float64_plans():
    vals = [1e8, 1.0, -1e8]
    for s in ("direct", "block", "kahan"):
        r = kp.compile(_red(), dtype="float32", backend="oracle", sum_scheme=s, acc_dtype="float64")
        assert _call(r, vals) == 1.0
    vals = [1e17, 1.0, 1.0, -1e17]
    d = kp.compile(_red(), dtype="float64", backend="oracle", sum_scheme="direct")
    k = kp.compile(_red(), dtype="float64", backend="oracle", sum_scheme="kahan")
    assert _call(d, vals, torch.float64) == 0.0 and _call(k, vals, torch.float64) == 2.0


def test_kahan_special_values():
    k = kp.compile(_red(), dtype="float32", backend="oracle", sum_scheme="kahan")
    assert _call(k, [float("inf"), 1.0, 2.0]) == float("inf")      # pas de inf − inf
    assert _call(k, [-float("inf"), 1.0]) == -float("inf")
    assert np.isnan(_call(k, [float("inf"), -float("inf")]))
    assert np.isnan(_call(k, [1.0, float("nan"), 2.0]))
    for s_ in ("direct", "block", "kahan"):  # N = 0 : neutre 0
        r = kp.compile(_red(), dtype="float32", backend="oracle", sum_scheme=s_)
        assert r(one=torch.ones(2, 1), w=torch.ones(0, 1)).tolist() == [[0.0], [0.0]]


def test_accumulator_merge_matches_sequential_neumaier():
    rng = np.random.default_rng(0)
    v = (rng.standard_normal((3, 3 * SUM_BLOCK + 5, 2)) * 10.0 ** rng.integers(-3, 8, (3, 3 * SUM_BLOCK + 5, 2)))
    v = v.astype(np.float64)
    acc = _Accumulator("kahan", (3, 2), np.float64)
    acc.add(v[:, :100])
    acc.add(v[:, 100:])
    import math

    for r in range(3):
        for c in range(2):
            assert abs(acc.result()[r, c] - math.fsum(v[r, :, c])) <= 4 * 2**-53 * np.abs(v[r, :, c]).sum()


def test_scheme_ignored_for_non_summing_reductions():
    x, y = Vi("x", 2), Vj("y", 2)
    d2 = ((x - y) ** 2).sum()
    X, Y = torch.randn(5, 2), torch.randn(7, 2)
    for red in (d2.min(over="j"), d2.argkmin(3, over="j"), (d2 < 1.0).sum_over("j")):
        a = kp.compile(red, dtype="float32", backend="oracle")(x=X, y=Y)
        b = kp.compile(red, dtype="float32", backend="oracle", sum_scheme="kahan")(x=X, y=Y)
        assert torch.equal(a, b)


def test_logsumexp_and_ssmw_schemes_agree_on_easy_data():
    x, y = Vi("x", 2), Vj("y", 2)
    d2 = ((x - y) ** 2).sum()
    X, Y = torch.randn(5, 2, dtype=torch.float64), torch.randn(700, 2, dtype=torch.float64)
    for red in ((-d2).logsumexp(over="j"), (-d2).sumsoftmaxweight(y, over="j")):
        outs = [kp.compile(red, dtype="float64", backend="oracle", sum_scheme=s)(x=X, y=Y)
                for s in ("direct", "block", "kahan")]
        for o in outs[1:]:
            torch.testing.assert_close(o, outs[0], rtol=1e-13, atol=1e-14)


def test_gradient_uses_forward_sum_scheme():
    x, y, b = Vi("x", 3), Vj("y", 3), Vj("b", 1)
    red = (kp.exp(-((x - y) ** 2).sum()) * b).sum_over("j")
    run = kp.compile(red, dtype="float64", backend="oracle", sum_scheme="kahan")
    gp, i_red, j_red = run._gradient(frozenset({"x", "y", "b"}))
    assert gp.i_plan.sum_scheme == "kahan" and gp.j_plan.sum_scheme == "kahan"
    assert i_red.plan.sum_scheme == "kahan" and j_red.plan.sum_scheme == "kahan"
    X = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    Y = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    B = torch.randn(6, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda X, Y, B: run(x=X, y=Y, b=B), (X, Y, B))


class _DirectOnly:
    name = "directonly"

    def __init__(self, honest):
        self.honest = honest

    def supports(self, plan, device):
        if self.honest and plan.sum_scheme != "direct":
            return Support(False, f"sum_scheme={plan.sum_scheme!r} non implémenté")
        return Support(True)

    def compile(self, plan, device):  # pragma: no cover - jamais atteint
        raise AssertionError("ne doit pas compiler")


@pytest.mark.parametrize("honest", [True, False])
def test_backend_without_scheme_is_refused_never_direct(honest):
    name = f"directonly_{honest}"
    register_backend(name, lambda: _DirectOnly(honest), replace=True)
    run = kp.compile(_red(), dtype="float32", backend=name, sum_scheme="kahan")
    with pytest.raises(UnsupportedError, match="sum_scheme"):
        run(one=torch.ones(2, 1), w=torch.ones(3, 1))
    plan = dataclasses.replace(_red().to_plan("float32"), sum_scheme="block")
    assert plan.sum_scheme == "block"
