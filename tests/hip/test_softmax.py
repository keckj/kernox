"""LOGSUMEXP / SUMSOFTMAXWEIGHT sur le backend hip (contrat v0.5), R9700.

Partition de j forcée (tranches entièrement à −inf, donc états partiels vides),
tuiles minuscules, valeurs 1e4 / −1e4 / ±inf / NaN par ligne, toutes les
variantes ``patient``, acc_dtype FP64, fp_contract on/off. Référence : oracle
Torch FP64 du noyau sur les entrées arrondies au dtype.
"""

import math

import pytest
import torch

import kernox as kp
from kernox.backends.base import Sizes
from kernox.backends.hip import HipBackend, HipOptions
from kernox.oracle import OracleReduction
from kernox.plan import output_shapes

pytestmark = pytest.mark.gpu
DEV = "cuda"
DTYPES = ["float32", "float64"]
INF, NAN = math.inf, math.nan


def _exprs():
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", 3)
    f = x - y
    return {"lse_vec": kp.cat(f, 2 * f - 1.0).logsumexp(over="j"),
            "ssmw": f.sumsoftmaxweight(w, over="j")}


def _inputs(M, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, 1, generator=g, dtype=torch.float64)
    x[1], x[2], x[3], x[4], x[5] = 1e4, -1e4, INF, -INF, NAN
    y = 3 * torch.randn(N, 1, generator=g, dtype=torch.float64)
    y[: N // 2] = INF                     # première moitié de j : f = −inf (tranches vides)
    w = torch.randn(N, 3, generator=g, dtype=torch.float64)
    return {"x": x, "y": y, "w": w}


def _ref(plan, host):
    p64 = plan if plan.dtype == "float64" and plan.acc_dtype == "float64" else \
        kp.builder.make_plan(plan.outputs, plan.reduction, "float64", args=plan.args)
    M = host["x"].shape[0]
    outs = {k: torch.empty(s, dtype=getattr(torch, d)) for k, (s, d) in output_shapes(p64, M).items()}
    cast = {k: host[k].to(getattr(torch, plan.dtype)).double() for k in (a.name for a in plan.args)}
    OracleReduction(p64, "cpu", 4096).launch(cast, outs)
    return outs["value"]


def _run(be, plan, host, variant=None):
    T = getattr(torch, plan.dtype)
    M = host["x"].shape[0]
    inp = {a.name: host[a.name].to(T).to(DEV) for a in plan.args}
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(plan, M).items()}
    cr = be.compile(plan, DEV, variant) if variant is not None else be.compile(plan, DEV)
    cr.launch(inp, outs)
    torch.cuda.synchronize()
    return outs["value"].cpu(), cr


def _close(got, ref, dtype, N):
    assert got.shape == ref.shape
    fin = torch.isfinite(ref)
    assert torch.equal(torch.isnan(got), torch.isnan(ref)), (got, ref)
    assert torch.equal(got[~fin & ~torch.isnan(ref)], ref[~fin & ~torch.isnan(ref)])
    u = 2.0 ** -24 if dtype == "float32" else 2.0 ** -53
    # |f| <= 1e4 + 10 : erreur des termes u·|f| (arrondi de f dans le dtype) + sommation
    tol = 4 * u * (2e4 + 20) + 8 * N * u * (1 + ref[fin].abs())
    assert torch.all((got[fin].double() - ref[fin]).abs() <= tol), (got[fin] - ref[fin]).abs().max()


SPLIT_OPTS = [dict(split="off"), dict(split="force", tile_j=8), dict(split="force", tile_j=8, jchunk=24),
              dict(split="force", tile_j=32, jchunk=40), dict(split="auto"),
              # insertion à un niveau (état de ligne direct) et à deux niveaux forcés
              dict(split="off", tile_j=8, soft_levels=1), dict(split="force", tile_j=8, jchunk=24, soft_levels=1),
              dict(split="off", tile_j=8, soft_levels=2)]


@pytest.mark.parametrize("opts", SPLIT_OPTS, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", list(_exprs()))
def test_split_special_rows(opts, dtype, name):
    plan = _exprs()[name].to_plan(dtype)
    host = _inputs(37, 150, seed=len(opts))
    got, cr = _run(HipBackend(HipOptions(**opts)), plan, host)
    if opts.get("split") == "force":
        assert cr.last_launch["S"] > 1 and cr.last_launch["temp_bytes"] > 0
    _close(got, _ref(plan, host), dtype, 150)
    # couverture : x = ±1e4 finis (décalage), x = +inf (inf − inf : NaN), x = −inf (tous −inf), NaN
    ref = _ref(plan, host)
    assert torch.isfinite(ref[1]).all() and torch.isfinite(ref[2]).all() and torch.isnan(ref[3]).all()
    assert torch.isnan(ref[5]).all()
    assert (ref[4] == -INF).all() if name != "ssmw" else torch.isnan(ref[4]).all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", list(_exprs()))
def test_pos_inf_and_empty_rows(dtype, name):
    """y sans +inf : x = +inf donne f = +inf partout (LSE +inf, SSMW NaN) ; N = 0."""
    plan = _exprs()[name].to_plan(dtype)
    for N in (0, 1, 300):
        host = _inputs(9, max(N, 1), seed=N)
        host["y"] = torch.linspace(-2, 2, N, dtype=torch.float64).reshape(N, 1)
        host["w"] = host["w"][:N]
        for opts in (dict(split="off"), dict(split="force", tile_j=8), dict(split="force", tile_j=8, soft_levels=1)):
            got, _ = _run(HipBackend(HipOptions(**opts)), plan, host)
            ref = _ref(plan, host)
            _close(got, ref, dtype, max(N, 1))
            if N == 0:
                assert ((got == -INF).all() if name != "ssmw" else torch.isnan(got).all())
            else:
                assert ((got[3] == INF).all() if name != "ssmw" else torch.isnan(got[3]).all())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", list(_exprs()))
def test_all_patient_variants_agree(dtype, name):
    be = HipBackend()
    plan = _exprs()[name].to_plan(dtype)
    M, N = 300, 5000
    host = _inputs(M, N, seed=1)
    ref = _ref(plan, host)
    vs = be.variants(plan, DEV, Sizes(M, N), "patient")
    assert len(vs) >= 2
    for v in vs:
        got, cr = _run(be, plan, host, v)
        assert cr.info.variant == v
        _close(got, ref, dtype, N)


@pytest.mark.parametrize("name", ["lse", "ssmw"])
def test_fp32_acc_fp64(name):
    # Termes f = x − y : une seule soustraction FP32, arrondie à l'identique sur GPU et par
    # Torch CPU. Référence : oracle du même plan (termes FP32, état FP64) ; seule l'erreur
    # d'accumulation FP64 subsiste (une accumulation FP32 serait ~1e-7 relatif).
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", 3)
    expr = (x - y).logsumexp(over="j") if name == "lse" else (x - y).sumsoftmaxweight(w, over="j")
    plan = expr.to_plan("float32", acc_dtype="float64")
    host = _inputs(40, 2000, seed=2)
    got, _ = _run(HipBackend(HipOptions(split="force", tile_j=16)), plan, host)
    assert got.dtype == torch.float64
    M = host["x"].shape[0]
    ref = torch.empty(output_shapes(plan, M)["value"][0], dtype=torch.float64)
    OracleReduction(plan, "cpu", 4096).launch({a.name: host[a.name].float() for a in plan.args}, {"value": ref})
    fin = torch.isfinite(ref)
    assert torch.equal(torch.isnan(got), torch.isnan(ref))
    assert torch.all((got[fin] - ref[fin]).abs() <= 1e-12 * (1 + ref[fin].abs()))


@pytest.mark.parametrize("dtype", DTYPES)
def test_fp_contract_modes(dtype):
    x, y, s = kp.Vi("x", 3), kp.Vj("y", 3), kp.Param("s")
    expr = (-((x - y) ** 2).sum() * s).logsumexp(over="j")
    host = {"x": torch.randn(64, 3, dtype=torch.float64), "y": torch.randn(3000, 3, dtype=torch.float64),
            "s": torch.tensor([2.5], dtype=torch.float64)}
    outs = []
    for fc in (True, False):
        plan = expr.to_plan(dtype, fp_contract=fc)
        got, cr = _run(HipBackend(), plan, host)
        assert ("-ffp-contract=on" if fc else "-ffp-contract=off") in cr.info.options
        outs.append(got)
        ref = _ref(plan, host)
        u = 2.0 ** -24 if dtype == "float32" else 2.0 ** -53
        assert torch.all((got.double() - ref).abs() <= 64 * u * (1 + ref.abs() + 50))
    u = 2.0 ** -24 if dtype == "float32" else 2.0 ** -53
    assert torch.all((outs[0] - outs[1]).abs() <= 128 * u * (1 + outs[0].abs() + 50))


def test_default_variant_and_coop_refused():
    be = HipBackend()
    plan = _exprs()["ssmw"].to_plan("float32")
    assert be.default_variant(plan, DEV, Sizes(100, 10**6)).name == "jsplit"
    assert be.default_variant(plan, DEV, Sizes(10**6, 10**6)).name == "row"
    assert not HipBackend(HipOptions(strategy="coop")).supports(plan, DEV).ok


@pytest.mark.parametrize("dtype,acc,E,levels", [("float32", None, 3, 2), ("float32", None, 64, 1),
                                                ("float64", None, 3, 1), ("float32", "float64", 3, 1)])
def test_default_levels_and_variants(dtype, acc, E, levels):
    """Règle du niveau d'insertion (FP64 ou état > 256 o : un niveau) ; l'autre niveau est
    proposé en ``measure`` et en ``patient`` ; les deux niveaux concordent."""
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", E)
    plan = (x - y).sumsoftmaxweight(w, over="j").to_plan(dtype, **({"acc_dtype": acc} if acc else {}))
    be = HipBackend()
    base = be.default_variant(plan, DEV, Sizes(300, 5000))
    assert dict(base.params)["levels"] == levels
    for effort in ("measure", "patient"):
        vs = be.variants(plan, DEV, Sizes(300, 5000), effort)
        assert {dict(v.params)["levels"] for v in vs} == {1, 2}
    host = _inputs(300, 5000, seed=3)
    host["w"] = torch.randn(5000, E, dtype=torch.float64)
    outs = [_run(HipBackend(HipOptions(soft_levels=lv, split="off")), plan, host)[0] for lv in (1, 2)]
    assert torch.equal(torch.isnan(outs[0]), torch.isnan(outs[1]))
    fin = torch.isfinite(outs[0])
    u = 2.0 ** -53 if (acc or dtype) == "float64" else 2.0 ** -24
    assert torch.all((outs[0][fin] - outs[1][fin]).abs().double() <= 64 * 5000 * u * (1 + outs[0][fin].abs().double()))


def test_levels_option_validated():
    with pytest.raises(Exception):
        HipBackend(HipOptions(soft_levels=3)).compile(_exprs()["ssmw"].to_plan("float32"), DEV)
