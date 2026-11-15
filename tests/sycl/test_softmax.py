"""LOGSUMEXP / SUMSOFTMAXWEIGHT on the sycl backend (v0.5 contract), on the G31.

Modeled on tests/hip/test_softmax.py: forced j partition (slices entirely
at −inf, hence empty partial states), tiny tiles, values
1e4 / −1e4 / ±inf / NaN per row, ``patient`` variants, FP64 acc_dtype,
fp_contract on/off, summation schemes. Reference: the core FP64 Torch oracle
on the inputs rounded to the dtype.
"""

import math

import pytest
import torch

import kernox as kp
from kernox.backends.base import Sizes
from kernox.backends.sycl import _variant
from kernox.oracle import OracleReduction
from kernox.plan import output_shapes

from ._helpers import DEV, backend

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
    y[: N // 2] = INF                     # first half of j: f = −inf (empty slices)
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


def _run(plan, host, variant=None, groups=None):
    T = getattr(torch, plan.dtype)
    M = host["x"].shape[0]
    inp = {a.name: host[a.name].to(T).to(DEV) for a in plan.args}
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV)
            for k, (s, d) in output_shapes(plan, M).items()}
    cr = backend().compile(plan, DEV, variant)
    if groups is not None:
        cr.target_groups = groups
    cr.launch(inp, outs)
    torch.xpu.synchronize()
    return outs["value"].cpu(), cr


def _close(got, ref, dtype, N):
    assert got.shape == ref.shape
    fin = torch.isfinite(ref)
    assert torch.equal(torch.isnan(got), torch.isnan(ref)), (got, ref)
    assert torch.equal(got[~fin & ~torch.isnan(ref)].double(), ref[~fin & ~torch.isnan(ref)])
    u = 2.0 ** -24 if dtype == "float32" else 2.0 ** -53
    # |f| <= 1e4 + 10: term error u·|f| (rounding of f to the dtype) + summation
    tol = 4 * u * (2e4 + 20) + 8 * N * u * (1 + ref[fin].abs())
    assert torch.all((got[fin].double() - ref[fin]).abs() <= tol), (got[fin].double() - ref[fin]).abs().max()


# (j tile, work-group target): 0 = heuristics; tiles of 8 and many
# work-groups -> many slices, the first ones entirely at −inf.
SPLITS = [(0, 1), (8, None), (8, 4096), (32, 2048), (0, None)]


@pytest.mark.parametrize("scheme", ["direct", "kahan"])
@pytest.mark.parametrize("tj,groups", SPLITS, ids=lambda v: str(v))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", list(_exprs()))
def test_split_special_rows(tj, groups, dtype, name, scheme):
    plan = _exprs()[name].to_plan(dtype, sum_scheme=scheme)
    host = _inputs(37, 1500, seed=tj + (groups or 0))
    got, cr = _run(plan, host, _variant(tj=tj), groups)
    if groups and groups > 1:
        assert cr.last_split[0] > 1 and cr.last_split[2] > 0, cr.last_split
    if groups == 1:
        assert cr.last_split[0] == 1
    ref = _ref(plan, host)
    _close(got, ref, dtype, 1500)
    # coverage: x = ±1e4 finite (shift), x = +inf (inf − inf: NaN), x = −inf (all −inf), NaN
    assert torch.isfinite(ref[1]).all() and torch.isfinite(ref[2]).all() and torch.isnan(ref[3]).all()
    assert torch.isnan(ref[5]).all()
    assert (ref[4] == -INF).all() if name != "ssmw" else torch.isnan(ref[4]).all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", list(_exprs()))
def test_pos_inf_and_empty_rows(dtype, name):
    """y without +inf: x = +inf gives f = +inf everywhere (LSE +inf, SSMW NaN); N = 0, 1."""
    plan = _exprs()[name].to_plan(dtype)
    for N in (0, 1, 3000):
        host = _inputs(9, max(N, 1), seed=N)
        host["y"] = torch.linspace(-2, 2, N, dtype=torch.float64).reshape(N, 1)
        host["w"] = host["w"][:N]
        for tj, groups in ((0, 1), (8, 4096)):
            got, _ = _run(plan, host, _variant(tj=tj), groups)
            ref = _ref(plan, host)
            _close(got, ref, dtype, max(N, 1))
            if N == 0:
                assert ((got == -INF).all() if name != "ssmw" else torch.isnan(got).all())
            else:
                assert ((got[3] == INF).all() if name != "ssmw" else torch.isnan(got[3]).all())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", list(_exprs()))
def test_all_patient_variants_agree(dtype, name):
    be = backend()
    plan = _exprs()[name].to_plan(dtype)
    M, N = 300, 5000
    host = _inputs(M, N, seed=1)
    ref = _ref(plan, host)
    vs = be.variants(plan, DEV, Sizes(M, N), "patient")
    assert len(vs) >= 2
    for v in vs:
        got, cr = _run(plan, host, v)
        assert cr.info.variant == v
        _close(got, ref, dtype, N)


@pytest.mark.parametrize("scheme", ["direct", "block", "kahan"])
@pytest.mark.parametrize("name", ["lse", "ssmw"])
def test_fp32_acc_fp64(name, scheme):
    # Terms f = x − y: a single FP32 subtraction, rounded identically on the GPU and by
    # Torch on CPU. Reference: oracle of the same plan (FP32 terms, FP64 state).
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", 3)
    expr = (x - y).logsumexp(over="j") if name == "lse" else (x - y).sumsoftmaxweight(w, over="j")
    plan = expr.to_plan("float32", acc_dtype="float64", sum_scheme=scheme)
    host = _inputs(40, 2000, seed=2)
    got, _ = _run(plan, host, _variant(tj=16), 4096)
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
    g = torch.Generator().manual_seed(5)
    host = {"x": torch.randn(64, 3, generator=g, dtype=torch.float64),
            "y": torch.randn(3000, 3, generator=g, dtype=torch.float64),
            "s": torch.tensor([2.5], dtype=torch.float64)}
    u = 2.0 ** -24 if dtype == "float32" else 2.0 ** -53
    outs = []
    for fc in (True, False):
        plan = expr.to_plan(dtype, fp_contract=fc)
        got, cr = _run(plan, host)
        assert ("-ffp-contract=off" in cr.info.options) is (not fc)
        outs.append(got)
        ref = _ref(plan, host)
        assert torch.all((got.double() - ref).abs() <= 64 * u * (1 + ref.abs() + 50))
    assert torch.all((outs[0] - outs[1]).abs() <= 128 * u * (1 + outs[0].abs() + 50))


def test_supports_and_options():
    be = backend()
    plan = _exprs()["ssmw"].to_plan("float32")
    assert be.supports(plan, DEV).ok
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    assert be.supports((x - y).logsumexp(over="j").to_plan("float32", acc_dtype="float64"), DEV).ok
    cr = be.compile(plan, DEV)
    assert "sum_scheme=direct" in cr.info.options
    assert any(o.startswith("sum_block=") for o in cr.info.options)
    cr = be.compile(_exprs()["ssmw"].to_plan("float32", sum_scheme="kahan"), DEV)
    assert "sum_scheme=kahan" in cr.info.options
