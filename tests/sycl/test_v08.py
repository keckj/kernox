"""MIN_ARGMIN, MAX_ARGMAX and LOGSUMEXPWEIGHT (v0.8 amendment) on the sycl backend, on the G31.

- MIN_ARGMIN / MAX_ARGMAX: bit-exact equality with MIN + ARGMIN (resp. MAX +
  ARGMAX) of the same backend, themselves checked by the conformance suite;
  integer data (many ties), NaN, rows entirely NaN, N = 0, forced j partition,
  variants, batch and ranges;
- LOGSUMEXPWEIGHT: the core FP64 Torch oracle on the inputs rounded to the
  dtype; special rows (±1e4, ±inf, NaN), zero and negative weights, j partition
  with empty slices, summation schemes, FP64 acc_dtype.
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
SPLITS = [(0, 1), (8, 4096), (32, 2048), (0, None)]


def _launch(plan, host, variant=None, groups=None):
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
    return {k: v.cpu() for k, v in outs.items()}, cr


# ------------------------------------------------------------- MIN_ARGMIN / MAX_ARGMAX

def _int_inputs(M, N, D, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-3, 4, (M, D), generator=g).double()
    y = torch.randint(-3, 4, (N, D), generator=g).double()
    if N > 7:
        y[7, 0] = NAN
    x[2, :] = NAN  # row entirely NaN: (±inf, −1)
    return {"x": x, "y": y}


def _pairs(is_min):
    x, y = kp.Vi("x", 2), kp.Vj("y", 2)
    f = x * y
    both = (f.min_argmin if is_min else f.max_argmax)(over="j")
    val = (f.min if is_min else f.max)(over="j")
    idx = (f.argmin if is_min else f.argmax)(over="j")
    return both, val, idx


@pytest.mark.parametrize("tj,groups", SPLITS, ids=str)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_min", [True, False])
def test_min_argmin_equals_separate(is_min, dtype, tj, groups):
    both, val, idx = _pairs(is_min)
    for N in (0, 1, 2500):
        host = _int_inputs(70, N, 2, seed=N + tj)
        got, cr = _launch(both.to_plan(dtype), host, _variant(tj=tj), groups)
        v, _ = _launch(val.to_plan(dtype), host, _variant(tj=tj), groups)
        i, _ = _launch(idx.to_plan(dtype), host, _variant(tj=tj), groups)
        assert got["value"].dtype == getattr(torch, dtype) and got["index"].dtype == torch.int64
        assert torch.equal(got["value"], v["value"]) and torch.equal(got["index"], i["index"])
        neutral = INF if is_min else -INF
        assert (got["value"][2] == neutral).all() and (got["index"][2] == -1).all()
        if N == 0:
            assert (got["index"] == -1).all() and (got["value"] == neutral).all()
        if groups and groups > 1 and N > 1000:
            assert cr.last_split[0] > 1


@pytest.mark.parametrize("is_min", [True, False])
def test_min_argmin_patient_variants(is_min):
    both, val, idx = _pairs(is_min)
    plan = both.to_plan("float32")
    host = _int_inputs(300, 4000, 2, seed=3)
    ref_v, _ = _launch(val.to_plan("float32"), host)
    ref_i, _ = _launch(idx.to_plan("float32"), host)
    vs = backend().variants(plan, DEV, Sizes(300, 4000), "patient")
    assert len(vs) >= 2
    for v in vs:
        got, _ = _launch(plan, host, v)
        assert torch.equal(got["value"], ref_v["value"]) and torch.equal(got["index"], ref_i["index"])


def test_min_argmin_batch_and_ranges_runtime():
    both, _, _ = _pairs(True)
    g = torch.Generator().manual_seed(4)
    X = torch.randint(-3, 4, (3, 1, 130, 2), generator=g).float()
    Y = torch.randint(-3, 4, (1, 2, 900, 2), generator=g).float()
    run = kp.compile(both, dtype="float32", backend="sycl")
    ref = kp.compile(both, dtype="float32", backend="oracle")
    v, i = run(x=X.to(DEV), y=Y.to(DEV))
    rv, ri = ref(x=X, y=Y)
    assert torch.equal(v.cpu(), rv) and torch.equal(i.cpu(), ri)
    # ranges: labels (3 classes), kept blocks are diagonal
    x2, y2 = X[0, 0], Y[0, 0]
    li = torch.arange(130) * 3 // 130
    lj = torch.arange(900) * 3 // 900
    keep = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=torch.bool)
    r_dev = kp.ranges_from_labels(li.to(DEV), lj.to(DEV), keep.to(DEV))
    r_cpu = kp.ranges_from_labels(li, lj, keep)
    v, i = run.with_ranges(r_dev)(x=x2.to(DEV), y=y2.to(DEV))
    rv, ri = ref.with_ranges(r_cpu)(x=x2, y=y2)
    assert torch.equal(v.cpu(), rv) and torch.equal(i.cpu(), ri)


# ------------------------------------------------------------- LOGSUMEXPWEIGHT

def _lsew_expr():
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", 3)
    return (x - y).logsumexp(over="j", weight=w)


def _lsew_inputs(M, N, seed=0, negative=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, 1, generator=g, dtype=torch.float64)
    x[1], x[2], x[3], x[4], x[5] = 1e4, -1e4, INF, -INF, NAN
    y = 3 * torch.randn(N, 1, generator=g, dtype=torch.float64)
    y[: N // 2] = INF                     # first half of j: f = −inf (empty slices)
    w = torch.rand(N, 3, generator=g, dtype=torch.float64) + 0.1
    w[: N // 2, 0] = NAN                  # NaN g with zero weight: ignored
    if negative:
        w[:, 2] -= 0.8                    # Σ g < 0 possible: NaN (log)
    return {"x": x, "y": y, "w": w}


def _ref(plan, host):
    p64 = plan if plan.dtype == "float64" and plan.acc_dtype == "float64" else \
        kp.builder.make_plan(plan.outputs, plan.reduction, "float64", args=plan.args)
    M = host["x"].shape[0]
    outs = {k: torch.empty(s, dtype=getattr(torch, d)) for k, (s, d) in output_shapes(p64, M).items()}
    cast = {k: host[k].to(getattr(torch, plan.dtype)).double() for k in (a.name for a in plan.args)}
    OracleReduction(p64, "cpu", 4096).launch(cast, outs)
    return outs["value"]


def _close(got, ref, dtype, N):
    assert got.shape == ref.shape
    fin = torch.isfinite(ref)
    assert torch.equal(torch.isnan(got), torch.isnan(ref)), (got, ref)
    assert torch.equal(got[~fin & ~torch.isnan(ref)].double(), ref[~fin & ~torch.isnan(ref)])
    u = 2.0 ** -24 if dtype == "float32" else 2.0 ** -53
    tol = 4 * u * (2e4 + 20) + 8 * N * u * (1 + ref[fin].abs())
    assert torch.all((got[fin].double() - ref[fin]).abs() <= tol), (got[fin].double() - ref[fin]).abs().max()


@pytest.mark.parametrize("scheme", ["direct", "block", "kahan"])
@pytest.mark.parametrize("tj,groups", SPLITS, ids=str)
@pytest.mark.parametrize("dtype", DTYPES)
def test_lsew_split_special_rows(dtype, tj, groups, scheme):
    plan = _lsew_expr().to_plan(dtype, sum_scheme=scheme)
    host = _lsew_inputs(37, 1500, seed=tj + (groups or 0))
    got, cr = _launch(plan, host, _variant(tj=tj), groups)
    got = got["value"]
    if groups and groups > 1:
        assert cr.last_split[0] > 1
    ref = _ref(plan, host)
    _close(got, ref, dtype, 1500)
    # x = +inf: inf − inf = NaN on the half where y = +inf; x = −inf: −inf; NaN: NaN
    assert torch.isnan(ref[3]).all() and (ref[4] == -INF).all() and torch.isnan(ref[5]).all()
    assert torch.isfinite(ref[1]).all() and torch.isfinite(ref[2]).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_lsew_negative_weights_and_empty(dtype):
    plan = _lsew_expr().to_plan(dtype)
    host = _lsew_inputs(40, 2000, seed=9, negative=True)
    for tj, groups in ((0, 1), (8, 4096)):
        got, _ = _launch(plan, host, _variant(tj=tj), groups)
        ref = _ref(plan, host)
        assert torch.isnan(ref[:, 2]).any()
        _close(got["value"], ref, dtype, 2000)
    # f = +inf without NaN: +inf if t_e > 0, NaN if t_e <= 0 (negative weights)
    hp = {"x": torch.tensor([[INF], [0.0]], dtype=torch.float64), "y": torch.tensor([[0.0], [1.0]], dtype=torch.float64),
          "w": torch.tensor([[1.0, -1.0, 0.5], [2.0, 0.5, -0.5]], dtype=torch.float64)}
    got, _ = _launch(plan, hp)
    v = got["value"]
    assert v[0, 0] == INF and torch.isnan(v[0, 1]) and torch.isnan(v[0, 2])  # t = 3, −0.5, 0
    _close(v, _ref(plan, hp), dtype, 2)
    # N = 0: −inf everywhere
    host0 = {"x": host["x"], "y": host["y"][:0], "w": host["w"][:0]}
    got, _ = _launch(plan, host0)
    assert (got["value"] == -INF).all()


def test_lsew_acc_fp64_and_options():
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", 3)
    plan = (x - y).logsumexp(over="j", weight=w).to_plan("float32", acc_dtype="float64", sum_scheme="kahan")
    assert backend().supports(plan, DEV).ok
    host = _lsew_inputs(40, 2000, seed=2)
    got, cr = _launch(plan, host, _variant(tj=16), 4096)
    assert "sum_scheme=kahan" in cr.info.options
    got = got["value"]
    assert got.dtype == torch.float64
    ref = torch.empty(output_shapes(plan, 40)["value"][0], dtype=torch.float64)
    OracleReduction(plan, "cpu", 4096).launch({a.name: host[a.name].float() for a in plan.args}, {"value": ref})
    fin = torch.isfinite(ref)
    assert torch.equal(torch.isnan(got), torch.isnan(ref))
    assert torch.all((got[fin] - ref[fin]).abs() <= 1e-12 * (1 + ref[fin].abs()))


def test_lsew_batch_and_ranges_runtime():
    expr = _lsew_expr()
    g = torch.Generator().manual_seed(6)
    X = torch.randn(2, 1, 90, 1, generator=g, dtype=torch.float64)
    Y = torch.randn(1, 3, 700, 1, generator=g, dtype=torch.float64)
    Wt = torch.rand(1, 3, 700, 3, generator=g, dtype=torch.float64) + 0.1
    run = kp.compile(expr, dtype="float64", backend="sycl")
    ref = kp.compile(expr, dtype="float64", backend="oracle")
    got = run(x=X.to(DEV), y=Y.to(DEV), w=Wt.to(DEV)).cpu()
    torch.testing.assert_close(got, ref(x=X, y=Y, w=Wt), rtol=1e-12, atol=1e-12)
    li = torch.arange(90) * 3 // 90
    lj = torch.arange(700) * 3 // 700
    keep = torch.tensor([[1, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=torch.bool)  # class 2: no pair
    got = run.with_ranges(kp.ranges_from_labels(li.to(DEV), lj.to(DEV), keep.to(DEV)))(
        x=X[0, 0].to(DEV), y=Y[0, 0].to(DEV), w=Wt[0, 0].to(DEV)).cpu()
    exp = ref.with_ranges(kp.ranges_from_labels(li, lj, keep))(x=X[0, 0], y=Y[0, 0], w=Wt[0, 0])
    torch.testing.assert_close(got, exp, rtol=1e-12, atol=1e-12)
    assert (got[li == 2] == -INF).all()
