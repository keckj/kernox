"""Native FP16 (amendment v0.9) on the hip backend: ``kp_h`` type (``_Float16``, each
operation rounded to binary16), FP32 accumulator by default, ``acc_dtype="float16"``
with ``block`` by default, FP16 outputs.

The terms are single-operation formulas (``x * y``, ``x - y``), whose FP16 rounding
does not depend on evaluation order: MIN/MAX/ARG*/KMIN are compared bitwise to the
oracle, sums against an accumulation error bound. FP16 arithmetic is proven by
overflow (``x·y/1000`` with ``x·y > 65504``), the final conversion by an FP32 sum
that exceeds 65504.
"""

import dataclasses
import math

import pytest
import torch

import kernox as kp
from kernox.backends.base import BatchLayout, RangesLayout
from kernox.backends.hip import HipBackend, HipOptions
from kernox.oracle import OracleReduction
from kernox.plan import output_shapes
from kernox.ranges import Triplet, work_list

pytestmark = pytest.mark.gpu
DEV = "cuda"
U16 = 2.0 ** -11
U32 = 2.0 ** -24
INF = math.inf
SPLITS = [dict(split="off"), dict(split="force", tile_j=8, jchunk=24), dict(split="auto")]


def _sym():
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", 2)
    return x, y, w, x * y


def _exprs():
    x, y, w, f = _sym()
    return {
        "min": f.min(over="j"), "max": f.max(over="j"), "argmin": f.argmin(over="j"),
        "argmax": f.argmax(over="j"), "min_argmin": f.min_argmin(over="j"),
        "kmin_argkmin5": f.kmin_argkmin(5, over="j"), "kmin_argkmin64": f.kmin_argkmin(64, over="j"),
    }


def _data(M, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-20, 21, (M, 1), generator=g).to(torch.float16) / 4
    y = torch.randn(N, 1, generator=g).to(torch.float16)
    w = torch.randn(N, 2, generator=g).to(torch.float16)
    if N > 5:
        y[5, 0] = math.nan
    return {"x": x, "y": y, "w": w}


def _run(be, plan, host, M):
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(plan, M).items()}
    cr = be.compile(plan, DEV)
    cr.launch({a.name: host[a.name].to(DEV) for a in plan.args}, outs)
    torch.cuda.synchronize()
    return {k: v.cpu() for k, v in outs.items()}, cr


def _oracle(plan, host, M):
    outs = {k: torch.empty(s, dtype=getattr(torch, d)) for k, (s, d) in output_shapes(plan, M).items()}
    OracleReduction(plan, "cpu", 4096).launch({a.name: host[a.name] for a in plan.args}, outs)
    return outs


@pytest.mark.parametrize("opts", SPLITS, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))
@pytest.mark.parametrize("name", list(_exprs()))
def test_selection_bitwise(opts, name):
    plan = _exprs()[name].to_plan("float16")
    assert plan.acc_dtype == "float16"            # no accumulator: FP16
    host = _data(61, 900, seed=len(opts))
    got, cr = _run(HipBackend(HipOptions(**opts)), plan, host, 61)
    if name == "kmin_argkmin64":
        assert cr.config.strategy == "coop"
    ref = _oracle(plan, host, 61)
    for k in ref:
        assert got[k].dtype == ref[k].dtype, k
        assert torch.equal(got[k].nan_to_num(), ref[k].nan_to_num()), (name, k)


def _sum_l1(host):
    """Σ_j |x_i·y_j| (M, 1)."""
    return (host["x"].double() * host["y"].double().T).abs().sum(1, keepdim=True)


@pytest.mark.parametrize("opts", SPLITS, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))
@pytest.mark.parametrize("acc", ["float32", "float16"])
def test_sum_accumulators(opts, acc):
    x, y, _w, f = _sym()
    plan = f.sum_over("j").to_plan("float16", acc_dtype=acc)
    assert plan.sum_scheme == ("block" if acc == "float16" else "direct")
    host = _data(61, 900, seed=1)
    host["y"][5, 0] = 0.5
    got, cr = _run(HipBackend(HipOptions(**opts)), plan, host, 61)
    assert got["value"].dtype == torch.float16
    assert f"sum_scheme={plan.sum_scheme}" in cr.info.options
    # x·y terms exact to within 1/2 FP16 ulp; accumulation: block (FP16 acc) ~ (T + N/T)·u16,
    # direct (FP32 acc) N·u32; final conversion u16·|result|
    r64 = (host["x"].double() * host["y"].double().T).sum(1, keepdim=True)
    l1, N = _sum_l1(host), host["y"].shape[0]
    T = cr.config.tile_j
    k = (T + N / T) * U16 if acc == "float16" else N * U32
    tol = (k + U16) * l1 + U16 * r64.abs() + 1e-7
    assert torch.all((got["value"].double() - r64).abs() <= tol)
    if acc == "float32":
        # same FP32 accumulator as the oracle: at most one FP16 ulp of the result apart
        ref = _oracle(plan, host, 61)["value"].double()
        ulp = torch.clamp(ref.abs(), min=2.0 ** -14) * 2.0 ** -10
        assert torch.all((got["value"].double() - ref).abs() <= ulp + N * U32 * l1)


def test_fp16_compute_proven_by_overflow():
    """x·y = 90000 > 65504: +inf in FP16, hence sqrt(x·y) = +inf (300 if computed in FP32).
    (``x·y/1000`` does not work: symox rewrites it as ``0.001·x·y``.)"""
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    plan = kp.sqrt(x * y).max(over="j").to_plan("float16")
    host = {"x": torch.tensor([[300.0], [2.0]], dtype=torch.float16),
            "y": torch.tensor([[300.0], [1.0]], dtype=torch.float16)}
    got, _ = _run(HipBackend(), plan, host, 2)
    assert torch.equal(got["value"], _oracle(plan, host, 2)["value"])
    assert got["value"][0, 0] == INF


def test_sum_output_overflow_and_acc_fp32():
    """FP32 accumulation: 3 × 30000 = 90000 → ±inf at the final conversion; 3 × 20000 =
    60000 stays finite (representable). FP16 accumulation: 30000 + 30000 + 30000 overflows
    during the sum."""
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    plan = (x * y).sum_over("j").to_plan("float16")
    host = {"x": torch.tensor([[30000.0], [20000.0], [-30000.0]], dtype=torch.float16),
            "y": torch.ones(4, 1, dtype=torch.float16)}
    host["y"][3, 0] = 0.0
    got, _ = _run(HipBackend(HipOptions(split="off")), plan, host, 3)
    v = got["value"][:, 0]
    assert v[0] == INF and v[1] == 60000.0 and v[2] == -INF
    # FP16 acc: the partial sum 30000 + 30000 already overflows
    p16 = (x * y).sum_over("j").to_plan("float16", acc_dtype="float16")
    got16, _ = _run(HipBackend(HipOptions(split="off")), p16, host, 3)
    assert got16["value"][0, 0] == INF


@pytest.mark.parametrize("acc", ["float32", "float16"])
@pytest.mark.parametrize("name", ["lse", "ssmw", "lsew"])
def test_soft_family(acc, name):
    x, y, w, _f = _sym()
    f = x - y
    expr = {"lse": f.logsumexp(over="j"), "ssmw": f.sumsoftmaxweight(w, over="j"),
            "lsew": f.logsumexp(over="j", weight=kp.abs(w) + 0.5)}[name]
    plan = expr.to_plan("float16", acc_dtype=acc)
    host = _data(37, 400, seed=2)
    host["y"][5, 0] = 0.25
    for opts in SPLITS:
        got, _ = _run(HipBackend(HipOptions(**opts)), plan, host, 37)
        assert got["value"].dtype == torch.float16
        p64 = kp.builder.make_plan(plan.outputs, plan.reduction, "float64", args=plan.args)
        ref = _oracle(p64, {k: v.double() for k, v in host.items()}, 37)["value"]
        # terms: f exact to within 1/2 FP16 ulp (|f| <= 9); state in acc_dtype
        u = U16 if acc == "float16" else U32
        tol = 9 * U16 * 4 + 400 * u * 4 * (1 + ref.abs()) + U16 * ref.abs()
        assert torch.all((got["value"].double() - ref).abs() <= tol), (name, acc, opts)


def test_batch_and_ranges_fp16():
    x, y, w, f = _sym()
    be = HipBackend(HipOptions(block=64))
    for expr in (f.kmin_argkmin(5, over="j"), f.sum_over("j")):
        plan = expr.to_plan("float16")
        host = _data(150, 700, seed=3)
        host["y"][5, 0] = 1.0
        # batch: 3 elements = 3 non-batched calls, bitwise
        xs = torch.stack([host["x"], -host["x"], 2 * host["x"]])
        ys = torch.stack([host["y"], host["y"] / 2, -host["y"]])
        pb = dataclasses.replace(plan, batch_rank=1)
        cr = be.compile(pb, DEV)
        offs = {"x": torch.arange(3) * 150, "y": torch.arange(3) * 700}
        outs = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(pb, 150, (3,)).items()}
        cr.launch({"x": xs.to(DEV), "y": ys.to(DEV)}, outs,
                  batch=BatchLayout((3,), 3, {k: v.to(DEV) for k, v in offs.items()}, offs))
        torch.cuda.synchronize()
        for b in range(3):
            one, _ = _run(be, plan, {"x": xs[b], "y": ys[b]}, 150)
            for k in one:
                assert torch.equal(outs[k][b].cpu(), one[k]), (b, k)
        # dense ranges = call without ranges, bitwise
        r = kp.Ranges.dense(150, 700, DEV)
        tr = Triplet(r.ranges_i, r.slices_i, r.redranges_j)
        crr = be.compile(dataclasses.replace(plan, ranges=True), DEV)
        work = work_list(tr, crr.ranges_tile_m)
        red = tr.red.to(torch.int64).contiguous()
        o2 = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(plan, 150).items()}
        crr.launch({"x": host["x"].to(DEV), "y": host["y"].to(DEV)}, o2,
                   ranges=RangesLayout(work, red, crr.ranges_tile_m, work.cpu(), red.cpu()))
        torch.cuda.synchronize()
        one, _ = _run(HipBackend(HipOptions(block=64, split="off")), plan, host, 150)
        for k in one:
            if k == "value" and plan.reduction.kind.value == "sum":
                assert torch.allclose(o2[k].cpu().float(), one[k].float(), rtol=2e-3, atol=1e-2)
            else:
                assert torch.equal(o2[k].cpu(), one[k]), k


def test_fp16_frontend_matches_oracle():
    x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 2)
    d2 = (x - y).sqnorm2()
    g = torch.Generator().manual_seed(0)
    X, Y = torch.randn(200, 3, generator=g).half(), torch.randn(300, 3, generator=g).half()
    for expr in (d2.argkmin(4, over="j"), d2.min(over="j")):
        out = kp.compile(expr, dtype="float16", backend="hip")(x=X.to(DEV), y=Y.to(DEV))
        ref = kp.compile(expr, dtype="float16", backend="oracle")(x=X, y=Y)
        assert out.dtype == ref.dtype
        assert (out.cpu() != ref).float().mean() < 0.02
