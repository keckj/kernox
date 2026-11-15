"""MIN_ARGMIN, MAX_ARGMAX and LOGSUMEXPWEIGHT (amendment v0.8) on the CUDA backend.

MIN_ARGMIN / MAX_ARGMAX: bitwise against MIN and ARGMIN (resp. MAX, ARGMAX) of
the same backend and against the oracle (ties, NaN, N = 0), forced j partition,
unordered ranges, batch. LOGSUMEXPWEIGHT: special values of the contract
(empty, +inf, negative weights, NaN), forced partition, block/kahan schemes,
acc_dtype FP64; reference: FP64 Torch oracle on the rounded inputs.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.version.hip is not None:  # pragma: no cover
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

import kernox as kp  # noqa: E402
from kernox.backends.base import BatchLayout, RangesLayout, Variant  # noqa: E402
from kernox.backends.cuda import CudaBackend  # noqa: E402
from kernox.oracle import OracleReduction  # noqa: E402
from kernox.plan import output_shapes  # noqa: E402
from kernox.ranges import Triplet, work_list  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda", 0)
DTYPES = ["float32", "float64"]
INF, NAN = math.inf, math.nan
SPLITS = [dict(split="off"), dict(split="x8", tile=8), dict(split="x4", tile=16), dict()]


def _launch(plan, inp, split=None, tile=None, ranges=None):
    M = inp[next(a.name for a in plan.args if a.role == "i")].shape[0]
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV)
            for k, (s, d) in output_shapes(plan, M).items()}
    cr = CudaBackend().compile(plan, DEV)
    if split is not None or tile is not None:
        p = dict(cr.variant.params)
        cr = cr.with_variant(Variant("tiled", (("block", p["block"]), ("split", split or p["split"]),
                                               ("tile", tile or p["tile"]))))
    lay = None
    if ranges is not None:
        work = work_list(ranges, cr.ranges_tile_m)
        red = ranges.red.to(torch.int64).contiguous()
        lay = RangesLayout(work, red, cr.ranges_tile_m, work.cpu(), red.cpu())
    cr.launch(inp, outs, ranges=lay) if lay is not None else cr.launch(inp, outs)
    torch.cuda.synchronize()
    return {k: v.cpu() for k, v in outs.items()}, cr, lay


def _oracle(plan, inp, lay=None):
    M = inp[next(a.name for a in plan.args if a.role == "i")].shape[0]
    outs = {k: torch.empty(s, dtype=getattr(torch, d)) for k, (s, d) in output_shapes(plan, M).items()}
    host = {k: v.cpu() for k, v in inp.items()}
    if lay is not None:
        lay = RangesLayout(lay.host_work, lay.host_redranges, lay.tile_m, lay.host_work, lay.host_redranges)
        OracleReduction(plan, "cpu", 4096).launch(host, outs, ranges=lay)
    else:
        OracleReduction(plan, "cpu", 4096).launch(host, outs)
    return outs


def _arg_inputs(dtype, M=53, N=700, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = getattr(torch, dtype)
    X = torch.randint(-3, 4, (M, 2), generator=g).to(T)
    Y = torch.randint(-3, 4, (N, 2), generator=g).to(T)
    if N > 7:
        Y[7, 0] = NAN
    Y[N // 2:N // 2 + 3, 1] = INF                 # valid ±inf (sign depends on x)
    return {"x": X.to(DEV), "y": Y.to(DEV)}


def _arg_exprs():
    x, y = kp.Vi("x", 2), kp.Vj("y", 2)
    f = x * y                                     # two components: D = 2
    return f


@pytest.mark.parametrize("opts", SPLITS, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()) or "auto")
@pytest.mark.parametrize("is_min", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_min_argmin_bitwise(opts, is_min, dtype):
    f = _arg_exprs()
    inp = _arg_inputs(dtype, seed=len(opts))
    both = (f.min_argmin if is_min else f.max_argmax)(over="j").to_plan(dtype)
    got, cr, _ = _launch(both, inp, **opts)
    if opts.get("split") not in (None, "off"):
        assert cr.launch_config(53, 700)[0] > 1
    val, _, _ = _launch((f.min if is_min else f.max)(over="j").to_plan(dtype), inp, **opts)
    idx, _, _ = _launch((f.argmin if is_min else f.argmax)(over="j").to_plan(dtype), inp, **opts)
    assert torch.equal(got["value"], val["value"]) and torch.equal(got["index"], idx["index"])
    ref = _oracle(both, inp)
    assert torch.equal(got["value"], ref["value"]) and torch.equal(got["index"], ref["index"])


@pytest.mark.parametrize("is_min", [True, False])
def test_min_argmin_empty(is_min):
    f = _arg_exprs()
    inp = _arg_inputs("float32", N=0)
    got, _, _ = _launch((f.min_argmin if is_min else f.max_argmax)(over="j").to_plan("float32"), inp)
    assert (got["index"] == -1).all() and (got["value"] == (INF if is_min else -INF)).all()


@pytest.mark.parametrize("is_min", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_min_argmin_ranges_shuffled(is_min, dtype):
    g = torch.Generator().manual_seed(2)
    M, N = 300, 700
    li = torch.sort(torch.randint(0, 4, (M,), generator=g)).values
    lj = torch.sort(torch.randint(0, 9, (N,), generator=g)).values
    keep = torch.rand(4, 9, generator=g) < 0.5
    keep[3] = False
    r = kp.ranges_from_labels(li.to(DEV), lj.to(DEV), keep.to(DEV))
    red = r.redranges_j.clone()
    lo = 0
    for hi in r.slices_i.tolist():
        red[lo:hi] = red[lo:hi].flip(0)
        lo = hi
    tr = Triplet(r.ranges_i, r.slices_i, red)
    f = _arg_exprs()
    inp = _arg_inputs(dtype, M=M, N=N, seed=5)
    plan = dataclasses.replace((f.min_argmin if is_min else f.max_argmax)(over="j").to_plan(dtype), ranges=True)
    got, _, lay = _launch(plan, inp, ranges=tr)
    ref = _oracle(plan, inp, lay)
    assert torch.equal(got["value"], ref["value"]) and torch.equal(got["index"], ref["index"])


def test_min_argmin_batch():
    f = _arg_exprs()
    plan = dataclasses.replace(f.min_argmin(over="j").to_plan("float32"), batch_rank=1)
    g = torch.Generator().manual_seed(3)
    X = torch.randint(-3, 4, (5, 40, 2), generator=g).float().to(DEV)
    Y = torch.randint(-3, 4, (5, 300, 2), generator=g).float().to(DEV)
    host = {"x": torch.arange(5) * 80, "y": torch.arange(5) * 600}
    lay = BatchLayout((5,), 5, {k: v.to(DEV) for k, v in host.items()}, host)
    outs = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(plan, 40, (5,)).items()}
    CudaBackend().compile(plan, DEV).launch({"x": X, "y": Y}, outs, batch=lay)
    p1 = f.min_argmin(over="j").to_plan("float32")
    for b in range(5):
        o, _, _ = _launch(p1, {"x": X[b], "y": Y[b]})
        assert torch.equal(outs["value"][b].cpu(), o["value"]) and torch.equal(outs["index"][b].cpu(), o["index"])


# ------------------------------------------------------------------ LOGSUMEXPWEIGHT
def _lsew_plan(dtype, E=3, scheme=None, acc=None):
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", E)
    plan = (x - y).logsumexp(over="j", weight=w).to_plan(dtype)
    if scheme:
        plan = dataclasses.replace(plan, sum_scheme=scheme)
    if acc:
        plan = dataclasses.replace(plan, acc_dtype=acc)
    return plan


def _lsew_inputs(M, N, seed=0, E=3, negative=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, 1, generator=g, dtype=torch.float64)
    x[1], x[2], x[3], x[4], x[5] = 1e4, -1e4, INF, -INF, NAN
    y = 3 * torch.randn(N, 1, generator=g, dtype=torch.float64)
    y[: N // 2] = INF                         # half of the j: f = −inf (empty slices)
    w = torch.rand(N, E, generator=g, dtype=torch.float64) + 0.1
    if negative:
        w[:, 1] -= 0.8                        # component 1: weights of mixed signs (t < 0 possible)
    w[: N // 2, 2] = NAN                      # NaN on zero-weight terms: ignored
    return {"x": x, "y": y, "w": w}


def _lsew_ref(plan, host):
    p64 = dataclasses.replace(plan, dtype="float64", acc_dtype="float64")
    cast = {k: v.to(getattr(torch, plan.dtype)).double() for k, v in host.items()}
    return _oracle(p64, cast)["value"]


def _close(got, ref, dtype, N):
    assert got.shape == ref.shape
    assert torch.equal(torch.isnan(got), torch.isnan(ref)), (got, ref)
    fin = torch.isfinite(ref)
    assert torch.equal(got[~fin & ~torch.isnan(ref)], ref[~fin & ~torch.isnan(ref)])
    u = 2.0 ** -24 if dtype == "float32" else 2.0 ** -53
    tol = 4 * u * (2e4 + 20) + 16 * N * u * (1 + ref[fin].abs())
    assert torch.all((got[fin].double() - ref[fin]).abs() <= tol), (got[fin].double() - ref[fin]).abs().max()


@pytest.mark.parametrize("opts", SPLITS, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()) or "auto")
@pytest.mark.parametrize("scheme", ["direct", "block", "kahan"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_lsew_special(opts, scheme, dtype):
    plan = _lsew_plan(dtype, scheme=scheme)
    N = 150
    host = _lsew_inputs(37, N, seed=len(opts))
    inp = {k: v.to(getattr(torch, dtype)).to(DEV) for k, v in host.items()}
    got, _, _ = _launch(plan, inp, **opts)
    ref = _lsew_ref(plan, host)
    _close(got["value"], ref, dtype, N)
    # contract coverage: x = −inf (empty: −inf), x = +inf (inf − inf: NaN), NaN, weight < 0
    assert (ref[4] == -INF).all() and torch.isnan(ref[5]).all()
    assert torch.isnan(ref[:, 1]).any() and torch.isfinite(ref[:, 0]).any()


def test_lsew_pos_inf_and_empty():
    for N in (0, 1, 300):
        host = _lsew_inputs(9, max(N, 1), seed=N, negative=False)
        host["y"] = torch.linspace(-2, 2, N, dtype=torch.float64).reshape(N, 1)
        host["w"] = host["w"][:N].nan_to_num(0.5)
        host["w"][:, 1] = 0.0                  # t = 0: log 0 = −inf; with f = +inf: NaN
        for opts in (dict(split="off"), dict(split="x8", tile=8)):
            for dtype in DTYPES:
                plan = _lsew_plan(dtype)
                inp = {k: v.to(getattr(torch, dtype)).to(DEV) for k, v in host.items()}
                got, _, _ = _launch(plan, inp, **opts)
                _close(got["value"], _lsew_ref(plan, host), dtype, max(N, 1))
                if N == 0:
                    assert (got["value"][~inp["x"].cpu().isnan().reshape(-1)] == -INF).all()
                elif N == 300:   # x = +inf: f = +inf everywhere -> +inf (t > 0), NaN (t = 0)
                    assert got["value"][3, 0] == INF and torch.isnan(got["value"][3, 1])


def test_lsew_acc64():
    plan = _lsew_plan("float32", acc="float64")
    host = _lsew_inputs(37, 2000, seed=4)
    inp = {k: v.float().to(DEV) for k, v in host.items()}
    got, _, _ = _launch(plan, inp, split="x4", tile=16)
    assert got["value"].dtype == torch.float64
    ref = _lsew_ref(plan, host)
    _close(got["value"], ref, "float64", 2000 * 2 ** 29)   # terms rounded to FP32: FP32 bound on f


def test_lsew_frontend_matches_logsumexp():
    """weight = 1: LOGSUMEXPWEIGHT = LOGSUMEXP (same state)."""
    x, y = kp.Vi("x", 3), kp.Vj("y", 3)
    f = -((x - y) ** 2).sum()
    g = torch.Generator().manual_seed(1)
    X, Y = torch.randn(100, 3, generator=g).to(DEV), torch.randn(900, 3, generator=g).to(DEV)
    a = kp.compile(f.logsumexp(over="j", weight=kp.Vj("w", 1)), dtype="float32", backend="cuda")(
        x=X, y=Y, w=torch.ones(900, 1, device=DEV))
    b = kp.compile(f.logsumexp(over="j"), dtype="float32", backend="cuda")(x=X, y=Y)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
