"""Native batch (amendment v0.6 §1) and block-sparse ranges (§2) of the hip backend.

Batch: bitwise equality with a loop of non-batched calls over each
element (same variants), broadcasting with 0 strides, batched params, forced
j partition (partials (nb, S, M, W)), coop top-K, LSE/SSMW, launches in chunks
of the z dimension (lowered GRID_Z_MAX). Ranges: one block per work tile;
unordered intervals and ties broken by the smallest j of J(i), uncovered rows
at the neutral value, blocks taller than ``ranges_tile_m``, coop top-K,
K > |J(i)|; reference: masked oracle (ranges plan) on the host.
"""

import dataclasses
import math

import pytest
import torch

import kernox as kp
from kernox.backends.base import BatchLayout, RangesLayout
from kernox.backends.hip import HipBackend, HipOptions
from kernox.backends.hip import backend as hip_backend
from kernox.oracle import OracleReduction
from kernox.plan import output_shapes
from kernox.ranges import Triplet, work_list

pytestmark = pytest.mark.gpu
DEV = "cuda"


def _exprs():
    x, y, b, s = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 2), kp.Param("s", 1)
    d2 = ((x - y) ** 2).sum()
    return {
        "sum": (kp.exp(-d2 * s) * b).sum_over("j"),
        "argmin": d2.argmin(over="j"),
        "kmin_argkmin5": d2.kmin_argkmin(5, over="j"),
        "kmin_argkmin40": d2.kmin_argkmin(40, over="j"),     # coop
        "lse": (-d2 * s).logsumexp(over="j"),
        "ssmw": (-d2 * s).sumsoftmaxweight(b, over="j"),
        "min_argmin": d2.min_argmin(over="j"),             # amendment v0.8
        "max_argmax": (-d2).max_argmax(over="j"),
        "lsew": (-d2 * s).logsumexp(over="j", weight=kp.abs(b) + 0.5),
    }


def _offsets(t, B, tail):
    off = torch.zeros(B, dtype=torch.int64)
    bshape, bstride = t.shape[:-tail], t.stride()[:-tail]
    lead = len(B) - len(bshape)
    for k, (size, stride) in enumerate(zip(bshape, bstride)):
        if size != 1 and stride:
            view = [1] * len(B)
            view[lead + k] = size
            off = off + torch.arange(size, dtype=torch.int64).view(view) * stride
    return off.reshape(-1).contiguous()


def _element(t, B, b, tail):
    """Element b (flattened index) of t broadcast to B, contiguous."""
    idx = []
    for size in reversed(B):
        idx.append(b % size)
        b //= size
    idx = idx[::-1]
    lead = len(B) - (t.dim() - tail)
    sel = tuple(0 if t.shape[k] == 1 else idx[lead + k] for k in range(t.dim() - tail))
    return t[sel].contiguous()


def _run_batched(be, plan, tensors, B):
    nb = math.prod(B)
    pb = dataclasses.replace(plan, batch_rank=len(B))
    cr = be.compile(pb, DEV)
    host = {a.name: _offsets(tensors[a.name], B, 1 if a.role == "param" else 2) for a in plan.args}
    layout = BatchLayout(B, nb, {k: v.to(DEV) for k, v in host.items()}, host)
    M = tensors[next(a.name for a in plan.args if a.role == "i")].shape[-2]
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV)
            for k, (s, d) in output_shapes(pb, M, (nb,)).items()}
    cr.launch(tensors, outs, batch=layout)
    torch.cuda.synchronize()
    return outs, cr


def _run_loop(be, plan, tensors, B):
    nb = math.prod(B)
    cr = be.compile(plan, DEV)
    M = tensors[next(a.name for a in plan.args if a.role == "i")].shape[-2]
    res = {k: [] for k in output_shapes(plan, M)}
    for b in range(nb):
        sub = {a.name: _element(tensors[a.name], B, b, 1 if a.role == "param" else 2) for a in plan.args}
        outs = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(plan, M).items()}
        cr.launch(sub, outs)
        for k in outs:
            res[k].append(outs[k])
    torch.cuda.synchronize()
    return {k: torch.stack(v) for k, v in res.items()}


def _batch_inputs(dtype, M=37, N=150, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = getattr(torch, dtype)
    x = torch.randint(-3, 4, (3, 1, M, 3), generator=g).to(T)          # (B1, 1, M, 3)
    y = torch.randint(-3, 4, (1, 4, N, 3), generator=g).to(T)          # (1, B2, N, 3)
    y[0, 1, 7] = math.nan
    b = torch.randn(4, N, 2, generator=g).to(T).expand(3, 4, N, 2)     # leading 0 strides
    s = torch.tensor([[0.3], [0.7], [1.1]]).to(T).reshape(3, 1, 1)      # batched param (B1, 1, 1)
    return {k: v.to(DEV) for k, v in dict(x=x, y=y, b=b, s=s).items()}, (3, 4)


SPLITS = [dict(split="off"), dict(split="force", tile_j=16), dict(split="force", tile_j=8, jchunk=40)]


@pytest.mark.parametrize("opts", SPLITS, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", list(_exprs()))
def test_batch_bitwise_vs_loop(opts, dtype, name):
    plan = _exprs()[name].to_plan(dtype)
    t, B = _batch_inputs(dtype, seed=len(opts))
    be = HipBackend(HipOptions(**opts))
    got, cr = _run_batched(be, plan, t, B)
    ref = _run_loop(be, plan, t, B)
    if opts.get("split") == "force":
        assert cr.last_launch["S"] > 1
    for k in ref:
        assert torch.equal(got[k].isnan(), ref[k].isnan())
        assert torch.equal(torch.nan_to_num(got[k]), torch.nan_to_num(ref[k])), (name, k)


def test_batch_grid_z_chunks(monkeypatch):
    monkeypatch.setattr(hip_backend, "GRID_Z_MAX", 5)   # 12 elements → chunks of 5, 5, 2
    for opts in (dict(split="off"), dict(split="force", tile_j=16)):
        plan = _exprs()["kmin_argkmin5"].to_plan("float32")
        t, B = _batch_inputs("float32", seed=3)
        be = HipBackend(HipOptions(**opts))
        got, _ = _run_batched(be, plan, t, B)
        ref = _run_loop(be, plan, t, B)
        for k in ref:
            assert torch.equal(torch.nan_to_num(got[k]), torch.nan_to_num(ref[k]))


def test_batch_empty_and_n0():
    plan = _exprs()["argmin"].to_plan("float32")
    t, B = _batch_inputs("float32")
    t["y"] = t["y"][:, :, :0]
    t["b"] = t["b"][:, :, :0]
    got, _ = _run_batched(HipBackend(), plan, t, B)
    assert (got["index"] == -1).all()


# ------------------------------------------------------------------ ranges
def _ranges_setup(dtype, m=300, n=700, density=0.5, seed=0, shuffle=False):
    g = torch.Generator().manual_seed(seed)
    li = torch.sort(torch.randint(0, 4, (m,), generator=g)).values
    lj = torch.sort(torch.randint(0, 9, (n,), generator=g)).values
    keep = torch.rand(4, 9, generator=g) < density
    keep[3] = False                         # i block without intervals (rows at the neutral value)
    r = kp.ranges_from_labels(li.to(DEV), lj.to(DEV), keep.to(DEV))
    tr = Triplet(r.ranges_i, r.slices_i, r.redranges_j)
    if shuffle:  # intervals of each block in reverse order
        red = r.redranges_j.clone()
        sl = r.slices_i.tolist()
        lo = 0
        for hi in sl:
            red[lo:hi] = red[lo:hi].flip(0)
            lo = hi
        tr = Triplet(r.ranges_i, r.slices_i, red)
    # uncovered rows: block 3 removed from the i ranges? ranges_from_labels keeps it with 0 intervals
    X = torch.randint(-2, 3, (m, 3), generator=g).double()
    Y = torch.randint(-2, 3, (n, 3), generator=g).double()          # many d² ties
    Y[11, 0] = math.nan
    Bv = torch.randn(n, 2, generator=g, dtype=torch.float64)
    T = getattr(torch, dtype)
    return tr, {"x": X.to(DEV, T), "y": Y.to(DEV, T), "b": Bv.to(DEV, T),
                "s": torch.tensor([0.4], dtype=T, device=DEV)}


def _oracle_ranged(plan, t, tr, tile_m):
    pr = dataclasses.replace(plan, ranges=True)
    M = t["x"].shape[0]
    host = {k: v.cpu() for k, v in t.items()}
    work = work_list(tr, tile_m)
    red = tr.red.to(torch.int64).contiguous()
    lay = RangesLayout(work.cpu(), red.cpu(), tile_m, work.cpu(), red.cpu())
    outs = {k: torch.empty(s, dtype=getattr(torch, d)) for k, (s, d) in output_shapes(pr, M).items()}
    OracleReduction(pr, "cpu", 1 << 16).launch(host, outs, ranges=lay)
    return outs


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", list(_exprs()))
def test_ranges_vs_oracle(name, dtype, shuffle):
    plan = _exprs()[name].to_plan(dtype)
    tr, t = _ranges_setup(dtype, shuffle=shuffle, seed=hash(name) % 100)
    be = HipBackend(HipOptions(block=64))     # 64-row tiles (taller i blocks)
    cr = be.compile(dataclasses.replace(plan, ranges=True), DEV)
    tile_m = cr.ranges_tile_m
    work = work_list(tr, tile_m)
    red = tr.red.to(torch.int64).contiguous()
    M = t["x"].shape[0]
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV)
            for k, (s, d) in output_shapes(cr.plan, M).items()}
    cr.launch(t, outs, ranges=RangesLayout(work, red, tile_m, work.cpu(), red.cpu()))
    torch.cuda.synchronize()
    ref = _oracle_ranged(plan, t, tr, tile_m)
    for k in ref:
        g_, r_ = outs[k].cpu(), ref[k]
        assert torch.equal(g_.isnan(), r_.isnan()), k
        if k == "index" or g_.dtype == torch.int64 or name in ("argmin", "kmin_argkmin5", "kmin_argkmin40", "min_argmin", "max_argmax"):
            assert torch.equal(torch.nan_to_num(g_), torch.nan_to_num(r_)), (name, k)
        else:
            u = 2.0 ** -24 if dtype == "float32" else 2.0 ** -53
            fin = torch.isfinite(r_)
            assert torch.equal(g_[~fin & ~r_.isnan()], r_[~fin & ~r_.isnan()])
            assert torch.all((g_[fin].double() - r_[fin].double()).abs() <= 64 * 700 * u * (1 + r_[fin].abs().double()))


@pytest.mark.parametrize("name", ["sum", "argmin", "kmin_argkmin40", "ssmw"])
def test_dense_ranges_bitwise(name):
    plan = _exprs()[name].to_plan("float32")
    _tr, t = _ranges_setup("float32")
    M, N = t["x"].shape[0], t["y"].shape[0]
    be = HipBackend(HipOptions(split="off"))
    plain = be.compile(plan, DEV)
    o1 = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(plan, M).items()}
    plain.launch(t, o1)
    r = kp.Ranges.dense(M, N, DEV)
    tr = Triplet(r.ranges_i, r.slices_i, r.redranges_j)
    cr = be.compile(dataclasses.replace(plan, ranges=True), DEV)
    work = work_list(tr, cr.ranges_tile_m)
    red = tr.red.to(torch.int64).contiguous()
    o2 = {k: torch.empty_like(v) for k, v in o1.items()}
    cr.launch(t, o2, ranges=RangesLayout(work, red, cr.ranges_tile_m, work.cpu(), red.cpu()))
    torch.cuda.synchronize()
    for k in o1:
        assert torch.equal(torch.nan_to_num(o1[k]), torch.nan_to_num(o2[k])), k


def test_ranges_batch_refused():
    plan = dataclasses.replace(_exprs()["sum"].to_plan("float32"), ranges=True, batch_rank=1)
    assert not HipBackend().supports(plan, DEV).ok
