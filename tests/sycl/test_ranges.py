"""Native block-sparse ranges of the sycl backend (v0.6 amendment §2), on the G31.

Reference: the oracle (CPU) with the same ranges; bit-exact equality except for
floating-point sums. Cases: intervals out of order (smallest-j rule on ties),
blocks taller than ``tile_m``, uncovered rows (neutral elements),
blocks without intervals, empty work list, cooperative top-K, LSE/SSMW.
"""

import pytest
import torch

import kernox as kp
from kernox import Ranges, Vi, Vj
from kernox.backends.sycl import _variant

from ._helpers import DEV, backend


def _formulas():
    x, y, w = Vi("x", 3), Vj("y", 3), Vj("w", 2)
    d2 = ((x - y) ** 2).sum()
    return {
        "sum": (kp.exp(-0.1 * d2) * w).sum_over("j"),
        "count": (d2 < 6.0).sum_over("j"),
        "min": d2.min(over="j"),
        "argmin": d2.argmin(over="j"),
        "argmax": d2.argmax(over="j"),
        "kmin_argkmin": d2.kmin_argkmin(7, over="j"),
        "kmin_argkmin_coop": d2.kmin_argkmin(40, over="j"),
        "lse": (-0.1 * d2).logsumexp(over="j"),
        "ssmw": (-0.1 * d2).sumsoftmaxweight(w, over="j"),
    }


def _ranges(m, n, device):
    """i blocks: [0, 300) (taller than tile_m), [300, 301), [310, 600);
    [301, 310) and [600, m) not covered. j intervals out of order,
    empty, adjacent; block [300, 301) has no interval."""
    ranges_i = torch.tensor([[0, 300], [300, 301], [310, 600]], dtype=torch.int64)
    red = torch.tensor([[700, n], [0, 50], [50, 51], [400, 400], [100, 300],    # block 0
                        [900, 950], [10, 20], [0, 5]], dtype=torch.int64)       # block 2
    slices = torch.tensor([5, 5, 8], dtype=torch.int64)
    return Ranges(ranges_i.to(device), slices.to(device), red.to(device))


def _inputs(dtype, m, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = getattr(torch, dtype)
    x = torch.randint(-3, 4, (m, 3), generator=g).to(T)   # many ties
    y = torch.randint(-3, 4, (n, 3), generator=g).to(T)
    w = torch.randn(n, 2, generator=g, dtype=torch.float64).to(T)
    return {"x": x, "y": y, "w": w}


def _pick(red, t):
    names = {a.name for a in red.to_plan("float32").args}
    return {k: v for k, v in t.items() if k in names}


def _as_tuple(o):
    return o if isinstance(o, tuple) else (o,)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", list(_formulas()))
def test_ranges_match_oracle(name, dtype):
    red = _formulas()[name]
    m, n = 700, 1000
    t = _pick(red, _inputs(dtype, m, n, seed=len(name)))
    ref = _as_tuple(kp.compile(red, dtype=dtype, backend="oracle").with_ranges(_ranges(m, n, "cpu"))(**t))
    run = kp.compile(red, dtype=dtype, backend="sycl")
    got = _as_tuple(run.with_ranges(_ranges(m, n, DEV))(**{k: v.to(DEV) for k, v in t.items()}))
    torch.xpu.synchronize()
    assert len(got) == len(ref)
    for g_, r_ in zip(got, ref):
        g_ = g_.cpu()
        assert g_.shape == r_.shape and g_.dtype == r_.dtype
        if name in ("sum", "lse", "ssmw"):
            # terms of mixed signs (Gaussian weights), |J(i)| <= 450, Σ|term| <= 450:
            # summation order differs from the oracle -> absolute difference of a few 1e-6 in FP32
            torch.testing.assert_close(g_, r_, rtol=1e-5 if dtype == "float32" else 1e-12,
                                       atol=2e-5 if dtype == "float32" else 1e-12, equal_nan=True)
        else:
            assert torch.equal(g_, r_), (name, (g_ != r_).nonzero()[:5])
    # uncovered rows: neutral elements
    if name == "argmin":
        assert (got[0].cpu()[301:310] == -1).all() and (got[0].cpu()[600:] == -1).all()


@pytest.mark.parametrize("G", [0, 4, 8])
def test_ranges_variants(G):
    """All variants (one work-item per row, coop G): same result, tile_m exposed."""
    red = _formulas()["kmin_argkmin_coop"]
    m, n = 700, 1000
    t = _pick(red, _inputs("float32", m, n, seed=G))
    ref = kp.compile(red, dtype="float32", backend="oracle").with_ranges(_ranges(m, n, "cpu"))(**t)
    plan = red.to_plan("float32")
    import dataclasses

    from kernox.backends.base import RangesLayout
    from kernox.plan import output_shapes
    from kernox.ranges import work_list

    rp = dataclasses.replace(plan, ranges=True)
    cr = backend().compile(rp, DEV, _variant(sg=16, coop=G) if G else _variant())
    tr = _ranges(m, n, DEV).oriented(False)[0]
    work = work_list(tr, cr.ranges_tile_m)
    redr = tr.red.to(torch.int64).contiguous()
    lay = RangesLayout(work, redr, cr.ranges_tile_m, work.cpu(), redr.cpu())
    outs = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(rp, m).items()}
    cr.launch({k: v.to(DEV) for k, v in t.items()}, outs, ranges=lay)
    torch.xpu.synchronize()
    assert torch.equal(outs["value"].cpu(), ref[0]) and torch.equal(outs["index"].cpu(), ref[1])
    assert cr.ranges_tile_m == cr.layout.rows


def test_empty_work_and_bad_layout():
    red = _formulas()["min"]
    m, n = 50, 80
    t = {k: v.to(DEV) for k, v in _pick(red, _inputs("float32", m, n)).items()}
    run = kp.compile(red, dtype="float32", backend="sycl")
    empty = Ranges(torch.zeros(0, 2, dtype=torch.int64, device=DEV), torch.zeros(0, dtype=torch.int64, device=DEV),
                   torch.zeros(0, 2, dtype=torch.int64, device=DEV))
    out = run.with_ranges(empty)(**t)
    assert torch.isinf(out.cpu()).all()
    # bounds checked on the host before launch (memory safety)
    import dataclasses

    from kernox.backends.base import RangesLayout
    from kernox.plan import output_shapes

    rp = dataclasses.replace(red.to_plan("float32"), ranges=True)
    cr = backend().compile(rp, DEV)
    outs = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(rp, m).items()}
    work = torch.tensor([[0, 10, 0, 1]], dtype=torch.int64)
    for redr in (torch.tensor([[0, n + 1]]), torch.tensor([[-1, 3]])):
        lay = RangesLayout(work.to(DEV), redr.to(DEV), cr.ranges_tile_m, work, redr)
        with pytest.raises(ValueError, match="outside"):
            cr.launch(t, outs, ranges=lay)
    bad = torch.tensor([[0, m + 1, 0, 1]], dtype=torch.int64)
    redr = torch.tensor([[0, 3]])
    with pytest.raises(ValueError, match="out of bounds"):
        cr.launch(t, outs, ranges=RangesLayout(bad.to(DEV), redr.to(DEV), cr.ranges_tile_m, bad, redr))
