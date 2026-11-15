"""direct / block / kahan summation schemes on the sycl backend (v0.7 amendment), on the G31.

Ill-conditioned data (cancelling ±1e8 terms plus small terms, FP32),
FP64 ``math.fsum`` reference of the exact terms; forced j partition
(multiple slices, empty slices), various tiles, FP64 acc_dtype,
fp_contract on/off, special values.
"""

import math

import pytest
import torch

import kernox as kp
from kernox import Vi, Vj
from kernox.backends.sycl import _variant
from kernox.plan import output_shapes

from ._helpers import DEV, backend

U32 = 2.0 ** -24
SCALES = torch.tensor([[1.0], [-1.0], [0.5], [2.0]], dtype=torch.float64)  # exact products


def _plan(dtype="float32", **kw):
    return (Vj("w", 1) * Vi("a", 1)).sum_over("j").to_plan(dtype, **kw)


def _higham(N, seed=0):
    """±1e8 pairs (opposite signs, shuffled positions) + small terms in [0, 1)."""
    g = torch.Generator().manual_seed(seed)
    nb = N // 8
    big = (1e8 * (1 + torch.rand(nb, generator=g, dtype=torch.float64))).float().double()
    vals = torch.cat([big, -big, torch.rand(N - 2 * nb, generator=g, dtype=torch.float64)])
    vals = vals[torch.randperm(N, generator=g)].float()
    return vals.reshape(N, 1)


def _run(plan, w, variant=None, groups=None, a=SCALES):
    T = getattr(torch, plan.dtype)
    inp = {"a": a.to(T).to(DEV), "w": w.to(T).to(DEV)}
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV)
            for k, (s, d) in output_shapes(plan, a.shape[0]).items()}
    cr = backend().compile(plan, DEV, variant)
    if groups is not None:
        cr.target_groups = groups
    cr.launch(inp, outs)
    torch.xpu.synchronize()
    return outs["value"].cpu().double()[:, 0], cr


def _errors(got, w, a=SCALES):
    """(absolute error per row, Σ|term| per row) against math.fsum of the exact terms."""
    terms = w.double()[:, 0]
    ref = torch.tensor([math.fsum((terms * s).tolist()) for s in a[:, 0].tolist()], dtype=torch.float64)
    mag = torch.tensor([float(s) for s in a[:, 0].abs()], dtype=torch.float64) * terms.abs().sum()
    return (got - ref).abs(), mag


def _block_T(cr):
    (t,) = [o for o in cr.info.options if o.startswith("sum_block=")]
    return int(t.split("=")[1])


@pytest.mark.parametrize("N", [1000, 100_000, 1_000_000])
@pytest.mark.parametrize("tj,groups", [(0, 1), (0, None), (16, 4096)], ids=str)
def test_bounds_ill_conditioned(N, tj, groups):
    w = _higham(N, seed=N)
    err = {}
    for scheme in ("direct", "block", "kahan"):
        got, cr = _run(_plan(sum_scheme=scheme), w, _variant(tj=tj), groups)
        e, mag = _errors(got, w)
        err[scheme] = e
        assert f"sum_scheme={scheme}" in cr.info.options
        if scheme == "kahan":
            assert torch.all(e <= 4 * U32 * mag), (e / (U32 * mag)).max()
        else:
            T = _block_T(cr)
            S = cr.last_split[0]
            # sum over tiles of T terms, then over tiles, then over S slice partials
            assert torch.all(e <= (N / T + T + S + 2) * U32 * mag), (e / (U32 * mag)).max()
    if N >= 100_000:
        assert torch.all(err["kahan"] * 10 < err["direct"]), (err["kahan"], err["direct"])


@pytest.mark.parametrize("fp_contract", [True, False])
def test_kahan_fp_contract_modes(fp_contract):
    """The compensation (additions only) withstands the w·a + s contraction."""
    w = _higham(200_000, seed=7)
    got, cr = _run(_plan(sum_scheme="kahan", fp_contract=fp_contract), w, groups=4096)
    assert ("-ffp-contract=off" in cr.info.options) is (not fp_contract)
    e, mag = _errors(got, w)
    assert torch.all(e <= 4 * U32 * mag), (e / (U32 * mag)).max()


@pytest.mark.parametrize("scheme", ["direct", "block", "kahan"])
def test_acc_float64(scheme):
    w = torch.tensor([1e8, 1.0, -1e8] * 1000, dtype=torch.float32).reshape(-1, 1)
    got, cr = _run(_plan(acc_dtype="float64", sum_scheme=scheme), w, _variant(tj=16), 4096)
    assert cr.last_split[0] > 1
    torch.testing.assert_close(got, 1000.0 * SCALES[:, 0], rtol=0, atol=0)


def test_float64_kahan_beats_direct():
    w = torch.tensor([1e17, 1.0, 1.0, -1e17] * 4096, dtype=torch.float64).reshape(-1, 1)
    got, _ = _run(_plan("float64", sum_scheme="kahan"), w, _variant(tj=16), 4096)
    torch.testing.assert_close(got, 8192.0 * SCALES[:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("scheme", ["direct", "block", "kahan"])
def test_special_values_and_empty(scheme):
    inf, nan = math.inf, math.nan
    a = torch.ones(1, 1, dtype=torch.float64)
    cases = [([inf, 1.0, 2.0], inf), ([-inf, 1.0], -inf), ([inf, -inf], nan), ([1.0, nan, 2.0], nan)]
    for vals, want in cases:
        for tj, groups in ((0, 1), (16, 4096)):
            w = torch.tensor(vals + [0.0] * 5000, dtype=torch.float64).reshape(-1, 1)
            got, _ = _run(_plan(sum_scheme=scheme), w, _variant(tj=tj), groups, a=a)
            assert (math.isnan(got[0]) if math.isnan(want) else got[0].item() == want), (vals, got)
    got, _ = _run(_plan(sum_scheme=scheme), torch.zeros(0, 1), a=a)
    assert got.tolist() == [0.0]


def test_scheme_ignored_for_non_summing():
    x, y = Vi("x", 2), Vj("y", 2)
    d2 = ((x - y) ** 2).sum()
    X, Y = torch.randn(50, 2).to(DEV), torch.randn(700, 2).to(DEV)
    for red in (d2.min(over="j"), d2.argkmin(3, over="j")):
        outs = []
        for s in ("direct", "kahan"):
            plan = red.to_plan("float32", sum_scheme=s)
            assert backend().supports(plan, DEV).ok
            cr = backend().compile(plan, DEV)
            assert not any(o.startswith(("sum_scheme=", "sum_block=")) for o in cr.info.options)
            o = {k: torch.empty(sh, dtype=getattr(torch, d), device=DEV)
                 for k, (sh, d) in output_shapes(plan, 50).items()}
            cr.launch({"x": X, "y": Y}, o)
            outs.append({k: v.cpu() for k, v in o.items()})
        assert all(torch.equal(outs[0][k], outs[1][k]) for k in outs[0])
