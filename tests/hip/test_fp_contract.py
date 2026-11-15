"""Contract v0.3: FMA contraction allowed by default, forbidden if ``plan.fp_contract=False``."""

import dataclasses

import pytest
import torch
from _cases import GAUSS, SQDIST, make_plan, pair_provider, ref_reduce

from kernox.backends.hip import HipBackend, _hip
from kernox.plan import ReductionKind as R
from kernox.plan import output_shapes

pytestmark = pytest.mark.gpu
DEV = "cuda"


@pytest.fixture(scope="module")
def be():
    return HipBackend(pair_provider=pair_provider)


def _run(be, plan, host, M):
    T = getattr(torch, plan.dtype)
    inp = {k: v.to(T).to(DEV) for k, v in host.items()}
    outs = {n: torch.empty(s, dtype=getattr(torch, d), device=DEV)
            for n, (s, d) in output_shapes(plan, M).items()}
    cr = be.compile(plan, DEV)
    cr.launch(inp, outs)
    torch.cuda.synchronize()
    return {k: v.cpu() for k, v in outs.items()}, cr


def _host(M, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, 3, generator=g, dtype=torch.float64),
            "s": torch.tensor([1.1], dtype=torch.float64),
            "y": torch.randn(N, 3, generator=g, dtype=torch.float64),
            "b": torch.randn(N, 2, generator=g, dtype=torch.float64)}


def test_default_plan_allows_contraction():
    from kernox.plan import Plan
    assert Plan.__dataclass_fields__["fp_contract"].default is True


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_flags_key_and_tolerance(be, dtype):
    on = make_plan(GAUSS, R.SUM, dtype, fp_contract=True)
    off = make_plan(GAUSS, R.SUM, dtype, fp_contract=False)
    M, N = 300, 2000
    h = _host(M, N, seed=1)
    o_on, c_on = _run(be, on, h, M)
    o_off, c_off = _run(be, off, h, M)
    assert "-ffp-contract=on" in c_on.info.options and "-ffp-contract=off" not in c_on.info.options
    assert "-ffp-contract=off" in c_off.info.options and "-ffp-contract=on" not in c_off.info.options
    assert all("fast" not in f for f in c_on.info.options if f.startswith("-ffp-contract"))
    assert c_on.cache_key != c_off.cache_key
    T = getattr(torch, dtype)
    F = GAUSS.torch_f({k: v.to(T) for k, v in h.items()}).double()
    ref = F.sum(1)
    eps = torch.finfo(T).eps
    l1 = F.abs().sum(1)
    for o in (o_on, o_off):
        assert torch.all((o["value"].double() - ref).abs() <= 64 * eps * l1)


def test_contraction_really_changes_code(be):
    """Proof that the flag takes effect: d0*d0 + d1*d1 + d2*d2 in FP64 differs bitwise
    between the two modes on at least one pair, and only ``off`` matches the Torch
    reference (without FMA) bitwise."""
    M, N = 256, 512
    h = _host(M, N, seed=2)
    on = make_plan(SQDIST, R.KMIN, "float64", 1, fp_contract=True)
    off = make_plan(SQDIST, R.KMIN, "float64", 1, fp_contract=False)
    o_on, _ = _run(be, on, {"x": h["x"], "y": h["y"]}, M)
    o_off, _ = _run(be, off, {"x": h["x"], "y": h["y"]}, M)
    ref = ref_reduce(SQDIST.torch_f({"x": h["x"], "y": h["y"]}), R.KMIN, 1)["value"]
    assert torch.equal(o_off["value"], ref)
    assert not torch.equal(o_on["value"], ref)
    torch.testing.assert_close(o_on["value"], ref, rtol=1e-15, atol=1e-15)


def test_explicit_override_option():
    from kernox.backends.hip import HipOptions
    plan = make_plan(SQDIST, R.SUM, "float64", fp_contract=True)
    assert "-ffp-contract=off" in HipOptions(fp_contract="off").flags("gfx1201", plan)
    assert "-ffp-contract=on" in HipOptions().flags("gfx1201", plan)
    assert "-ffp-contract=off" in HipOptions().flags("gfx1201", dataclasses.replace(plan, fp_contract=False))
    assert _hip.same_runtime_as_torch()


def test_runtime_accepts_fp_contract_false():
    """The runtime requires ``fp_contract_off`` from the backend for fp_contract=False (contract v0.3)."""
    import kernox as kp

    x, y = kp.Vi("x", 2), kp.Vj("y", 2)
    run = kp.compile(((x - y) ** 2).sum().sum_over("j"), dtype="float64", backend="hip", fp_contract=False)
    out = run(x=torch.randn(5, 2, dtype=torch.float64, device="cuda"),
              y=torch.randn(7, 2, dtype=torch.float64, device="cuda"))
    assert out.shape == (5, 1)
