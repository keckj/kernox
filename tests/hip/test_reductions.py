"""Correctness of the hip backend on the real GPU against the CPU Torch reference."""

import math

import pytest
import torch
from _cases import ABS1, COUNT_BALL, GAUSS, SQDIST, VEC, make_plan, ref_reduce

from kernox.backends.base import UnsupportedError
from kernox.plan import ReductionKind as R
from kernox.plan import output_shapes

pytestmark = pytest.mark.gpu
DEV = "cuda"
DTYPES = ["float64", "float32"]


def run(backend, plan, host_inputs, M):
    T = getattr(torch, plan.dtype)
    inp = {k: v.to(T).to(DEV) for k, v in host_inputs.items()}
    outs = {n: torch.full(s, 7, dtype=getattr(torch, d), device=DEV)  # sentinel
            for n, (s, d) in output_shapes(plan, M).items()}
    cr = backend.compile(plan, DEV)
    cr.launch(inp, outs)
    torch.cuda.synchronize()
    return {k: v.cpu() for k, v in outs.items()}, {k: v.to(T) for k, v in host_inputs.items()}


def check(out, ref, kind, F=None):
    for n, r in ref.items():
        o = out[n]
        assert o.dtype == r.dtype, (n, o.dtype, r.dtype)
        if kind is R.SUM:
            # tolerance relative to the L1 norm of the terms (summation order is free)
            l1 = F.abs().to(r.dtype).sum(1)
            eps = torch.finfo(F.dtype).eps
            assert torch.all((o - r).abs() <= 8 * eps * F.shape[1] ** 0.5 * l1 + 1e-300), n
        else:
            assert torch.equal(o, r), (n, o, r)


def inputs_for(case, M, N, gen, special=None):
    t = {}
    for a in case.args:
        if a.role == "i":
            t[a.name] = torch.randn(M, a.dim, generator=gen, dtype=torch.float64)
        elif a.role == "j":
            t[a.name] = torch.randn(N, a.dim, generator=gen, dtype=torch.float64)
        else:
            t[a.name] = torch.full((a.dim,), 1.3, dtype=torch.float64)
    return t


SIZES = [(1, 1), (5, 3), (300, 517), (257, 256), (64, 1000)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M,N", SIZES)
@pytest.mark.parametrize("case,kind,k", [
    (SQDIST, R.SUM, None), (GAUSS, R.SUM, None), (COUNT_BALL, R.COUNT, None),
    (VEC, R.MIN, None), (VEC, R.MAX, None), (VEC, R.ARGMIN, None), (VEC, R.ARGMAX, None),
    (SQDIST, R.KMIN, 4), (SQDIST, R.ARGKMIN, 7), (SQDIST, R.KMIN_ARGKMIN, 1),
    (SQDIST, R.KMIN_ARGKMIN, 16),
], ids=lambda v: getattr(v, "name", getattr(v, "value", v)))
def test_random(backend, case, kind, k, dtype, M, N):
    gen = torch.Generator().manual_seed(M * 1000 + N)
    host = inputs_for(case, M, N, gen)
    plan = make_plan(case, kind, dtype, k)
    out, cast = run(backend, plan, host, M)
    F = case.torch_f(cast)
    check(out, ref_reduce(F, kind, k, acc=getattr(torch, plan.acc_dtype)), kind, F)


def test_sum_fp32_with_fp64_accumulator(backend):
    gen = torch.Generator().manual_seed(1)
    host = inputs_for(GAUSS, 200, 3000, gen)
    plan = make_plan(GAUSS, R.SUM, "float32", acc_dtype="float64")
    out, cast = run(backend, plan, host, 200)
    assert out["value"].dtype == torch.float64
    F = GAUSS.torch_f(cast).to(torch.float64)
    ref = F.sum(1)
    assert torch.all((out["value"] - ref).abs() <= 1e-6 * F.abs().sum(1))  # FP32 exp is not correctly rounded


ALL = [(VEC, R.SUM, None), (COUNT_BALL, R.COUNT, None), (VEC, R.MIN, None), (VEC, R.MAX, None),
       (VEC, R.ARGMIN, None), (VEC, R.ARGMAX, None), (SQDIST, R.KMIN, 3),
       (SQDIST, R.ARGKMIN, 3), (SQDIST, R.KMIN_ARGKMIN, 3)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case,kind,k", ALL, ids=lambda v: getattr(v, "name", getattr(v, "value", v)))
def test_n_zero_gives_neutral(backend, case, kind, k, dtype):
    M = 37
    host = inputs_for(case, M, 0, torch.Generator().manual_seed(0))
    plan = make_plan(case, kind, dtype, k)
    out, cast = run(backend, plan, host, M)
    check(out, ref_reduce(case.torch_f(cast), kind, k, acc=getattr(torch, dtype)), kind,
          case.torch_f(cast))
    if kind in (R.MIN, R.KMIN):
        assert torch.all(out["value"] == math.inf)
    if "index" in out:
        assert torch.all(out["index"] == -1)


@pytest.mark.parametrize("case,kind,k", ALL, ids=lambda v: getattr(v, "name", getattr(v, "value", v)))
def test_m_zero_no_launch(backend, case, kind, k):
    host = inputs_for(case, 0, 50, torch.Generator().manual_seed(0))
    plan = make_plan(case, kind, "float64", k)
    out, _ = run(backend, plan, host, 0)
    for n, (s, _d) in output_shapes(plan, 0).items():
        assert tuple(out[n].shape) == s


@pytest.mark.parametrize("kind", [R.KMIN, R.ARGKMIN, R.KMIN_ARGKMIN])
def test_k_zero(backend, kind):
    host = inputs_for(SQDIST, 10, 20, torch.Generator().manual_seed(0))
    plan = make_plan(SQDIST, kind, "float64", 0)
    out, _ = run(backend, plan, host, 10)
    for v in out.values():
        assert tuple(v.shape) == (10, 0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_k_greater_than_n(backend, dtype):
    host = inputs_for(SQDIST, 33, 5, torch.Generator().manual_seed(2))
    plan = make_plan(SQDIST, R.KMIN_ARGKMIN, dtype, 9)
    out, cast = run(backend, plan, host, 33)
    check(out, ref_reduce(SQDIST.torch_f(cast), R.KMIN_ARGKMIN, 9), R.KMIN_ARGKMIN)
    assert torch.all(out["index"][:, 5:] == -1) and torch.all(out["value"][:, 5:] == math.inf)


def _tie_nan_inputs(M, N, nan_frac, inf_frac, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-3, 4, (M, 1), generator=g).to(torch.float64)
    y = torch.randint(-3, 4, (N, 1), generator=g).to(torch.float64)
    u = torch.rand(N, generator=g)
    y[u < nan_frac, 0] = math.nan
    y[(u >= nan_frac) & (u < nan_frac + inf_frac), 0] = math.inf
    return {"x": x, "y": y}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind,k", [(R.MIN, None), (R.MAX, None), (R.ARGMIN, None), (R.ARGMAX, None),
                                    (R.KMIN, 6), (R.ARGKMIN, 6), (R.KMIN_ARGKMIN, 40)],
                         ids=lambda v: getattr(v, "value", v))
@pytest.mark.parametrize("nan_frac,inf_frac", [(0.0, 0.0), (0.3, 0.1), (0.95, 0.0), (1.0, 0.0), (0.0, 1.0)])
def test_ties_nan_inf(backend, kind, k, dtype, nan_frac, inf_frac):
    """Massive ties (integer values), NaN ignored, +inf is a valid candidate."""
    M, N = 70, 600
    host = _tie_nan_inputs(M, N, nan_frac, inf_frac, seed=int(nan_frac * 100 + inf_frac * 10))
    plan = make_plan(ABS1, kind, dtype, k)
    out, cast = run(backend, plan, host, M)
    check(out, ref_reduce(ABS1.torch_f(cast), kind, k), kind)


def test_sum_propagates_nan(backend):
    host = _tie_nan_inputs(10, 100, 0.05, 0.0, seed=5)
    plan = make_plan(ABS1, R.SUM, "float64")
    out, cast = run(backend, plan, host, 10)
    assert torch.all(torch.isnan(out["value"]))


def test_count_nan_is_false(backend):
    g = torch.Generator().manual_seed(9)
    host = inputs_for(COUNT_BALL, 40, 300, g)
    host["y"][::3, 1] = math.nan
    plan = make_plan(COUNT_BALL, R.COUNT, "float32")
    out, cast = run(backend, plan, host, 40)
    assert torch.equal(out["value"], ref_reduce(COUNT_BALL.torch_f(cast), R.COUNT)["value"])


# --------------------------------------------------------------- explicit refusals
@pytest.mark.parametrize("make,why", [
    (lambda: make_plan(VEC, R.KMIN, "float64", 3), "scalar"),
    (lambda: make_plan(SQDIST, R.KMIN, "float64", -1), "invalid K"),
    (lambda: make_plan(SQDIST, R.KMIN, "float64", 2.0), "invalid K"),
    (lambda: make_plan(SQDIST, R.KMIN, "float64", 10**6), "K_MAX"),
    (lambda: make_plan(SQDIST, R.SUM, "float64", acc_dtype="float32"), "acc_dtype"),
])
def test_unsupported(backend, make, why):
    plan = make()
    s = backend.supports(plan, DEV)
    assert not s.ok and why in s.reason
    with pytest.raises(UnsupportedError):
        backend.compile(plan, DEV)


def test_fast_math_refused(backend):
    import dataclasses
    plan = dataclasses.replace(make_plan(SQDIST, R.SUM, "float64"), fast_math=True)
    with pytest.raises(UnsupportedError):
        backend.compile(plan, DEV)


def test_cpu_device_refused(backend):
    assert not backend.supports(make_plan(SQDIST, R.SUM, "float64"), "cpu").ok


def test_launch_revalidates(backend):
    plan = make_plan(SQDIST, R.SUM, "float64")
    cr = backend.compile(plan, DEV)
    x = torch.randn(4, 3, device=DEV, dtype=torch.float64)
    y = torch.randn(5, 3, device=DEV, dtype=torch.float64)
    out = torch.empty(4, 1, device=DEV, dtype=torch.float64)
    with pytest.raises(ValueError):
        cr.launch({"x": x.float(), "y": y}, {"value": out})
    with pytest.raises(ValueError):
        cr.launch({"x": x, "y": y.t().contiguous().t()}, {"value": out})  # non-contiguous
    with pytest.raises(ValueError):
        cr.launch({"x": x.cpu(), "y": y}, {"value": out})
    with pytest.raises(ValueError):
        cr.launch({"x": x, "y": y}, {"value": torch.empty(3, 1, device=DEV, dtype=torch.float64)})


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", [32, 64, 128, 256])
def test_large_k_with_spills(backend, dtype, k):
    """Large K (spills measured from K=64 FP64 / K=128 FP32): exact correctness."""
    M, N = 130, 700
    host = _tie_nan_inputs(M, N, 0.1, 0.05, seed=k)
    plan = make_plan(ABS1, R.KMIN_ARGKMIN, dtype, k)
    out, cast = run(backend, plan, host, M)
    check(out, ref_reduce(ABS1.torch_f(cast), R.KMIN_ARGKMIN, k), R.KMIN_ARGKMIN)
