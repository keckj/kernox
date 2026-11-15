"""End to end: kernox frontend → hip backend on R9700, compared to the core oracle.

The blockwise Torch oracle runs on the same device (a reference independent of
the kernel); ``reference_numpy`` (FP64, full grid) is used for small cases.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import corpus  # noqa: E402  (shared core corpus, tests/corpus.py)

import kernox as kp  # noqa: E402
from kernox.oracle import reference_numpy  # noqa: E402
from kernox.plan import ReductionKind as R  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = "cuda"


def _inputs(plan, M, N, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    T = getattr(torch, dtype)
    out = {}
    for a in plan.args:
        if a.role == "param":
            out[a.name] = torch.full((a.dim,), 0.7, dtype=T)
        else:
            L = M if a.role == "i" else N
            out[a.name] = torch.randn(L, a.dim, generator=g, dtype=torch.float64).to(T)
    return out


def _compare(plan, got, ref, F_l1=None):
    kind = plan.reduction.kind
    got = got if isinstance(got, tuple) else (got,)
    ref = ref if isinstance(ref, tuple) else (ref,)
    for g, r in zip(got, ref):
        g, r = g.cpu(), r.cpu()
        assert g.dtype == r.dtype and g.shape == r.shape
        if g.dtype == torch.int64 and kind in (R.COUNT,):
            assert torch.equal(g, r)
        elif g.dtype == torch.int64:
            # indices: exact equality expected except for rounding near-ties; we
            # tolerate different indices only if the values coincide.
            if not torch.equal(g, r):
                return False
        else:
            tol = 1e-10 if plan.dtype == "float64" else 2e-4
            assert torch.allclose(g, r, rtol=tol, atol=tol, equal_nan=True), (g, r)
    return True


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("name", sorted(corpus.reductions()))
def test_corpus_vs_oracle(name, dtype):
    red = corpus.reductions()[name]
    hip = red.compile(dtype, backend="hip")
    orc = red.compile(dtype, backend="oracle")
    plan = hip.plan
    M, N = 333, 1201
    host = _inputs(plan, M, N, dtype, seed=len(name))
    dev_in = {k: v.to(DEV) for k, v in host.items()}
    got = hip(**dev_in)
    ref = orc(**dev_in)
    torch.cuda.synchronize()
    if dtype == "float32" and plan.reduction.kind is R.SUM:
        # FP32, contract criterion: error relative to the L1 norm of the terms,
        # measured against an FP64 reference on the same widened FP32 inputs.
        # Bound: summation (TJ + N/TJ)·eps·L1, plus the intrinsic error of the terms
        # (conditioning of the formula) estimated by that of the FP32 oracle.
        d64 = {k: v.double() for k, v in dev_in.items()}
        ref64 = red.compile("float64", backend="oracle")(**d64)
        l1 = kp.abs(red.formula).sum_over("j").compile("float64", backend="oracle")(**d64)
        eps = torch.finfo(torch.float32).eps
        tj = hip.compiled(torch.device(DEV)).config.tile_j
        bound = (tj + N / tj + 4) * eps * l1 + 4 * (ref.double() - ref64).abs().amax()
        assert torch.all((got.double() - ref64).abs() <= bound)
        return
    ok = _compare(plan, got, ref)
    if not ok:
        pytest.fail(f"{name}/{dtype}: indices differ from the oracle")


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_small_vs_numpy_fp64_oracle(dtype):
    x, y = kp.Vi("x", 2), kp.Vj("y", 2)
    d2 = ((x - y) ** 2).sum()
    for red in [d2.sum_over("j"), d2.min(over="j"), d2.argmax(over="j"),
                d2.kmin_argkmin(5, over="j") if hasattr(d2, "kmin_argkmin") else d2.argkmin(5, over="j"),
                (d2 < 1.0).sum_over("j")]:
        run = red.compile(dtype, backend="hip")
        host = _inputs(run.plan, 17, 29, dtype, seed=1)
        got = run(**{k: v.to(DEV) for k, v in host.items()})
        ref = reference_numpy(run.plan, {k: v.numpy() for k, v in host.items()})
        got = got if isinstance(got, tuple) else (got,)
        for g, (key, r) in zip(got, ref.items()):
            r = torch.as_tensor(r)
            if g.dtype == torch.int64:
                assert torch.equal(g.cpu(), r.to(torch.int64)), key
            else:
                assert torch.allclose(g.cpu().double(), r.double(), rtol=1e-6, atol=1e-6), key


def test_auto_backend_is_hip_and_over_i():
    x, y = kp.Vi("x", 3), kp.Vj("y", 3)
    red = ((x - y) ** 2).sum().argkmin(2, over="i")
    run = red.compile("float64")  # auto → hip on Torch ROCm
    X = torch.randn(40, 3, dtype=torch.float64, device=DEV)
    Y = torch.randn(25, 3, dtype=torch.float64, device=DEV)
    got = run(x=X, y=Y)
    ref = red.compile("float64", backend="oracle")(x=X, y=Y)
    assert type(kp.get_backend("hip")).__name__ == "HipBackend"
    assert torch.equal(got, ref) and got.shape == (25, 2)


def test_nan_inputs_vs_oracle():
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    f = abs(x - y).sum() if hasattr(abs(x - y), "sum") else abs(x - y)
    X = torch.randint(-3, 4, (50, 1), device=DEV).double()
    Y = torch.randint(-3, 4, (400, 1), device=DEV).double()
    Y[::3] = math.nan
    for red in [f.min(over="j"), f.argmin(over="j"), f.argkmin(7, over="j"), f.max(over="j")]:
        got = red.compile("float64", backend="hip")(x=X, y=Y)
        ref = red.compile("float64", backend="oracle")(x=X, y=Y)
        assert torch.equal(got, ref)
