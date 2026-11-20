"""Amendment v0.11: ``flush_denormals=True`` (subnormal numbers may be flushed to zero).

The option allows a backend to flush subnormal inputs and results of
floating-point operations to zero; it never requires it (the oracle keeps
them). Every other rule is unchanged, so the result stays within the IEEE
tolerance plus the flushed mass: each of the N terms of a row may lose at most
``tiny`` (smallest normal number) times the factor it is multiplied by.

Terms here: ``exp(x_i - y_j) · b_j`` with ``x_i - y_j`` from -120 to 0, so a
row sums normal, subnormal and zero terms. Reference: FP64 on CPU (Torch) on
the same inputs. The cpu backend (FTZ + DAZ, every dtype) and the cuda backend
(``--ftz=true``, float32) must actually flush: a row whose terms are all
subnormal sums to 0. A backend that does not implement v0.11 raises
``UnsupportedError`` (hip, sycl so far): checked, not silently skipped.
"""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox import Vi, Vj
from kernox.autograd import gradient_plans
from kernox.errors import UnsupportedError

from ._support import synchronize

M, N = 64, 500
U = {"float32": 2.0**-24, "float64": 2.0**-53}
DT = {"float32": torch.float32, "float64": torch.float64}
LO = {"float32": -120.0, "float64": -760.0}  # below the smallest subnormal
# a row whose terms are all subnormal: exp(SUB - y), y in [0, 1]
SUB = {"float32": -95.0, "float64": -720.0}
FLUSHES = {("cpu", "float32"), ("cpu", "float64"), ("cuda", "float32")}


def _data(dtype):
    g = torch.Generator().manual_seed(11)
    x = torch.linspace(LO[dtype] + 1, 0.0, M, dtype=torch.float64)[:, None]
    x[0, 0] = SUB[dtype]
    y = torch.rand(N, 1, generator=g, dtype=torch.float64)
    b = torch.rand(N, 1, generator=g, dtype=torch.float64) + 0.5
    return {k: v.to(DT[dtype]) for k, v in dict(x=x, y=y, b=b).items()}


def _sum():
    return (kp.exp(Vi("x", 1) - Vj("y", 1)) * Vj("b", 1)).sum_over("j")


def _run(red, bname, dev, t, dtype, **opts):
    run = red.compile(dtype, backend=bname, **opts)
    out = run(**{k: v.to(dev) for k, v in t.items() if k in run.input_names})
    synchronize(dev)
    return out.double().cpu(), run


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_flush_denormals_sum(backend, dtype):
    bname, dev = backend
    t = _data(dtype)
    try:
        got, run = _run(_sum(), bname, dev, t, dtype, flush_denormals=True)
    except UnsupportedError:
        assert bname in ("hip", "sycl"), f"{bname} must implement amendment v0.11"
        pytest.skip(f"{bname}: flush_denormals not implemented (UnsupportedError, amendment v0.11)")
    x, y, b = (t[k].double() for k in ("x", "y", "b"))
    terms = torch.exp(x - y.T) * b.T
    ref, l1 = terms.sum(1, keepdim=True), terms.abs().sum(1, keepdim=True)
    tiny = torch.finfo(DT[dtype]).tiny
    bound = 4 * N * U[dtype] * l1 + 2 * N * tiny * b.abs().max()
    assert ((got - ref).abs() <= bound).all(), ((got - ref).abs() - bound).max()
    ieee, _ = _run(_sum(), bname, dev, t, dtype)
    assert ieee[0, 0] > 0  # the IEEE row of subnormal terms is a subnormal number
    if (bname, dtype) in FLUSHES:
        assert got[0, 0] == 0, got[0, 0]
        opts = run.compiled(torch.device(dev), M, N).info.options
        assert any("ftz" in o for o in opts), opts
    elif bname == "oracle":
        assert torch.equal(got, ieee)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_flush_denormals_logsumexp(backend, dtype):
    bname, dev = backend
    t = _data(dtype)
    red = (Vi("x", 1) - Vj("y", 1)).logsumexp(over="j")
    try:
        got, _ = _run(red, bname, dev, t, dtype, flush_denormals=True)
    except UnsupportedError:
        assert bname in ("hip", "sycl"), f"{bname} must implement amendment v0.11"
        pytest.skip(f"{bname}: flush_denormals not implemented")
    x, y = t["x"].double(), t["y"].double()
    ref = torch.logsumexp(x - y.T, 1, keepdim=True)
    assert ((got - ref).abs() <= 8 * U[dtype] * (1 + ref.abs())).all()


def test_flush_denormals_is_opt_in_keyed_and_inherited():
    red = _sum()
    p0, p1 = red.to_plan("float32"), red.to_plan("float32", flush_denormals=True)
    assert not p0.flush_denormals and p1.flush_denormals and p0.key() != p1.key()
    g = gradient_plans(p1, ("x", "y"))
    assert g.i_plan.flush_denormals and g.j_plan.flush_denormals
    with pytest.raises(kp.ValidationError):
        red.compile("float32", flush_denormals=1)
