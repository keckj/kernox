"""Summation schemes (amendment v0.7): direct, block, kahan (Neumaier).

Run on the oracle (CPU reference that actually simulates each scheme in
``acc_dtype``) and on each available GPU backend. A backend that does not
implement a scheme (or an option of the case: ``fp_contract=False``, LSE/SSMW)
answers ``Support(False)``: the case is **explicitly skipped** (reason shown),
never replaced by ``direct``.

Higham data: ``±1e8`` terms that cancel out (positions 4k and 4k+2) between
small terms in [0, 1), in FP32; ``math.fsum`` reference in FP64 of the FP32
values. Requirements (``u = 2⁻²⁴``, ``L1 = Σ|term|``):

- ``kahan``: error ≤ 4·u·L1 (margin: none) for any N, with ``fp_contract``
  true and false;
- ``block``: error ≤ (N/T + T + 2)·u·L1, ``T`` published by the backend in
  ``CompileInfo.options`` (``sum_block=T``);
- ``direct``: error ≤ N·u·L1;
- N ≥ 10⁵, FP32: error(kahan) ≤ error(direct) / 10.

Same requirements for the ``t`` sum of SUMSOFTMAXWEIGHT (uniform weights,
Higham ``g``); for LOGSUMEXP (positive terms), kahan is at least 10× more
accurate than direct for N ≥ 10⁵.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import kernox as kp
from kernox import Vi, Vj
from kernox.errors import UnsupportedError

from ._support import BACKENDS, backend_status, synchronize

U32 = 2.0**-24
SIZES = (10**3, 10**4 + 7, 10**5, 10**6 + 3)


@pytest.fixture(params=("oracle",) + BACKENDS)
def target(request):
    name = request.param
    if name == "oracle":
        return name, "cpu"
    dev, reason = backend_status(name)
    if dev is None:
        pytest.skip(f"backend {name} unavailable: {reason}")
    return name, dev


def higham(n, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.random(n).astype(np.float32)
    big = np.zeros(n, np.float32)
    big[0::4], big[2::4] = 1e8, -1e8
    return np.where(big != 0, big, w).astype(np.float32)


def _run(red, name, dev, inputs, **opts):
    """Result (first row, first component) and compiled object; skips if unsupported."""
    run = kp.compile(red, dtype="float32", backend=name, **opts)
    try:
        out = run(**{k: v.to(dev) for k, v in inputs.items()})
    except UnsupportedError as exc:
        # explicit refusal (scheme, fp_contract=False, reduction not yet
        # implemented...): skipped with its reason, never replaced by direct
        pytest.skip(f"{name}: {exc}")
    if dev != "cpu":
        synchronize(dev)
    m = next(iter(inputs.values())).shape[0]
    n = max(v.shape[0] for v in inputs.values())
    return out.double().cpu(), run.compiled(torch.device(dev), m, n).info


def _sum_block(info):
    opts = [o for o in info.options if o.startswith("sum_block=")]
    assert opts, f"block scheme: tile size not published in CompileInfo.options {info.options}"
    return int(opts[0].split("=")[1])


def _bound(scheme, n, l1, info):
    if scheme == "kahan":
        return 4 * U32 * l1
    if scheme == "block":
        T = _sum_block(info)
        return (n / T + T + 2) * U32 * l1
    return n * U32 * l1


# ------------------------------------------------------------------- SUM
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("fp_contract", [True, False])
@pytest.mark.parametrize("acc", ["float32", "float64"])
def test_sum_schemes_higham(target, n, fp_contract, acc):
    name, dev = target
    W = higham(n)
    ref = math.fsum(W.astype(np.float64))
    l1 = math.fsum(np.abs(W.astype(np.float64)))
    red = (Vj("w", 1) * Vi("one", 1)).sum_over("j")
    inputs = dict(one=torch.ones(3, 1), w=torch.from_numpy(W)[:, None])
    err = {}
    for scheme in ("direct", "block", "kahan"):
        out, info = _run(red, name, dev, inputs, sum_scheme=scheme, fp_contract=fp_contract, acc_dtype=acc)
        assert torch.equal(out[0], out[1]) and torch.equal(out[0], out[2])
        err[scheme] = abs(out[0, 0].item() - ref)
        assert err[scheme] <= _bound(scheme, n, l1, info), (scheme, err[scheme], info.options)
    if n >= 10**5 and acc == "float32":
        assert err["kahan"] * 10 <= err["direct"], err


# ------------------------------------------------------ SUMSOFTMAXWEIGHT
@pytest.mark.parametrize("n", SIZES)
def test_sumsoftmaxweight_schemes_higham(target, n):
    name, dev = target
    W = higham(n, seed=1)
    ref = math.fsum(W.astype(np.float64)) / n
    l1 = math.fsum(np.abs(W.astype(np.float64))) / n
    one, f, g = Vi("one", 1), Vj("f", 1), Vj("g", 1)
    red = (f * one).sumsoftmaxweight(g * one, over="j")
    inputs = dict(one=torch.ones(2, 1), f=torch.zeros(n, 1), g=torch.from_numpy(W)[:, None])
    err = {}
    for scheme in ("direct", "kahan"):
        out, info = _run(red, name, dev, inputs, sum_scheme=scheme)
        err[scheme] = abs(out[0, 0].item() - ref)
        # t and s accumulated with the scheme, then t / s: one extra rounded division
        assert err[scheme] <= _bound(scheme, n, l1, info) + 2 * U32 * abs(ref), (scheme, err[scheme])
    if n >= 10**5:
        assert err["kahan"] * 10 <= err["direct"], err


# ------------------------------------------------------------- LOGSUMEXP
@pytest.mark.parametrize("n", [10**5, 10**6 + 3])
def test_logsumexp_kahan_more_accurate(target, n):
    name, dev = target
    F = (np.random.default_rng(2).standard_normal(n) * 3).astype(np.float32)
    mx = float(F.max())
    ref = mx + math.log(math.fsum(np.exp(F.astype(np.float64) - mx)))
    one, f = Vi("one", 1), Vj("f", 1)
    red = (f * one).logsumexp(over="j")
    inputs = dict(one=torch.ones(2, 1), f=torch.from_numpy(F)[:, None])
    err = {}
    for scheme in ("direct", "kahan"):
        out, _ = _run(red, name, dev, inputs, sum_scheme=scheme)
        err[scheme] = abs(out[0, 0].item() - ref)
    # kahan: error of the (s, c) state ~ u, plus rounded FP32 log and exp
    assert err["kahan"] <= 8 * U32 * max(1.0, abs(ref)), err
    # Floor: the output is FP32, rounding ref to FP32 is unavoidable. The x10 gain
    # is required on the error above this floor (direct may already be at it).
    floor = abs(float(np.float32(ref)) - ref)
    assert err["kahan"] - floor <= max(err["direct"] - floor, 0.0) / 10 + 1e-12, (err, floor)
