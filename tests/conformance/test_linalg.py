"""Linear operator and ``solve`` on each backend, against the oracle (FP64)."""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox.errors import UnsupportedError

from ._support import synchronize


def _run(target):
    x, y, b, s = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 2), kp.Param("s")
    k = kp.exp(-(x - y).sqnorm2() / (2.0 * s**2))
    return kp.compile((k * b).sum_over("j"), dtype="float64", backend=target)


def test_operator_and_solve_match_oracle(backend):
    bname, dev = backend
    g = torch.Generator().manual_seed(0)
    x = torch.randn(300, 3, dtype=torch.float64, generator=g)
    v = torch.randn(300, 2, dtype=torch.float64, generator=g)
    rhs = torch.randn(300, 2, dtype=torch.float64, generator=g)
    w = torch.randn(300, 2, dtype=torch.float64, generator=g)
    results = []
    for target, device in ((bname, dev), ("oracle", "cpu")):
        run = _run(target)
        xl = x.to(device).requires_grad_(True)
        sl = torch.tensor(0.6, dtype=torch.float64, device=device, requires_grad=True)
        rl = rhs.to(device).requires_grad_(True)
        try:
            op = run.linear_operator("b", x=xl, y=xl, s=sl)
            mv = op @ v.to(device)
            rv = op.rmatvec(v.to(device))
            a, info = op.solve(rl, alpha=0.05, eps=1e-12, return_info=True)
            grads = torch.autograd.grad((a * w.to(device)).sum(), [xl, sl, rl])
        except UnsupportedError as e:
            pytest.skip(f"{target}: {e}")
        synchronize(device)
        assert info["converged"]
        results.append([t.detach().cpu() for t in (mv, rv, a, *grads)])
    for got, want in zip(*results):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-9)


def test_batched_solve_matches_oracle(backend):
    bname, dev = backend
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 150, 3, dtype=torch.float64, generator=g)
    rhs = torch.randn(150, 2, dtype=torch.float64, generator=g)
    w = torch.randn(2, 150, 2, dtype=torch.float64, generator=g)
    results = []
    for target, device in ((bname, dev), ("oracle", "cpu")):
        run = _run(target)
        xl = x.to(device).requires_grad_(True)
        rl = rhs.to(device).requires_grad_(True)
        try:
            a = run.solve("b", rl, alpha=0.05, eps=1e-12, x=xl, y=xl, s=0.6)
            grads = torch.autograd.grad((a * w.to(device)).sum(), [xl, rl])
        except UnsupportedError as e:
            pytest.skip(f"{target}: {e}")
        synchronize(device)
        results.append([t.detach().cpu() for t in (a, *grads)])
    for got, want in zip(*results):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-9)
