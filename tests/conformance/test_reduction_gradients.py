"""Gradients des réductions LOGSUMEXP/SUMSOFTMAXWEIGHT (plan SUM substitut) et
MIN/MAX/KMIN (terme retenu) sur chaque backend, contre l'oracle (FP64)."""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox.errors import UnsupportedError

from ._support import synchronize


def _exprs():
    x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 2)
    f = -(x - y).sqnorm2() * 2.0 + (x * y).sum()
    return {
        "logsumexp": f.logsumexp(over="j"),
        "logsumexp_weight": f.logsumexp(over="j", weight=b * b + 0.1),
        "sumsoftmaxweight": f.sumsoftmaxweight(b, over="j"),
        "min": f.min(over="j"),
        "max_i": f.max(over="i"),
        "kmin": f.kmin(4, over="j"),
    }


@pytest.mark.parametrize("name", list(_exprs()))
def test_gradients_match_oracle(backend, name):
    bname, dev = backend
    expr = _exprs()[name]
    g = torch.Generator().manual_seed(0)
    data = dict(x=torch.randn(150, 3, dtype=torch.float64, generator=g),
                y=torch.randn(230, 3, dtype=torch.float64, generator=g),
                b=torch.randn(230, 2, dtype=torch.float64, generator=g))
    results = []
    for target, device in ((bname, dev), ("oracle", "cpu")):
        run = kp.compile(expr, dtype="float64", backend=target)
        names = {a.name for a in run.plan.args}
        leaves = {k: v.to(device).requires_grad_(True) for k, v in data.items() if k in names}
        try:
            out = run(**leaves)
            G = torch.randn(out.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1)).to(device)
            grads = torch.autograd.grad((out * G).sum(), list(leaves.values()))
        except UnsupportedError as e:
            pytest.skip(f"{target} : {e}")
        synchronize(device)
        results.append([out.cpu(), *(t.cpu() for t in grads)])
    for a, w in zip(*results):
        torch.testing.assert_close(a, w, rtol=1e-10, atol=1e-10)
