"""Complex formulas (stored as interleaved reals) on each backend, against the oracle."""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox.errors import UnsupportedError

from ._support import synchronize


def _exprs():
    z, w, x = kp.Vi("z", 2, complex=True), kp.Vj("w", 2, complex=True), kp.Vi("x", 1)
    return {
        "mult_exp": kp.exp(z * w * 0.3).sum_over("j"),
        "divide_conj": (z.conj() / (w + 2.0) + x.exp1j() * w).sum_over("j"),
        "abs_angle": (abs(z - w) + kp.angle(z * w)).sum_over("j"),
        "over_i": (z * w.conj()).sum().sum_over("i"),
    }


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", list(_exprs()))
def test_complex_matches_oracle(backend, name, dtype):
    bname, dev = backend
    g = torch.Generator().manual_seed(0)
    ct = torch.complex64 if dtype == "float32" else torch.complex128
    rt = getattr(torch, dtype)
    data = dict(z=torch.randn(150, 2, dtype=ct, generator=g), w=torch.randn(230, 2, dtype=ct, generator=g),
                x=torch.randn(150, 1, dtype=rt, generator=g))
    expr = _exprs()[name]
    results = []
    for target, device in ((bname, dev), ("oracle", "cpu")):
        run = kp.compile(expr, dtype=dtype, backend=target)
        try:
            out = run(**{k: v.to(device) for k, v in data.items() if k in run.input_names})
        except UnsupportedError as e:
            pytest.skip(f"{target}: {e}")
        synchronize(device)
        results.append(out.cpu())
    tol = 2e-5 if dtype == "float32" else 1e-12
    torch.testing.assert_close(results[0], results[1], rtol=tol, atol=tol)
