"""FP16 (amendment v0.9) on each backend: explicit refusal (``UnsupportedError``,
before any launch) as long as the backend does not declare it, otherwise matches the oracle."""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox.errors import UnsupportedError

from ._support import synchronize


def _exprs():
    x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 2)
    d2 = (x - y).sqnorm2()
    return {"sum": (kp.exp(-d2) * b).sum_over("j"), "logsumexp": (-d2).logsumexp(over="j"),
            "min": d2.min(over="j"), "argkmin": d2.argkmin(4, over="j")}


@pytest.mark.parametrize("name", list(_exprs()))
def test_fp16_refused_or_matches_oracle(backend, name):
    bname, dev = backend
    g = torch.Generator().manual_seed(0)
    data = dict(x=torch.randn(150, 3, generator=g), y=torch.randn(230, 3, generator=g),
                b=torch.randn(230, 2, generator=g))
    expr = _exprs()[name]
    run = kp.compile(expr, dtype="float16", backend=bname)
    ins = {k: v.to(dev, torch.float16) for k, v in data.items() if k in run.input_names}
    try:
        out = run(**ins)
    except UnsupportedError as e:
        assert "float16" in str(e) or "dtype" in str(e), e
        pytest.skip(f"{bname}: explicit refusal ({e})")
    synchronize(dev)
    ref = kp.compile(expr, dtype="float16", backend="oracle")(**{k: v.cpu() for k, v in ins.items()})
    assert out.dtype == ref.dtype
    if out.is_floating_point():
        # The order of additions in a formula differs between printers (C: x0, x1, x2;
        # Torch: x2, x0, x1): in FP16, each term may differ from the oracle by one ulp, and
        # a sum of N terms of mixed signs then exceeds 2e-3 absolute. Criterion: gap to the
        # FP16 oracle within 2e-3, or error against the FP64 reference (same FP16 inputs)
        # at most twice that of the oracle, plus 2e-3.
        close = torch.isclose(out.cpu(), ref, rtol=2e-3, atol=2e-3)
        if not bool(close.all()):
            r64 = kp.compile(expr, dtype="float64", backend="oracle")(
                **{k: v.cpu().double() for k, v in ins.items()})
            err = (out.cpu().double() - r64).abs().max().item()
            err_o = (ref.double() - r64).abs().max().item()
            assert err <= 2 * err_o + 2e-3, (err, err_o)
    else:
        assert (out.cpu() != ref).float().mean() < 0.02
