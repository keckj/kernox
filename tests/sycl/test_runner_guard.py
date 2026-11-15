"""XPU runner guard: ``check_self_idle`` must detect actual use of the G31
by the runner's own context (long kernel launched without synchronization)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from kernox.backends.sycl import _variant
from kernox.plan import output_shapes

from ._helpers import DEV, arg, backend, plan

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import runner_xpu as R  # noqa: E402

pytestmark = pytest.mark.slow


def _long_kernel():
    """coop G=32 on a sub-group of 32, K = 1024, FP32, 20000²: ~12 s (measured)."""
    x, y = arg("x", "i", 3), arg("y", "j", 3)
    p = plan([x, y], [sum((a - b) ** 2 for a, b in zip(x.symbols, y.symbols))], "argkmin", "float32", k=1024)
    c = backend().compile(p, DEV, _variant(sg=32, coop=32, grf=256))
    g = torch.Generator(device=DEV).manual_seed(0)
    X = torch.randn(20000, 3, device=DEV, generator=g)
    Y = torch.randn(20000, 3, device=DEV, generator=g)
    outs = {n: torch.empty(s, dtype=getattr(torch, t), device=DEV) for n, (s, t) in output_shapes(p, 20000).items()}
    torch.xpu.synchronize()
    return c, {"x": X, "y": Y}, outs


def test_self_busy_detected_while_own_kernel_runs():
    c, inp, outs = _long_kernel()
    assert R.own_gpu_busy(1.0) < 5.0  # context idle at the start
    c.launch(inp, outs)  # without synchronization
    busy = R.own_gpu_busy(2.0)
    print(f"own utilization while the kernel runs: {busy:.1f} %")
    assert busy > 5.0, busy
    with pytest.raises(R.SelfBusyError):
        R.check_self_idle(torch, timeout=3.0)
    torch.xpu.synchronize()
    assert R.check_self_idle(torch, timeout=60.0) < 5.0  # idle once the kernel has finished
