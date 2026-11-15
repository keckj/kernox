"""Region-based CSE (``pair._region_cse``): subexpressions hoisted out of ``where``.

The inactive branches produce NaN, ±inf or a division by zero (integer data:
ties and ``d2 == 1`` are reached); the selected result must stay that of the
term-by-term FP64 reference (no contamination, no abort).
"""

from __future__ import annotations

import numpy as np
import pytest

from ._support import run_case

import kernox as kp
from kernox import Vi, Vj, where

DTYPES = ("float32", "float64")


def _formulas():
    x, y, b = Vi("x", 3), Vj("y", 3), Vj("b", 16)
    d2 = ((x - y) ** 2).sum()
    inv = 1 / (x[0] - y[0])
    h = kp.exp(d2 * 1000.0)
    return {
        # exp(-d2) shared by both branches; log(<=0) and sqrt(<0) inactive
        "nan_inactive": where(d2 > 1.0, kp.log(d2 - 1.0) * kp.exp(-d2), kp.exp(-d2) * kp.sqrt(1.0 - d2)),
        # 1/(x-y) hoisted (two where); 1/0 = inf never selected
        "div0_inactive": where(kp.eq(x[0], y[0]), 0.0, inv) + where(x[0] > y[0], inv, 2.0),
        # h = exp(1000 d2) hoisted (two where): +inf and inf - inf = NaN outside d2 = 0
        "inf_inactive": where(d2 < 0.5, h, -d2) + where(d2 < 0.5, 2.0 * h - h, d2 * d2),
        # truncated vector Gaussian (E = 16)
        "truncated_vec": where(d2 < 4.0, kp.exp(-d2) * b, 0.0),
        # nested where sharing the outer condition
        "nested": where(d2 > 0.5, where(x[1] > y[1], kp.sqrt(d2 - 0.5), -kp.sqrt(d2 - 0.5)), kp.log(d2)),
    }


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", list(_formulas()))
@pytest.mark.parametrize("kind", ["sum", "max", "argmin"])
def test_where_cse_inactive_branches(backend, dtype, name, kind):
    dev_name, dev = backend
    f = _formulas()[name]
    if kind == "sum":
        red = f.sum_over("j")
    elif kind == "max":
        red = f.max(over="j")
    else:
        if name == "truncated_vec":
            pytest.skip("argmin: scalar formula")
        red = f.argmin(over="j")
    rng = np.random.default_rng(7)
    m, n = 37, 53
    arrays = {"x": rng.integers(-2, 3, (m, 3)).astype(float),
              "y": rng.integers(-2, 3, (n, 3)).astype(float),
              "b": rng.integers(-3, 4, (n, 16)).astype(float)}
    plan = red.to_plan(dtype)
    arrays = {a.name: arrays[a.name] for a in plan.args}
    run_case(dev_name, dev, red, arrays, dtype)
