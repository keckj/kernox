"""End to end: frontend -> runtime -> sycl backend (core pair function)
on the G31, compared with the core FP64 NumPy oracle (shared corpus)."""

from __future__ import annotations

import zlib

import numpy as np
import pytest
import torch

from corpus import reductions
from conftest import make_inputs
from kernox.oracle import evaluate_numpy, reduce_numpy
from kernox.plan import ReductionKind

pytestmark = pytest.mark.gpu

RTOL = {"float32": 2e-5, "float64": 1e-12}


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", list(reductions()))
@pytest.mark.parametrize("m,n", [(97, 301), (300, 1)])
def test_corpus_sycl_vs_oracle(name, dtype, m, n):
    expr = reductions()[name]
    red = expr.compile(dtype, backend="sycl")
    plan = red.plan
    # stable seed: hash() of a str depends on PYTHONHASHSEED (different data in each
    # process; FP32 where_sqrt then exceeds the tolerance for some seeds, because
    # 1/sqrt(d2 - 1) is ill-conditioned near d2 = 1, with the same error in FP32 on CPU)
    inputs = make_inputs(plan, m, n, seed=zlib.crc32(name.encode()) % 1000, dtype=dtype)
    kwargs = {k: v.to("xpu") for k, v in inputs.items()}
    out = red(**kwargs)
    torch.xpu.synchronize()
    if plan.reduction.kind is ReductionKind.KMIN_ARGKMIN:
        got = {"value": out[0].cpu().numpy(), "index": out[1].cpu().numpy()}
    else:
        (key,) = [k for k in ("value", "index") if k in _keys(plan, m)]
        got = {key: out.cpu().numpy()}
    vals = evaluate_numpy(plan, inputs)
    ref = reduce_numpy(plan, vals)
    kind = plan.reduction.kind
    rtol = RTOL[dtype]
    if kind is ReductionKind.COUNT:
        np.testing.assert_array_equal(got["value"], ref["value"])
        return
    if kind is ReductionKind.SUM:
        l1 = np.abs(np.nan_to_num(vals)).sum(1)
        err = np.abs(got["value"] - ref["value"])
        assert (err <= 8 * rtol * (l1 + 1e-30)).all(), err.max()
        return
    if "value" in got:
        g, r = got["value"], ref["value"]
        same_inf = np.isinf(r) & (g == r)
        close = np.abs(g - r) <= rtol * np.maximum(1.0, np.abs(np.where(np.isinf(r), 0, r)))
        assert (same_inf | close).all()
    if "index" in got:
        gi, ri = got["index"], ref["index"]
        np.testing.assert_array_equal(gi < 0, ri < 0)
        # near-exact ties: the value at the returned point must equal the optimum
        f = vals[..., 0] if kind in (ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN) else None
        for i in range(m):
            for c in range(gi.shape[1]):
                if gi[i, c] < 0:
                    continue
                if f is not None:
                    a, b = f[i, gi[i, c]], f[i, ri[i, c]]
                else:
                    a, b = vals[i, gi[i, c], c], vals[i, ri[i, c], c]
                assert abs(a - b) <= rtol * max(1.0, abs(b)) or a == b, (i, c, a, b)


def _keys(plan, m):
    from kernox.plan import output_shapes

    return output_shapes(plan, m)
