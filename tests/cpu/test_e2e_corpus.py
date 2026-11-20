"""End to end: frontend -> runtime -> cpu backend, compared with the core FP64
NumPy oracle (shared corpus). (5, 20000) splits j (partials + merge pass)."""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from conftest import make_inputs
from corpus import reductions
from kernox.oracle import evaluate_numpy, reduce_numpy
from kernox.plan import ReductionKind, output_shapes

RTOL = {"float32": 2e-5, "float64": 1e-12}


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", list(reductions()))
@pytest.mark.parametrize("m,n", [(97, 301), (300, 1), (5, 20000)])
def test_corpus_cpu_vs_oracle(name, dtype, m, n):
    red = reductions()[name].compile(dtype, backend="cpu")
    plan = red.plan
    inputs = make_inputs(plan, m, n, seed=zlib.crc32(name.encode()) % 1000, dtype=dtype)
    out = red(**inputs)
    kind = plan.reduction.kind
    if kind is ReductionKind.KMIN_ARGKMIN:
        got = {"value": out[0].numpy(), "index": out[1].numpy()}
    else:
        (key,) = [k for k in ("value", "index") if k in output_shapes(plan, m)]
        got = {key: out.numpy()}
    vals = evaluate_numpy(plan, inputs)
    ref = reduce_numpy(plan, vals)
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
        same = (np.isinf(r) & (g == r)) | (np.isnan(g) & np.isnan(r))
        close = np.abs(g - r) <= rtol * np.maximum(1.0, np.abs(np.where(np.isinf(r), 0, r)))
        assert (same | close).all()
    if "index" in got:
        gi, ri = got["index"], ref["index"]
        np.testing.assert_array_equal(gi < 0, ri < 0)
        f = vals[..., 0] if kind in (ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN) else None
        for i in range(m):
            for c in range(gi.shape[1]):
                if gi[i, c] < 0:
                    continue
                a, b = (f[i, gi[i, c]], f[i, ri[i, c]]) if f is not None else \
                    (vals[i, gi[i, c], c], vals[i, ri[i, c], c])
                assert abs(a - b) <= rtol * max(1.0, abs(b)) or a == b, (i, c, a, b)
