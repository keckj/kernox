import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
symox = pytest.importorskip("symox")


def make_inputs(plan, m, n, *, seed=0, dtype=None, device="cpu", scale=1.0):
    """Random inputs matching the plan (names -> Torch tensors)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    dt = {"float32": torch.float32, "float64": torch.float64}[dtype or plan.dtype]
    out = {}
    for a in plan.args:
        rows = {"i": m, "j": n, "param": None}[a.role]
        shape = (a.dim,) if rows is None else (rows, a.dim)
        out[a.name] = (scale * torch.randn(shape, generator=g, dtype=torch.float64)).to(dt).to(device)
    return out
