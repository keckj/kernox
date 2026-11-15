"""Bout en bout : frontend + fonction de paire du noyau + backend CUDA, contre l'oracle.

Corpus partagé ``tests/corpus.py``. ``pair.py`` contourne le bug
de parenthésage de ``symox.cudacode`` (anciennes révisions) grâce
aux helpers ``kp_pow*`` ; ce test le vérifie sur tout le corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("symox")
if not torch.cuda.is_available() or torch.version.hip is not None:  # pragma: no cover
    pytest.skip("GPU NVIDIA requis", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus import reductions  # noqa: E402

import kernox as kp  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda", 0)
DIMS = {"x": ("i", 3), "y": ("j", 3), "w": ("j", 2)}


def _inputs(M, N, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, (role, d) in DIMS.items():
        rows = M if role == "i" else N
        out[name] = (2 * torch.rand(rows, d, generator=g, dtype=torch.float64) - 1).to(DEV, getattr(torch, dtype))
    out["sigma"] = 0.7
    return out


def _flat(r):
    return r if isinstance(r, tuple) else (r,)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("name", sorted(reductions()))
@pytest.mark.parametrize("M,N", [(37, 211), (5, 9000)])
def test_frontend_vs_oracle(name, dtype, M, N):
    expr = reductions()[name]
    inp = _inputs(M, N, dtype, seed=M + N)
    run = kp.compile(expr, dtype=dtype, backend="cuda")
    inp = {k: v for k, v in inp.items() if k in run.input_names}
    got = _flat(run(**inp))
    ref = _flat(kp.compile(expr, dtype=dtype, backend="oracle")(**inp))
    torch.cuda.synchronize()
    for a, b in zip(got, ref):
        assert a.shape == b.shape and a.dtype == b.dtype
        if a.dtype == torch.int64 and name != "count" and name != "logic":
            # indices : comparer les valeurs atteintes serait plus robuste ; données
            # continues aléatoires -> égalités exactes improbables
            assert torch.equal(a.cpu(), b.cpu()), (a[:3], b[:3])
        elif a.dtype == torch.int64:
            assert torch.equal(a.cpu(), b.cpu())
        else:
            rtol = 1e-11 if dtype == "float64" else 2e-4
            fin = torch.isfinite(b)
            assert torch.equal(fin.cpu(), torch.isfinite(a).cpu())
            assert torch.equal(a[~fin].cpu(), b[~fin].cpu())
            scale = b[fin].abs().max().clamp_min(1.0) if fin.any() else 1.0
            assert float((a[fin] - b[fin]).abs().max()) <= rtol * float(scale) if fin.any() else True
