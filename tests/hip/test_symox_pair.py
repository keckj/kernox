"""M1/M3 avec de vraies expressions symox (fonction de paire générée).

Utilise ``codegen/pair.py`` du noyau (dialecte ``hip``).
"""

import math

import pytest
import torch
from _cases import ref_reduce

from kernox.backends.hip import HipBackend
from kernox.plan import Arg, Plan, Reduction
from kernox.plan import ReductionKind as R
from kernox.plan import output_shapes

sx = pytest.importorskip("symox")
pytestmark = pytest.mark.gpu
DEV = "cuda"

X = sx.symbols("x0:3", real=True)
Y = sx.symbols("y0:3", real=True)
B = sx.symbols("b0:2", real=True)
S = sx.Symbol("s", positive=True)
D2 = sum((a - b) ** 2 for a, b in zip(X, Y))
ARGS = (Arg("x", "i", 3, X), Arg("y", "j", 3, Y), Arg("b", "j", 2, B), Arg("s", "param", 1, (S,)))


def _plan(outputs, kind, dtype, k=None, acc=None):
    return Plan(args=ARGS, outputs=tuple(outputs), reduction=Reduction(kind, k), dtype=dtype,
                acc_dtype=acc or dtype)


def _data(M, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, 3, generator=g, dtype=torch.float64),
            "y": torch.randn(N, 3, generator=g, dtype=torch.float64),
            "b": torch.randn(N, 2, generator=g, dtype=torch.float64),
            "s": torch.tensor([0.8], dtype=torch.float64)}


def _torch_d2(t):
    return ((t["x"][:, None, :] - t["y"][None, :, :]) ** 2).sum(-1)


def _run(plan, host, M):
    T = getattr(torch, plan.dtype)
    inp = {k: v.to(T).to(DEV) for k, v in host.items()}
    outs = {n: torch.empty(s, dtype=getattr(torch, d), device=DEV)
            for n, (s, d) in output_shapes(plan, M).items()}
    cr = HipBackend().compile(plan, DEV)
    cr.launch(inp, outs)
    torch.cuda.synchronize()
    return {k: v.cpu() for k, v in outs.items()}, cr


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_gaussian_vector_sum(dtype):
    g = sx.exp(-D2 / (2 * S ** 2))
    plan = _plan([g * B[0], g * B[1]], R.SUM, dtype)
    host = _data(400, 1500)
    out, cr = _run(plan, host, 400)
    assert "kp_pair" in cr.source and "exp" in cr.source
    G = torch.exp(-_torch_d2(host) / (2 * 0.8 ** 2))
    F = G[..., None] * host["b"][None]
    ref = F.sum(1)
    tol = {"float64": 1e-12, "float32": 1e-5}[dtype]
    assert torch.all((out["value"].double() - ref).abs() <= tol * F.abs().sum(1))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_count_rational_radius(dtype):
    plan = _plan([sx.Le(D2, sx.Rational(1, 4) * 3)], R.COUNT, dtype)
    host = _data(300, 2000, seed=3)
    out, _ = _run(plan, host, 300)
    T = getattr(torch, dtype)
    ref = (_torch_d2({k: v.to(T) for k, v in host.items()}) <= 0.75).sum(1, keepdim=True)
    # frontière : tolérer les seuls couples à moins de 1e-5 du rayon en FP32
    diff = (out["value"] - ref).abs()
    near = ((_torch_d2(host) - 0.75).abs() < 1e-5).sum(1, keepdim=True)
    assert torch.all(diff <= near)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_piecewise_branch_preserved(dtype):
    """Branche inactive sqrt(d2 - 1) jamais évaluée hors domaine : pas de NaN."""
    pw = sx.Piecewise((sx.sqrt(D2 - 1), D2 > 1), (0, True))
    plan = _plan([pw], R.MAX, dtype)
    host = _data(200, 700, seed=4)
    out, cr = _run(plan, host, 200)
    assert "?" in cr.source  # conditionnelle conservée
    d2 = _torch_d2(host)
    ref = torch.where(d2 > 1, torch.sqrt((d2 - 1).clamp_min(0)), torch.zeros_like(d2)).amax(1, keepdim=True)
    assert not torch.isnan(out["value"]).any()
    torch.testing.assert_close(out["value"].double(), ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_knn_argkmin(dtype):
    plan = _plan([D2], R.KMIN_ARGKMIN, dtype, k=8)
    host = _data(256, 3000, seed=5)
    out, _ = _run(plan, host, 256)
    T = getattr(torch, dtype)
    # référence : d2 calculée comme le kernel (d0² + d1² + d2², même ordre), FP64 pour départager
    d2 = _torch_d2(host)
    vals, idx = torch.sort(d2, dim=1, stable=True)
    assert torch.equal(out["index"], idx[:, :8]) or dtype == "float32"
    if dtype == "float32":
        # FP32 : quasi-égalités possibles ; on vérifie que les valeurs retournées sont exactes
        rec = d2.gather(1, out["index"])
        torch.testing.assert_close(out["value"].double(), rec, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(out["value"].double(), vals[:, :8], rtol=1e-5, atol=1e-6)
    else:
        torch.testing.assert_close(out["value"], vals[:, :8].to(T), rtol=1e-13, atol=1e-14)


def test_python_constants_and_inf():
    # constantes Python usuelles dans l'expression, +inf comme valeur valide
    e = sx.Piecewise((sx.oo, D2 > 4.5), (D2 * 0.25 + 2, True))
    plan = _plan([e], R.ARGMIN, "float64")
    host = _data(64, 200, seed=6)
    out, _ = _run(plan, host, 64)
    d2 = _torch_d2(host)
    F = torch.where(d2 > 4.5, torch.full_like(d2, math.inf), d2 * 0.25 + 2)[..., None]
    assert torch.equal(out["index"], ref_reduce(F, R.ARGMIN)["index"])
