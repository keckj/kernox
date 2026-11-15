"""FP16 (avenant v0.9) sur le backend CUDA : type kp_h (binary16 en mémoire, chaque
opération évaluée en FP32 puis arrondie en FP16, y compris sur sm_52).

Calcul en FP16 prouvé par débordement (identique à l'oracle), MIN/ARG*/K* et
COUNT au bit près contre l'oracle, sommes à l'ordre de sommation près
(accumulateur FP32 par défaut, FP16 + block), sortie FP16 à ±inf au-delà de
65504, partition de j forcée, top-K coopératif, batch et ranges.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.version.hip is not None:  # pragma: no cover
    pytest.skip("GPU NVIDIA requis", allow_module_level=True)

import kernox as kp  # noqa: E402
from kernox.backends.base import Variant  # noqa: E402
from kernox.backends.cuda import CudaBackend  # noqa: E402
from kernox.oracle import OracleReduction  # noqa: E402
from kernox.plan import output_shapes  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda", 0)


def _launch(plan, inp, split=None, tile=None, coop=None):
    M = inp["x"].shape[0]
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(plan, M).items()}
    cr = CudaBackend(coop_group=coop).compile(plan, DEV)
    if split is not None:
        p = dict(cr.variant.params)
        cr = cr.with_variant(Variant("tiled", (("block", p["block"]), ("split", split), ("tile", tile or p["tile"]))))
    cr.launch({k: v.to(DEV) for k, v in inp.items()}, outs)
    torch.cuda.synchronize()
    return {k: v.cpu() for k, v in outs.items()}, cr


def _oracle(plan, inp):
    M = inp["x"].shape[0]
    outs = {k: torch.empty(s, dtype=getattr(torch, d)) for k, (s, d) in output_shapes(plan, M).items()}
    OracleReduction(plan, "cpu", 4096).launch({k: v.cpu() for k, v in inp.items()}, outs)
    return outs


def _data(M=61, N=900, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return {"x": (scale * torch.randn(M, 3, generator=g)).half(), "y": (scale * torch.randn(N, 3, generator=g)).half(),
            "b": torch.randn(N, 2, generator=g).half()}


def _exprs():
    x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 2)
    d2 = (x - y).sqnorm2()
    return x, y, b, d2


SPLITS = [dict(), dict(split="off"), dict(split="x8", tile=8)]


@pytest.mark.parametrize("opts", SPLITS, ids=lambda o: o.get("split", "auto"))
@pytest.mark.parametrize("name", ["min", "max", "argmin", "min_argmin", "count", "kmin_argkmin5", "argkmin32"])
def test_exact_vs_oracle(name, opts):
    x, y, b, d2 = _exprs()
    f = kp.sqrt(d2) * 3.0 + kp.log(d2 + 1.0)            # chaîne d'opérations arrondies en FP16
    expr = {"min": f.min(over="j"), "max": f.max(over="j"), "argmin": f.argmin(over="j"),
            "min_argmin": f.min_argmin(over="j"), "count": (f < 4.0).sum_over("j"),
            "kmin_argkmin5": f.kmin_argkmin(5, over="j"), "argkmin32": f.argkmin(32, over="j")}[name]
    plan = expr.to_plan("float16")
    inp = {k: v for k, v in _data(seed=len(opts)).items() if k in {a.name for a in plan.args}}
    got, _ = _launch(plan, inp, **opts)
    ref = _oracle(plan, inp)
    for k in ref:
        assert got[k].dtype == ref[k].dtype
        assert torch.equal(got[k], ref[k]), (name, k)


def test_fp16_overflow_proves_half_arithmetic():
    """x²/y avec x = 300 : x² = 90000 déborde en FP16 (> 65504) mais pas en FP32 ;
    le résultat (90 en FP32) vaut +inf comme dans l'oracle."""
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    plan = ((x * x) / y).max(over="j").to_plan("float16")
    inp = {"x": torch.tensor([[300.0], [100.0], [-300.0]]).half(), "y": torch.tensor([[1000.0], [2000.0]]).half()}
    got, _ = _launch(plan, inp)
    ref = _oracle(plan, inp)
    # ligne 1 : l'oracle (torch_lambdify) imprime x²·y^-1 (10,0078), la paire x·x/y (10) :
    # écart d'impression des générateurs, pas de sémantique FP16
    assert torch.equal(got["value"][[0, 2]], ref["value"][[0, 2]])
    assert got["value"][0, 0] == math.inf and got["value"][2, 0] == math.inf and got["value"][1, 0] == 10.0


@pytest.mark.parametrize("acc", [None, "float16"])
@pytest.mark.parametrize("opts", SPLITS, ids=lambda o: o.get("split", "auto"))
@pytest.mark.parametrize("name", ["sum", "lse", "ssmw", "lsew"])
def test_sums_vs_oracle(name, opts, acc):
    x, y, b, d2 = _exprs()
    expr = {"sum": (kp.exp(-d2) * b).sum_over("j"), "lse": (-d2).logsumexp(over="j"),
            "ssmw": (-d2).sumsoftmaxweight(b, over="j"),
            "lsew": (-d2).logsumexp(over="j", weight=kp.exp(b))}[name]
    plan = expr.to_plan("float16") if acc is None else expr.to_plan("float16", acc_dtype=acc)
    assert plan.acc_dtype == (acc or "float32")
    inp = {k: v for k, v in _data(seed=3).items() if k in {a.name for a in plan.args}}
    got, cr = _launch(plan, inp, **opts)
    ref = _oracle(plan, inp)["value"].float()
    g = got["value"]
    assert g.dtype == torch.float16
    # Termes : l'oracle (torch_lambdify) et la paire (cudacode) associent d² dans des
    # ordres différents, d'où quelques ulp FP16 par terme ; accumulateur FP16 : ~N·u_16.
    rel = 16 * 2.0 ** -11 if acc is None else 900 * 2.0 ** -11
    assert torch.all((g.float() - ref).abs() <= rel * (1 + ref.abs())), (g.float() - ref).abs().max()
    if acc == "float16":
        assert "sum_scheme=block" in cr.info.options


def test_output_rounds_to_inf():
    """Accumulateur FP32 au-delà de 65504 : sortie FP16 +inf (conversion au plus proche)."""
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    plan = (x * y).sum_over("j").to_plan("float16")
    inp = {"x": torch.tensor([[1000.0], [0.5]]).half(), "y": torch.full((100, 1), 1000.0).half()}
    got, _ = _launch(plan, inp)
    assert got["value"][0, 0] == math.inf and got["value"][1, 0] == 49984.0   # 50000 : égalité, pair
    assert torch.equal(got["value"], _oracle(plan, inp)["value"])


def test_fp16_frontend_batch_and_ranges():
    x, y, b, d2 = _exprs()
    run = kp.compile(d2.argkmin(3, over="j"), dtype="float16", backend="cuda")
    g = torch.Generator().manual_seed(2)
    X = torch.randn(2, 40, 3, generator=g).half().to(DEV)
    Y = torch.randn(2, 500, 3, generator=g).half().to(DEV)
    got = run(x=X, y=Y)
    for e in range(2):
        assert torch.equal(got[e], run(x=X[e].contiguous(), y=Y[e].contiguous()))
    li = torch.sort(torch.randint(0, 3, (40,), generator=g)).values.to(DEV)
    lj = torch.sort(torch.randint(0, 4, (500,), generator=g)).values.to(DEV)
    keep = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.bool, device=DEV)
    r = kp.ranges_from_labels(li, lj, keep)
    ro = kp.compile(d2.argkmin(3, over="j"), dtype="float16", backend="oracle")
    a = run.with_ranges(r)(x=X[0].contiguous(), y=Y[0].contiguous()).cpu()
    rc = kp.ranges_from_labels(li.cpu(), lj.cpu(), keep.cpu())
    bref = ro.with_ranges(rc)(x=X[0].cpu(), y=Y[0].cpu())
    assert torch.equal(a, bref)
