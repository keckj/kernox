"""LOGSUMEXP / SUMSOFTMAXWEIGHT on the CUDA backend (contract v0.5).

Forced j partition (first half of j at f = −inf: fully empty slices,
neutral partial states), tiny tiles, values 1e4 / −1e4 / ±inf /
NaN per row, acc_dtype FP64, fp_contract on/off, large-E SSMW in FP64.
Reference: the core FP64 Torch oracle on inputs rounded to the dtype.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.version.hip is not None:  # pragma: no cover
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

import kernox as kp  # noqa: E402
from kernox.backends.base import Variant  # noqa: E402
from kernox.backends.cuda import CudaBackend  # noqa: E402
from kernox.oracle import OracleReduction  # noqa: E402
from kernox.plan import output_shapes  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda", 0)
DTYPES = ["float32", "float64"]
INF, NAN = math.inf, math.nan


def _exprs(E=3):
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", E)
    f = x - y
    return {"lse_vec": kp.cat(f, 2 * f - 1.0).logsumexp(over="j"),
            "ssmw": f.sumsoftmaxweight(w, over="j")}


def _inputs(M, N, seed=0, E=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, 1, generator=g, dtype=torch.float64)
    x[1], x[2], x[3], x[4], x[5] = 1e4, -1e4, INF, -INF, NAN
    y = 3 * torch.randn(N, 1, generator=g, dtype=torch.float64)
    y[: N // 2] = INF                     # first half of j: f = −inf (empty slices)
    w = torch.randn(N, E, generator=g, dtype=torch.float64)
    return {"x": x, "y": y, "w": w}


def _ref(plan, host):
    p64 = plan if plan.dtype == "float64" and plan.acc_dtype == "float64" else \
        kp.builder.make_plan(plan.outputs, plan.reduction, "float64", args=plan.args)
    M = host["x"].shape[0]
    outs = {k: torch.empty(s, dtype=getattr(torch, d)) for k, (s, d) in output_shapes(p64, M).items()}
    cast = {a.name: host[a.name].to(getattr(torch, plan.dtype)).double() for a in plan.args}
    OracleReduction(p64, "cpu", 4096).launch(cast, outs)
    return outs["value"]


def _run(plan, host, split=None, tile=None):
    T = getattr(torch, plan.dtype)
    M = host["x"].shape[0]
    inp = {a.name: host[a.name].to(T).to(DEV) for a in plan.args}
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV)
            for k, (s, d) in output_shapes(plan, M).items()}
    cr = CudaBackend().compile(plan, DEV)
    if split is not None or tile is not None:
        p = dict(cr.variant.params)
        cr = cr.with_variant(Variant("tiled", (("block", p["block"]), ("split", split or p["split"]),
                                               ("tile", tile or p["tile"]))))
    cr.launch(inp, outs)
    torch.cuda.synchronize()
    return outs["value"].cpu(), cr


def _close(got, ref, dtype, N):
    assert got.shape == ref.shape
    fin = torch.isfinite(ref)
    assert torch.equal(torch.isnan(got), torch.isnan(ref)), (got, ref)
    assert torch.equal(got[~fin & ~torch.isnan(ref)], ref[~fin & ~torch.isnan(ref)])
    u = 2.0 ** -24 if dtype == "float32" else 2.0 ** -53
    # |f| <= 1e4 + 10: term error u·|f| (rounding of f to the dtype) + summation
    tol = 4 * u * (2e4 + 20) + 8 * N * u * (1 + ref[fin].abs())
    assert torch.all((got[fin].double() - ref[fin]).abs() <= tol), (got[fin] - ref[fin]).abs().max()


SPLITS = [dict(split="off"), dict(split="x8", tile=8), dict(split="x4", tile=16),
          dict(split="x2", tile=32), dict()]


@pytest.mark.parametrize("opts", SPLITS, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()) or "auto")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", list(_exprs()))
def test_split_special_rows(opts, dtype, name):
    plan = _exprs()[name].to_plan(dtype)
    N = 150
    host = _inputs(37, N, seed=len(opts))
    got, cr = _run(plan, host, **opts)
    if opts.get("split", "off") != "off":
        assert cr.launch_config(37, N)[0] > 1
    ref = _ref(plan, host)
    _close(got, ref, dtype, N)
    # coverage: finite x = ±1e4 (shift), x = +inf (inf − inf: NaN), x = −inf (all −inf), NaN
    assert torch.isfinite(ref[1]).all() and torch.isfinite(ref[2]).all() and torch.isnan(ref[3]).all()
    assert torch.isnan(ref[5]).all()
    assert (ref[4] == -INF).all() if name != "ssmw" else torch.isnan(ref[4]).all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", list(_exprs()))
def test_pos_inf_and_empty_rows(dtype, name):
    """y without +inf: x = +inf gives f = +inf everywhere (LSE +inf, SSMW NaN); N = 0 and 1."""
    plan = _exprs()[name].to_plan(dtype)
    for N in (0, 1, 300):
        host = _inputs(9, max(N, 1), seed=N)
        host["y"] = torch.linspace(-2, 2, N, dtype=torch.float64).reshape(N, 1)
        host["w"] = host["w"][:N]
        for opts in (dict(split="off"), dict(split="x8", tile=8)):
            got, _ = _run(plan, host, **opts)
            ref = _ref(plan, host)
            _close(got, ref, dtype, max(N, 1))
            if N == 0:
                assert ((got == -INF).all() if name != "ssmw" else torch.isnan(got).all())
            else:
                assert ((got[3] == INF).all() if name != "ssmw" else torch.isnan(got[3]).all())


def test_nan_weight_on_dead_term_ignored():
    """NaN g on a zero-weight term (f = −inf): ignored; on a live term: NaN."""
    plan = _exprs()["ssmw"].to_plan("float64")
    host = {"x": torch.zeros(2, 1, dtype=torch.float64),
            "y": torch.tensor([[INF], [0.0], [1.0]], dtype=torch.float64),
            "w": torch.tensor([[NAN] * 3, [1.0] * 3, [2.0] * 3], dtype=torch.float64)}
    got, _ = _run(plan, host, split="off")
    assert torch.isfinite(got).all()
    host["y"][0] = 5.0
    got, _ = _run(plan, host, split="off")
    assert torch.isnan(got).all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", list(_exprs()))
def test_large_mn_split_auto(dtype, name):
    plan = _exprs()[name].to_plan(dtype)
    M, N = 300, 50_000
    host = _inputs(M, N, seed=1)
    got, cr = _run(plan, host)
    assert cr.launch_config(M, N)[0] > 1
    _close(got, _ref(plan, host), dtype, N)


@pytest.mark.parametrize("name", ["lse", "ssmw"])
def test_fp32_acc_fp64(name):
    # Terms f = x − y: a single FP32 subtraction, rounded identically on the GPU and by
    # Torch CPU; only the FP64 accumulation error remains.
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", 3)
    expr = (x - y).logsumexp(over="j") if name == "lse" else (x - y).sumsoftmaxweight(w, over="j")
    plan = expr.to_plan("float32", acc_dtype="float64")
    host = _inputs(40, 2000, seed=2)
    got, _ = _run(plan, host, split="x8", tile=16)
    assert got.dtype == torch.float64
    M = host["x"].shape[0]
    ref = torch.empty(output_shapes(plan, M)["value"][0], dtype=torch.float64)
    OracleReduction(plan, "cpu", 4096).launch({a.name: host[a.name].float() for a in plan.args}, {"value": ref})
    fin = torch.isfinite(ref)
    assert torch.equal(torch.isnan(got), torch.isnan(ref))
    assert torch.all((got[fin] - ref[fin]).abs() <= 1e-12 * (1 + ref[fin].abs()))


@pytest.mark.parametrize("dtype", DTYPES)
def test_fp_contract_modes(dtype):
    x, y, s = kp.Vi("x", 3), kp.Vj("y", 3), kp.Param("s")
    expr = (-((x - y) ** 2).sum() * s).logsumexp(over="j")
    g = torch.Generator().manual_seed(5)
    host = {"x": torch.randn(64, 3, generator=g, dtype=torch.float64),
            "y": torch.randn(3000, 3, generator=g, dtype=torch.float64),
            "s": torch.tensor([2.5], dtype=torch.float64)}
    u = 2.0 ** -24 if dtype == "float32" else 2.0 ** -53
    outs = []
    for fc in (True, False):
        plan = expr.to_plan(dtype, fp_contract=fc)
        got, cr = _run(plan, host)
        assert ("--fmad=true" if fc else "--fmad=false") in cr.info.options
        outs.append(got)
        ref = _ref(plan, host)
        assert torch.all((got.double() - ref).abs() <= 64 * u * (1 + ref.abs() + 50))
    assert torch.all((outs[0] - outs[1]).abs() <= 128 * u * (1 + outs[0].abs() + 50))


@pytest.mark.parametrize("E", [16, 64])
def test_ssmw_fp64_large_e(E):
    """FP64 SSMW with large E: (2 + E) doubles of state per thread; the kernel must stay
    resident (no launch failure) and exact, with or without partition."""
    plan = _exprs(E)["ssmw"].to_plan("float64")
    N = 3000
    host = _inputs(50, N, seed=E, E=E)
    ref = _ref(plan, host)
    for opts in (dict(split="off"), dict()):
        got, _ = _run(plan, host, **opts)
        _close(got, ref, "float64", N)
