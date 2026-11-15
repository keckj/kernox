"""direct / block / kahan summation schemes on the CUDA backend (contract v0.7).

Ill-conditioned data (cancelling ±1e8 terms plus small terms, FP32,
``math.fsum`` reference), forced j partition, fp_contract on/off; LOGSUMEXP /
SUMSOFTMAXWEIGHT under the three schemes with the special rows of v0.5.
"""

from __future__ import annotations

import copy
import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.version.hip is not None:  # pragma: no cover
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

import kernox as kp  # noqa: E402
from kernox.backends.cuda import CudaBackend  # noqa: E402
from kernox.plan import Plan  # noqa: E402

from .test_cuda_softmax import _close, _exprs, _inputs, _ref, _run  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda", 0)
SCHEMES = ["direct", "block", "kahan"]


def with_scheme(plan, scheme):
    """Plan with ``sum_scheme``; without the field (older versions), the attribute is set
    on a copy (the backend reads it through ``getattr``)."""
    if any(f.name == "sum_scheme" for f in dataclasses.fields(Plan)):
        return dataclasses.replace(plan, sum_scheme=scheme)
    p = copy.copy(plan)
    object.__setattr__(p, "sum_scheme", scheme)
    return p


def _higham(N, seed):
    """FP32 terms: scattered ±1e8 pairs (zero sum) + small terms in [0, 1)."""
    g = torch.Generator().manual_seed(seed)
    small = torch.rand(N, generator=g, dtype=torch.float64).float().double()
    t = small.clone()
    nb = N // 10
    pos = torch.randperm(N, generator=g)[: 2 * nb]
    t[pos[:nb]] = 1e8
    t[pos[nb:]] = -1e8
    return t


def _sum_plan(scheme, fc=True):
    x, b = kp.Vi("x", 1), kp.Vj("b", 1)
    return with_scheme((b * x).sum_over("j").to_plan("float32", fp_contract=fc), scheme)


def _sum_run(plan, t, split=None, tile=None):
    host = {"x": torch.ones(4, 1, dtype=torch.float64), "b": t.reshape(-1, 1)}
    got, cr = _run(plan, host, split=split, tile=tile)
    return got[:, 0].double(), cr


@pytest.mark.parametrize("N", [1000, 100_000, 1_000_000])
@pytest.mark.parametrize("fc", [True, False])
def test_kahan_ill_conditioned(N, fc):
    t = _higham(N, N)
    ref = math.fsum(t.tolist())
    l1 = float(t.abs().sum())
    u = 2.0 ** -24
    errs, T = {}, None
    for sch in SCHEMES:
        got, cr = _sum_run(_sum_plan(sch, fc), t)
        assert f"sum_scheme={sch}" in " ".join(cr.info.options)
        if sch == "block":
            T = int([o for o in cr.info.options if o.startswith("sum_block=")][0].split("=")[1])
            assert T == dict(cr.variant.params)["tile"]
        errs[sch] = float((got - ref).abs().max())
    assert errs["kahan"] <= 4 * u * l1 + 1e-30, errs
    assert errs["block"] <= (N / T + T + 2) * u * l1, errs
    if N >= 100_000:
        assert errs["kahan"] * 10 < errs["direct"], errs


@pytest.mark.parametrize("sch", SCHEMES)
@pytest.mark.parametrize("opts", [dict(split="off"), dict(split="x8", tile=16), dict(split="x4", tile=8)],
                         ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))
def test_sum_split_and_empty_slices(sch, opts):
    t = _higham(20_000, 7)
    t[:8000] = 0.0  # leading slices with zero sum
    ref = math.fsum(t.tolist())
    got, cr = _sum_run(_sum_plan(sch), t, **opts)
    if opts["split"] != "off":
        assert cr.launch_config(4, 20_000)[0] > 1
    l1 = float(t.abs().sum())
    tol = 4 * 2.0 ** -24 * l1 if sch == "kahan" else 20_000 * 2.0 ** -24 * l1
    assert float((got - ref).abs().max()) <= tol


@pytest.mark.parametrize("sch", ["kahan"])
def test_sum_kahan_inf_nan(sch):
    """The correction has no effect on ±inf / NaN: inf stays inf (no spurious NaN)."""
    for bad, want in ((math.inf, math.inf), (-math.inf, -math.inf), (math.nan, math.nan)):
        t = torch.ones(3000, dtype=torch.float64)
        t[1234] = bad
        for opts in (dict(split="off"), dict(split="x8", tile=16)):
            got, _ = _sum_run(_sum_plan(sch), t, **opts)
            assert (torch.isnan(got).all() if math.isnan(want) else (got == want).all()), got


def test_sum_acc_fp64_kahan():
    x, b = kp.Vi("x", 1), kp.Vj("b", 1)
    plan = with_scheme((b * x).sum_over("j").to_plan("float32", acc_dtype="float64"), "kahan")
    t = _higham(100_000, 3)
    got, _ = _sum_run(plan, t, split="x8", tile=16)
    assert float((got - math.fsum(t.tolist())).abs().max()) <= 4 * 2.0 ** -53 * float(t.abs().sum())


@pytest.mark.parametrize("sch", SCHEMES)
@pytest.mark.parametrize("opts", [dict(split="off"), dict(split="x8", tile=8), dict()],
                         ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()) or "auto")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", ["lse_vec", "ssmw"])
def test_soft_schemes_special_rows(sch, opts, dtype, name):
    plan = with_scheme(_exprs()[name].to_plan(dtype), sch)
    host = _inputs(37, 150, seed=3)
    got, cr = _run(plan, host, **opts)
    assert f"sum_scheme={sch}" in " ".join(cr.info.options)
    _close(got, _ref(plan, host), dtype, 150)


@pytest.mark.parametrize("name", ["lse", "ssmw"])
def test_soft_kahan_many_terms(name):
    """Many terms of equal weight (f ≈ 0): FP32 kahan within a few u of the FP64
    reference, clearly better than direct for N = 10⁶."""
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", 1)
    f = x - y
    expr = f.logsumexp(over="j") if name == "lse" else f.sumsoftmaxweight(w, over="j")
    N = 1_000_000
    g = torch.Generator().manual_seed(11)
    host = {"x": torch.zeros(4, 1, dtype=torch.float64),
            "y": (1e-3 * torch.rand(N, 1, generator=g, dtype=torch.float64)).float().double(),
            "w": torch.rand(N, 1, generator=g, dtype=torch.float64).float().double()}
    errs = {}
    for sch in ("direct", "kahan"):
        plan = with_scheme(expr.to_plan("float32"), sch)
        got, _ = _run(plan, host, split="off")
        ref = _ref(plan, host)
        errs[sch] = float(((got.double() - ref).abs() / ref.abs()).max())
    assert errs["kahan"] <= 8 * 2.0 ** -24, errs
    assert errs["kahan"] < errs["direct"], errs


def test_unknown_scheme_refused():
    plan = with_scheme(_exprs()["ssmw"].to_plan("float32"), "pairwise")
    assert not CudaBackend().supports(plan, DEV).ok
    # not applicable to reductions that do not accumulate floats: ignored
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    assert CudaBackend().supports(with_scheme((x - y).min(over="j").to_plan("float32"), "kahan"), DEV).ok
