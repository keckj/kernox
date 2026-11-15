"""Native FP16 (v0.9 amendment) on the sycl backend, on the G31.

The formula is evaluated in ``sycl::half`` (each operation rounded to binary16,
symox half printer); FP32 accumulator by default for SUM and the LOGSUMEXP
family, ``acc_dtype="float16"`` with ``sum_scheme="block"`` by default;
FP16 outputs (``plan.accumulated_output_dtype``), ±inf beyond 65504.
Reference: the core FP16 Torch oracle (same rules) and the FP64 oracle on the
inputs rounded to FP16.
"""

import pytest
import torch

import kernox as kp
from kernox.backends.sycl import _variant
from kernox.plan import accumulated_output_dtype

from ._helpers import DEV, backend

H = torch.float16
INF = float("inf")


def _exprs():
    x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 2)
    d2 = (x - y).sqnorm2()
    k = kp.exp(-d2)
    return {
        "sum": (k * b).sum_over("j"),
        "logsumexp": (-d2).logsumexp(over="j"),
        "logsumexp_weight": (-d2).logsumexp(over="j", weight=b * b + 0.1),
        "sumsoftmaxweight": (-d2).sumsoftmaxweight(b, over="j"),
        "min": d2.min(over="j"),
        "argmin": d2.argmin(over="j"),
        "max_argmax": (-d2).max_argmax(over="j"),
        "kmin": d2.kmin(4, over="j"),
        "argkmin": d2.argkmin(4, over="j"),
        "kmin32": d2.kmin(32, over="j"),     # coop
        "count": (d2 < 1.0).sum_over("j"),
    }


def _data(seed=0, m=300, n=2000):
    g = torch.Generator().manual_seed(seed)
    return dict(x=torch.randn(m, 3, dtype=torch.float64, generator=g),
                y=torch.randn(n, 3, dtype=torch.float64, generator=g),
                b=torch.randn(n, 2, dtype=torch.float64, generator=g))


def _assert_sum_close(got, ref, ins):
    """Weighted Gaussian SUM: FP16 terms identical up to 1 ulp (libm exp),
    FP32 sums in a different order, then rounding to FP16:
    |difference| <= ulp16(|ref|) + 2 u16 Σ|term| (small cases: dense reference)."""
    d = ins["x"].double().unsqueeze(-2) - ins["y"].double().unsqueeze(-3)
    l1 = (torch.exp(-(d * d).sum(-1)).unsqueeze(-1) * ins["b"].double().unsqueeze(-3)).abs().sum(-2)
    tol = 2.0 ** -10 * ref.double().abs() + 2 * 2.0 ** -11 * l1 + 1e-6
    assert got.dtype == ref.dtype
    assert torch.all((got.double() - ref.double()).abs() <= tol)


def _first(o):
    return o[0] if isinstance(o, tuple) else o


@pytest.mark.parametrize("name", list(_exprs()))
def test_fp16_matches_oracle(name):
    expr = _exprs()[name]
    run = kp.compile(expr, dtype="float16", backend="sycl")
    data = _data()
    ins = {k: v.to(H) for k, v in data.items() if k in run.input_names}
    out = run(**{k: v.to(DEV) for k, v in ins.items()})
    ref = kp.compile(expr, dtype="float16", backend="oracle")(**ins)
    torch.xpu.synchronize()
    if isinstance(out, tuple):
        for a, b in zip(out, ref):
            assert a.dtype == b.dtype
        # values: equal to the oracle (same FP16 formula, no accumulation)
        torch.testing.assert_close(out[0].cpu(), ref[0], rtol=0, atol=0)
        assert (out[1].cpu() != ref[1]).float().mean() < 0.02  # FP16 ties
        return
    assert out.dtype == ref.dtype
    if name == "sum":
        _assert_sum_close(out.cpu(), ref, ins)
    elif out.is_floating_point():
        exact = name in ("min", "kmin", "kmin32")  # selection: identical FP16 values
        torch.testing.assert_close(out.cpu(), ref, rtol=0 if exact else 4e-3, atol=0 if exact else 4e-3)
    else:
        assert (out.cpu() != ref).float().mean() < 0.02


def test_defaults_and_output_dtype():
    be = backend()
    x, y = kp.Vi("x"), kp.Vj("y")
    plan = (x * y).sum_over("j").to_plan("float16")
    assert plan.acc_dtype == "float32" and accumulated_output_dtype(plan) == "float16"
    assert be.supports(plan, DEV).ok
    p16 = (x * y).sum_over("j").to_plan("float16", acc_dtype="float16")
    assert p16.sum_scheme == "block" and be.supports(p16, DEV).ok
    cr = be.compile(p16, DEV)
    assert "sum_scheme=block" in cr.info.options


def test_formula_is_evaluated_in_fp16():
    """300·300 overflows in FP16 (max 65504) despite the FP32 accumulator; the FP32
    sum 1000·60 = 60000 fits; 2000·60 = 120000 overflows at the final conversion."""
    x, y = kp.Vi("x"), kp.Vj("y")
    run = kp.compile((x * y).sum_over("j"), dtype="float16", backend="sycl")
    out = run(x=torch.tensor([[300.0], [2.0]], dtype=H, device=DEV),
              y=torch.tensor([[300.0], [1.0]], dtype=H, device=DEV)).cpu()
    assert out.dtype == H and out[0, 0] == INF and out[1, 0] == 602.0
    for n, expect in ((1000, 60000.0), (2000, INF)):
        o = run(x=torch.tensor([[60.0]], dtype=H, device=DEV), y=torch.ones(n, 1, dtype=H, device=DEV)).cpu()
        assert o[0, 0] == expect


@pytest.mark.parametrize("scheme", ["direct", "block", "kahan"])
def test_acc_float16_schemes_against_oracle(scheme):
    """acc_dtype float16: FP16 accumulator rounded at each addition. The schemes
    stay within their error bound with respect to the FP64 reference."""
    x, y = kp.Vi("x"), kp.Vj("y")
    expr = (x * y).sum_over("j")
    g = torch.Generator().manual_seed(3)
    X = torch.rand(64, 1, generator=g).to(H)
    Y = torch.rand(4000, 1, generator=g).to(H) * 0.01
    run = kp.compile(expr, dtype="float16", backend="sycl", acc_dtype="float16", sum_scheme=scheme)
    out = run(x=X.to(DEV), y=Y.to(DEV)).cpu().double()
    exact = (X.double() * Y.double().T).to(H).double().sum(1, keepdim=True)  # FP16 terms, exact sum
    err = (out - exact).abs().max().item()
    u = 2.0 ** -11
    l1 = exact.abs().max().item()
    bound = {"kahan": 4 * u * l1 + 1e-3, "block": (4000 / 16 + 256 + 2) * u * l1, "direct": 4000 * u * l1}[scheme]
    assert err <= bound, (scheme, err, bound)
    if scheme == "kahan":
        naive = kp.compile(expr, dtype="float16", backend="oracle", acc_dtype="float16",
                           sum_scheme="direct")(x=X, y=Y).double()
        assert err < (naive - exact).abs().max().item()


@pytest.mark.parametrize("tj,groups", [(0, 1), (8, 4096)], ids=str)
@pytest.mark.parametrize("name", ["sum", "logsumexp", "sumsoftmaxweight", "min", "argkmin"])
def test_fp16_split_j(name, tj, groups):
    expr = _exprs()[name]
    plan = expr.to_plan("float16")
    data = _data(seed=5, m=40, n=3000)
    ins = {a.name: data[a.name].to(H) for a in plan.args}
    from kernox.plan import output_shapes

    outs = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(plan, 40).items()}
    cr = backend().compile(plan, DEV, _variant(tj=tj))
    cr.target_groups = groups
    cr.launch({k: v.to(DEV) for k, v in ins.items()}, outs)
    torch.xpu.synchronize()
    if groups > 1:
        assert cr.last_split[0] > 1
    ref = kp.compile(expr, dtype="float16", backend="oracle")(**ins)
    got = outs.get("value", outs.get("index")).cpu()
    ref = _first(ref)
    if name == "sum":
        _assert_sum_close(got, ref, ins)
    elif got.is_floating_point():
        torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)
    else:
        assert (got != ref).float().mean() < 0.02


def test_fp16_batch_runtime():
    expr = _exprs()["sum"]
    g = torch.Generator().manual_seed(8)
    X = torch.randn(2, 1, 50, 3, generator=g).to(H)
    Y = torch.randn(1, 3, 400, 3, generator=g).to(H)
    B = torch.randn(1, 3, 400, 2, generator=g).to(H)
    out = kp.compile(expr, dtype="float16", backend="sycl")(x=X.to(DEV), y=Y.to(DEV), b=B.to(DEV)).cpu()
    ref = kp.compile(expr, dtype="float16", backend="oracle")(x=X, y=Y, b=B)
    assert out.dtype == H and out.shape == ref.shape
    _assert_sum_close(out, ref, {"x": X, "y": Y, "b": B})
