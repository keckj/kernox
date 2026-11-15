"""FP16 (avenant v0.9, parité PyKeOps ``dtype=float16``) : plan, constantes, oracle, gradients."""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox.errors import ConstantConversionError, ValidationError

H = torch.float16


def _exprs():
    x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 2)
    d2 = (x - y).sqnorm2()
    k = kp.exp(-d2)
    return {
        "sum": ((k * b).sum_over("j"), "float32", H),
        "logsumexp": ((-d2).logsumexp(over="j"), "float32", H),
        "logsumexp_weight": ((-d2).logsumexp(over="j", weight=b * b + 0.1), "float32", H),
        "sumsoftmaxweight": ((-d2).sumsoftmaxweight(b, over="j"), "float32", H),
        "min": (d2.min(over="j"), "float16", H),
        "max_argmax": ((-d2).max_argmax(over="j"), "float16", H),
        "kmin": (d2.kmin(4, over="j"), "float16", H),
        "argkmin": (d2.argkmin(4, over="j"), "float16", torch.int64),
        "count": ((d2 < 1.0).sum_over("j"), "float16", torch.int64),
    }


def _data(seed=0, m=120, n=170):
    g = torch.Generator().manual_seed(seed)
    return dict(x=torch.randn(m, 3, dtype=torch.float64, generator=g),
                y=torch.randn(n, 3, dtype=torch.float64, generator=g),
                b=torch.randn(n, 2, dtype=torch.float64, generator=g))


@pytest.mark.parametrize("name", list(_exprs()))
def test_fp16_plan_and_oracle(name):
    expr, acc, out_dtype = _exprs()[name]
    run = kp.compile(expr, dtype="float16", backend="oracle")
    assert run.plan.dtype == "float16" and run.plan.acc_dtype == acc
    data = _data()
    names = run.input_names
    out = run(**{k: v.to(H) for k, v in data.items() if k in names})
    ref = kp.compile(expr, dtype="float64", backend="oracle")(
        **{k: v.to(H).double() for k, v in data.items() if k in names})  # mêmes entrées arrondies
    out, ref = (out[0], ref[0]) if isinstance(out, tuple) else (out, ref)
    assert out.dtype == out_dtype
    if out.is_floating_point():
        torch.testing.assert_close(out.double(), ref, rtol=5e-3, atol=5e-3)
    else:
        assert (out != ref).float().mean() < 0.05  # indices/compte : quelques ex aequo FP16


def test_formula_is_evaluated_in_fp16():
    """Le produit 300·300 déborde en FP16 (max 65504) même avec un accumulateur FP32."""
    x, y = kp.Vi("x"), kp.Vj("y")
    run = kp.compile((x * y).sum_over("j"), dtype="float16", backend="oracle")
    out = run(x=torch.tensor([[300.0], [2.0]], dtype=H), y=torch.tensor([[300.0], [1.0]], dtype=H))
    assert out.dtype == H  # avenant v0.9 (b) : sortie FP16
    assert torch.isinf(out[0, 0]) and out[1, 0] == 602.0
    # accumulateur FP32 : la somme intermédiaire dépasse 65504 sans déborder, la
    # sortie FP16 est finie si le résultat l'est, ±inf sinon
    xy = kp.compile((x * y).sum_over("j"), dtype="float16", backend="oracle")
    ys = torch.tensor([[60000.0], [60000.0], [-60000.0]], dtype=H)
    assert xy(x=torch.ones(1, 1, dtype=H), y=ys).item() == 60000.0
    assert torch.isinf(xy(x=torch.ones(1, 1, dtype=H), y=ys[:2])).all()
    xy16 = kp.compile((x * y).sum_over("j"), dtype="float16", acc_dtype="float16", sum_scheme="direct",
                      backend="oracle")
    assert torch.isinf(xy16(x=torch.ones(1, 1, dtype=H), y=ys)).all()  # débordement dans l'accumulateur


def test_fp16_accumulator_rounds_each_addition():
    x, y = kp.Vi("x"), kp.Vj("y")
    run = kp.compile((x + y * 0.0).sum_over("j"), dtype="float16", acc_dtype="float16", sum_scheme="direct",
                     backend="oracle")
    # 2048 + 1 + 1 + ... : en FP16 (11 bits) 2048 + 1 s'arrondit à 2048 (pair)
    out = run(x=torch.ones(1, 1, dtype=H), y=torch.zeros(3000, 1, dtype=H))
    assert out.dtype == H and out.item() == 2048.0
    out32 = kp.compile((x + y * 0.0).sum_over("j"), dtype="float16", backend="oracle")(
        x=torch.ones(1, 1, dtype=H), y=torch.zeros(3000, 1, dtype=H))
    assert out32.item() == 3000.0


def test_fp16_acc_dtype_rules():
    x, y = kp.Vi("x"), kp.Vj("y")
    e = (x * y).sum_over("j")
    # avenant v0.9 (c) : accumulateur FP16 = PyKeOps (half, block_sum par défaut)
    p16 = kp.compile(e, dtype="float16", acc_dtype="float16", backend="oracle").plan
    assert (p16.acc_dtype, p16.sum_scheme) == ("float16", "block")
    p32 = kp.compile(e, dtype="float16", backend="oracle").plan
    assert (p32.acc_dtype, p32.sum_scheme) == ("float32", "direct")
    assert kp.compile(d2 := (x * y).min(over="j"), dtype="float16", backend="oracle").plan.sum_scheme == "direct"
    del d2
    with pytest.raises(ValidationError, match="non permis"):
        kp.compile(e, dtype="float16", acc_dtype="float64", backend="oracle")
    with pytest.raises(ValidationError, match="inférieur"):
        kp.compile(e, dtype="float32", acc_dtype="float16", backend="oracle")
    with pytest.raises(ValidationError, match="SUM"):
        kp.compile((x * y).min(over="j"), dtype="float16", acc_dtype="float32", backend="oracle")
    # les schémas de sommation s'appliquent à l'accumulateur
    for scheme in ("block", "kahan"):
        r = kp.compile(e, dtype="float16", sum_scheme=scheme, backend="oracle")
        out = r(x=torch.ones(2, 1, dtype=H), y=torch.full((5000, 1), 0.1, dtype=H))
        assert out.dtype == H
        torch.testing.assert_close(out, torch.full((2, 1), 5000 * float(torch.tensor(0.1, dtype=H)), dtype=H),
                                   rtol=0, atol=0)


def test_fp16_constants_and_params():
    x, y, s = kp.Vi("x"), kp.Vj("y"), kp.Param("s")
    ok = kp.compile((x * y * 65504.0).sum_over("j"), dtype="float16", backend="oracle")
    assert ok.plan is not None
    with pytest.raises(ConstantConversionError, match="dépassement"):
        kp.compile((x * y * 65520.0).sum_over("j"), dtype="float16", backend="oracle")
    with pytest.raises(ConstantConversionError, match="s'arrondit à 0"):
        kp.compile((x * y * 1e-8).sum_over("j"), dtype="float16", backend="oracle")
    kp.compile((x * y * 3e-8).sum_over("j"), dtype="float16", backend="oracle")  # sous-normal FP16
    run = kp.compile((x * y * s).sum_over("j"), dtype="float16", backend="oracle")
    xs, ys = torch.ones(1, 1, dtype=H), torch.ones(1, 1, dtype=H)
    assert run(x=xs, y=ys, s=0.1).item() == float(torch.tensor(0.1, dtype=H))
    with pytest.raises(ConstantConversionError):
        run(x=xs, y=ys, s=1e5)
    with pytest.raises(ValidationError, match="dtype"):
        run(x=xs.float(), y=ys.float(), s=0.1)


def test_fp16_gradients():
    data = _data(3, 60, 80)
    x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 2)
    expr = (kp.exp(-(x - y).sqnorm2()) * b).sum_over("j")
    leaves16 = {k: v.to(H).requires_grad_() for k, v in data.items()}
    leaves64 = {k: v.to(H).double().requires_grad_() for k, v in data.items()}
    o16 = kp.compile(expr, dtype="float16", backend="oracle")(**leaves16)
    o64 = kp.compile(expr, dtype="float64", backend="oracle")(**leaves64)
    g16 = torch.autograd.grad(o16.sum(), list(leaves16.values()))
    g64 = torch.autograd.grad(o64.sum(), list(leaves64.values()))
    for a, r, t in zip(g16, g64, leaves16.values()):
        assert a.dtype == H and a.shape == t.shape
        torch.testing.assert_close(a.double(), r, rtol=2e-2, atol=2e-2)
