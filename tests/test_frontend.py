from fractions import Fraction

import numpy as np
import pytest
import symox

import kernox as kp
from kernox import Param, Vi, Vj, exp, sqrt, where
from kernox.errors import ConstantConversionError, UnsupportedError, ValidationError
from kernox.plan import ReductionKind


def floats(expr):
    return sorted(float(a) for a in expr.atoms(symox.Float))


# ---------------------------------------------------------------- constants
def test_python_constants_without_intcst():
    x, y = Vi("x", 2), Vj("y", 2)
    f = 0 + 2.0 * x - y * 0.25 + 1 + True + Fraction(1, 3) + symox.Rational(2, 7)
    p = f.sum().sum_over("j").to_plan("float64")
    (e,) = p.outputs
    assert 0.25 in floats(e) or -0.25 in floats(e)
    assert (x ** 2).components[0].exp == 2 and (x ** 2).components[0].exp.is_Integer


def test_float_constants_exact_in_float64_rounded_in_float32():
    x, y = Vi("x"), Vj("y")
    r = (0.1 * x * y).sum_over("j")
    (e64,) = r.to_plan("float64").outputs
    (e32,) = r.to_plan("float32").outputs
    assert floats(e64) == [0.1]
    assert floats(e32) == [float(np.float32(0.1))]
    assert float(np.float32(0.1)) != 0.1


def test_numpy_scalars_are_exact():
    x, y = Vi("x"), Vj("y")
    (e,) = (np.float32(0.1) * x * y).sum_over("j").to_plan("float64").outputs
    assert floats(e) == [float(np.float32(0.1))]


@pytest.mark.parametrize("c", [1e300, 1e-300, 10**40])
def test_unrepresentable_float32_constant_errors(c):
    x, y = Vi("x"), Vj("y")
    r = (c * x * y).sum_over("j")
    r.to_plan("float64")
    with pytest.raises(ConstantConversionError):
        r.to_plan("float32")


def test_unrepresentable_float64_constant_errors():
    x, y = Vi("x"), Vj("y")
    with pytest.raises(ConstantConversionError):
        (symox.Rational(1, 10**400) * x * y).sum_over("j").to_plan("float64")
    with pytest.raises(ConstantConversionError):
        (symox.Integer(10**400) * x * y).sum_over("j").to_plan("float64")


def test_inf_constant_kept():
    x, y = Vi("x"), Vj("y")
    p = where(x < y, x, float("inf")).min(over="j").to_plan("float32")
    assert p.outputs[0].has(symox.oo)


def test_bad_constant_type():
    with pytest.raises(TypeError):
        Vi("x") + "1.0"
    with pytest.raises(TypeError):
        Vi("x") + symox.Symbol("z")


# ---------------------------------------------------------------- formulas
def test_broadcast_and_dims():
    x, y = Vi("x", 3), Vj("y", 3)
    assert (x - y).dim == 3 and (x * Param("s")).dim == 3
    with pytest.raises(ValidationError):
        x + Vj("z", 2)
    assert x[0].dim == 1 and x[1:].dim == 2 and x[[0, 2]].dim == 2 and x[-1].dim == 1
    with pytest.raises(IndexError):
        x[3]
    assert kp.cat(x, y[0]).dim == 4


def test_component_sum_vs_axis_sum():
    x, y = Vi("x", 3), Vj("y", 3)
    s = (x - y).sum()
    assert s.dim == 1
    with pytest.raises(TypeError, match="sum_over"):
        (x - y).sum("j")
    with pytest.raises(TypeError, match="sum_over"):
        (x - y).sum(over="j")
    r = (x - y).sum_over("j")
    assert r.kind is ReductionKind.SUM and r.to_plan("float32").out_dim == 3


def test_symbols_are_scalar_symox_symbols():
    x = Vi("x", 2)
    assert all(isinstance(c, symox.Symbol) and c.is_real for c in x.components)


def test_name_conflicts():
    with pytest.raises(ValidationError):
        Vi("x", 3) + Vj("x", 3)
    with pytest.raises(ValidationError):
        Vi("x", 3)[0] + Vi("x", 2)[0]
    with pytest.raises(ValidationError):
        Vi("_kp_x")
    with pytest.raises(ValidationError):
        Vi("for")
    with pytest.raises(ValidationError):
        Vi("x", 0)


def test_predicates_and_count():
    x, y = Vi("x", 3), Vj("y", 3)
    d2 = ((x - y) ** 2).sum()
    pred = (d2 <= 0.25) & ~(d2 < 0.0)
    assert pred.is_boolean
    r = pred.sum_over("j")
    assert r.kind is ReductionKind.COUNT
    with pytest.raises(TypeError):
        pred + 1
    with pytest.raises(TypeError):
        pred.min(over="j")
    with pytest.raises(TypeError):
        bool(d2 < 1)
    with pytest.raises(TypeError):
        d2 & pred
    assert kp.eq(x[0], y[0]).is_boolean and x[0].ne(1).is_boolean


def test_where_and_functions():
    x, y = Vi("x", 3), Vj("y", 3)
    d2 = ((x - y) ** 2).sum()
    g = where(d2 < 4.5, exp(-d2), 0.0)
    assert g.components[0].has(symox.Piecewise)
    for fn in (kp.exp, kp.log, kp.sqrt, kp.sin, kp.cos, kp.tan, kp.abs, kp.tanh, kp.sinh,
               kp.cosh, kp.atan, kp.sign, kp.rsqrt, kp.square):
        assert fn(x).dim == 3
    assert kp.atan2(x, y).dim == 3 and kp.minimum(x, 0).dim == 3 and kp.clamp(x, -1, 1).dim == 3
    assert abs(x).dim == 3 and (-x).dim == 3 and (2 ** x).dim == 3
    with pytest.raises(TypeError):
        where(d2, 1, 0)
    with pytest.raises(TypeError):
        where(d2 < 1, d2 < 2, 0.0)


def test_k_validation():
    x, y = Vi("x"), Vj("y")
    d = (x - y) ** 2
    for bad in (-1, 1.5, True, "3", None):
        with pytest.raises(ValidationError):
            d.kmin(bad, over="j")
    with pytest.raises(ValidationError):
        d.min(over="k")
    with pytest.raises(UnsupportedError):
        (Vi("a", 2) - Vj("b", 2)).argkmin(3, over="j").to_plan("float32")
    assert d.kmin(0, over="j").to_plan("float32").reduction.k == 0


def test_over_i_swaps_roles():
    x, y, s = Vi("x", 3), Vj("y", 2), Param("s")
    p = (x[0] * y[1] * s).argmin(over="i").to_plan("float32")
    roles = {a.name: a.role for a in p.args}
    assert roles == {"x": "j", "y": "i", "s": "param"}
    assert [a.role for a in p.args] == ["i", "j", "param"]
    assert p.meta["swapped_ij"] is True


def test_requires_i_and_j():
    with pytest.raises(UnsupportedError):
        (Vi("x") * 2).sum_over("j").to_plan("float32")


def test_acc_dtype_rules():
    x, y = Vi("x"), Vj("y")
    r = (x * y).sum_over("j")
    assert r.to_plan("float32", acc_dtype="float64").acc_dtype == "float64"
    with pytest.raises(ValidationError):
        r.to_plan("float64", acc_dtype="float32")
    with pytest.raises(ValidationError):
        (x * y).min(over="j").to_plan("float32", acc_dtype="float64")
    with pytest.raises(ValidationError):
        r.to_plan("bfloat16")
    assert r.to_plan("float16").acc_dtype == "float32"  # v0.9 amendment


def test_builder_minimal():
    from kernox.builder import make_arg, make_plan
    x, y = make_arg("x", "i", 3), make_arg("y", "j", 3)
    d2 = sum((a - b) ** 2 for a, b in zip(x.symbols, y.symbols))
    p = make_plan([symox.exp(-d2)], "sum", "float32", args=[x, y])
    assert p.out_dim == 1 and p.args_of("i") == (x,)
    with pytest.raises(ValidationError):
        make_plan([symox.Symbol("z")], "sum", "float32", args=[x, y])
    with pytest.raises(ValidationError):
        make_plan([d2 < 1], "sum", "float32", args=[x, y])
    with pytest.raises(ValidationError):
        make_plan([d2], "count", "float32", args=[x, y])
    with pytest.raises(ValidationError):
        make_plan([d2], "kmin", "float32", args=[x, y])


def test_realify_guarded_complex_branches():
    import symox

    from kernox.builder import realify

    yv = symox.Symbol("y", real=True)
    inner = symox.Piecewise((symox.sqrt(yv), yv >= 0), (-1.5, True))
    out = realify(symox.sqrt(inner))
    assert not out.has(symox.I) and out.has(symox.nan)
    pw = symox.Piecewise((symox.sqrt(yv), yv >= 0), (0.5, True))
    out = realify(symox.Abs(symox.exp(3 * symox.sin(pw))))
    assert not out.has(symox.re, symox.im)


def test_unguarded_complex_constant_still_refused():
    from kernox import UnsupportedError

    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    with pytest.raises(UnsupportedError):
        (kp.sqrt(kp.constant(-1.5)) * x[0] * y[0]).sum_over("j").to_plan("float32")


def test_evaluate_numpy_keeps_full_float_precision():
    import numpy as np

    from kernox.oracle import evaluate_numpy

    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    plan = (0.5**0.5 * (1 + x[0] * y[0])).sum_over("j").to_plan("float64")
    vals = evaluate_numpy(plan, {"x": np.zeros((2, 1)), "y": np.zeros((3, 1))})
    assert vals[0, 0, 0] == 0.5**0.5


def test_constant_function_calls_are_folded():
    import symox

    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    for dt in ("float32", "float64"):
        plan = (kp.maximum(kp.cos(-1), x[0] * y[0]) + where(x[0] < kp.log(3), kp.sqrt(2), y[0])).sum_over("j").to_plan(dt)
        out = plan.outputs[0]
        assert not out.has(symox.cos, symox.log) and not out.has(symox.sqrt(2))
