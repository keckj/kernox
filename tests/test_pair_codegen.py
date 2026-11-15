import re

import pytest
import symox

import kernox as kp

from kernox import Param, Vi, Vj, exp, sqrt, where
from kernox.codegen.pair import DIALECTS, pair_function
from kernox.errors import UnsupportedError, ValidationError

from corpus import reductions


@pytest.mark.parametrize("name", list(reductions()))
@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_structure(name, dialect, dtype):
    plan = reductions()[name].to_plan(dtype)
    ps = pair_function(plan, dialect)
    T = "double" if dtype == "float64" else "float"
    src = ps.source
    assert ps.name == "kp_pair" and ps.out_type == "kp_out" and ps.scalar_type == T
    assert len(ps.arg_layout) == len(plan.args)
    assert [l.arg for l in ps.arg_layout] == [a.name for a in plan.args]
    qual = "__device__ __forceinline__" if dialect != "sycl" else "inline"
    params = ", ".join(f"const {T}* __restrict__ {l.param}" for l in ps.arg_layout)
    assert f"{qual} kp_out kp_pair({params})" in src
    want = "bool" if plan.reduction.kind.value == "count" else T
    assert ps.out_ctypes == (want,) * plan.out_dim
    for f in ps.out_fields:
        assert f"    {want} {f};" in src
    # numerical policy: no fast-math, no rsqrt, no integer pow
    for bad in ("__expf", "__fdividef", "__logf", "rsqrt", "__fsqrt", "M_PI", "M_E"):
        assert bad not in src
    assert not re.search(r"pow[f]?\([^()]*(\([^()]*\))*[^()]*,\s*[234]\)", src)
    if dtype == "float64":
        assert not re.search(r"\d[fF]\b", src.split("struct kp_out")[1]), "float literal in FP64 code"


def test_layout_names_follow_roles():
    x, y, z, p, q = Vi("x", 2), Vj("y", 2), Vi("z"), Param("p"), Param("q", 2)
    plan = (x[0] * y[1] + z * p + q[1]).sum_over("j").to_plan("float32")
    ps = pair_function(plan, "cuda")
    assert [(l.param, l.arg, l.role, l.dim) for l in ps.arg_layout] == [
        ("xi_0", "x", "i", 2), ("xi_1", "z", "i", 1), ("yj_0", "y", "j", 2),
        ("p_0", "p", "param", 1), ("p_1", "q", "param", 2)]
    # unused component not loaded; fully unused pointer: (void)
    assert "xi_0[1]" not in ps.source and "q[0]" not in ps.source and "p_1[0]" not in ps.source


def test_unused_pointer_is_voided():
    x, y, z = Vi("x"), Vj("y"), Vj("z")
    plan = (x * y + 0 * z).sum_over("j").to_plan("float32")
    assert "(void)yj_1;" in pair_function(plan, "sycl").source


def test_cse_without_conditionals():
    x, y = Vi("x", 3), Vj("y", 3)
    d2 = ((x - y) ** 2).sum()
    plan = (exp(-d2) * sqrt(d2) + d2 * exp(-d2)).sum_over("j").to_plan("float64")
    src = pair_function(plan, "cuda").source
    assert "_kp_t0" in src or "_sx_t0" in src


def test_region_cse_with_where():
    """Region-based CSE: d2 (condition) and exp(-d2) (both branches) are hoisted,
    the square root specific to the active branch stays inside the ternary."""
    x, y = Vi("x", 3), Vj("y", 3)
    d2 = ((x - y) ** 2).sum()
    f = where(d2 > 1.0, 1 / sqrt(d2 - 1.0) + exp(-d2), exp(-d2))
    plan = f.sum_over("j").to_plan("float64")
    for dialect in DIALECTS:
        src = pair_function(plan, dialect).source
        body = src.split("kp_pair(")[1]
        assert "_kp_t" in body, dialect
        assert len(re.findall(r"\bexp\(|sycl::exp\(", body)) == 1, dialect
        first_q = body.index("?")
        assert all(m.start() > first_q for m in re.finditer(r"sqrt|pow\(", body)), dialect


def test_region_cse_single_branch_not_hoisted():
    x, y = Vi("x", 1), Vj("y", 1)
    g = kp.log(x[0] - y[0])
    plan = where(x[0] > y[0], g * g + g, 0.0).sum_over("j").to_plan("float32")
    body = pair_function(plan, "cuda").source.split("kp_pair(")[1]
    first_q = body.index("?")
    assert all(m.start() > first_q for m in re.finditer(r"logf", body))


def test_region_cse_vector_truncated_gaussian():
    x, y, b = Vi("x", 3), Vj("y", 3), Vj("b", 16)
    d2 = ((x - y) ** 2).sum()
    plan = where(d2 < 4.0, exp(-d2) * b, 0.0).sum_over("j").to_plan("float32")
    body = pair_function(plan, "cuda").source.split("kp_pair(")[1]
    assert body.count("expf(") == 1 and body.count("kp_pow2(") == 3


def test_rename_and_ident_validation():
    x, y = Vi("x"), Vj("y")
    plan = (x * y).sum_over("j").to_plan("float32")
    ps = pair_function(plan, "hip", name="pair_a", out_type="out_a")
    assert "struct out_a" in ps.source and "out_a pair_a(" in ps.source
    with pytest.raises(ValidationError):
        pair_function(plan, "opencl")
    with pytest.raises(ValidationError):
        pair_function(plan, "cuda", name="1abc")
    from dataclasses import replace
    with pytest.raises(UnsupportedError):
        pair_function(replace(plan, fast_math=True), "cuda")


def test_constants_printed_in_dtype():
    x, y = Vi("x"), Vj("y")
    r = (0.1 * x * y + symox.Rational(1, 3)).sum_over("j")
    s32 = pair_function(r.to_plan("float32"), "cuda").source
    s64 = pair_function(r.to_plan("float64"), "cuda").source
    assert "0.100000001F" in s32 and "3.0F" in s32
    assert "0.10000000000000001" in s64 and "3.0F" not in s64
