"""Builder for :class:`kernox.plan.Plan` (core, usable without a frontend).

This module is the only recommended way to build a ``Plan``: it validates the
plan (names, roles, bound symbols, boolean types, K, dtypes) and applies the
rules for converting constants to ``dtype`` (see ``docs/numerics.md``,
section "Constants").

Minimal example (backends, tests)::

    from kernox.builder import make_arg, make_plan
    from kernox.plan import ReductionKind
    import symox

    x, y = make_arg("x", "i", 3), make_arg("y", "j", 3)
    d2 = sum((a - b) ** 2 for a, b in zip(x.symbols, y.symbols))
    plan = make_plan([symox.exp(-d2)], ReductionKind.SUM, "float32", args=[x, y])
"""

from __future__ import annotations

import keyword
import math
from fractions import Fraction

from .errors import ConstantConversionError, UnsupportedError, ValidationError
from .plan import SUM_SCHEMES, Arg, DType, Plan, Reduction, ReductionKind

_DTYPES = ("float16", "float32", "float64")
_RANK = {"float16": 0, "float32": 1, "float64": 2}
#: accumulators allowed per dtype (as in PyKeOps: FP16 -> FP16 or FP32, FP32 -> FP32 or FP64)
_ACC_ALLOWED = {"float16": ("float16", "float32"), "float32": ("float32", "float64"), "float64": ("float64",)}
_ROLES = ("i", "j", "param")
_K_KINDS = (ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN)
_ACC_KINDS = (ReductionKind.SUM, ReductionKind.LOGSUMEXP, ReductionKind.SUMSOFTMAXWEIGHT,
              ReductionKind.LOGSUMEXPWEIGHT)
# Largest finite value and smallest positive subnormal per dtype.
_FMAX = {"float16": 65504.0, "float32": float.fromhex("0x1.fffffep+127"), "float64": float.fromhex("0x1.fffffffffffffp+1023")}


def component_symbol(name: str, index: int):
    """Real scalar symox symbol for component ``index`` of argument ``name``."""
    import symox

    return symox.Symbol(f"{name}[{index}]", real=True)


def check_name(name) -> str:
    if not isinstance(name, str) or not name.isidentifier() or not name.isascii():
        raise ValidationError(f"invalid argument name {name!r}: an ASCII identifier is required")
    if keyword.iskeyword(name) or name.startswith(("_kp", "_sx")):
        raise ValidationError(f"reserved argument name: {name!r}")
    return name


def make_arg(name: str, role: str, dim: int = 1) -> Arg:
    """Builds an :class:`Arg` and its component symbols."""
    check_name(name)
    if role not in _ROLES:
        raise ValidationError(f"invalid role {role!r} (expected one of {_ROLES})")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValidationError(f"invalid dimension {dim!r} for {name!r}: an integer >= 1 is required")
    return Arg(name, role, dim, tuple(component_symbol(name, c) for c in range(dim)))


def is_boolean(expr) -> bool:
    """True if the symox expression is boolean (same rule as the symox printers)."""
    from symox import _core

    return _core._sycl_type(expr) == "bool"


def check_k(k) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValidationError(f"K must be a Python int, got {k!r}")
    if k < 0:
        raise ValidationError(f"K must be >= 0, got {k}")
    return k


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------


def _round_to(value: float, dtype: DType) -> float:
    if dtype == "float64":
        return value
    import numpy as np

    with np.errstate(over="ignore", under="ignore"):
        return float(np.float16(value) if dtype == "float16" else np.float32(value))


def convert_float(value: float, dtype: DType, what: str = "constant") -> float:
    """Rounds a Python ``float`` to ``dtype`` (round to nearest, ties to even).

    Raises if the result overflows (finite |x| > dtype max) or if a nonzero
    value rounds to zero. Explicit ``inf``/``nan`` are kept.
    """
    if math.isnan(value) or math.isinf(value):
        return value
    rounded = _round_to(value, dtype)
    if math.isinf(rounded):
        raise ConstantConversionError(f"{what} {value!r} not representable in {dtype} (overflow)")
    if rounded == 0.0 and value != 0.0:
        raise ConstantConversionError(f"{what} {value!r} not representable in {dtype} (rounds to 0)")
    return rounded


def _exact_to_float(fr: Fraction, what) -> float:
    try:
        return float(fr)  # correctly rounded to binary64
    except OverflowError:
        raise ConstantConversionError(f"{what} not representable in float64 (overflow)") from None


def convert_constants(expr, dtype: DType):
    """Applies the plan's constant policy to a symox expression.

    - ``Float``: rounded once to ``dtype`` (to nearest, ties to even); the
      resulting ``Float`` (53-bit precision) holds exactly the dtype value;
      the printer emits it with enough digits to identify it.
      A ``Float`` with more than 53 bits of precision is first rounded to
      binary64.
    - Exact ``Rational``/``Integer``: kept exact (the printers emit ``p/q``
      in the dtype, which yields the correctly rounded p/q when p and q are
      representable); only their representability is checked.
    - Overflow, or underflow of a nonzero value to 0:
      :class:`ConstantConversionError`.
    """
    import symox

    if dtype not in _DTYPES:
        raise ValidationError(f"invalid dtype {dtype!r}")
    replacements = {}
    for atom in expr.atoms(symox.Number):
        if atom.is_Float:
            value = float(atom)
            what = f"constant {atom}"
            new = convert_float(value, dtype, what)
            if new != value or getattr(atom, "_prec", 53) != 53:
                replacements[atom] = symox.Float(new)
        elif atom.is_Rational:
            p, q = int(atom.p), int(atom.q)
            what = f"constant {atom}"
            for part in {p, q} - {0, 1, -1}:
                convert_float(_exact_to_float(Fraction(part), what), dtype, what)
            convert_float(_exact_to_float(Fraction(p, q), what), dtype, what)
        elif atom.is_finite is False or atom is symox.nan:
            continue
        else:  # non-real (I...) or unknown numbers
            raise UnsupportedError(f"non-real or unsupported constant: {atom}")
    return expr.xreplace(replacements) if replacements else expr


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


def realify(expr):
    """Maps the engine's complex artifacts back to real IEEE arithmetic.

    All Kernox variables are real, but the symbolic engine (like SymPy)
    distributes functions over the branches of a ``Piecewise`` and cannot
    always prove that a guarded branch is real:

    - ``sqrt(where(c, a, -1.5))`` becomes ``Piecewise((sqrt(a), c), (1.22*I, True))``
      and ``1/where(c, x, 0)`` introduces ``zoo``; in branch values, ``I`` and
      ``zoo`` are replaced with ``nan``, which is the real IEEE result
      (``sqrt(-1.5)``, ``0/0``); a branch reached this way is out of domain
      and the enclosing ``where`` discards it;
    - ``abs(exp(z))`` becomes ``exp(re(z))``: ``re(a)`` is ``a`` and ``im(a)``
      is 0 for a real expression (without ``I``).

    An ``I`` or ``zoo`` outside any branch is still rejected by :func:`make_plan`.
    """
    import symox

    bad = {symox.I: symox.nan, symox.zoo: symox.nan}
    memo = {}  # shared DAG: each subexpression only once (exponential otherwise)

    def walk(e):
        if not e.args:
            return e
        hit = memo.get(e)
        if hit is not None:
            return hit
        if isinstance(e, symox.Piecewise):
            out = symox.Piecewise(
                *((walk(v).xreplace(bad), walk(c)) for v, c in e.args)
            )
        else:
            out = e.func(*(walk(a) for a in e.args))
        memo[e] = out
        return out

    if not expr.has(symox.Piecewise, symox.re, symox.im):
        return expr
    out = walk(expr)
    for _ in range(64):
        new = out.replace(lambda n: isinstance(n, symox.re) and not n.args[0].has(symox.I),
                          lambda n: n.args[0])
        new = new.replace(lambda n: isinstance(n, symox.im) and not n.args[0].has(symox.I),
                          lambda n: symox.S.Zero)
        if new == out:
            break
        out = walk(new)
    return out


def fold_constants(expr):
    """Numerically evaluates non-atomic constant subexpressions.

    ``cos(1)``, ``sqrt(2)``, ``log(3)`` stay symbolic in the engine and would
    be printed as is; ``cos(1)`` in double is ambiguous for NVRTC
    (``cos(float)``/``cos(double)`` overloads for an integer argument). The
    value is computed in extended precision, then rounded to the dtype by
    :func:`convert_constants`.
    """
    import symox

    memo = {}  # shared DAG: each subexpression only once (exponential otherwise)

    def walk(e):
        if not e.args:
            return e
        hit = memo.get(e)
        if hit is not None:
            return hit
        boolean = isinstance(e, symox.logic.boolalg.Boolean)
        if not boolean and getattr(e, "is_number", False) and not e.free_symbols:
            v = e.evalf(30)
            if v.is_Float or v.is_Rational:
                out = symox.Float(v, 30) if v.is_Float else v
            else:
                out = e
        else:
            out = e.func(*(walk(a) for a in e.args))
        memo[e] = out
        return out

    return walk(expr)


def make_plan(
    outputs,
    reduction,
    dtype: DType,
    *,
    args,
    k: int | None = None,
    acc_dtype: DType | None = None,
    fp_contract: bool = True,
    sum_scheme: str | None = "direct",
    meta: dict | None = None,
    fast_math: bool = False,
) -> Plan:
    """Builds and validates a :class:`Plan`.

    ``outputs``: sequence of scalar symox expressions (or Python values/booleans).
    ``reduction``: :class:`ReductionKind`, its value (``"sum"``...) or a
    :class:`Reduction`. ``args``: sequence of :class:`Arg`, kept in the given
    order (the frontend provides the canonical order i, j, param, then name).
    ``fp_contract``: intra-expression FMA contraction allowed (contract v0.3).
    ``fast_math`` (amendment v0.10): approximate exp/log allowed (opt-in, IEEE by default).
    ``sum_scheme``: ``"direct"``, ``"block"`` or ``"kahan"`` (amendment v0.7); ``None``:
    default, ``"block"`` for FP16 accumulated in FP16 (as in PyKeOps), ``"direct"`` otherwise.
    """
    import symox

    if not isinstance(fast_math, bool):
        raise ValidationError(f"fast_math must be a bool, got {fast_math!r}")
    if not isinstance(fp_contract, bool):
        raise ValidationError(f"fp_contract must be a bool, got {fp_contract!r}")
    if sum_scheme is not None and sum_scheme not in SUM_SCHEMES:
        raise ValidationError(f"invalid sum_scheme {sum_scheme!r} (expected one of {SUM_SCHEMES})")
    if dtype not in _DTYPES:
        raise ValidationError(f"invalid dtype {dtype!r} (expected one of {_DTYPES})")

    if isinstance(reduction, Reduction):
        if k is not None and k != reduction.k:
            raise ValidationError("K given twice with different values")
        kind, k = reduction.kind, reduction.k
    else:
        try:
            kind = ReductionKind(reduction)
        except ValueError:
            raise ValidationError(f"unknown reduction {reduction!r}") from None
    if kind in _K_KINDS:
        if k is None:
            raise ValidationError(f"K is required for {kind.value}")
        k = check_k(k)
    elif k is not None:
        raise ValidationError(f"K is meaningless for {kind.value}")
    if acc_dtype is None:
        # amendment v0.9: FP16 accumulates in FP32 by default (SUM and LOGSUMEXP families)
        acc_dtype = "float32" if dtype == "float16" and kind in _ACC_KINDS else dtype
    if acc_dtype not in _DTYPES:
        raise ValidationError(f"invalid acc_dtype {acc_dtype!r}")
    if _RANK[acc_dtype] < _RANK[dtype]:
        raise ValidationError(f"acc_dtype {acc_dtype} is narrower than dtype {dtype}")
    if acc_dtype not in _ACC_ALLOWED[dtype]:
        raise ValidationError(f"acc_dtype {acc_dtype} not allowed for dtype {dtype} "
                              f"(expected one of {_ACC_ALLOWED[dtype]}, as in PyKeOps)")
    if sum_scheme is None:
        # amendment v0.9 (c): acc_dtype="float16" reproduces PyKeOps (half, block_sum)
        sum_scheme = "block" if acc_dtype == "float16" and kind in _ACC_KINDS else "direct"
    if acc_dtype != dtype and kind not in _ACC_KINDS:
        raise ValidationError("acc_dtype different from dtype is only defined for SUM, LOGSUMEXP, "
                              "SUMSOFTMAXWEIGHT and LOGSUMEXPWEIGHT")

    args = tuple(args)
    seen_names, seen_symbols = set(), set()
    for a in args:
        if not isinstance(a, Arg):
            raise ValidationError(f"argument {a!r} is not a kernox.plan.Arg")
        check_name(a.name)
        if a.role not in _ROLES:
            raise ValidationError(f"invalid role {a.role!r} for {a.name!r}")
        if isinstance(a.dim, bool) or not isinstance(a.dim, int) or a.dim < 1 or len(a.symbols) != a.dim:
            raise ValidationError(f"inconsistent dimension/symbols for {a.name!r}")
        if a.name in seen_names:
            raise ValidationError(f"duplicate argument name: {a.name!r}")
        seen_names.add(a.name)
        for s in a.symbols:
            if not isinstance(s, symox.Symbol) or s in seen_symbols:
                raise ValidationError(f"invalid or duplicate symbol {s!r} in {a.name!r}")
            seen_symbols.add(s)
    if not any(a.role == "i" for a in args) or not any(a.role == "j" for a in args):
        raise UnsupportedError(
            "v0: the formula must depend on at least one i variable and one j variable "
            "(otherwise M or N cannot be inferred from the inputs)"
        )

    exprs = tuple(symox.sympify(e) for e in outputs)
    if not exprs:
        raise ValidationError("at least one output is required")
    exprs = tuple(realify(e) for e in exprs)
    for e in exprs:
        unbound = e.free_symbols - seen_symbols
        if unbound:
            raise ValidationError("undeclared symbols: " + ", ".join(sorted(map(str, unbound))))
        if e.has(symox.I) or e.has(symox.zoo):
            raise UnsupportedError(f"non-real expression: {e}")
    booleans = [is_boolean(e) for e in exprs]
    if kind is ReductionKind.COUNT:
        if not all(booleans):
            raise ValidationError("COUNT requires boolean outputs")
    elif any(booleans):
        raise ValidationError(
            f"{kind.value} requires real outputs; use where(...) or sum_over on a "
            "predicate (COUNT)"
        )
    if kind in (ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT) and len(exprs) < 2:
        raise ValidationError(f"{kind.name}: outputs (f, g_1, ..., g_E) with E >= 1")
    if kind in _K_KINDS and len(exprs) != 1:
        raise UnsupportedError(f"{kind.value}: scalar formula only in v0 (out_dim={len(exprs)})")

    exprs = tuple(convert_constants(fold_constants(e), dtype) for e in exprs)
    return Plan(
        args=args,
        outputs=exprs,
        reduction=Reduction(kind, k),
        dtype=dtype,
        acc_dtype=acc_dtype,
        fp_contract=fp_contract,
        sum_scheme=sum_scheme,
        fast_math=fast_math,
        meta=dict(meta or {}),
    )
