"""kernox frontend: i/j variables, parameters, formulas.

A :class:`Formula` is a vector (dimension ``dim``) of scalar symox
expressions; the components of variables are real ``symox.Symbol`` objects.
No string is parsed: Python operators directly build the symox DAG. See
the Concepts and Formula operations pages of the documentation.
"""

from __future__ import annotations

import numbers
from fractions import Fraction

from .builder import check_k, check_name, component_symbol, is_boolean, make_arg, make_plan
from .errors import ValidationError
from .plan import Plan, ReductionKind

_ROLE_ORDER = {"i": 0, "j": 1, "param": 2}


# --------------------------------------------------------------------------
# Python constants -> symox
# --------------------------------------------------------------------------


def constant(value, *, boolean: bool = False):
    """Convert a Python constant into an **exact** symox number.

    - ``bool``: ``symox.true/false`` in a boolean context, otherwise integer 0/1
      (Python semantics);
    - ``int`` → exact ``Integer`` (arbitrary size);
    - ``float`` → 53-bit precision ``Float`` carrying exactly the binary64
      value of the Python ``float`` (no FP32 rounding at this stage); ``inf`` →
      ``oo``, ``nan`` → ``nan``;
    - ``fractions.Fraction`` → exact ``Rational``;
    - NumPy integer/floating scalars: like ``int``/``float`` (an
      ``np.float32`` is converted exactly to binary64);
    - symox numbers (``Rational(1, 3)``, ``pi``, ``E``...): kept exact.

    Rounding to the plan's dtype only happens when the plan is built
    (:func:`kernox.builder.convert_constants`).
    """
    import math

    import symox

    if isinstance(value, bool) or (type(value).__name__ == "bool_" and type(value).__module__ == "numpy"):
        value = bool(value)
        if boolean:
            return symox.true if value else symox.false
        return symox.Integer(int(value))
    if isinstance(value, numbers.Integral):
        return symox.Integer(int(value))
    if isinstance(value, Fraction):
        return symox.Rational(value.numerator, value.denominator)
    if isinstance(value, numbers.Real) and not isinstance(value, symox.Basic):
        v = float(value)
        if math.isnan(v):
            return symox.nan
        if math.isinf(v):
            return symox.oo if v > 0 else -symox.oo
        return symox.Float(v)
    if isinstance(value, symox.Basic):
        if value.free_symbols:
            raise TypeError(
                "symox expression with free symbols is not accepted as a constant: use Vi/Vj/Param "
                f"({sorted(map(str, value.free_symbols))})"
            )
        return value
    raise TypeError(f"unsupported constant: {value!r} ({type(value).__name__})")


# --------------------------------------------------------------------------
# Formula
# --------------------------------------------------------------------------


def _merge_vars(*maps):
    out = {}
    for m in maps:
        for name, spec in m.items():
            if out.setdefault(name, spec) != spec:
                raise ValidationError(
                    f"variable {name!r} declared with incompatible roles/dimensions: "
                    f"{out[name]} and {spec}"
                )
    return out


def _complex_cls():
    from .complex import ComplexFormula

    return ComplexFormula


def _complex_operand(o) -> bool:
    """Complex operand: ComplexFormula or non-real Python complex scalar."""
    if isinstance(o, numbers.Complex) and not isinstance(o, numbers.Real):
        return True
    return type(o).__name__ == "ComplexFormula" and isinstance(o, _complex_cls())


class Formula:
    """Vector of scalar symox expressions and the variables it depends on.

    ``components``: tuple of symox expressions; ``dim``: length;
    ``variables``: ``{name: (role, dim)}``.
    """

    __slots__ = ("_comps", "_vars", "_bool")
    __array_priority__ = 1000  # so that ndarray.__add__ defers to us

    def __init__(self, components, variables):
        comps = tuple(components)
        if not comps:
            raise ValidationError("empty formula")
        kinds = {is_boolean(c) for c in comps}
        if len(kinds) != 1:
            raise ValidationError("formula mixes boolean and real components")
        self._comps = comps
        self._vars = dict(variables)
        self._bool = kinds.pop()

    # ---- introspection
    @property
    def components(self):
        return self._comps

    @property
    def dim(self) -> int:
        return len(self._comps)

    @property
    def variables(self) -> dict:
        return dict(self._vars)

    @property
    def is_boolean(self) -> bool:
        return self._bool

    def __len__(self):
        return len(self._comps)

    def __repr__(self):
        kind = "bool" if self._bool else "real"
        return f"Formula(dim={self.dim}, {kind}, {list(self._comps)})"

    def __bool__(self):
        raise TypeError(
            "a Formula has no truth value; use &, |, ~ and where(...) "
            "instead of and/or/not/if"
        )

    __hash__ = None  # unhashable: a formula is not a key; == stays identity

    # ---- internal construction
    @staticmethod
    def _lift(value, boolean=False) -> "Formula":
        if isinstance(value, Formula):
            return value
        if isinstance(value, ReductionExpr):
            raise TypeError("a reduction cannot be used inside a formula (v0)")
        return Formula((constant(value, boolean=boolean),), {})

    @staticmethod
    def _broadcast(*formulas):
        dims = {f.dim for f in formulas} - {1}
        if len(dims) > 1:
            raise ValidationError(f"incompatible dimensions: {[f.dim for f in formulas]}")
        d = dims.pop() if dims else 1
        return d, [f._comps * d if f.dim == 1 and d != 1 else f._comps for f in formulas]

    @classmethod
    def _map(cls, fn, *operands, boolean_in=None, boolean=False):
        fs = [cls._lift(o, boolean=boolean) for o in operands]
        if boolean_in is not None:
            for f in fs:
                if f._bool != boolean_in:
                    if boolean_in:
                        raise TypeError("expected a boolean operand")
                    raise TypeError(
                        "arithmetic operation on a boolean formula: use "
                        "where(pred, a, b)"
                    )
        _, comps = cls._broadcast(*fs)
        return Formula((fn(*args) for args in zip(*comps)), _merge_vars(*(f._vars for f in fs)))

    # ---- arithmetic
    def __add__(self, o):
        if _complex_operand(o):
            return self.real2complex() + o
        return Formula._map(lambda a, b: a + b, self, o, boolean_in=False)

    def __radd__(self, o):
        if _complex_operand(o):
            return o + self.real2complex()
        return Formula._map(lambda a, b: a + b, o, self, boolean_in=False)

    def __sub__(self, o):
        if _complex_operand(o):
            return self.real2complex() - o
        return Formula._map(lambda a, b: a - b, self, o, boolean_in=False)

    def __rsub__(self, o):
        if _complex_operand(o):
            return o - self.real2complex()
        return Formula._map(lambda a, b: a - b, o, self, boolean_in=False)

    def __mul__(self, o):
        if _complex_operand(o):
            return _complex_cls()._real_scal_or_mult(o, self)
        return Formula._map(lambda a, b: a * b, self, o, boolean_in=False)

    def __rmul__(self, o):
        if _complex_operand(o):
            return _complex_cls()._real_scal_or_mult(o, self)
        return Formula._map(lambda a, b: a * b, o, self, boolean_in=False)

    def __truediv__(self, o):
        if _complex_operand(o):
            return self.real2complex() / o
        return Formula._map(lambda a, b: a / b, self, o, boolean_in=False)

    def __rtruediv__(self, o):
        if _complex_operand(o):
            return o / self.real2complex()
        return Formula._map(lambda a, b: a / b, o, self, boolean_in=False)

    def __pow__(self, o):
        return Formula._map(lambda a, b: a**b, self, o, boolean_in=False)

    def __rpow__(self, o):
        return Formula._map(lambda a, b: a**b, o, self, boolean_in=False)

    def __neg__(self):
        return Formula._map(lambda a: -a, self, boolean_in=False)

    def __pos__(self):
        return Formula._map(lambda a: a, self, boolean_in=False)

    def __abs__(self):
        import symox

        return Formula._map(symox.Abs, self, boolean_in=False)

    # ---- comparisons (predicates). == / != stay Python identity:
    # use eq()/ne().
    def __lt__(self, o):
        return Formula._map(lambda a, b: a < b, self, o, boolean_in=False)

    def __le__(self, o):
        return Formula._map(lambda a, b: a <= b, self, o, boolean_in=False)

    def __gt__(self, o):
        return Formula._map(lambda a, b: a > b, self, o, boolean_in=False)

    def __ge__(self, o):
        return Formula._map(lambda a, b: a >= b, self, o, boolean_in=False)

    def eq(self, o):
        import symox

        return Formula._map(symox.Eq, self, o, boolean_in=False)

    def ne(self, o):
        import symox

        return Formula._map(symox.Ne, self, o, boolean_in=False)

    # ---- logic
    def __and__(self, o):
        import symox

        return Formula._map(symox.And, self, o, boolean_in=True, boolean=True)

    __rand__ = __and__

    def __or__(self, o):
        import symox

        return Formula._map(symox.Or, self, o, boolean_in=True, boolean=True)

    __ror__ = __or__

    def __xor__(self, o):
        import symox

        return Formula._map(symox.Xor, self, o, boolean_in=True, boolean=True)

    __rxor__ = __xor__

    def __invert__(self):
        import symox

        return Formula._map(symox.Not, self, boolean_in=True, boolean=True)

    # ---- components
    def __getitem__(self, index):
        d = self.dim
        if isinstance(index, bool):
            raise TypeError("boolean index not supported")
        if isinstance(index, numbers.Integral):
            i = int(index)
            if not -d <= i < d:
                raise IndexError(f"component {i} out of [0, {d})")
            comps = (self._comps[i],)
        elif isinstance(index, slice):
            comps = self._comps[index]
        elif isinstance(index, (list, tuple)):
            comps = tuple(self[i]._comps[0] for i in index)
        else:
            raise TypeError(f"unsupported index: {index!r}")
        if not comps:
            raise IndexError("empty component selection")
        return Formula(comps, self._vars)

    def sum(self, *args, **kwargs) -> "Formula":
        """Sum of the **components** (scalar). To reduce an axis: ``sum_over``."""
        if args or kwargs:
            raise TypeError(
                "Formula.sum() sums the components and takes no argument; "
                "to reduce a pairwise axis use .sum_over('j') or .sum_over('i')"
            )
        if self._bool:
            raise TypeError("sum of boolean components: use where(...)")
        import symox

        return Formula((symox.Add(*self._comps),), self._vars)

    def sqnorm2(self) -> "Formula":
        return (self * self).sum()

    def norm2(self) -> "Formula":
        return sqrt(self.sqnorm2())

    def dot(self, other) -> "Formula":
        other = Formula._lift(other)
        if self.dim != other.dim and 1 not in (self.dim, other.dim):
            raise ValidationError(f"dot: incompatible dimensions {self.dim} and {other.dim}")
        return (self * other).sum()

    # ---- pairwise reductions
    def sum_over(self, axis: str) -> "ReductionExpr":
        """Sum over the axis ``axis``; a boolean formula gives a count (COUNT)."""
        kind = ReductionKind.COUNT if self._bool else ReductionKind.SUM
        return ReductionExpr(self, kind, axis)

    def min(self, *, over: str) -> "ReductionExpr":
        return ReductionExpr(self, ReductionKind.MIN, over)

    def max(self, *, over: str) -> "ReductionExpr":
        return ReductionExpr(self, ReductionKind.MAX, over)

    def argmin(self, *, over: str) -> "ReductionExpr":
        return ReductionExpr(self, ReductionKind.ARGMIN, over)

    def argmax(self, *, over: str) -> "ReductionExpr":
        return ReductionExpr(self, ReductionKind.ARGMAX, over)

    def kmin(self, K: int, *, over: str) -> "ReductionExpr":
        return ReductionExpr(self, ReductionKind.KMIN, over, K)

    def argkmin(self, K: int, *, over: str) -> "ReductionExpr":
        return ReductionExpr(self, ReductionKind.ARGKMIN, over, K)

    def kmin_argkmin(self, K: int, *, over: str) -> "ReductionExpr":
        return ReductionExpr(self, ReductionKind.KMIN_ARGKMIN, over, K)

    def min_argmin(self, *, over: str) -> "ReductionExpr":
        """MIN and ARGMIN in a single pass (PyKeOps ``min_argmin``): ``(values, indices)``."""
        return ReductionExpr(self, ReductionKind.MIN_ARGMIN, over)

    def max_argmax(self, *, over: str) -> "ReductionExpr":
        """MAX and ARGMAX in a single pass (PyKeOps ``max_argmax``): ``(values, indices)``."""
        return ReductionExpr(self, ReductionKind.MAX_ARGMAX, over)

    def logsumexp(self, *, over: str, weight=None) -> "ReductionExpr":
        """``log Σ exp f`` over the axis ``over``, per component (PyKeOps ``logsumexp``), output (M, D).

        ``weight=g`` (PyKeOps ``logsumexp(weight=g)``): ``log Σ exp(f) g_e`` with
        scalar ``f`` and ``g`` of dimension ``E``, output (M, E).
        """
        if weight is None:
            return ReductionExpr(self, ReductionKind.LOGSUMEXP, over)
        if self.dim != 1:
            raise ValidationError(f"logsumexp(weight=): expected a scalar score (dim={self.dim})")
        g = Formula._lift(weight)
        if g.is_boolean:
            raise TypeError("logsumexp(weight=): boolean weights, use where(...)")
        return ReductionExpr(cat(self, g), ReductionKind.LOGSUMEXPWEIGHT, over)

    def sumsoftmaxweight(self, g, *, over: str) -> "ReductionExpr":
        """``Σ softmax(f) · g`` over the axis ``over`` (PyKeOps ``sumsoftmaxweight``), output (M, E).

        ``self``: scalar score ``f``; ``g``: weights of dimension ``E >= 1``.
        """
        if self.dim != 1:
            raise ValidationError(f"sumsoftmaxweight: expected a scalar score (dim={self.dim})")
        g = Formula._lift(g)
        if g.is_boolean:
            raise TypeError("sumsoftmaxweight: boolean weights, use where(...)")
        return ReductionExpr(cat(self, g), ReductionKind.SUMSOFTMAXWEIGHT, over)


def _variable(name, role, dim, is_complex=False):
    check_name(name)
    if isinstance(dim, bool) or not isinstance(dim, numbers.Integral) or dim < 1:
        raise ValidationError(f"invalid dimension {dim!r} for {name!r}")
    dim = int(dim)
    if is_complex:
        from .complex import ComplexDim, ComplexFormula

        d2 = ComplexDim(2 * dim)
        return ComplexFormula(Formula((component_symbol(name, c) for c in range(2 * dim)), {name: (role, d2)}))
    return Formula((component_symbol(name, c) for c in range(dim)), {name: (role, dim)})


def Vi(name: str, dim: int = 1, *, complex: bool = False) -> Formula:  # noqa: A002
    """Variable indexed by i: tensor (M, dim) passed under the name ``name``.

    ``complex=True``: :class:`~kernox.complex.ComplexFormula`; complex tensor
    (M, dim) or its interleaved real view (M, 2·dim).
    """
    return _variable(name, "i", dim, complex)


def Vj(name: str, dim: int = 1, *, complex: bool = False) -> Formula:  # noqa: A002
    """Variable indexed by j: tensor (N, dim) passed under the name ``name`` (``complex``: see Vi)."""
    return _variable(name, "j", dim, complex)


def Param(name: str, dim: int = 1, *, complex: bool = False) -> Formula:  # noqa: A002
    """Parameter constant over the pairs: tensor (dim,) or Python scalar if dim=1."""
    return _variable(name, "param", dim, complex)


# --------------------------------------------------------------------------
# Reductions
# --------------------------------------------------------------------------


class ReductionExpr:
    """Pairwise reduction of a formula (terminal node, compiled with :func:`kernox.compile`)."""

    __slots__ = ("formula", "kind", "over", "k")

    def __init__(self, formula: Formula, kind: ReductionKind, over: str, k=None):
        if over not in ("i", "j"):
            raise ValidationError(f"invalid reduction axis {over!r} (expected 'i' or 'j')")
        if kind in (ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN):
            k = check_k(k)
        if formula.is_boolean and kind is not ReductionKind.COUNT:
            raise TypeError(f"{kind.value} on a boolean formula: use where(...)")
        self.formula, self.kind, self.over, self.k = formula, kind, over, k

    def __repr__(self):
        k = f", K={self.k}" if self.k is not None else ""
        return f"ReductionExpr({self.kind.value}, over={self.over!r}{k}, {self.formula!r})"

    def to_plan(self, dtype, *, acc_dtype=None, fp_contract=True, sum_scheme=None, fast_math=False,
                flush_denormals=False) -> Plan:
        """Normalized plan (reduction over j): if ``over='i'``, the i/j roles are swapped."""
        swap = {"i": "j", "j": "i", "param": "param"} if self.over == "i" else None
        specs = []
        for name, (role, dim) in self.formula.variables.items():
            role = swap[role] if swap else role
            specs.append((_ROLE_ORDER[role], name, role, dim))
        args = [make_arg(name, role, int(dim)) for _, name, role, dim in sorted(specs)]
        from .complex import complex_args

        cplx = complex_args(self.formula.variables)
        plan = make_plan(
            self.formula.components,
            self.kind,
            dtype,
            args=args,
            k=self.k,
            acc_dtype=acc_dtype,
            fp_contract=fp_contract,
            sum_scheme=sum_scheme,
            fast_math=fast_math,
            flush_denormals=flush_denormals,
            meta={"over": self.over, "swapped_ij": self.over == "i"},
        )
        if cplx:
            plan.meta["complex_args"] = cplx
        return plan

    def compile(self, dtype, *, backend="auto", acc_dtype=None, tuning="estimate", fp_contract=None,
                sum_scheme=None, fast_math=None, flush_denormals=None):
        from .runtime import compile as _compile

        return _compile(self, dtype=dtype, backend=backend, acc_dtype=acc_dtype, tuning=tuning,
                        fp_contract=fp_contract, sum_scheme=sum_scheme, fast_math=fast_math,
                        flush_denormals=flush_denormals)


# --------------------------------------------------------------------------
# Common functions (component-wise)
# --------------------------------------------------------------------------


def _unary(symox_name, doc):
    def fn(x):
        import symox

        if _complex_operand(x):
            if symox_name == "exp":
                return x.exp()
            raise TypeError(f"{doc} of a complex formula is not defined (exp, abs, angle, conj, real, imag are)")
        return Formula._map(getattr(symox, symox_name), x, boolean_in=False)

    fn.__name__ = fn.__qualname__ = doc
    fn.__doc__ = f"{doc}(x) component-wise (symox.{symox_name})."
    return fn


exp = _unary("exp", "exp")
log = _unary("log", "log")
sqrt = _unary("sqrt", "sqrt")
sin = _unary("sin", "sin")
cos = _unary("cos", "cos")
tan = _unary("tan", "tan")
asin = _unary("asin", "asin")
acos = _unary("acos", "acos")
atan = _unary("atan", "atan")
sinh = _unary("sinh", "sinh")
cosh = _unary("cosh", "cosh")
tanh = _unary("tanh", "tanh")
sign = _unary("sign", "sign")
floor = _unary("floor", "floor")
ceiling = _unary("ceiling", "ceiling")


def abs(x):  # noqa: A001 - intended public name
    """Component-wise absolute value (modulus for a complex formula)."""
    import symox

    if _complex_operand(x):
        return _complex_cls()._lift(x).abs()
    return Formula._map(symox.Abs, x, boolean_in=False)


def rsqrt(x):
    """Exact ``1/sqrt(x)`` (no approximate rsqrt instruction)."""
    return 1 / sqrt(x)


def square(x):
    return Formula._lift(x) ** 2


def atan2(y, x):
    import symox

    return Formula._map(symox.atan2, y, x, boolean_in=False)


def minimum(a, b):
    """Component-wise minimum (symox ``Min``)."""
    import symox

    return Formula._map(symox.Min, a, b, boolean_in=False)


def maximum(a, b):
    import symox

    return Formula._map(symox.Max, a, b, boolean_in=False)


def clamp(x, lo, hi):
    return minimum(maximum(x, lo), hi)


def eq(a, b):
    return Formula._lift(a).eq(b)


def ne(a, b):
    return Formula._lift(a).ne(b)


def where(cond, a, b):
    """``a`` if ``cond`` else ``b``, component-wise (``Piecewise``).

    The result is always that of the active branch: an invalid value in the
    inactive branch (``sqrt(-1)``, ``log(0)``, ``1/0``) affects neither the
    value nor the gradient.
    """
    import symox

    c = Formula._lift(cond, boolean=True)
    if not c.is_boolean:
        raise TypeError("where: the condition must be a boolean predicate")
    fa, fb = Formula._lift(a), Formula._lift(b)
    if fa.is_boolean != fb.is_boolean:
        raise TypeError("where: mixed boolean and real branches")
    _, (cc, ca, cb) = Formula._broadcast(c, fa, fb)
    comps = (symox.Piecewise((x, k), (y, True)) for k, x, y in zip(cc, ca, cb))
    return Formula(comps, _merge_vars(c._vars, fa._vars, fb._vars))


def cat(*formulas):
    """Concatenation of the components."""
    fs = [Formula._lift(f) for f in formulas]
    if not fs:
        raise ValidationError("cat() called without arguments")
    comps = tuple(c for f in fs for c in f.components)
    return Formula(comps, _merge_vars(*(f._vars for f in fs)))


__all__ = [
    "Formula", "ReductionExpr", "Vi", "Vj", "Param", "constant",
    "exp", "log", "sqrt", "rsqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "abs", "sign", "floor", "ceiling", "square",
    "minimum", "maximum", "clamp", "where", "eq", "ne", "cat",
]
