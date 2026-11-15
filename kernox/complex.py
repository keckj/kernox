"""Complex formulas (parity with PyKeOps ``ComplexLazyTensor``, keopscore ``formulas/complex``).

keopscore representation: a complex formula of dimension D is a real
:class:`~kernox.expr.Formula` of dimension 2D, with **interleaved**
components ``(re_0, im_0, re_1, im_1, …)``. Each operation is the
keopscore definition re-expressed with real symox primitives (no complex
type in the engine nor in the backends):

- ``Real2Complex(x) = (x, 0)``, ``Imag2Complex(x) = (0, x)``,
  ``ComplexExp1j(x) = (cos x, sin x)``;
- ``ComplexReal``, ``ComplexImag``, ``Conj = (re, −im)``,
  ``ComplexSquareAbs = re² + im²``, ``ComplexAbs = sqrt(re² + im²)``,
  ``ComplexAngle = atan2(im, re)``,
  ``ComplexExp = (exp(re)·cos(im), exp(re)·sin(im))``;
- ``ComplexAdd``, ``ComplexSubtract``, ``ComplexMult = (ac − bd, ad + bc)``,
  ``ComplexRealScal`` (real of dimension 1 × complex),
  ``ComplexDivide(f, g) = Real2Complex(1/|g|²) · (f · conj(g))``,
  ``ComplexSum`` (sum of the complex components);
- broadcasting of a complex dimension 1 to D, like ``VectorizedComplexScalarOp``.

Inputs: ``Vi(name, D, complex=True)`` (likewise ``Vj``, ``Param``) expects a
complex tensor ``(…, L, D)`` (``complex64`` in FP32, ``complex128`` in FP64)
or its interleaved real view ``(…, L, 2D)``. Reduction: ``z.sum_over(ax)`` returns
a complex tensor ``(…, M, D)``. Gradients go through
``torch.view_as_real`` (Torch convention for complex numbers).
"""

from __future__ import annotations

import numbers

from .errors import ValidationError
from .expr import Formula, ReductionExpr, _merge_vars
from .plan import ReductionKind


class ComplexDim(int):
    """Real dimension (2D) of a complex variable: marker carried by ``Formula._vars``."""

    def __eq__(self, other):
        return isinstance(other, ComplexDim) and int(self) == int(other)

    def __ne__(self, other):
        return not self == other

    __hash__ = int.__hash__

    def __repr__(self):
        return f"complex({int(self) // 2})"


def complex_args(variables: dict) -> tuple:
    """Names of the complex variables of a dictionary ``{name: (role, dim)}``."""
    return tuple(sorted(n for n, (_r, d) in variables.items() if isinstance(d, ComplexDim)))


def _pairs(f: Formula):
    c = f.components
    return [(c[2 * k], c[2 * k + 1]) for k in range(len(c) // 2)]


def _from_pairs(pairs, variables) -> "ComplexFormula":
    return ComplexFormula(Formula([v for p in pairs for v in p], variables))


def _lift(value) -> "ComplexFormula":
    """Operand to ComplexFormula: complex, real (Real2Complex), Python constant."""
    import symox

    if isinstance(value, ComplexFormula):
        return value
    if isinstance(value, Formula):
        return value.real2complex()
    if isinstance(value, ReductionExpr):
        raise TypeError("a reduction cannot be used inside a formula (v0)")
    if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
        re, im = Formula._lift(float(value.real)), Formula._lift(float(value.imag))
        return ComplexFormula(Formula(re.components + im.components, {}))
    if isinstance(value, (numbers.Real, symox.Basic)):
        return Formula._lift(value).real2complex()
    raise TypeError(f"unsupported complex operand: {value!r}")


def _broadcast(a: "ComplexFormula", b: "ComplexFormula"):
    pa, pb = _pairs(a._f), _pairs(b._f)
    if len(pa) != len(pb):
        if len(pa) == 1:
            pa = pa * len(pb)
        elif len(pb) == 1:
            pb = pb * len(pa)
        else:
            raise ValidationError(f"incompatible complex dimensions: {a.dim} and {b.dim}")
    return pa, pb, _merge_vars(a._f._vars, b._f._vars)


def _mult(p, q):
    (a, b), (c, d) = p, q
    return (a * c - b * d, a * d + b * c)


class ComplexFormula:
    """Complex formula: interleaved real :class:`Formula` of dimension 2·``dim``."""

    __slots__ = ("_f",)
    __array_priority__ = 1001
    is_complex = True

    def __init__(self, formula: Formula):
        if not isinstance(formula, Formula) or formula.is_boolean or formula.dim % 2:
            raise ValidationError("complex formula: expected a real Formula of even dimension")
        self._f = formula

    # ---- introspection
    @property
    def dim(self) -> int:
        return self._f.dim // 2

    @property
    def formula(self) -> Formula:
        """Interleaved real formula ``(re_0, im_0, …)``."""
        return self._f

    @property
    def variables(self) -> dict:
        return self._f.variables

    def __len__(self):
        return self.dim

    def __repr__(self):
        return f"ComplexFormula(dim={self.dim}, {list(self._f.components)})"

    def __bool__(self):
        raise TypeError("a ComplexFormula has no truth value")

    __hash__ = None

    # ---- parts
    @property
    def real(self) -> Formula:
        return self._f[0::2]

    @property
    def imag(self) -> Formula:
        return self._f[1::2]

    def conj(self) -> "ComplexFormula":
        return _from_pairs([(a, -b) for a, b in _pairs(self._f)], self._f._vars)

    def square_abs(self) -> Formula:
        """``re² + im²`` (keopscore ``ComplexSquareAbs``)."""
        return Formula([a * a + b * b for a, b in _pairs(self._f)], self._f._vars)

    def abs(self) -> Formula:  # noqa: A003
        """Modulus ``sqrt(re² + im²)`` (keopscore ``ComplexAbs``, without hypot)."""
        import symox

        return Formula([symox.sqrt(a * a + b * b) for a, b in _pairs(self._f)], self._f._vars)

    __abs__ = abs

    def angle(self) -> Formula:
        """Argument ``atan2(im, re)`` (keopscore ``ComplexAngle``)."""
        import symox

        return Formula([symox.atan2(b, a) for a, b in _pairs(self._f)], self._f._vars)

    def exp(self) -> "ComplexFormula":
        """``exp(re)·(cos im, sin im)`` (keopscore ``ComplexExp``)."""
        import symox

        out = []
        for a, b in _pairs(self._f):
            r = symox.exp(a)
            out.append((r * symox.cos(b), r * symox.sin(b)))
        return _from_pairs(out, self._f._vars)

    def sum(self, *args, **kwargs) -> "ComplexFormula":
        """Sum of the complex components (keopscore ``ComplexSum``), dimension 1."""
        if args or kwargs:
            raise TypeError("ComplexFormula.sum() sums the components; to reduce an axis: .sum_over('j')")
        import symox

        ps = _pairs(self._f)
        return _from_pairs([(symox.Add(*(a for a, _ in ps)), symox.Add(*(b for _, b in ps)))], self._f._vars)

    def __getitem__(self, index):
        d = self.dim
        if isinstance(index, bool) or not isinstance(index, numbers.Integral):
            raise TypeError("ComplexFormula: only an integer component index is supported")
        i = int(index)
        if not -d <= i < d:
            raise IndexError(f"component {i} out of [0, {d})")
        i %= d
        return ComplexFormula(self._f[2 * i:2 * i + 2])

    def factorize(self, g):
        """Identity (CSE always applied), see :func:`kernox.ops.factorize`."""
        from .ops import factorize

        factorize(self._f, g)
        return self

    def auto_factorize(self):
        return self

    # ---- explicit refusals (like PyKeOps)
    def real2complex(self):
        raise ValidationError("real2complex does not apply to a complex formula")

    def imag2complex(self):
        raise ValidationError("imag2complex does not apply to a complex formula")

    def exp1j(self):
        raise ValidationError("exp1j does not apply to a complex formula")

    # ---- arithmetic
    def _binary(self, other, op, reverse=False):
        a, b = (_lift(other), self) if reverse else (self, _lift(other))
        pa, pb, variables = _broadcast(a, b)
        return _from_pairs([op(p, q) for p, q in zip(pa, pb)], variables)

    def __add__(self, o):
        return self._binary(o, lambda p, q: (p[0] + q[0], p[1] + q[1]))

    def __radd__(self, o):
        return self._binary(o, lambda p, q: (p[0] + q[0], p[1] + q[1]), reverse=True)

    def __sub__(self, o):
        return self._binary(o, lambda p, q: (p[0] - q[0], p[1] - q[1]))

    def __rsub__(self, o):
        return self._binary(o, lambda p, q: (p[0] - q[0], p[1] - q[1]), reverse=True)

    def __mul__(self, o):
        if isinstance(o, Formula) and o.dim == 1:  # ComplexRealScal
            return self._real_scal(o)
        return self._binary(o, _mult)

    def __rmul__(self, o):
        if isinstance(o, Formula) and o.dim == 1:
            return self._real_scal(o)
        return self._binary(o, _mult, reverse=True)

    @staticmethod
    def _real_scal_or_mult(o, s: Formula):
        """``s · o`` for real ``s`` and complex ``o`` (formula or Python scalar)."""
        o = _lift(o)
        return o._real_scal(s) if s.dim == 1 else s.real2complex() * o

    @staticmethod
    def _lift(value):
        return _lift(value)

    def _real_scal(self, s: Formula):
        (c,) = s.components
        return _from_pairs([(c * a, c * b) for a, b in _pairs(self._f)], _merge_vars(s._vars, self._f._vars))

    def __truediv__(self, o):
        return _divide(self, _lift(o))

    def __rtruediv__(self, o):
        return _divide(_lift(o), self)

    def __neg__(self):
        return _from_pairs([(-a, -b) for a, b in _pairs(self._f)], self._f._vars)

    def __pos__(self):
        return self

    def __pow__(self, o):
        raise TypeError("power of a complex formula is not defined (like PyKeOps)")

    # ---- reduction
    def sum_over(self, axis: str) -> "ComplexReductionExpr":
        return ComplexReductionExpr(self, axis)

    def _no_reduction(self, *args, **kwargs):
        raise TypeError("reduction not defined on a complex formula: only sum_over (SUM) is; "
                        "reduce .real, .imag or .abs() for the others")

    min = max = argmin = argmax = kmin = argkmin = kmin_argkmin = min_argmin = max_argmax = _no_reduction
    logsumexp = sumsoftmaxweight = _no_reduction


def _divide(f: ComplexFormula, g: ComplexFormula) -> ComplexFormula:
    """keopscore: ``ComplexMult(Real2Complex(Inv(ComplexSquareAbs(g))), ComplexMult(f, Conj(g)))``."""
    pf, pg, variables = _broadcast(f, g)
    out = []
    for p, q in zip(pf, pg):
        c, d = q
        inv = 1 / (c * c + d * d)
        num = _mult(p, (c, -d))
        out.append((inv * num[0], inv * num[1]))
    return _from_pairs(out, variables)


class ComplexReductionExpr(ReductionExpr):
    """SUM of a complex formula: real plan of dimension 2D, output returned as complex."""

    __slots__ = ("complex_formula",)

    def __init__(self, cf: ComplexFormula, over: str):
        super().__init__(cf.formula, ReductionKind.SUM, over)
        self.complex_formula = cf

    def __repr__(self):
        return f"ComplexReductionExpr(sum, over={self.over!r}, {self.complex_formula!r})"

    def to_plan(self, dtype, **kwargs):
        plan = super().to_plan(dtype, **kwargs)
        plan.meta["complex_output"] = True
        return plan


# --------------------------------------------------------------------------
# constructors from real formulas (Formula methods, kp.* functions)
# --------------------------------------------------------------------------


def real2complex(x) -> ComplexFormula:
    """``(x, 0)`` component-wise (keopscore ``Real2Complex``)."""
    if isinstance(x, ComplexFormula):
        return x.real2complex()
    import symox

    f = Formula._lift(x)
    return ComplexFormula(Formula([v for c in f.components for v in (c, symox.Integer(0))], f._vars))


def imag2complex(x) -> ComplexFormula:
    """``(0, x)`` (keopscore ``Imag2Complex``)."""
    if isinstance(x, ComplexFormula):
        return x.imag2complex()
    import symox

    f = Formula._lift(x)
    return ComplexFormula(Formula([v for c in f.components for v in (symox.Integer(0), c)], f._vars))


def exp1j(x) -> ComplexFormula:
    """``(cos x, sin x)`` (keopscore ``ComplexExp1j``)."""
    if isinstance(x, ComplexFormula):
        return x.exp1j()
    import symox

    f = Formula._lift(x)
    return ComplexFormula(Formula([v for c in f.components for v in (symox.cos(c), symox.sin(c))], f._vars))


def complex_formula(re, im) -> ComplexFormula:
    """``re + i·im`` from two real formulas of the same dimension (or broadcast)."""
    return real2complex(re) + imag2complex(im)


def conj(z):
    return z.conj()


def angle(z):
    return z.angle()


def real(z):
    return z.real


def imag(z):
    return z.imag


for _name, _fn in (("real2complex", real2complex), ("imag2complex", imag2complex), ("exp1j", exp1j)):
    setattr(Formula, _name, _fn)


# --------------------------------------------------------------------------
# runtime: tensor conversion
# --------------------------------------------------------------------------


def complex_input(name: str, value, dim2: int):
    """Complex tensor ``(…, L, D)`` → interleaved real view ``(…, L, 2D)`` (differentiable)."""
    import torch

    if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
        value = [value]
    if isinstance(value, (list, tuple)) and any(isinstance(v, numbers.Complex) and not isinstance(v, numbers.Real)
                                                 for v in value):
        return [p for v in value for p in (complex(v).real, complex(v).imag)]
    if isinstance(value, torch.Tensor) and value.is_complex():
        if value.shape[-1] * 2 != dim2:
            raise ValidationError(f"{name!r}: last complex dimension is {value.shape[-1]} instead of {dim2 // 2}")
        if value.dim() > 1 and value.stride(-1) != 1:
            value = value.contiguous()
        return torch.view_as_real(value.resolve_conj()).reshape(*value.shape[:-1], dim2)
    return value


def complex_output(out):
    """Interleaved real output ``(…, M, 2D)`` → complex tensor ``(…, M, D)``."""
    import torch

    return torch.view_as_complex(out.reshape(*out.shape[:-1], out.shape[-1] // 2, 2).contiguous())


__all__ = ["ComplexFormula", "ComplexReductionExpr", "real2complex", "imag2complex", "exp1j",
           "complex_formula", "conj", "angle", "real", "imag"]
