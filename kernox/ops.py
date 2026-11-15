"""Formula operations for PyKeOps parity (keopscore ``formulas/maths``).

Each operation reproduces the **exact semantics** of keopscore (reference
sources: ``keopscore/formulas/maths/*.py`` and
``keopscore/utils/math_functions.py``), expressed with symox primitives
(``Piecewise``, ``floor``, comparisons, arithmetic): no codegen extension,
and the same DAG is printed by the three backends, the Torch oracle and the
NumPy reference. Deliberate deviations (behaviors left undefined by
keopscore) are flagged in each docstring and in the
"Coming from PyKeOps" page of the documentation.

Memory layout of "matrix" vectors: row-major, like keopscore
(``C_CONTIGUOUS 1``).
"""

from __future__ import annotations

import numbers

from .errors import ValidationError
from .expr import Formula, _merge_vars
from .expr import clamp as _clamp


def _sx():
    import symox

    return symox


def _pw(*pairs):
    return _sx().Piecewise(*pairs)


def _int(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValidationError(f"{name}: expected an integer, got {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValidationError(f"{name}: expected an integer ≥ {minimum}, got {value}")
    return value


def _real(f, what):
    f = Formula._lift(f)
    if f.is_boolean:
        raise TypeError(f"{what}: boolean operand; use where(pred, a, b)")
    return f


def _formula(comps, *formulas):
    return Formula(tuple(comps), _merge_vars(*(f._vars for f in formulas)))


# ------------------------------------------------------------ scalars
def relu(x):
    """``(x < 0) ? 0 : x`` (keopscore ``ReLU``; NaN → NaN)."""
    sx = _sx()
    return Formula._map(lambda a: _pw((sx.Integer(0), sx.StrictLessThan(a, 0)), (a, True)), x, boolean_in=False)


def step(x):
    """``(x < 0) ? 0 : 1`` (keopscore ``Step``; NaN → 1, like keopscore)."""
    sx = _sx()
    return Formula._map(lambda a: _pw((sx.Integer(0), sx.StrictLessThan(a, 0)), (sx.Integer(1), True)), x,
                        boolean_in=False)


def sinxdivx(x):
    """``x ? sin(x)/x : 1`` (keopscore ``SinXDivX``)."""
    sx = _sx()
    return Formula._map(lambda a: _pw((sx.Integer(1), sx.Eq(a, 0)), (sx.sin(a) / a, True)), x, boolean_in=False)


def sinc(x):
    """``sin(πx)/(πx)``, i.e. ``sinxdivx(π·x)`` (PyKeOps ``LazyTensor.sinc``)."""
    return sinxdivx(_sx().pi * _real(x, "sinc"))


def xlogx(x):
    """``x ? x·log(x) : 0`` (keopscore ``XLogX``)."""
    sx = _sx()
    return Formula._map(lambda a: _pw((sx.Integer(0), sx.Eq(a, 0)), (a * sx.log(a), True)), x, boolean_in=False)


def inv(x):
    """``1/x`` (keopscore ``Inv``)."""
    return 1 / _real(x, "inv")


def _round0(a):
    """C ``round(a)`` (half away from zero), exact, including the sign of zero.

    ``b = |a|``, ``r = floor(b)``, ``f = b − r``: exact (Sterbenz for
    ``r ≥ 1``, trivial for ``r = 0``); ``q = r + 1`` if ``f ≥ 1/2``, otherwise
    ``r``; result ``−q`` if ``a < 0`` (hence ``−0`` for ``a ∈ ]−1/2, 0[``,
    like C, and ``±0 → ±0``). ``±inf``: ``f`` is NaN, ``q = r``; NaN propagates.
    """
    sx = _sx()
    b = sx.Abs(a)
    r = sx.floor(b)
    q = _pw((r + 1, sx.GreaterThan(b - r, sx.Rational(1, 2))), (r, True))
    return _pw((-q, sx.StrictLessThan(a, 0)), (a, sx.Eq(a, 0)), (q, True))  # ±0 → ±0


def round(x, d=0):  # noqa: A001 - PyKeOps name
    """Rounding to ``d`` decimal places (keopscore ``Round``): ``round(x·10^d)/10^d``.

    ``d = 0``: C ``round(x)`` (half away from zero). The factor ``10^d``
    is an exact constant; the final division is rounded once.
    """
    d = _int(d, "round: number of decimals")
    sx = _sx()
    if d == 0:
        return Formula._map(_round0, x, boolean_in=False)
    scale = sx.Integer(10) ** d
    return Formula._map(lambda a: _round0(a * scale) / scale, x, boolean_in=False)


def mod(x, modulus, offset=0):
    """``x − n·floor((x − d)/n)`` (keopscore ``Mod``, ``n = modulus``, ``d = offset``)."""
    sx = _sx()
    return Formula._map(lambda a, n, o: a - n * sx.floor((a - o) / n), x, modulus, offset, boolean_in=False)


def clamp(x, lo, hi):
    """Alias of :func:`kernox.clamp` (``min(max(x, lo), hi)``).

    keopscore writes ``(x < a) ? a : ((x > b) ? b : x)``; the two coincide
    except for NaN (keopscore propagates NaN, for ``min``/``max`` it depends on the backend).
    The deviation is listed in the "Coming from PyKeOps" documentation page.
    """
    return _clamp(x, lo, hi)


def clampint(x, lo: int, hi: int):
    """``clamp`` with fixed integer bounds (keopscore ``ClampInt``)."""
    return clamp(x, _int(lo, "clampint: a"), _int(hi, "clampint: b"))


def ifelse(x, a, b):
    """``(x >= 0) ? a : b`` (keopscore ``IfElse``)."""
    sx = _sx()
    return Formula._map(lambda c, u, v: _pw((u, sx.GreaterThan(c, 0)), (v, True)), x, a, b, boolean_in=False)


# ------------------------------------------------------------- vectors
def _rsqrt_keops(s):
    """``(s == 0) ? 0 : 1/sqrt(s)`` (keopscore ``Rsqrt``, exact here: no approximate ``rsqrt``)."""
    sx = _sx()
    return _pw((sx.Integer(0), sx.Eq(s, 0)), (1 / sx.sqrt(s), True))


def normalize(x):
    """``x · rsqrt(|x|²)`` with ``rsqrt(0) = 0`` (keopscore ``Normalize``: zero vector → 0)."""
    x = _real(x, "normalize")
    r = _rsqrt_keops(x.sqnorm2()._comps[0])
    return _formula((c * r for c in x._comps), x)


def sqdist(x, y):
    """``|x − y|²`` (keopscore ``SqDist``)."""
    return (_real(x, "sqdist") - y).sqnorm2()


def weightedsqnorm(x, s):
    """Norm of ``x`` weighted by ``s`` (PyKeOps ``x.weightedsqnorm(s)``).

    ``dim(s) = 1``: ``s·|x|²``; ``dim(s) = D``: ``Σ s_k x_k²``;
    ``dim(s) = D²``: ``Σ_{kl} s_{kl} x_k x_l`` (row-major matrix). For
    ``D = 1``, the isotropic case applies (like keopscore).
    """
    x, s = _real(x, "weightedsqnorm"), _real(s, "weightedsqnorm")
    d = x.dim
    if s.dim == 1:
        return s * x.sqnorm2()
    if s.dim == d:
        return _formula((_sx().Add(*(a * (b * b) for a, b in zip(s._comps, x._comps))),), s, x)
    if s.dim == d * d:
        return (s * tensorprod(x, x)).sum()
    raise ValidationError(f"weightedsqnorm: expected weights of dimension 1, {d} or {d * d}, got {s.dim}")


def weightedsqdist(x, y, s):
    """``weightedsqnorm(x − y, s)`` (PyKeOps ``x.weightedsqdist(y, s)``)."""
    return weightedsqnorm(_real(x, "weightedsqdist") - y, s)


def one_hot(x, D: int):
    """Vector of dimension ``D`` equal to 1 at ``(int)(x + 0.5)`` (keopscore ``OneHot``).

    keopscore computes ``x + .5`` in double then truncates toward zero: the
    component k ≥ 1 is 1 if ``k − 1/2 ≤ x < k + 1/2``, the component 0 if
    ``−3/2 < x < 1/2`` (exact comparisons, without rounding ``x + 0.5``).
    Outside ``[0, D)`` (undefined behavior in keopscore) and NaN: zero vector.
    """
    D = _int(D, "one_hot: D", minimum=1)
    x = _real(x, "one_hot")
    if x.dim != 1:
        raise ValidationError("one_hot: scalar formula only")
    sx = _sx()
    a = x._comps[0]
    half = sx.Rational(1, 2)
    comps = []
    for k in range(D):
        if k == 0:
            cond = sx.And(sx.StrictGreaterThan(a, -3 * half), sx.StrictLessThan(a, half))
        else:
            cond = sx.And(sx.GreaterThan(a, k - half), sx.StrictLessThan(a, k + half))
        comps.append(_pw((sx.Integer(1), cond), (sx.Integer(0), True)))
    return _formula(comps, x)


def elem(x, i: int):
    """Component ``i`` (keopscore ``Elem``), ``0 ≤ i < dim``."""
    i = _int(i, "elem: index", minimum=0)
    if i >= x.dim:
        raise ValidationError(f"elem: index {i} out of [0, {x.dim})")
    return x[i]


def extract(x, i: int, d: int):
    """Components ``i … i+d−1`` (keopscore ``Extract``)."""
    i, d = _int(i, "extract: start", minimum=0), _int(d, "extract: dimension", minimum=1)
    if i + d > x.dim:
        raise ValidationError(f"extract: [{i}, {i + d}) out of [0, {x.dim})")
    return x[i:i + d]


def matvecmult(A, v):
    """``(A v)_i = Σ_k A[i·m + k] v_k`` with ``A`` row-major (n, m), ``m = dim(v)``."""
    A, v = _real(A, "matvecmult"), _real(v, "matvecmult")
    m = v.dim
    if A.dim % m:
        raise ValidationError(f"matvecmult: dim(A)={A.dim} is not a multiple of dim(v)={m}")
    n = A.dim // m
    sx = _sx()
    comps = (sx.Add(*(A._comps[i * m + k] * v._comps[k] for k in range(m))) for i in range(n))
    return _formula(comps, A, v)


def vecmatmult(v, A):
    """``(v A)_i = Σ_k v_k A[k·n + i]`` with ``A`` row-major (m, n), ``m = dim(v)``."""
    v, A = _real(v, "vecmatmult"), _real(A, "vecmatmult")
    m = v.dim
    if A.dim % m:
        raise ValidationError(f"vecmatmult: dim(A)={A.dim} is not a multiple of dim(v)={m}")
    n = A.dim // m
    sx = _sx()
    comps = (sx.Add(*(v._comps[k] * A._comps[k * n + i] for k in range(m))) for i in range(n))
    return _formula(comps, v, A)


def tensorprod(a, b):
    """Tensor product ``out[k·dim(b) + l] = a_k b_l`` (keopscore ``TensorProd``)."""
    a, b = _real(a, "tensorprod"), _real(b, "tensorprod")
    return _formula((x * y for x in a._comps for y in b._comps), a, b)


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def tensordot(a, b, dimfa, dimfb, contfa, contfb, permute=None):
    """Contracted tensor product (keopscore ``TensorDot``, PyKeOps ``keops_tensordot``).

    ``a`` (resp. ``b``) viewed as a row-major tensor of shape ``dimfa``
    (``dimfb``); contraction of the axes ``contfa`` with ``contfb`` (like
    ``numpy.tensordot``), remaining axes of ``a`` then of ``b``. ``permute``:
    permutation in the keopscore sense, the output is the tensor transposed by
    ``argsort(permute)`` then flattened row-major.
    """
    import numpy as np

    a, b = _real(a, "tensordot"), _real(b, "tensordot")
    dimfa, dimfb = [_int(d, "tensordot: dimfa", minimum=1) for d in dimfa], [_int(d, "tensordot: dimfb", minimum=1) for d in dimfb]
    contfa, contfb = [_int(c, "tensordot: contfa", minimum=0) for c in contfa], [_int(c, "tensordot: contfb", minimum=0) for c in contfb]
    if _prod(dimfa) != a.dim or _prod(dimfb) != b.dim:
        raise ValidationError(f"tensordot: prod(dimfa)={_prod(dimfa)}/dim(a)={a.dim}, prod(dimfb)={_prod(dimfb)}/dim(b)={b.dim}")
    if len(contfa) != len(contfb) or any(c >= len(dimfa) for c in contfa) or any(c >= len(dimfb) for c in contfb):
        raise ValidationError("tensordot: invalid contraction axes")
    if [dimfa[c] for c in contfa] != [dimfb[c] for c in contfb]:
        raise ValidationError("tensordot: contracted dimensions differ")
    A = np.empty(len(a._comps), dtype=object)
    A[:] = list(a._comps)
    B = np.empty(len(b._comps), dtype=object)
    B[:] = list(b._comps)
    T = np.tensordot(A.reshape(dimfa), B.reshape(dimfb), axes=(contfa, contfb))
    T = np.asarray(T, dtype=object)
    nk = T.ndim
    if permute is not None:
        permute = [_int(p, "tensordot: permute", minimum=0) for p in permute]
        if sorted(permute) != list(range(nk)):
            raise ValidationError(f"tensordot: permute must be a permutation of range({nk})")
        T = T.transpose(np.argsort(permute))
    comps = [_sx().sympify(c) for c in T.reshape(-1).tolist()]
    return _formula(comps, a, b)


def kron(a, b, dimfa, dimfb):
    """Kronecker product over the inner dimensions (keopscore ``Kron``, ``numpy.kron``)."""
    if len(dimfa) != len(dimfb):
        raise ValidationError("kron: dimfa and dimfb must have the same length")
    n = len(dimfa)
    permute = list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2))
    return tensordot(a, b, dimfa, dimfb, [], [], permute=permute)


def bspline(knots, x, k: int = 0):
    """B-spline functions of order ``k`` on the knots ``knots``, evaluated at scalar ``x``.

    keopscore ``BSpline`` (in-place De Boor algorithm, same order of
    operations): output of dimension ``dim(knots) − k − 1``. Order 0:
    indicator ``t_j ≤ x < t_{j+1}``; ratios ``ω = (x − t_i)/(t_{i+k} − t_i)``
    if ``t_i < t_{i+k}``, 0 otherwise. Differentiable here with respect to ``x`` and to the
    knots (keopscore is not).
    """
    k = _int(k, "bspline: order", minimum=0)
    knots, x = _real(knots, "bspline"), _real(x, "bspline")
    if x.dim != 1:
        raise ValidationError("bspline: expected a scalar x")
    if knots.dim < k + 2:
        raise ValidationError(f"bspline of order {k}: expected at least {k + 2} knots, got {knots.dim}")
    sx = _sx()
    t, xv = knots._comps, x._comps[0]
    zero, one = sx.Integer(0), sx.Integer(1)
    B = [_pw((one, sx.And(sx.LessThan(t[j], xv), sx.StrictLessThan(xv, t[j + 1]))), (zero, True))
         for j in range(len(t) - 1)]

    def ratio(i, kk):
        return _pw(((xv - t[i]) / (t[i + kk] - t[i]), sx.StrictLessThan(t[i], t[i + kk])), (zero, True))

    for kk in range(1, k + 1):
        r1 = ratio(0, kk)
        for i in range(len(t) - 1 - kk):
            r2 = ratio(i + 1, kk)
            B[i] = r1 * B[i] + (one - r2) * B[i + 1]
            r1 = r2
    return _formula(B[:len(t) - k - 1], knots, x)


def softdtw_sqdist(x, y, gamma):
    """Soft-DTW between the series ``x`` (dim n) and ``y`` (dim m), cost ``(x_i − y_j)²``.

    keopscore ``SoftDTW_SqDist``: dynamic programming ``r(i, j)`` with
    ``r(0,0) = d²``, cumulative borders, then
    ``r = d² + (μ − γ·log(e^{(μ−r(i−1,j−1))/γ} + e^{(μ−r(i−1,j))/γ} + e^{(μ−r(i,j−1))/γ}))``,
    ``μ = min(r(i−1,j−1), r(i,j−1), r(i−1,j))``; output ``r(n−1, m−1)``.
    The recurrence is unrolled symbolically (DAG of size O(n·m), shared
    through CSE): the gradient (with respect to x, y **and** γ) is the symbolic
    derivative of this DAG. keopscore codes a loop and a dedicated gradient, and
    refuses the derivative with respect to γ.
    """
    x, y, gamma = _real(x, "softdtw_sqdist"), _real(y, "softdtw_sqdist"), _real(gamma, "softdtw_sqdist")
    if gamma.dim != 1:
        raise ValidationError("softdtw_sqdist: expected a scalar gamma")
    sx = _sx()
    X, Y, g = x._comps, y._comps, gamma._comps[0]
    n, m = len(X), len(Y)

    def d2(i, j):
        t = X[i] - Y[j]
        return t * t

    prev = [None] * n  # column j − 1
    acc = None
    for i in range(n):
        acc = d2(i, 0) if i == 0 else d2(i, 0) + acc
        prev[i] = acc
    for j in range(1, m):
        cur = [d2(0, j) + prev[0]]
        for i in range(1, n):
            diag, up, left = prev[i - 1], cur[i - 1], prev[i]
            mu = sx.Min(diag, left, up)  # keopscore MIN3; n-ary Min: native kernel
            soft = mu - g * sx.log(sx.exp((mu - diag) / g) + sx.exp((mu - up) / g) + sx.exp((mu - left) / g))
            cur.append(d2(i, j) + soft)
        prev = cur
    return _formula((prev[n - 1],), x, y, gamma)


# ---------------------------------------------------------- derivatives
def _wrt(v, what):
    """Component symbols of a variable (``Vi``, ``Vj`` or ``Param``)."""
    from .builder import component_symbol

    if isinstance(v, Formula) and len(v._vars) == 1:
        (name, (_role, dim)), = v._vars.items()
        syms = tuple(component_symbol(name, c) for c in range(dim))
        if v._comps == syms:
            return syms
    raise ValidationError(f"{what}: expected a variable (Vi, Vj or Param), got {v!r}")


def _d(expr, symbol):
    from .autograd import derivative

    return derivative(expr, symbol)


def grad(f, v, e):
    """``(∂f/∂v)ᵀ e`` (keopscore ``Grad(f, v, e)``): dimension of ``v``, ``e`` of dimension of ``f``.

    Symbolic derivative (rules of :mod:`kernox.autograd`: ``Heaviside`` of
    ``min``/``max`` → 1/2 at ties, ``where`` differentiated branch by branch).
    """
    f, e = _real(f, "grad"), _real(e, "grad")
    syms = _wrt(v, "grad")
    if e.dim != f.dim:
        raise ValidationError(f"grad: expected e of dimension {f.dim} (that of f), got {e.dim}")
    sx = _sx()
    comps = [sx.Add(*(ek * _d(fk, s) for fk, ek in zip(f._comps, e._comps))) for s in syms]
    return _formula(comps, f, v, e)


def diff(f, v, u):
    """``(∂f/∂v) u`` (keopscore ``Diff(f, v, u)``, differential applied to ``u``): dimension of ``f``."""
    f, u = _real(f, "diff"), _real(u, "diff")
    syms = _wrt(v, "diff")
    if u.dim != len(syms):
        raise ValidationError(f"diff: expected u of dimension {len(syms)} (that of v), got {u.dim}")
    sx = _sx()
    comps = [sx.Add(*(_d(fk, s) * uc for s, uc in zip(syms, u._comps))) for fk in f._comps]
    return _formula(comps, f, v, u)


def grad_matrix(f, v):
    """Matrix ``∂f_k/∂v_c`` flattened as ``k·dim(v) + c`` (keopscore ``GradMatrix``)."""
    f = _real(f, "grad_matrix")
    syms = _wrt(v, "grad_matrix")
    return _formula([_d(fk, s) for fk in f._comps for s in syms], f, v)


def divergence(f, v):
    """``Σ_c ∂f_c/∂v_c`` (keopscore ``Divergence``); ``dim(f) = dim(v)``."""
    f = _real(f, "divergence")
    syms = _wrt(v, "divergence")
    if f.dim != len(syms):
        raise ValidationError(f"divergence: expected f of dimension {len(syms)} (that of v), got {f.dim}")
    return _formula([_sx().Add(*(_d(fc, s) for fc, s in zip(f._comps, syms)))], f, v)


def laplacian(f, v):
    """``Σ_c ∂²f/∂v_c²`` (keopscore ``Laplacian``); scalar ``f``."""
    f = _real(f, "laplacian")
    syms = _wrt(v, "laplacian")
    if f.dim != 1:
        raise ValidationError(f"laplacian: expected a scalar f, got dimension {f.dim}")
    return _formula([_sx().Add(*(_d(_d(f._comps[0], s), s) for s in syms))], f, v)


def trace_operator(f, v):
    """Trace of the linear map ``v ↦ f(v)`` (keopscore ``TraceOperator``): ``Σ_c ∂f_c/∂v_c``.

    keopscore requires ``f`` to be linear in ``v``; the trace of the differential is
    then the trace of the map, and that is what is computed here for any ``f``.
    """
    f = _real(f, "trace_operator")
    syms = _wrt(v, "trace_operator")
    if f.dim != len(syms):
        raise ValidationError(f"trace_operator: expected f of dimension {len(syms)}, got {f.dim}")
    return divergence(f, v)


# ------------------------------------------------- Formula methods
def factorize(f, g):
    """``f.factorize(g)`` (keopscore ``Factorize``): evaluation hint, **with no semantic effect**.

    keopscore evaluates ``g`` once into a temporary then ``f``; here code
    generation always applies symox's CSE (per region in the
    presence of ``where``, see ``codegen.pair._region_cse``), which already
    factors out any repeated subexpression: the formula is returned unchanged.
    """
    if not isinstance(g, Formula) and type(g).__name__ != "ComplexFormula":
        raise TypeError(f"factorize expects a formula, got {type(g).__name__}")
    return f


def auto_factorize(f):
    """``f.auto_factorize()`` (keopscore ``AutoFactorize``): identity, CSE always applied."""
    return f


_METHODS = {
    "relu": relu, "step": step, "sinxdivx": sinxdivx, "sinc": sinc, "xlogx": xlogx, "inv": inv,
    "round": round, "mod": mod, "clamp": clamp, "clampint": clampint, "ifelse": ifelse,
    "normalize": normalize, "sqdist": sqdist, "weightedsqnorm": weightedsqnorm,
    "weightedsqdist": weightedsqdist, "one_hot": one_hot, "elem": elem, "extract": extract,
    "matvecmult": matvecmult, "vecmatmult": vecmatmult, "tensorprod": tensorprod,
    "keops_tensordot": tensordot, "tensordot": tensordot, "keops_kron": kron, "kron": kron,
    "bspline": bspline, "softdtw_sqdist": softdtw_sqdist,
    "grad": grad, "diff": diff, "grad_matrix": grad_matrix, "divergence": divergence,
    "laplacian": laplacian, "trace_operator": trace_operator,
    "factorize": factorize, "auto_factorize": auto_factorize,
}
for _name, _fn in _METHODS.items():
    setattr(Formula, _name, _fn)

__all__ = sorted(set(_METHODS) - {"keops_tensordot", "keops_kron"})
