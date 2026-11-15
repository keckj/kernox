"""kernox: fused symbolic reductions on GPUs.

Documentation: https://keckj.github.io/kernox/. Expressions are symox objects; ``compile`` produces a
callable that validates its inputs before any native launch.
"""

from .errors import (
    BackendUnavailableError,
    ConstantConversionError,
    KernoxError,
    UnsupportedError,
    ValidationError,
)
from .expr import (
    Formula,
    Param,
    ReductionExpr,
    Vi,
    Vj,
    abs,
    acos,
    asin,
    atan,
    atan2,
    cat,
    ceiling,
    clamp,
    constant,
    cos,
    cosh,
    eq,
    exp,
    floor,
    log,
    maximum,
    minimum,
    ne,
    rsqrt,
    sign,
    sin,
    sinh,
    sqrt,
    square,
    tan,
    tanh,
    where,
)
from .ops import (  # noqa: E402 - attache aussi les méthodes PyKeOps à Formula
    bspline,
    clampint,
    elem,
    extract,
    ifelse,
    inv,
    kron,
    matvecmult,
    mod,
    normalize,
    one_hot,
    relu,
    round,
    sinc,
    sinxdivx,
    softdtw_sqdist,
    grad,
    diff,
    grad_matrix,
    divergence,
    laplacian,
    trace_operator,
    sqdist,
    step,
    tensordot,
    tensorprod,
    vecmatmult,
    weightedsqdist,
    weightedsqnorm,
    xlogx,
)
from .complex import ComplexFormula, angle, complex_formula, conj, exp1j, imag, imag2complex, real, real2complex
from .linalg import LinearOperator, conjugate_gradient
from .ranges import Ranges, ranges_from_labels
from .runtime import RangedReduction, Reduction, compile, get_backend, register_backend
from .tuning import export_wisdom, forget_wisdom, import_wisdom
from .diagnostics import diagnostics  # noqa: E402 - aussi `python -m kernox.diagnostics`

__all__ = [
    "Vi", "Vj", "Param", "Formula", "ReductionExpr", "Reduction", "RangedReduction", "compile",
    "Ranges", "ranges_from_labels", "LinearOperator", "conjugate_gradient",
    "ComplexFormula", "real2complex", "imag2complex", "exp1j", "complex_formula", "conj", "angle", "real", "imag",
    "register_backend", "get_backend", "diagnostics", "export_wisdom", "import_wisdom", "forget_wisdom", "constant", "where", "eq", "ne", "cat",
    "exp", "log", "sqrt", "rsqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "abs", "sign", "floor", "ceiling", "square",
    "minimum", "maximum", "clamp",
    "relu", "step", "sinxdivx", "sinc", "xlogx", "inv", "round", "mod", "clampint", "ifelse",
    "normalize", "sqdist", "weightedsqnorm", "weightedsqdist", "one_hot", "elem", "extract",
    "matvecmult", "vecmatmult", "tensorprod", "tensordot", "kron", "bspline", "softdtw_sqdist",
    "grad", "diff", "grad_matrix", "divergence", "laplacian", "trace_operator",
    "KernoxError", "ValidationError", "ConstantConversionError",
    "BackendUnavailableError", "UnsupportedError",
]
