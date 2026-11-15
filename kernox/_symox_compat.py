"""Detection of the symox capabilities used by Kernox.

Kernox is pinned to symox ``53d9576e`` (GitHub main), which contains the
codegen extensions below, and remains compatible with ``2e623e66``
(minimum revision), which lacks them:

- ``cudacode``: logical operators ``And/Or/Xor/Not``;
- ``cudacode``: ``reciprocal_sqrt`` option (``1/sqrt(x)``, opt-in);
- ``symox.extra.cudacode_function``: generic CUDA/HIP device function
  (pointer arguments, named output struct, no math macro);
- native NumPy ``lambdify`` with NumPy broadcasting.

Detection tests the **behavior** (not a version number: both revisions
carry version 0.1.1). Each probe is cached; the environment variable
``KERNOX_SYMOX_COMPAT=pinned`` forces the pinned-commit code path
(regression tests).
"""

from __future__ import annotations

import os
from functools import lru_cache

from ._env import getenv

PINNED_SYMOX = "53d9576e9f39db16e9ca8b3483d754385ebaf986"  # https://github.com/keckj/symox (main, printer SYCL sycl::half)
# Minimum revision still supported (compatibility path, without the
# codegen extensions); KERNOX_SYMOX_COMPAT=pinned forces this path.
MIN_SUPPORTED_SYMOX = "2e623e66bb31882e31f6dbfff6f77ae4bd945408"


def _forced_pinned() -> bool:
    return getenv("KERNOX_SYMOX_COMPAT", "") == "pinned"


@lru_cache(maxsize=None)
def _cuda_logic() -> bool:
    import symox
    from symox.extra import cudacode

    x, y = symox.symbols("x y", real=True)
    try:
        return "&&" in cudacode(symox.And(x < 1, y < 1))
    except NotImplementedError:
        return False


@lru_cache(maxsize=None)
def _cuda_reciprocal_sqrt() -> bool:
    import symox
    from symox.extra import cudacode

    x = symox.Symbol("x", real=True)
    try:
        return cudacode(1 / symox.sqrt(x) + 1, use_rsqrt=False, reciprocal_sqrt=True) == "1 + 1.0/sqrt(x)"
    except TypeError:
        return False


@lru_cache(maxsize=None)
def _cudacode_function() -> bool:
    import symox.extra

    return "cudacode_function" in getattr(symox.extra, "__all__", ())


@lru_cache(maxsize=None)
def _lambdify_broadcasts() -> bool:
    import numpy as np
    import symox

    x, y = symox.symbols("x y", real=True)
    f = symox.lambdify((x, y), symox.Piecewise((x, x < y), (y, True)), modules="numpy")
    try:
        return np.shape(f(np.zeros((2, 1)), np.ones((1, 3)))) == (2, 3)
    except ValueError:
        return False


@lru_cache(maxsize=None)
def _sycl_half() -> bool:
    import symox
    from symox import _core

    try:
        return "sycl::half" in _core._syclcode(symox.Symbol("x") + 1, dtype="float16")
    except (ValueError, TypeError):
        return False


def sycl_half() -> bool:
    """``_syclcode(..., dtype="float16")`` prints ``sycl::half`` (sycl backend)."""
    return not _forced_pinned() and _sycl_half()


def cuda_logic() -> bool:
    """``cudacode`` prints ``And/Or/Xor/Not``."""
    return not _forced_pinned() and _cuda_logic()


def cuda_reciprocal_sqrt() -> bool:
    """``cudacode(..., reciprocal_sqrt=True)`` is available."""
    return not _forced_pinned() and _cuda_reciprocal_sqrt()


def cudacode_function() -> bool:
    """``symox.extra.cudacode_function`` is available (implies ``cuda_logic``)."""
    return not _forced_pinned() and _cudacode_function() and _cuda_logic()


def lambdify_broadcasts() -> bool:
    """The native NumPy ``lambdify`` broadcasts shapes like NumPy."""
    return not _forced_pinned() and _lambdify_broadcasts()


def capabilities() -> dict:
    """Summary (diagnostics, benchmark reports)."""
    import symox

    return {
        "symox_version": getattr(symox, "__version__", "?"),
        "forced_pinned": _forced_pinned(),
        "cuda_logic": cuda_logic(),
        "cuda_reciprocal_sqrt": cuda_reciprocal_sqrt(),
        "cudacode_function": cudacode_function(),
        "lambdify_broadcasts": lambdify_broadcasts(),
        "sycl_half": sycl_half(),
    }
