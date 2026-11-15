"""Public kernox exceptions.

All validation errors are raised **before** any native launch.
"""

from __future__ import annotations

from .backends.base import UnsupportedError


class KernoxError(Exception):
    """Base class for kernox-specific errors (except ``UnsupportedError``)."""


class ValidationError(KernoxError, ValueError):
    """Invalid input (shape, dtype, device, name, K...) detected before launch."""


class ConstantConversionError(KernoxError, ValueError):
    """Constant not representable in the requested dtype (overflow, underflow to 0)."""


class BackendUnavailableError(KernoxError, RuntimeError):
    """Requested backend missing or not loadable. Never a silent fallback."""


__all__ = [
    "KernoxError",
    "ValidationError",
    "ConstantConversionError",
    "BackendUnavailableError",
    "UnsupportedError",
]
