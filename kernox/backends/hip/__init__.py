"""AMD ROCm/HIP backend (hipRTC + Torch's HIP runtime). See docs/backend-hip.md."""

from .backend import HipBackend, HipCompiledReduction, HipOptions


def get_backend() -> HipBackend:
    """Entry point for the ``kernox.runtime`` registry (default options)."""
    return HipBackend()


__all__ = ["HipBackend", "HipCompiledReduction", "HipOptions", "get_backend"]
