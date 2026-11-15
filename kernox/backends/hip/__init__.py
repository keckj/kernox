"""Backend AMD ROCm/HIP (hipRTC + runtime HIP de Torch). Voir docs/backend-hip.md."""

from .backend import HipBackend, HipCompiledReduction, HipOptions


def get_backend() -> HipBackend:
    """Point d'entrée du registre ``kernox.runtime`` (options par défaut)."""
    return HipBackend()


__all__ = ["HipBackend", "HipCompiledReduction", "HipOptions", "get_backend"]
