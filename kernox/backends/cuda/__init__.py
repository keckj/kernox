"""Backend CUDA (NVRTC + CUDA Driver API). Voir ``backend.py``."""

from .backend import CudaBackend, CudaCompiledReduction

__all__ = ["CudaBackend", "CudaCompiledReduction", "get_backend"]


def get_backend():
    """Fabrique attendue par ``kernox.runtime.get_backend('cuda')``."""
    return CudaBackend()
