"""CUDA backend (NVRTC + CUDA Driver API). See ``backend.py``."""

from .backend import CudaBackend, CudaCompiledReduction

__all__ = ["CudaBackend", "CudaCompiledReduction", "get_backend"]


def get_backend():
    """Factory expected by ``kernox.runtime.get_backend('cuda')``."""
    return CudaBackend()
