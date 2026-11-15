"""Variables d'environnement KERNOX_*."""

import pytest

from kernox._env import getenv

_NAMES = ("KERNOX_CACHE_DIR", "KERNOX_CUDA_CACHE_DIR", "KERNOX_SYCL_TARGET", "KERNOX_ONEAPI_ROOT")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)


def test_absent_et_vide(monkeypatch):
    assert getenv("KERNOX_SYCL_TARGET") is None
    assert getenv("KERNOX_ONEAPI_ROOT", "/opt") == "/opt"
    monkeypatch.setenv("KERNOX_SYCL_TARGET", "")
    assert getenv("KERNOX_SYCL_TARGET", "x") == "x"
    monkeypatch.setenv("KERNOX_SYCL_TARGET", "spir64")
    assert getenv("KERNOX_SYCL_TARGET", "x") == "spir64"


def test_cache_cuda_priorites(monkeypatch, tmp_path):
    from kernox.backends.cuda.backend import _cache_dir

    monkeypatch.setenv("KERNOX_CACHE_DIR", str(tmp_path / "commun"))
    assert _cache_dir() == tmp_path / "commun" / "cuda"
    monkeypatch.setenv("KERNOX_CUDA_CACHE_DIR", str(tmp_path / "cuda"))
    assert _cache_dir() == tmp_path / "cuda" / "cuda"
