"""Conformance suite shared by the GPU backends (cuda, hip, sycl).

Each test is parametrized by backend; a backend without a device or without
its module is **skipped** (never simulated). ``KP_CONFORMANCE_BACKENDS=hip,cuda``
restricts the list. Quick profile (CI): ``-m "not slow"``; long profile:
``-m slow``.
"""

import os

import pytest

from ._support import BACKENDS, backend_status


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache(tmp_path_factory):
    # isolated compilation and wisdom caches: nothing is written to the user's directories
    if "KERNOX_CACHE_DIR" in os.environ:
        yield None
        return
    d = tmp_path_factory.mktemp("kp-conformance-cache")
    os.environ["KERNOX_CACHE_DIR"] = str(d)
    yield d
    os.environ.pop("KERNOX_CACHE_DIR", None)


@pytest.fixture(params=BACKENDS)
def backend(request):
    """``(name, device)`` of an actually available backend."""
    dev, reason = backend_status(request.param)
    if dev is None:
        pytest.skip(f"backend {request.param} unavailable: {reason}")
    return request.param, dev


def pytest_collection_modifyitems(items):
    for item in items:
        if "conformance" in str(item.fspath):
            item.add_marker(pytest.mark.gpu)
