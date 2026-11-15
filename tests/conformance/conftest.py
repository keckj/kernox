"""Suite de conformance commune aux backends GPU (cuda, hip, sycl).

Chaque test est paramétré par backend ; un backend sans device ou sans module
est **sauté** (jamais simulé). ``KP_CONFORMANCE_BACKENDS=hip,cuda`` restreint
la liste. Profil rapide (CI) : ``-m "not slow"`` ; profil long :
``-m slow``.
"""

import os

import pytest

from ._support import BACKENDS, backend_status


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache(tmp_path_factory):
    # caches de compilation et wisdom isolés : rien n'est écrit chez l'utilisateur
    if "KERNOX_CACHE_DIR" in os.environ:
        yield None
        return
    d = tmp_path_factory.mktemp("kp-conformance-cache")
    os.environ["KERNOX_CACHE_DIR"] = str(d)
    yield d
    os.environ.pop("KERNOX_CACHE_DIR", None)


@pytest.fixture(params=BACKENDS)
def backend(request):
    """``(nom, device)`` d'un backend réellement disponible."""
    dev, reason = backend_status(request.param)
    if dev is None:
        pytest.skip(f"backend {request.param} absent : {reason}")
    return request.param, dev


def pytest_collection_modifyitems(items):
    for item in items:
        if "conformance" in str(item.fspath):
            item.add_marker(pytest.mark.gpu)
