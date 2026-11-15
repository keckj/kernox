import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

torch = pytest.importorskip("torch")

if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("Torch ROCm + GPU AMD requis pour tests/hip", allow_module_level=True)


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache(tmp_path_factory):
    d = tmp_path_factory.mktemp("kpcache")
    old = os.environ.get("KERNOX_CACHE_DIR")
    os.environ["KERNOX_CACHE_DIR"] = str(d)
    yield d
    if old is None:
        os.environ.pop("KERNOX_CACHE_DIR", None)
    else:
        os.environ["KERNOX_CACHE_DIR"] = old


@pytest.fixture(scope="session")
def backend():
    from _cases import pair_provider

    from kernox.backends.hip import HipBackend

    return HipBackend(pair_provider=pair_provider)
