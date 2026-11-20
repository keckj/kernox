import shutil

import pytest

pytest.importorskip("torch")
pytest.importorskip("symox")

from kernox._env import getenv  # noqa: E402

if shutil.which(getenv("KERNOX_CPU_CXX", "g++")) is None:
    pytest.skip("no C++ compiler: cpu backend tests skipped", allow_module_level=True)
