import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("symox")

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("aucun device XPU : tests du backend sycl sautés", allow_module_level=True)
