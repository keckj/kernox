import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("symox")

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("no XPU device: sycl backend tests skipped", allow_module_level=True)
