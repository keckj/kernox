"""Coexistence of hip and torch.compile (triton) in the same process.

Regression: reopening Torch's HIP libraries with RTLD_GLOBAL promoted
libLLVM.so (_rocm_sdk_core, via libamd_comgr) into the global scope; importing
``torch._dynamo`` (hence ``libtriton.so``, with its embedded LLVM) then crashed with
SIGSEGV in ``llvm::cl::Option::addArgument``. Each scenario runs in a
subprocess: a native crash fails the test, not the whole suite.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.gpu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_KP = """
import kernox as kp
from kernox.backends.hip import _hip
x, y = kp.Vi("x", 3), kp.Vj("y", 3)
r = kp.compile(((x - y) ** 2).sum().sum_over("j"), dtype="float32", backend="hip")
xa, ya = torch.randn(10, 3, device="cuda"), torch.randn(20, 3, device="cuda")
out = r(x=xa, y=ya)
ref = ((xa[:, None] - ya[None]) ** 2).sum(-1).sum(1, keepdim=True)
assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
assert _hip.hiprtc() is not None
"""

_COMPILE = """
import torch._dynamo
f = torch.compile(lambda a: torch.sin(a) * 2 + a.cos())
a = torch.randn(1000, device="cuda")
assert torch.allclose(f(a), torch.sin(a) * 2 + a.cos(), rtol=1e-5, atol=1e-5)
"""


def _run(code):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")])))
    p = subprocess.run([sys.executable, "-c", "import torch\n" + textwrap.dedent(code) + "\nprint('ok')"],
                       capture_output=True, text=True, timeout=900, env=env)
    assert p.returncode == 0 and p.stdout.strip().endswith("ok"), f"code {p.returncode}; {p.stderr[-800:]}"


@pytest.mark.parametrize("order", ["hip_then_compile", "compile_then_hip", "interleaved"])
def test_torch_compile_after_or_before_hip(order):
    pytest.importorskip("triton")
    code = {"hip_then_compile": _KP + _COMPILE,
            "compile_then_hip": _COMPILE + _KP,
            "interleaved": _KP + _COMPILE + _KP.replace("20, 3", "33, 3")}[order]
    _run(code)


def test_hip_libraries_opened_locally():
    """The ctypes handles of the runtime/hipRTC must never be opened with RTLD_GLOBAL."""
    from kernox.backends.hip import _hip

    assert _hip._DLOPEN_MODE & os.RTLD_GLOBAL == 0
