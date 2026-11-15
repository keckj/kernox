"""Minimal ctypes bindings to the HIP runtime and hipRTC **of Torch ROCm**.

Core rule (see ``docs/backend-hip.md``): load exactly the ``libamdhip64`` and
``libhiprtc`` libraries that Torch has already loaded into the process (the
``_rocm_sdk_core`` package of the official ROCm wheels, or ``torch/lib`` for
older wheels). Loading a second HIP runtime (e.g. from ``/opt/rocm/lib``) would
create a second runtime state, in which Torch's contexts, streams and
allocations would not be valid. ``hip_runtime()`` therefore checks through
``/proc/self/maps`` that the opened library is indeed Torch's.
"""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass

_lock = threading.Lock()
_rt = None
_rtc = None

# hipFunction_attribute (hip/driver_types.h, enumeration starting at 0)
FUNC_ATTR_MAX_THREADS_PER_BLOCK = 0
FUNC_ATTR_SHARED_SIZE_BYTES = 1
FUNC_ATTR_LOCAL_SIZE_BYTES = 3
FUNC_ATTR_NUM_REGS = 4


# Always RTLD_LOCAL. Reopening with RTLD_GLOBAL a library that Torch already loaded
# locally promotes it, along with its dependencies (libamd_comgr → libLLVM.so from
# _rocm_sdk_core), to global scope. The static constructors of ``triton/_C/libtriton.so``
# (bundled LLVM, visible symbols) then bind to libLLVM.so's
# ``llvm::cl::Option::addArgument``: SIGSEGV when importing torch._dynamo / triton
# (torch.compile after Kernox). ctypes resolves our symbols through the handle, so global
# scope is never needed.
_DLOPEN_MODE = os.RTLD_LOCAL | os.RTLD_NOW


class HipError(RuntimeError):
    pass


def _loaded_path(fragment: str) -> str | None:
    """Path of a library already mapped into the process."""
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and fragment in os.path.basename(parts[-1]):
                    return parts[-1]
    except OSError:
        return None
    return None


def _torch_hip_path(fragment: str) -> str:
    import torch

    if torch.version.hip is None:
        raise HipError("Torch is not a ROCm build (torch.version.hip is None)")
    # Make sure Torch has initialized and mapped its HIP runtime.
    torch.cuda.init()
    path = _loaded_path(fragment)
    if path is None and fragment == "libhiprtc":
        # hipRTC may be loaded lazily: look for it next to the runtime.
        rt = _loaded_path("libamdhip64")
        if rt is not None:
            d = os.path.dirname(rt)
            for name in sorted(os.listdir(d)):
                if name.startswith("libhiprtc.so") and "builtins" not in name:
                    path = os.path.join(d, name)
                    break
    if path is None:
        raise HipError(f"Torch library {fragment} not found in the process")
    return path


class _Runtime:
    def __init__(self, path: str):
        self.path = path
        lib = ctypes.CDLL(path, mode=_DLOPEN_MODE)
        self.lib = lib
        vp, i, u, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_size_t
        P = ctypes.POINTER
        sig = {
            "hipGetErrorString": (ctypes.c_char_p, [i]),
            "hipGetLastError": (i, []),
            "hipGetDevice": (i, [P(i)]),
            "hipSetDevice": (i, [i]),
            "hipRuntimeGetVersion": (i, [P(i)]),
            "hipDriverGetVersion": (i, [P(i)]),
            "hipDeviceGetAttribute": (i, [P(i), i, i]),
            "hipModuleLoadData": (i, [P(vp), vp]),
            "hipModuleUnload": (i, [vp]),
            "hipModuleGetFunction": (i, [P(vp), vp, ctypes.c_char_p]),
            "hipFuncGetAttribute": (i, [P(i), i, vp]),
            "hipModuleLaunchKernel": (i, [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)]),
        }
        for name, (res, args) in sig.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        del sz

    def check(self, err: int, what: str) -> None:
        if err != 0:
            # Consume the runtime's "last error": otherwise Torch picks it up at its
            # next launch check and attributes it to its own kernel.
            self.lib.hipGetLastError()
            msg = self.lib.hipGetErrorString(err)
            raise HipError(f"{what}: hipError {err} ({msg.decode() if msg else '?'})")

    def runtime_version(self) -> int:
        v = ctypes.c_int()
        self.check(self.lib.hipRuntimeGetVersion(ctypes.byref(v)), "hipRuntimeGetVersion")
        return v.value

    def driver_version(self) -> int:
        v = ctypes.c_int()
        self.check(self.lib.hipDriverGetVersion(ctypes.byref(v)), "hipDriverGetVersion")
        return v.value

    def get_device(self) -> int:
        v = ctypes.c_int()
        self.check(self.lib.hipGetDevice(ctypes.byref(v)), "hipGetDevice")
        return v.value

    def device_attribute(self, attr: int, dev: int) -> int:
        v = ctypes.c_int()
        self.check(self.lib.hipDeviceGetAttribute(ctypes.byref(v), attr, dev), "hipDeviceGetAttribute")
        return v.value

    def module_load(self, image: bytes) -> ctypes.c_void_p:
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(image, len(image))
        self.check(self.lib.hipModuleLoadData(ctypes.byref(mod), buf), "hipModuleLoadData")
        return mod

    def module_unload(self, mod) -> None:
        self.check(self.lib.hipModuleUnload(mod), "hipModuleUnload")

    def get_function(self, mod, name: str) -> ctypes.c_void_p:
        fn = ctypes.c_void_p()
        self.check(self.lib.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode()), "hipModuleGetFunction")
        return fn

    def func_attribute(self, fn, attr: int) -> int:
        v = ctypes.c_int()
        self.check(self.lib.hipFuncGetAttribute(ctypes.byref(v), attr, fn), "hipFuncGetAttribute")
        return v.value

    def launch(self, fn, grid: tuple, block: tuple, shmem: int, stream: int, args: list) -> None:
        """``args``: ctypes objects (values); they must stay alive until the call returns."""
        n = len(args)
        params = (ctypes.c_void_p * n)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        err = self.lib.hipModuleLaunchKernel(
            fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
            shmem, ctypes.c_void_p(stream), params, None,
        )
        self.check(err, "hipModuleLaunchKernel")


@dataclass(frozen=True)
class RtcResult:
    code: bytes
    log: str


class _Rtc:
    def __init__(self, path: str):
        self.path = path
        lib = ctypes.CDLL(path, mode=_DLOPEN_MODE)
        self.lib = lib
        vp, i, P, sz = ctypes.c_void_p, ctypes.c_int, ctypes.POINTER, ctypes.c_size_t
        cpp = P(ctypes.c_char_p)
        sig = {
            "hiprtcGetErrorString": (ctypes.c_char_p, [i]),
            "hiprtcVersion": (i, [P(i), P(i)]),
            "hiprtcCreateProgram": (i, [P(vp), ctypes.c_char_p, ctypes.c_char_p, i, cpp, cpp]),
            "hiprtcCompileProgram": (i, [vp, i, cpp]),
            "hiprtcGetProgramLogSize": (i, [vp, P(sz)]),
            "hiprtcGetProgramLog": (i, [vp, ctypes.c_char_p]),
            "hiprtcGetCodeSize": (i, [vp, P(sz)]),
            "hiprtcGetCode": (i, [vp, ctypes.c_char_p]),
            "hiprtcDestroyProgram": (i, [P(vp)]),
        }
        for name, (res, args) in sig.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args

    def check(self, err: int, what: str, log: str = "") -> None:
        if err != 0:
            msg = self.lib.hiprtcGetErrorString(err)
            extra = f"\n--- hipRTC log ---\n{log}" if log else ""
            raise HipError(f"{what}: hiprtcResult {err} ({msg.decode() if msg else '?'}){extra}")

    def version(self) -> tuple[int, int]:
        a, b = ctypes.c_int(), ctypes.c_int()
        self.check(self.lib.hiprtcVersion(ctypes.byref(a), ctypes.byref(b)), "hiprtcVersion")
        return a.value, b.value

    def compile(self, source: str, name: str, options: list[str]) -> RtcResult:
        prog = ctypes.c_void_p()
        self.check(
            self.lib.hiprtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(), 0, None, None),
            "hiprtcCreateProgram",
        )
        try:
            opts = (ctypes.c_char_p * len(options))(*[o.encode() for o in options])
            err = self.lib.hiprtcCompileProgram(prog, len(options), opts)
            n = ctypes.c_size_t()
            self.lib.hiprtcGetProgramLogSize(prog, ctypes.byref(n))
            log = ""
            if n.value > 1:
                buf = ctypes.create_string_buffer(n.value)
                self.lib.hiprtcGetProgramLog(prog, buf)
                log = buf.value.decode(errors="replace")
            self.check(err, "hiprtcCompileProgram", log)
            self.check(self.lib.hiprtcGetCodeSize(prog, ctypes.byref(n)), "hiprtcGetCodeSize")
            buf = ctypes.create_string_buffer(n.value)
            self.check(self.lib.hiprtcGetCode(prog, buf), "hiprtcGetCode")
            return RtcResult(code=buf.raw[: n.value], log=log)
        finally:
            self.lib.hiprtcDestroyProgram(ctypes.byref(prog))


def hip_runtime() -> _Runtime:
    global _rt
    with _lock:
        if _rt is None:
            _rt = _Runtime(_torch_hip_path("libamdhip64"))
        return _rt


def hiprtc() -> _Rtc:
    global _rtc
    with _lock:
        if _rtc is None:
            _rtc = _Rtc(_torch_hip_path("libhiprtc"))
        return _rtc


def same_runtime_as_torch() -> bool:
    """True if exactly one ``libamdhip64`` is mapped and it is ours."""
    rt = hip_runtime()
    paths = set()
    with open("/proc/self/maps") as f:
        for line in f:
            p = line.split()[-1] if line.split() else ""
            if os.path.basename(p).startswith("libamdhip64"):
                paths.add(os.path.realpath(p))
    return paths == {os.path.realpath(rt.path)}
