"""Minimal ctypes binding: CUDA Driver API + NVRTC (compilation to CUBIN).

Origin: structure inspired by ``symox/python/symox/extra/codegen/cuda.py``
(keckj/symox @ 2e623e66, MIT/Apache-2.0 license, same author), rewritten for
Kernox: CUBIN output (no PTX JIT), occupancy, function attributes, primary
context shared with Torch. See ``THIRD_PARTY.md``.

No memory allocation or copy happens here: buffers come from the Torch
allocator.
"""

from __future__ import annotations

import contextlib
import ctypes as ct
import ctypes.util
import os
import sys
import threading
from pathlib import Path

from ..._env import getenv


class CudaError(RuntimeError):
    """Failure of a CUDA Driver or NVRTC call."""


def _bind(lib, name, restype, *argtypes):
    fn = getattr(lib, name)
    fn.argtypes = list(argtypes)
    fn.restype = restype
    return fn


# --------------------------------------------------------------------------- driver

# CUdevice_attribute values used.
ATTR_MAX_SHARED_PER_BLOCK = 8
ATTR_MULTIPROCESSOR_COUNT = 16
ATTR_CC_MAJOR = 75
ATTR_CC_MINOR = 76
# CUfunction_attribute
FUNC_MAX_THREADS_PER_BLOCK = 0
FUNC_SHARED_SIZE_BYTES = 1
FUNC_LOCAL_SIZE_BYTES = 3
FUNC_NUM_REGS = 4


class Driver:
    def __init__(self):
        override = getenv("KERNOX_CUDA_LIBRARY")
        names = [override] if override else ["libcuda.so.1", "libcuda.so"]
        errors = []
        for n in names:
            try:
                self.lib = ct.CDLL(n)
                break
            except OSError as e:  # pragma: no cover - host-dependent
                errors.append(str(e))
        else:  # pragma: no cover
            raise ImportError("libcuda not found: " + "; ".join(errors))
        p, i, u, z = ct.c_void_p, ct.c_int, ct.c_uint, ct.c_size_t
        I = ct.c_int
        specs = {
            "cuInit": (u,),
            "cuDriverGetVersion": (ct.POINTER(i),),
            "cuDeviceGet": (ct.POINTER(i), i),
            "cuDeviceGetAttribute": (ct.POINTER(i), i, i),
            "cuDeviceGetName": (ct.c_char_p, i, i),
            "cuDevicePrimaryCtxRetain": (ct.POINTER(p), i),
            "cuDevicePrimaryCtxRelease_v2": (i,),
            "cuCtxPushCurrent_v2": (p,),
            "cuCtxGetCurrent": (ct.POINTER(p),),
            "cuCtxPopCurrent_v2": (ct.POINTER(p),),
            "cuModuleLoadData": (ct.POINTER(p), p),
            "cuModuleGetFunction": (ct.POINTER(p), p, ct.c_char_p),
            "cuModuleUnload": (p,),
            "cuFuncGetAttribute": (ct.POINTER(i), i, p),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor": (ct.POINTER(i), p, i, z),
            "cuLaunchKernel": (p, u, u, u, u, u, u, u, p, ct.POINTER(p), ct.POINTER(p)),
            "cuGetErrorName": (i, ct.POINTER(ct.c_char_p)),
            "cuGetErrorString": (i, ct.POINTER(ct.c_char_p)),
        }
        for name, args in specs.items():
            setattr(self, name, _bind(self.lib, name, I, *args))
        self.check(self.cuInit(0), "cuInit")
        v = ct.c_int()
        self.check(self.cuDriverGetVersion(ct.byref(v)), "cuDriverGetVersion")
        self.version = v.value

    def check(self, result, op):
        if result:
            name, msg = ct.c_char_p(), ct.c_char_p()
            self.cuGetErrorName(result, ct.byref(name))
            self.cuGetErrorString(result, ct.byref(msg))
            raise CudaError(
                f"{op}: {(name.value or b'?').decode()} ({(msg.value or b'').decode()})"
            )

    def call(self, name, *args):
        self.check(getattr(self, name)(*args), name)

    def device(self, ordinal: int) -> int:
        d = ct.c_int()
        self.call("cuDeviceGet", ct.byref(d), ordinal)
        return d.value

    def attribute(self, dev: int, attr: int) -> int:
        v = ct.c_int()
        self.call("cuDeviceGetAttribute", ct.byref(v), attr, dev)
        return v.value

    def name(self, dev: int) -> str:
        buf = ct.create_string_buffer(256)
        self.call("cuDeviceGetName", buf, 256, dev)
        return buf.value.decode()


_DRIVER = None
_DRIVER_LOCK = threading.Lock()


def driver() -> Driver:
    global _DRIVER
    with _DRIVER_LOCK:
        if _DRIVER is None:
            _DRIVER = Driver()
    return _DRIVER


class PrimaryContext:
    """Primary context of the device (the one used by the Torch CUDA runtime)."""

    def __init__(self, ordinal: int):
        self.drv = driver()
        self.dev = self.drv.device(ordinal)
        self.handle = ct.c_void_p()
        self.drv.call("cuDevicePrimaryCtxRetain", ct.byref(self.handle), self.dev)

    @contextlib.contextmanager
    def current(self):
        self.drv.call("cuCtxPushCurrent_v2", self.handle)
        try:
            yield
        finally:
            prev = ct.c_void_p()
            self.drv.call("cuCtxPopCurrent_v2", ct.byref(prev))

    # No release in __del__: the primary context is shared with Torch and must
    # live until the end of the process (a single retain per device).


_CONTEXTS: dict[int, PrimaryContext] = {}


def primary_context(ordinal: int) -> PrimaryContext:
    with _DRIVER_LOCK:
        ctx = _CONTEXTS.get(ordinal)
    if ctx is None:
        ctx = PrimaryContext(ordinal)
        with _DRIVER_LOCK:
            ctx = _CONTEXTS.setdefault(ordinal, ctx)
    return ctx


class Module:
    """CUBIN module loaded in the primary context; functions looked up by name."""

    def __init__(self, ctx: PrimaryContext, image: bytes):
        self.ctx = ctx
        self.handle = ct.c_void_p()
        self._image = ct.create_string_buffer(image, len(image))
        with ctx.current():
            ctx.drv.call("cuModuleLoadData", ct.byref(self.handle), self._image)

    def function(self, name: str) -> "Function":
        f = ct.c_void_p()
        with self.ctx.current():
            self.ctx.drv.call("cuModuleGetFunction", ct.byref(f), self.handle, name.encode())
        return Function(self, f)

    def __del__(self):  # pragma: no cover - process teardown
        try:
            if self.handle:
                with self.ctx.current():
                    self.ctx.drv.cuModuleUnload(self.handle)
        except Exception:
            pass


class Function:
    def __init__(self, module: Module, handle):
        self.module = module
        self.handle = handle
        drv = module.ctx.drv
        self.attrs = {}
        for key, code in (
            ("max_threads", FUNC_MAX_THREADS_PER_BLOCK),
            ("static_smem", FUNC_SHARED_SIZE_BYTES),
            ("local_bytes", FUNC_LOCAL_SIZE_BYTES),
            ("regs", FUNC_NUM_REGS),
        ):
            v = ct.c_int()
            drv.call("cuFuncGetAttribute", ct.byref(v), code, handle)
            self.attrs[key] = v.value

    def occupancy(self, block: int, smem: int) -> int:
        v = ct.c_int()
        ctx = self.module.ctx
        with ctx.current():
            ctx.drv.call(
                "cuOccupancyMaxActiveBlocksPerMultiprocessor", ct.byref(v), self.handle, block, smem
            )
        return v.value

    def launch(self, args, grid, block, smem, stream: int):
        """Asynchronous launch; ``args``: ctypes values."""
        ptrs = (ct.c_void_p * len(args))(*(ct.addressof(a) for a in args))
        self.launch_params(ptrs, grid, block, smem, stream)

    def launch_params(self, params, grid, block, smem, stream: int):
        """Asynchronous launch; ``params``: ctypes array of pointers to the
        parameter values. Fast path: if the primary context is already current
        on this thread (the usual case with Torch), two driver calls
        (cuCtxGetCurrent, cuLaunchKernel) instead of four (push/pop)."""
        ctx = self.module.ctx
        drv = ctx.drv
        cur = ct.c_void_p()
        r = drv.cuCtxGetCurrent(ct.byref(cur))
        if r == 0 and cur.value == ctx.handle.value:
            r = drv.cuLaunchKernel(self.handle, grid[0], grid[1], grid[2], block, 1, 1, smem,
                                   stream, params, None)
            if r:
                drv.check(r, "cuLaunchKernel")
            return
        with ctx.current():
            drv.call("cuLaunchKernel", self.handle, grid[0], grid[1], grid[2],
                     block, 1, 1, smem, ct.c_void_p(stream), params, None)


# --------------------------------------------------------------------------- NVRTC


def _nvrtc_candidates():
    override = getenv("KERNOX_NVRTC_LIBRARY")
    if override:
        yield override
        return
    # Priority: NVRTC installed in the environment (nvidia-cuda-nvrtc wheel,
    # pinned with torch), then the system toolkit.
    for root in sys.path:
        for p in sorted(Path(root, "nvidia/cuda_nvrtc/lib").glob("libnvrtc.so.*")):
            if "builtins" not in p.name and "alt" not in p.name:
                yield str(p)
        for p in sorted(Path(root, "nvidia/cu13/lib").glob("libnvrtc.so.*")):
            if "builtins" not in p.name:
                yield str(p)
    found = ctypes.util.find_library("nvrtc")
    if found:
        yield found
    for root in [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"]:
        if root:
            for p in sorted(Path(root, "lib64").glob("libnvrtc.so*")):
                if "builtins" not in p.name:
                    yield str(p)


class Nvrtc:
    def __init__(self):
        errors = []
        for cand in dict.fromkeys(_nvrtc_candidates()):
            try:
                self.lib = ct.CDLL(cand)
                self.path = cand
                break
            except OSError as e:
                errors.append(str(e))
        else:
            raise ImportError("libnvrtc not found: " + "; ".join(errors))
        p, i, z, c = ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_char_p
        specs = {
            "nvrtcVersion": (ct.POINTER(i), ct.POINTER(i)),
            "nvrtcCreateProgram": (ct.POINTER(p), c, c, i, ct.POINTER(c), ct.POINTER(c)),
            "nvrtcCompileProgram": (p, i, ct.POINTER(c)),
            "nvrtcGetProgramLogSize": (p, ct.POINTER(z)),
            "nvrtcGetProgramLog": (p, p),
            "nvrtcGetCUBINSize": (p, ct.POINTER(z)),
            "nvrtcGetCUBIN": (p, p),
            "nvrtcDestroyProgram": (ct.POINTER(p),),
        }
        for name, args in specs.items():
            setattr(self, name, _bind(self.lib, name, ct.c_int, *args))
        ma, mi = ct.c_int(), ct.c_int()
        self._check(self.nvrtcVersion(ct.byref(ma), ct.byref(mi)), "nvrtcVersion")
        self.version = (ma.value, mi.value)

    @staticmethod
    def _check(r, op):
        if r:
            raise CudaError(f"{op}: NVRTC error {r}")

    def compile_cubin(self, source: str, options: tuple[str, ...], filename="kernox.cu"):
        prog = ct.c_void_p()
        self._check(
            self.nvrtcCreateProgram(ct.byref(prog), source.encode(), filename.encode(), 0, None, None),
            "nvrtcCreateProgram",
        )
        try:
            opts = (ct.c_char_p * len(options))(*(o.encode() for o in options))
            r = self.nvrtcCompileProgram(prog, len(options), opts)
            size = ct.c_size_t()
            self.nvrtcGetProgramLogSize(prog, ct.byref(size))
            log = ct.create_string_buffer(size.value)
            self.nvrtcGetProgramLog(prog, log)
            log = log.value.decode(errors="replace")
            if r:
                raise CudaError(f"nvrtcCompileProgram failed ({r}):\n{log}")
            self._check(self.nvrtcGetCUBINSize(prog, ct.byref(size)), "nvrtcGetCUBINSize")
            buf = ct.create_string_buffer(size.value)
            self._check(self.nvrtcGetCUBIN(prog, buf), "nvrtcGetCUBIN")
            return buf.raw, log
        finally:
            self.nvrtcDestroyProgram(ct.byref(prog))


_NVRTC = None


def nvrtc() -> Nvrtc:
    global _NVRTC
    with _DRIVER_LOCK:
        if _NVRTC is None:
            _NVRTC = Nvrtc()
    return _NVRTC
