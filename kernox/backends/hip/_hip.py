"""Liaisons ctypes minimales vers le runtime HIP et hipRTC **de Torch ROCm**.

Règle centrale (voir ``docs/backend-hip.md``) : on charge exactement les
bibliothèques ``libamdhip64`` et ``libhiprtc`` que Torch a déjà chargées dans
le processus (paquet ``_rocm_sdk_core`` des wheels ROCm officielles, ou
``torch/lib`` pour les anciennes wheels). Charger un second runtime HIP (par ex.
``/opt/rocm/lib``) créerait un second état de runtime : contextes, streams et
allocations de Torch n'y seraient pas valides. ``hip_runtime()`` vérifie donc
via ``/proc/self/maps`` que la bibliothèque ouverte est bien celle de Torch.
"""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass

_lock = threading.Lock()
_rt = None
_rtc = None

# hipFunction_attribute (hip/driver_types.h, énumération commençant à 0)
FUNC_ATTR_MAX_THREADS_PER_BLOCK = 0
FUNC_ATTR_SHARED_SIZE_BYTES = 1
FUNC_ATTR_LOCAL_SIZE_BYTES = 3
FUNC_ATTR_NUM_REGS = 4


# Toujours RTLD_LOCAL. Rouvrir avec RTLD_GLOBAL une bibliothèque que Torch a déjà chargée
# en local la promeut, avec ses dépendances (libamd_comgr → libLLVM.so de _rocm_sdk_core),
# dans la portée globale. Les constructeurs statiques de ``triton/_C/libtriton.so`` (LLVM
# embarquée, symboles visibles) se lient alors à ``llvm::cl::Option::addArgument`` de
# libLLVM.so : SIGSEGV à l'import de torch._dynamo / triton (torch.compile après Kernox).
# ctypes résout nos symboles par le handle : la portée globale n'est jamais nécessaire.
_DLOPEN_MODE = os.RTLD_LOCAL | os.RTLD_NOW


class HipError(RuntimeError):
    pass


def _loaded_path(fragment: str) -> str | None:
    """Chemin d'une bibliothèque déjà projetée dans le processus."""
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
        raise HipError("Torch n'est pas une build ROCm (torch.version.hip est None)")
    # Garantit que Torch a initialisé et projeté son runtime HIP.
    torch.cuda.init()
    path = _loaded_path(fragment)
    if path is None and fragment == "libhiprtc":
        # hipRTC peut être chargé paresseusement : on cherche à côté du runtime.
        rt = _loaded_path("libamdhip64")
        if rt is not None:
            d = os.path.dirname(rt)
            for name in sorted(os.listdir(d)):
                if name.startswith("libhiprtc.so") and "builtins" not in name:
                    path = os.path.join(d, name)
                    break
    if path is None:
        raise HipError(f"bibliothèque {fragment} de Torch introuvable dans le processus")
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
            # Consomme l'erreur « last error » du runtime : sinon Torch la relève
            # au prochain contrôle de lancement et l'attribue à son propre kernel.
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
        """``args`` : objets ctypes (valeurs) ; ils doivent vivre jusqu'au retour."""
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
            extra = f"\n--- journal hipRTC ---\n{log}" if log else ""
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
    """Vrai si une seule ``libamdhip64`` est projetée et que c'est la nôtre."""
    rt = hip_runtime()
    paths = set()
    with open("/proc/self/maps") as f:
        for line in f:
            p = line.split()[-1] if line.split() else ""
            if os.path.basename(p).startswith("libamdhip64"):
                paths.add(os.path.realpath(p))
    return paths == {os.path.realpath(rt.path)}
