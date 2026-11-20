"""Host C++ toolchain (g++ + OpenMP) and on-disk cache of the ``cpu`` backend.

Choices:

- compiler: ``g++`` (``KERNOX_CPU_CXX`` overrides it). The module is linked
  against ``libgomp.so.1``; the CPU wheels of PyTorch ship a ``libgomp.so.1``
  with the same soname, already loaded by ``import torch``, so the dynamic
  loader reuses it: one OpenMP runtime in the process, no second thread pool.
  The generated code only uses ``omp parallel`` (``GOMP_parallel``, GOMP 4.0)
  and ``omp_get_thread_num``, available in every libgomp still in use;
- module: shared library exposing an ``extern "C"`` function loaded through
  ``ctypes`` (no Torch headers, no coupling with the Torch C++ ABI);
- target: ``-march=native``. The cache key holds the architecture resolved by
  the compiler and the full list of target options it implies, so a cache
  shared between machines (NFS home directory) never hands a binary to a CPU
  that cannot run it;
- numeric options: never fast-math. ``-fno-math-errno`` (libm does not set
  ``errno``, which lets ``sqrt`` compile to one instruction) and
  ``-fno-trapping-math`` (floating-point exception flags are not observed,
  which lets ``where`` branches be vectorized with blends) change no value.
  ``-fexcess-precision=standard`` rounds every ``_Float16`` operation to
  binary16. Contraction: ``-ffp-contract=fast`` when ``fp_contract`` is True,
  ``-ffp-contract=off`` otherwise;
- vector math: libmvec (see ``libmvec_functions``), linked with ``-lmvec``.

Cache: ``$KERNOX_CACHE_DIR/cpu`` or ``~/.cache/kernox/cpu``. Key = sha256 of the
generated source (which contains the full plan) + generator version + exact
compiler version + options + target. Atomic write (temporary file then
``os.replace``); concurrent compilation is guarded by a per-key ``fcntl``
lock.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..._env import getenv

GENERATOR_VERSION = "cpu-tiled-lanes-v1"

NUMERIC_OPTIONS = ("-fno-fast-math", "-fno-math-errno", "-fno-trapping-math", "-fexcess-precision=standard")
BASE_OPTIONS = ("-O3", "-march=native", "-std=c++17", "-fPIC", "-shared", "-fopenmp") + NUMERIC_OPTIONS


class ToolchainError(RuntimeError):
    pass


def compiler() -> str:
    cxx = getenv("KERNOX_CPU_CXX", "g++")
    if shutil.which(cxx) is None:
        raise ToolchainError(f"C++ compiler {cxx!r} not found (set KERNOX_CPU_CXX)")
    return cxx


@lru_cache(maxsize=4)
def _version(cxx: str) -> str:
    out = subprocess.run([cxx, "--version"], check=True, capture_output=True, text=True).stdout
    return out.splitlines()[0].strip()


def compiler_version() -> str:
    return _version(compiler())


@lru_cache(maxsize=4)
def _native_target(cxx: str) -> tuple[str, str]:
    """(architecture resolved by ``-march=native``, sha256 of the implied target options)."""
    q = subprocess.run([cxx, "-march=native", "-Q", "--help=target"], capture_output=True, text=True)
    m = re.search(r"^\s*-march=\s*(\S+)", q.stdout, re.M)
    arch = m.group(1) if m else "native"
    # The cc1plus command line lists every -m option implied by -march=native.
    v = subprocess.run([cxx, "-march=native", "-x", "c++", "-E", "-", "-###"], input="",
                       capture_output=True, text=True)
    lines = [ln for ln in v.stderr.splitlines() if "cc1" in ln]
    flags = " ".join(sorted(t for t in " ".join(lines).split() if t.strip('"').startswith(("-m", "--param"))))
    return arch, hashlib.sha256(flags.encode()).hexdigest()[:16]


def native_target() -> tuple[str, str]:
    return _native_target(compiler())


def cache_dir() -> Path:
    d = getenv("KERNOX_CACHE_DIR")
    base = Path(d) if d else Path.home() / ".cache" / "kernox"
    p = base / "cpu"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class BuildResult:
    path: Path
    key: str
    cache_hit: bool
    compile_seconds: float
    options: tuple[str, ...]
    compiler: str


def build_options(fp_contract: bool = True) -> tuple[str, ...]:
    return BASE_OPTIONS + (("-ffp-contract=fast",) if fp_contract else ("-ffp-contract=off",))


def build_shared_object(source: str, fp_contract: bool = True, libs: tuple[str, ...] = ()) -> BuildResult:
    """Compile ``source`` into a shared module, with an atomic cache and a lock."""
    cxx = compiler()
    options = build_options(fp_contract) + tuple(libs)
    version = _version(cxx)
    arch, flags = _native_target(cxx)
    h = hashlib.sha256()
    for part in (GENERATOR_VERSION, cxx, version, " ".join(options), arch, flags, source):
        h.update(part.encode())
        h.update(b"\0")
    key = h.hexdigest()
    root = cache_dir()
    so_path = root / f"{key}.so"
    if so_path.exists():
        return BuildResult(so_path, key, True, 0.0, options, version)
    lock_path = root / f"{key}.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if so_path.exists():  # compiled by another process while waiting
                return BuildResult(so_path, key, True, 0.0, options, version)
            with tempfile.TemporaryDirectory(dir=root, prefix=f".{key[:16]}-") as tmp:
                src = Path(tmp) / "kernel.cpp"
                src.write_text(source)
                out = Path(tmp) / "kernel.so"
                cmd = [cxx, *build_options(fp_contract), str(src), "-o", str(out), *libs]
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                dt = time.perf_counter() - t0
                if proc.returncode != 0:
                    raise ToolchainError(
                        "CPU compilation failed:\n" + " ".join(cmd) + "\n" + proc.stderr[-8000:]
                    )
                # Keep the source next to the binary for diagnostics.
                (root / f"{key}.cpp").write_text(source)
                os.replace(out, so_path)  # atomic on the same filesystem
            return BuildResult(so_path, key, False, dt, options, version)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# libmvec (glibc vector math): functions whose SIMD variants are declared to the
# compiler (``__attribute__((simd("notinbranch")))``), so that a t loop calling
# them vectorizes. glibc only declares them itself under ``-ffast-math``; here no
# fast-math option is used (NaN, inf, contraction, exceptions unchanged): only the
# libm changes. Measured accuracy (glibc 2.43, x86-64): <= 4 ulp, the accuracy
# class of the CUDA/SYCL device libm. ``KERNOX_CPU_LIBMVEC=0`` disables it.
LIBMVEC_CORE = ("exp", "log", "sin", "cos", "pow")
LIBMVEC_EXTRA = ("acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "cos", "cosh", "erf",
                 "erfc", "exp10", "exp2", "expm1", "hypot", "log10", "log1p", "log2", "sinh", "tan", "tanh")
_ARITY2 = ("atan2", "pow", "hypot")


def libmvec_declarations(functions) -> str:
    out = ['extern "C" {']
    for f in functions:
        args = "float, float" if f in _ARITY2 else "float"
        dargs = "double, double" if f in _ARITY2 else "double"
        out.append(f'__attribute__((__simd__("notinbranch"))) float {f}f({args}) noexcept;')
        out.append(f'__attribute__((__simd__("notinbranch"))) double {f}({dargs}) noexcept;')
    out.append("}")
    return "\n".join(out)


def _link_ok(cxx: str, functions) -> bool:
    calls = "\n".join(
        f"  for (int i = 0; i < n; ++i) {{ a[i] = {f}f(a[i]{', a[i]' if f in _ARITY2 else ''}); "
        f"b[i] = {f}(b[i]{', b[i]' if f in _ARITY2 else ''}); }}"
        for f in functions)
    src = ("#include <cmath>\n" + libmvec_declarations(functions)
           + "\nextern \"C\" void kp_probe(float* __restrict__ a, double* __restrict__ b, int n) {\n"
           + "#pragma omp simd\n" + calls.replace("\n  for", "\n#pragma omp simd\n  for") + "\n}\n")
    with tempfile.TemporaryDirectory() as tmp:
        cpp = Path(tmp) / "probe.cpp"
        cpp.write_text(src)
        proc = subprocess.run([cxx, "-O2", "-march=native", "-fopenmp", "-fno-math-errno", "-fPIC", "-shared",
                               str(cpp), "-o", str(Path(tmp) / "probe.so"), "-Wl,--no-undefined", "-lmvec", "-lm"],
                              capture_output=True, text=True)
        return proc.returncode == 0


@lru_cache(maxsize=4)
def _libmvec(cxx: str) -> tuple[str, ...]:
    full = tuple(sorted(set(LIBMVEC_CORE) | set(LIBMVEC_EXTRA)))
    for cand in (full, LIBMVEC_CORE):
        if _link_ok(cxx, cand):
            return cand
    return ()


def libmvec_functions() -> tuple[str, ...]:
    """Math functions vectorized through libmvec (empty: disabled or unavailable)."""
    if getenv("KERNOX_CPU_LIBMVEC", "1").strip().lower() in ("0", "false", "no", "off"):
        return ()
    try:
        return _libmvec(compiler())
    except (ToolchainError, OSError):
        return ()


@lru_cache(maxsize=4)
def _float16_ok(cxx: str) -> bool:
    """The compiler accepts ``_Float16`` arithmetic for the target (GCC >= 12 on x86-64)."""
    src = "_Float16 f(_Float16 a, _Float16 b) { return a * b + a; }\n"
    proc = subprocess.run([cxx, "-march=native", "-std=c++17", "-fexcess-precision=standard",
                           "-x", "c++", "-fsyntax-only", "-"], input=src, capture_output=True, text=True)
    return proc.returncode == 0


def float16_supported() -> bool:
    try:
        return _float16_ok(compiler())
    except (ToolchainError, OSError):
        return False
