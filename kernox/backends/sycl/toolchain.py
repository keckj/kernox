"""SYCL toolchain (DPC++/icpx) and on-disk cache of the ``sycl`` backend.

Choices (validated by running on Intel Arc / Battlemage G31):

- compiler: ``icpx -fsycl`` (oneAPI DPC++), same lineage as the SYCL runtime
  shipped with Torch XPU (``libsycl.so.9``). The produced module has no RPATH:
  it links against the ``libsycl.so.9`` already loaded by ``import torch``;
- module: shared library exposing an ``extern "C"`` function loaded through
  ``ctypes`` (no Torch headers, shorter compilation and no coupling with the
  Torch C++ ABI);
- numeric options: ``-fp-model=precise`` is enforced, because icpx compiles
  with ``-fp-model=fast`` by default (reassociation and flush-to-zero measured
  on the G31);
- target: AOT for known architectures (no JIT on first launch), otherwise
  SPIR-V (``spir64``, JIT-compiled by the driver).

Cache: ``$KERNOX_CACHE_DIR`` or ``~/.cache/kernox/sycl``. Key = sha256 of the
generated source (which contains the full plan) + generator version + exact
compiler version + options + target. Atomic write (temporary file then
``os.replace``); concurrent compilation is guarded by a per-key ``fcntl``
lock.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..._env import getenv

GENERATOR_VERSION = "sycl-tiled-1d-v0"

# Enforced numeric options: never any implicit fast-math.
NUMERIC_OPTIONS = ("-fp-model=precise", "-fno-fast-math")
BASE_OPTIONS = ("-fsycl", "-O3", "-shared", "-fPIC", "-std=c++17") + NUMERIC_OPTIONS

# Intel PCI IDs -> DPC++ AOT target (verified by actual execution only for
# those marked "qualified").
AOT_TARGETS = {
    0xE223: "intel_gpu_bmg_g31",  # Battlemage G31 (qualified)
    0xE220: "intel_gpu_bmg_g31",
    0xE221: "intel_gpu_bmg_g31",
    0xE222: "intel_gpu_bmg_g31",
    0xE20B: "intel_gpu_bmg_g21",
    0xE20C: "intel_gpu_bmg_g21",
    0xE20D: "intel_gpu_bmg_g21",
    0xE210: "intel_gpu_bmg_g21",
    0xE211: "intel_gpu_bmg_g21",
    0xE212: "intel_gpu_bmg_g21",
}


class ToolchainError(RuntimeError):
    pass


def _oneapi_root() -> Path:
    return Path(getenv("KERNOX_ONEAPI_ROOT", "/opt/intel/oneapi"))


@lru_cache(maxsize=1)
def compiler_env() -> dict:
    """Environment of the compiler subprocess only (never that of the Python
    process, which keeps Torch's SYCL runtime).

    ``icpx`` from PATH if present, otherwise ``$KERNOX_ONEAPI_ROOT/compiler/latest/bin``
    (default ``/opt/intel/oneapi``): icpx finds its own headers and libraries,
    so ``setvars.sh`` (about 4 s) is not needed."""
    env = dict(os.environ)
    if shutil.which("icpx", path=env.get("PATH")):
        return env
    bindir = _oneapi_root() / "compiler" / "latest" / "bin"
    if not (bindir / "icpx").exists():
        raise ToolchainError(
            f"icpx not found: neither in PATH nor in {bindir} (set KERNOX_ONEAPI_ROOT)"
        )
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    return env


@lru_cache(maxsize=1)
def compiler_version() -> str:
    env = compiler_env()
    out = subprocess.run(["icpx", "--version"], env=env, check=True,
                         capture_output=True, text=True).stdout
    return out.splitlines()[0].strip()


def has_ocloc() -> bool:
    return shutil.which("ocloc", path=compiler_env().get("PATH")) is not None


def target_for_device_id(device_id: int) -> str:
    """AOT target if the architecture is known AND ``ocloc`` is available,
    otherwise generic SPIR-V compiled by the driver on first launch (JIT).
    ``KERNOX_SYCL_TARGET`` forces the target."""
    forced = getenv("KERNOX_SYCL_TARGET")
    if forced:
        return forced
    aot = AOT_TARGETS.get(int(device_id))
    if aot is not None and has_ocloc():
        return aot
    return "spir64"


def cache_dir() -> Path:
    d = getenv("KERNOX_CACHE_DIR")
    base = Path(d) if d else Path.home() / ".cache" / "kernox"
    p = base / "sycl"
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


# Contract v0.3: fp_contract=False forbids FMA contraction (icpx front end);
# verified on the G31 (tests/sycl/test_numerics.py): no contraction in the
# driver's final code, denormals preserved.
NO_CONTRACT_OPTIONS = ("-ffp-contract=off",)


def build_options(target: str, fp_contract: bool = True) -> tuple[str, ...]:
    extra = () if fp_contract else NO_CONTRACT_OPTIONS
    return BASE_OPTIONS + extra + (f"-fsycl-targets={target}",)


def build_shared_object(source: str, target: str, fp_contract: bool = True) -> BuildResult:
    """Compile ``source`` into a shared module, with an atomic cache and a lock."""
    options = build_options(target, fp_contract)
    compiler = compiler_version()
    h = hashlib.sha256()
    for part in (GENERATOR_VERSION, compiler, " ".join(options), target, source):
        h.update(part.encode())
        h.update(b"\0")
    key = h.hexdigest()
    root = cache_dir()
    so_path = root / f"{key}.so"
    if so_path.exists():
        return BuildResult(so_path, key, True, 0.0, options, compiler)
    lock_path = root / f"{key}.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if so_path.exists():  # compiled by another process while waiting
                return BuildResult(so_path, key, True, 0.0, options, compiler)
            with tempfile.TemporaryDirectory(dir=root, prefix=f".{key[:16]}-") as tmp:
                src = Path(tmp) / "kernel.cpp"
                src.write_text(source)
                out = Path(tmp) / "kernel.so"
                cmd = ["icpx", *options, str(src), "-o", str(out)]
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, env=compiler_env(), capture_output=True, text=True)
                dt = time.perf_counter() - t0
                if proc.returncode != 0:
                    raise ToolchainError(
                        "SYCL compilation failed:\n" + " ".join(cmd) + "\n" + proc.stderr[-8000:]
                    )
                # Keep the source next to the binary for diagnostics.
                (root / f"{key}.cpp").write_text(source)
                os.replace(out, so_path)  # atomic on the same filesystem
            return BuildResult(so_path, key, False, dt, options, compiler)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
