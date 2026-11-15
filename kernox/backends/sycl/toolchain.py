"""Chaîne de compilation SYCL (DPC++/icpx) et cache disque du backend ``sycl``.

Choix (validés par exécution sur Intel Arc / Battlemage G31) :

- compilateur : ``icpx -fsycl`` (oneAPI DPC++), même lignée que le runtime SYCL
  embarqué par Torch XPU (``libsycl.so.9``). Le module produit n'a pas de RPATH :
  il se lie au ``libsycl.so.9`` déjà chargé par ``import torch`` ;
- module : bibliothèque partagée exposant une fonction ``extern "C"`` chargée
  par ``ctypes`` (pas d'en-têtes Torch, compilation plus courte et pas de
  couplage avec l'ABI C++ de Torch) ;
- options numériques : ``-fp-model=precise`` imposé, car icpx compile par défaut
  en ``-fp-model=fast`` (réassociation et flush-to-zero mesurés sur le G31) ;
- cible : AOT pour les architectures connues (pas de JIT au premier lancement),
  SPIR-V (``spir64``, JIT par le pilote) sinon.

Cache : ``$KERNOX_CACHE_DIR`` ou ``~/.cache/kernox/sycl``. Clé = sha256 de la
source générée (qui contient le plan complet) + version du générateur +
version exacte du compilateur + options + cible. Écriture atomique (fichier
temporaire puis ``os.replace``) et compilation concurrente protégée par un
verrou ``fcntl`` par clé.
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

# Options numériques imposées : jamais de fast-math implicite.
NUMERIC_OPTIONS = ("-fp-model=precise", "-fno-fast-math")
BASE_OPTIONS = ("-fsycl", "-O3", "-shared", "-fPIC", "-std=c++17") + NUMERIC_OPTIONS

# Identifiants PCI Intel -> cible AOT DPC++ (vérifiés à l'exécution réelle
# uniquement pour ceux marqués « qualifié »).
AOT_TARGETS = {
    0xE223: "intel_gpu_bmg_g31",  # Battlemage G31 (qualifié)
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
    """Environnement du sous-processus compilateur uniquement (jamais celui du
    processus Python, qui garde le runtime SYCL de Torch).

    ``icpx`` du PATH s'il existe, sinon ``$KERNOX_ONEAPI_ROOT/compiler/latest/bin``
    (défaut ``/opt/intel/oneapi``) : icpx localise seul ses en-têtes et
    bibliothèques, ``setvars.sh`` (≈ 4 s) n'est pas nécessaire."""
    env = dict(os.environ)
    if shutil.which("icpx", path=env.get("PATH")):
        return env
    bindir = _oneapi_root() / "compiler" / "latest" / "bin"
    if not (bindir / "icpx").exists():
        raise ToolchainError(
            f"icpx introuvable : ni dans PATH, ni dans {bindir} (définir KERNOX_ONEAPI_ROOT)"
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
    """Cible AOT si l'architecture est connue ET ``ocloc`` est disponible,
    sinon SPIR-V générique compilé par le pilote au premier lancement (JIT).
    ``KERNOX_SYCL_TARGET`` force la cible."""
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


# Contrat v0.3 : fp_contract=False interdit la contraction FMA (frontal icpx) ;
# vérifié sur le G31 (tests/sycl/test_numerics.py) : pas de contraction dans
# le code final du pilote, dénormaux conservés.
NO_CONTRACT_OPTIONS = ("-ffp-contract=off",)


def build_options(target: str, fp_contract: bool = True) -> tuple[str, ...]:
    extra = () if fp_contract else NO_CONTRACT_OPTIONS
    return BASE_OPTIONS + extra + (f"-fsycl-targets={target}",)


def build_shared_object(source: str, target: str, fp_contract: bool = True) -> BuildResult:
    """Compile ``source`` en module partagé, avec cache atomique et verrou."""
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
            if so_path.exists():  # compilé par un autre processus pendant l'attente
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
                        "échec de compilation SYCL :\n" + " ".join(cmd) + "\n" + proc.stderr[-8000:]
                    )
                # Conserver la source à côté du binaire pour diagnostic.
                (root / f"{key}.cpp").write_text(source)
                os.replace(out, so_path)  # atomique sur le même système de fichiers
            return BuildResult(so_path, key, False, dt, options, compiler)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
