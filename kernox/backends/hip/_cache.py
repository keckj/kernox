"""Cache disque des code objects HIP : écriture atomique + verrou de fichier.

- Répertoire : ``$KERNOX_CACHE_DIR/hip`` ou ``~/.cache/kernox/hip``.
- Fichier ``<clé>.co`` écrit dans un temporaire du même répertoire puis
  ``os.replace`` (atomique sur un même système de fichiers) : un lecteur ne voit
  jamais un artefact partiel.
- Verrou ``fcntl.flock`` exclusif sur ``<clé>.lock`` autour de « relire puis
  compiler » : deux processus qui compilent la même clé ne compilent qu'une
  fois. Les verrous ne sont pas supprimés (supprimer un fichier de verrou
  pendant qu'un autre processus l'attend casserait l'exclusion).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from typing import Callable

from ..._env import getenv

_thread_lock = threading.Lock()


def cache_dir() -> str:
    root = getenv("KERNOX_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "kernox")
    d = os.path.join(root, "hip")
    os.makedirs(d, exist_ok=True)
    return d


def make_key(fields: dict) -> str:
    blob = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def get_or_build(key: str, build: Callable[[], bytes]) -> tuple[bytes, bool]:
    """Retourne (code object, cache_hit)."""
    d = cache_dir()
    path = os.path.join(d, key + ".co")
    try:
        with open(path, "rb") as f:
            data = f.read()
        if data:
            return data, True
    except FileNotFoundError:
        pass
    with _thread_lock, open(os.path.join(d, key + ".lock"), "a+b") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                if data:
                    return data, True
            except FileNotFoundError:
                pass
            data = build()
            fd, tmp = tempfile.mkstemp(dir=d, prefix=key[:16] + ".", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except FileNotFoundError:
                    pass
                raise
            return data, False
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def invalidate(key: str) -> None:
    try:
        os.unlink(os.path.join(cache_dir(), key + ".co"))
    except FileNotFoundError:
        pass
