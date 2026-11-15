"""On-disk cache of HIP code objects: atomic writes + file lock.

- Directory: ``$KERNOX_CACHE_DIR/hip`` or ``~/.cache/kernox/hip``.
- The ``<key>.co`` file is written to a temporary file in the same directory,
  then moved with ``os.replace`` (atomic within a filesystem): a reader never
  sees a partial artifact.
- An exclusive ``fcntl.flock`` lock on ``<key>.lock`` wraps "re-read, then
  compile": two processes compiling the same key only compile it once. Lock
  files are never deleted (deleting a lock file while another process waits
  on it would break mutual exclusion).
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
    """Return (code object, cache_hit)."""
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
