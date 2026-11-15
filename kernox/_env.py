"""Reading of the ``KERNOX_*`` environment variables.

An empty value is treated as an unset variable.
"""

from __future__ import annotations

import os


def getenv(name: str, default: str | None = None) -> str | None:
    """Value of ``name`` if it is set and non-empty, otherwise ``default``."""
    return os.environ.get(name) or default
