"""Lecture des variables d'environnement ``KERNOX_*``.

Une valeur vide équivaut à une variable absente.
"""

from __future__ import annotations

import os


def getenv(name: str, default: str | None = None) -> str | None:
    """Valeur de ``name`` si elle est définie et non vide, sinon ``default``."""
    return os.environ.get(name) or default
