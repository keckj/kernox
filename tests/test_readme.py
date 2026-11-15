"""Exécute les blocs ```python du README dans l'ordre, dans un même espace de noms."""

import re
from pathlib import Path

import pytest

pytest.importorskip("torch")


def _blocks():
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def test_readme_examples_run():
    import torch

    blocks = _blocks()
    assert len(blocks) >= 3
    ns = {}
    for code in blocks:
        exec(compile(code, "README.md", "exec"), ns)  # noqa: S102 - exemples du dépôt
    assert ns["out"].shape == (2000, 2) and ns["out"].dtype == torch.float32
    assert ns["idx"].shape == (2000, 8) and ns["idx"].dtype == torch.int64
    assert ns["counts"].shape == (2000, 1)
    g = ns["X64"].grad
    assert g is not None and g.shape == (2000, 3) and torch.isfinite(g).all()
