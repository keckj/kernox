"""Contrat v0.3 : contraction FMA (``Plan.fp_contract``) sur le backend sycl.

``a*b - c`` avec a = b = 1 + 2^-h et c = 1 + 2^(1-h) : le produit exact vaut
c + 2^-2h ; arrondi seul (sans FMA), il donne c (égalité à mi-chemin, pair) et
le résultat est 0 ; contracté en FMA, il vaut exactement 2^-2h.
h = 12 en FP32, h = 27 en FP64.
"""

from __future__ import annotations

import pytest
import torch

from ._helpers import DEV, arg, backend, plan, run

H = {"float32": 12, "float64": 27}


def _fma_case(dtype, fp_contract):
    x, y = arg("x", "i", 2), arg("y", "j", 1)
    f = x.symbols[0] * y.symbols[0] - x.symbols[1]
    p = plan([x, y], [f], "sum", dtype, fp_contract=fp_contract)
    h = H[dtype]
    td = getattr(torch, dtype)
    a = 1 + 2.0 ** -h
    X = torch.tensor([[a, 1 + 2.0 ** (1 - h)]], dtype=td)
    Y = torch.tensor([[a]], dtype=td)
    out, c = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    return float(out["value"][0, 0]), c, 2.0 ** (-2 * h)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fp_contract_false_forbids_fma(dtype):
    v, c, _ = _fma_case(dtype, False)
    assert v == 0.0
    assert "-ffp-contract=off" in c.info.options


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fp_contract_true_allows_fma(dtype):
    """Défaut : contraction autorisée (et effectivement appliquée sur le G31)."""
    v, c, fma = _fma_case(dtype, True)
    assert v in (0.0, fma)
    assert "-ffp-contract=off" not in c.info.options
    assert v == fma  # constaté sur le G31 (icpx 2026.2, IGC 2.28)


def test_fp_contract_in_cache_key():
    v1, c1, _ = _fma_case("float32", True)
    v0, c0, _ = _fma_case("float32", False)
    assert c0.info.options != c1.info.options
    assert c0._lib is not c1._lib


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_no_contract_keeps_denormals(dtype):
    x, y = arg("x", "i", 1), arg("y", "j", 1)
    p = plan([x, y], [x.symbols[0] * y.symbols[0]], "sum", dtype, fp_contract=False)
    tiny = (1e-30, 1e-10) if dtype == "float32" else (1e-300, 1e-10)
    td = getattr(torch, dtype)
    X = torch.full((3, 1), tiny[0], dtype=td)
    Y = torch.full((1, 1), tiny[1], dtype=td)
    out, _ = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    expected = float(torch.tensor(tiny[0], dtype=td) * torch.tensor(tiny[1], dtype=td))
    assert expected != 0.0 and abs(expected) < torch.finfo(td).tiny
    assert (out["value"] == expected).all()

