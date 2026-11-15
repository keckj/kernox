"""Top-K coopératif (variante ``coop`` : G lanes d'un sous-groupe par ligne i).

Contre la référence FP64 CPU, résultats exacts attendus : données entières
(nombreuses égalités, règle du plus petit j), NaN injectés, K aux bornes
(1, multiple ou non de G, K = N, K > N), partition de l'axe j (fusion
coopérative), lignes partiellement actives (M non multiple des lignes par
work-group).
"""

from __future__ import annotations

import pytest
import torch

from kernox.backends.sycl import _variant
from kernox.backends.sycl.codegen import COOP_GROUPS

from ._helpers import DEV, arg, backend, plan, reduce_ref, run

DTYPES = ["float32", "float64"]


def _sq(x, y):
    return sum((a - b) ** 2 for a, b in zip(x.symbols, y.symbols))


def _case(dtype, kind, k, M, N, D=3, seed=0, nan=True, lo=-3, hi=4):
    x, y = arg("x", "i", D), arg("y", "j", D)
    p = plan([x, y], [_sq(x, y)], kind, dtype, k=k)
    gen = torch.Generator().manual_seed(seed)
    td = getattr(torch, dtype)
    X = torch.randint(lo, hi, (M, D), generator=gen).to(td)
    Y = torch.randint(lo, hi, (N, D), generator=gen).to(td)
    if nan and N:
        Y[::13, 0] = float("nan")
    F = ((X.double()[:, None, :] - Y.double()[None, :, :]) ** 2).sum(-1)[:, :, None]
    return p, X, Y, reduce_ref("kmin", F, k=k)


def _check(p, X, Y, ref, variant, split=None):
    c = backend().compile(p, DEV, variant)
    if split is not None:
        c.target_groups = split
    out, c = run(p, {"x": X.to(DEV), "y": Y.to(DEV)}, compiled=c)
    for key in out:
        assert torch.equal(out[key].to(ref[key].dtype), ref[key]), key
    return c


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("G", [2, 4, 8, 16])
@pytest.mark.parametrize("k", [1, 7, 32, 64, 129])
def test_coop_kmin_argkmin_ties_nan(dtype, G, k):
    p, X, Y, ref = _case(dtype, "kmin_argkmin", k, M=203, N=1500, seed=G * 1000 + k)
    _check(p, X, Y, ref, _variant(sg=16, coop=G))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["kmin", "argkmin"])
def test_coop_single_outputs(dtype, kind):
    p, X, Y, ref = _case(dtype, kind, 64, M=77, N=900, seed=3)
    _check(p, X, Y, ref, _variant(sg=16, coop=4))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k,N", [(50, 50), (64, 37), (256, 300), (130, 0)])
def test_coop_k_bounds(dtype, k, N):
    """K = N, K > N (rangs vides : +inf / -1), K = 256, N = 0."""
    p, X, Y, ref = _case(dtype, "kmin_argkmin", k, M=45, N=N, seed=k + N)
    _check(p, X, Y, ref, _variant(sg=16, coop=16))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("G,k", [(4, 64), (8, 128), (16, 128), (16, 256)])
def test_coop_split_j_merge(dtype, G, k):
    """Petit M, grand N : plusieurs tranches + fusion coopérative ; égalités entre tranches."""
    p, X, Y, ref = _case(dtype, "kmin_argkmin", k, M=21, N=60001, D=2, seed=G + k, lo=-5, hi=6)
    c = _check(p, X, Y, ref, _variant(sg=16, coop=G))
    assert c.last_split[0] > 1, c.last_split


def test_coop_split_j_forced_many_slices():
    """Beaucoup de tranches courtes (fusion longue), valeurs variées exactes en FP32."""
    x, y = arg("x", "i", 3), arg("y", "j", 3)
    p = plan([x, y], [_sq(x, y)], "kmin_argkmin", "float32", k=100)
    gen = torch.Generator().manual_seed(9)
    X = torch.randint(-1000, 1000, (5, 3), generator=gen).float() / 8
    Y = torch.randint(-1000, 1000, (40000, 3), generator=gen).float() / 8
    F = ((X.double()[:, None] - Y.double()[None]) ** 2).sum(-1)[:, :, None]
    ref = reduce_ref("kmin", F, k=100)
    c = _check(p, X, Y, ref, _variant(sg=16, coop=8, groups=4096))
    assert c.last_split[0] > 30, c.last_split


@pytest.mark.parametrize("dtype", DTYPES)
def test_coop_all_nan_rows(dtype):
    p, X, Y, ref = _case(dtype, "kmin_argkmin", 40, M=33, N=500, seed=1, nan=False)
    Y[:, 1] = float("nan")
    F = torch.full((33, 500, 1), float("nan"), dtype=torch.float64)
    ref = reduce_ref("kmin", F, k=40)
    _check(p, X, Y, ref, _variant(sg=16, coop=8))


def test_coop_inf_values():
    x, y = arg("x", "i", 1), arg("y", "j", 1)
    p = plan([x, y], [x.symbols[0] * y.symbols[0]], "kmin_argkmin", "float32", k=33)
    X = torch.tensor([[1.0], [-1.0], [0.0]])
    Y = torch.tensor([[float("inf")], [2.0], [float("-inf")], [float("nan")], [2.0]] * 20)
    F = (X.double() * Y.double().T)[:, :, None]
    ref = reduce_ref("kmin", F, k=33)
    _check(p, X, Y, ref, _variant(sg=16, coop=4))


def test_coop_sg32():
    p, X, Y, ref = _case("float32", "kmin_argkmin", 128, M=100, N=2000, seed=5)
    _check(p, X, Y, ref, _variant(sg=32, coop=32))


def test_coop_invalid_refused():
    from kernox.backends.base import UnsupportedError

    p, *_ = _case("float32", "kmin_argkmin", 64, M=4, N=4)
    with pytest.raises(UnsupportedError):
        backend().compile(p, DEV, _variant(sg=0, coop=4))
    with pytest.raises(UnsupportedError):
        backend().compile(p, DEV, _variant(sg=16, coop=3))
    with pytest.raises(UnsupportedError):
        backend().compile(p, DEV, _variant(sg=16, coop=32))
    x, y = arg("x", "i", 3), arg("y", "j", 3)
    ps = plan([x, y], [_sq(x, y)], "argmin", "float32")
    with pytest.raises(UnsupportedError):
        backend().compile(ps, DEV, _variant(sg=16, coop=4))
    assert set(COOP_GROUPS) >= {2, 4, 8, 16}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", [512, 1024])
def test_coop_very_large_k_default_variant(dtype, k):
    """L = K/16 > 16 rangs par lane (GRF 256), variante par défaut ; K > N et partition j."""
    p, X, Y, ref = _case(dtype, "kmin_argkmin", k, M=40, N=3000, seed=k)
    v = backend().default_variant(p, DEV, None)
    assert dict(v.params)["coop"] in (16, 32) and dict(v.params)["grf"] == 256
    _check(p, X, Y, ref, v)
    p, X, Y, ref = _case(dtype, "argkmin", k, M=30, N=700, seed=k + 1)
    _check(p, X, Y, ref, v)
    p, X, Y, ref = _case(dtype, "kmin_argkmin", k, M=7, N=40001, D=2, seed=k + 2, lo=-5, hi=6)
    c = _check(p, X, Y, ref, v)
    assert c.last_split[0] > 1, c.last_split


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("G", [2, 4, 8])
def test_coop_k16(dtype, G):
    """Seuil coop : K = 16 en variante coop (comparaison au registre par ligne)."""
    p, X, Y, ref = _case(dtype, "kmin_argkmin", 16, M=150, N=2000, seed=G)
    _check(p, X, Y, ref, _variant(sg=16, coop=G))


def test_default_threshold_k16_by_dtype():
    """Seuils mesurés : coop dès K = 16 en FP64, dès K = 32 en FP32."""
    for dtype, k, coop in (("float64", 16, 4), ("float32", 16, 0), ("float32", 32, 8), ("float64", 8, 0)):
        p, *_ = _case(dtype, "argkmin", k, M=4, N=4)
        assert dict(backend().default_variant(p, DEV, None).params)["coop"] == coop, (dtype, k)


def test_default_k512_fp32_uses_sg32():
    p, X, Y, ref = _case("float32", "kmin_argkmin", 512, M=40, N=3000, seed=11)
    v = backend().default_variant(p, DEV, None)
    assert dict(v.params)["coop"] == 32 and dict(v.params)["sg"] == 32
    _check(p, X, Y, ref, v)
    for dt, k in (("float64", 512), ("float32", 1024), ("float32", 256)):
        q, *_ = _case(dt, "argkmin", k, M=4, N=4)
        assert dict(backend().default_variant(q, DEV, None).params)["coop"] == 16, (dt, k)
