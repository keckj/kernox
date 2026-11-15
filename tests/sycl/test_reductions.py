"""Correction du backend sycl sur le GPU réel contre la référence FP64 CPU."""

from __future__ import annotations

import math

import pytest
import symox
import torch

from ._helpers import DEV, arg, backend, plan, reduce_ref, run

pytestmark = pytest.mark.gpu

DTYPES = ["float32", "float64"]
EPS = {"float32": 2.0**-23, "float64": 2.0**-52}


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _sq(x, y):
    return sum((a - b) ** 2 for a, b in zip(x.symbols, y.symbols))


def _pairwise_sq(X, Y):
    return ((X[:, None, :].double() - Y[None, :, :].double()) ** 2).sum(-1)


# ------------------------------------------------------------------- SUM

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M,N", [(1, 1), (37, 300), (300, 37), (1000, 777), (513, 2049)])
def test_sum_gaussian_vector(dtype, M, N):
    x, y, b, s = arg("x", "i", 3), arg("y", "j", 3), arg("b", "j", 2), arg("s", "param", 1)
    g = symox.exp(-_sq(x, y) / (2 * s.symbols[0] ** 2))
    p = plan([x, y, b, s], [g * b.symbols[0], g * b.symbols[1]], "sum", dtype)
    gen = _gen(M * 1000 + N)
    td = getattr(torch, dtype)
    X, Y, Bv = (torch.randn(M, 3, generator=gen, dtype=td), torch.randn(N, 3, generator=gen, dtype=td),
                torch.randn(N, 2, generator=gen, dtype=td))
    S = torch.tensor([0.7], dtype=td)
    out, c = run(p, {"x": X.to(DEV), "y": Y.to(DEV), "b": Bv.to(DEV), "s": S.to(DEV)})
    G = torch.exp(-_pairwise_sq(X, Y) / (2 * 0.7**2 if dtype == "float64" else 2 * float(S[0]) ** 2))
    F = G[:, :, None] * Bv.double()[None]
    ref = reduce_ref("sum", F)["value"]
    l1 = F.abs().sum(1)
    tol = 64 * EPS[dtype] * (math.sqrt(N) + 4) * l1 + 1e-300
    assert out["value"].dtype == td
    assert ((out["value"].double() - ref).abs() <= tol).all()
    from kernox.backends.sycl import TEMP_CAP_BYTES

    assert c.info.temp_bytes_per_launch == TEMP_CAP_BYTES
    assert c.last_split[2] <= TEMP_CAP_BYTES
    assert "-fp-model=precise" in c.info.options


def test_sum_float32_with_float64_accumulator():
    x, y = arg("x", "i", 1), arg("y", "j", 1)
    p = plan([x, y], [x.symbols[0] * y.symbols[0]], "sum", "float32", acc_dtype="float64")
    M, N = 64, 20000
    X = torch.ones(M, 1)
    Y = torch.full((N, 1), 0.1, dtype=torch.float32)
    out, _ = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    assert out["value"].dtype == torch.float64
    exact = N * float(torch.tensor(0.1, dtype=torch.float32))
    assert (out["value"] - exact).abs().max() < 1e-9 * exact


def test_sum_conditional_preserves_domain():
    """Branche inactive avec racine d'un négatif : aucune NaN ne doit fuir."""
    x, y = arg("x", "i", 1), arg("y", "j", 1)
    d = x.symbols[0] - y.symbols[0]
    expr = symox.Piecewise((symox.sqrt(d), d > 0), (0, True))
    p = plan([x, y], [expr], "sum", "float32")
    X = torch.linspace(-3, 3, 101).reshape(-1, 1)
    Y = torch.linspace(-2, 2, 57).reshape(-1, 1)
    out, _ = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    D = X.double() - Y.double().T
    ref = torch.where(D > 0, D.clamp(min=0).sqrt(), torch.zeros_like(D)).sum(1, keepdim=True)
    assert not torch.isnan(out["value"]).any()
    assert torch.allclose(out["value"].double(), ref, rtol=1e-5, atol=1e-5)


def test_sum_nan_propagates():
    x, y = arg("x", "i", 1), arg("y", "j", 1)
    p = plan([x, y], [x.symbols[0] * y.symbols[0]], "sum", "float32")
    Y = torch.ones(10, 1)
    Y[3] = float("nan")
    out, _ = run(p, {"x": torch.ones(4, 1, device=DEV), "y": Y.to(DEV)})
    assert torch.isnan(out["value"]).all()


def test_no_flush_to_zero_fp32():
    """icpx compile en -fp-model=fast par défaut (FTZ mesuré) : le backend doit l'éviter."""
    x, y = arg("x", "i", 1), arg("y", "j", 1)
    p = plan([x, y], [x.symbols[0] * y.symbols[0]], "sum", "float32")
    X = torch.full((3, 1), 1e-30)
    Y = torch.full((1, 1), 1e-10)
    out, _ = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    expected = float(torch.tensor(1e-30, dtype=torch.float32) * torch.tensor(1e-10, dtype=torch.float32))
    assert expected != 0.0  # dénormal FP32
    assert (out["value"] == expected).all()


# ----------------------------------------------------------------- COUNT

@pytest.mark.parametrize("dtype", DTYPES)
def test_count_radius_integer_data(dtype):
    x, y = arg("x", "i", 2), arg("y", "j", 2)
    p = plan([x, y], [_sq(x, y) <= 4, _sq(x, y) < 1], "count", dtype)
    gen = _gen(1)
    td = getattr(torch, dtype)
    X = torch.randint(-5, 6, (777, 2), generator=gen).to(td)
    Y = torch.randint(-5, 6, (1531, 2), generator=gen).to(td)
    Y[5] = float("nan")  # comparaison avec NaN fausse
    out, _ = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    D2 = _pairwise_sq(X, Y)
    ref = reduce_ref("count", torch.stack([D2 <= 4, D2 < 1], -1))["value"]
    assert out["value"].dtype == torch.int64
    assert torch.equal(out["value"], ref)


# ------------------------------------------------------- MIN / MAX / ARG*

def _minmax_data(dtype, M=301, N=1029, seed=3):
    gen = _gen(seed)
    td = getattr(torch, dtype)
    X = torch.randint(-4, 5, (M, 2), generator=gen).to(td)
    Y = torch.randint(-4, 5, (N, 2), generator=gen).to(td)
    Y[7, 0] = float("nan")
    Y[11] = float("inf")
    Y[13, 1] = float("-inf")
    return X, Y


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["min", "max", "argmin", "argmax"])
def test_minmax_vector_ties_nan_inf(dtype, kind):
    x, y = arg("x", "i", 2), arg("y", "j", 2)
    # deux composantes : produit (±inf, NaN possibles) et distance (beaucoup d'égalités)
    outs = [x.symbols[0] * y.symbols[1] - y.symbols[0], _sq(x, y)]
    p = plan([x, y], outs, kind, dtype)
    X, Y = _minmax_data(dtype)
    out, _ = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    Xd, Yd = X.double(), Y.double()
    F = torch.stack([Xd[:, None, 0] * Yd[None, :, 1] - Yd[None, :, 0], _pairwise_sq(X, Y)], -1)
    ref = reduce_ref(kind, F)
    key = "index" if kind.startswith("arg") else "value"
    if key == "value":
        assert torch.equal(out["value"].double(), ref["value"])
    else:
        assert torch.equal(out["index"], ref["index"])


@pytest.mark.parametrize("kind", ["min", "max", "argmin", "argmax"])
def test_minmax_all_nan_and_inf_candidates(kind):
    x, y = arg("x", "i", 1), arg("y", "j", 1)
    p = plan([x, y], [x.symbols[0] * y.symbols[0]], kind, "float32")
    X = torch.tensor([[float("nan")], [1.0], [0.0]])
    Y = torch.tensor([[float("inf")], [float("inf")]])
    out, _ = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    # ligne 0 : que des NaN ; ligne 1 : +inf, +inf (valides) ; ligne 2 : 0*inf = NaN
    if kind == "min":
        assert out["value"].flatten().tolist() == [math.inf, math.inf, math.inf]
    elif kind == "max":
        assert out["value"].flatten().tolist() == [-math.inf, math.inf, -math.inf]
    else:
        assert out["index"].flatten().tolist() == [-1, 0, -1]


# ------------------------------------------------------------------ K-min

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("K", [1, 5, 32, 64])
@pytest.mark.parametrize("kind", ["kmin", "argkmin", "kmin_argkmin"])
def test_kmin_ties_nan(dtype, K, kind):
    x, y = arg("x", "i", 3), arg("y", "j", 3)
    p = plan([x, y], [_sq(x, y)], kind, dtype, k=K)
    gen = _gen(K)
    td = getattr(torch, dtype)
    X = torch.randint(-3, 4, (517, 3), generator=gen).to(td)
    Y = torch.randint(-3, 4, (1100, 3), generator=gen).to(td)
    Y[::17, 1] = float("nan")
    out, _ = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    ref = reduce_ref("kmin", _pairwise_sq(X, Y)[:, :, None], k=K)
    if kind in ("kmin", "kmin_argkmin"):
        assert out["value"].shape == (517, K)
        assert torch.equal(out["value"].double(), ref["value"])
    if kind in ("argkmin", "kmin_argkmin"):
        assert torch.equal(out["index"], ref["index"])


@pytest.mark.parametrize("dtype", DTYPES)
def test_kmin_random_values(dtype):
    x, y = arg("x", "i", 4), arg("y", "j", 4)
    p = plan([x, y], [_sq(x, y)], "kmin_argkmin", dtype, k=16)
    gen = _gen(9)
    td = getattr(torch, dtype)
    X, Y = torch.randn(1000, 4, generator=gen, dtype=td), torch.randn(3000, 4, generator=gen, dtype=td)
    out, _ = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    D2 = _pairwise_sq(X, Y)
    ref = reduce_ref("kmin", D2[:, :, None], k=16)
    tol = 16 * EPS[dtype] * (D2.max())
    assert ((out["value"].double() - ref["value"]).abs() <= tol).all()
    # indices : la valeur exacte au point retourné doit égaler la valeur retournée à tol près
    picked = torch.gather(D2, 1, out["index"])
    assert ((picked - out["value"].double()).abs() <= tol).all()
    assert (out["index"] >= 0).all()


def test_kmin_k_greater_than_n():
    x, y = arg("x", "i", 1), arg("y", "j", 1)
    p = plan([x, y], [(x.symbols[0] - y.symbols[0]) ** 2], "kmin_argkmin", "float32", k=7)
    X = torch.tensor([[0.0], [10.0]])
    Y = torch.tensor([[3.0], [1.0], [float("nan")]])
    out, _ = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    inf = math.inf
    assert out["value"].tolist() == [[1, 9, inf, inf, inf, inf, inf], [49, 81, inf, inf, inf, inf, inf]]
    assert out["index"].tolist() == [[1, 0, -1, -1, -1, -1, -1], [0, 1, -1, -1, -1, -1, -1]]


def test_kmin_k_zero_no_launch():
    x, y = arg("x", "i", 1), arg("y", "j", 1)
    p = plan([x, y], [(x.symbols[0] - y.symbols[0]) ** 2], "kmin_argkmin", "float32", k=0)
    out, _ = run(p, {"x": torch.zeros(5, 1, device=DEV), "y": torch.zeros(3, 1, device=DEV)})
    assert out["value"].shape == (5, 0) and out["index"].shape == (5, 0)


@pytest.mark.parametrize("k", [-1, 2.5, True])
def test_kmin_invalid_k_refused(k):
    from kernox.backends.base import UnsupportedError

    x, y = arg("x", "i", 1), arg("y", "j", 1)
    p = plan([x, y], [(x.symbols[0] - y.symbols[0]) ** 2], "kmin", "float32", k=k)
    assert not backend().supports(p, DEV).ok
    with pytest.raises(UnsupportedError):
        backend().compile(p, DEV)


def test_kmin_vector_formula_refused():
    x, y = arg("x", "i", 1), arg("y", "j", 1)
    p = plan([x, y], [x.symbols[0], y.symbols[0]], "argkmin", "float32", k=3)
    assert not backend().supports(p, DEV).ok


# ---------------------------------------------------------- tailles vides

NEUTRAL = {
    "sum": ("value", 0.0), "count": ("value", 0), "min": ("value", math.inf),
    "max": ("value", -math.inf), "argmin": ("index", -1), "argmax": ("index", -1),
    "kmin": ("value", math.inf), "argkmin": ("index", -1),
}


@pytest.mark.parametrize("kind", list(NEUTRAL))
def test_n_zero_gives_neutral(kind):
    x, y = arg("x", "i", 2), arg("y", "j", 2)
    o = _sq(x, y) < 1 if kind == "count" else _sq(x, y)
    p = plan([x, y], [o], kind, "float32", k=3 if "kmin" in kind else None)
    out, _ = run(p, {"x": torch.randn(9, 2, device=DEV), "y": torch.empty(0, 2, device=DEV)})
    key, val = NEUTRAL[kind]
    assert (out[key] == val).all() and out[key].shape[0] == 9


@pytest.mark.parametrize("kind", list(NEUTRAL))
def test_m_zero(kind):
    x, y = arg("x", "i", 2), arg("y", "j", 2)
    o = _sq(x, y) < 1 if kind == "count" else _sq(x, y)
    p = plan([x, y], [o], kind, "float32", k=3 if "kmin" in kind else None)
    out, _ = run(p, {"x": torch.empty(0, 2, device=DEV), "y": torch.randn(5, 2, device=DEV)})
    assert all(v.shape[0] == 0 for v in out.values())


# ----------------------------------------------------- grandes dimensions

@pytest.mark.parametrize("dtype", DTYPES)
def test_large_dim_tile_shrinks(dtype):
    """D=64 en FP64 : la tuile j est réduite pour tenir dans le budget local."""
    x, y = arg("x", "i", 64), arg("y", "j", 64)
    p = plan([x, y], [_sq(x, y)], "argmin", dtype)
    gen = _gen(5)
    td = getattr(torch, dtype)
    X = torch.randint(-2, 3, (333, 64), generator=gen).to(td)
    Y = torch.randint(-2, 3, (1000, 64), generator=gen).to(td)
    out, c = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    ref = reduce_ref("argmin", _pairwise_sq(X, Y)[:, :, None])
    assert torch.equal(out["index"], ref["index"])


# ------------------------------------------------- partition de l'axe j

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind,k", [("sum", None), ("count", None), ("min", None), ("max", None),
                                    ("argmin", None), ("argmax", None), ("kmin_argkmin", 7),
                                    ("argkmin", 40)])
def test_split_j_small_m_large_n(dtype, kind, k):
    """Petit M, grand N : plusieurs tranches j + fusion. Données entières (égalités
    nombreuses entre tranches), NaN injectés : résultats exacts attendus."""
    x, y = arg("x", "i", 2), arg("y", "j", 2)
    d2 = _sq(x, y)
    outs = [d2 <= 5] if kind == "count" else ([d2] if k else [d2, x.symbols[0] * y.symbols[1]])
    p = plan([x, y], outs, kind, dtype, k=k)
    gen = _gen(17)
    td = getattr(torch, dtype)
    X = torch.randint(-6, 7, (37, 2), generator=gen).to(td)
    Y = torch.randint(-6, 7, (70001, 2), generator=gen).to(td)
    if kind != "sum":
        Y[::101, 0] = float("nan")
    out, c = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    assert c.last_split[0] > 1, c.last_split
    Xd, Yd = X.double(), Y.double()
    D2 = _pairwise_sq(X, Y)
    if kind == "count":
        F = (D2 <= 5)[:, :, None]
    elif k:
        F = D2[:, :, None]
    else:
        F = torch.stack([D2, Xd[:, None, 0] * Yd[None, :, 1]], -1)
    ref = reduce_ref("kmin" if k else kind, F, k=k)
    if kind == "sum":  # termes entiers : exact en FP64, à quelques ulps en FP32 (> 2^24)
        tol = 0 if dtype == "float64" else 1e-6 * ref["value"].abs()
        assert ((out["value"].double() - ref["value"]).abs() <= tol).all()
        return
    for key in out:
        assert torch.equal(out[key].to(ref[key].dtype), ref[key]), key


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["min", "argmin"])
def test_huge_dim_rows_from_global(dtype, kind):
    """D = 990 (conformance test_min_argmin_chunked) : ligne i lue en mémoire
    globale au lieu de registres ; auparavant, refus natif du pilote (scratch)."""
    x, y = arg("x", "i", 990), arg("y", "j", 990)
    f = symox.exp(-sum((a + b) ** 2 for a, b in zip(x.symbols, y.symbols)) / 990)
    p = plan([x, y], [f], kind, dtype)
    gen = _gen(42)
    td = getattr(torch, dtype)
    X = (torch.rand(10, 990, generator=gen, dtype=torch.float64) * 0.1).to(td)
    Y = (torch.rand(6, 990, generator=gen, dtype=torch.float64) * 0.1).to(td)
    out, c = run(p, {"x": X.to(DEV), "y": Y.to(DEV)})
    F = torch.exp(-((X.double()[:, None] + Y.double()[None]) ** 2).sum(-1) / 990)[:, :, None]
    ref = reduce_ref(kind, F)
    if kind == "min":
        tol = 1e-5 if dtype == "float32" else 1e-12
        assert torch.allclose(out["value"].double(), ref["value"], rtol=tol, atol=0)
    else:
        assert torch.equal(out["index"], ref["index"])


def test_driver_refusal_is_unsupported_error():
    """Le refus du pilote est remonté par compile() (kp_probe), jamais au lancement."""
    import ctypes

    from kernox.backends.base import UnsupportedError
    from kernox.backends.sycl import _probe

    class Fake:
        class kp_probe:  # noqa: N801
            restype = argtypes = None

            def __call__(self, q, priv, err, n):
                ctypes.memmove(err, b"total scratch space exceeds HW supported limit", 46)
                return 1

        kp_probe = kp_probe()

    with pytest.raises(UnsupportedError, match="scratch"):
        _probe(Fake(), 0)
