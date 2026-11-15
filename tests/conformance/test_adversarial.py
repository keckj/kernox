"""Données adverses : égalités massives, NaN, ±inf, −0.0, dénormaux, valeurs
énormes ou minuscules, doublons exacts, entrées constantes, et domaines
invalides exclus par ``where`` (la branche inactive ne doit pas contaminer le
résultat).

Les formules ``x - y`` et ``abs(x - y)`` transportent les valeurs spéciales
sans les créer : avec ``x = 0`` la valeur du couple est exactement ``-y_j``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ._support import TINY, R, run_case

import kernox as kp

DTYPES = ("float32", "float64")
KINDS = (R.SUM, R.MIN, R.MAX, R.ARGMIN, R.ARGMAX, R.KMIN, R.ARGKMIN, R.KMIN_ARGKMIN, R.COUNT)
SIZES = ((1, 1), (5, 33), (64, 257), (3, 1031))


def _reduce(f, kind, k=4):
    return {
        R.SUM: lambda: f.sum_over("j"), R.MIN: lambda: f.min(over="j"), R.MAX: lambda: f.max(over="j"),
        R.ARGMIN: lambda: f.argmin(over="j"), R.ARGMAX: lambda: f.argmax(over="j"),
        R.KMIN: lambda: f.kmin(k, over="j"), R.ARGKMIN: lambda: f.argkmin(k, over="j"),
        R.KMIN_ARGKMIN: lambda: f.kmin_argkmin(k, over="j"),
    }[kind]()


def _special(dtype, rng, shape, family):
    fi = np.finfo(np.float32 if dtype == "float32" else np.float64)
    big, small, denorm = float(fi.max) / 4, float(fi.tiny) * 4, float(fi.tiny) / 8
    base = rng.integers(-3, 4, shape).astype(np.float64)
    if family == "ties":
        return base
    if family == "constant":
        return np.full(shape, 1.5)
    if family == "duplicates":
        rows = rng.integers(-3, 4, (max(1, shape[0] // 4), shape[1])).astype(np.float64)
        return rows[rng.integers(0, rows.shape[0], shape[0])]
    pool = {
        "nan": [math.nan, 1.0, -2.0, 0.0],
        "inf": [math.inf, -math.inf, 1.0, -1.0, 0.0],
        "signed_zero": [0.0, -0.0, 1.0],
        "denormal": [denorm, -denorm, 2 * denorm, 0.0, -0.0],
        "huge": [big, -big, big / 2, 1.0],
        "tiny": [small, -small, 3 * small, 1.0],
        "mixed": [math.nan, math.inf, -math.inf, -0.0, 0.0, denorm, big, -small, 1.0],
    }[family]
    return rng.choice(np.array(pool), shape)


FAMILIES = ("ties", "constant", "duplicates", "nan", "inf", "signed_zero", "denormal", "huge", "tiny", "mixed")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", KINDS, ids=lambda v: v.value)
def test_special_values(backend, kind, dtype, family):
    name, dev = backend
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    if kind is R.COUNT:
        expr = (y <= x).sum_over("j")
    else:
        expr = _reduce(x - y, kind)
    for m, n in SIZES:
        rng = np.random.default_rng([m, n, FAMILIES.index(family)])
        arrays = {"x": np.zeros((m, 1)) if family not in ("ties", "duplicates", "constant") else
                  _special(dtype, rng, (m, 1), family),
                  "y": _special(dtype, rng, (n, 1), family)}
        try:
            run_case(name, dev, expr, arrays, dtype)
        except AssertionError as exc:
            raise AssertionError(f"[{name} {dtype} {family} M={m} N={n}] {exc}") from exc


@pytest.mark.parametrize("dtype", DTYPES)
def test_signed_zero_ties(backend, dtype):
    # -0.0 == 0.0 : égalité au sens du contrat -> plus petit j, quel que soit le signe
    name, dev = backend
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    ys = np.array([[1.0], [-0.0], [0.0], [-0.0], [2.0]])
    for expr in ((x * y).argmin(over="j"), (x * y).kmin_argkmin(3, over="j"), (x * y).argmax(over="j")):
        run_case(name, dev, expr, {"x": np.array([[1.0], [-1.0]]), "y": ys}, dtype)


GUARDED = {
    # la branche inactive serait NaN/inf/erreur de domaine si elle était sélectionnée
    "sqrt": lambda x, y: kp.where(x > y, kp.sqrt(x - y), -1.0),
    "div": lambda x, y: kp.where(kp.ne(y, 0.0), x / y, 0.0),
    "log": lambda x, y: kp.where(x * y > 0.0, kp.log(x * y), 0.0),
    "nested": lambda x, y: kp.where(x > 0.0, kp.where(y > 0.0, kp.log(x) + kp.log(y), kp.sqrt(x) / (y * y + 1.0)),
                                    kp.where(y < 0.0, kp.sqrt(-x) * kp.log(-y), 0.0)),
    "pow": lambda x, y: kp.where(x - y >= 0.0, (x - y) ** 0.5 + (x - y) ** 1.5, (y - x) ** -0.5),
    "logic": lambda x, y: kp.where((x > 0.0) & (y > 0.0) | (x < -1.0), kp.log(kp.abs(x) + y * y), x / (y - y + 1.0)),
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", sorted(GUARDED))
@pytest.mark.parametrize("kind", (R.SUM, R.MIN, R.ARGMAX, R.KMIN_ARGKMIN), ids=lambda v: v.value)
def test_where_excludes_invalid_domains(backend, kind, guard, dtype):
    name, dev = backend
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    expr = _reduce(GUARDED[guard](x, y), kind)
    for m, n in ((7, 129), (64, 33)):
        rng = np.random.default_rng([m, n, len(guard)])
        # entiers et zéros exacts : divisions par zéro et racines de négatifs dans la branche inactive
        arrays = {"x": rng.integers(-3, 4, (m, 1)).astype(np.float64),
                  "y": rng.integers(-3, 4, (n, 1)).astype(np.float64)}
        try:
            run_case(name, dev, expr, arrays, dtype)
        except AssertionError as exc:
            raise AssertionError(f"[{name} {dtype} where/{guard} M={m} N={n}] {exc}") from exc


@pytest.mark.parametrize("dtype", DTYPES)
def test_where_result_has_no_nan(backend, dtype):
    # aucune valeur de la branche inactive (NaN, inf) ne doit atteindre la somme
    name, dev = backend
    x, y = kp.Vi("x", 1), kp.Vj("y", 1)
    for guard in sorted(GUARDED):
        run = kp.compile(GUARDED[guard](x, y).sum_over("j"), dtype=dtype, backend=name)
        import torch

        T = {"float32": torch.float32, "float64": torch.float64}[dtype]
        xs = torch.arange(-3, 4, dtype=T, device=dev).reshape(-1, 1)
        out = run(x=xs, y=xs.clone())
        assert torch.isfinite(out).all(), (guard, out.cpu().ravel().tolist())


def test_denormal_floor_documented():
    # garde-fou : le plancher absolu de tolérance est 4 x plus petit normal (dénormaux / FTZ)
    assert TINY["float32"] == float(np.finfo(np.float32).tiny)
