"""Schémas de sommation (avenant v0.7) sur le backend hip : block et kahan.

Complète ``tests/conformance/test_sum_scheme.py`` (données de Higham sans
partition) : partition de j forcée (tranches vides comprises), petites tuiles,
lignes spéciales (±inf, NaN) dont le motif doit être exactement celui de la
référence (``kahan`` rend ``s`` si ``s`` n'est pas fini), acc FP64,
fp_contract off, publication ``sum_block=T`` et niveaux imposés pour LSE/SSMW.
"""

import math

import numpy as np
import pytest
import torch

import kernox as kp
from kernox.backends.hip import HipBackend, HipOptions
from kernox.oracle import OracleReduction
from kernox.plan import output_shapes

from .test_softmax import _close, _exprs, _inputs, _ref, _run

pytestmark = pytest.mark.gpu
DEV = "cuda"
U32 = 2.0 ** -24
SPLITS = [dict(split="off"), dict(split="force", tile_j=8, jchunk=24), dict(split="force", tile_j=32, jchunk=40)]


def _plan(name, dtype, scheme, **kw):
    if name == "sum":
        x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", 3)
        expr = ((x - y) * w).sum_over("j")
    else:
        expr = _exprs()[name]
    return expr.to_plan(dtype, sum_scheme=scheme, **kw)


@pytest.mark.parametrize("opts", SPLITS, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("scheme", ["block", "kahan"])
@pytest.mark.parametrize("name", ["sum", "lse_vec", "ssmw"])
def test_schemes_split_special_rows(opts, dtype, scheme, name):
    plan = _plan(name, dtype, scheme)
    host = _inputs(37, 150, seed=len(opts))
    got, cr = _run(HipBackend(HipOptions(**opts)), plan, host)
    if opts.get("split") == "force":
        assert cr.last_launch["S"] > 1
    assert f"sum_scheme={scheme}" in cr.info.options
    if scheme == "block":
        assert f"sum_block={cr.config.tile_j}" in cr.info.options
    if name != "sum":
        assert cr.config.levels == (2 if scheme == "block" else 1)
    ref = _ref(plan, host)
    if name == "sum":
        assert torch.equal(torch.isnan(got), torch.isnan(ref))
        fin = torch.isfinite(ref)
        assert torch.equal(got[~fin & ~torch.isnan(ref)], ref[~fin & ~torch.isnan(ref)])
        u = U32 if dtype == "float32" else 2.0 ** -53
        mag = (host["x"].abs().max() + host["y"][torch.isfinite(host["y"])].abs().max()) * host["w"].abs().sum(0).max()
        assert torch.all((got[fin].double() - ref[fin]).abs() <= 8 * 150 * u * float(mag))
    else:
        _close(got, ref, dtype, 150)


def _higham(n, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.random(n).astype(np.float32)
    big = np.zeros(n, np.float32)
    big[0::4], big[2::4] = 1e8, -1e8
    return np.where(big != 0, big, w).astype(np.float32)


@pytest.mark.parametrize("opts", [dict(split="force", tile_j=8, jchunk=4001), dict(split="force", tile_j=32, jchunk=3333),
                                  dict(split="force", tile_j=64, jchunk=10000), dict(split="off")],
                         ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))
@pytest.mark.parametrize("fp_contract", [True, False])
def test_kahan_higham_with_split(opts, fp_contract):
    n = 200_003
    W = _higham(n, seed=4)
    ref = math.fsum(W.astype(np.float64))
    l1 = math.fsum(np.abs(W.astype(np.float64)))
    red = (kp.Vj("w", 1) * kp.Vi("x", 1)).sum_over("j")
    errs = {}
    for scheme in ("direct", "block", "kahan"):
        plan = red.to_plan("float32", sum_scheme=scheme, fp_contract=fp_contract)
        got, cr = _run(HipBackend(HipOptions(**opts)), plan,
                       {"x": torch.ones(4, 1, dtype=torch.float64), "w": torch.from_numpy(W).double()[:, None]})
        errs[scheme] = abs(got[0, 0].item() - ref)
        if scheme == "kahan":
            assert errs[scheme] <= 4 * U32 * l1, errs
        if scheme == "block":
            T = cr.config.tile_j
            assert errs[scheme] <= (n / T + T + 2) * U32 * l1, errs
    assert errs["kahan"] * 10 <= errs["direct"], errs


def test_ssmw_lse_kahan_higham_split():
    n = 100_000
    W = _higham(n, seed=5)
    one, f, g = kp.Vi("x", 1), kp.Vj("f", 1), kp.Vj("g", 1)
    host = {"x": torch.ones(2, 1, dtype=torch.float64), "f": torch.zeros(n, 1, dtype=torch.float64),
            "g": torch.from_numpy(W).double()[:, None]}
    ref = math.fsum(W.astype(np.float64)) / n
    l1 = math.fsum(np.abs(W.astype(np.float64))) / n
    errs = {}
    for scheme in ("direct", "kahan"):
        plan = (f * one).sumsoftmaxweight(g * one, over="j").to_plan("float32", sum_scheme=scheme)
        got, _ = _run(HipBackend(HipOptions(split="force", tile_j=16, jchunk=4000)), plan, host)
        errs[scheme] = abs(got[0, 0].item() - ref)
    assert errs["kahan"] <= 4 * U32 * l1 + 2 * U32 * abs(ref), errs
    assert errs["kahan"] * 10 <= errs["direct"], errs


def test_levels_imposed_by_scheme():
    plan = _plan("ssmw", "float32", "kahan")
    with pytest.raises(Exception):
        HipBackend(HipOptions(soft_levels=2)).compile(plan, DEV)
    plan = _plan("ssmw", "float64", "block")
    with pytest.raises(Exception):
        HipBackend(HipOptions(soft_levels=1)).compile(plan, DEV)


@pytest.mark.parametrize("scheme", ["block", "kahan"])
def test_scheme_acc64(scheme):
    plan = _plan("sum", "float32", scheme, acc_dtype="float64")
    host = _inputs(40, 3000, seed=6)
    got, cr = _run(HipBackend(HipOptions(split="force", tile_j=16)), plan, host)
    assert got.dtype == torch.float64
    # référence : oracle du même plan (termes FP32 identiques, accumulation FP64 selon le schéma)
    ref = torch.empty(output_shapes(plan, 40)["value"][0], dtype=torch.float64)
    OracleReduction(plan, "cpu", 4096).launch({a.name: host[a.name].float() for a in plan.args}, {"value": ref})
    fin = torch.isfinite(ref)
    assert torch.equal(torch.isnan(got), torch.isnan(ref))
    assert torch.equal(got[~fin], ref[~fin].to(got.dtype)) or torch.isnan(ref[~fin]).all()
    assert torch.all((got[fin] - ref[fin]).abs() <= 1e-12 * (1 + ref[fin].abs()) * 3000)
