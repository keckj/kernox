"""MIN_ARGMIN, MAX_ARGMAX, LOGSUMEXPWEIGHT (avenant v0.8) natifs sur le backend hip.

Complète ``tests/conformance/test_reductions_v08.py`` : partition de j forcée
(partiels valeur + indice, tranches entièrement NaN), égalités nombreuses
(données entières), NaN, N = 0 ; LSEW sous tous les schémas de sommation et
niveaux d'insertion, lignes spéciales (±1e4, ±inf, NaN), poids positifs,
négatifs (NaN) et nuls (−inf). Batch et ranges : ``test_batch_ranges.py``.
"""

import math

import pytest
import torch

import kernox as kp
from kernox.backends.hip import HipBackend, HipOptions
from kernox.oracle import OracleReduction
from kernox.plan import output_shapes

from .test_softmax import SPLIT_OPTS, _close, _inputs

pytestmark = pytest.mark.gpu
DEV = "cuda"
DTYPES = ["float32", "float64"]
INF = math.inf
ARG_SPLITS = [dict(split="off"), dict(split="force", tile_j=8), dict(split="force", tile_j=8, jchunk=24),
              dict(split="force", tile_j=32, jchunk=40), dict(split="auto")]


def _launch(be, plan, inp, M):
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(plan, M).items()}
    cr = be.compile(plan, DEV)
    cr.launch(inp, outs)
    torch.cuda.synchronize()
    return {k: v.cpu() for k, v in outs.items()}, cr


def _oracle(plan, inp, M):
    outs = {k: torch.empty(s, dtype=getattr(torch, d)) for k, (s, d) in output_shapes(plan, M).items()}
    OracleReduction(plan, "cpu", 4096).launch({k: v.cpu() for k, v in inp.items()}, outs)
    return outs


def _arg_data(M, N, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    T = getattr(torch, dtype)
    x = torch.randint(-3, 4, (M, 2), generator=g).to(T)
    y = torch.randint(-3, 4, (N, 2), generator=g).to(T)
    if N > 10:
        y[3, 0] = math.nan
        y[N // 2:N // 2 + 40, 1] = math.nan          # tranches où la colonne 1 est NaN
    if M > 4:
        x[2, 0] = math.nan                            # ligne sans candidat : neutre
        x[4, 1] = INF
    return {"x": x.to(DEV), "y": y.to(DEV)}


@pytest.mark.parametrize("opts", ARG_SPLITS, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_min", [True, False], ids=["min_argmin", "max_argmax"])
def test_min_argmin_split_ties_nan(opts, dtype, is_min):
    x, y = kp.Vi("x", 2), kp.Vj("y", 2)
    f = x * y
    both = (f.min_argmin if is_min else f.max_argmax)(over="j").to_plan(dtype)
    val = (f.min if is_min else f.max)(over="j").to_plan(dtype)
    idx = (f.argmin if is_min else f.argmax)(over="j").to_plan(dtype)
    M, N = 53, 700
    inp = _arg_data(M, N, dtype, seed=len(opts))
    be = HipBackend(HipOptions(**opts))
    got, cr = _launch(be, both, inp, M)
    if opts.get("split") == "force":
        assert cr.last_launch["S"] > 1
    gv, _ = _launch(be, val, inp, M)
    gi, _ = _launch(be, idx, inp, M)
    ref = _oracle(both, inp, M)
    assert got["value"].dtype == getattr(torch, dtype) and got["index"].dtype == torch.int64
    assert torch.equal(got["index"], gi["index"]) and torch.equal(got["index"], ref["index"])
    assert torch.equal(got["value"].nan_to_num(), gv["value"].nan_to_num())
    assert torch.equal(got["value"].nan_to_num(), ref["value"].nan_to_num())
    # valeur = f(i, index) ; ligne NaN : neutre (±inf, −1)
    assert (got["index"][2, 0] == -1) and got["value"][2, 0] == (INF if is_min else -INF)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_min", [True, False], ids=["min_argmin", "max_argmax"])
def test_min_argmin_empty(dtype, is_min):
    x, y = kp.Vi("x", 2), kp.Vj("y", 2)
    plan = ((x * y).min_argmin if is_min else (x * y).max_argmax)(over="j").to_plan(dtype)
    inp = _arg_data(6, 0, dtype, seed=0)
    for opts in (dict(split="off"), dict(split="force", tile_j=8)):
        got, _ = _launch(HipBackend(HipOptions(**opts)), plan, inp, 6)
        assert (got["index"] == -1).all() and (got["value"] == (INF if is_min else -INF)).all()


def _lsew_plan(dtype, scheme="direct"):
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", 3)
    return (x - y).logsumexp(over="j", weight=w).to_plan(dtype, sum_scheme=scheme)


def _lsew_inputs(M, N, seed):
    h = _inputs(M, N, seed)
    w = h["w"].abs() + 0.1
    w[:, 1] = -w[:, 1]          # t < 0 : NaN
    w[:, 2] = 0.0               # t = 0 : −inf
    h["w"] = w
    return h


def _lsew_ref(plan, host):
    p64 = kp.builder.make_plan(plan.outputs, plan.reduction, "float64", args=plan.args)
    M = host["x"].shape[0]
    cast = {k: host[k].to(getattr(torch, plan.dtype)).double() for k in ("x", "y", "w")}
    return _oracle(p64, cast, M)["value"]


@pytest.mark.parametrize("opts", SPLIT_OPTS, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))
@pytest.mark.parametrize("dtype", DTYPES)
def test_lsew_split_special_rows(opts, dtype):
    plan = _lsew_plan(dtype)
    host = _lsew_inputs(37, 150, seed=len(opts))
    T = getattr(torch, dtype)
    got, cr = _launch(HipBackend(HipOptions(**opts)), plan, {k: v.to(T).to(DEV) for k, v in host.items()}, 37)
    got = got["value"]
    if opts.get("split") == "force":
        assert cr.last_launch["S"] > 1
    ref = _lsew_ref(plan, host)
    _close(got, ref, dtype, 150)
    assert torch.isnan(got[:, 1][torch.isfinite(host["x"][:, 0])]).all()
    assert (got[0, 2] == -INF) and torch.isnan(got[5]).all() and (got[4] == -INF).all()


@pytest.mark.parametrize("opts", [dict(split="off"), dict(split="force", tile_j=8, jchunk=24)],
                         ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scheme", ["block", "kahan"])
def test_lsew_schemes(opts, dtype, scheme):
    plan = _lsew_plan(dtype, scheme)
    host = _lsew_inputs(37, 150, seed=3)
    T = getattr(torch, dtype)
    got, cr = _launch(HipBackend(HipOptions(**opts)), plan, {k: v.to(T).to(DEV) for k, v in host.items()}, 37)
    assert f"sum_scheme={scheme}" in cr.info.options
    assert cr.config.levels == (2 if scheme == "block" else 1)
    _close(got["value"], _lsew_ref(plan, host), dtype, 150)


@pytest.mark.parametrize("dtype", DTYPES)
def test_lsew_pos_inf_and_empty(dtype):
    """f = +inf : +inf si Σ g > 0, NaN si < 0 ou = 0 ; N = 0 : −inf."""
    plan = _lsew_plan(dtype)
    T = getattr(torch, dtype)
    for N in (0, 1, 300):
        host = _lsew_inputs(9, max(N, 1), seed=N)
        host["y"] = torch.linspace(-2, 2, N, dtype=torch.float64).reshape(N, 1)
        host["w"] = host["w"][:N]
        for opts in (dict(split="off"), dict(split="force", tile_j=8), dict(split="force", tile_j=8, soft_levels=1)):
            got, _ = _launch(HipBackend(HipOptions(**opts)), plan, {k: v.to(T).to(DEV) for k, v in host.items()}, 9)
            got = got["value"]
            _close(got, _lsew_ref(plan, host), dtype, max(N, 1))
            if N == 0:
                assert (got == -INF).all()
            else:
                assert got[3, 0] == INF and torch.isnan(got[3, 1:]).all()


def test_lsew_acc_fp64():
    x, y, w = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("w", 2)
    plan = (x - y).logsumexp(over="j", weight=w).to_plan("float32", acc_dtype="float64")
    g = torch.Generator().manual_seed(0)
    host = {"x": torch.randn(64, 1, generator=g), "y": torch.randn(5000, 1, generator=g),
            "w": torch.rand(5000, 2, generator=g) + 0.1}
    got, _ = _launch(HipBackend(), plan, {k: v.to(DEV) for k, v in host.items()}, 64)
    assert got["value"].dtype == torch.float64
    ref = _oracle(plan, host, 64)["value"]
    assert torch.allclose(got["value"], ref, rtol=1e-12, atol=1e-12)
