"""Batch natif (avenant v0.6 §1) du backend CUDA.

Égalité au bit près avec une boucle d'appels non batchés sur chaque élément
(même configuration de lancement : partition de j imposée), diffusion à
strides 0, params batchés, partition de j (partiels (nb, M, S, ·)), top-K par
thread et coopératif, LSE/SSMW, schémas block/kahan, lancements par paquets de
la dimension z (GRID_Z_MAX abaissé), nb et N nuls, frontend batché.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.version.hip is not None:  # pragma: no cover
    pytest.skip("GPU NVIDIA requis", allow_module_level=True)

import kernox as kp  # noqa: E402
from kernox.backends.base import BatchLayout  # noqa: E402
from kernox.backends.cuda import CudaBackend  # noqa: E402
from kernox.backends.cuda import backend as cuda_backend  # noqa: E402
from kernox.plan import output_shapes  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda", 0)


def _exprs():
    x, y, b, s = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 2), kp.Param("s", 1)
    d2 = ((x - y) ** 2).sum()
    return {
        "sum": (kp.exp(-d2 * s) * b).sum_over("j"),
        "argmin": d2.argmin(over="j"),
        "kmin_argkmin5": d2.kmin_argkmin(5, over="j"),
        "kmin_argkmin40": d2.kmin_argkmin(40, over="j"),     # coop G = 32
        "lse": (-d2 * s).logsumexp(over="j"),
        "ssmw": (-d2 * s).sumsoftmaxweight(b, over="j"),
    }


def _offsets(t, B, tail):
    off = torch.zeros(B, dtype=torch.int64)
    bshape, bstride = t.shape[:-tail], t.stride()[:-tail]
    lead = len(B) - len(bshape)
    for k, (size, stride) in enumerate(zip(bshape, bstride)):
        if size != 1 and stride:
            view = [1] * len(B)
            view[lead + k] = size
            off = off + torch.arange(size, dtype=torch.int64).view(view) * stride
    return off.reshape(-1).contiguous()


def _element(t, B, b, tail):
    """Élément b (indice aplati) de t diffusé à B, contigu."""
    idx = []
    for size in reversed(B):
        idx.append(b % size)
        b //= size
    idx = idx[::-1]
    lead = len(B) - (t.dim() - tail)
    sel = tuple(0 if t.shape[k] == 1 else idx[lead + k] for k in range(t.dim() - tail))
    return t[sel].contiguous()


def _m(plan, tensors):
    return tensors[next(a.name for a in plan.args if a.role == "i")].shape[-2]


def _run_batched(be, plan, tensors, B):
    nb = math.prod(B)
    pb = dataclasses.replace(plan, batch_rank=len(B))
    cr = be.compile(pb, DEV)
    host = {a.name: _offsets(tensors[a.name], B, 1 if a.role == "param" else 2) for a in plan.args}
    layout = BatchLayout(B, nb, {k: v.to(DEV) for k, v in host.items()}, host)
    M = _m(plan, tensors)
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV)
            for k, (s, d) in output_shapes(pb, M, (nb,)).items()}
    cr.launch(tensors, outs, batch=layout)
    torch.cuda.synchronize()
    return outs, cr


def _run_loop(be, plan, tensors, B):
    nb = math.prod(B)
    cr = be.compile(plan, DEV)
    M = _m(plan, tensors)
    res = {k: [] for k in output_shapes(plan, M)}
    for b in range(nb):
        sub = {a.name: _element(tensors[a.name], B, b, 1 if a.role == "param" else 2) for a in plan.args}
        outs = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(plan, M).items()}
        cr.launch(sub, outs)
        for k in outs:
            res[k].append(outs[k])
    torch.cuda.synchronize()
    return {k: torch.stack(v) for k, v in res.items()}


def _batch_inputs(dtype, M=37, N=150, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = getattr(torch, dtype)
    x = torch.randn(3, 1, M, 3, generator=g).to(T)                     # (B1, 1, M, 3)
    y = torch.randn(1, 4, N, 3, generator=g).to(T)                     # (1, B2, N, 3)
    y[0, 1, 7] = math.nan
    b = torch.randn(4, N, 2, generator=g).to(T).expand(3, 4, N, 2)     # strides 0 en tête
    s = torch.tensor([[0.3], [0.7], [1.1]]).to(T).reshape(3, 1, 1)      # param batché (B1, 1, 1)
    return {k: v.to(DEV) for k, v in dict(x=x, y=y, b=b, s=s).items()}, (3, 4)


# (S, jchunk, tuile) imposés aux deux chemins : même partition, comparaison au bit près
CONFIGS = {"S1": None, "S4": (4, 40, 8), "S10": (10, 16, 16)}


def _force(monkeypatch, cfg):
    if cfg is None:
        monkeypatch.setattr(cuda_backend.CudaCompiledReduction, "launch_config",
                            lambda self, m, n, nb=1: (1, max(n, 1), self.tile))
    else:
        S, jchunk, T = cfg
        monkeypatch.setattr(cuda_backend.CudaCompiledReduction, "launch_config",
                            lambda self, m, n, nb=1: (-(-n // jchunk), jchunk, T))


def _same(got, ref, label):
    for k in ref:
        assert got[k].shape == ref[k].shape, (label, k)
        assert torch.equal(got[k].isnan(), ref[k].isnan()), (label, k)
        assert torch.equal(torch.nan_to_num(got[k]), torch.nan_to_num(ref[k])), (label, k)


@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", list(_exprs()))
def test_batch_bitwise_vs_loop(monkeypatch, cfg, dtype, name):
    _force(monkeypatch, CONFIGS[cfg])
    plan = _exprs()[name].to_plan(dtype)
    t, B = _batch_inputs(dtype, seed=len(cfg))
    be = CudaBackend()
    got, _ = _run_batched(be, plan, t, B)
    ref = _run_loop(be, plan, t, B)
    _same(got, ref, (name, dtype, cfg))


@pytest.mark.parametrize("scheme", ["block", "kahan"])
@pytest.mark.parametrize("name", ["sum", "lse", "ssmw"])
def test_batch_sum_schemes(monkeypatch, scheme, name):
    _force(monkeypatch, CONFIGS["S4"])
    plan = dataclasses.replace(_exprs()[name].to_plan("float32"), sum_scheme=scheme)
    t, B = _batch_inputs("float32", seed=5)
    be = CudaBackend()
    _same(_run_batched(be, plan, t, B)[0], _run_loop(be, plan, t, B), (name, scheme))


@pytest.mark.parametrize("G", [16, 32])
def test_batch_coop_k16(monkeypatch, G):
    _force(monkeypatch, CONFIGS["S10"])
    plan = _exprs()["kmin_argkmin5"].to_plan("float32")
    t, B = _batch_inputs("float32", seed=7)
    be = CudaBackend(coop_group=G)
    _same(_run_batched(be, plan, t, B)[0], _run_loop(be, plan, t, B), G)


@pytest.mark.parametrize("cfg", ["S1", "S4"])
def test_batch_grid_z_chunks(monkeypatch, cfg):
    monkeypatch.setattr(cuda_backend, "GRID_Z_MAX", 5)   # 12 éléments -> paquets de 5, 5, 2
    _force(monkeypatch, CONFIGS[cfg])
    for name in ("kmin_argkmin5", "sum"):
        plan = _exprs()[name].to_plan("float32")
        t, B = _batch_inputs("float32", seed=3)
        be = CudaBackend()
        _same(_run_batched(be, plan, t, B)[0], _run_loop(be, plan, t, B), (name, cfg))


def test_batch_auto_split_close():
    """Configuration automatique (S selon nb) : proche de la boucle, indices identiques."""
    plan = _exprs()["sum"].to_plan("float64")
    t, B = _batch_inputs("float64", M=5, N=20000, seed=11)
    be = CudaBackend()
    got, cr = _run_batched(be, plan, t, B)
    ref = _run_loop(be, plan, t, B)
    assert cr.launch_config(5, 20000, 12)[0] > 1
    torch.testing.assert_close(got["value"], ref["value"], rtol=1e-12, atol=1e-12, equal_nan=True)


def test_batch_empty_and_n0():
    plan = _exprs()["argmin"].to_plan("float32")
    t, B = _batch_inputs("float32")
    t["y"] = t["y"][:, :, :0]
    t["b"] = t["b"][:, :, :0]
    got, _ = _run_batched(CudaBackend(), plan, t, B)
    assert (got["index"] == -1).all()
    plan = _exprs()["sum"].to_plan("float32")
    got, _ = _run_batched(CudaBackend(), plan, t, B)
    assert (got["value"] == 0).all()
    # nb = 0 : aucun lancement, sorties intactes
    t, _ = _batch_inputs("float32")
    t = {k: v[:0] if k != "y" else v for k, v in t.items()}
    got, _ = _run_batched(CudaBackend(), _exprs()["sum"].to_plan("float32"), t, (0, 4))
    assert got["value"].numel() == 0


def test_batch_rejects_bad_layout():
    plan = _exprs()["sum"].to_plan("float32")
    t, B = _batch_inputs("float32")
    pb = dataclasses.replace(plan, batch_rank=2)
    cr = CudaBackend().compile(pb, DEV)
    host = {a.name: _offsets(t[a.name], B, 1 if a.role == "param" else 2) for a in plan.args}
    outs = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV)
            for k, (s, d) in output_shapes(pb, 37, (12,)).items()}
    bad = dict(host)
    bad["y"] = host["y"][:5]
    with pytest.raises(ValueError):
        cr.launch(t, outs, batch=BatchLayout(B, 12, {k: v.to(DEV) for k, v in bad.items()}, bad))
    t2 = dict(t)
    t2["x"] = t["x"].transpose(-1, -2).contiguous().transpose(-1, -2)   # lignes non contiguës
    lay = BatchLayout(B, 12, {k: v.to(DEV) for k, v in host.items()}, host)
    with pytest.raises(ValueError):
        cr.launch(t2, outs, batch=lay)
    with pytest.raises(Exception):
        CudaBackend().compile(plan, DEV).launch(t, outs, batch=lay)


def test_batch_frontend():
    """Chemin runtime (Reduction batchée sur le backend cuda) contre la boucle."""
    x, y = kp.Vi("x", 3), kp.Vj("y", 3)
    red = ((x - y) ** 2).sum().kmin_argkmin(3, over="j")
    run = kp.compile(red, dtype="float32", backend="cuda")
    g = torch.Generator().manual_seed(1)
    X = torch.randn(2, 3, 50, 3, generator=g).to(DEV)
    Y = torch.randn(1, 3, 400, 3, generator=g).to(DEV)
    v, i = run(x=X, y=Y)
    for a in range(2):
        for b in range(3):
            v1, i1 = run(x=X[a, b].contiguous(), y=Y[0, b].contiguous())
            assert torch.equal(i[a, b], i1) and torch.equal(v[a, b], v1)
