"""Dimensions de batch natives du backend sycl (avenant v0.6 §1), G31.

Chaque élément de batch doit être identique au bit près à l'appel v0 sur
l'élément (copie contiguë), quelle que soit la variante : un work-item par
ligne, top-K coopératif, partition de j forcée (temporaires (NB, M, S, W)),
diffusion par strides nuls, param batché, ``nb`` premier.
"""

import dataclasses
import itertools

import pytest
import torch

import kernox as kp
from kernox import Param, Vi, Vj
from kernox.backends.base import BatchLayout
from kernox.backends.sycl import _variant
from kernox.plan import output_shapes

from ._helpers import DEV, backend

B = (3, 1, 5)  # nb = 15 après diffusion (3, 2, 5)
BF = (3, 2, 5)


def _formulas():
    x, y, w, s = Vi("x", 3), Vj("y", 3), Vj("w", 2), Param("s", 1)
    d2 = ((x - y) ** 2).sum() * s[0]
    return {
        "sum": (kp.exp(-d2) * w).sum_over("j"),
        "lse": (-d2).logsumexp(over="j"),
        "ssmw": (-d2).sumsoftmaxweight(w, over="j"),
        "count": (d2 < 2.0).sum_over("j"),
        "argmin": d2.argmin(over="j"),
        "kmin_argkmin": d2.kmin_argkmin(40, over="j"),
    }


def _inputs(dtype, M, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = getattr(torch, dtype)

    def r(*shape):
        return torch.randint(-3, 4, shape, generator=g).to(T).to(DEV)

    x = r(3, 1, 1, M, 3).expand(3, 2, 1, M, 3)          # stride 0 sur l'axe 2 (avant diffusion)
    y = r(1, 2, 5, N + 3, 3)[..., 3:, :]                 # tranche : offset de stockage non nul
    w = r(3, 2, 5, N, 2)
    s = torch.tensor([0.7], dtype=T, device=DEV)         # param sans batch
    return {"x": x, "y": y, "w": w, "s": s}


def _layout(plan, t):
    """BatchLayout du runtime (même calcul que Reduction._layout)."""
    host, dev = {}, {}
    nb = 1
    for b in BF:
        nb *= b
    for a in plan.args:
        v = t[a.name]
        tail = 1 if a.role == "param" else 2
        bshape, bstride = tuple(v.shape[:-tail]), v.stride()[:-tail]
        off = torch.zeros(BF, dtype=torch.int64)
        lead = len(BF) - len(bshape)
        for k, (size, stride) in enumerate(zip(bshape, bstride)):
            if size != 1 and stride:
                view = [1] * len(BF)
                view[lead + k] = size
                off = off + torch.arange(size, dtype=torch.int64).view(view) * stride
        host[a.name] = off.reshape(nb).contiguous()
        dev[a.name] = host[a.name].to(DEV)
    return BatchLayout(BF, nb, dev, host)


def _launch(plan, t, M, variant=None, groups=None, batch=None):
    nb = batch.nb if batch is not None else 1
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV)
            for k, (s, d) in output_shapes(plan, M, (nb,) if batch is not None else ()).items()}
    cr = backend().compile(plan, DEV, variant)
    if groups is not None:
        cr.target_groups = groups
    cr.launch(t, outs, batch=batch)
    torch.xpu.synchronize()
    return {k: v.cpu() for k, v in outs.items()}, cr


VARIANTS = [(None, None), (_variant(tj=8), 4096), (_variant(tj=0), 1)]


@pytest.mark.parametrize("vi", range(len(VARIANTS)))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", list(_formulas()))
def test_batched_equals_loop(name, dtype, vi):
    variant, groups = VARIANTS[vi]
    red = _formulas()[name]
    M, N = 37, 700
    t = _inputs(dtype, M, N, seed=vi)
    p0 = red.to_plan(dtype)
    pb = dataclasses.replace(p0, batch_rank=3)
    lay = _layout(pb, t)
    got, cr = _launch(pb, t, M, variant, groups, lay)
    if groups and groups > 1:
        assert cr.last_split[0] > 1
    for idx, b in enumerate(itertools.product(*[range(k) for k in BF])):
        elem = {}
        for a in p0.args:
            v = t[a.name]
            if a.role == "param":
                elem[a.name] = v
            else:
                v = v.expand(*BF, *v.shape[-2:])
                elem[a.name] = v[b].contiguous()
        ref, rc = _launch(p0, elem, M, variant, groups)
        for k in ref:
            g_, r_ = got[k][idx].nan_to_num(123.0), ref[k].nan_to_num(123.0)
            if name in ("sum", "lse", "ssmw") and rc.last_split[0] != cr.last_split[0]:
                # partition de j propre au batch (split_j(..., nb)) : ordre de sommation différent
                torch.testing.assert_close(g_, r_, rtol=1e-5 if dtype == "float32" else 1e-13, atol=1e-6)
            else:
                assert torch.equal(g_, r_), (name, b, k)


@pytest.mark.parametrize("G", [4, 8])
def test_batched_coop(G):
    red = _formulas()["kmin_argkmin"]
    M, N = 37, 5000
    t = _inputs("float32", M, N, seed=G)
    p0 = red.to_plan("float32")
    pb = dataclasses.replace(p0, batch_rank=3)
    lay = _layout(pb, t)
    for groups in (1, 4096):
        v = _variant(sg=16, coop=G)
        got, cr = _launch(pb, t, M, v, groups, lay)
        assert (cr.last_split[0] > 1) is (groups > 1)
        for idx, b in enumerate(itertools.product(*[range(k) for k in BF])):
            elem = {a.name: (t[a.name] if a.role == "param" else
                             t[a.name].expand(*BF, *t[a.name].shape[-2:])[b].contiguous()) for a in p0.args}
            ref, _ = _launch(p0, elem, M, v, groups)
            for k in ref:
                assert torch.equal(got[k][idx], ref[k]), (b, k)


def test_runtime_batched_matches_oracle():
    """Par le runtime (BatchLayout réel, param batché, nb premier) contre l'oracle."""
    x, y, w, s = Vi("x", 2), Vj("y", 2), Vj("w", 1), Param("s", 1)
    red = (kp.exp(-((x - y) ** 2).sum() * s[0]) * w).sum_over("j")
    g = torch.Generator().manual_seed(3)
    X = torch.randn(7, 1, 20, 2, generator=g, dtype=torch.float64)
    Y = torch.randn(1, 1, 300, 2, generator=g, dtype=torch.float64)
    W = torch.randn(7, 1, 300, 1, generator=g, dtype=torch.float64)
    S = torch.rand(7, 1, 1, generator=g, dtype=torch.float64) + 0.5
    ref = kp.compile(red, dtype="float64", backend="oracle")(x=X, y=Y, w=W, s=S)
    got = kp.compile(red, dtype="float64", backend="sycl")(x=X.to(DEV), y=Y.to(DEV), w=W.to(DEV), s=S.to(DEV))
    assert got.shape == ref.shape == (7, 1, 20, 1)
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-12, atol=1e-12)
    empty = kp.compile(red, dtype="float64", backend="sycl")(
        x=X[:0].to(DEV), y=Y.to(DEV), w=W[:0].to(DEV), s=S[:0].to(DEV))
    assert empty.shape == (0, 1, 20, 1)


def test_check_rejects_bad_layout():
    red = _formulas()["sum"]
    M, N = 5, 20
    t = _inputs("float32", M, N)
    p0 = red.to_plan("float32")
    pb = dataclasses.replace(p0, batch_rank=3)
    lay = _layout(pb, t)
    with pytest.raises(ValueError, match="BatchLayout"):
        _launch(pb, t, M)
    bad = dict(lay.host_offsets)
    bad["w"] = bad["w"] + 10**9
    with pytest.raises(ValueError, match="hors du stockage"):
        _launch(pb, t, M, batch=BatchLayout(lay.shape, lay.nb, lay.offsets, bad))
