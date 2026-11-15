"""Ranges block-sparse natifs (avenant v0.6 §2) du backend CUDA.

Un bloc par tuile de travail ; intervalles dans le désordre et égalités au plus
petit j de J(i) (liste par thread, ARGMIN, top-K coopératif), lignes non
couvertes au neutre, blocs i plus hauts que ``ranges_tile_m``, NaN ignorés,
schémas block/kahan ; référence : oracle masqué (plan ranges) sur l'hôte.
Ranges denses : égalité au bit près avec le lancement sans ranges (S = 1).
"""

from __future__ import annotations

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.version.hip is not None:  # pragma: no cover
    pytest.skip("GPU NVIDIA requis", allow_module_level=True)

import kernox as kp  # noqa: E402
from kernox.backends.base import RangesLayout, Variant  # noqa: E402
from kernox.backends.cuda import CudaBackend  # noqa: E402
from kernox.errors import UnsupportedError  # noqa: E402
from kernox.oracle import OracleReduction  # noqa: E402
from kernox.plan import output_shapes  # noqa: E402
from kernox.ranges import Triplet, work_list  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda", 0)


def _exprs():
    x, y, b, s = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 2), kp.Param("s", 1)
    d2 = ((x - y) ** 2).sum()
    return {
        "sum": (kp.exp(-d2 * s) * b).sum_over("j"),
        "count": ((d2 * s < 2.0)).sum_over("j"),
        "min": (d2 * s).min(over="j"),
        "argmin": d2.argmin(over="j"),
        "argmax": d2.argmax(over="j"),
        "kmin_argkmin5": d2.kmin_argkmin(5, over="j"),
        "kmin_argkmin40": d2.kmin_argkmin(40, over="j"),     # coop G = 32
        "lse": (-d2 * s).logsumexp(over="j"),
        "ssmw": (-d2 * s).sumsoftmaxweight(b, over="j"),
    }


EXACT = ("count", "min", "argmin", "argmax", "kmin_argkmin5", "kmin_argkmin40")


def _ranges_setup(dtype, m=300, n=700, density=0.5, seed=0, shuffle=False):
    g = torch.Generator().manual_seed(seed)
    li = torch.sort(torch.randint(0, 4, (m,), generator=g)).values
    lj = torch.sort(torch.randint(0, 9, (n,), generator=g)).values
    keep = torch.rand(4, 9, generator=g) < density
    keep[3] = False                         # bloc i sans intervalle (lignes au neutre)
    r = kp.ranges_from_labels(li.to(DEV), lj.to(DEV), keep.to(DEV))
    tr = Triplet(r.ranges_i, r.slices_i, r.redranges_j)
    if shuffle:  # intervalles de chaque bloc dans l'ordre inverse
        red = r.redranges_j.clone()
        lo = 0
        for hi in r.slices_i.tolist():
            red[lo:hi] = red[lo:hi].flip(0)
            lo = hi
        tr = Triplet(r.ranges_i, r.slices_i, red)
    X = torch.randint(-2, 3, (m, 3), generator=g).double()
    Y = torch.randint(-2, 3, (n, 3), generator=g).double()          # nombreuses égalités de d²
    Y[11, 0] = math.nan
    Bv = torch.randn(n, 2, generator=g, dtype=torch.float64)
    T = getattr(torch, dtype)
    return tr, {"x": X.to(DEV, T), "y": Y.to(DEV, T), "b": Bv.to(DEV, T),
                "s": torch.tensor([0.4], dtype=T, device=DEV)}


def _layout(tr, tile_m):
    work = work_list(tr, tile_m)
    red = tr.red.to(torch.int64).contiguous()
    return RangesLayout(work, red, tile_m, work.cpu(), red.cpu())


def _oracle_ranged(plan, t, tr, tile_m):
    pr = dataclasses.replace(plan, ranges=True)
    M = t["x"].shape[0]
    host = {k: v.cpu() for k, v in t.items()}
    lay = _layout(tr, tile_m)
    lay = RangesLayout(lay.host_work, lay.host_redranges, tile_m, lay.host_work, lay.host_redranges)
    outs = {k: torch.empty(s, dtype=getattr(torch, d)) for k, (s, d) in output_shapes(pr, M).items()}
    OracleReduction(pr, "cpu", 1 << 16).launch(host, outs, ranges=lay)
    return outs


def _compiled(be, plan, block=None):
    cr = be.compile(dataclasses.replace(plan, ranges=True), DEV)
    if block is not None:
        p = dict(cr.variant.params)
        cr = cr.with_variant(Variant("tiled", (("block", block), ("split", p["split"]), ("tile", min(p["tile"], 2 * block)))))
    return cr


def _check(name, dtype, outs, ref):
    for k in ref:
        g_, r_ = outs[k].cpu(), ref[k]
        assert torch.equal(g_.isnan(), r_.isnan()), (name, k)
        if k == "index" or g_.dtype == torch.int64 or name in EXACT:
            assert torch.equal(torch.nan_to_num(g_), torch.nan_to_num(r_)), (name, k)
        else:
            u = 2.0 ** -24 if dtype == "float32" else 2.0 ** -53
            fin = torch.isfinite(r_)
            assert torch.equal(g_[~fin & ~r_.isnan()], r_[~fin & ~r_.isnan()]), (name, k)
            err = (g_[fin].double() - r_[fin].double()).abs()
            assert torch.all(err <= 64 * 700 * u * (1 + r_[fin].abs().double())), (name, k, err.max())


def _seed(name):
    return sum(map(ord, name)) % 100


@pytest.mark.parametrize("block", [None, 64])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", list(_exprs()))
def test_ranges_vs_oracle(name, dtype, shuffle, block):
    plan = _exprs()[name].to_plan(dtype)
    tr, t = _ranges_setup(dtype, shuffle=shuffle, seed=_seed(name))
    cr = _compiled(CudaBackend(), plan, block)
    tile_m = cr.ranges_tile_m
    if block is not None:
        assert tile_m == block // cr.lanes
    M = t["x"].shape[0]
    outs = {k: torch.full(s, 7, dtype=getattr(torch, d), device=DEV)
            for k, (s, d) in output_shapes(cr.plan, M).items()}
    cr.launch(t, outs, ranges=_layout(tr, tile_m))
    torch.cuda.synchronize()
    _check(name, dtype, outs, _oracle_ranged(plan, t, tr, tile_m))


@pytest.mark.parametrize("G", [16, 32])
@pytest.mark.parametrize("shuffle", [False, True])
def test_ranges_coop_forced(G, shuffle):
    plan = _exprs()["kmin_argkmin5"].to_plan("float32")
    tr, t = _ranges_setup("float32", shuffle=shuffle, seed=3)
    cr = _compiled(CudaBackend(coop_group=G), plan)
    assert cr.lanes == G
    outs = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV)
            for k, (s, d) in output_shapes(cr.plan, 300).items()}
    cr.launch(t, outs, ranges=_layout(tr, cr.ranges_tile_m))
    torch.cuda.synchronize()
    _check("kmin_argkmin5", "float32", outs, _oracle_ranged(plan, t, tr, cr.ranges_tile_m))


@pytest.mark.parametrize("scheme", ["block", "kahan"])
@pytest.mark.parametrize("name", ["sum", "lse", "ssmw"])
def test_ranges_sum_schemes(scheme, name):
    plan = dataclasses.replace(_exprs()[name].to_plan("float32"), sum_scheme=scheme)
    tr, t = _ranges_setup("float32", shuffle=True, seed=9)
    cr = _compiled(CudaBackend(), plan)
    outs = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV)
            for k, (s, d) in output_shapes(cr.plan, 300).items()}
    cr.launch(t, outs, ranges=_layout(tr, cr.ranges_tile_m))
    torch.cuda.synchronize()
    _check(name, "float32", outs, _oracle_ranged(plan, t, tr, cr.ranges_tile_m))


@pytest.mark.parametrize("name", ["sum", "argmin", "kmin_argkmin5", "kmin_argkmin40", "ssmw"])
def test_dense_ranges_bitwise(monkeypatch, name):
    """Ranges denses découpés en tuiles : même ordre de sommation que le lancement S = 1."""
    from kernox.backends.cuda import backend as cb

    monkeypatch.setattr(cb.CudaCompiledReduction, "launch_config", lambda self, m, n, nb=1: (1, max(n, 1), self.tile))
    plan = _exprs()[name].to_plan("float32")
    _tr, t = _ranges_setup("float32")
    M, N = t["x"].shape[0], t["y"].shape[0]
    be = CudaBackend()
    plain = be.compile(plan, DEV)
    o1 = {k: torch.empty(s, dtype=getattr(torch, d), device=DEV) for k, (s, d) in output_shapes(plan, M).items()}
    plain.launch(t, o1)
    r = kp.Ranges.dense(M, N, DEV)
    tr = Triplet(r.ranges_i, r.slices_i, r.redranges_j)
    cr = _compiled(be, plan)
    o2 = {k: torch.empty_like(v) for k, v in o1.items()}
    cr.launch(t, o2, ranges=_layout(tr, cr.ranges_tile_m))
    torch.cuda.synchronize()
    for k in o1:
        assert torch.equal(torch.nan_to_num(o1[k]), torch.nan_to_num(o2[k])), k


def test_ranges_refusals():
    plan = _exprs()["sum"].to_plan("float32")
    assert not CudaBackend().supports(dataclasses.replace(plan, ranges=True, batch_rank=1), DEV).ok
    assert CudaBackend().supports(dataclasses.replace(plan, ranges=True), DEV).ok
    tr, t = _ranges_setup("float32")
    cr = _compiled(CudaBackend(), plan)
    outs = {"value": torch.empty(300, 2, device=DEV)}
    with pytest.raises(ValueError):   # tile_m différent de ranges_tile_m
        cr.launch(t, outs, ranges=_layout(tr, cr.ranges_tile_m + 1))
    with pytest.raises(UnsupportedError):   # plan sans ranges
        CudaBackend().compile(plan, DEV).launch(t, outs, ranges=_layout(tr, cr.ranges_tile_m))


def test_ranges_frontend():
    """Chemin runtime : run.with_ranges contre l'oracle masqué dense."""
    x, y = kp.Vi("x", 3), kp.Vj("y", 3)
    run = kp.compile(((x - y) ** 2).sum().argmin(over="j"), dtype="float32", backend="cuda")
    g = torch.Generator().manual_seed(4)
    m, n = 500, 900
    li = torch.sort(torch.randint(0, 5, (m,), generator=g)).values.to(DEV)
    lj = torch.sort(torch.randint(0, 7, (n,), generator=g)).values.to(DEV)
    keep = (torch.rand(5, 7, generator=g) < 0.4).to(DEV)
    X = torch.randn(m, 3, generator=g).to(DEV)
    Y = torch.randn(n, 3, generator=g).to(DEV)
    got = run.with_ranges(kp.ranges_from_labels(li, lj, keep))(x=X, y=Y).reshape(-1)
    mask = keep[li][:, lj]
    d2 = ((X[:, None, :] - Y[None]) ** 2).sum(-1).masked_fill(~mask, math.inf)
    ref = torch.where(mask.any(1), d2.argmin(1), torch.full_like(got, -1))
    assert torch.equal(got, ref)


@pytest.mark.parametrize("name", ["argmin", "kmin_argkmin5", "min"])
def test_ranges_mono_variant(name):
    """Intervalles triés et disjoints : noyau sans départage (kp_reduce_ranges_mono) ;
    dans le désordre : noyau à départage par j. Seuls ARG*/K* non coopératifs en ont deux."""
    plan = _exprs()[name].to_plan("float32")
    cr = _compiled(CudaBackend(), plan)
    has = name != "min"
    assert (cr._franges_mono is not None) == has
    for shuffle, want in ((False, True), (True, False)):
        tr, _ = _ranges_setup("float32", shuffle=shuffle, seed=_seed(name))
        lay = _layout(tr, cr.ranges_tile_m)
        assert cr._j_sorted(lay) is want
        assert cr._j_sorted(lay) is want   # cache par identité


@pytest.mark.parametrize("ci,cj,density", [(3, 4, 0.5), (300, 250, 0.2), (40, 40, 1.0), (5, 7, 0.0)])
def test_ranges_from_labels_on_device_matches_host(ci, cj, density):
    """Hors XPU, ranges_from_labels reste sur le device : mêmes six tenseurs que le chemin hôte."""
    g = torch.Generator().manual_seed(ci * 1000 + cj)
    li = torch.sort(torch.randint(0, ci, (5 * ci,), generator=g)).values
    lj = torch.sort(torch.randint(0, cj, (4 * cj,), generator=g)).values
    keep = torch.rand(ci, cj, generator=g) < density
    ref = kp.ranges_from_labels(li, lj, keep)
    got = kp.ranges_from_labels(li.to(DEV), lj.to(DEV), keep.to(DEV))
    for name in ("ranges_i", "slices_i", "redranges_j", "ranges_j", "slices_j", "redranges_i"):
        t = getattr(got, name)
        assert t.device == DEV, name
        assert torch.equal(t.cpu(), getattr(ref, name)), name
    with pytest.raises(kp.ValidationError, match="labels_i"):
        kp.ranges_from_labels(li.flip(0).to(DEV), lj.to(DEV), keep.to(DEV))
