"""Contrat v0.2 : default_variant / variants / compile(variant) du backend hip.

Toutes les variantes ``patient`` d'un plan donnent des résultats identiques
pour les réductions exactes et restent dans la tolérance pour SUM.
"""

import math

import pytest
import torch
from _cases import ABS1, COUNT_BALL, GAUSS, SQDIST, VEC, make_plan, pair_provider, ref_reduce

from kernox.backends.base import Sizes, UnsupportedError, Variant
from kernox.backends.hip import HipBackend
from kernox.plan import ReductionKind as R
from kernox.plan import output_shapes

pytestmark = pytest.mark.gpu
DEV = "cuda"


@pytest.fixture(scope="module")
def be():
    return HipBackend(pair_provider=pair_provider)


def _inputs(case, M, N, seed):
    g = torch.Generator().manual_seed(seed)
    t = {}
    for a in case.args:
        L = M if a.role == "i" else N
        t[a.name] = (torch.randn(L, a.dim, generator=g, dtype=torch.float64) if a.role != "param"
                     else torch.full((a.dim,), 1.3, dtype=torch.float64))
    if case is ABS1:  # égalités massives + NaN + inf
        t = {"x": torch.randint(-3, 4, (M, 1), generator=g).double(),
             "y": torch.randint(-3, 4, (N, 1), generator=g).double()}
        t["y"][::9] = math.nan
        t["y"][4::13] = math.inf
    return t


def _launch(be, plan, variant, host, M):
    T = getattr(torch, plan.dtype)
    inp = {k: v.to(T).to(DEV) for k, v in host.items()}
    outs = {n: torch.full(s, 7, dtype=getattr(torch, d), device=DEV)
            for n, (s, d) in output_shapes(plan, M).items()}
    cr = be.compile(plan, DEV, variant)
    cr.launch(inp, outs)
    torch.cuda.synchronize()
    return {k: v.cpu() for k, v in outs.items()}, cr


def test_default_variant_heuristic(be):
    big = Sizes(10**6, 10**6)
    small = Sizes(100, 10**6)
    assert be.default_variant(make_plan(SQDIST, R.SUM, "float32"), DEV, big).name == "row"
    assert be.default_variant(make_plan(SQDIST, R.SUM, "float32"), DEV, small).name == "jsplit"
    assert be.default_variant(make_plan(SQDIST, R.KMIN, "float32", 8), DEV, big).name == "row"
    v = be.default_variant(make_plan(SQDIST, R.ARGKMIN, "float64", 128), DEV, small)
    assert v.name == "coop_topk" and dict(v.params)["group"] == 16 and dict(v.params)["split"] == "auto"
    # top-K K>=32 : pas de partition dès 4 blocs/CU (mesuré : M=50 000, K=32 → G2, 391 blocs)
    v = be.default_variant(make_plan(SQDIST, R.KMIN, "float32", 32), DEV, Sizes(50_000, 50_000))
    assert v.name == "coop_topk" and dict(v.params)["group"] == 2 and dict(v.params)["split"] == "off"
    # G16 : partition sous 4 blocs/CU, dimensionnée sur 8 blocs/CU (mesuré : M=1024 → S=4)
    v = be.default_variant(make_plan(SQDIST, R.KMIN, "float32", 128), DEV, Sizes(1024, 200_000))
    assert dict(v.params) == {"block": 256, "tile_j": dict(v.params)["tile_j"], "group": 16,
                              "split": "auto", "blocks_per_cu": 8}
    cr = be.compile(make_plan(SQDIST, R.KMIN, "float32", 128), DEV, v)
    if cr.cu_count == 32:
        assert cr.split_plan(1024, 200_000)[0] == 4
    v = be.default_variant(make_plan(SQDIST, R.KMIN, "float32", 128), DEV, Sizes(2048, 200_000))
    assert dict(v.params)["split"] == "off"
    # K=48 : G2 (24 rangs par lane)
    assert dict(be.default_variant(make_plan(SQDIST, R.KMIN, "float32", 48), DEV, big).params)["group"] == 2
    # déterministe
    assert be.default_variant(make_plan(SQDIST, R.SUM, "float32"), DEV, small) == \
        be.default_variant(make_plan(SQDIST, R.SUM, "float32"), DEV, small)


@pytest.mark.parametrize("k", [None, 4, 32, 64, 256])
@pytest.mark.parametrize("sizes", [Sizes(100, 10**6), Sizes(10**6, 10**6)], ids=["smallM", "bigM"])
def test_variant_counts_and_validity(be, k, sizes):
    plan = make_plan(SQDIST, R.SUM if k is None else R.KMIN_ARGKMIN, "float32", k)
    d = be.default_variant(plan, DEV, sizes)
    m = be.variants(plan, DEV, sizes, "measure")
    p = be.variants(plan, DEV, sizes, "patient")
    assert m[0] == d and p[0] == d
    assert 2 <= len(m) <= 4 and len(p) <= 32 and len(p) >= len(m)
    assert len(set(m)) == len(m) and len(set(p)) == len(p)
    for v in m + p:
        assert v.name in ("row", "jsplit", "coop_topk")
        prm = dict(v.params)
        assert prm["block"] % 64 == 0 and prm["tile_j"] >= 1
        assert (prm["split"] == "auto") == ("blocks_per_cu" in prm)
    with pytest.raises(ValueError):
        be.variants(plan, DEV, sizes, "exhaustive")


def test_invalid_variants_refused(be):
    plan = make_plan(SQDIST, R.SUM, "float64")
    for v in [Variant("coop_topk", (("group", 8),)), Variant("row", (("block", 96),)),
              Variant("nope"), Variant("row", (("split", "auto"),)), Variant("row", (("foo", 1),))]:
        with pytest.raises(UnsupportedError):
            be.compile(plan, DEV, v)


def test_variant_in_cache_key_and_info(be):
    plan = make_plan(SQDIST, R.MIN, "float32")
    vs = be.variants(plan, DEV, Sizes(50, 5000), "measure")
    crs = [be.compile(plan, DEV, v) for v in vs]
    assert [c.info.variant for c in crs] == vs
    assert len({c.cache_key for c in crs}) == len(vs)
    assert be.compile(plan, DEV).info.variant is not None  # variant=None : variante par défaut publiée


CASES = [(VEC, R.MIN, None, "float32"), (VEC, R.ARGMAX, None, "float64"), (ABS1, R.ARGMIN, None, "float32"),
         (COUNT_BALL, R.COUNT, None, "float64"), (ABS1, R.KMIN_ARGKMIN, 8, "float32"),
         (ABS1, R.KMIN_ARGKMIN, 70, "float64"), (SQDIST, R.ARGKMIN, 130, "float32"),
         (GAUSS, R.SUM, None, "float64"), (GAUSS, R.SUM, None, "float32")]


@pytest.mark.parametrize("case,kind,k,dtype", CASES,
                         ids=lambda v: getattr(v, "name", getattr(v, "value", v)))
@pytest.mark.parametrize("M,N", [(40, 700), (700, 300)])
def test_all_patient_variants_agree(be, case, kind, k, dtype, M, N):
    plan = make_plan(case, kind, dtype, k)
    host = _inputs(case, M, N, seed=M + N + (k or 0))
    variants = be.variants(plan, DEV, Sizes(M, N), "patient")
    T = getattr(torch, dtype)
    cast = {kk: v.to(T) for kk, v in host.items()}
    F = case.torch_f(cast)
    ref = ref_reduce(F, kind, k, acc=T)
    first = None
    for v in variants:
        out, cr = _launch(be, plan, v, host, M)
        assert cr.info.variant == v
        for n, r in ref.items():
            if kind is R.SUM:
                eps = torch.finfo(T).eps
                l1 = F.abs().sum(1)
                assert torch.all((out[n] - r).abs() <= 8 * eps * N ** 0.5 * l1), v
            else:
                assert torch.equal(out[n], r), v
        if kind is not R.SUM:
            if first is None:
                first = out
            assert all(torch.equal(out[n], first[n]) for n in out), v


def test_default_variant_heavy_row():
    """Ligne j >= 64 dimensions : thread à K=32, coop dès K=48 (mesuré, D64)."""
    import kernox as kp

    from kernox.backends.hip._kernel import COOP_HEAVY_D, COOP_K_MIN_HEAVY

    be = HipBackend()
    sizes = Sizes(50_000, 50_000)

    def plan(d, k, dtype="float32"):
        return (((kp.Vi("x", d) - kp.Vj("y", d)) ** 2).sum()).argkmin(k, over="j").to_plan(dtype)

    assert COOP_HEAVY_D == 64 and COOP_K_MIN_HEAVY == 48
    assert be.default_variant(plan(64, 32), DEV, sizes).name == "row"
    assert be.default_variant(plan(48, 32), DEV, sizes).name == "coop_topk"
    assert be.default_variant(plan(3, 32, "float64"), DEV, sizes).name == "coop_topk"
    v = be.default_variant(plan(64, 48), DEV, sizes)
    assert v.name == "coop_topk" and dict(v.params)["group"] == 2
