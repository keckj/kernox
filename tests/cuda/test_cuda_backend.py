"""Correction du backend CUDA sur GPU réel (référence Torch locale).

À remplacer/compléter par l'oracle du noyau quand il est livré.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.version.hip is not None:  # pragma: no cover
    pytest.skip("GPU NVIDIA requis", allow_module_level=True)

from kernox.backends.base import UnsupportedError  # noqa: E402
from kernox.backends.cuda import CudaBackend  # noqa: E402
from kernox.plan import output_shapes  # noqa: E402

from ._helpers import (  # noqa: E402
    count_body, gauss_body, gauss_ref, ident_body, make_plan, pair_from, ref_kmin,
    ref_minmax, sqdist_body, sqdist_ref, two_comp_body, two_comp_ref,
)

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda", 0)


def run(backend, plan, inputs, m):
    comp = backend.compile(plan, DEV)
    outs = {name: torch.empty(shape, dtype=getattr(torch, dt), device=DEV)
            for name, (shape, dt) in output_shapes(plan, m).items()}
    comp.launch(inputs, outs)
    torch.cuda.synchronize()
    return outs, comp


# ------------------------------------------------------------------ SUM


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("M,N,D,E", [(1, 1, 3, 1), (300, 517, 3, 1), (64, 5000, 16, 2),
                                      (2000, 70, 64, 1), (5, 100_000, 3, 3)])
def test_gauss_sum(dtype, M, N, D, E):
    g = torch.Generator(device="cpu").manual_seed(M * 7 + N)
    tdt = getattr(torch, dtype)
    x = torch.rand(M, D, generator=g, dtype=torch.float64).to(DEV, tdt)
    y = torch.rand(N, D, generator=g, dtype=torch.float64).to(DEV, tdt)
    b = torch.randn(N, E, generator=g, dtype=torch.float64).to(DEV, tdt)
    s = torch.tensor([0.3 * math.sqrt(D)], dtype=tdt, device=DEV)
    plan = make_plan([("x", "i", D), ("y", "j", D), ("b", "j", E), ("s", "param", 1)], E,
                     "sum", dtype, tag=f"gauss{D}_{E}")
    out, _ = run(CudaBackend(pair_from(gauss_body)), plan, {"x": x, "y": y, "b": b, "s": s}, M)
    ref = gauss_ref(x.cpu(), y.cpu(), b.cpu(), s.cpu())
    l1 = (torch.exp(-sqdist_ref(x.cpu().double(), y.cpu().double()) / (2 * s.cpu().double()[0] ** 2))
          @ b.cpu().double().abs())
    eps = 1e-13 if dtype == "float64" else 2e-5
    err = (out["value"].cpu().double() - ref).abs()
    assert bool((err <= eps * (l1 + 1e-300) + 1e-300).all()), float((err / (l1 + 1e-300)).max())


def test_sum_fp32_acc_fp64():
    M, N, D = 50, 20000, 3
    x = torch.rand(M, D, device=DEV)
    y = torch.rand(N, D, device=DEV)
    b = torch.rand(N, 1, device=DEV)
    s = torch.tensor([0.5], device=DEV)
    plan = make_plan([("x", "i", D), ("y", "j", D), ("b", "j", 1), ("s", "param", 1)], 1,
                     "sum", "float32", acc_dtype="float64", tag="g")
    out, _ = run(CudaBackend(pair_from(gauss_body)), plan, {"x": x, "y": y, "b": b, "s": s}, M)
    assert out["value"].dtype == torch.float64
    ref = gauss_ref(x.cpu(), y.cpu(), b.cpu(), s.cpu())
    assert torch.allclose(out["value"].cpu(), ref, rtol=1e-5)


def test_sum_nan_propagates():
    x = torch.zeros(3, 1, dtype=torch.float64, device=DEV)
    y = torch.tensor([[1.0], [float("nan")], [2.0]], dtype=torch.float64, device=DEV)
    plan = make_plan([("x", "i", 1), ("y", "j", 1)], 1, "sum", tag="id")
    out, _ = run(CudaBackend(pair_from(ident_body)), plan, {"x": x, "y": y}, 3)
    assert torch.isnan(out["value"]).all()


# ------------------------------------------------------------------ COUNT


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_count(dtype):
    tdt = getattr(torch, dtype)
    # coordonnées entières : distances exactes, rayon exactement atteint
    g = torch.Generator().manual_seed(3)
    x = torch.randint(0, 6, (257, 3), generator=g).to(DEV, tdt)
    y = torch.randint(0, 6, (3001, 3), generator=g).to(DEV, tdt)
    y[5, 0] = float("nan")
    r2 = torch.tensor([4.0], dtype=tdt, device=DEV)
    plan = make_plan([("x", "i", 3), ("y", "j", 3), ("r2", "param", 1)], 1, "count", dtype, tag="c")
    out, _ = run(CudaBackend(pair_from(count_body)), plan, {"x": x, "y": y, "r2": r2}, 257)
    d2 = sqdist_ref(x.cpu().double(), y.cpu().double())
    ref = (d2 <= 4.0).sum(1, keepdim=True)  # NaN <= 4 est faux
    assert out["value"].dtype == torch.int64
    assert torch.equal(out["value"].cpu(), ref)


# ------------------------------------------------------------------ MIN/MAX/ARG


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("kind", ["min", "max", "argmin", "argmax"])
@pytest.mark.parametrize("M,N", [(37, 129), (3, 4000), (600, 33)])
def test_minmax_two_components(kind, dtype, M, N):
    tdt = getattr(torch, dtype)
    g = torch.Generator().manual_seed(M + N)
    # petits entiers : nombreuses égalités exactes
    x = torch.randint(-3, 4, (M, 2), generator=g).to(DEV, tdt)
    y = torch.randint(-3, 4, (N, 2), generator=g).to(DEV, tdt)
    y[1, 0] = float("nan")
    y[N // 2, 1] = float("nan")
    plan = make_plan([("x", "i", 2), ("y", "j", 2)], 2, kind, dtype, tag="tc")
    out, _ = run(CudaBackend(pair_from(two_comp_body)), plan, {"x": x, "y": y}, M)
    rv, ri = ref_minmax(two_comp_ref(x.cpu().double(), y.cpu().double()), kind)
    if kind in ("min", "max"):
        assert torch.equal(out["value"].cpu().double(), rv)
    else:
        assert out["index"].dtype == torch.int64
        assert torch.equal(out["index"].cpu(), ri)


def test_arg_all_nan_and_inf():
    # ligne 0 : que des NaN -> -1 / +inf ; ligne 1 : +inf valide -> indice >= 0
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64, device=DEV)
    nan, inf = float("nan"), float("inf")
    y = torch.tensor([[nan], [inf], [nan]], dtype=torch.float64, device=DEV)

    def body(plan, T):  # ligne 0 -> NaN partout ; ligne 1 -> y_j
        return " r.o0 = (a0[0] == 0.0) ? (" + T + ")NAN : a1[0];"

    be = CudaBackend(pair_from(body))
    out, _ = run(be, make_plan([("x", "i", 1), ("y", "j", 1)], 1, "argmin", tag="nan"),
                 {"x": x, "y": y}, 2)
    assert out["index"].cpu().tolist() == [[-1], [1]]
    out, _ = run(be, make_plan([("x", "i", 1), ("y", "j", 1)], 1, "min", tag="nan"),
                 {"x": x, "y": y}, 2)
    assert out["value"].cpu().tolist() == [[inf], [inf]]


# ------------------------------------------------------------------ KMIN


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("kind", ["kmin", "argkmin", "kmin_argkmin"])
@pytest.mark.parametrize("K", [1, 2, 8, 33, 128])
@pytest.mark.parametrize("M,N", [(70, 300), (4, 6000)])
def test_kmin_ties(kind, dtype, K, M, N):
    tdt = getattr(torch, dtype)
    g = torch.Generator().manual_seed(K * 11 + M)
    x = torch.randint(0, 4, (M, 3), generator=g).to(DEV, tdt)
    y = torch.randint(0, 4, (N, 3), generator=g).to(DEV, tdt)
    y[7, 1] = float("nan")
    plan = make_plan([("x", "i", 3), ("y", "j", 3)], 1, kind, dtype, k=K, tag="sq")
    out, _ = run(CudaBackend(pair_from(sqdist_body)), plan, {"x": x, "y": y}, M)
    rv, ri = ref_kmin(sqdist_ref(x.cpu().double(), y.cpu().double()), K)
    if "value" in out:
        assert torch.equal(out["value"].cpu().double(), rv)
    if "index" in out:
        assert torch.equal(out["index"].cpu(), ri)


def test_kmin_random_float_values_match():
    # données continues : comparaison des valeurs, cohérence indice -> valeur
    M, N, K = 200, 20000, 16
    x = torch.rand(M, 3, dtype=torch.float64, device=DEV)
    y = torch.rand(N, 3, dtype=torch.float64, device=DEV)
    plan = make_plan([("x", "i", 3), ("y", "j", 3)], 1, "kmin_argkmin", k=K, tag="sq")
    out, _ = run(CudaBackend(pair_from(sqdist_body)), plan, {"x": x, "y": y}, M)
    d2 = sqdist_ref(x.cpu(), y.cpu())
    rv = d2.sort(1).values[:, :K]
    assert torch.allclose(out["value"].cpu(), rv, rtol=1e-12, atol=0)
    got = d2.gather(1, out["index"].cpu())
    assert torch.allclose(got, out["value"].cpu(), rtol=1e-12, atol=0)


@pytest.mark.parametrize("K", [3, 5])
def test_k_greater_than_n(K):
    x = torch.zeros(3, 1, dtype=torch.float64, device=DEV)
    y = torch.tensor([[2.0], [float("inf")], [1.0]], dtype=torch.float64, device=DEV)
    plan = make_plan([("x", "i", 1), ("y", "j", 1)], 1, "kmin_argkmin", k=K, tag="id")
    out, _ = run(CudaBackend(pair_from(ident_body)), plan, {"x": x, "y": y}, 3)
    rv, ri = ref_kmin(y.cpu().double().T.expand(3, 3), K)
    assert torch.equal(out["value"].cpu(), rv)
    assert torch.equal(out["index"].cpu(), ri)


def test_k_padding_with_inf_candidate():
    x = torch.zeros(2, 1, dtype=torch.float64, device=DEV)
    y = torch.tensor([[float("inf")], [float("nan")], [3.0]], dtype=torch.float64, device=DEV)
    plan = make_plan([("x", "i", 1), ("y", "j", 1)], 1, "kmin_argkmin", k=4, tag="id")
    out, _ = run(CudaBackend(pair_from(ident_body)), plan, {"x": x, "y": y}, 2)
    assert out["index"].cpu().tolist() == [[2, 0, -1, -1]] * 2
    assert out["value"].cpu().tolist() == [[3.0, math.inf, math.inf, math.inf]] * 2


# ------------------------------------------------------------------ bords


def test_k_zero_no_launch():
    plan = make_plan([("x", "i", 1), ("y", "j", 1)], 1, "argkmin", k=0, tag="id")
    x = torch.zeros(4, 1, dtype=torch.float64, device=DEV)
    y = torch.zeros(5, 1, dtype=torch.float64, device=DEV)
    out, _ = run(CudaBackend(pair_from(ident_body)), plan, {"x": x, "y": y}, 4)
    assert out["index"].shape == (4, 0)


@pytest.mark.parametrize("kind,k", [("sum", None), ("count", None), ("min", None), ("max", None),
                                     ("argmin", None), ("argmax", None), ("kmin_argkmin", 3)])
def test_n_zero_neutral(kind, k):
    x = torch.zeros(5, 1, dtype=torch.float64, device=DEV)
    y = torch.zeros(0, 1, dtype=torch.float64, device=DEV)
    body = ident_body if kind != "count" else (lambda p, T: " r.o0 = a1[0] <= a0[0];")
    plan = make_plan([("x", "i", 1), ("y", "j", 1)], 1, kind, k=k, tag="id")
    out, _ = run(CudaBackend(pair_from(body)), plan, {"x": x, "y": y}, 5)
    if kind in ("sum", "count"):
        assert (out["value"] == 0).all()
    if kind in ("min", "kmin_argkmin"):
        assert (out["value"] == math.inf).all()
    if kind == "max":
        assert (out["value"] == -math.inf).all()
    if "index" in out:
        assert (out["index"] == -1).all()


def test_m_zero():
    x = torch.zeros(0, 1, dtype=torch.float64, device=DEV)
    y = torch.ones(10, 1, dtype=torch.float64, device=DEV)
    plan = make_plan([("x", "i", 1), ("y", "j", 1)], 1, "kmin_argkmin", k=2, tag="id")
    out, _ = run(CudaBackend(pair_from(ident_body)), plan, {"x": x, "y": y}, 0)
    assert out["value"].shape == (0, 2)


def test_unsupported_explicit():
    be = CudaBackend(pair_from(two_comp_body))
    with pytest.raises(UnsupportedError):
        be.compile(make_plan([("x", "i", 2), ("y", "j", 2)], 2, "kmin", k=3), DEV)
    with pytest.raises(UnsupportedError):
        be.compile(make_plan([("x", "i", 1), ("y", "j", 1)], 1, "kmin", k=1025), DEV)
    with pytest.raises(UnsupportedError):
        be.compile(make_plan([("x", "i", 1), ("y", "j", 1)], 1, "kmin", k=-1), DEV)
    with pytest.raises(UnsupportedError):
        be.compile(make_plan([("x", "i", 1), ("y", "j", 1)], 1, "sum"), torch.device("cpu"))


def test_bad_input_rejected_before_launch():
    plan = make_plan([("x", "i", 1), ("y", "j", 1)], 1, "sum", tag="id")
    comp = CudaBackend(pair_from(ident_body)).compile(plan, DEV)
    out = {"value": torch.empty(4, 1, dtype=torch.float64, device=DEV)}
    with pytest.raises(ValueError):
        comp.launch({"x": torch.zeros(4, 2, dtype=torch.float64, device=DEV)[:, :1],
                     "y": torch.zeros(3, 1, dtype=torch.float64, device=DEV)}, out)
    with pytest.raises(ValueError):
        comp.launch({"x": torch.zeros(4, 1, device=DEV),
                     "y": torch.zeros(3, 1, dtype=torch.float64, device=DEV)}, out)


# ------------------------------------------------------------------ interop


def test_non_default_stream_and_split():
    """Entrées produites sur un stream non défaut ; lancement sur ce stream ;
    pas de synchro hôte dans launch ; partition de j active (M petit, N grand)."""
    be = CudaBackend(pair_from(sqdist_body))
    plan = make_plan([("x", "i", 3), ("y", "j", 3)], 1, "kmin_argkmin", k=4, tag="sq")
    s = torch.cuda.Stream(device=DEV)
    with torch.cuda.stream(s):
        x = torch.rand(8, 3, dtype=torch.float64, device=DEV)
        y = torch.rand(200_000, 3, dtype=torch.float64, device=DEV)
        comp = be.compile(plan, DEV)
        S, _ = comp.split(8, 200_000)
        assert S > 1
        outs = {n: torch.empty(sh, dtype=getattr(torch, dt), device=DEV)
                for n, (sh, dt) in output_shapes(plan, 8).items()}
        comp.launch({"x": x, "y": y}, outs)
        res = outs["value"] + 0  # consommation sur le même stream
    s.synchronize()
    rv = sqdist_ref(x.cpu(), y.cpu()).sort(1).values[:, :4]
    assert torch.allclose(res.cpu(), rv, rtol=1e-12, atol=0)


@pytest.mark.parametrize("n", [1000, 1003, 37])
def test_split_reduces_launch_tile(n):
    """split-policy-3 : N petit devant la tuile, partition active via une tuile
    de lancement réduite (multiple de 4, >= 16) ; résultat exact (KMIN)."""
    be = CudaBackend(pair_from(sqdist_body))
    plan = make_plan([("x", "i", 3), ("y", "j", 3)], 1, "kmin_argkmin", k=3, tag="sq")
    comp = be.compile(plan, DEV)
    S, jchunk, tj = comp.launch_config(8, n)
    assert tj % 4 == 0 and tj <= comp.tile and S * jchunk >= n
    assert S == 1 or jchunk % tj == 0
    if n >= 4 * 16 * 2:
        assert S > 1 and tj < comp.tile
    g = torch.Generator().manual_seed(n)
    x = torch.rand(8, 3, dtype=torch.float64, generator=g).to(DEV)
    y = torch.rand(n, 3, dtype=torch.float64, generator=g).to(DEV)
    outs = {nm: torch.empty(sh, dtype=getattr(torch, dt), device=DEV)
            for nm, (sh, dt) in output_shapes(plan, 8).items()}
    comp.launch({"x": x, "y": y}, outs)
    torch.cuda.synchronize(DEV)
    ref = sqdist_ref(x.cpu(), y.cpu()).sort(1)
    assert torch.allclose(outs["value"].cpu(), ref.values[:, :3], rtol=1e-12, atol=0)


def test_cache_hit_second_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNOX_CUDA_CACHE_DIR", str(tmp_path))
    plan = make_plan([("x", "i", 1), ("y", "j", 1)], 1, "sum", tag="cache")
    c1 = CudaBackend(pair_from(ident_body)).compile(plan, DEV)
    c2 = CudaBackend(pair_from(ident_body)).compile(plan, DEV)
    assert not c1.info.cache_hit and c2.info.cache_hit
    assert c1.info.source_sha256 == c2.info.source_sha256
    assert not list(tmp_path.glob("cuda/.tmp-*"))


@pytest.mark.parametrize("kind,k", [("argmin", None), ("kmin_argkmin", 5)])
def test_wide_index_variant_matches(kind, k):
    """Variante à indices locaux int64 (utilisée si N >= 2^31) forcée sur un petit cas."""
    g = torch.Generator().manual_seed(9)
    x = torch.randint(0, 4, (50, 3), generator=g).to(DEV, torch.float64)
    y = torch.randint(0, 4, (3000, 3), generator=g).to(DEV, torch.float64)
    plan = make_plan([("x", "i", 3), ("y", "j", 3)], 1, kind, k=k, tag="sq")
    be = CudaBackend(pair_from(sqdist_body))
    outs = []
    for li in ("int", "long long"):
        comp = be.compile(plan, DEV, _local_index=li)
        assert f"local_index={li}" in comp.info.options
        o = {n: torch.empty(sh, dtype=getattr(torch, dt), device=DEV)
             for n, (sh, dt) in output_shapes(plan, 50).items()}
        comp.launch({"x": x, "y": y}, o)
        outs.append(o)
    torch.cuda.synchronize()
    for key in outs[0]:
        assert torch.equal(outs[0][key], outs[1][key])


# ------------------------------------------------------------------ variantes (contrat v0.2)


@pytest.mark.parametrize("effort", ["measure", "patient"])
def test_variants_same_semantics(effort):
    from kernox.backends.base import Sizes, Variant

    g = torch.Generator().manual_seed(4)
    x = torch.randint(0, 4, (300, 3), generator=g).to(DEV, torch.float64)
    y = torch.randint(0, 4, (20000, 3), generator=g).to(DEV, torch.float64)
    plan = make_plan([("x", "i", 3), ("y", "j", 3)], 1, "kmin_argkmin", k=7, tag="sq")
    be = CudaBackend(pair_from(sqdist_body))
    sizes = Sizes(300, 20000)
    dv = be.default_variant(plan, DEV, sizes)
    vs = be.variants(plan, DEV, sizes, effort)
    assert vs[0] == dv and len(vs) == len(set(vs)) and len(vs) <= (4 if effort == "measure" else 32)
    assert len(vs) > 1
    ref = None
    for v in vs:
        comp = be.compile(plan, DEV, v)
        assert comp.info.variant == v and comp.variant == v
        o = {n: torch.empty(sh, dtype=getattr(torch, dt), device=DEV)
             for n, (sh, dt) in output_shapes(plan, 300).items()}
        comp.launch({"x": x, "y": y}, o)
        torch.cuda.synchronize()
        if ref is None:
            ref = o
        for k in o:
            assert torch.equal(o[k], ref[k]), v
    with pytest.raises(UnsupportedError):
        be.compile(plan, DEV, Variant("tiled", (("block", 4096), ("split", "auto"), ("tile", 64))))
    with pytest.raises(UnsupportedError):
        be.compile(plan, DEV, Variant("coop", ()))


def test_split_policy4_auto_factor():
    """split-policy-4 : "auto" vise 2 vagues pour les listes K >= 8 bon marché
    (FP32, j <= 16 composantes) ou M très petit ; 4 vagues sinon. Les modes
    explicites x2/x4 sont des variantes valides et proposées au tuning."""
    from kernox.backends.base import Sizes

    be = CudaBackend(pair_from(sqdist_body))
    def comp_for(kind, k, dtype, d):
        plan = make_plan([("x", "i", d), ("y", "j", d)], 1, kind, k=k, tag="sq", dtype=dtype)
        return plan, be.compile(plan, DEV)
    _, c = comp_for("kmin_argkmin", 32, "float32", 3)
    assert c.auto_factor() == 2
    _, c = comp_for("kmin_argkmin", 4, "float32", 3)
    assert c.auto_factor() == 4
    plan, c = comp_for("kmin_argkmin", 32, "float64", 3)
    big = c.waves_args(10 ** 7)
    assert c.auto_factor(*big) == 4 and c.auto_factor(*c.waves_args(c.block)) == 2
    for eff in ("measure", "patient"):
        modes = {dict(v.params)["split"] for v in be.variants(plan, DEV, Sizes(10 ** 7, 20000), eff)}
        assert "x2" in modes and "off" in modes, (eff, modes)


def test_fp_contract_controls_fmad():
    """Contrat v0.3 : --fmad suit Plan.fp_contract (défaut True) ; l'option fait partie
    de la clé de cache. x = y = 1 + 2^-30, z = -(1 + 2^-29) en FP64 :
    sans contraction x*y arrondi = 1 + 2^-29 -> 0 ; avec FMA -> 2^-60 exactement."""
    from dataclasses import replace

    def fma_body(plan, T):
        return " r.o0 = a0[0] * a1[0] + a2[0];"

    be = CudaBackend(pair_from(fma_body))
    base = make_plan([("x", "i", 1), ("y", "j", 1), ("z", "j", 1)], 1, "sum", tag="fma")
    assert base.fp_contract is True
    x = torch.full((4, 1), 1 + 2.0 ** -30, dtype=torch.float64, device=DEV)
    y = torch.full((1, 1), 1 + 2.0 ** -30, dtype=torch.float64, device=DEV)
    z = torch.full((1, 1), -(1 + 2.0 ** -29), dtype=torch.float64, device=DEV)
    res = {}
    for fc in (True, False):
        plan = replace(base, fp_contract=fc)
        comp = be.compile(plan, DEV)
        assert ("--fmad=true" if fc else "--fmad=false") in comp.info.options
        assert ("--fmad=false" if fc else "--fmad=true") not in comp.info.options
        o = {n: torch.empty(sh, dtype=getattr(torch, dt), device=DEV) for n, (sh, dt) in output_shapes(plan, 4).items()}
        comp.launch({"x": x, "y": y, "z": z}, o)
        torch.cuda.synchronize()
        res[fc] = (o["value"].cpu(), comp.info.source_sha256, comp)
    assert torch.equal(res[True][0], torch.full((4, 1), 2.0 ** -60, dtype=torch.float64))
    assert torch.equal(res[False][0], torch.zeros(4, 1, dtype=torch.float64))
    assert res[True][2] is not res[False][2]  # clés de cache distinctes (options)


def test_fp_contract_off_through_api():
    """Contrat v0.3 : le backend déclare fp_contract_off ; l'API accepte fp_contract=False
    et l'option --fmad=false atteint NVRTC (x*y + z de test_fp_contract_controls_fmad)."""
    import kernox as kp

    assert CudaBackend.fp_contract_off is True
    x = kp.Vi("x", 1)
    y = kp.Vj("y", 1)
    z = kp.Vj("z", 1)
    xs = torch.full((4, 1), 1 + 2.0 ** -30, dtype=torch.float64, device=DEV)
    ys = torch.full((1, 1), 1 + 2.0 ** -30, dtype=torch.float64, device=DEV)
    zs = torch.full((1, 1), -(1 + 2.0 ** -29), dtype=torch.float64, device=DEV)
    for fc in (True, False):
        r = kp.compile((x * y + z).sum_over("j"), dtype="float64", backend="cuda", fp_contract=fc)
        assert r.plan.fp_contract is fc
        out = r(x=xs, y=ys, z=zs)
        assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("D", [384, 990, 2000])
def test_large_dim_global_staging(dtype, D):
    """Grand D : une ligne j ne tient plus en mémoire partagée (tuile de 16) ; lecture
    globale directe (staging=global), même résultat que la référence Torch FP64.
    Conformance : test_min_argmin_chunked (D = 990), qui levait UnsupportedError."""
    import kernox as kp
    from kernox.backends.cuda.backend import row_bytes, staging_for

    x, y = kp.Vi("x", D), kp.Vj("y", D)
    g = torch.Generator().manual_seed(D)
    xs = (torch.rand(37, D, generator=g, dtype=torch.float64) / D).to(DEV, getattr(torch, dtype))
    ys = (torch.rand(53, D, generator=g, dtype=torch.float64) / D).to(DEV, getattr(torch, dtype))
    d2 = ((xs.double()[:, None, :] - ys.double()[None, :, :]) ** 2).sum(-1)
    r = kp.compile(kp.exp(-((x - y) ** 2).sum()).sum_over("j"), dtype=dtype, backend="cuda")
    expect = "smem" if 16 * row_bytes(r.plan) <= 48 << 10 else "global"
    assert staging_for(r.plan) == expect
    comp = CudaBackend().compile(r.plan, DEV)
    assert f"staging={expect}" in comp.info.options
    out = r(x=xs, y=ys)
    ref = torch.exp(-d2).sum(1, keepdim=True)
    tol = 1e-5 if dtype == "float32" else 1e-12
    assert torch.allclose(out.double().cpu(), ref.cpu(), rtol=tol, atol=0)
    rm = kp.compile(((x - y) ** 2).sum().argmin(over="i"), dtype=dtype, backend="cuda")
    idx = rm(x=xs, y=ys).reshape(-1).cpu()
    d2c = d2.cpu()
    best = d2c.min(0).values
    assert torch.all(d2c[idx, torch.arange(53)] <= best * (1 + 4 * D * torch.finfo(getattr(torch, dtype)).eps))


def test_launch_cache_per_variant():
    """Coût hôte : configurations de lancement mémorisées par (M, N), propres à chaque
    variante (with_variant ne partage pas le cache) ; résultats identiques."""
    import kernox as kp
    from kernox.backends.base import Variant

    x, y = kp.Vi("x", 3), kp.Vj("y", 3)
    r = kp.compile(((x - y) ** 2).sum().min(over="j"), dtype="float32", backend="cuda")
    base = CudaBackend().compile(r.plan, DEV)
    xs, ys = torch.rand(100, 3, device=DEV), torch.rand(20000, 3, device=DEV)
    outs = {}
    for sp in ("auto", "off", "x8"):
        p = dict(base.variant.params)
        c = base if sp == "auto" else base.with_variant(Variant("tiled", (("block", p["block"]), ("split", sp), ("tile", p["tile"]))))
        o = {c._outs[0]: torch.empty(100, 1, device=DEV)}
        for _ in range(2):
            c.launch({"x": xs, "y": ys}, o)
        assert c._lc["configs"][(100, 20000)] == c.launch_config(100, 20000)
        outs[sp] = o[c._outs[0]].clone()
    assert base._lc is not None and all(torch.equal(outs["auto"], v) for v in outs.values())
    assert base.launch_config(100, 20000)[0] != 1  # la partition est bien exercée
