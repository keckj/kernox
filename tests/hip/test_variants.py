"""Variantes du backend hip : partition de j (+ fusion), KMIN coopératif, sélection auto.

Petits cas uniquement (GPU partagé). Références : sémantique du contrat
(``ref_reduce``) et oracle du noyau.
"""

import math

import pytest
import torch
from _cases import ABS1, COUNT_BALL, GAUSS, SQDIST, VEC, make_plan, pair_provider, ref_reduce

from kernox.backends.hip import HipBackend, HipOptions
from kernox.backends.hip._kernel import COOP_K_MIN
from kernox.plan import ReductionKind as R
from kernox.plan import output_shapes

pytestmark = pytest.mark.gpu
DEV = "cuda"
DTYPES = ["float64", "float32"]


def _be(**kw):
    return HipBackend(HipOptions(**kw), pair_provider=pair_provider)


def _run(be, plan, host, M):
    T = getattr(torch, plan.dtype)
    inp = {k: v.to(T).to(DEV) for k, v in host.items()}
    outs = {n: torch.full(s, 7, dtype=getattr(torch, d), device=DEV)
            for n, (s, d) in output_shapes(plan, M).items()}
    cr = be.compile(plan, DEV)
    cr.launch(inp, outs)
    torch.cuda.synchronize()
    return {k: v.cpu() for k, v in outs.items()}, {k: v.to(T) for k, v in host.items()}, cr


def _ties(M, N, nan=0.1, inf=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-3, 4, (M, 1), generator=g).double()
    y = torch.randint(-3, 4, (N, 1), generator=g).double()
    u = torch.rand(N, generator=g)
    y[u < nan, 0] = math.nan
    y[(u >= nan) & (u < nan + inf), 0] = math.inf
    return {"x": x, "y": y}


def _rand(case, M, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = {}
    for a in case.args:
        L = M if a.role == "i" else N
        t[a.name] = (torch.randn(L, a.dim, generator=g, dtype=torch.float64) if a.role != "param"
                     else torch.full((a.dim,), 1.3, dtype=torch.float64))
    return t


def _check(out, ref, kind, F=None):
    for n, r in ref.items():
        o = out[n]
        assert o.dtype == r.dtype
        if kind is R.SUM:
            eps = torch.finfo(F.dtype).eps
            l1 = F.abs().to(r.dtype).sum(1)
            assert torch.all((o - r).abs() <= 8 * eps * max(F.shape[1], 1) ** 0.5 * l1 + 1e-300), n
        else:
            assert torch.equal(o, r), (n, (o != r).nonzero()[:5])


# ------------------------------------------------------------- partition de j
SPLIT_OPTS = [dict(split="force", tile_j=8), dict(split="force", tile_j=8, jchunk=24),
              dict(split="force", tile_j=32, jchunk=40), dict(split="auto")]


@pytest.mark.parametrize("opts", SPLIT_OPTS, ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind,k", [(R.MIN, None), (R.MAX, None), (R.ARGMIN, None), (R.ARGMAX, None),
                                    (R.KMIN, 5), (R.ARGKMIN, 9), (R.KMIN_ARGKMIN, 1),
                                    (R.KMIN_ARGKMIN, 20), (R.KMIN_ARGKMIN, 300)],
                         ids=lambda v: getattr(v, "value", v))
def test_split_ties_nan_inf(opts, dtype, kind, k):
    """Égalités massives à cheval sur les plages : départage par plus petit j après fusion."""
    if k is not None and k > 256:
        k = 200  # K > N (N = 150) avec liste longue
    M, N = 37, 150
    be = _be(**opts)
    plan = make_plan(ABS1, kind, dtype, k)
    out, cast, cr = _run(be, plan, _ties(M, N, seed=hash(str(opts)) % 1000), M)
    if opts.get("split") == "force":
        assert cr.last_launch["S"] > 1 and cr.last_launch["temp_bytes"] > 0
    _check(out, ref_reduce(ABS1.torch_f(cast), kind, k), kind)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case,kind", [(GAUSS, R.SUM), (COUNT_BALL, R.COUNT), (VEC, R.MIN), (VEC, R.ARGMAX)],
                         ids=lambda v: getattr(v, "name", getattr(v, "value", v)))
def test_split_vector_formulas(dtype, case, kind):
    M, N = 50, 1000
    be = _be(split="force", tile_j=16)
    plan = make_plan(case, kind, dtype)
    out, cast, cr = _run(be, plan, _rand(case, M, N, seed=3), M)
    assert cr.last_launch["S"] == 63  # min(S_MAX, ceil(1000/16)) = 63
    F = case.torch_f(cast)
    _check(out, ref_reduce(F, kind, acc=getattr(torch, dtype)), kind, F)


def test_split_sum_fp32_acc_fp64():
    M, N = 20, 3000
    be = _be(split="force", tile_j=32)
    plan = make_plan(GAUSS, R.SUM, "float32", acc_dtype="float64")
    out, cast, cr = _run(be, plan, _rand(GAUSS, M, N, seed=4), M)
    F = GAUSS.torch_f(cast).double()
    assert cr.last_launch["S"] > 1 and out["value"].dtype == torch.float64
    assert torch.all((out["value"] - F.sum(1)).abs() <= 1e-6 * F.abs().sum(1))


@pytest.mark.parametrize("kind,k", [(R.SUM, None), (R.ARGMIN, None), (R.KMIN_ARGKMIN, 4)],
                         ids=lambda v: getattr(v, "value", v))
def test_split_n_zero(kind, k):
    be = _be(split="force", tile_j=8)
    case = VEC if k is None else SQDIST
    plan = make_plan(case, kind, "float64", k)
    out, cast, cr = _run(be, plan, _rand(case, 9, 0), 9)
    assert cr.last_launch["S"] == 1
    _check(out, ref_reduce(case.torch_f(cast), kind, k, acc=torch.float64), kind, case.torch_f(cast))


def test_split_auto_decision_and_temp_bound():
    be = _be()
    plan = make_plan(SQDIST, R.SUM, "float32")
    cr = be.compile(plan, DEV)
    big_m = cr.config.rows_per_block * cr.cu_count * HipOptions().split_blocks_per_cu
    assert cr.split_plan(big_m, 10**6)[0] == 1             # grille déjà pleine
    S, jc = cr.split_plan(100, 10**6)
    assert S > 1 and jc % cr.config.tile_j == 0 and S * jc >= 10**6
    assert cr.temp_bytes(100, 10**6) == S * 100 * 4
    assert cr.info.temp_bytes_per_launch == HipOptions().temp_max_bytes
    small = HipBackend(HipOptions(temp_max_bytes=1000), pair_provider=pair_provider).compile(plan, DEV)
    assert small.temp_bytes(100, 10**6) <= 1000
    off = _be(split="off").compile(plan, DEV)
    assert off.split_plan(1, 10**6) == (1, 10**6) and off.info.temp_bytes_per_launch == 0


# ------------------------------------------------------------- KMIN coopératif
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("k", [1, 7, 33, 64])
def test_coop_kmin_groups(dtype, group, k):
    M, N = 45, 400
    be = _be(strategy="coop", group=group, split="off")
    plan = make_plan(ABS1, R.KMIN_ARGKMIN, dtype, k)
    out, cast, cr = _run(be, plan, _ties(M, N, seed=group * 100 + k), M)
    assert cr.config.strategy == "coop" and cr.config.group == group
    _check(out, ref_reduce(ABS1.torch_f(cast), R.KMIN_ARGKMIN, k), R.KMIN_ARGKMIN)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k,split", [(64, "off"), (128, "off"), (256, "off"), (128, "force"), (256, "force")])
def test_coop_auto_large_k(dtype, k, split):
    M, N = 33, 500
    be = _be(split=split, tile_j=64 if split == "force" else None)
    plan = make_plan(SQDIST, R.KMIN_ARGKMIN, dtype, k)
    host = _rand(SQDIST, M, N, seed=k)
    host["y"][::7] = host["y"][3]   # doublons exacts → égalités de valeur
    host["y"][5::11, 0] = math.nan
    out, cast, cr = _run(be, plan, host, M)
    assert cr.config.strategy == ("coop" if k >= COOP_K_MIN else "thread")
    _check(out, ref_reduce(SQDIST.torch_f(cast), R.KMIN_ARGKMIN, k), R.KMIN_ARGKMIN)


def test_coop_k_greater_than_n_and_argkmin_only():
    be = _be(strategy="coop", group=8)
    for kind in (R.KMIN, R.ARGKMIN):
        plan = make_plan(ABS1, kind, "float32", 100)
        out, cast, _ = _run(be, plan, _ties(10, 30, seed=1), 10)
        _check(out, ref_reduce(ABS1.torch_f(cast), kind, 100), kind)


def test_coop_refused_for_non_kmin():
    s = _be(strategy="coop").supports(make_plan(VEC, R.MIN, "float64"), DEV)
    assert not s.ok and "coop" in s.reason


# ------------------------------------------------------------- frontend vs oracle
@pytest.mark.parametrize("opts", [dict(split="force", tile_j=16), dict(strategy="coop", group=8)],
                         ids=["split", "coop"])
def test_frontend_vs_oracle(opts):
    import kernox as kp

    x, y, w = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("w", 2)
    d2 = ((x - y) ** 2).sum()
    reds = [d2.argkmin(70, over="j"), d2.kmin(70, over="j")]
    if opts.get("split"):
        reds += [(kp.exp(-d2) * w).sum_over("j"), (d2 <= 1.5).sum_over("j"), (x - y).min(over="j"),
                 d2.argmin(over="j"), d2.argkmin(3, over="i")]
    X = torch.randn(40, 3, dtype=torch.float64, device=DEV)
    Y = torch.randn(300, 3, dtype=torch.float64, device=DEV)
    W = torch.randn(300, 2, dtype=torch.float64, device=DEV)
    be = HipBackend(HipOptions(**opts))
    for red in reds:
        plan = red.to_plan("float64")
        names = {a.name for a in plan.args}
        args = {k: v for k, v in {"x": X, "y": Y, "w": W}.items() if k in names}
        ref = red.compile("float64", backend="oracle")(**args)
        ref = ref if isinstance(ref, tuple) else (ref,)
        M = args[plan.args_of("i")[0].name].shape[0]
        outs = {n: torch.empty(s, dtype=getattr(torch, d), device=DEV)
                for n, (s, d) in output_shapes(plan, M).items()}
        be.compile(plan, DEV).launch(args, outs)
        for got, r in zip(outs.values(), ref):
            if got.dtype == torch.int64:
                assert torch.equal(got, r), red
            else:
                torch.testing.assert_close(got, r, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k,N,opts", [
    (512, 1500, dict(split="off")), (512, 1500, dict(split="force", tile_j=64)),
    (512, 1500, dict(strategy="coop", group=16, split="off")),
    (1024, 2500, dict(split="off")), (1024, 2500, dict(split="force", tile_j=64, jchunk=600)),
    (1024, 700, dict(split="off")), (1000, 700, dict(split="force", tile_j=32, jchunk=96)),   # K > N
], ids=lambda v: "-".join(f"{a}{b}" for a, b in v.items()) if isinstance(v, dict) else str(v))
def test_coop_k_up_to_1024(dtype, k, N, opts):
    """K = 512/1024 (coop G=16/32, <= 32 rangs par lane, sans débordement) : égalités de
    valeur, NaN, K > N, partition de j ; correction exacte contre la référence Torch."""
    M = 29
    be = _be(**opts)
    for kind in (R.KMIN_ARGKMIN, R.ARGKMIN):
        plan = make_plan(SQDIST, kind, dtype, k)
        host = _rand(SQDIST, M, N, seed=k + N)
        host["y"][::7] = host["y"][3]   # doublons exacts → égalités de valeur
        host["y"][5::11, 0] = math.nan
        out, cast, cr = _run(be, plan, host, M)
        assert cr.config.strategy == "coop" and cr.config.slots(k) <= 32
        assert cr.config.group == opts.get("group", 16 if (k <= 512 and dtype == "float64") else 32)
        if opts.get("split") == "force":
            assert cr.last_launch["S"] > 1
        _check(out, ref_reduce(SQDIST.torch_f(cast), kind, k), kind)


def test_k_limits():
    be = _be()
    assert not be.supports(make_plan(SQDIST, R.KMIN, "float32", 1025), DEV).ok
    assert be.supports(make_plan(SQDIST, R.KMIN, "float32", 1024), DEV).ok
    s = _be(strategy="thread").supports(make_plan(SQDIST, R.KMIN, "float32", 300), DEV)
    assert not s.ok and "thread" in s.reason
