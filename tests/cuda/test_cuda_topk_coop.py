"""Cooperative top-K (16 <= K <= 1024) of the CUDA backend: G lanes per row i.

Integer data (many ties, exact values), NaN ignored, valid ±inf,
N < K (padding with +inf / −1), forced j partition (cooperative merge
of the slices), G imposed to 8 / 16 / 32: result identical to the
(value, j) ordered reference.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.version.hip is not None:  # pragma: no cover
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from kernox.backends.base import Variant  # noqa: E402
from kernox.backends.cuda import CudaBackend  # noqa: E402
from kernox.backends.cuda.backend import coop_group  # noqa: E402
from kernox.plan import output_shapes  # noqa: E402

from ._helpers import make_plan, pair_from, sqdist_body  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda", 0)


def ref_topk(vals, K):
    """(M, N) FP64 -> (values, indices) (M, K): (value, j) order, NaN ignored,
    padding (+inf, −1)."""
    M, N = vals.shape
    nan = torch.isnan(vals)
    key = torch.where(nan, torch.full_like(vals, math.inf), vals)
    sv, si = torch.sort(key, dim=1, stable=True)            # ties: smallest j first
    valid = ~torch.gather(nan, 1, si)
    # NaN placed last (key +inf, after the real +inf thanks to a stable sort on valid)
    order = torch.sort((~valid).to(torch.int8), dim=1, stable=True).indices
    sv, si, valid = sv.gather(1, order), si.gather(1, order), valid.gather(1, order)
    ov = torch.full((M, K), math.inf, dtype=torch.float64)
    oi = torch.full((M, K), -1, dtype=torch.int64)
    k = min(K, N)
    ov[:, :k] = torch.where(valid[:, :k], sv[:, :k], torch.full_like(sv[:, :k], math.inf))
    oi[:, :k] = torch.where(valid[:, :k], si[:, :k], torch.full_like(si[:, :k], -1))
    return ov, oi


def _run(be, plan, x, y, split=None, tile=None):
    comp = be.compile(plan, DEV)
    if split is not None:
        p = dict(comp.variant.params)
        comp = comp.with_variant(Variant("tiled", (("block", p["block"]), ("split", split),
                                                   ("tile", tile or p["tile"]))))
    M = x.shape[0]
    outs = {n: torch.full(sh, 7, dtype=getattr(torch, dt), device=DEV)
            for n, (sh, dt) in output_shapes(plan, M).items()}
    comp.launch({"x": x, "y": y}, outs)
    torch.cuda.synchronize()
    return outs, comp


def _data(M, N, dtype, seed):
    tdt = getattr(torch, dtype)
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 6, (M, 3), generator=g).to(tdt)
    y = torch.randint(0, 6, (N, 3), generator=g).to(tdt)
    if N > 20:
        y[7, 1] = math.nan
        y[11, 0] = math.inf
    return x, y


def _check(out, x, y, K):
    d2 = ((x.double()[:, None, :] - y.double()[None, :, :]) ** 2).sum(-1)
    rv, ri = ref_topk(d2, K)
    if "value" in out:
        assert torch.equal(out["value"].cpu().double(), rv)
    if "index" in out:
        assert torch.equal(out["index"].cpu(), ri)


def test_coop_group_policy():
    assert coop_group(15) == 0 and coop_group(16) == 16 and coop_group(32) == 16 and coop_group(33) == 32
    assert coop_group(256) == 32 and coop_group(512) == 32 and coop_group(1024) == 32
    # Maxwell (cc 5.x): per-thread list up to K = 31 (measured on a 980 Ti)
    assert coop_group(16, 5) == 0 and coop_group(31, 5) == 0 and coop_group(32, 5) == 16
    assert coop_group(16, 7) == 16 and coop_group(16, 12) == 16 and coop_group(64, 5) == 32


@pytest.mark.parametrize("kind", ["kmin", "argkmin", "kmin_argkmin"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("K", [16, 32, 33, 128, 257, 300, 512, 777, 1024])
@pytest.mark.parametrize("M,N", [(37, 3000), (5, 20000)])
def test_coop_topk_matches(kind, dtype, K, M, N):
    x, y = _data(M, N, dtype, K + M)
    plan = make_plan([("x", "i", 3), ("y", "j", 3)], 1, kind, dtype, k=K, tag="sq")
    out, comp = _run(CudaBackend(pair_from(sqdist_body)), plan, x.to(DEV), y.to(DEV))
    g = coop_group(K, torch.cuda.get_device_capability(DEV)[0])
    assert f"coop={g}" in comp.info.options and comp.lanes == max(g, 1)
    _check(out, x, y, K)


@pytest.mark.parametrize("G", [8, 16, 32])
@pytest.mark.parametrize("opts", [dict(split="off"), dict(split="x8", tile=16), dict(split="x2", tile=64)],
                         ids=lambda o: "-".join(f"{k}{v}" for k, v in o.items()))
def test_coop_groups_and_split(G, opts):
    K = 260 if G == 8 else 400
    M, N = 21, 9000
    x, y = _data(M, N, "float32", G)
    plan = make_plan([("x", "i", 3), ("y", "j", 3)], 1, "kmin_argkmin", "float32", k=K, tag="sq")
    out, comp = _run(CudaBackend(pair_from(sqdist_body), coop_group=G), plan, x.to(DEV), y.to(DEV), **opts)
    assert comp.lanes == G
    if opts["split"] != "off":
        assert comp.launch_config(M, N)[0] > 1
    _check(out, x, y, K)


@pytest.mark.parametrize("N", [0, 1, 100, 300])
def test_coop_k_greater_than_n(N):
    K = 300
    x, y = _data(9, N, "float64", N)
    plan = make_plan([("x", "i", 3), ("y", "j", 3)], 1, "kmin_argkmin", "float64", k=K, tag="sq")
    for opts in (dict(), dict(split="x8", tile=16)):
        out, _ = _run(CudaBackend(pair_from(sqdist_body)), plan, x.to(DEV), y.to(DEV), **opts)
        _check(out, x, y, K)


def test_coop_large_m_partial_warps():
    """M not a multiple of B / G: inactive groups in an active warp (uniform control flow)."""
    K = 600
    x, y = _data(1001, 2500, "float32", 5)
    plan = make_plan([("x", "i", 3), ("y", "j", 3)], 1, "argkmin", "float32", k=K, tag="sq")
    out, _ = _run(CudaBackend(pair_from(sqdist_body)), plan, x.to(DEV), y.to(DEV))
    _check(out, x, y, K)


def test_k_above_1024_refused():
    plan = make_plan([("x", "i", 3), ("y", "j", 3)], 1, "argkmin", "float32", k=1025, tag="sq")
    assert not CudaBackend(pair_from(sqdist_body)).supports(plan, DEV).ok
