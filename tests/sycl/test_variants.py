"""v0.2 contract: sycl backend variants (WG size, j tile, sub-group,
j split). They must all give the same semantics on the G31."""

from __future__ import annotations

import pytest
import torch

from kernox.backends.base import Sizes, UnsupportedError, Variant

from ._helpers import DEV, arg, backend, plan, reduce_ref, run

pytestmark = pytest.mark.gpu


def _p(kind, k=None, dtype="float32"):
    x, y = arg("x", "i", 3), arg("y", "j", 3)
    d2 = sum((a - b) ** 2 for a, b in zip(x.symbols, y.symbols))
    return plan([x, y], [d2 <= 4] if kind == "count" else [d2], kind, dtype, k=k)


def test_variant_counts_and_default_first():
    p = _p("argkmin", 16)
    be = backend()
    d = be.default_variant(p, DEV, Sizes(1000, 1000))
    for effort, cap in (("measure", 4), ("patient", 32)):
        vs = be.variants(p, DEV, Sizes(1000, 1000), effort)
        assert vs[0] == d and len(vs) <= cap and len(set(vs)) == len(vs)
    with pytest.raises(ValueError):
        be.variants(p, DEV, Sizes(1, 1), "exhaustive")


def test_info_variant_and_cache_key():
    p = _p("sum")
    be = backend()
    v0 = be.default_variant(p, DEV, Sizes(10, 10))
    c0 = be.compile(p, DEV)
    assert c0.info.variant == v0
    v1 = be.variants(p, DEV, Sizes(10, 10), "measure")[1]
    c1 = be.compile(p, DEV, v1)
    assert c1.info.variant == v1 and c1.info.source_sha256 != c0.info.source_sha256


def test_invalid_variant_refused():
    p = _p("sum")
    with pytest.raises(UnsupportedError):
        backend().compile(p, DEV, Variant("other"))
    with pytest.raises(UnsupportedError):
        backend().compile(p, DEV, Variant("tiled1d_jsplit", (("wg", 4096),)))
    with pytest.raises(UnsupportedError):
        backend().compile(p, DEV, Variant("tiled1d_jsplit", (("foo", 1),)))


@pytest.mark.parametrize("kind,k", [("argmin", None), ("count", None), ("kmin_argkmin", 9), ("kmin_argkmin", 64)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_all_patient_variants_same_semantics(kind, k, dtype):
    p = _p(kind, k, dtype)
    gen = torch.Generator().manual_seed(3)
    td = getattr(torch, dtype)
    X = torch.randint(-3, 4, (300, 3), generator=gen).to(td)
    Y = torch.randint(-3, 4, (5000, 3), generator=gen).to(td)
    Y[::13, 2] = float("nan")
    D2 = ((X[:, None].double() - Y[None].double()) ** 2).sum(-1)
    F = (D2 <= 4)[..., None] if kind == "count" else D2[..., None]
    ref = reduce_ref("kmin" if k else kind, F, k=k)
    for v in backend().variants(p, DEV, Sizes(300, 5000), "patient"):
        c = backend().compile(p, DEV, v)
        out, _ = run(p, {"x": X.to(DEV), "y": Y.to(DEV)}, compiled=c)
        for key in out:
            assert torch.equal(out[key].to(ref[key].dtype), ref[key]), (v, key)
