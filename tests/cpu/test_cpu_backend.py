"""cpu backend specifics: thread-count independence, libmvec switch, auto."""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox import Param, Vi, Vj
from kernox.backends.cpu import toolchain
from kernox.runtime import auto_backend_name


def _cases():
    x, y, b, s = Vi("x", 3), Vj("y", 3), Vj("b", 2), Param("s")
    d2 = ((x - y) ** 2).sum()
    return {
        "gauss": (kp.exp(-d2 / s) * b).sum_over("j"),
        "lse": (-d2 / s).logsumexp(over="j"),
        "ssmw": (-d2 / s).sumsoftmaxweight(b, over="j"),
        "argkmin": d2.argkmin(5, over="j"),
        "min": d2.min(over="j"),
    }


def _inputs(m, n, dtype):
    g = torch.Generator().manual_seed(3)
    t = dict(x=torch.rand(m, 3, generator=g), y=torch.rand(n, 3, generator=g),
             b=torch.randn(n, 2, generator=g), s=torch.tensor([0.3]))
    return {k: v.to(dtype) for k, v in t.items()}


@pytest.fixture
def threads():
    old = torch.get_num_threads()
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("scheme", ["direct", "block", "kahan"])
@pytest.mark.parametrize("name", list(_cases()))
@pytest.mark.parametrize("m,n", [(37, 5000), (3, 40000)])
def test_bitwise_independent_of_thread_count(threads, name, scheme, m, n):
    """The j split depends on the sizes only: same bits with 1 and 4 threads."""
    red = _cases()[name]
    run = kp.compile(red, dtype="float32", backend="cpu", sum_scheme=scheme)
    t = {k: v for k, v in _inputs(m, n, torch.float32).items() if k in run.input_names}
    torch.set_num_threads(1)
    a = run(**t)
    torch.set_num_threads(4)
    b = run(**t)
    assert torch.equal(a, b)
    if (m, n) == (3, 40000):
        assert run.compiled(torch.device("cpu"), m, n).last_split[0] > 1  # partials + merge exercised


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_libmvec_switch(monkeypatch, dtype):
    """KERNOX_CPU_LIBMVEC=0: scalar glibc libm; both builds agree within a few ulp."""
    red = _cases()["gauss"]
    t = _inputs(64, 3000, dtype)
    on = kp.compile(red, dtype=str(dtype).split(".")[1], backend="cpu")
    got_on = on(**t)
    monkeypatch.setenv("KERNOX_CPU_LIBMVEC", "0")
    assert toolchain.libmvec_functions() == ()
    off = kp.compile(red, dtype=str(dtype).split(".")[1], backend="cpu")
    got_off = off(**t)
    assert "libmvec=off" in off.compiled(torch.device("cpu"), 64, 3000).info.options
    eps = torch.finfo(dtype).eps
    l1 = (torch.exp(-torch.cdist(t["x"].double(), t["y"].double()) ** 2 / 0.3) @ t["b"].double().abs())
    assert ((got_on.double() - got_off.double()).abs() <= 64 * eps * l1).all()


@pytest.mark.parametrize("dtype,lo", [(torch.float32, -110.0), (torch.float64, -760.0)])
def test_exp_below_fast_range(dtype, lo):
    """Past libmvec's fast range (about -87.3 / -708.4) exp keeps its subnormal
    results and its special values; log-sum-exp ignores the underflowing terms."""
    name = str(dtype).split(".")[1]
    xs = torch.cat([torch.linspace(lo, 0.0, 4001, dtype=torch.float64).to(dtype),
                    torch.tensor([float("-inf"), float("inf"), float("nan"), 80.0, 1000.0], dtype=dtype)])
    ref = torch.exp(xs.double()).to(dtype).double()
    one = torch.ones(1, 1, dtype=dtype)
    got = kp.compile((kp.exp(Vi("x", 1)) * Vj("one", 1)).sum_over("j"), dtype=name, backend="cpu")(
        x=xs[:, None], one=one)[:, 0].double()
    tiny = 2.0 ** (-149 if dtype == torch.float32 else -1074)
    eps = torch.finfo(dtype).eps
    fin = ref.isfinite()
    assert ((got[fin] - ref[fin]).abs() <= 4 * eps * ref[fin] + tiny).all()
    assert torch.equal(got[~fin].isnan(), ref[~fin].isnan()) and torch.equal(got[~fin & ~ref.isnan()], ref[~fin & ~ref.isnan()])
    ys = xs[:4001]  # 0 and terms down to lo
    lse = kp.compile((Vj("y", 1) * Vi("one", 1)).logsumexp(over="j"), dtype=name, backend="cpu")(
        y=ys[:, None], one=one)[0, 0].double()
    want = torch.logsumexp(ys.double(), 0)
    assert abs(lse - want) <= 4 * eps * abs(want)


def test_flush_denormals_leaves_the_threads_clean(threads):
    """FTZ/DAZ hold inside the kernel only: PyTorch shares the OpenMP pool and
    the calling thread, and must still see subnormal numbers afterwards."""
    sub = torch.full((1 << 20,), 1e-30) * 1e-10
    assert (sub > 0).all()
    torch.set_num_threads(4)
    red = (Vi("x", 1) * Vj("y", 1)).sum_over("j")
    run = kp.compile(red, dtype="float32", backend="cpu", flush_denormals=True)
    x = torch.full((4000, 1), 1e-40)  # subnormal input: DAZ reads it as 0
    out = run(x=x, y=torch.ones(3000, 1))
    assert (out == 0).all()
    assert "flush_denormals=ftz+daz" in run.compiled(torch.device("cpu"), 4000, 3000).info.options
    assert torch.equal(torch.full((1 << 20,), 1e-30) * 1e-10, sub)  # main thread and pool restored
    keep = kp.compile(red, dtype="float32", backend="cpu")(x=x, y=torch.ones(3000, 1))
    assert (keep > 0).all()


def test_flush_denormals_refused_without_support(monkeypatch):
    from kernox.backends.cpu import CpuBackend

    monkeypatch.setattr(CpuBackend, "flush_denormals", False)
    run = kp.compile(_cases()["min"], dtype="float32", backend="cpu", flush_denormals=True)
    with pytest.raises(kp.UnsupportedError, match="flush_denormals"):
        run(**{k: v for k, v in _inputs(4, 5, torch.float32).items() if k in run.input_names})


def test_auto_selects_cpu_for_cpu_tensors():
    assert auto_backend_name(torch.device("cpu")) == "cpu"
    red = _cases()["min"]
    run = kp.compile(red, dtype="float32")
    t = {k: v for k, v in _inputs(10, 20, torch.float32).items() if k in run.input_names}
    out = run(**t)
    assert run.compiled(torch.device("cpu"), 10, 20).info.backend == "cpu"
    ref = torch.cdist(t["x"].double(), t["y"].double()).square().min(1, keepdim=True).values
    assert torch.allclose(out.double(), ref, rtol=1e-5, atol=1e-6)


def test_non_x86_64_host_is_refused(monkeypatch):
    import platform

    from kernox.backends.cpu import get_backend

    plan = _cases()["min"].to_plan("float32")
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    s = get_backend().supports(plan, torch.device("cpu"))
    assert not s.ok and "x86-64" in s.reason


@pytest.mark.parametrize("dim,dtype,tile", [(3, "float32", 256), (16, "float32", 256), (16, "float64", 128),
                                            (32, "float32", 128), (32, "float64", 64), (100, "float64", 64)])
def test_default_tile_fits_l1(dim, dtype, tile):
    from kernox.backends.cpu import codegen, get_backend

    x, y, b = Vi("x", dim), Vj("y", dim), Vj("b", 1)
    plan = (kp.exp(-((x - y) ** 2).sum()) * b).sum_over("j").to_plan(dtype)
    assert codegen.default_tile_j(plan) == tile
    assert dict(get_backend().default_variant(plan, torch.device("cpu")).params)["tj"] == tile
