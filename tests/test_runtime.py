import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import kernox as kp
from kernox import Param, Vi, Vj, exp, where
from kernox.backends.base import CompileInfo, Support
from kernox.errors import (BackendUnavailableError, ConstantConversionError, UnsupportedError,
                             ValidationError)
from kernox.oracle import reference_numpy
from kernox.runtime import auto_backend_name, register_backend


def gaussian():
    x, y, w, sigma = Vi("x", 3), Vj("y", 3), Vj("w", 1), Param("sigma")
    d2 = ((x - y) ** 2).sum()
    return x, y, w, sigma, d2, exp(-d2 / (2.0 * sigma ** 2))


def data(m=6, n=9, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(x=torch.randn(m, 3, generator=g, dtype=dtype), y=torch.randn(n, 3, generator=g, dtype=dtype),
                w=torch.randn(n, 1, generator=g, dtype=dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_plan_example_runs_on_oracle(dtype):
    x, y, w, sigma, d2, g = gaussian()
    tdt = getattr(torch, dtype)
    d = data(dtype=tdt)
    run = kp.compile((g * w).sum_over("j"), dtype=dtype, backend="oracle")
    out = run(**d, sigma=0.7)
    assert out.shape == (6, 1) and out.dtype == tdt
    ref = reference_numpy(run.plan, {**d, "sigma": torch.tensor([0.7], dtype=tdt)})["value"]
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5 if dtype == "float32" else 1e-12)

    nb = kp.compile(d2.argkmin(4, over="j"), dtype=dtype, backend="oracle")(x=d["x"], y=d["y"])
    assert nb.shape == (6, 4) and nb.dtype == torch.int64
    cnt = kp.compile((d2 <= 0.25).sum_over("j"), dtype=dtype, backend="oracle")(x=d["x"], y=d["y"])
    assert cnt.dtype == torch.int64
    vals, idx = kp.compile(d2.kmin_argkmin(3, over="i"), dtype=dtype, backend="oracle")(x=d["x"], y=d["y"])
    assert vals.shape == (9, 3) and idx.shape == (9, 3) and int(idx.max()) < 6
    b = kp.compile(where(d2 < 4.5, g, 0.0).max(over="j"), dtype=dtype, backend="oracle")(**{k: d[k] for k in "xy"}, sigma=0.7)
    assert b.shape == (6, 1)


def test_over_i_equals_transposed_over_j():
    x, y = Vi("x", 3), Vj("y", 3)
    d = data()
    a = kp.compile((x - y).sum().argmin(over="i"), dtype="float64", backend="oracle")(x=d["x"], y=d["y"])
    xb, yb = Vi("x", 3), Vj("y", 3)
    b = kp.compile((yb - xb).sum().argmin(over="j"), dtype="float64", backend="oracle")(x=d["y"], y=d["x"])
    assert a.shape == (9, 1)
    assert torch.equal(a, b)


def test_errors_before_launch():
    x, y, w, sigma, d2, g = gaussian()
    run = kp.compile((g * w).sum_over("j"), dtype="float64", backend="oracle")
    d = data()
    with pytest.raises(ValidationError, match="missing arguments"):
        run(x=d["x"], y=d["y"], sigma=1.0)
    with pytest.raises(ValidationError, match="unexpected"):
        run(**d, sigma=1.0, z=1)
    with pytest.raises(ValidationError, match="dtype"):
        run(**{**d, "x": d["x"].float()}, sigma=1.0)
    with pytest.raises(ValidationError, match="contiguous"):
        run(**{**d, "x": torch.randn(3, 6, dtype=torch.float64).t()}, sigma=1.0)
    with pytest.raises(ValidationError, match="shape"):
        run(**{**d, "x": torch.randn(6, 2, dtype=torch.float64)}, sigma=1.0)
    with pytest.raises(ValidationError, match="N="):
        run(**{**d, "w": torch.randn(4, 1, dtype=torch.float64)}, sigma=1.0)
    with pytest.raises(ValidationError, match="device"):
        run(**{**d, "w": torch.empty((9, 1), dtype=torch.float64, device="meta")}, sigma=1.0)
    with pytest.raises(ValidationError):
        run(**d, sigma=torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValidationError):
        run(**d, sigma=True)
    with pytest.raises(ValidationError):
        run(**d, sigma=[1.0, 2.0])
    # SUM is differentiable since v0.4 (tests/test_autograd.py)
    assert run(**{**d, "x": d["x"].clone().requires_grad_()}, sigma=1.0).requires_grad
    with torch.no_grad():
        run(**{**d, "x": d["x"].clone().requires_grad_()}, sigma=1.0)
    run32 = kp.compile((g * w).sum_over("j"), dtype="float32", backend="oracle")
    d32 = data(dtype=torch.float32)
    with pytest.raises(ConstantConversionError):
        run32(**d32, sigma=1e300)


def test_backend_selection_is_explicit():
    x, y = Vi("x"), Vj("y")
    r = (x * y).sum_over("j")
    # auto: native backend of the tensors' device (CPU -> "cpu"), never the oracle
    assert auto_backend_name(torch.device("cpu")) == "cpu"
    with pytest.raises(UnsupportedError, match="oracle"):
        auto_backend_name(torch.device("meta"))
    with pytest.raises(BackendUnavailableError):
        kp.compile(r, dtype="float32", backend="nope")
    with pytest.raises(TypeError):
        kp.compile(x * y, dtype="float32")
    with pytest.raises(TypeError):
        kp.compile(r)  # dtype is required


def test_native_backend_missing_is_explicit():
    import importlib.util
    x, y = Vi("x"), Vj("y")
    for name in ("cuda", "hip", "sycl"):
        if importlib.util.find_spec(f"kernox.backends.{name}") is None:
            with pytest.raises(BackendUnavailableError, match="no fallback"):
                kp.compile((x * y).sum_over("j"), dtype="float32", backend=name)


class FakeCompiled:
    def __init__(self, plan, calls):
        self.plan, self.calls = plan, calls
        self.info = CompileInfo("fake", "cpu", "none", "none", (), False, 0.0, 0, "")

    def launch(self, inputs, outputs):
        self.calls.append((dict(inputs), {k: (tuple(v.shape), v.dtype) for k, v in outputs.items()}))
        for v in outputs.values():
            v.fill_(7)


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.calls, self.compiles = [], 0

    def supports(self, plan, device):
        return Support(plan.dtype == "float32", "float32 only")

    def compile(self, plan, device):
        self.compiles += 1
        return FakeCompiled(plan, self.calls)


def test_registry_validation_and_output_allocation():
    fake = FakeBackend()
    register_backend("fake", lambda: fake, replace=True)
    x, y = Vi("x", 2), Vj("y", 2)
    d2 = ((x - y) ** 2).sum()
    run = kp.compile(d2.kmin_argkmin(5, over="j"), dtype="float32", backend="fake")
    xs, ys = torch.randn(4, 2), torch.randn(3, 2)
    with pytest.raises(ValidationError):
        run(x=xs.double(), y=ys)
    assert fake.calls == []  # no failed validation reaches the backend
    v, i = run(x=xs, y=ys)
    assert fake.calls[-1][1] == {"value": ((4, 5), torch.float32), "index": ((4, 5), torch.int64)}
    run(x=xs, y=ys)
    assert fake.compiles == 1  # per-device cache
    n = len(fake.calls)
    v, i = run(x=xs[:0], y=ys)  # M = 0: no launch
    assert v.shape == (0, 5) and len(fake.calls) == n
    v, i = kp.compile(d2.kmin_argkmin(0, over="j"), dtype="float32", backend="fake")(x=xs, y=ys)
    assert v.shape == (4, 0) and len(fake.calls) == n  # K = 0: no launch
    with pytest.raises(UnsupportedError, match="float32 only"):
        kp.compile(d2.sum_over("j"), dtype="float64", backend="fake")(x=xs.double(), y=ys.double())


def test_empty_inputs_oracle():
    x, y = Vi("x"), Vj("y")
    run = kp.compile((x * y).min(over="j"), dtype="float64", backend="oracle")
    out = run(x=torch.ones(3, 1, dtype=torch.float64), y=torch.ones(0, 1, dtype=torch.float64))
    assert (out == math.inf).all()
    out = run(x=torch.ones(0, 1, dtype=torch.float64), y=torch.ones(4, 1, dtype=torch.float64))
    assert out.shape == (0, 1)


def test_fp_contract_option():
    from kernox import Vi, Vj

    x, y = Vi("x", 1), Vj("y", 1)
    red = (x[0] * y[0] + 1.0).sum_over("j")
    on, off = red.to_plan("float32"), red.to_plan("float32", fp_contract=False)
    assert on.fp_contract is True and off.fp_contract is False and on.key() != off.key()
    assert kp.compile(red, dtype="float32", backend="oracle").plan.fp_contract is True
    run = kp.compile(red, dtype="float32", backend="oracle", fp_contract=False)
    assert run.plan == off
    with pytest.raises(kp.ValidationError):
        kp.compile(red, dtype="float32", fp_contract="off")
    with pytest.raises(kp.ValidationError):
        kp.compile(off, dtype="float32", fp_contract=True)
    assert kp.compile(off, dtype="float32", backend="oracle").plan.fp_contract is False


def test_fp_contract_off_refused_by_backend_without_support():
    import torch

    from kernox import Vi, Vj
    from kernox.backends.base import Support

    class NoOff:
        name = "nooff"

        def supports(self, plan, device):
            return Support(True)

    kp.register_backend("nooff_test", NoOff, replace=True)
    x, y = Vi("x", 1), Vj("y", 1)
    run = kp.compile((x[0] * y[0]).sum_over("j"), dtype="float32", backend="nooff_test", fp_contract=False)
    with pytest.raises(kp.UnsupportedError, match="fp_contract"):
        run(x=torch.ones(2, 1), y=torch.ones(3, 1))


def test_call_caches_keep_validation_and_param_values():
    x, y, s = Vi("x", 1), Vj("y", 1), Param("s")
    run = kp.compile((x[0] * y[0] * s[0]).sum_over("j"), dtype="float64", backend="oracle")
    X, Y = torch.ones(2, 1, dtype=torch.float64), torch.ones(3, 1, dtype=torch.float64)
    assert run(x=X, y=Y, s=0.5)[0, 0].item() == 1.5
    assert run(x=X, y=Y, s=2)[0, 0].item() == 6.0
    assert run(x=X, y=Y, s=0.5)[0, 0].item() == 1.5
    with pytest.raises(ValidationError):
        run(x=X.float(), y=Y, s=0.5)
    with pytest.raises(ValidationError):
        run(x=X, y=Y, s=True)
    with pytest.raises(ValidationError):
        run(x=X, y=torch.ones(3, 2, dtype=torch.float64)[:, :1], s=0.5)  # not contiguous
    with pytest.raises(ValidationError):
        run(x=X, y=Y)
    assert run(x=torch.ones(5, 1, dtype=torch.float64), y=Y, s=0.5).shape == (5, 1)
