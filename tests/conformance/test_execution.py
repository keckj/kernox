"""Contract execution rules (M1) on each available GPU backend.

- non-default stream / queue: inputs produced on that stream behind a long
  computation, kernel enqueued on the current stream, output consumed on that stream;
- zero elements: M = 0, N = 0 (contract neutral values) and K = 0;
- destruction of the reduction and of the compiled object while a kernel runs;
- no host ↔ device copy during the call (Torch dispatch hook, all backends;
  device activity profiler when Torch provides one);
- multi-GPU: only if Torch sees at least two devices of the same backend
  (otherwise skipped with a reason; see ``docs/conformance.md``).
"""

from __future__ import annotations

import gc
import math
import os
import subprocess
import sys
import textwrap
import weakref

import pytest
import torch
import torch._dynamo  # noqa: F401 - before any backend: see test_dynamo_import_after_backend_load
from torch.utils._python_dispatch import TorchDispatchMode

import kernox as kp
from kernox import Param, Vi, Vj

from ._support import synchronize


# ------------------------------------------------------------------- tools
def _api(device):
    return torch.xpu if torch.device(device).type == "xpu" else torch.cuda


def _gauss(sigma_tag=0.0):
    x, y, b, s = Vi("x", 3), Vj("y", 3), Vj("b", 2), Param("s")
    d2 = ((x - y) ** 2).sum()
    return (kp.exp(-d2 * s[0] + sigma_tag) * b).sum_over("j")


def _inputs(m, n, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(m, 3, generator=g)
    Y = torch.randn(n, 3, generator=g)
    B = torch.rand(n, 2, generator=g) + 0.1  # positive weights: L1 norm = value
    S = torch.tensor([0.7])
    return {k: v.to(device) for k, v in dict(x=X, y=Y, b=B, s=S).items()}


def _reference(t, rows=None, sigma_tag=0.0):
    """FP64 Gaussian sum on CPU (only rows ``rows`` if given)."""
    X, Y, B, s = (t[k].detach().double().cpu() for k in ("x", "y", "b", "s"))
    if rows is not None:
        X = X[rows]
    out = torch.zeros(X.shape[0], 2, dtype=torch.float64)
    for j0 in range(0, Y.shape[0], 8192):
        d2 = ((X[:, None, :] - Y[None, j0:j0 + 8192, :]) ** 2).sum(-1)
        out += torch.exp(-d2 * s[0] + sigma_tag) @ B[j0:j0 + 8192]
    return out


def _assert_close_l1(got, want, rtol=2e-5):
    # FP32, positive terms: relative error bounded by ~N u; a wide margin, but
    # incompatible with zero or stale inputs.
    got = got.double().cpu()
    err = (got - want).abs() / want.abs().clamp_min(1e-300)
    assert torch.isfinite(got).all()
    assert err.max().item() <= rtol, err.max().item()


# ----------------------------------------------------- non-default stream
def test_non_default_stream_dependencies(backend):
    """The kernel follows the current stream's dependencies and produces its output there.

    The inputs are written on ``s`` behind a few hundred ms of computation:
    a kernel launched on another stream (or unordered) would read zeros.
    """
    bname, dev = backend
    api = _api(dev)
    run = kp.compile(_gauss(), dtype="float32", backend=bname)
    m, n = 4096, 8192
    true = _inputs(m, n, dev)
    run(**true)  # compilation outside the measurement
    synchronize(dev)

    s = api.Stream(device=dev)
    assert s != api.current_stream(dev)
    with api.stream(s):
        t = {k: torch.zeros_like(v) for k, v in true.items()}
        big = torch.rand(2048, 2048, device=dev)
        for _ in range(40):  # delays stream s
            big = torch.tanh(big @ big)
        for k in t:
            t[k].copy_(true[k] + 0.0 * big[0, 0])  # dependency on the long computation
        out = run(**t)
        consumed = out * 2.0  # consumption on s
        done = api.Event()
        done.record(s)
    assert out.device == torch.device(dev)
    done.synchronize()  # synchronize s only
    want = _reference(true)
    _assert_close_l1(consumed / 2.0, want)


def test_stream_dependency_on_output_consumer(backend):
    """Output consumed on a second stream after ``wait_stream``: final values."""
    bname, dev = backend
    api = _api(dev)
    run = kp.compile(_gauss(), dtype="float32", backend=bname)
    t = _inputs(2048, 65536, dev, seed=1)
    run(**t)
    synchronize(dev)
    s1, s2 = api.Stream(device=dev), api.Stream(device=dev)
    with api.stream(s1):
        out = run(**t)
    with api.stream(s2):
        s2.wait_stream(s1)
        consumed = out.clone()
    s2.synchronize()
    _assert_close_l1(consumed, _reference(t))


# ----------------------------------------------------------- zero elements
def _kinds():
    x, y = Vi("x", 3), Vj("y", 3)
    d2 = ((x - y) ** 2).sum()
    f = d2 * 1.0
    return {
        "sum": (f.sum_over("j"), lambda m: [torch.zeros(m, 1)]),
        "count": ((d2 < 1.0).sum_over("j"), lambda m: [torch.zeros(m, 1, dtype=torch.int64)]),
        "min": (f.min(over="j"), lambda m: [torch.full((m, 1), math.inf)]),
        "max": (f.max(over="j"), lambda m: [torch.full((m, 1), -math.inf)]),
        "argmin": (f.argmin(over="j"), lambda m: [torch.full((m, 1), -1, dtype=torch.int64)]),
        "argmax": (f.argmax(over="j"), lambda m: [torch.full((m, 1), -1, dtype=torch.int64)]),
        "kmin": (f.kmin(3, over="j"), lambda m: [torch.full((m, 3), math.inf)]),
        "argkmin": (f.argkmin(3, over="j"), lambda m: [torch.full((m, 3), -1, dtype=torch.int64)]),
        "kmin_argkmin": (f.kmin_argkmin(3, over="j"),
                         lambda m: [torch.full((m, 3), math.inf), torch.full((m, 3), -1, dtype=torch.int64)]),
    }


def _as_list(out):
    return list(out) if isinstance(out, tuple) else [out]


@pytest.mark.parametrize("kind", list(_kinds()))
@pytest.mark.parametrize("m,n", [(0, 17), (17, 0), (0, 0)])
def test_zero_elements(backend, kind, m, n):
    bname, dev = backend
    red, neutral = _kinds()[kind]
    run = kp.compile(red, dtype="float32", backend=bname)
    out = _as_list(run(x=torch.randn(m, 3, device=dev), y=torch.randn(n, 3, device=dev)))
    synchronize(dev)
    want = neutral(m)
    assert len(out) == len(want)
    for o, w in zip(out, want):
        assert o.device == torch.device(dev)
        assert o.shape == w.shape and o.dtype == (w.dtype if w.dtype == torch.int64 else torch.float32)
        assert torch.equal(o.cpu(), w.to(o.dtype)), (kind, o.cpu())


@pytest.mark.parametrize("kind", ["kmin", "argkmin", "kmin_argkmin"])
def test_k_zero(backend, kind):
    bname, dev = backend
    x, y = Vi("x", 3), Vj("y", 3)
    d2 = ((x - y) ** 2).sum()
    run = kp.compile(getattr(d2, kind)(0, over="j"), dtype="float32", backend=bname)
    for o in _as_list(run(x=torch.randn(11, 3, device=dev), y=torch.randn(13, 3, device=dev))):
        assert o.shape == (11, 0) and o.device == torch.device(dev)


def test_zero_elements_gradient(backend):
    """N = 0: zero sum, zero gradients (no matrix, no useless launch)."""
    bname, dev = backend
    run = kp.compile(_gauss(), dtype="float32", backend=bname)
    t = _inputs(9, 0, dev)
    for v in t.values():
        v.requires_grad_(True)
    out = run(**t)
    assert out.shape == (9, 2) and torch.equal(out.detach().cpu(), torch.zeros(9, 2))
    out.sum().backward()
    assert torch.equal(t["x"].grad.cpu(), torch.zeros(9, 3))
    assert t["y"].grad.shape == (0, 3) and t["b"].grad.shape == (0, 2)
    assert torch.equal(t["s"].grad.cpu(), torch.zeros(1))


# ------------------------------------------- destruction during execution
def test_destroy_compiled_while_running(backend):
    """``del`` of the reduction and of the compiled object during the kernel: result intact.

    Unique plan (test-specific constant): no other Python object of the test
    shares the module. The backend's internal caches may keep it alive; the
    contract only requires the enqueued kernel to stay valid.
    """
    bname, dev = backend
    api = _api(dev)
    tag = 1e-3 * (hash(bname) % 997) / 997.0
    run = kp.compile(_gauss(tag), dtype="float32", backend=bname)
    m, n = 32768, 131072
    t = _inputs(m, n, dev, seed=2)
    run(**_inputs(64, 64, dev))
    synchronize(dev)
    comp = run.compiled(torch.device(dev), m, n)
    try:
        ref = weakref.ref(comp)
    except TypeError:
        ref = None
    out = run(**t)
    ev = api.Event()
    ev.record()
    still_running = not ev.query()
    del run, comp
    gc.collect()
    # new device activity after the destruction
    other = kp.compile(_gauss(), dtype="float32", backend=bname)(**_inputs(128, 256, dev, seed=3))
    synchronize(dev)
    rows = torch.arange(0, m, 509)
    _assert_close_l1(out[rows], _reference(t, rows, sigma_tag=tag))
    _assert_close_l1(other, _reference(_inputs(128, 256, "cpu", seed=3)))
    print(f"kernel still running at destruction: {still_running}; "
          f"compiled object collected: {None if ref is None else ref() is None}")


# ------------------------------------------------------- no host copy
class _HostTransfers(TorchDispatchMode):
    """Records every ATen op that moves data between host and device."""

    def __init__(self):
        super().__init__()
        self.events = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        ins = [a for a in torch.utils._pytree.tree_leaves((args, kwargs)) if isinstance(a, torch.Tensor)]
        outs = [o for o in torch.utils._pytree.tree_leaves(out) if isinstance(o, torch.Tensor)]
        devs = {t.device.type for t in ins + outs}
        name = str(func)
        if "_local_scalar_dense" in name or ("cpu" in devs and len(devs) > 1):
            self.events.append(name)
        return out


def _device_activity(device):
    acts = torch.profiler.supported_activities()
    want = {"cuda": "CUDA", "xpu": "XPU"}[torch.device(device).type]
    return next((a for a in acts if a.name == want), None)


@pytest.mark.parametrize("params", ["tensor", "python_cached"])
def test_no_host_copy_during_call(backend, params):
    bname, dev = backend
    run = kp.compile(_gauss(), dtype="float32", backend=bname)
    t = _inputs(1024, 2048, dev)
    if params == "python_cached":
        t["s"] = 0.7
    run(**t)  # compilation, variant selection, parameter caching
    synchronize(dev)

    hook = _HostTransfers()
    with hook:
        run(**t)
    synchronize(dev)
    assert hook.events == [], hook.events

    act = _device_activity(dev)
    if act is None:
        pytest.skip(f"profiler has no device activity for {dev} (ATen hook checked)")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, act]) as prof:
        run(**t)
        synchronize(dev)
    names = [e.name for e in prof.events()]
    copies = [nm for nm in names if any(p in nm.lower() for p in (
        "memcpy htod", "memcpy dtoh", "memcpyhtod", "memcpydtoh", "memcpy host", "copydevicetohost",
        "copyhosttodevice", "_local_scalar_dense"))]
    assert copies == [], copies
    kernels = [nm for nm in names if "kp" in nm.lower() or "kernox" in nm.lower()]
    print(f"profiler {act.name}: {len(names)} events, kernox kernels seen: {sorted(set(kernels))[:4]}")


# ------------------------------------------ coexistence with torch.compile
def test_dynamo_import_after_backend_load(backend):
    """Import ``torch._dynamo`` (hence triton) after a backend launch.

    Case of a user calling ``torch.compile`` after Kernox (hip: SIGSEGV fixed
    by opening the HIP libraries with RTLD_LOCAL, see docs/backend-hip.md). Run
    in a subprocess: a native crash is a failure, not the end of the suite
    (the test module imports ``torch._dynamo`` upfront for that reason).
    """
    bname, dev = backend
    code = textwrap.dedent(f"""
        import torch, kernox as kp
        from kernox import Vi, Vj
        x, y = Vi("x", 3), Vj("y", 3)
        r = kp.compile(((x - y) ** 2).sum().sum_over("j"), dtype="float32", backend={bname!r})
        out = r(x=torch.randn(10, 3, device={dev!r}), y=torch.randn(20, 3, device={dev!r}))
        import torch._dynamo
        print("ok", tuple(out.shape))
    """)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=env)
    assert p.returncode == 0 and "ok (10, 1)" in p.stdout, (
        f"code {p.returncode}; stderr: {p.stderr[-600:]}")


# ------------------------------------------------------------- multi-GPU
def _same_backend_devices(dev):
    return _api(dev).device_count()


def test_multi_gpu_same_backend(backend):
    bname, dev = backend
    count = _same_backend_devices(dev)
    if count < 2:
        pytest.skip(f"{bname}: {count} visible device; multi-GPU cannot be tested on this host "
                    "(docs/conformance.md, multi-GPU limitation)")
    kind = torch.device(dev).type
    run = kp.compile(_gauss(), dtype="float32", backend=bname)
    outs = {}
    for i in (count - 1, 0):
        d = f"{kind}:{i}"
        t = _inputs(512, 1024, d, seed=4)
        outs[i] = run(**t)
        assert outs[i].device == torch.device(d)
    for i, o in outs.items():
        synchronize(o.device)
        _assert_close_l1(o, _reference(_inputs(512, 1024, "cpu", seed=4)))
    t = _inputs(8, 8, f"{kind}:0")
    t["y"] = t["y"].to(f"{kind}:1")
    with pytest.raises(kp.ValidationError if hasattr(kp, "ValidationError") else Exception):
        run(**t)
