"""Preuve d'interop M1 : queue Torch XPU, USM, ordre, durée de vie, cache."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import time

import pytest
import symox
import torch

from kernox.backends.base import UnsupportedError

from ._helpers import DEV, HOST_BLOCK_ELEMS, alloc_outputs, arg, backend, plan

pytestmark = pytest.mark.gpu


def _gauss_plan(dtype="float32", d=3):
    x, y = arg("x", "i", d), arg("y", "j", d)
    g = symox.exp(-sum((a - b) ** 2 for a, b in zip(x.symbols, y.symbols)))
    return plan([x, y], [g], "sum", dtype)


def _ref(X, Y):
    """Référence par blocs de lignes : mémoire hôte O(bloc·N), jamais O(M·N)."""
    Yd = Y.double()
    bi = max(1, HOST_BLOCK_ELEMS // max(1, Yd.shape[0] * Yd.shape[1]))
    return torch.cat([torch.exp(-((X[i0:i0 + bi].double()[:, None] - Yd[None]) ** 2).sum(-1)).sum(1, keepdim=True)
                      for i0 in range(0, X.shape[0], bi)])


def test_single_sycl_runtime_shared_with_torch():
    c = backend().compile(_gauss_plan(), DEV)
    assert c is not None
    libs = {line.split()[-1] for line in open("/proc/self/maps") if "libsycl.so" in line}
    assert len(libs) == 1, libs  # un seul runtime SYCL : celui de Torch XPU


def test_non_default_stream_and_cross_stream_dependency():
    c = backend().compile(_gauss_plan(), DEV)
    s1, s2 = torch.xpu.Stream(), torch.xpu.Stream()
    assert s1.sycl_queue != torch.xpu.current_stream().sycl_queue
    M, N = 4096, 8192
    with torch.xpu.stream(s1):
        # entrées produites sur s1 par une chaîne d'opérations non triviale
        X = torch.randn(M, 3, device=DEV)
        Y = torch.randn(N, 3, device=DEV)
        for _ in range(20):
            Y = Y * 1.0000001 + 0.0
        done = torch.xpu.Event()
        done.record(s1)
    s2.wait_event(done)
    with torch.xpu.stream(s2):
        out = alloc_outputs(c.plan, M)
        c.launch({"x": X, "y": Y}, out)
        consumed = out["value"] * 2.0  # consommation sur le même stream
    s2.synchronize()
    ref = _ref(X.cpu(), Y.cpu())
    assert torch.allclose(out["value"].double().cpu(), ref, rtol=1e-4)
    assert torch.allclose(consumed.cpu().double(), 2 * ref, rtol=1e-4)


def test_launch_is_asynchronous_no_host_sync():
    c = backend().compile(_gauss_plan(), DEV)
    M, N = 65536, 262144
    X, Y = torch.randn(M, 3, device=DEV), torch.randn(N, 3, device=DEV)
    out = alloc_outputs(c.plan, M)
    c.launch({"x": X, "y": Y}, out)  # échauffement
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    c.launch({"x": X, "y": Y}, out)
    t_launch = time.perf_counter() - t0
    torch.xpu.synchronize()
    t_total = time.perf_counter() - t0
    assert t_total > 0.02, "cas trop petit pour conclure"
    assert t_launch < 0.25 * t_total, (t_launch, t_total)


def test_no_host_device_copy_in_launch():
    """Profileur Torch : la seule activité device de launch est le kernel."""
    from torch.profiler import ProfilerActivity, profile

    c = backend().compile(_gauss_plan(), DEV)
    X, Y = torch.randn(1000, 3, device=DEV), torch.randn(2000, 3, device=DEV)
    out = alloc_outputs(c.plan, 1000)
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        c.launch({"x": X, "y": Y}, out)
        torch.xpu.synchronize()
    names = [e.name for e in prof.events()]
    assert not any("memcpy" in n.lower() or "copy" in n.lower() for n in names), names
    assert any("kp_kernel" in n for n in names), names


def test_input_lifetime_after_launch():
    """Les entrées peuvent être libérées juste après launch (allocateur ordonné par stream)."""
    c = backend().compile(_gauss_plan(), DEV)
    M, N = 2048, 100000
    Xc, Yc = torch.randn(M, 3), torch.randn(N, 3)
    X, Y = Xc.to(DEV), Yc.to(DEV)
    out = alloc_outputs(c.plan, M)
    c.launch({"x": X, "y": Y}, out)
    del X, Y
    junk = [torch.full((N, 3), float("nan"), device=DEV) for _ in range(4)]  # réutilise la mémoire
    torch.xpu.synchronize()
    del junk
    assert torch.allclose(out["value"].double().cpu(), _ref(Xc, Yc), rtol=1e-4)


def test_compiled_object_destruction_and_reload():
    p = _gauss_plan()
    c1 = backend().compile(p, DEV)
    X, Y = torch.randn(10, 3, device=DEV), torch.randn(20, 3, device=DEV)
    out = alloc_outputs(p, 10)
    c1.launch({"x": X, "y": Y}, out)
    del c1
    gc.collect()
    from kernox.backends.sycl import SyclBackend

    c2 = SyclBackend().compile(p, DEV)  # nouveau backend : cache disque
    assert c2.info.cache_hit
    out2 = alloc_outputs(p, 10)
    c2.launch({"x": X, "y": Y}, out2)
    torch.xpu.synchronize()
    assert torch.equal(out["value"], out2["value"])


def test_input_validation_errors():
    c = backend().compile(_gauss_plan(), DEV)
    out = alloc_outputs(c.plan, 4)
    good = torch.randn(4, 3, device=DEV)
    with pytest.raises(ValueError):
        c.launch({"x": torch.randn(4, 3), "y": good}, out)  # CPU
    with pytest.raises(ValueError):
        c.launch({"x": torch.randn(3, 4, device=DEV).T, "y": good}, out)  # non contigu
    with pytest.raises(ValueError):
        c.launch({"x": good.double(), "y": good}, out)  # dtype
    with pytest.raises(ValueError):
        c.launch({"x": good, "y": torch.randn(4, 2, device=DEV)}, out)  # dim
    with pytest.raises(ValueError):
        c.launch({"x": good, "y": good}, alloc_outputs(c.plan, 5))  # forme de sortie


def test_cpu_device_refused():
    assert not backend().supports(_gauss_plan(), "cpu").ok
    with pytest.raises(UnsupportedError):
        backend().compile(_gauss_plan(), "cpu")


def test_fp64_refused_when_device_lacks_it(monkeypatch):
    real = torch.xpu.get_device_properties

    class NoFP64:
        def __init__(self, p):
            self._p = p

        def __getattr__(self, k):
            return False if k == "has_fp64" else getattr(self._p, k)

    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda i=None: NoFP64(real(i)))
    sup = backend().supports(_gauss_plan("float64"), DEV)
    assert not sup.ok and "float64" in sup.reason
    x, y = arg("x", "i", 1), arg("y", "j", 1)
    p = plan([x, y], [x.symbols[0] * y.symbols[0]], "sum", "float32", acc_dtype="float64")
    assert not backend().supports(p, DEV).ok
    assert backend().supports(_gauss_plan("float32"), DEV).ok


def test_fast_math_plan_refused():
    import dataclasses

    p = dataclasses.replace(_gauss_plan(), fast_math=True)
    assert not backend().supports(p, DEV).ok


def test_concurrent_cold_compilation(tmp_path):
    """Deux processus compilent le même plan sur un cache vide : un seul binaire, pas d'artefact partiel."""
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {src!r})
        import torch, symox
        from tests.sycl._helpers import arg, plan, run
        x, y = arg("x", "i", 2), arg("y", "j", 2)
        p = plan([x, y], [symox.cos(x.symbols[0] * y.symbols[1]) + 0.123456], "sum", "float32")
        out, c = run(p, {{"x": torch.ones(3, 2, device="xpu"), "y": torch.ones(5, 2, device="xpu")}})
        print(c.info.cache_hit, float(out["value"][0, 0]))
    """)
    env = dict(os.environ, KERNOX_CACHE_DIR=str(tmp_path))
    procs = [subprocess.Popen([sys.executable, "-c", script], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for _ in range(3)]
    results = [p.communicate(timeout=600) for p in procs]
    for p, (o, e) in zip(procs, results):
        assert p.returncode == 0, e
    lines = [o.split() for o, _ in results]
    assert sum(1 for l in lines if l[0] == "False") == 1  # une seule compilation
    assert len({l[1] for l in lines}) == 1
    files = sorted(os.listdir(tmp_path / "sycl"))
    assert len([f for f in files if f.endswith(".so")]) == 1
    assert not [f for f in files if f.startswith(".")], files  # pas de temporaire résiduel


def test_kernel_names_unique_across_modules():
    """Régression : le runtime SYCL indexe les kernels par nom pour tout le processus.
    Deux plans de même formule mais de réductions différentes, chargés dans le même
    processus, ne doivent pas se confondre (MAX renvoyait le résultat de MIN)."""
    from ._helpers import run

    x, y = arg("x", "i", 1), arg("y", "j", 1)
    e = (x.symbols[0] - y.symbols[0]) ** 2
    X = torch.tensor([[0.0], [5.0]], device=DEV)
    Y = torch.tensor([[1.0], [3.0], [4.0]], device=DEV)
    results = {}
    for kind in ("min", "max", "sum"):
        results[kind] = run(plan([x, y], [e], kind), {"x": X, "y": Y})[0]["value"].flatten().tolist()
    assert results == {"min": [1.0, 1.0], "max": [16.0, 16.0], "sum": [26.0, 21.0]}
