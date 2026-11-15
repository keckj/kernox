"""M1 : interop HIP/Torch ROCm : runtime partagé, streams, dépendances, durée de vie."""

import gc
import subprocess
import sys
import textwrap

import pytest
import torch
from _cases import SQDIST, make_plan, pair_provider

from kernox.backends.hip import HipBackend, _hip
from kernox.plan import ReductionKind as R

pytestmark = pytest.mark.gpu
DEV = "cuda"
SLEEP_CYCLES = 20_000_000  # ~10 ms : garde le GPU occupé sans charge mémoire


def _ref_sum(x, y):
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1).sum(1, keepdim=True)


def test_single_hip_runtime_shared_with_torch():
    rt = _hip.hip_runtime()
    assert "libamdhip64" in rt.path
    assert _hip.same_runtime_as_torch()
    rtc = _hip.hiprtc()
    import os
    assert os.path.dirname(os.path.realpath(rtc.path)) == os.path.dirname(os.path.realpath(rt.path))
    # le device courant HIP est celui de Torch
    assert rt.get_device() == torch.cuda.current_device()


def test_non_default_stream_dependencies(backend):
    """Entrées produites sur un stream non défaut, consommation sur ce même stream.

    Le stream est d'abord bloqué par un kernel de sommeil, puis les entrées y
    sont écrites : si le kernel était lancé sur un autre stream, il lirait des
    entrées non initialisées (zéros) et le résultat serait faux.
    """
    cr = backend.compile(make_plan(SQDIST, R.SUM, "float64"), DEV)
    xs = torch.randn(500, 3, dtype=torch.float64, device=DEV)
    ys = torch.randn(700, 3, dtype=torch.float64, device=DEV)
    ref = _ref_sum(xs, ys)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        x = torch.zeros_like(xs)
        y = torch.zeros_like(ys)
        out = torch.zeros(500, 1, dtype=torch.float64, device=DEV)
        torch.cuda._sleep(SLEEP_CYCLES)
        x.copy_(xs)
        y.copy_(ys)
        cr.launch({"x": x, "y": y}, {"value": out})
        res = out * 2.0  # consommation de la sortie sur le même stream
    torch.cuda.current_stream().wait_stream(s)
    torch.testing.assert_close(res, 2.0 * ref, rtol=1e-12, atol=1e-12)


def test_cross_stream_event(backend):
    """Producteur sur s1, kernel sur s2 après attente d'événement."""
    cr = backend.compile(make_plan(SQDIST, R.SUM, "float64"), DEV)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(s1):
        torch.cuda._sleep(SLEEP_CYCLES)
        x = torch.randn(300, 3, dtype=torch.float64, device=DEV)
        y = torch.randn(400, 3, dtype=torch.float64, device=DEV)
        ev = torch.cuda.Event()
        ev.record(s1)
    with torch.cuda.stream(s2):
        s2.wait_event(ev)
        out = torch.empty(300, 1, dtype=torch.float64, device=DEV)
        cr.launch({"x": x, "y": y}, {"value": out})
    torch.cuda.synchronize()
    torch.testing.assert_close(out, _ref_sum(x, y), rtol=1e-12, atol=1e-12)


def test_launch_is_async_and_never_syncs(backend):
    """Aucune synchronisation hôte ni copie D2H dans launch (mode debug Torch)."""
    cr = backend.compile(make_plan(SQDIST, R.KMIN_ARGKMIN, "float32", 4), DEV)
    x = torch.randn(1000, 3, device=DEV)
    y = torch.randn(1000, 3, device=DEV)
    v = torch.empty(1000, 4, device=DEV)
    i = torch.empty(1000, 4, dtype=torch.int64, device=DEV)
    torch.cuda.synchronize()
    torch.cuda._sleep(SLEEP_CYCLES)
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        cr.launch({"x": x, "y": y}, {"value": v, "index": i})
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    # le kernel est derrière le sommeil : le stream ne doit pas encore être vide
    assert not torch.cuda.current_stream().query()
    torch.cuda.synchronize()
    d = SQDIST.torch_f({"x": x, "y": y})[..., 0]
    assert torch.equal(i[:, 0], d.argmin(1))


def test_buffer_lifetime_caching_allocator(backend):
    """Entrées libérées par Python juste après launch (kernel encore en file) :
    l'allocateur de Torch ne réutilise ces blocs que dans l'ordre du stream, donc
    l'écrasement des nouveaux tenseurs se produit après la lecture du kernel."""
    cr = backend.compile(make_plan(SQDIST, R.SUM, "float64"), DEV)
    xh = torch.randn(400, 3, dtype=torch.float64)
    yh = torch.randn(900, 3, dtype=torch.float64)
    ref = _ref_sum(xh, yh)
    out = torch.empty(400, 1, dtype=torch.float64, device=DEV)
    torch.cuda._sleep(SLEEP_CYCLES)
    x, y = xh.to(DEV), yh.to(DEV)
    px, py = x.data_ptr(), y.data_ptr()
    cr.launch({"x": x, "y": y}, {"value": out})
    del x, y
    junk = [torch.full((400, 3), 1e300, dtype=torch.float64, device=DEV),
            torch.full((900, 3), 1e300, dtype=torch.float64, device=DEV)]
    reused = {junk[0].data_ptr(), junk[1].data_ptr()} & {px, py}
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-12, atol=1e-12)
    assert reused, "le test n'a pas provoqué de réutilisation mémoire (non discriminant)"


def test_zero_elements_do_not_launch(backend):
    cr = backend.compile(make_plan(SQDIST, R.SUM, "float64"), DEV)
    x = torch.empty(0, 3, dtype=torch.float64, device=DEV)
    y = torch.randn(10, 3, dtype=torch.float64, device=DEV)
    out = torch.empty(0, 1, dtype=torch.float64, device=DEV)
    cr.launch({"x": x, "y": y}, {"value": out})  # M = 0
    assert out.shape == (0, 1)


def test_destruction_and_recompile_from_cache():
    plan = make_plan(SQDIST, R.ARGMIN, "float64")
    b1 = HipBackend(pair_provider=pair_provider)
    cr = b1.compile(plan, DEV)
    key = cr.cache_key
    del cr, b1
    gc.collect()
    b2 = HipBackend(pair_provider=pair_provider)
    cr2 = b2.compile(plan, DEV)  # nouveau backend : cache disque (ou module déjà chargé)
    assert cr2.cache_key == key and cr2.info.cache_hit
    x = torch.randn(50, 3, dtype=torch.float64, device=DEV)
    y = torch.randn(60, 3, dtype=torch.float64, device=DEV)
    i = torch.empty(50, 1, dtype=torch.int64, device=DEV)
    cr2.launch({"x": x, "y": y}, {"index": i})
    torch.cuda.synchronize()
    assert torch.equal(i[:, 0], ((x[:, None] - y[None]) ** 2).sum(-1).argmin(1))


def test_corrupted_cache_entry_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNOX_CACHE_DIR", str(tmp_path))
    from kernox.backends.hip import backend as hb
    plan = make_plan(SQDIST, R.MAX, "float32")
    b = HipBackend(pair_provider=pair_provider)
    cr = b.compile(plan, DEV)
    path = tmp_path / "hip" / (cr.cache_key + ".co")
    assert path.exists() and not cr.info.cache_hit
    path.write_bytes(b"\x7fELF garbage")
    hb._modules.pop((cr.cache_key, torch.cuda.current_device()), None)
    b2 = HipBackend(pair_provider=pair_provider)
    cr2 = b2.compile(plan, DEV)
    assert path.read_bytes()[:4] == b"\x7fELF" and len(path.read_bytes()) > 100
    x = torch.randn(9, 3, device=DEV)
    y = torch.randn(11, 3, device=DEV)
    v = torch.empty(9, 1, device=DEV)
    cr2.launch({"x": x, "y": y}, {"value": v})
    torch.cuda.synchronize()
    assert torch.equal(v[:, 0], SQDIST.torch_f({"x": x, "y": y})[..., 0].amax(1))


def test_concurrent_compilation_processes(tmp_path):
    """Deux processus compilent la même clé : un seul build, artefact complet."""
    code = textwrap.dedent(f"""
        import os, sys, json
        sys.path.insert(0, {repr(str(__import__('pathlib').Path(__file__).parent))})
        os.environ['KERNOX_CACHE_DIR'] = {repr(str(tmp_path))}
        from _cases import SQDIST, make_plan, pair_provider
        from kernox.backends.hip import HipBackend
        from kernox.plan import ReductionKind as R
        cr = HipBackend(pair_provider=pair_provider).compile(make_plan(SQDIST, R.KMIN, 'float64', 11), 'cuda')
        print(json.dumps([cr.cache_key, cr.info.cache_hit]))
    """)
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
             for _ in range(3)]
    res = [__import__("json").loads(p.communicate(timeout=300)[0].strip().splitlines()[-1]) for p in procs]
    assert all(p.returncode == 0 for p in procs)
    assert len({r[0] for r in res}) == 1
    assert sum(1 for r in res if not r[1]) == 1  # une seule compilation effective
    leftovers = [p for p in (tmp_path / "hip").iterdir() if p.suffix == ".tmp"]
    assert not leftovers


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="un seul GPU AMD : multi-GPU non testable")
def test_multi_gpu():  # pragma: no cover
    pass
