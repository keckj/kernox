"""Règles d'exécution du contrat (M1) sur chaque backend GPU présent.

- stream / queue non défaut : entrées produites sur ce stream derrière un long
  calcul, kernel enfilé sur le stream courant, sortie consommée sur ce stream ;
- zéro élément : M = 0, N = 0 (neutres du contrat) et K = 0 ;
- destruction de la réduction et de l'objet compilé pendant qu'un kernel tourne ;
- absence de copie hôte ↔ device pendant l'appel (hook de dispatch Torch, tous
  backends ; profileur d'activité device quand Torch le fournit) ;
- multi-GPU : seulement si Torch voit au moins deux devices du même backend
  (sinon sauté avec motif ; cf. ``docs/conformance.md``).
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
import torch._dynamo  # noqa: F401 - avant tout backend : voir test_dynamo_import_after_backend_load
from torch.utils._python_dispatch import TorchDispatchMode

import kernox as kp
from kernox import Param, Vi, Vj

from ._support import synchronize


# ------------------------------------------------------------------ outils
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
    B = torch.rand(n, 2, generator=g) + 0.1  # poids positifs : norme L1 = valeur
    S = torch.tensor([0.7])
    return {k: v.to(device) for k, v in dict(x=X, y=Y, b=B, s=S).items()}


def _reference(t, rows=None, sigma_tag=0.0):
    """Somme gaussienne FP64 sur CPU (lignes ``rows`` seulement si donné)."""
    X, Y, B, s = (t[k].detach().double().cpu() for k in ("x", "y", "b", "s"))
    if rows is not None:
        X = X[rows]
    out = torch.zeros(X.shape[0], 2, dtype=torch.float64)
    for j0 in range(0, Y.shape[0], 8192):
        d2 = ((X[:, None, :] - Y[None, j0:j0 + 8192, :]) ** 2).sum(-1)
        out += torch.exp(-d2 * s[0] + sigma_tag) @ B[j0:j0 + 8192]
    return out


def _assert_close_l1(got, want, rtol=2e-5):
    # FP32, termes positifs : erreur relative bornée par ~N u ; marge large mais
    # incompatible avec des entrées nulles ou périmées.
    got = got.double().cpu()
    err = (got - want).abs() / want.abs().clamp_min(1e-300)
    assert torch.isfinite(got).all()
    assert err.max().item() <= rtol, err.max().item()


# ------------------------------------------------------- stream non défaut
def test_non_default_stream_dependencies(backend):
    """Le kernel suit les dépendances du stream courant et y produit sa sortie.

    Les entrées sont écrites sur ``s`` derrière ~centaines de ms de calcul :
    un kernel lancé sur un autre stream (ou sans ordre) lirait des zéros.
    """
    bname, dev = backend
    api = _api(dev)
    run = kp.compile(_gauss(), dtype="float32", backend=bname)
    m, n = 4096, 8192
    true = _inputs(m, n, dev)
    run(**true)  # compilation hors de la mesure
    synchronize(dev)

    s = api.Stream(device=dev)
    assert s != api.current_stream(dev)
    with api.stream(s):
        t = {k: torch.zeros_like(v) for k, v in true.items()}
        big = torch.rand(2048, 2048, device=dev)
        for _ in range(40):  # retarde le stream s
            big = torch.tanh(big @ big)
        for k in t:
            t[k].copy_(true[k] + 0.0 * big[0, 0])  # dépendance sur le long calcul
        out = run(**t)
        consumed = out * 2.0  # consommation sur s
        done = api.Event()
        done.record(s)
    assert out.device == torch.device(dev)
    done.synchronize()  # synchronisation de s seulement
    want = _reference(true)
    _assert_close_l1(consumed / 2.0, want)


def test_stream_dependency_on_output_consumer(backend):
    """Sortie consommée sur un second stream après ``wait_stream`` : valeurs finales."""
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


# ------------------------------------------------------------ zéro élément
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
    """N = 0 : somme nulle, gradients nuls (aucune matrice, aucun lancement inutile)."""
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


# ---------------------------------------- destruction pendant l'exécution
def test_destroy_compiled_while_running(backend):
    """``del`` de la réduction et de l'objet compilé pendant le kernel : résultat intact.

    Plan unique (constante propre au test) : aucun autre objet Python du test
    ne partage le module. Les caches internes du backend peuvent le garder en
    vie ; le contrat exige seulement que le kernel enfilé reste valide.
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
    # nouvelle activité sur le device après la destruction
    other = kp.compile(_gauss(), dtype="float32", backend=bname)(**_inputs(128, 256, dev, seed=3))
    synchronize(dev)
    rows = torch.arange(0, m, 509)
    _assert_close_l1(out[rows], _reference(t, rows, sigma_tag=tag))
    _assert_close_l1(other, _reference(_inputs(128, 256, "cpu", seed=3)))
    print(f"kernel encore en cours à la destruction : {still_running} ; "
          f"objet compilé collecté : {None if ref is None else ref() is None}")


# ---------------------------------------------------- aucune copie hôte
class _HostTransfers(TorchDispatchMode):
    """Relève toute op ATen qui fait passer des données entre hôte et device."""

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
    run(**t)  # compilation, sélection de variante, mise en cache du paramètre
    synchronize(dev)

    hook = _HostTransfers()
    with hook:
        run(**t)
    synchronize(dev)
    assert hook.events == [], hook.events

    act = _device_activity(dev)
    if act is None:
        pytest.skip(f"profileur sans activité device pour {dev} (hook ATen vérifié)")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, act]) as prof:
        run(**t)
        synchronize(dev)
    names = [e.name for e in prof.events()]
    copies = [nm for nm in names if any(p in nm.lower() for p in (
        "memcpy htod", "memcpy dtoh", "memcpyhtod", "memcpydtoh", "memcpy host", "copydevicetohost",
        "copyhosttodevice", "_local_scalar_dense"))]
    assert copies == [], copies
    kernels = [nm for nm in names if "kp" in nm.lower() or "kernox" in nm.lower()]
    print(f"profileur {act.name} : {len(names)} évènements, noyaux kernox vus : {sorted(set(kernels))[:4]}")


# ------------------------------------------- cohabitation avec torch.compile
def test_dynamo_import_after_backend_load(backend):
    """Importer ``torch._dynamo`` (donc triton) après un lancement du backend.

    Cas d'un utilisateur qui appelle ``torch.compile`` après Kernox (hip : SIGSEGV
    corrigé par l'ouverture RTLD_LOCAL des bibliothèques HIP, cf. docs/backend-hip.md). Exécuté
    dans un sous-processus : un plantage natif est un échec, pas la fin de la
    suite (le module de test importe ``torch._dynamo`` d'emblée pour cela).
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
        f"code {p.returncode} ; stderr : {p.stderr[-600:]}")


# ------------------------------------------------------------- multi-GPU
def _same_backend_devices(dev):
    return _api(dev).device_count()


def test_multi_gpu_same_backend(backend):
    bname, dev = backend
    count = _same_backend_devices(dev)
    if count < 2:
        pytest.skip(f"{bname} : {count} device visible ; multi-GPU non testable sur cet hôte "
                    "(docs/conformance.md, limite multi-GPU)")
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
