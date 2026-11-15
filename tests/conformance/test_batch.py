"""Dimensions de batch (avenant v0.6), sur l'oracle et sur chaque backend GPU.

Un backend qui ne prend pas encore le batch en charge répond
``Support(False)`` : ses cas sont **sautés explicitement** (motif affiché),
jamais simulés. L'oracle (référence Torch, CPU) exécute toujours la suite.
Critère : égalité au bit près avec une boucle d'appels v0 sur le même backend
pour COUNT/MIN/MAX/ARG*/K* ; SUM et LOGSUMEXP dans la tolérance relative à la
norme L1 (ordre de sommation libre) ; dans tous les cas, accord avec l'oracle.
"""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox import Param, Vi, Vj
from kernox.errors import UnsupportedError

from ._support import BACKENDS, backend_status, synchronize


@pytest.fixture(params=("oracle",) + BACKENDS)
def target(request):
    name = request.param
    if name == "oracle":
        return name, "cpu"
    dev, reason = backend_status(name)
    if dev is None:
        pytest.skip(f"backend {name} absent : {reason}")
    return name, dev


def _compile_or_skip(red, dtype, name, probe):
    run = kp.compile(red, dtype=dtype, backend=name)
    try:
        run(**probe)
    except UnsupportedError as exc:
        if "batch" not in str(exc):
            raise
        pytest.skip(f"{name} : {exc}")
    return run


def _formulas():
    x, y, b, s = Vi("x", 3), Vj("y", 3), Vj("b", 2), Param("s", 1)
    d2 = ((x - y) ** 2).sum()
    return {
        "sum_gauss": ((kp.exp(-d2 * s[0]) * b).sum_over("j"), True),
        "sum_over_i": ((kp.exp(-d2 * s[0]) * Vi("v", 2)).sum_over("i"), True),
        "count": ((d2 * s[0] < 2.0).sum_over("j"), False),
        "min": ((d2 * s[0]).min(over="j"), False),
        "argmax": ((d2 * s[0]).argmax(over="j"), False),
        "kmin_argkmin": ((d2 * s[0]).kmin_argkmin(5, over="j"), False),
    }


def _inputs(device, dtype, integer=False):
    g = torch.Generator().manual_seed(7)

    def r(*shape):
        t = torch.randint(-2, 3, shape, generator=g).double() if integer else torch.randn(*shape, generator=g, dtype=torch.float64)
        return t.to(dtype=dtype, device=device)

    return dict(x=r(2, 1, 33, 3), y=r(1, 3, 47, 3), b=r(2, 3, 47, 2), v=r(2, 3, 33, 2),
                s=torch.tensor([0.7], dtype=dtype, device=device))


def _pick(red_names, t):
    return {k: v for k, v in t.items() if k in red_names}


def _loop(run, B, t):
    outs = []
    for i0 in range(B[0]):
        for i1 in range(B[1]):
            call = {k: (v.expand(*B, *v.shape[-2:])[i0, i1].contiguous() if v.dim() == 4 else v) for k, v in t.items()}
            outs.append(run(**call))
    if isinstance(outs[0], tuple):
        return tuple(torch.stack([o[k] for o in outs]).reshape(*B, *outs[0][k].shape) for k in range(2))
    return torch.stack(outs).reshape(*B, *outs[0].shape)


def _flat(o):
    return o if isinstance(o, tuple) else (o,)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", list(_formulas()))
def test_batch_matches_loop_and_oracle(target, name, dtype):
    bname, dev = target
    red, is_sum = _formulas()[name]
    tdt = getattr(torch, dtype)
    t = _inputs(dev, tdt, integer=not is_sum)
    t = _pick(set(kp.compile(red, dtype=dtype, backend="oracle").input_names), t)
    run = _compile_or_skip(red, dtype, bname, t)
    out = run(**t)
    synchronize(dev) if dev != "cpu" else None
    loop = _loop(run, (2, 3), t)
    ref = kp.compile(red, dtype=dtype, backend="oracle")(**{k: v.cpu() for k, v in t.items()})
    for o, lp, rf in zip(_flat(out), _flat(loop), _flat(ref)):
        assert o.shape == rf.shape and o.dtype == rf.dtype and o.device == torch.device(dev)
        if is_sum:
            tol = dict(rtol=1e-4, atol=1e-5) if dtype == "float32" else dict(rtol=1e-11, atol=1e-12)
            torch.testing.assert_close(o.cpu(), lp.cpu(), **tol)
            torch.testing.assert_close(o.cpu(), rf, **tol)
        else:
            assert torch.equal(o.cpu(), lp.cpu()) and torch.equal(o.cpu(), rf)


def test_batch_zero_sizes(target):
    bname, dev = target
    red, _ = _formulas()["sum_gauss"]
    t = _inputs(dev, torch.float32)
    probe = _pick({"x", "y", "b", "s"}, t)
    run = _compile_or_skip(red, "float32", bname, probe)
    for shapes, want in (
        (dict(x=(0, 4, 3), y=(0, 5, 3), b=(0, 5, 2)), (0, 4, 2)),
        (dict(x=(2, 0, 3), y=(2, 5, 3), b=(2, 5, 2)), (2, 0, 2)),
        (dict(x=(2, 4, 3), y=(2, 0, 3), b=(2, 0, 2)), (2, 4, 2)),
    ):
        call = {k: torch.randn(*sh, device=dev) for k, sh in shapes.items()}
        out = run(**call, s=probe["s"])
        assert out.shape == want and torch.equal(out.cpu(), torch.zeros(want))


def test_batch_gradients(target):
    bname, dev = target
    red, _ = _formulas()["sum_gauss"]
    t = _pick({"x", "y", "b", "s"}, _inputs(dev, torch.float64))
    run = _compile_or_skip(red, "float64", bname, t)
    t = {k: v.clone().requires_grad_(True) for k, v in t.items()}
    G = torch.randn(*run(**t).shape, dtype=torch.float64, generator=torch.Generator().manual_seed(3)).to(dev)
    got = torch.autograd.grad((run(**t) * G).sum(), list(t.values()))
    r = {k: v.detach().cpu().requires_grad_(True) for k, v in t.items()}
    D = ((r["x"][..., :, None, :] - r["y"][..., None, :, :]) ** 2).sum(-1)
    want = torch.autograd.grad(((torch.exp(-D * r["s"][0]) @ r["b"]) * G.cpu()).sum(), list(r.values()))
    for k, g, w in zip(t, got, want):
        assert g.shape == t[k].shape, k
        torch.testing.assert_close(g.cpu(), w, rtol=1e-10, atol=1e-11, msg=k)
