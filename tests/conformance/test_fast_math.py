"""Avenant v0.10 : ``fast_math=True`` (exp/log approchés, opt-in).

Référence : FP64 sur CPU (Torch), sur les mêmes entrées FP32. Tolérances
(``u = 2⁻²⁴``) :

- ``exp`` approché (CUDA : ``ex2.approx.f32(x·log₂e)``) : erreur relative
  ≤ 2⁻²¹ (instruction, marge ×3 sur la spécification PTX) + |x|·u (arrondi du
  produit x·log₂e, amplifié par 2^· : ln 2·|x·log₂e|·u = |x|·u) ;
- ``log`` approché (``lg2.approx.f32(x)·ln 2``) : erreur absolue ≤ 2⁻²¹ ;
- sommes de N termes positifs : + N·u (schéma direct, comme les tests IEEE).

D'où, avec X = max |argument d'exp| : ``tol = 8·(2⁻²¹ + X·u) + 2·N·u``,
relative pour les sommes de termes positifs, absolue/(1 + |ref|) pour
LOGSUMEXP ; pour une somme à annulation (log de signe variable) :
``|err| ≤ tol·Σ|terme| + 2·2⁻²¹·Σ|b|``. Les tests IEEE
existants (``fast_math=False``, défaut) ne changent pas : ils restent comparés
au bit près là où ils l'étaient. Un backend qui n'implémente pas v0.10 lève
``UnsupportedError`` (hip, sycl à ce jour) : vérifié, pas sauté en silence.
"""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox import Vi, Vj
from kernox.errors import UnsupportedError

from ._support import synchronize

U = 2.0**-24
M, N, S = 700, 2000, 4.0  # |arguments d'exp| ≤ S·D·1 + |h| ≤ 3·4 + 3


def _data(dev):
    g = torch.Generator().manual_seed(7)
    x = torch.rand(M, 3, generator=g, dtype=torch.float64)
    y = torch.rand(N, 3, generator=g, dtype=torch.float64)
    h = 3 * torch.rand(N, 1, generator=g, dtype=torch.float64) - 1.5
    b = torch.rand(N, 2, generator=g, dtype=torch.float64) + 0.1
    f32 = {k: v.float() for k, v in dict(x=x, y=y, h=h, b=b).items()}
    return f32, {k: v.to(dev).contiguous() for k, v in f32.items()}


def _cases():
    x, y, h, b = Vi("x", 3), Vj("y", 3), Vj("h", 1), Vj("b", 2)
    d2 = ((x - y) ** 2).sum()
    return {
        "gauss": ((kp.exp(-d2 * S) * b).sum_over("j"), "sum"),
        "log": ((kp.log(d2 + 0.5) * b).sum_over("j"), "sum_log"),
        "lse": ((h - d2 * S).logsumexp(over="j"), "lse"),
        "ssmw": ((h - d2 * S).sumsoftmaxweight(b, over="j"), "ssmw"),
    }


def _ref(name, t):
    x, y, h, b = (t[k].double() for k in ("x", "y", "h", "b"))
    d2 = ((x[:, None] - y[None]) ** 2).sum(-1)
    if name == "gauss":
        return torch.exp(-d2 * S) @ b
    if name == "log":
        return torch.log(d2 + 0.5) @ b
    if name == "log_l1":  # Σ|terme| et Σ|b| : bornes des sommes à annulation
        return torch.log(d2 + 0.5).abs() @ b.abs(), b.abs().sum(0, keepdim=True)
    f = h.view(1, -1) - d2 * S
    if name == "lse":
        return torch.logsumexp(f, 1, keepdim=True)
    w = torch.softmax(f, 1)
    return w @ b


TOL = 8 * (2.0**-21 + (3 * S + 1.5) * U) + 2 * N * U


@pytest.mark.parametrize("name", list(_cases()))
def test_fast_math(backend, name):
    bname, dev = backend
    red, _ = _cases()[name]
    f32, t = _data(dev)
    inputs = {a.name: t[a.name] for a in red.to_plan("float32").args}
    try:
        run = red.compile("float32", backend=bname, fast_math=True)
        got = run(**inputs)
    except UnsupportedError:
        assert bname in ("hip", "sycl"), f"{bname} doit implémenter l'avenant v0.10"
        pytest.skip(f"{bname} : fast_math non implémenté (UnsupportedError, avenant v0.10)")
    synchronize(dev)
    got = got.double().cpu()
    ref = _ref(name, f32)
    if name == "lse":
        err = ((got - ref).abs() / (1 + ref.abs())).max().item()
    elif name == "log":  # termes de signes variés : borne absolue en L1 (log : erreur absolue)
        l1, bsum = _ref("log_l1", f32)
        bound = TOL * l1 + 2 * 2.0**-21 * bsum
        err = ((got - ref).abs() / bound).max().item() * TOL
    else:
        err = ((got - ref).abs() / ref.abs().clamp_min(1e-30)).max().item()
    assert err <= TOL, (name, err, TOL)
    if bname == "cuda":  # l'approximation est bien active : sorties distinctes de l'IEEE
        ieee = red.compile("float32", backend=bname)(**inputs).double().cpu()
        assert not torch.equal(ieee, got), name


def test_fast_math_is_opt_in_and_keyed():
    red, _ = _cases()["lse"]
    p0, p1 = red.to_plan("float32"), red.to_plan("float32", fast_math=True)
    assert not p0.fast_math and p1.fast_math and p0.key() != p1.key()
