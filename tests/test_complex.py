"""Formules complexes (parité PyKeOps ``ComplexLazyTensor`` / keopscore ``formulas/complex``),
oracle contre l'arithmétique complexe dense de Torch."""

from __future__ import annotations

import pytest
import torch

import kernox as kp
from kernox.errors import ValidationError

C128 = torch.complex128


def _data(seed=0, m=7, n=9, d=2):
    g = torch.Generator().manual_seed(seed)

    def c(*shape):
        return torch.complex(torch.randn(*shape, dtype=torch.float64, generator=g),
                             torch.randn(*shape, dtype=torch.float64, generator=g))

    return dict(z=c(m, d), w=c(n, d), u=c(n, 1), x=torch.randn(m, d, dtype=torch.float64, generator=g),
                r=torch.rand(n, 1, dtype=torch.float64, generator=g) + 0.5)


def _sym():
    return dict(z=kp.Vi("z", 2, complex=True), w=kp.Vj("w", 2, complex=True), u=kp.Vj("u", 1, complex=True),
                x=kp.Vi("x", 2), r=kp.Vj("r", 1))


def _dense(t):
    """Vues (M, N, D) des entrées pour la référence dense."""
    return dict(z=t["z"][:, None, :], w=t["w"][None, :, :], u=t["u"][None, :, :], x=t["x"][:, None, :],
                r=t["r"][None, :, :])


# (formule kernox, référence Torch) ; sortie complexe si la formule l'est
CASES = {
    "mult": (lambda s: s["z"] * s["w"], lambda t: t["z"] * t["w"]),
    "add_sub": (lambda s: s["z"] + s["w"] - s["z"] * 0.5, lambda t: t["z"] + t["w"] - t["z"] * 0.5),
    "divide": (lambda s: s["z"] / s["w"], lambda t: t["z"] / t["w"]),
    "conj": (lambda s: s["z"].conj() * s["w"], lambda t: t["z"].conj() * t["w"]),
    "exp": (lambda s: kp.exp(s["z"] * s["w"] * 0.3), lambda t: torch.exp(t["z"] * t["w"] * 0.3)),
    "exp1j": (lambda s: s["x"].exp1j() * s["w"], lambda t: torch.exp(1j * t["x"]) * t["w"]),
    "real2complex": (lambda s: s["x"].real2complex() + kp.imag2complex(s["x"] * 2.0) * s["w"],
                     lambda t: t["x"] + 1j * (2.0 * t["x"]) * t["w"]),
    "real_scal": (lambda s: s["x"][0] * s["w"] + s["z"] * s["r"],
                  lambda t: t["x"][..., 0:1] * t["w"] + t["z"] * t["r"]),
    "real_dim_d": (lambda s: s["x"] * s["w"], lambda t: t["x"] * t["w"]),
    "broadcast_dim1": (lambda s: s["u"] * s["z"], lambda t: t["u"] * t["z"]),
    "complex_const": (lambda s: (2 - 3j) * s["z"] + s["w"] / 1j, lambda t: (2 - 3j) * t["z"] + t["w"] / 1j),
    "component_sum": (lambda s: (s["z"] * s["w"]).sum(), lambda t: (t["z"] * t["w"]).sum(-1, keepdim=True)),
    "component_index": (lambda s: s["z"][1] * s["w"][-1], lambda t: t["z"][..., 1:2] * t["w"][..., 1:2]),
    "complex_formula": (lambda s: kp.complex_formula(s["x"], s["x"] * s["r"]),
                        lambda t: torch.complex(t["x"], t["x"] * t["r"]).expand(-1, t["r"].shape[1], -1)),
    # sorties réelles
    "real": (lambda s: (s["z"] * s["w"]).real, lambda t: (t["z"] * t["w"]).real),
    "imag": (lambda s: kp.imag(s["z"] / s["w"]), lambda t: (t["z"] / t["w"]).imag),
    "abs": (lambda s: abs(s["z"] - s["w"]), lambda t: (t["z"] - t["w"]).abs()),
    "kp_abs": (lambda s: kp.abs(s["z"] * s["w"]), lambda t: (t["z"] * t["w"]).abs()),
    "angle": (lambda s: kp.angle(s["z"] * s["w"]), lambda t: (t["z"] * t["w"]).angle()),
    "square_abs": (lambda s: s["z"].square_abs() * s["r"], lambda t: t["z"].abs() ** 2 * t["r"]),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("over", ["j", "i"])
def test_complex_ops_against_torch(name, over):
    build, ref = CASES[name]
    t = _data()
    f = build(_sym())
    run = kp.compile(f.sum_over(over), dtype="float64", backend="oracle")
    out = run(**{k: v for k, v in t.items() if k in run.input_names})
    dense = ref(_dense(t))
    want = dense.sum(dim=1 if over == "j" else 0)
    assert out.is_complex() == getattr(f, "is_complex", False)
    torch.testing.assert_close(out, want, rtol=1e-12, atol=1e-12)


def test_interleaved_real_inputs_are_equivalent():
    t = _data(1)
    s = _sym()
    run = kp.compile((s["z"] * s["w"]).sum_over("j"), dtype="float64", backend="oracle")
    a = run(z=t["z"], w=t["w"])
    b = run(z=torch.view_as_real(t["z"]).reshape(7, 4), w=torch.view_as_real(t["w"]).reshape(9, 4))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert run.plan.meta["complex_args"] == ("w", "z") and [a.dim for a in run.plan.args] == [4, 4]
    # vue conjuguée paresseuse de Torch : résolue
    torch.testing.assert_close(run(z=t["z"].conj(), w=t["w"]), run(z=t["z"].conj().resolve_conj(), w=t["w"]))


def test_complex_float32_and_params():
    t = _data(2)
    z, w, p = kp.Vi("z", 2, complex=True), kp.Vj("w", 2, complex=True), kp.Param("p", 1, complex=True)
    run = kp.compile((z * w * p).sum_over("j"), dtype="float32", backend="oracle")
    out = run(z=t["z"].to(torch.complex64), w=t["w"].to(torch.complex64), p=0.5 - 2j)
    assert out.dtype == torch.complex64
    want = ((t["z"][:, None] * t["w"][None]) * (0.5 - 2j)).sum(1)
    torch.testing.assert_close(out.to(C128), want, rtol=1e-5, atol=1e-5)
    pt = torch.tensor([0.5 - 2j], dtype=torch.complex64)
    torch.testing.assert_close(run(z=t["z"].to(torch.complex64), w=t["w"].to(torch.complex64), p=pt), out)
    with pytest.raises(ValidationError, match="dtype"):
        run(z=t["z"], w=t["w"], p=0.5)  # complex128 pour un plan FP32


def test_complex_gradients_match_torch():
    t = _data(3)
    s = _sym()
    f = kp.exp(s["z"] * s["w"] * 0.2) / (s["w"] + 3.0) + s["x"].exp1j() * s["z"].conj()
    run = kp.compile(f.sum_over("j"), dtype="float64", backend="oracle")
    g = torch.Generator().manual_seed(9)
    a, b = torch.randn(7, 2, dtype=torch.float64, generator=g), torch.randn(7, 2, dtype=torch.float64, generator=g)
    leaves = {k: t[k].clone().requires_grad_() for k in ("z", "w", "x")}
    out = run(**leaves)
    got = torch.autograd.grad((out.real * a + out.imag * b).sum(), list(leaves.values()))
    lr = {k: t[k].clone().requires_grad_() for k in ("z", "w", "x")}
    d = _dense({**t, **lr})
    ref = (torch.exp(d["z"] * d["w"] * 0.2) / (d["w"] + 3.0) + torch.exp(1j * d["x"]) * d["z"].conj()).sum(1)
    want = torch.autograd.grad((ref.real * a + ref.imag * b).sum(), list(lr.values()))
    for x, y in zip(got, want):
        torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-11)


def test_complex_refusals():
    s = _sym()
    z, w = s["z"], s["w"]
    for red in ("min", "max", "argmin", "logsumexp"):
        with pytest.raises(TypeError, match="sum_over"):
            getattr(z * w, red)(over="j")
    with pytest.raises(ValidationError, match="real2complex"):
        z.real2complex()
    with pytest.raises(ValidationError, match="exp1j"):
        kp.exp1j(z)
    with pytest.raises(TypeError, match="complexe"):
        kp.sin(z)
    with pytest.raises(TypeError, match="puissance"):
        z ** 2
    with pytest.raises(ValidationError, match="incompatibles"):
        (z * kp.Vj("w", 4)).sum_over("j")  # même nom, réel et complexe
    with pytest.raises(ValidationError, match="dimensions complexes"):
        z * kp.Vj("v", 3, complex=True)
    with pytest.raises(ValidationError, match="dimension complexe"):
        kp.compile((z * w).sum_over("j"), dtype="float64", backend="oracle")(
            z=torch.zeros(3, 3, dtype=C128), w=torch.zeros(4, 2, dtype=C128))
