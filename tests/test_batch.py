"""Batch dimensions (v0.6 amendment): plan, runtime, oracle, autograd (CPU)."""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

import kernox as kp  # noqa: E402
from kernox import Param, Vi, Vj  # noqa: E402
from kernox.backends.base import Support  # noqa: E402
from kernox.errors import UnsupportedError, ValidationError  # noqa: E402
from kernox.plan import output_shapes  # noqa: E402
from kernox.runtime import register_backend  # noqa: E402

F64 = torch.float64


def _gauss():
    x, y, b, s = Vi("x", 3), Vj("y", 3), Vj("b", 2), Param("s", 1)
    d2 = ((x - y) ** 2).sum()
    return (kp.exp(-d2 * s[0]) * b).sum_over("j")


def _rand(*shape, seed=0, dtype=F64):
    return torch.randn(*shape, dtype=dtype, generator=torch.Generator().manual_seed(seed))


def _loop(run, B, inputs):
    """Reference: v0 calls element by element (inputs explicitly broadcast)."""
    full = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor) or v.dim() <= 1:
            full[k] = v
        else:
            tail = v.shape[-1:] if k == "s" else v.shape[-2:]
            full[k] = v.expand(*B, *tail)
    outs = []
    for idx in torch.cartesian_prod(*[torch.arange(b) for b in B]).reshape(-1, len(B)).tolist():
        call = {k: (v[tuple(idx)].contiguous() if isinstance(v, torch.Tensor) and v.dim() > 1 else v)
                for k, v in full.items()}
        outs.append(run(**call))
    if isinstance(outs[0], tuple):
        return tuple(torch.stack([o[i] for o in outs]).reshape(*B, *outs[0][i].shape) for i in range(2))
    return torch.stack(outs).reshape(*B, *outs[0].shape)


def _eq(a, b):
    if isinstance(a, tuple):
        return all(_eq(x, y) for x, y in zip(a, b))
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


# ------------------------------------------------------------------ plan
def test_plan_key_and_output_shapes():
    plan = _gauss().to_plan("float32")
    assert plan.batch_rank == 0
    p2 = dataclasses.replace(plan, batch_rank=2)
    assert p2.key() != plan.key()
    assert dataclasses.replace(p2, batch_rank=0).key() == plan.key()
    assert output_shapes(plan, 5) == {"value": ((5, 2), "float32")}
    assert output_shapes(p2, 5, (3, 4)) == {"value": ((3, 4, 5, 2), "float32")}


# ------------------------------------------------------------- broadcasting
CASES = {
    "full": dict(x=(4, 5, 3), y=(4, 7, 3), b=(4, 7, 2), s=(1,)),
    "bcast_x_y": dict(x=(2, 1, 5, 3), y=(1, 3, 7, 3), b=(2, 3, 7, 2), s=(1,)),
    "param_batched": dict(x=(3, 5, 3), y=(1, 7, 3), b=(1, 7, 2), s=(3, 1)),
    "param_bcast": dict(x=(2, 3, 5, 3), y=(2, 3, 7, 3), b=(1, 1, 7, 2), s=(1, 3, 1)),
    "prime_nb": dict(x=(7, 13, 3), y=(7, 11, 3), b=(1, 11, 2), s=(1,)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_batched_equals_loop_of_v0_calls(case):
    run = kp.compile(_gauss(), dtype="float64", backend="oracle")
    shapes = CASES[case]
    t = {k: _rand(*sh, seed=i) for i, (k, sh) in enumerate(shapes.items())}
    t["s"] = t["s"].abs() + 0.3
    out = run(**t)
    B = torch.broadcast_shapes(*(sh[:-2] for k, sh in shapes.items() if k != "s"),
                               *([shapes["s"][:-1]] if len(shapes["s"]) > 1 else []))
    assert out.shape == (*B, shapes["x"][-2], 2)
    assert _eq(out, _loop(run, tuple(B), t))
    # offset tables are cached: a second call adds no new entry
    child = run._batched[len(B)]
    before = len(child._layouts)
    run(**t)
    assert len(child._layouts) == before


def test_strided_and_expanded_batch_dims():
    run = kp.compile(_gauss(), dtype="float64", backend="oracle")
    base = _rand(6, 5, 3)
    X = base[::2]                          # non-standard batch stride
    Y = _rand(1, 7, 3).expand(3, 7, 3)     # stride 0
    Bw = _rand(7, 3, 2, seed=2).transpose(0, 1).contiguous()
    S = torch.tensor([0.5], dtype=F64)
    assert X.stride(0) == 30 and Y.stride(0) == 0
    out = run(x=X, y=Y, b=Bw, s=S)
    want = run(x=X.contiguous(), y=Y.contiguous(), b=Bw, s=S)
    assert _eq(out, want)
    assert _eq(out, _loop(run, (3,), dict(x=X, y=Y, b=Bw, s=S)))


def _all_kinds():
    x, y = Vi("x", 2), Vj("y", 2)
    d2 = ((x - y) ** 2).sum()
    return {
        "sum": (d2 * 1.0).sum_over("j"), "count": (d2 < 1.0).sum_over("j"),
        "min": d2.min(over="j"), "max": d2.max(over="j"),
        "argmin": d2.argmin(over="j"), "argmax": d2.argmax(over="j"),
        "kmin": d2.kmin(3, over="j"), "argkmin": d2.argkmin(3, over="j"),
        "kmin_argkmin": d2.kmin_argkmin(4, over="j"), "argmin_over_i": d2.argmin(over="i"),
        "logsumexp": (-d2).logsumexp(over="j") if hasattr(d2, "logsumexp") else None,
    }


@pytest.mark.parametrize("kind", [k for k, v in _all_kinds().items() if v is not None])
def test_all_reductions_batched_local_indices(kind):
    run = kp.compile(_all_kinds()[kind], dtype="float32", backend="oracle")
    # integer data: ties, indices local to each batch element
    X = torch.randint(-2, 3, (3, 1, 6, 2), generator=torch.Generator().manual_seed(1)).float()
    Y = torch.randint(-2, 3, (1, 2, 5, 2), generator=torch.Generator().manual_seed(2)).float()
    out = run(x=X, y=Y)
    assert _eq(out, _loop(run, (3, 2), dict(x=X, y=Y)))
    for o in out if isinstance(out, tuple) else (out,):
        assert o.shape[:2] == (3, 2)
        if o.dtype == torch.int64 and "count" not in kind:
            limit = 6 if kind.endswith("over_i") else 5
            assert int(o.max()) < limit and int(o.min()) >= -1


@pytest.mark.parametrize("shapes", [
    dict(x=(0, 5, 3), y=(0, 7, 3), b=(0, 7, 2)),     # nb = 0
    dict(x=(2, 0, 3), y=(2, 7, 3), b=(2, 7, 2)),     # M = 0
    dict(x=(2, 5, 3), y=(2, 0, 3), b=(2, 0, 2)),     # N = 0: neutral elements
])
def test_zero_sizes(shapes):
    run = kp.compile(_gauss(), dtype="float64", backend="oracle")
    t = {k: _rand(*sh) for k, sh in shapes.items()}
    out = run(**t, s=0.5)
    assert out.shape == (shapes["x"][0], shapes["x"][1], 2)
    assert torch.equal(out, torch.zeros_like(out))


def test_k_zero_and_empty_batch_argkmin():
    x, y = Vi("x", 2), Vj("y", 2)
    d2 = ((x - y) ** 2).sum()
    run = kp.compile(d2.kmin_argkmin(0, over="j"), dtype="float32", backend="oracle")
    v, i = run(x=torch.randn(2, 3, 2), y=torch.randn(2, 4, 2))
    assert v.shape == (2, 3, 0) and i.shape == (2, 3, 0)


def test_python_param_and_scalar_param_batched():
    run = kp.compile(_gauss(), dtype="float64", backend="oracle")
    t = dict(x=_rand(2, 5, 3), y=_rand(2, 7, 3, seed=1), b=_rand(2, 7, 2, seed=2))
    assert _eq(run(**t, s=0.5), run(**t, s=torch.tensor(0.5, dtype=F64)))


# ------------------------------------------------------------- validation
@pytest.mark.parametrize("bad,msg", [
    (dict(x=(2, 5, 3), y=(7, 3), b=(7, 2)), "rank shared"),
    (dict(x=(2, 5, 3), y=(3, 7, 3), b=(3, 7, 2)), "cannot be broadcast"),
    (dict(x=(2, 5, 3), y=(2, 7, 3), b=(2, 7, 2), s=(2, 2, 1)), r"\(1,\) or \(\*B, 1\)"),
    (dict(x=(2, 5, 3), y=(2, 7, 3), b=(2, 6, 2)), "N=6"),
    (dict(x=(2, 5, 4), y=(2, 7, 3), b=(2, 7, 2)), "instead of"),
])
def test_validation_errors(bad, msg):
    run = kp.compile(_gauss(), dtype="float64", backend="oracle")
    t = {k: _rand(*sh) for k, sh in bad.items()}
    t.setdefault("s", 0.5)
    with pytest.raises(ValidationError, match=msg):
        run(**t)


def test_non_contiguous_last_dims_refused():
    run = kp.compile(_gauss(), dtype="float64", backend="oracle")
    X = _rand(2, 3, 5).transpose(1, 2)  # (2, 5, 3), last dimension not contiguous
    with pytest.raises(ValidationError, match="non-contiguous last two"):
        run(x=X, y=_rand(2, 7, 3), b=_rand(2, 7, 2), s=0.5)


def test_batch_plan_called_without_batch_is_refused():
    plan = dataclasses.replace(_gauss().to_plan("float64"), batch_rank=1)
    run = kp.compile(plan, dtype="float64", backend="oracle")
    with pytest.raises(ValidationError, match=r"\(\*B, L, 3\)"):
        run(x=_rand(5, 3), y=_rand(7, 3), b=_rand(7, 2), s=0.5)
    assert run(x=_rand(2, 5, 3), y=_rand(2, 7, 3), b=_rand(2, 7, 2), s=0.5).shape == (2, 5, 2)


# ---------------------------------- backends without batch: refusal
class _NoBatchBackend:
    """Fake backend that accepts any plan but does not declare ``batch``."""

    name = "nobatch"

    def __init__(self, honest):
        self.honest = honest

    def supports(self, plan, device):
        if self.honest and plan.batch_rank:
            return Support(False, "batch dimensions (v0.6 amendment) not implemented")
        return Support(True)

    def compile(self, plan, device):  # pragma: no cover - never reached
        raise AssertionError("must not compile a batched plan")


@pytest.mark.parametrize("honest", [True, False])
def test_backend_without_batch_is_refused(honest):
    name = f"nobatch_{honest}"
    register_backend(name, lambda: _NoBatchBackend(honest), replace=True)
    run = kp.compile(_gauss(), dtype="float64", backend=name)
    with pytest.raises(UnsupportedError, match="batch"):
        run(x=_rand(2, 5, 3), y=_rand(2, 7, 3), b=_rand(2, 7, 2), s=0.5)
    with pytest.raises(UnsupportedError, match="batch"):  # even without a launch (nb = 0)
        run(x=_rand(0, 5, 3), y=_rand(0, 7, 3), b=_rand(0, 7, 2), s=0.5)


# --------------------------------------------------------------- autograd
def test_batched_gradcheck_with_broadcast():
    run = kp.compile(_gauss(), dtype="float64", backend="oracle")
    X = _rand(2, 1, 4, 3).requires_grad_()
    Y = _rand(1, 3, 5, 3, seed=1).requires_grad_()
    Bw = _rand(2, 3, 5, 2, seed=2).requires_grad_()
    S = (_rand(1, seed=3).abs() + 0.3).requires_grad_()
    f = lambda X, Y, Bw, S: run(x=X, y=Y, b=Bw, s=S)  # noqa: E731
    assert torch.autograd.gradcheck(f, (X, Y, Bw, S))
    assert torch.autograd.gradgradcheck(f, (X, Y, Bw, S))


def test_batched_gradients_match_dense_and_expanded_inputs():
    run = kp.compile(_gauss(), dtype="float64", backend="oracle")
    X = _rand(3, 5, 3).requires_grad_()
    Y0 = _rand(7, 3, seed=1).requires_grad_()
    Y = Y0.expand(3, 7, 3)                         # stride 0: gradient summed over the batch
    Bw = _rand(3, 7, 2, seed=2).requires_grad_()
    S = (_rand(3, 1, seed=3).abs() + 0.3).requires_grad_()   # per-element parameter
    G = _rand(3, 5, 2, seed=4)
    (run(x=X, y=Y, b=Bw, s=S) * G).sum().backward()
    got = [t.grad.clone() for t in (X, Y0, Bw, S)]
    for t in (X, Y0, Bw, S):
        t.grad = None
    D = ((X[:, :, None, :] - Y[:, None, :, :]) ** 2).sum(-1)
    ((torch.exp(-D * S[:, :, None]) @ Bw) * G).sum().backward()
    want = [t.grad for t in (X, Y0, Bw, S)]
    for g, w in zip(got, want):
        assert g.shape == w.shape
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-13)


def test_batch_keeps_sum_scheme_forward_and_gradient():
    x, w = Vi("one", 1), Vj("w", 1)
    run = kp.compile((w * x).sum_over("j"), dtype="float32", backend="oracle", sum_scheme="kahan")
    W = torch.tensor([[1e8, 1.0, 1.0, -1e8], [1.0, 1e8, -1e8, 1.0]])[..., None]
    out = run(one=torch.ones(2, 3, 1), w=W)
    assert out.shape == (2, 3, 1) and out[:, :, 0].tolist() == [[2.0] * 3, [2.0] * 3]
    assert run._batched[1].plan.sum_scheme == "kahan" and run._batched[1].plan.batch_rank == 1
    gp = run._gradient(frozenset({"w"}))[0]
    assert gp.j_plan.sum_scheme == "kahan"
