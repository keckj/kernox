"""Block-sparse ranges (v0.6 amendment §2): Ranges, validation, runtime, oracle, autograd (CPU)."""

from __future__ import annotations

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")

import kernox as kp  # noqa: E402
from kernox import Ranges, Vi, Vj, ranges_from_labels  # noqa: E402
from kernox.backends.base import Support  # noqa: E402
from kernox.errors import UnsupportedError, ValidationError  # noqa: E402
from kernox.ranges import Triplet, work_list  # noqa: E402
from kernox.runtime import register_backend  # noqa: E402

F64 = torch.float64


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _clusters(m, n, ci, cj, density, seed=0):
    g = _gen(seed)
    li = torch.sort(torch.randint(0, ci, (m,), generator=g)).values
    lj = torch.sort(torch.randint(0, cj, (n,), generator=g)).values
    keep = torch.rand(ci, cj, generator=g) < density
    return li, lj, keep, keep[li][:, lj]


def _formulas():
    x, y, b = Vi("x", 2), Vj("y", 2), Vj("b", 2)
    d2 = ((x - y) ** 2).sum()
    return {
        "sum": (kp.exp(-d2) * b).sum_over("j"),
        "count": (d2 < 2.0).sum_over("j"),
        "min": d2.min(over="j"),
        "argmax": d2.argmax(over="j"),
        "kmin_argkmin": d2.kmin_argkmin(4, over="j"),
        "logsumexp": (-d2).logsumexp(over="j"),
    }


def _dense_reference(kind, X, Y, Bw, mask):
    """Masked dense reference (FP64), contract semantics (neutral elements, smallest j)."""
    D = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    inf = math.inf
    if kind == "sum":
        return (torch.exp(-D) * mask) @ Bw
    if kind == "count":
        return ((D < 2.0) & mask).sum(1, keepdim=True)
    if kind == "min":
        return torch.where(mask, D, torch.full_like(D, inf)).amin(1, keepdim=True)
    if kind == "argmax":
        Dm = torch.where(mask, D, torch.full_like(D, -inf))
        idx = Dm.argmax(1)  # first maximum
        return torch.where(mask.any(1), idx, torch.full_like(idx, -1))[:, None]
    if kind == "kmin_argkmin":
        Dm = torch.where(mask, D, torch.full_like(D, inf))
        key = torch.argsort(Dm, dim=1, stable=True)[:, :4]
        vals = Dm.gather(1, key)
        idx = torch.where(torch.isinf(vals), torch.full_like(key, -1), key)
        return vals, idx
    if kind == "logsumexp":
        return torch.logsumexp(torch.where(mask, -D, torch.full_like(D, -inf)), dim=1, keepdim=True)
    raise ValueError(kind)


def _inputs(m, n, integer, seed=1):
    g = _gen(seed)
    if integer:
        X = torch.randint(-2, 3, (m, 2), generator=g).to(F64)
        Y = torch.randint(-2, 3, (n, 2), generator=g).to(F64)
    else:
        X, Y = torch.randn(m, 2, dtype=F64, generator=g), torch.randn(n, 2, dtype=F64, generator=g)
    return X, Y, torch.randn(n, 2, dtype=F64, generator=g)


# ------------------------------------------------------------------ objects
def test_plan_key_and_ranges_object():
    plan = _formulas()["sum"].to_plan("float32")
    assert plan.ranges is False
    assert dataclasses.replace(plan, ranges=True).key() != plan.key()
    r = Ranges.dense(3, 4)
    assert r.has_transpose and r.slices_i.tolist() == [1]
    with pytest.raises(ValidationError, match="complete or absent"):
        Ranges(r.ranges_i, r.slices_i, r.redranges_j, ranges_j=r.ranges_j)
    with pytest.raises(ValidationError, match="int32 or int64"):
        Ranges(r.ranges_i.float(), r.slices_i, r.redranges_j)
    with pytest.raises(ValidationError, match=r"\(C, 2\)"):
        Ranges(r.ranges_i.reshape(-1), r.slices_i, r.redranges_j)
    with pytest.raises(ValidationError, match="kernox.Ranges"):
        kp.compile(_formulas()["sum"], dtype="float64", backend="oracle").with_ranges((1, 2, 3))


def test_work_list_tiles_and_intervals():
    tr = Triplet(torch.tensor([[0, 5], [5, 5], [10, 2510]]), torch.tensor([2, 2, 3]),
                 torch.tensor([[0, 3], [7, 9], [1, 2]]))
    w = work_list(tr, 1024).tolist()
    assert w == [[0, 5, 0, 2], [10, 1034, 2, 3], [1034, 2058, 2, 3], [2058, 2510, 2, 3]]


def test_ranges_from_labels_matches_mask():
    li, lj, keep, mask = _clusters(40, 50, 4, 5, 0.5)
    r = ranges_from_labels(li, lj, keep)
    got = torch.zeros(40, 50, dtype=torch.bool)
    for c, (a, b) in enumerate(r.ranges_i.tolist()):
        k0 = 0 if c == 0 else int(r.slices_i[c - 1])
        for cc, d in r.redranges_j[k0:int(r.slices_i[c])].tolist():
            got[a:b, cc:d] = True
    assert torch.equal(got, mask)
    gotT = torch.zeros(50, 40, dtype=torch.bool)
    for c, (a, b) in enumerate(r.ranges_j.tolist()):
        k0 = 0 if c == 0 else int(r.slices_j[c - 1])
        for cc, d in r.redranges_i[k0:int(r.slices_j[c])].tolist():
            gotT[a:b, cc:d] = True
    assert torch.equal(gotT, mask.T)
    with pytest.raises(ValidationError, match="non-decreasing"):
        ranges_from_labels(torch.tensor([1, 0]), lj, keep)


# --------------------------------------------------------------- semantics
@pytest.mark.parametrize("kind", list(_formulas()))
def test_dense_ranges_equal_plain_call(kind):
    run = kp.compile(_formulas()[kind], dtype="float64", backend="oracle")
    X, Y, Bw = _inputs(37, 53, integer=kind != "sum")
    kw = dict(x=X, y=Y, **({"b": Bw} if "b" in run.input_names else {}))
    plain = run(**kw)
    ranged = run.with_ranges(Ranges.dense(37, 53), validate=True)(**kw)
    for p, q in zip(plain if isinstance(plain, tuple) else (plain,), ranged if isinstance(ranged, tuple) else (ranged,)):
        assert torch.equal(p, q)


@pytest.mark.parametrize("density", [1.0, 0.5, 0.05, 0.0])
@pytest.mark.parametrize("kind", list(_formulas()))
def test_label_masks_match_dense_reference(kind, density):
    m, n = 300, 2600  # blocks taller than the tile; intervals longer than 1024
    li, lj, keep, mask = _clusters(m, n, 3, 4, density, seed=3)
    X, Y, Bw = _inputs(m, n, integer=kind not in ("sum", "logsumexp"))
    run = kp.compile(_formulas()[kind], dtype="float64", backend="oracle")
    kw = dict(x=X, y=Y, **({"b": Bw} if "b" in run.input_names else {}))
    got = run.with_ranges(ranges_from_labels(li, lj, keep), validate=True)(**kw)
    want = _dense_reference(kind, X, Y, Bw, mask)
    if kind in ("sum", "logsumexp"):
        torch.testing.assert_close(got, want.to(got.dtype), rtol=1e-12, atol=1e-12)
    elif isinstance(want, tuple):
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    else:
        assert torch.equal(got, want.to(got.dtype))


def test_uncovered_rows_empty_blocks_and_unordered_intervals():
    # rows 0-3 not covered; empty block [4, 4); intervals out of order
    r = Ranges(torch.tensor([[4, 4], [4, 9]]), torch.tensor([0, 2]), torch.tensor([[6, 10], [0, 3]]))
    run = kp.compile(_formulas()["argmax"], dtype="float64", backend="oracle")
    X = torch.zeros(9, 2, dtype=F64)
    Y = torch.ones(10, 2, dtype=F64)  # all distances equal: smallest j of J = 0
    out = run.with_ranges(r, validate=True)(x=X, y=Y)
    assert out[:, 0].tolist() == [-1] * 4 + [0] * 5
    s = kp.compile(_formulas()["sum"], dtype="float64", backend="oracle").with_ranges(r)
    assert torch.equal(s(x=X, y=Y, b=torch.ones(10, 2, dtype=F64))[:4], torch.zeros(4, 2, dtype=F64))


def test_k_larger_than_reduced_set_completes():
    r = Ranges(torch.tensor([[0, 3]]), torch.tensor([1]), torch.tensor([[2, 4]]))
    run = kp.compile(_formulas()["kmin_argkmin"], dtype="float64", backend="oracle").with_ranges(r)
    v, i = run(x=torch.zeros(3, 2, dtype=F64), y=torch.arange(10, dtype=F64)[:, None].repeat(1, 2))
    assert i.tolist() == [[2, 3, -1, -1]] * 3 and torch.isinf(v[:, 2:]).all()


def test_sizes_zero():
    run = kp.compile(_formulas()["sum"], dtype="float64", backend="oracle")
    r = Ranges(torch.zeros(0, 2, dtype=torch.int64), torch.zeros(0, dtype=torch.int64), torch.zeros(0, 2, dtype=torch.int64))
    assert run.with_ranges(r)(x=torch.zeros(0, 2, dtype=F64), y=torch.ones(5, 2, dtype=F64),
                              b=torch.ones(5, 2, dtype=F64)).shape == (0, 2)
    out = run.with_ranges(r, validate=True)(x=torch.ones(3, 2, dtype=F64), y=torch.ones(5, 2, dtype=F64),
                                            b=torch.ones(5, 2, dtype=F64))
    assert torch.equal(out, torch.zeros(3, 2, dtype=F64))


def test_validate_accepts_empty_blocks_and_intervals():
    """Empty blocks and intervals overlap nothing (kernox.distributed clips ranges to spans)."""
    run = kp.compile(_formulas()["sum"], dtype="float64", backend="oracle")
    X, Y, Bw = _inputs(10, 20, integer=False)
    r = Ranges(torch.tensor([[0, 5], [0, 0], [3, 3], [5, 10]]), torch.tensor([3, 4, 4, 5]),
               torch.tensor([[0, 4], [4, 4], [2, 2], [0, 20], [5, 9]]))
    dense = Ranges(torch.tensor([[0, 5], [5, 10]]), torch.tensor([1, 2]), torch.tensor([[0, 4], [5, 9]]))
    out = run.with_ranges(r, validate=True)(x=X, y=Y, b=Bw)
    assert torch.equal(out, run.with_ranges(dense, validate=True)(x=X, y=Y, b=Bw))


@pytest.mark.parametrize("bad,msg", [
    (Ranges(torch.tensor([[0, 5], [3, 8]]), torch.tensor([1, 2]), torch.tensor([[0, 2], [0, 2]])), "overlapping row blocks"),
    (Ranges(torch.tensor([[0, 5]]), torch.tensor([2]), torch.tensor([[0, 4], [3, 6]])), "within a block"),
    (Ranges(torch.tensor([[0, 11]]), torch.tensor([1]), torch.tensor([[0, 2]])), "outside"),
    (Ranges(torch.tensor([[0, 5]]), torch.tensor([1]), torch.tensor([[0, 30]])), "outside"),
    (Ranges(torch.tensor([[0, 5]]), torch.tensor([3]), torch.tensor([[0, 2]])), "slices"),
])
def test_validate_rejects_bad_ranges(bad, msg):
    run = kp.compile(_formulas()["sum"], dtype="float64", backend="oracle")
    X, Y, Bw = _inputs(10, 20, integer=False)
    with pytest.raises(ValidationError, match=msg):
        run.with_ranges(bad, validate=True)(x=X, y=Y, b=Bw)


@pytest.mark.parametrize("bad", [
    Ranges(torch.tensor([[0, 11]]), torch.tensor([1]), torch.tensor([[0, 2]])),        # rows > M
    Ranges(torch.tensor([[-1, 5]]), torch.tensor([1]), torch.tensor([[0, 2]])),        # rows < 0
    Ranges(torch.tensor([[0, 5]]), torch.tensor([1]), torch.tensor([[0, 30]])),        # j > N
    Ranges(torch.tensor([[0, 5]]), torch.tensor([1]), torch.tensor([[-4, 2]])),        # j < 0
    Ranges(torch.tensor([[0, 5]]), torch.tensor([3]), torch.tensor([[0, 2]])),         # slices > R
    Ranges(torch.tensor([[0, 5]]), torch.tensor([1]), torch.tensor([[0, 2**62]])),     # endless loop
])
def test_bounds_checked_without_validate(bad):
    """Bounds checking is always on (a kernel must never read outside the tensors)."""
    run = kp.compile(_formulas()["sum"], dtype="float64", backend="oracle")
    X, Y, Bw = _inputs(10, 20, integer=False)
    with pytest.raises(ValidationError, match="out of bounds"):
        run.with_ranges(bad)(x=X, y=Y, b=Bw)


def test_bounds_ignore_empty_tiles_and_intervals():
    """Empty interval (d ≤ c) and empty block: never traversed, hence accepted."""
    run = kp.compile(_formulas()["sum"], dtype="float64", backend="oracle")
    X, Y, Bw = _inputs(10, 20, integer=False)
    r = Ranges(torch.tensor([[0, 5], [7, 7]]), torch.tensor([2, 2]), torch.tensor([[0, 3], [99, 99]]))
    mask = torch.zeros(10, 20, dtype=torch.bool)
    mask[:5, :3] = True
    want = _dense_reference("sum", X, Y, Bw, mask)
    torch.testing.assert_close(run.with_ranges(r)(x=X, y=Y, b=Bw), want)


def test_over_i_uses_transposed_triplet():
    li, lj, keep, mask = _clusters(30, 40, 3, 4, 0.5, seed=5)
    r = ranges_from_labels(li, lj, keep)
    x, y, v = Vi("x", 2), Vj("y", 2), Vi("v", 1)
    run = kp.compile((kp.exp(-((x - y) ** 2).sum()) * v).sum_over("i"), dtype="float64", backend="oracle")
    X, Y, _ = _inputs(30, 40, integer=False)
    V = torch.randn(30, 1, dtype=F64, generator=_gen(9))
    D = ((X[:, None] - Y[None]) ** 2).sum(-1)
    torch.testing.assert_close(run.with_ranges(r)(x=X, y=Y, v=V), (torch.exp(-D) * mask).T @ V)
    with pytest.raises(ValidationError, match="over='i'"):
        run.with_ranges(Ranges(r.ranges_i, r.slices_i, r.redranges_j))


def test_sum_scheme_kept_with_ranges():
    w, one = Vj("w", 1), Vi("one", 1)
    W = torch.tensor([1e8, 1.0, 1.0, -1e8, 5.0])[:, None]
    r = Ranges(torch.tensor([[0, 2]]), torch.tensor([1]), torch.tensor([[0, 4]]))
    out = {s: kp.compile((w * one).sum_over("j"), dtype="float32", backend="oracle", sum_scheme=s)
           .with_ranges(r)(one=torch.ones(2, 1), w=W)[0, 0].item() for s in ("direct", "kahan")}
    assert out == {"direct": 0.0, "kahan": 2.0}


def test_batch_with_ranges_refused():
    run = kp.compile(_formulas()["sum"], dtype="float64", backend="oracle")
    with pytest.raises(UnsupportedError, match="batch"):
        run.with_ranges(Ranges.dense(3, 4))(x=torch.ones(2, 3, 2, dtype=F64), y=torch.ones(2, 4, 2, dtype=F64),
                                            b=torch.ones(2, 4, 2, dtype=F64))


# --------------------------------------------------------------- autograd
def test_gradcheck_with_ranges():
    li, lj, keep, _ = _clusters(12, 15, 3, 3, 0.6, seed=7)
    r = ranges_from_labels(li, lj, keep)
    run = kp.compile(_formulas()["sum"], dtype="float64", backend="oracle").with_ranges(r)
    g = _gen(4)
    X, Y, Bw = (torch.randn(*s, dtype=F64, generator=g).requires_grad_() for s in ((12, 2), (15, 2), (15, 2)))
    f = lambda X, Y, Bw: run(x=X, y=Y, b=Bw)  # noqa: E731
    assert torch.autograd.gradcheck(f, (X, Y, Bw))
    assert torch.autograd.gradgradcheck(f, (X, Y, Bw))


def test_j_gradient_requires_transposed_triplet():
    r = Ranges.dense(6, 7)
    r_no_t = Ranges(r.ranges_i, r.slices_i, r.redranges_j)
    run = kp.compile(_formulas()["sum"], dtype="float64", backend="oracle").with_ranges(r_no_t)
    X = torch.randn(6, 2, dtype=F64, requires_grad=True)
    Y = torch.randn(7, 2, dtype=F64, requires_grad=True)
    Bw = torch.randn(7, 2, dtype=F64)
    with pytest.raises(UnsupportedError, match="transposed triplet"):
        run(x=X, y=Y, b=Bw).sum().backward()
    X.grad = None
    run(x=X, y=Y.detach(), b=Bw).sum().backward()  # i side only: accepted
    assert X.grad.shape == (6, 2)


# --------------------------------- backends without ranges: refusal
class _NoRanges:
    name = "noranges"

    def __init__(self, honest):
        self.honest = honest

    def supports(self, plan, device):
        if self.honest and plan.ranges:
            return Support(False, "ranges (v0.6 amendment) not implemented")
        return Support(True)

    def compile(self, plan, device):  # pragma: no cover - never reached
        raise AssertionError("must not compile")


@pytest.mark.parametrize("honest", [True, False])
def test_backend_without_ranges_is_refused(honest):
    name = f"noranges_{honest}"
    register_backend(name, lambda: _NoRanges(honest), replace=True)
    run = kp.compile(_formulas()["sum"], dtype="float64", backend=name)
    with pytest.raises(UnsupportedError, match="ranges"):
        run.with_ranges(Ranges.dense(3, 4))(x=torch.ones(3, 2, dtype=F64), y=torch.ones(4, 2, dtype=F64),
                                            b=torch.ones(4, 2, dtype=F64))


def test_ranges_from_labels_rejects_out_of_range_labels():
    keep = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ValidationError, match="labels_i"):
        ranges_from_labels(torch.tensor([0, 2]), torch.tensor([0, 1, 2]), keep)
    with pytest.raises(ValidationError, match="labels_j"):
        ranges_from_labels(torch.tensor([0, 1]), torch.tensor([-1, 0, 2]), keep)
    r = ranges_from_labels(torch.tensor([], dtype=torch.int64), torch.tensor([0, 2]), keep)
    assert r.ranges_i.tolist() == [[0, 0], [0, 0]] and r.ranges_j.tolist() == [[0, 1], [1, 1], [1, 2]]
