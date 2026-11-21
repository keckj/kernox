"""kernox.distributed: split reductions on a Gloo group of local processes (CPU).

Each worker runs every case with the same inputs as a single-process call
of the same reduction, and compares outputs and gradients.
"""

from __future__ import annotations

import os
import traceback

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

import kernox as kp  # noqa: E402
from kernox import Vi, Vj, ranges_from_labels  # noqa: E402

pytestmark = pytest.mark.skipif(not (dist.is_available() and dist.is_gloo_available()),
                                reason="torch.distributed with Gloo required")

F64 = torch.float64


def _formulas():
    x, y, b = Vi("x", 2), Vj("y", 2), Vj("b", 2)
    s = kp.Param("s", 1)
    d2 = ((x - y) ** 2).sum()
    score = -d2 * s
    return {
        "sum": (kp.exp(-d2 * s) * b).sum_over("j"),
        "sum_over_i": (kp.exp(-d2 * s) * x).sum_over("i"),
        "count": (d2 < 1.5).sum_over("j"),
        "min": d2.min(over="j"),
        "max": (d2 * s).max(over="j"),
        "argmin": d2.argmin(over="j"),
        "argmax": d2.argmax(over="j"),
        "min_argmin": (d2 * s).min_argmin(over="j"),
        "max_argmax": d2.max_argmax(over="j"),
        "kmin": (d2 * s).kmin(3, over="j"),
        "argkmin": d2.argkmin(3, over="j"),
        "kmin_argkmin": (d2 * s).kmin_argkmin(5, over="j"),
        "kmin_over_i": d2.kmin_argkmin(4, over="i"),
        "logsumexp": kp.cat(score, score * 0.5).logsumexp(over="j"),
        "lse_over_i": score.logsumexp(over="i"),
        "logsumexpweight": score.logsumexp(over="j", weight=kp.exp(b)),
        "sumsoftmaxweight": score.sumsoftmaxweight(b, over="j"),
    }


def _inputs(run, m, n, batch=(), seed=0, ties=False):
    g = torch.Generator().manual_seed(seed)
    kw = {}
    for a in run.plan.args:
        name = a.name
        if name == "s":
            kw[name] = torch.tensor([0.7], dtype=F64)
            continue
        rows = {"x": m, "y": n, "b": n}[name]
        kw[name] = torch.randn(*batch, rows, 2, generator=g, dtype=F64)
    if ties:  # second half of y = first half: equal values at j and j + n/2
        h = n // 2
        kw["y"][..., h:2 * h, :] = kw["y"][..., :h, :]
    return kw


def _as_tuple(out):
    return out if isinstance(out, tuple) else (out,)


def _compare(name, got, ref, rtol=1e-11, atol=1e-12):
    for k, (p, q) in enumerate(zip(_as_tuple(got), _as_tuple(ref))):
        assert p.shape == q.shape and p.dtype == q.dtype, (name, k, p.shape, q.shape, p.dtype, q.dtype)
        if p.is_floating_point():
            torch.testing.assert_close(p, q, rtol=rtol, atol=atol, equal_nan=True, msg=lambda s: f"{name}[{k}]: {s}")
        else:
            assert torch.equal(p, q), (name, k, (p != q).nonzero()[:5])


def _check(name, run, kw, split, grad=True, ranges=None):
    """Distributed vs local call: outputs, then gradients of a fixed linear loss."""
    import kernox.distributed as kd

    def leaves():
        return {k: (v.clone().requires_grad_(grad and v.is_floating_point()) if torch.is_tensor(v) else v)
                for k, v in kw.items()}

    local = run if ranges is None else run.with_ranges(ranges, validate=True)
    a, b = leaves(), leaves()
    ref = local(**a)
    got = kd.reduce(local, split=split, **b)  # validate=True: the clipped ranges are checked too
    _compare(f"{name}/{split}", got, ref)
    if not grad:
        return
    floats = [(p, q) for p, q in zip(_as_tuple(got), _as_tuple(ref)) if p.is_floating_point() and q.requires_grad]
    if not floats:
        return
    g = torch.Generator().manual_seed(7)
    weights = [torch.randn(q.shape, generator=g, dtype=q.dtype) for _, q in floats]

    def loss(outs):
        return sum((torch.nan_to_num(o, posinf=0.0, neginf=0.0) * w).sum() for o, w in zip(outs, weights))

    loss([q for _, q in floats]).backward()
    loss([p for p, _ in floats]).backward()
    for k in a:
        if torch.is_tensor(a[k]) and a[k].requires_grad:
            ga, gb = a[k].grad, b[k].grad
            if ga is None:
                assert gb is None or not gb.abs().any(), (name, k)
                continue
            torch.testing.assert_close(gb, ga, rtol=1e-10, atol=1e-12, msg=lambda s: f"{name}/{split} grad {k}: {s}")


def _labels(m, n, seed=3):
    g = torch.Generator().manual_seed(seed)
    li = torch.sort(torch.randint(0, 4, (m,), generator=g)).values
    lj = torch.sort(torch.randint(0, 5, (n,), generator=g)).values
    keep = torch.rand(4, 5, generator=g) < 0.5
    keep[0, 0] = True
    return li, lj, keep


def _cases(world):
    forms = _formulas()
    runs = {k: kp.compile(f, dtype="float64", backend="oracle") for k, f in forms.items()}
    for split in ("output", "reduced", None):
        for name, run in runs.items():
            yield f"dense/{name}", run, _inputs(run, 29, 23), split, None
        # batch dimensions
        for name in ("sum", "logsumexp", "kmin_argkmin", "sumsoftmaxweight"):
            run = runs[name]
            yield f"batch/{name}", run, _inputs(run, 13, 17, batch=(2,)), split, None
        # ties across shards: the smallest global index wins
        for name in ("argmin", "max_argmax", "kmin_argkmin", "argkmin"):
            yield f"ties/{name}", runs[name], _inputs(runs[name], 11, 24, ties=True), split, None
        # block-sparse ranges (both triplets), plan over j
        li, lj, keep = _labels(31, 37)
        r = ranges_from_labels(li, lj, keep)
        for name in ("sum", "count", "min", "argmax", "kmin_argkmin", "logsumexp", "logsumexpweight",
                     "sumsoftmaxweight"):
            run = runs[name]
            yield f"ranges/{name}", run, _inputs(run, 31, 37), split, r
        run = runs["lse_over_i"]  # over i: the transposed triplet drives the forward pass
        yield "ranges/lse_over_i", run, _inputs(run, 31, 37), split, r
    # K larger than a shard (and than N)
    run = kp.compile(((Vi("x", 2) - Vj("y", 2)) ** 2).sum().kmin_argkmin(9, over="j"),
                     dtype="float64", backend="oracle")
    yield "kmin/K>N", run, _inputs(run, 10, 7), "reduced", None
    # axis shorter than the group: local call
    run = runs["sum"]
    yield "fallback/short", run, _inputs(run, world - 1, 5), "output", None


def _worker(rank, world, init_file, errors_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    failures = []
    try:
        for name, run, kw, split, ranges in _cases(world):
            try:
                _check(name, run, kw, split, ranges=ranges)
            except Exception:  # noqa: BLE001 - reported by the parent
                failures.append(f"{name}/{split}:\n{traceback.format_exc()}")
    finally:
        dist.barrier()
        dist.destroy_process_group()
    if failures:
        with open(os.path.join(errors_dir, f"rank{rank}.txt"), "w") as f:
            f.write("\n".join(failures))


def _spawn(fn, world, tmp_path, *args):
    import torch.multiprocessing as mp

    init_file = tmp_path / "init"
    errors = tmp_path / "errors"
    errors.mkdir()
    mp.spawn(fn, args=(world, str(init_file), str(errors), *args), nprocs=world, join=True)
    reports = sorted(errors.iterdir())
    if reports:
        pytest.fail("\n".join(p.read_text() for p in reports)[:20000])


@pytest.mark.parametrize("world", [2, 3])
def test_distributed_matches_single_process(world, tmp_path):
    _spawn(_worker, world, tmp_path)


def test_without_group_is_a_plain_call():
    import kernox.distributed as kd

    assert not dist.is_initialized()
    run = kp.compile(_formulas()["sum"], dtype="float64", backend="oracle")
    kw = _inputs(run, 5, 6)
    torch.testing.assert_close(kd.reduce(run, **kw), run(**kw), rtol=0, atol=0)


def test_spans_balance_pairs():
    from kernox.distributed import _from_cumulative, _ranges_cumulative, _uniform
    from kernox.ranges import Triplet

    assert _uniform(10, 3) == ((0, 4), (4, 7), (7, 10))
    # rows [0, 10) meet 1 column, rows [10, 20) meet 9: the cut lands near row 15
    tr = Triplet(torch.tensor([[0, 10], [10, 20]]), torch.tensor([1, 2]), torch.tensor([[0, 1], [1, 10]]))
    cum = _ranges_cumulative(tr, 20, "i")
    assert int(cum[-1]) == 100
    (a0, b0), (a1, b1) = _from_cumulative(cum, 2)
    assert (a0, b1) == (0, 20) and b0 == a1 and 14 <= b0 <= 16
    cumj = _ranges_cumulative(tr, 10, "j")
    assert int(cumj[-1]) == 100 and int(cumj[0]) == 10
    # every rank keeps at least one row
    assert all(b > a for a, b in _from_cumulative(torch.tensor([0, 0, 0, 0, 100]), 4))


def test_init_from_env_reads_launchers():
    from kernox.distributed import _env_ranks

    assert _env_ranks({"OMPI_COMM_WORLD_RANK": "3", "OMPI_COMM_WORLD_SIZE": "8",
                       "OMPI_COMM_WORLD_LOCAL_RANK": "1", "OMPI_COMM_WORLD_LOCAL_SIZE": "4"}) == (3, 8, 1, 4)
    assert _env_ranks({"SLURM_PROCID": "5", "SLURM_NTASKS": "6", "SLURM_LOCALID": "2",
                       "SLURM_NTASKS_PER_NODE": "4(x2),2"}) == (5, 6, 2, 4)
    assert _env_ranks({"RANK": "0", "WORLD_SIZE": "2"}) == (0, 2, None, None)
    assert _env_ranks({}) is None


def test_master_addr_from_oar_nodefile(tmp_path):
    from kernox.distributed import _master_addr

    f = tmp_path / "nodes"
    f.write_text("\nkraken-c12\nkraken-c12\nkraken-c13\n")
    assert _master_addr({"OAR_NODEFILE": str(f)}) == "kraken-c12"
    assert _master_addr({"MASTER_ADDR": "h", "OAR_NODEFILE": str(f)}) == "h"
    assert _master_addr({}) == "127.0.0.1"


def _cpu_worker(rank, world, init_file, errors_dir, cache_dir):
    os.environ["KERNOX_CACHE_DIR"] = cache_dir  # fresh cache shared by the ranks: concurrent builds
    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    failures = []
    try:
        forms = _formulas()
        li, lj, keep = _labels(301, 257)
        r = ranges_from_labels(li, lj, keep)
        cases = [("sum", "output", None), ("min_argmin", "reduced", None), ("logsumexp", "reduced", r),
                 ("kmin_argkmin", "output", r)]
        for name, split, ranges in cases:
            run = kp.compile(forms[name], dtype="float64", backend="cpu")
            try:
                _check(f"cpu/{name}", run, _inputs(run, 301, 257), split, ranges=ranges)
            except Exception:  # noqa: BLE001 - reported by the parent
                failures.append(f"cpu/{name}/{split}:\n{traceback.format_exc()}")
    finally:
        dist.barrier()
        dist.destroy_process_group()
    if failures:
        with open(os.path.join(errors_dir, f"rank{rank}.txt"), "w") as f:
            f.write("\n".join(failures))


def test_cpu_backend_on_two_ranks(tmp_path):
    import shutil

    from kernox._env import getenv

    if shutil.which(getenv("KERNOX_CPU_CXX", "g++")) is None:
        pytest.skip("no C++ compiler")
    _spawn(_cpu_worker, 2, tmp_path, str(tmp_path / "cache"))


def test_auto_threads_respects_binding(monkeypatch):
    import kernox.distributed as kd

    a, b = frozenset(range(0, 4)), frozenset(range(4, 8))
    whole = a | b
    # bound by the launcher to disjoint cores: each rank keeps its own
    assert kd._share(a, 4, "n1", [("n1", a), ("n1", b), ("n2", a)]) == 4
    # unbound: the ranks of the node divide its cores, other hosts do not count
    assert kd._share(whole, 8, "n1", [("n1", whole), ("n1", whole), ("n2", whole)]) == 4
    assert kd._share(whole, 8, "n1", [("n1", whole)] * 3) == 2
    assert kd._share(frozenset({0}), 1, "n1", [("n1", frozenset({0}))] * 4) == 1
    # OMP_PROC_BIND pins the main thread to one core when torch starts:
    # the OpenMP team counts, not that core
    monkeypatch.delenv("OMP_PROC_BIND", raising=False)
    assert kd._cores(frozenset({0})) == 1
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    assert kd._cores(frozenset({0})) == max(1, kd._omp_places())
    monkeypatch.setenv("OMP_PROC_BIND", "false")
    assert kd._cores(frozenset({0})) == 1
