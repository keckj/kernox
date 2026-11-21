"""Strong scaling of kernox.distributed (see docs/guides/distributed.md).

A Gaussian kernel product sum_j exp(-|x_i - y_j|^2 / 2s^2) b_j in D = 3,
FP32, M = N, forward and forward + backward (gradient with respect to x),
dense (N^2 pairs) or block-sparse (a grid of about PER_CELL points per cell,
each cell against its 27 neighbours, a kernel a quarter of a cell wide:
about 27 PER_CELL pairs per point, up to 10^8 points). Launch one process per GPU or per node (or socket); with a single
process, kd.reduce is a plain call and gives the reference time. Rank 0 prints
one JSON line per case.

    torchrun --nproc-per-node 4 tools/bench_distributed.py --sizes 1e6 3e6 --sparse 1e7 1e8
    mpirun -np 4 --map-by ppr:1:node python tools/bench_distributed.py --sizes 1e6 --sparse 1e8
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.distributed as dist

import kernox as kp
import kernox.distributed as kd


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(fn, device, min_reps, min_time, warmup):
    for _ in range(warmup):  # compilation, caches
        fn()
    times = []
    start = time.perf_counter()
    while len(times) < min_reps or time.perf_counter() - start < min_time:
        if dist.is_initialized():
            dist.barrier()
        _sync(device)
        t = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t)
    return statistics.median(times), len(times)


def _grid_ranges(X, Y, per_cell):
    """Sort the points by cell of a G^3 grid (about ``per_cell`` points of X
    per cell) and keep, for each cell, its 27 neighbours: nine intervals of
    three consecutive cells in the sort order, built directly (no Ci x Cj
    matrix, so that it scales to 10^8 points)."""
    n = X.shape[0]
    G = max(1, round((n / per_cell) ** (1 / 3)))

    def sort(P):
        q = torch.floor(P * G).long().clamp_(0, G - 1)
        lab = (q[:, 0] * G + q[:, 1]) * G + q[:, 2]
        lab, order = torch.sort(lab)
        counts = torch.bincount(lab, minlength=G ** 3)
        ends = torch.cumsum(counts, 0)
        return P[order].contiguous(), torch.stack([ends - counts, ends], 1)

    X, bx = sort(X)
    Y, by = sort(Y)
    g = torch.arange(G)
    a, b, c = torch.meshgrid(g, g, g, indexing="ij")
    a, b, c = (t.reshape(-1, 1) for t in (a, b, c))
    d = torch.tensor([-1, 0, 1])
    da, db = (t.reshape(1, -1) for t in torch.meshgrid(d, d, indexing="ij"))
    a2, b2 = a + da, b + db  # (G^3, 9)
    ok = (a2 >= 0) & (a2 < G) & (b2 >= 0) & (b2 < G)
    lo = (a2 * G + b2) * G + (c - 1).clamp(min=0)
    hi = (a2 * G + b2) * G + (c + 1).clamp(max=G - 1)
    lo, hi = lo[ok], hi[ok]
    slices = torch.cumsum(ok.sum(1), 0)

    def triplet(rows, cols):
        return rows, slices, torch.stack([cols[lo, 0], cols[hi, 1]], 1)

    return X, Y, G, kp.Ranges(*triplet(bx, by), *triplet(by, bx))


def _pairs(r):
    """Number of (i, j) pairs kept by the ranges."""
    rows = (r.ranges_i[:, 1] - r.ranges_i[:, 0]).cpu().double()
    lens = (r.redranges_j[:, 1] - r.redranges_j[:, 0]).cpu().double()
    per_block = torch.zeros(rows.numel(), dtype=torch.float64)
    ends = r.slices_i.cpu().long()
    block = torch.repeat_interleave(torch.arange(rows.numel()), torch.diff(ends, prepend=ends.new_zeros(1)))
    per_block.index_add_(0, block, lens)
    return float((rows * per_block).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="*", default=["2e5"], help="sizes of the dense case")
    ap.add_argument("--sparse", nargs="*", default=[], metavar="N",
                    help="sizes of the block-sparse case (about 27 x PER_CELL pairs per point)")
    ap.add_argument("--per-cell", type=int, default=1000)
    ap.add_argument("--validate", action="store_true", help="check the ranges (slow)")
    ap.add_argument("--split", default=None, choices=[None, *kd.SPLITS])
    ap.add_argument("--min-reps", type=int, default=3)
    ap.add_argument("--min-time", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=1, help="untimed runs before each timing (0 for long "
                    "cases: the untimed check run already compiles the forward kernels)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    info = kd.init_from_env()
    device = info.device
    x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 1)
    gauss = kp.compile((kp.exp(-50.0 * x.sqdist(y)) * b).sum_over("j"), dtype="float32")
    cases = [("dense", int(float(s))) for s in a.sizes] + [("ranges", int(float(s))) for s in a.sparse]
    for case, n in cases:
        g = torch.Generator().manual_seed(0)
        X, Y = torch.rand(n, 3, generator=g), torch.rand(n, 3, generator=g)
        B = torch.randn(n, 1, generator=g)
        if case == "dense":
            run, pairs = gauss, float(n) * n
        else:
            X, Y, G, r = _grid_ranges(X, Y, a.per_cell)
            # width a quarter of a cell: beyond the 27 neighbouring cells, below exp(-8)
            k = kp.compile((kp.exp(-8.0 * G * G * x.sqdist(y)) * b).sum_over("j"), dtype="float32")
            r = kp.Ranges(*(t.to(device) for t in (r.ranges_i, r.slices_i, r.redranges_j,
                                                     r.ranges_j, r.slices_j, r.redranges_i)))
            run, pairs = k.with_ranges(r, validate=a.validate), _pairs(r)
        Xd, Yd, Bd = X.to(device), Y.to(device), B.to(device)
        Xg = Xd.clone().requires_grad_()
        del X, Y, B

        def fwd():
            return kd.reduce(run, split=a.split, x=Xd, y=Yd, b=Bd)

        def fwd_bwd():
            out = kd.reduce(run, split=a.split, x=Xg, y=Yd, b=Bd)
            (gx,) = torch.autograd.grad(out.sum(), Xg)
            return gx

        check = float(fwd().double().sum())
        tf, nf = _time(fwd, device, a.min_reps, a.min_time, a.warmup)
        tb, nb = _time(fwd_bwd, device, a.min_reps, a.min_time, a.warmup)
        if info.rank == 0:
            print(json.dumps({"case": case, "N": n, "P": info.world_size, "device": str(device),
                              "threads": torch.get_num_threads(), "split": a.split, "tag": a.tag,
                              "pairs": pairs, "fwd": tf, "fwd_reps": nf, "fwd_bwd": tb, "fwd_bwd_reps": nb,
                              "check": check}), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
