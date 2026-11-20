"""Benchmark of kernox against PyKeOps, on a GPU or on the CPU.

Each (library, case) runs in a fresh process that compiles one kernel only,
with a fresh PyKeOps build folder (PyKeOps through ``Genred``, its interface
with the least overhead per call): PyKeOps' best configuration. The processes alternate between the two
libraries, ``--repeat`` times, and the table reports the median times and the
ratio PyKeOps / kernox (above 1: kernox is faster). The results of the two
libraries are compared too.

    python tools/bench_pykeops.py --device cuda --size 100000
    OMP_PROC_BIND=close OMP_PLACES=cores \\
        python tools/bench_pykeops.py --device cpu --threads 192 --size 100000

Cases: ``lse_d2``, ``lse_d3`` (log-sum-exp of -s |x - y|², 1 output),
``gauss_d3`` (Gaussian kernel times a weight, summed over j), ``knn_d3``
(10 nearest neighbours), and ``*_ranges`` (the same with block-sparse ranges
built on a grid, about 3 neighbouring cells per dimension).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

CASES = ["lse_d2", "lse_d3", "lse_d2_ranges", "lse_d3_ranges", "gauss_d3", "gauss_d3_ranges", "knn_d3"]


def data(case, n, device):
    """Inputs of a case, identical in both processes (fixed seed)."""
    import torch

    kind, d = case.split("_")[0], int(case.split("_")[1][1:])
    g = torch.Generator().manual_seed(0)
    X = torch.rand(n, d, generator=g)
    Y = torch.rand(n, d, generator=g)
    B = torch.rand(n, 1, generator=g)
    s = {"lse": 10.0, "gauss": 10.0, "knn": None}[kind]
    ranges = None
    if case.endswith("_ranges"):
        import kernox as kp

        cells = {2: 16, 3: 8}[d]                       # a few hundred points per cell
        s = 1 / (2 * (0.25 / cells) ** 2)               # sigma = a quarter of a cell

        def labels(P):
            c = (P * cells).long().clamp(0, cells - 1)
            return sum(c[:, k] * cells ** (d - 1 - k) for k in range(d))

        lx, px = torch.sort(labels(X))
        ly, py = torch.sort(labels(Y))
        X, Y, B = X[px], Y[py], B[py]
        grid = torch.stack(torch.meshgrid(*[torch.arange(cells)] * d, indexing="ij"), -1).reshape(-1, d)
        keep = (grid[:, None, :] - grid[None, :, :]).abs().amax(-1) <= 1   # 3^d neighbouring cells
        r = kp.ranges_from_labels(lx, ly, keep)
        ranges = tuple(getattr(r, k).to(device=device, dtype=torch.int32)
                       for k in ("ranges_i", "slices_i", "redranges_j", "ranges_j", "slices_j", "redranges_i"))
    return kind, X.to(device), Y.to(device), B.to(device), s, ranges


def kernox_fn(kind, d, s, ranges, device, opts):
    import torch

    import kernox as kp

    x, y, b, p = kp.Vi("x", d), kp.Vj("y", d), kp.Vj("b", 1), kp.Param("s")
    d2 = x.sqdist(y)
    f = {"lse": lambda: (-d2 * p).logsumexp(over="j"),
         "gauss": lambda: (kp.exp(-d2 * p) * b).sum_over("j"),
         "knn": lambda: d2.argkmin(10, over="j")}[kind]()
    k = kp.compile(f, dtype="float32", **opts)
    if ranges is not None:
        k = k.with_ranges(kp.Ranges(*ranges))
    S = torch.tensor([s if s is not None else 1.0], device=device)
    if kind == "lse":
        return lambda X, Y, B: k(x=X, y=Y, s=S)
    if kind == "gauss":
        return lambda X, Y, B: k(x=X, y=Y, b=B, s=S)
    return lambda X, Y, B: k(x=X, y=Y)


def pykeops_fn(kind, d, s, ranges, device, opts):
    # Genred, the lowest-overhead PyKeOps interface (no LazyTensor parsing per call)
    import pykeops
    import torch
    from pykeops.torch import Genred

    pykeops.set_verbose(False)
    if os.environ.get("KEOPS_CACHE_FOLDER"):
        pykeops.clean_pykeops()   # fresh build folder: build the JIT binaries there
    S = torch.tensor([s if s is not None else 1.0], device=device)
    xy = [f"x = Vi({d})", f"y = Vj({d})"]
    if kind == "lse":
        g = Genred("-SqDist(x,y) * s", xy + ["s = Pm(1)"], reduction_op="LogSumExp", axis=1)
        return lambda X, Y, B: g(X, Y, S, ranges=ranges)
    if kind == "gauss":
        g = Genred("Exp(-SqDist(x,y) * s) * b", xy + ["b = Vj(1)", "s = Pm(1)"], reduction_op="Sum", axis=1)
        return lambda X, Y, B: g(X, Y, B, S, ranges=ranges)
    g = Genred("SqDist(x,y)", xy, reduction_op="ArgKMin", opt_arg=10, axis=1)
    return lambda X, Y, B: g(X, Y, ranges=ranges)


def child(a):
    if a.lib == "pykeops":
        # before torch: otherwise PyKeOps links its JIT binaries against the NVRTC
        # wheel loaded by torch, which has no libnvrtc.so, and fails in a fresh folder
        import pykeops  # noqa: F401
    import torch

    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    kind, X, Y, B, s, ranges = data(a.case, a.size, a.device)
    d = X.shape[1]
    opts = json.loads(a.kernox_opts)
    fn = (kernox_fn if a.lib == "kernox" else pykeops_fn)(kind, d, s, ranges, a.device, opts)
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    t0 = time.perf_counter()
    out = fn(X, Y, B)
    sync()
    first = time.perf_counter() - t0
    for _ in range(2):
        fn(X, Y, B)
    sync()
    times, t_start = [], time.perf_counter()
    while len(times) < a.min_reps or (time.perf_counter() - t_start < a.min_time and len(times) < 200):
        t0 = time.perf_counter()
        out = fn(X, Y, B)
        sync()
        times.append(time.perf_counter() - t0)
    torch.save(out.detach().cpu(), a.out)
    print(json.dumps({"ms": 1e3 * statistics.median(times), "first_s": first, "reps": len(times)}))


def driver(a):
    # The driver imports torch to compare the results; with OMP_PROC_BIND set,
    # OpenMP then pins it to one core, and the children would inherit that
    # affinity: they get the initial one back.
    cpus = os.sched_getaffinity(0)
    env = dict(os.environ)
    if a.device == "cpu":
        env["OMP_NUM_THREADS"] = str(a.threads)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in a.cases:
            res = {"kernox": [], "pykeops": []}
            outs = {}
            for r in range(a.repeat):
                for lib in ("pykeops", "kernox") if r % 2 else ("kernox", "pykeops"):
                    outs[lib] = os.path.join(tmp, f"{case}_{lib}.pt")
                    cmd = [sys.executable, __file__, "--child", "--lib", lib, "--case", case,
                           "--device", a.device, "--size", str(a.size), "--threads", str(a.threads),
                           "--kernox-opts", a.kernox_opts, "--out", outs[lib],
                           "--min-reps", str(a.min_reps), "--min-time", str(a.min_time)]
                    with tempfile.TemporaryDirectory() as build:
                        p = subprocess.run(cmd, env=env | {"KEOPS_CACHE_FOLDER": build},
                                           capture_output=True, text=True,
                                           preexec_fn=lambda: os.sched_setaffinity(0, cpus))
                    if p.returncode:
                        sys.stderr.write(p.stdout + p.stderr)
                        raise SystemExit(f"{lib} {case} failed")
                    res[lib].append(json.loads(p.stdout.strip().splitlines()[-1])["ms"])
            rows.append((case, statistics.median(res["kernox"]), statistics.median(res["pykeops"]),
                         agreement(case, outs)))
            print(f"{case}: kernox {res['kernox']} pykeops {res['pykeops']}", file=sys.stderr, flush=True)
    print(f"\nM = N = {a.size:,}, float32, device {a.device}"
          + (f", {a.threads} threads" if a.device == "cpu" else "") + f", kernox options {a.kernox_opts}\n")
    print("| Case | kernox | PyKeOps | PyKeOps / kernox | agreement |")
    print("| --- | --- | --- | --- | --- |")
    for case, k, p, agree in rows:
        print(f"| {case} | {fmt(k)} | {fmt(p)} | × {p / k:.2f} | {agree} |")


def fmt(ms):
    return f"{ms:.3g} ms" if ms < 1000 else f"{ms / 1e3:.3g} s"


def agreement(case, outs):
    import torch

    a, b = (torch.load(outs[k]) for k in ("kernox", "pykeops"))
    a, b = a.reshape(b.shape), b
    if case.startswith("knn"):
        return f"{(a == b).float().mean().item():.4%} same indices"
    return f"max rel. diff {((a - b).abs().max() / b.abs().max()).item():.1e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--size", type=int, default=100_000)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--repeat", type=int, default=2, help="fresh processes per library and case")
    ap.add_argument("--kernox-opts", default=None, help='JSON, default {"fast_math": true} on cuda')
    ap.add_argument("--min-reps", type=int, default=10)
    ap.add_argument("--min-time", type=float, default=1.0)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--lib")
    ap.add_argument("--case")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.kernox_opts is None:
        a.kernox_opts = json.dumps({"fast_math": True} if a.device == "cuda" else {})
    child(a) if a.child else driver(a)
