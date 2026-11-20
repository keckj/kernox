"""Benchmark of the ``cpu`` backend (see docs/backend-cpu.md).

Times kernox ``backend="cpu"`` on a few reductions, optionally against a
hand-written chunked PyTorch reference (``--baselines``), its tuning variants
(``--variants``) and a sweep of the j tile (``--tile-sweep``). Pin the threads
for stable numbers, e.g. ``OMP_PROC_BIND=close OMP_PLACES=cores``.

    python tools/bench_cpu.py --threads 1 8 --sizes 20000 --baselines
"""
import argparse
import json
import os
import time

import torch

import kernox as kp
from kernox import Param, Vi, Vj
from kernox.plan import output_shapes

p = argparse.ArgumentParser()
p.add_argument("--threads", type=int, nargs="+", default=[1, 8])
p.add_argument("--sizes", type=int, nargs="+", default=[100_000])
p.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
p.add_argument("--cases", nargs="+", default=["gauss", "lse", "knn", "gauss_d16", "funcs"],
               help="gauss lse knn gauss_d16 gauss_d32 funcs")
p.add_argument("--baselines", action="store_true")
p.add_argument("--variants", action="store_true", help="time all 'measure' variants")
p.add_argument("--tile-sweep", action="store_true", help="time j tiles 64..512 (32 rows per task)")
p.add_argument("--json", default=None)
args = p.parse_args()


def cases():
    x, y, b, s = Vi("x", 3), Vj("y", 3), Vj("b", 1), Param("s")
    d2 = ((x - y) ** 2).sum()
    x16, y16 = Vi("x", 16), Vj("y", 16)
    d16 = ((x16 - y16) ** 2).sum()
    x32, y32 = Vi("x", 32), Vj("y", 32)
    d32 = ((x32 - y32) ** 2).sum()
    return {
        "gauss": ((kp.exp(-d2 / s) * b).sum_over("j"), dict(x=("i", 3), y=("j", 3), b=("j", 1), s=("p", 1))),
        "gauss_d16": ((kp.exp(-d16 / s) * b).sum_over("j"), dict(x=("i", 16), y=("j", 16), b=("j", 1), s=("p", 1))),
        "gauss_d32": ((kp.exp(-d32 / s) * b).sum_over("j"), dict(x=("i", 32), y=("j", 32), b=("j", 1), s=("p", 1))),
        "lse": ((-d2 / s).logsumexp(over="j"), dict(x=("i", 3), y=("j", 3), s=("p", 1))),
        "knn": (d2.argkmin(10, over="j"), dict(x=("i", 3), y=("j", 3))),
        "funcs": ((kp.sin(x[0]) * kp.cos(y[1]) + kp.atan2(x[1], y[0]) + kp.tanh(x[2] * y[2])).sum_over("j"),
                  dict(x=("i", 3), y=("j", 3))),
    }


def inputs(spec, m, n, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, (role, d) in spec.items():
        rows = {"i": m, "j": n, "p": None}[role]
        shape = (d,) if rows is None else (rows, d)
        t = torch.rand(shape, generator=g, dtype=torch.float64)
        if role == "p":
            t = t + 0.5
        out[k] = t.to(getattr(torch, dtype)).contiguous()
    return out


def timeit(fn, min_time=1.0, max_rep=20):
    fn()
    ts = []
    t_end = time.perf_counter() + min_time
    while len(ts) < max_rep and (len(ts) < 3 or time.perf_counter() < t_end):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def torch_ref(name, t, chunk=4096):
    x, y = t["x"], t["y"]
    outs = []
    for i0 in range(0, x.shape[0], chunk):
        xi = x[i0:i0 + chunk]
        d2 = torch.cdist(xi, y).square_() if x.shape[1] > 3 else ((xi[:, None, :] - y[None]) ** 2).sum(-1)
        if name in ("gauss", "gauss_d16"):
            outs.append(torch.exp(-d2 / t["s"]) @ t["b"])
        elif name == "lse":
            outs.append(torch.logsumexp(-d2 / t["s"], dim=1, keepdim=True))
        elif name == "knn":
            outs.append(torch.topk(d2, 10, dim=1, largest=False).indices)
        else:
            return None
    return torch.cat(outs)


res = []
print(f"host={os.uname().nodename} torch={torch.__version__} libmvec={os.environ.get('KERNOX_CPU_LIBMVEC', '1')}")
C = cases()
for name in args.cases:
    expr, spec = C[name]
    for dtype in args.dtypes:
        for size in args.sizes:
            t = inputs(spec, size, size, dtype)
            run = kp.compile(expr, dtype=dtype, backend="cpu")
            run(**t)
            comp = run.compiled(torch.device("cpu"), size, size)
            for nth in args.threads:
                torch.set_num_threads(nth)
                dt = timeit(lambda: run(**t))
                row = dict(case=name, dtype=dtype, m=size, n=size, threads=nth, impl="kernox-cpu",
                           variant=dict(comp.info.variant.params), split=comp.last_split[:2], seconds=dt,
                           gpairs_s=size * size / dt / 1e9)
                res.append(row)
                print(f"{name:10s} {dtype} M=N={size:>8d} th={nth:3d} kernox-cpu {dt*1e3:9.2f} ms  "
                      f"{row['gpairs_s']:7.2f} Gpairs/s split={row['split']}", flush=True)
                if args.baselines:
                    ref = None
                    if size * size <= 4e10 / max(1, 96 // nth):
                        try:
                            if torch_ref(name, inputs(spec, 64, 64, dtype)) is not None:
                                dref = timeit(lambda: torch_ref(name, t), min_time=0.5, max_rep=5)
                                ref = dref
                                res.append(dict(row, impl="torch-chunked", seconds=dref, gpairs_s=size * size / dref / 1e9))
                                print(f"{'':10s} {'':7s} {'':10s} {'':7s} torch      {dref*1e3:9.2f} ms  "
                                      f"{size*size/dref/1e9:7.2f} Gpairs/s  (x{dref/dt:.1f})", flush=True)
                        except Exception as e:  # noqa: BLE001
                            print("torch ref failed", e)
            if args.variants or args.tile_sweep:
                torch.set_num_threads(args.threads[-1])
                be = kp.get_backend("cpu")
                plan = run.plan
                vs = be.variants(plan, torch.device("cpu"), None, "measure") if args.variants else []
                if args.tile_sweep:
                    base = be.default_variant(plan, torch.device("cpu"))
                    vs += [type(base)(base.name, tuple(sorted({**dict(base.params), "tj": tj, "bi": 32}.items())))
                           for tj in (64, 128, 256, 512)]
                    vs = list(dict.fromkeys(vs))
                for v in vs:
                    c = be.compile(plan, torch.device("cpu"), v)
                    m = size
                    outs = {k: torch.empty(sh, dtype=getattr(torch, dt_)) for k, (sh, dt_) in
                            output_shapes(plan, m).items()}
                    dv = timeit(lambda: c.launch(t, outs))
                    print(f"   variant {dict(v.params)} {dv*1e3:9.2f} ms {size*size/dv/1e9:7.2f} Gpairs/s", flush=True)
                    res.append(dict(case=name, dtype=dtype, m=size, n=size, threads=args.threads[-1], impl="variant",
                                    variant=dict(v.params), seconds=dv, gpairs_s=size * size / dv / 1e9))
if args.json:
    with open(args.json, "w") as f:
        json.dump(res, f, indent=1)
