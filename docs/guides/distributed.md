# Several GPUs or nodes

`kernox.distributed` splits one reduction across the processes of a
`torch.distributed` group: several GPUs of a node with NCCL, several CPU
nodes with Gloo. Every process passes the same full inputs and receives the
full result, so the rest of the program does not change. Each process only
computes its share of the $M \times N$ pairs.

## A distributed call

```python
import torch
import kernox as kp
import kernox.distributed as kd

device = "cuda"

x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 1)
gauss = kp.compile((kp.exp(-x.sqdist(y)) * b).sum_over("j"), dtype="float32")

X = torch.randn(4_000, 3, device=device)
Y = torch.randn(8_000, 3, device=device)
B = torch.randn(8_000, 1, device=device)

out = kd.reduce(gauss, x=X, y=Y, b=B)      # (4000, 1) on every rank
```

`kd.reduce(run, **inputs)` takes the same keyword arguments as `run(**inputs)`.
Without an initialized group, or with a single process, it is a plain call:
the same script runs on a laptop and on a cluster.

## Launching

Each process initializes the default group once, at the start of the script:

<!-- skip-test -->
```python
import kernox.distributed as kd

info = kd.init_from_env()     # rank, world_size, local_rank, device
```

`init_from_env` reads the environment of the launcher: `torchrun`, Open MPI
`mpirun`, MPICH and Intel MPI, PMIx, or SLURM `srun`. It uses NCCL when CUDA
is available (one GPU per process, `local_rank % device_count`) and Gloo
otherwise. `MASTER_ADDR` defaults to the first host of `$OAR_NODEFILE` (or
`$PBS_NODEFILE`, then the SLURM launch node), `MASTER_PORT` to 29500. On the
CPU, it divides the cores of the node between the processes that share it
(`torch.set_num_threads`).

<!-- skip-test -->
```bash
torchrun --nproc-per-node 4 script.py                         # 4 GPUs of a node
mpirun -np 8 --map-by ppr:2:node:pe=96 python script.py       # 4 nodes, 2 processes each
```

[Running on a cluster](cluster.md) goes through it step by step: testing on
a laptop, batch scripts for SLURM and OAR, and what to do when it hangs.

If your program already calls `torch.distributed.init_process_group`, skip
`init_from_env`: `kd.reduce` uses the default group, or the one passed as
`group=`.

## Two ways to split

![The two ways kd.reduce splits a reduction across four processes](../images/distributed-split.svg)

`split="output"` (the default): each process owns a contiguous span of the
output rows, computes them exactly, and an `all_gather` assembles the result.
The communication is the size of the output.

`split="reduced"`: each process reduces over a span of the reduced axis, and
the partial results are merged. This suits short outputs over long axes, such
as a few queries against millions of points. Every reduction is supported,
with the semantics of a single call:

| Reduction | Merge of the partial results |
| --- | --- |
| `sum_over`, count | sum |
| `min`, `max`, `argmin`, `argmax` | best value, then the smallest index on ties |
| `kmin`, `argkmin` | the K smallest of the candidates, `(+inf, -1)` padding |
| `logsumexp` | log-sum-exp of the partial values |
| `logsumexp(weight=)`, `sumsoftmaxweight` | log-sum-exp state `(m, s)` of each process |

`split=None` chooses `"reduced"` when the output axis is shorter than the
reduced axis and than 32 rows per process, `"output"` otherwise. When the split
axis is shorter than the number of processes, every process makes the full call.

## Gradients

Autograd works through `kd.reduce`. In the backward pass, the gradient of
each input that requires one is summed over the processes (`all_reduce`), so
every process ends with the full gradient, as in the forward pass. The
gradient of the output must be the same on every process: this holds when the
loss is computed from the replicated result.

## Block-sparse reductions and batches

Pass the ranges as for a plain call, with `run.with_ranges(r)` or `ranges=r`:

<!-- skip-test -->
```python
out = kd.reduce(gauss.with_ranges(r), x=Xs, y=Ys, b=Bs)
```

The processes then share the pairs that the ranges keep, not the rows: the
spans are cut so that each process gets the same number of pairs. The spans
are computed once per ranges object (one host synchronization). Batch
dimensions are split along the rows as well.

## Reproducibility

The collectives are plain `all_gather` and `all_reduce`, so the result is the
same on every process and does not change from one run to the next with the
same number of processes. It is not bitwise identical to a single-process
call: the sums are done in another order.

## Scaling

Strong scaling measured with `tools/bench_distributed.py`: a Gaussian kernel
product in $D = 3$, float32, $M = N$, forward and backward (gradient with respect
to `x`), median time in seconds. *Dense* computes the $N^2$ pairs. *Block-sparse*
sorts the points on a grid of about 1,000 points per cell and keeps, for each
cell, its 27 neighbours: about 27,000 pairs per point ($2.7 \times 10^{12}$ pairs for
$10^8$ points).

![Strong scaling of kernox.distributed: speed-up against the number of devices](../images/scaling.svg)

Speed-up = time on one device / time on $P$ devices; the times are in the
tables below.

AMD Instinct MI210, one process per GPU, 4 GPUs per node, RCCL:

| Case | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs (2 nodes) | Speed-up at 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense, $N = 10^6$ | 7.40 | 3.29 | 1.69 | 1.05 | × 7.1 |
| Dense, $N = 10^7$ | 756 | 378 | 205 | 94.6 | × 8.0 |
| Block-sparse, $N = 10^7$ | 1.63 | 0.82 | 0.56 | 0.44 | × 3.7 |
| Block-sparse, $N = 10^8$ | 21.4 | 9.55 | 5.01 | 2.79 | × 7.7 |

CPU nodes of 2 × AMD EPYC 9654 (192 cores), one process per node with all
its cores, Gloo over TCP:

| Case | 1 node | 2 nodes | 4 nodes | 8 nodes | Speed-up at 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense, $N = 10^6$ | 10.3 | 5.22 | 2.66 | 1.50 | × 6.9 |
| Dense, $N = 3 \times 10^6$ | 94.9 | 47.9 | 24.1 | 13.2 | × 7.2 |
| Block-sparse, $N = 10^7$ | 2.44 | 1.34 | 0.76 | 0.53 | × 4.6 |
| Block-sparse, $N = 10^8$ | 28.0 | 15.0 | 8.27 | 5.40 | × 5.2 |

Each process computes $1/P$ of the pairs, but every process still receives
the whole output in the forward pass and the whole gradient in the backward
pass: 400 MB and 1.2 GB for $10^8$ points. Splitting pays when the computation
is long compared to these exchanges. From a few seconds on one device
(dense, or block-sparse with $10^8$ points), the speed-up is close to the number
of GPUs. Gloo between CPU nodes exchanges data more slowly than RCCL, so the
block-sparse case with $10^8$ points gains × 5.2 on 8 nodes. Below about a
second on one device, the exchanges dominate.

To measure on your machine:

<!-- skip-test -->
```bash
torchrun --nproc-per-node 4 tools/bench_distributed.py --sizes 1e6 --sparse 1e7 1e8
```
