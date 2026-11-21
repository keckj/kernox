# Running on a cluster

This page takes a kernox script from a laptop to a batch job on several GPUs
or CPU nodes, step by step. [Several GPUs or nodes](distributed.md) explains
what `kd.reduce` splits and how.

![A launcher starts one process per GPU; init_from_env joins them into one group](../images/cluster.svg)

kernox does not need `mpi4py` or an MPI build of PyTorch: `mpirun`, `srun`
or `torchrun` only start the processes, and they talk to each other through
`torch.distributed` (NCCL between GPUs, Gloo over TCP between CPU nodes).

## 1. Write the script once

<!-- skip-test -->
```python
# run.py
import torch
import kernox as kp
import kernox.distributed as kd

info = kd.init_from_env()      # first thing: joins the other processes
device = info.device           # this process's GPU, or the CPU

x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 1)
gauss = kp.compile((kp.exp(-50.0 * x.sqdist(y)) * b).sum_over("j"), dtype="float32")

g = torch.Generator().manual_seed(0)     # the same data on every process
X = torch.rand(1_000_000, 3, generator=g).to(device)
Y = torch.rand(1_000_000, 3, generator=g).to(device)
B = torch.randn(1_000_000, 1, generator=g).to(device)

out = kd.reduce(gauss, x=X, y=Y, b=B)    # the full result, on every process

if info.rank == 0:                       # print and save on one process only
    print(f"{info.world_size} processes: {out.sum().item():.6e}")
```

Three rules:

- **Every process runs the whole script.** Each one must call `kd.reduce`
  the same number of times, in the same order, with inputs of the same
  shapes. A call inside `if info.rank == 0:` hangs the others.
- **Every process passes the same inputs.** Generate them with the same
  seed, or load the same file, or send them from the first process with
  `torch.distributed.broadcast(X, src=0)`.
- **Write on one process only.** Logs, prints and files: `if info.rank == 0`.

The same script runs as a single process, with no change: `python run.py`.

## 2. Test it on your machine

<!-- skip-test -->
```bash
python run.py                          # 1 process
torchrun --nproc-per-node 2 run.py     # 2 processes: 2 GPUs, or the CPU cores shared in two
```

Both print the same value, up to rounding. If the second command works,
the script is ready for the cluster.

## 3. Launch it on the cluster

Pick the case that matches your machine. In every case, `init_from_env`
finds the rank of each process, its GPU and the address of the first node by
itself.

### One node, several GPUs

<!-- skip-test -->
```bash
torchrun --nproc-per-node 4 run.py     # one process per GPU
```

### Several GPU nodes

With SLURM, one task per GPU:

<!-- skip-test -->
```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
srun python run.py
```

Without SLURM, start one `torchrun` per node, with the hostname of the first
node as `--master-addr`:

<!-- skip-test -->
```bash
# on the first node, node0
torchrun --nnodes 2 --node-rank 0 --nproc-per-node 4 --master-addr node0 --master-port 29500 run.py
# on the second node
torchrun --nnodes 2 --node-rank 1 --nproc-per-node 4 --master-addr node0 --master-port 29500 run.py
```

### CPU nodes, with MPI

Run one process per node; it uses all the cores of its node:

<!-- skip-test -->
```bash
export OMP_PROC_BIND=close OMP_PLACES=cores
mpirun -np 8 --map-by ppr:1:node --bind-to none -x OMP_PROC_BIND -x OMP_PLACES python run.py
```

Or one process per socket (here, nodes of two 96-core sockets), each bound
to its cores:

<!-- skip-test -->
```bash
mpirun -np 16 --map-by ppr:2:node:pe=96 --bind-to core -x OMP_PROC_BIND -x OMP_PLACES python run.py
```

`init_from_env` sets the number of threads of each process: all the cores it
may run on, divided between the processes of the node that share them.
Pass `threads=` to choose it yourself.

With SLURM:

<!-- skip-test -->
```bash
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
export OMP_PROC_BIND=close OMP_PLACES=cores
srun python run.py
```

With OAR, Open MPI starts the remote processes through `oarsh`:

<!-- skip-test -->
```bash
#!/bin/bash
#OAR -l /nodes=8
export OMPI_MCA_plm_rsh_agent=oarsh PRTE_MCA_plm_ssh_agent=oarsh
export OMP_PROC_BIND=close OMP_PLACES=cores
mpirun -np 8 --host $(uniq $OAR_NODEFILE | paste -sd,) --map-by ppr:1:node --bind-to none \
       -x PATH -x OMP_PROC_BIND -x OMP_PLACES python run.py
```

Use the `python` of your environment (a virtual environment: its full path,
or `-x PATH` as above), the same on every node.

## 4. Check the setup

Before a long run, print what each process sees, with the same launcher:

<!-- skip-test -->
```bash
mpirun -np 2 --map-by ppr:1:node --bind-to none python -c \
  "import torch, kernox.distributed as kd; i = kd.init_from_env(); print(i, torch.get_num_threads())"
```

Each process prints its own `rank`, its device (a different GPU per process
of a node) and its number of threads (the cores of its node, or its share).

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `init_from_env` never returns | The processes cannot reach the first one. Set `MASTER_ADDR` to the hostname of the first node, and `MASTER_PORT` to a free port if two jobs share a node (default 29500). `init_from_env(timeout=60)` raises an error instead of waiting. |
| Stuck in `kd.reduce`, or a Gloo/NCCL error about sizes | The processes did not make the same calls: a call on one rank only, or inputs of different shapes. |
| Slow across nodes | Gloo or NCCL chose a slow network. Name the fast one (see `ip -br addr`): `GLOO_SOCKET_IFNAME=ib0`, `NCCL_SOCKET_IFNAME=ib0`. |
| Slow on the CPU | Check `torch.get_num_threads()` in each process (step 4). Bind each process to its own cores, or pass `threads=`. |
| Several processes on the same GPU | The launcher did not give a local rank: start with `torchrun` or `srun`, or set `CUDA_VISIBLE_DEVICES` per process. |
| The result differs slightly from one process | Expected: the sums are done in another order. It is the same on every process and from one run to the next. |

## When it pays

Each process computes `1/P` of the pairs, and the processes exchange one
copy of the output. Splitting pays when one process takes a few seconds or more:
see [the measurements](distributed.md#scaling).
