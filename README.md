# kernox

<p align="center">
  <img src="docs/images/kernox-banner.png" alt="kernox logo" width="560">
</p>

kernox computes pairwise reductions on GPUs and CPUs (Gaussian kernel sums, nearest
neighbours, log-sum-exp, softmax-weighted sums, ...) without ever building the
M×N matrix. You write the formula with symbolic variables indexed by `i` and
`j`, and kernox turns it into a single fused GPU kernel for NVIDIA (CUDA),
AMD (ROCm/HIP) or Intel (SYCL) cards, or into vectorized multithreaded C++ on
x86-64 CPUs. Support for other architectures and accelerators may be added
later.

kernox is a successor to [PyKeOps](https://github.com/getkeops/keops), built on
the [symox](https://github.com/keckj/symox) symbolic engine. It works with
PyTorch tensors.

- Documentation: <https://keckj.github.io/kernox/>
- Getting started: <https://keckj.github.io/kernox/getting-started/>

## Why kernox

- **No M×N matrix.** Memory stays linear in the number of points, so each row
  can be reduced over millions of points.
- **Three GPU vendors and the CPU.** One formula runs on NVIDIA, AMD and Intel
  GPUs, and on x86-64 CPUs (OpenMP, SIMD).
- **Several GPUs or nodes.** `kernox.distributed` splits a reduction across
  the processes of a `torch.distributed` group (NCCL, Gloo), with gradients.
- **Formulas, not strings.** Formulas are Python expressions. Derivatives are
  computed symbolically and run as fused reductions too.
- **Predictable numerics.** IEEE arithmetic by default, an explicit opt-in for
  approximate `exp` and `log`, and an explicit error instead of a silent
  fallback when something is not supported.
- **Familiar to PyKeOps users.** The PyKeOps reductions and formula operations
  are available for PyTorch on GPUs and CPUs. kernox has no NumPy interface;
  the other differences are listed in
  [Coming from PyKeOps](docs/pykeops.md).

## Installation

kernox needs Python 3.10 or later, PyTorch, and a supported GPU or an x86-64
Linux CPU with `g++` 12 or later. Install PyTorch
for your platform first, then kernox from PyPI (symox, its symbolic engine, comes
with it as a dependency):

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu126   # NVIDIA
pip install "kernox[cuda]"
python -m kernox.diagnostics --check
```

On the CPU only:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install kernox
```

AMD and Intel instructions, and the details, are in
[Getting started](docs/getting-started.md).

## Performance

The goal of kernox is to be **at least as fast as PyKeOps** wherever PyKeOps
runs, and to bring the same performance to AMD and Intel GPUs, which PyKeOps
does not support. In every case measured so far, kernox is faster.

M = N = 10⁵, FP32, on an RTX 5090; kernox with `fast_math=True`, PyKeOps 2.3
through `Genred` in its best configuration (each kernel compiled alone in a
fresh process); median of three runs. The last column is the speed-up of
kernox (PyKeOps time / kernox time):

| Case | kernox | PyKeOps | Speed-up |
| --- | --- | --- | --- |
| Log-sum-exp, D = 2 | 3.60 ms | 7.02 ms | × 1.95 |
| Log-sum-exp, D = 3 | 4.36 ms | 8.03 ms | × 1.84 |
| Log-sum-exp, D = 2, block-sparse ranges | 0.249 ms | 0.552 ms | × 2.22 |
| Log-sum-exp, D = 3, block-sparse ranges | 0.359 ms | 0.739 ms | × 2.06 |
| Gaussian kernel sum, D = 3 | 3.49 ms | 5.24 ms | × 1.50 |
| Gaussian kernel sum, D = 3, block-sparse ranges | 0.321 ms | 0.563 ms | × 1.76 |
| 10 nearest neighbours, D = 3 | 5.29 ms | 8.70 ms | × 1.64 |

These figures cover only the cases listed; they are not a general claim of
superiority over PyKeOps on every workload or card. A case where kernox is
slower than PyKeOps is treated as a bug. To reproduce:
`python tools/bench_pykeops.py --device cuda --repeat 3`.

On the CPU, kernox uses every core and SIMD instructions (AVX2, AVX-512) with
glibc's vectorized math functions. Against the CPU backend of PyKeOps 2.3,
M = N = 10⁵, FP32, IEEE arithmetic on both sides, on a node with 2 × AMD
EPYC 9654 (192 cores, 192 threads), same protocol as above:

| Case | kernox | PyKeOps | Speed-up |
| --- | --- | --- | --- |
| Log-sum-exp, D = 2 | 35.2 ms | 344 ms | × 9.8 |
| Log-sum-exp, D = 3 | 37.4 ms | 374 ms | × 10.0 |
| Log-sum-exp, D = 2, block-sparse ranges | 1.35 ms | 21.6 ms | × 16.0 |
| Log-sum-exp, D = 3, block-sparse ranges | 1.72 ms | 33.7 ms | × 19.6 |
| Gaussian kernel sum, D = 3 | 47.2 ms | 343 ms | × 7.3 |
| Gaussian kernel sum, D = 3, block-sparse ranges | 1.94 ms | 36.1 ms | × 18.6 |
| 10 nearest neighbours, D = 3 | 13.8 ms | 110 ms | × 7.9 |

Against PyTorch computing the distance matrix by blocks (same node, 192
threads, FP32, M = N = 20,000):

| Case | kernox | PyTorch | Speed-up |
| --- | --- | --- | --- |
| Gaussian kernel sum, D = 3 | 2.1 ms | 242 ms | × 113 |
| Log-sum-exp, D = 3 | 1.7 ms | 157 ms | × 91 |
| 10 nearest neighbours, D = 3 | 1.4 ms | 113 ms | × 79 |
| Gaussian kernel sum, D = 16 | 3.4 ms | 170 ms | × 50 |

Details, thread scaling and larger sizes: [CPU backend](docs/backend-cpu.md).

## Learn more

- [Getting started](docs/getting-started.md): installation and a first
  reduction, step by step.
- [Concepts](docs/concepts.md): variables, formulas, reductions, devices and
  dtypes.
- How-to guides: [kernel sums](docs/guides/kernel-sums.md),
  [nearest neighbours](docs/guides/nearest-neighbours.md),
  [log-sum-exp and Sinkhorn](docs/guides/logsumexp.md),
  [gradients](docs/guides/gradients.md),
  [linear systems](docs/guides/linear-systems.md),
  [several GPUs or nodes](docs/guides/distributed.md) and more.
- [Coming from PyKeOps](docs/pykeops.md): a migration guide.
- [Backends and hardware](docs/backends.md): what runs where, and the known
  limitations.
- [CPU backend](docs/backend-cpu.md): how the CPU kernels are built, and
  measured performance.

## About the name and logo

symox is *symbolic* plus *ox*: a rusty ox, since symox is written in Rust.
kernox is *kernels* plus the ox of symox, and also a nod to the redox chemistry
of copper. The logo goes from orange copper (the ox) to turquoise verdigris
(the trails and the compute grid), that is, from copper to its oxidation
patina. And redox contains *reduction*, which is what kernox does: sum, min and
max, as drawn in the logo.

## License

At your option, [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
Provenance of third-party code and algorithms (KeOps, symox):
[`THIRD_PARTY.md`](THIRD_PARTY.md).

## Projects using kernox

- [remblais](https://github.com/keckj/remblais): a faster GeomLoss (Sinkhorn
  divergences, kernel norms and optimal transport) on NVIDIA, AMD and Intel
  GPUs and on CPUs. Documentation: <https://keckj.github.io/remblais/>

## Related projects

- symox, symbolic engine and code generation: <https://github.com/keckj/symox>
- KeOps / PyKeOps: <https://github.com/getkeops/keops>
