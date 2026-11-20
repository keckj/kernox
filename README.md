# kernox

<p align="center">
  <img src="docs/images/kernox-banner.png" alt="kernox logo" width="560">
</p>

kernox computes pairwise reductions on GPUs and CPUs (Gaussian kernel sums, nearest
neighbours, log-sum-exp, softmax-weighted sums, ...) without ever building the
M×N matrix. You write the formula with symbolic variables indexed by `i` and
`j`, and kernox turns it into a single fused GPU kernel for NVIDIA (CUDA),
AMD (ROCm/HIP) or Intel (SYCL) cards, or into vectorized multithreaded C++ on
x86-64 CPUs. Support for other accelerators may be added later.

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
does not support. On NVIDIA GPUs, kernox is already faster than PyKeOps in
several cases.

LogSumExp reduction, M = N = 10⁵, FP32, on an RTX 2080 Ti; kernox with
`fast_math=True`, PyKeOps 2.4b in its best configuration (each kernel compiled
alone in a fresh process); median of two runs:

| Case | kernox | PyKeOps | kernox / PyKeOps |
| --- | --- | --- | --- |
| D = 2, dense | 20.1 ms | 21.9 ms | ×0.92 |
| D = 3, dense | 24.6 ms | 25.8 ms | ×0.95 |
| D = 2, block-sparse ranges | 8.4 ms | 8.9 ms | ×0.95 |
| D = 3, block-sparse ranges | 4.2 ms | 4.2 ms | ×1.01 (same within noise) |

These figures cover only the cases listed; they are not a general claim of
superiority over PyKeOps on every workload or card. A case where kernox is
slower than PyKeOps is treated as a bug.

On the CPU, kernox uses every core and SIMD instructions (AVX2, AVX-512) with
glibc's vectorized math functions. On a node with 2 × AMD EPYC 9654 (192
cores), FP32, M = N = 20,000, against PyTorch computing the distance matrix by
blocks:

| Case | kernox | PyTorch | Speed-up |
| --- | --- | --- | --- |
| Gaussian kernel sum, D = 3 | 2.1 ms | 242 ms | ×113 |
| Log-sum-exp, D = 3 | 1.7 ms | 157 ms | ×91 |
| 10 nearest neighbours, D = 3 | 1.4 ms | 113 ms | ×79 |
| Gaussian kernel sum, D = 16 | 3.4 ms | 170 ms | ×50 |

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
  [linear systems](docs/guides/linear-systems.md) and more.
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
