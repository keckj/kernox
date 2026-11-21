# Changelog

All notable changes to kernox. Versions follow
[semantic versioning](https://semver.org/): a minor version adds features
without breaking code written for the previous one. Each section below is also
the text of the corresponding [GitHub release](https://github.com/keckj/kernox/releases).

## 1.2.0 (2026-11-21)

One reduction can now run on several GPUs or nodes: `kernox.distributed`
splits it across the processes of a `torch.distributed` group, from
`torchrun`, `mpirun`, `srun` or OAR, with a step-by-step guide for clusters.

### Added

- `kernox.distributed`: one reduction split across the processes of a
  `torch.distributed` group, NCCL on GPUs or Gloo on CPUs. Every process passes
  the same inputs and receives the full result, with autograd.
  `split="output"` shares the output rows; `split="reduced"` shares the
  reduced axis and merges the partial results of every reduction with the
  semantics of a single call. Block-sparse ranges are split by number of
  pairs. `init_from_env()` starts the group under `torchrun`, `mpirun`, `srun`
  or OAR. See the guides [Several GPUs or nodes](https://keckj.github.io/kernox/guides/distributed/)
  and [Running on a cluster](https://keckj.github.io/kernox/guides/cluster/).
- `tools/bench_distributed.py`: strong scaling of a dense and a block-sparse
  Gaussian kernel product, forward and backward; the block-sparse case keeps
  the 27 neighbouring cells of a grid and runs up to 10⁸ points.
  Measured on 8 AMD MI210 (2 nodes): × 8.0 for a dense product of 10⁷
  points, × 7.7 for a block-sparse one of 10⁸ points; × 7.2 on 8 CPU nodes
  (2 × EPYC 9654) for a dense product of 3 × 10⁶ points. See
  [Scaling](https://keckj.github.io/kernox/guides/distributed/#scaling).

### Fixed

- `with_ranges(..., validate=True)` no longer reports empty row blocks or
  empty intervals as overlapping their neighbours.

## 1.1.1 (2026-11-21)

A performance fix for the `cpu` backend on large nodes, and measurements
against PyKeOps.

### Fixed

- `cpu` backend: small reductions no longer slow down the PyTorch operations
  that follow them on large nodes. The kernel now always runs with the full
  OpenMP team of PyTorch instead of one thread per task; changing the team
  size cost libgomp about 10 ms on 192 pinned threads. A Sinkhorn divergence
  of remblais on 10³ points goes from 12 ms to 3.8 ms on 2 × EPYC 9654.

### Added

- `tools/bench_pykeops.py`: benchmark against PyKeOps on a GPU or on the CPU,
  one fresh process and one fresh PyKeOps build folder per measurement.
- Documentation: measurements against PyKeOps 2.3 on an RTX 5090 (× 1.5 to
  × 2.2) and on 2 × AMD EPYC 9654 (× 7.3 to × 19.6); speed-ups are now given as
  PyKeOps time / kernox time.

## 1.1.0 (2026-11-20)

kernox now runs on CPUs: a native, multithreaded and vectorized `cpu`
backend joins `cuda`, `hip` and `sycl`. A new compile option,
`flush_denormals`, lets backends flush subnormal numbers to zero.

### Added

- **Native CPU backend `cpu`** (Linux x86-64, `g++` ≥ 12 with OpenMP, a CPU
  build of PyTorch). Each reduction is generated as C++17, compiled with
  `g++ -O3 -march=native -fopenmp` and loaded through `ctypes`, without any
  dependency on the PyTorch C++ ABI.
  - Every core: rows are split across OpenMP threads, and the `j` axis too
    when there are few rows. A reduction uses `torch.get_num_threads()`
    threads from the OpenMP pool it shares with PyTorch, so
    `torch.set_num_threads` and `OMP_NUM_THREADS` apply to both; results are
    bitwise identical whatever the number of threads.
  - Every vector unit: the inner loop over `j` is vectorized for the host by
    `-march=native` (AVX2, AVX-512, native AVX512-FP16 when present), and
    `exp`, `log`, `sin`, `cos`, … go through glibc's vector math library
    (libmvec).
  - Same rules as the GPU backends: IEEE arithmetic (never `-ffast-math`),
    accumulators per `acc_dtype`, every `sum_scheme`, FP16 through
    `_Float16`, block-sparse `ranges`, batch dimensions, gradients. The whole
    conformance suite passes.
  - Compiled modules are cached on disk (`$KERNOX_CACHE_DIR/cpu`, default
    `~/.cache/kernox/cpu`). The key holds the source, the exact compiler
    version and the target CPU, so a cache shared over NFS between different
    machines never hands a binary to a CPU that cannot run it. Concurrent
    processes compile each kernel only once.
  - Speed, float32, M = N = 20,000, on 2 × AMD EPYC 9654 (192 threads), in
    billions of pairs per second against a blockwise PyTorch implementation:
    Gaussian kernel sum 187 vs 1.65 (× 113), log-sum-exp 233 vs 2.55 (× 91),
    10 nearest neighbours 280 vs 3.55 (× 79).
  - `KERNOX_CPU_CXX` selects another compiler, `KERNOX_CPU_LIBMVEC=0`
    disables libmvec. `kernox.diagnostics` reports the CPU, the compiler,
    the number of threads, the vector functions and FP16 support.
  - Documentation: [CPU backend](https://keckj.github.io/kernox/backend-cpu/)
    (design, vector math, threads, tuning, measured performance).
- **`flush_denormals` compile option** (`kp.compile(..., flush_denormals=True)`,
  default `False`). It allows the backend to flush subnormal numbers (below
  about 10⁻³⁸ in float32, 10⁻³⁰⁸ in float64) to zero, as PyKeOps' `-use_fast_math`
  does. Operations that produce subnormal numbers are many times slower on
  CPUs, and narrow Gaussian kernels produce many: with the option, a Gaussian
  kernel sum with a small bandwidth runs 4.8 times faster on an Intel Xeon
  and 1.5 times faster on an AMD EPYC. Results only lose terms smaller than
  the smallest normal number.
  - `cpu`: FTZ and DAZ set for the kernel's threads only, and restored
    afterwards: PyTorch, which shares the threads, still sees subnormal
    numbers.
  - `cuda`: NVRTC `--ftz=true` (float32; float64 is unchanged).
  - `oracle`: accepted, subnormal numbers kept.
  - `hip`, `sycl`: refused with `UnsupportedError`.
  - Part of the compilation key only when enabled (existing caches stay
    valid); gradients inherit it. See
    [Numerics](https://keckj.github.io/kernox/numerics/#flushing-subnormal-numbers).

### Changed

- `backend="auto"` on CPU tensors now selects the `cpu` backend, where kernox
  1.0 raised an error asking for `backend="oracle"`. The oracle is still
  available explicitly.

### Upgrading

Nothing to change in existing code. To run on the CPU, install a CPU build of
PyTorch and `g++` 12 or later, and pass CPU tensors.

## 1.0.2 (2026-11-15)

### Fixed

- **Intel GPUs: installing `kernox[xpu]` no longer replaces the XPU build of
  PyTorch.** The `xpu` extra required `intel-sycl-rt>=2026.2.0`, while
  `torch 2.15.0+xpu` pins `intel-sycl-rt==2026.1.2`: to satisfy both, pip
  installed the CUDA build of PyTorch from PyPI. The extra now only requires
  `torch`, and
  [Getting started](https://keckj.github.io/kernox/getting-started/#intel-gpus)
  explains how to align the SYCL runtime with `icpx`
  (`pip install --no-deps`).
- When a compiled SYCL module fails to load with an undefined symbol, the
  error now says that the SYCL runtime is older than `icpx` and points to that
  section.

## 1.0.1 (2026-11-15)

kernox is installable from PyPI: `pip install kernox[cuda]` (or `[rocm]`,
`[xpu]`).

### Changed

- **symox comes from PyPI.** symox now ships prebuilt wheels: the git install
  and the Rust toolchain are no longer needed. `symox>=0.1.1` is a regular
  dependency, and the diagnostics warn about missing symox capabilities
  instead of checking a pinned build revision.
- All messages (errors, warnings, logs, diagnostics), comments and docstrings
  are in English. Generated kernel sources changed, so kernel caches are
  rebuilt once.
- Documentation rewritten for users and published at
  <https://keckj.github.io/kernox/>: getting started, concepts, formula
  operations, numerics, eleven how-to guides, a migration guide from
  PyKeOps with a feature parity table, backends and hardware, contributing.
  Every Python example of the documentation runs in the test suite.

### Added

- Release workflow: each version is tested, built, published to PyPI through
  Trusted Publishing with build provenance attestations (PEP 740), and
  released on GitHub.

## 1.0.0 (2026-11-15)

First release. kernox writes pairwise computations (Gaussian sums, nearest
neighbours, log-sum-exp, softmax-weighted sums, …) as symbolic formulas on
variables indexed by `i` and `j`, and runs them as fused GPU reductions: the
M×N pairwise matrix is never materialized.

- **GPUs from three vendors:** NVIDIA (`cuda`, NVRTC), AMD (`hip`, hipRTC from
  PyTorch ROCm) and Intel (`sycl`, oneAPI `icpx`).
- **A successor to PyKeOps:** sum, min/max, argmin/argmax, k-min/argkmin,
  log-sum-exp (with weights), softmax-weighted sum, batch dimensions,
  block-sparse `ranges`, summation schemes, FP16, complex formulas, `solve`
  and linear operators, gradients through `torch.autograd`.
- **Built on [symox](https://github.com/keckj/symox):** formulas are symbolic
  expressions, derivatives are computed symbolically and run as fused
  reductions, device code is generated for each backend.
- **No silent fallback:** an unsupported operation, dtype or device raises an
  explicit error before any launch. The `oracle` backend, a blockwise PyTorch
  reference, runs only when requested.
- Autotuning, `fast_math` (approximate `exp` and `log` on NVIDIA GPUs) and a
  conformance suite shared by every backend.
