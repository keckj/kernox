# CPU backend (x86-64, OpenMP)

This page explains how the `cpu` backend compiles and runs its kernels, why,
and how fast it is. Code: [`kernox/backends/cpu/`](../kernox/backends/cpu/).
Environment: a CPU build of PyTorch, `g++` ≥ 12 with OpenMP, Linux x86-64.
Tests: `tests/cpu/` and the conformance suite (`KP_CONFORMANCE_BACKENDS=cpu`).
Benchmark: [`tools/bench_cpu.py`](../tools/bench_cpu.py).

## Toolchain choice

**C++17 generated per plan, compiled by `g++ -O3 -march=native -fopenmp` into a
shared library loaded through `ctypes`** (`toolchain.py`).

- `ctypes` and an `extern "C"` entry point: no Torch headers and no coupling
  with the Torch C++ ABI. The call releases the GIL.
- One OpenMP runtime. The module links against `libgomp.so.1`; the CPU wheels
  of PyTorch ship a `libgomp.so.1` with the same soname, already loaded by
  `import torch`, so the dynamic loader reuses it: kernox and PyTorch share
  one thread pool. The generated code only uses `omp parallel`
  (`GOMP_parallel`) and `omp_get_thread_num`.
- `-march=native`: the binary uses every vector extension of the host
  (see [Vector instruction sets](backends.md#vector-instruction-sets-on-the-cpu)).
  The cache key holds the architecture resolved by the compiler and the full
  list of target options it implies, so a cache shared over NFS between
  different CPUs never hands a binary to a CPU that cannot run it.
- `KERNOX_CPU_CXX` replaces `g++`. Clang is not tested.

Numeric options: **never fast-math**. `-fno-math-errno` (lets `sqrt` compile to
one instruction) and `-fno-trapping-math` (lets `where` branches compile to
blends) change no value. `-fexcess-precision=standard` rounds every `_Float16`
operation to binary16. Contraction follows the plan: `-ffp-contract=fast`, or
`-ffp-contract=off` when contraction is disabled. `fast_math=True` is accepted
and generates the same IEEE code (option `fast_math=ieee` in
`CompileInfo.options`): the approximation is allowed, not required, and the
code is already vectorized. `flush_denormals=True` sets FTZ and DAZ in the
MXCSR register of each thread at the start of the parallel region and restores
it at the end (option `flush_denormals=ftz+daz`): PyTorch shares the OpenMP
threads and the calling thread, and must not inherit the flags.

Cache: `$KERNOX_CACHE_DIR/cpu` (default `~/.cache/kernox/cpu`). Key = sha256 of
the source (which contains the plan), generator version, exact compiler
version, options and target. Atomic write (temporary file, then `os.replace`),
one `fcntl` lock per key against concurrent compilation. The source is kept
next to the binary (`<key>.cpp`) for diagnostics.

## Kernel structure

- **Task**: a block of at most `bi` rows `i` against one slice of the `j` axis
  (with ranges: against the intervals of one work-list tile). Tasks are handed
  out dynamically to the threads of a single parallel region through an
  atomic counter.
- **j tile**: a task walks its slice in tiles of `tj` points. Each tile of the
  `j` operands is transposed into a thread-local SoA buffer (`yb[c * tj + t]`)
  and reused by the `bi` rows of the task, so it must stay in the L1 cache
  (see [Default j tile](#default-j-tile)). Above 64 `j` components in one
  operand, or 128 in all, the operands are read in place instead and the loop
  over `j` is not vectorized (only the vector operations inside the formula
  may be).
- **Evaluation**: per row and per tile, the formula is evaluated over the tile
  in an `omp simd` loop, one SIMD lane per `j` point; the `i` row and the
  parameters are uniform. `fmin`/`fmax` are replaced by branch-free helpers
  (NaN gives the other operand, as in C) so that the loop vectorizes.
- **Reduction**: the tile values are reduced with 16 explicit lane
  accumulators folded in a fixed pairwise tree, then merged into the row state.
  Sums: one fresh accumulator per tile added to the row total (block
  summation with block `tj`, for `direct` and `block`; published as
  `sum_block`), Neumaier per lane for `kahan`. LOGSUMEXP family: maximum of the
  tile first, then one exponential per term. Top-K: sorted list per row, with
  a vectorized prefilter that skips groups of 16 points that cannot enter it.
- **j split**: when there are too few row blocks to occupy the threads, the
  `j` axis is cut into `S` slices (multiples of `tj`, at least 8 tiles each)
  so that rows × slices reach about 1024 tasks (variant parameter `tasks`).
  Partial states go to a temporary (at most 256 MiB), merged per row in
  increasing slice order by a second parallel region. `run.compiled(...).last_split`
  gives `(S, slice length)`.

**Determinism**: `tj`, `bi` and the slices depend on the plan and the sizes
only, never on the number of threads. The same call gives bit for bit the
same result with 1 or 192 threads (`tests/cpu/test_cpu_backend.py`).

Threads: `torch.get_num_threads()` (the calling thread plus OpenMP workers).
For stable timings on a multi-socket machine, pin them:
`OMP_PROC_BIND=close OMP_PLACES=cores`.
Every parallel region uses that full team, even when there are fewer tasks
than threads (the idle threads leave at once): PyTorch runs its own parallel
regions in the same OpenMP runtime, and when two regions in a row have
different team sizes, libgomp reorganizes its pinned threads. On 192 pinned
threads, that cost about 10 ms per switch, and it made a Sinkhorn divergence
on 10³ points three times slower (12 ms instead of 3.8 ms).

## Vector math (libmvec)

Without help, a call to `exp` or `sin` keeps the loop scalar: GCC only knows the
vector variants of glibc's math functions under `-ffast-math`. The generated
code declares them itself (`__attribute__((simd("notinbranch")))`) and links
`-lmvec`, without any fast-math option: NaN, infinities, contraction and
signed zeros behave as before, only the libm changes.

- Functions: `exp`, `log`, `sin`, `cos`, `pow`, and when the glibc has them
  `exp2`, `exp10`, `expm1`, `log2`, `log10`, `log1p`, `tan`, `atan`, `atan2`,
  `asin`, `acos`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `erf`,
  `erfc`, `cbrt`, `hypot`. Detected once by a link test; `python -m
  kernox.diagnostics` lists them.
- Accuracy: ≤ 4 ulp measured (glibc 2.43, x86-64), the accuracy class of the
  CUDA and SYCL device libraries. The scalar libm is ≤ 1 ulp.
- The other functions compile to vector instructions directly (`sqrt`,
  `rsqrt`, `abs`, `floor`, `ceiling`, `round`, `mod`, `sign`, `minimum`,
  `maximum`, `clamp`, `relu`, `step`, `where`) or are built from the ones above
  (`sinc`, `sinxdivx`, `xlogx`, `**`). Checked with GCC's vectorization report
  (`-fopt-info-vec`): for every function of
  [Formula operations](operations.md), in float32 and float64, the loop over the
  `j` points of a tile is vectorized.
- Underflowing `exp`: libmvec's vector `exp` hands every lane below about
  −87.3 (float32) or −708.4 (float64) to a scalar fallback, 5 to 8 times
  slower. That is the common case of a log-sum-exp with a small temperature
  (Sinkhorn with a small ε) or a narrow Gaussian kernel. kernox keeps the
  argument in the fast range:
    - In a formula, `exp(x)` is computed as `exp(x + 40) · e⁻⁴⁰` below that
      bound. The result keeps its subnormal value, with one extra rounding,
      and is 0 where the exact result rounds to 0.
    - In the LOGSUMEXP sum, a term below `e⁻⁸⁷` (float32) or `e⁻⁷⁰⁸` (float64)
      is counted as that bound. The sum holds a term equal to 1, so the change
      stays below half an ulp of the result unless there are more than about
      10³⁰ terms.
    - SUMSOFTMAXWEIGHT keeps the exact form, because its weights could scale
      a tiny term up.

  Measured on a Xeon with 8 threads, M = N = 10⁴, D = 3, float32, in
  G pairs/s, with `s` the scale in `exp(−s·|x−y|²)`:

  | Formula | `s` | before | after | after, `flush_denormals=True` |
  | --- | --- | --- | --- | --- |
  | log-sum-exp | 1 | 7.7 | 7.2 | 7.2 |
  | log-sum-exp | 100 | 1.5 | 7.2 | 7.2 |
  | log-sum-exp | 1000 | 0.9 | 7.1 | 7.2 |
  | Gaussian kernel sum | 1 | 7.7 | 5.7 | 5.7 |
  | Gaussian kernel sum | 100 | 1.0 | 1.2 | 5.7 |
  | Gaussian kernel sum | 1000 | 0.9 | 2.4 | 5.7 |

  A Sinkhorn divergence from remblais (N = 10⁴, forward and backward) went
  from 2.4 s to 1.0 s.

  The Gaussian kernel sum still pays for its subnormal terms after the fix:
  kernox keeps them by default, and on Intel CPUs each operation that
  produces one costs a microcode assist. `flush_denormals=True` (see
  [Numerics](numerics.md#flushing-subnormal-numbers)) removes that cost:
  the last column. In float64, the Gaussian kernel sum at `s = 1000` goes
  from 1.5 to 3.0 G pairs/s. Log-sum-exp produces no subnormal numbers and
  does not change.

  On an AMD EPYC 9654 (8 threads, same formulas, float32), log-sum-exp stays
  at 13.2 G pairs/s for every `s`, where it used to fall to 0.9. The Gaussian
  kernel sum goes from 1.7 to 6.5 at `s = 100` and from 0.9 to 7.9 at
  `s = 1000`, but from 15.8 to 10.7 at `s = 1`. Subnormal numbers cost less
  than on Intel CPUs, but still cost: with `flush_denormals=True`, the
  Gaussian kernel sum runs at 10.1 for every `s`, and from 3.8 to 5.2 at
  `s = 1000` in float64.

  A remblais Sinkhorn loss (N = 10⁴, forward and backward) does not depend
  on the option: 732 → 714 ms on the Xeon, 392 ms on the EPYC, same loss to
  the last bit.
- `KERNOX_CPU_LIBMVEC=0` falls back to the scalar libm. It is 4 to 11 times
  slower on formulas with transcendental functions (see
  [Measured performance](#measured-performance)).

## FP16

`float16` uses GCC's `_Float16` (GCC ≥ 12), with every operation rounded to
binary16 (`-fexcess-precision=standard`). On CPUs with AVX512-FP16 (Intel
Sapphire Rapids and later) the operations are native; elsewhere GCC computes
in float and rounds after each operation, with the same results.
Accumulation follows `acc_dtype`, as on the GPU backends.

## Variants

`Variant("tiled_lanes_jsplit", {tj, bi, tasks})`: `j` tile (multiple of 16),
rows per task, target number of tasks for the `j` split. The variant is part
of the cache key and is published in `info.variant`.

- `default_variant` (`"estimate"`): `tj` from the rule below, `bi = 32`,
  `tasks = 1024`. Depends on the plan only.
- `variants(..., "measure")` (4): default; half tile with 16 rows; double
  tile; default tile with 64 rows.
- `variants(..., "patient")` (32): tiles 64–512 × rows 8–64 × tasks 1024/4096.

### Default j tile

The SoA copy of a `j` tile is reread by the 32 rows of a task: it must fit in
the 32 KiB L1 data cache of current x86 cores, next to the tile values and the
row state. The default tile is 256 points, halved (down to 64) while the tile
takes more than 24 KiB (`codegen.default_tile_j`):

| `j` values per point | float32 | float64 |
| --- | --- | --- |
| 1–12 | 256 | 256 |
| 13–24 | 256 | 128 |
| 25–48 | 128 | 64 |
| more than 48 | 64 | 64 |

A Gaussian sum in dimension D has D + 1 `j` values per point (`y` and the
weight `b`): D = 16 keeps 256 points in float32 and gets 128 in float64.

Measured on 2 × EPYC 9654 (M = N = 200,000, 192 threads, 32 rows per task,
billions of pairs per second; default tile in bold):

| Formula | `j` values per point | tile 64 | 128 | 256 | 512 |
| --- | --- | --- | --- | --- | --- |
| gauss, float32 | 4 | 281 | 297 | **305** | 319 |
| lse, float32 | 4 | 184 | 208 | **270** | 284 |
| knn, float32 | 3 | 720 | 776 | **816** | 828 |
| gauss, D = 16, float32 | 17 | 120 | 138 | **143** | 131 |
| gauss, D = 16, float64 | 17 | 60.4 | **60.2** | 51.5 | 39.1 |
| gauss, D = 32, float32 | 33 | 67.3 | **75.8** | 69.1 | 48.0 |
| gauss, D = 32, float64 | 33 | **33.6** | 29.2 | 21.4 | 14.5 |

In dimension 16 and 32, the rule picks the fastest tile, or one within 1 %. Compared with a fixed tile of 256 points, it gains 17 % for D = 16
in float64, 10 % for D = 32 in float32 and 57 % for D = 32 in float64. For
small dimensions a tile of 512 points would gain 1 to 5 %; it is not the
default because it halves the number of `j` slices available to occupy the
threads when M is small.

## Measured performance

Machine: one node with 2 × AMD EPYC 9654 (Zen 4, 2 × 96 cores, AVX-512 with
512-bit vectors), Debian, glibc 2.41, g++ 14.2, PyTorch 2.15.0+cpu, threads
pinned (`OMP_PROC_BIND=close OMP_PLACES=cores`). Throughput in billions of
pairs (i, j) per second, M = N, median of several runs after a warm-up call
(`tools/bench_cpu.py`). Formulas:

- `gauss`: `exp(-|x - y|² / s) * b` summed over `j`, D = 3;
- `lse`: `logsumexp(-|x - y|² / s)` over `j`, D = 3;
- `knn`: `argkmin(|x - y|², K = 10)` over `j`, D = 3;
- `gauss_d16`: `gauss` with D = 16;
- `funcs`: `sin(x₀) cos(y₁) + atan2(x₁, y₀) + tanh(x₂ y₂)` summed over `j`.

**Against PyKeOps** (PyKeOps 2.3, CPU backend, through `Genred`; one fresh
process and one fresh PyKeOps build folder per measurement; same machine,
192 threads; float32, M = N = 10⁵; kernox with its default options, IEEE
arithmetic like PyKeOps, which compiles with `-O3 -fopenmp` without
`-march=native` or fast math). Times in milliseconds, speed-up = PyKeOps time /
kernox time. `ranges`: block-sparse ranges on a grid, 3ᴰ neighbouring cells
kept around each cell:

| Formula | kernox | PyKeOps | Speed-up |
| --- | --- | --- | --- |
| lse, D = 2 | 35.2 | 344 | × 9.8 |
| lse, D = 3 | 37.4 | 374 | × 10.0 |
| lse, D = 2, ranges | 1.35 | 21.6 | × 16.0 |
| lse, D = 3, ranges | 1.72 | 33.7 | × 19.6 |
| gauss, D = 3 | 47.2 | 343 | × 7.3 |
| gauss, D = 3, ranges | 1.94 | 36.1 | × 18.6 |
| knn, D = 3 | 13.8 | 110 | × 7.9 |

The results agree to 2·10⁻⁷ (dense) and 5·10⁻⁶ (ranges) relative, and 99.9998 %
of the nearest-neighbour indices are identical. On 8 threads (M = N = 30,000),
the speed-ups are × 10.3 to × 10.6 for `lse`, × 7.3 for `gauss`, × 4.6 for
`knn` and × 5.2 to × 6.2 with ranges.

**Against PyTorch** (float32, M = N = 20,000, 192 threads). The PyTorch
reference computes the distance matrix by blocks of 4096 rows, then
`exp` + matrix product, `logsumexp` or `topk`:

| Formula | kernox `cpu` | PyTorch | Speed-up |
| --- | --- | --- | --- |
| gauss | 187 | 1.65 | × 113 |
| lse | 233 | 2.55 | × 91 |
| knn | 280 | 3.55 | × 79 |
| gauss_d16 | 119 | 2.36 | × 50 |

**Thread scaling** (float32, M = N = 20,000; measured before the
underflow-safe `exp` described in [Vector math](#vector-math-libmvec), which
costs a wide Gaussian kernel about a third, so the gauss row is about a third
lower today):

| Threads | 1 | 8 | 24 | 48 | 96 | 192 |
| --- | --- | --- | --- | --- | --- | --- |
| gauss | 2.1 | 16.2 | 47.0 | 89.2 | 143 | 254 |
| knn | 1.9 | 15.3 | 45.0 | 87.5 | 148 | 268 |

From 1 to 192 threads, gauss runs 121 times faster. The step from 48 to 96
threads (one full socket) gains only × 1.6, the step from 96 to 192 threads
× 2: probably the clock dropping when the whole socket is busy (not
measured).

**Larger sizes** (192 threads):

| Formula | M = N | float32 | float64 |
| --- | --- | --- | --- |
| gauss | 200,000 | 207 | 99 |
| lse | 200,000 | 263 | 117 |
| knn | 200,000 | 821 | 523 |
| gauss_d16 | 200,000 | 128 | 53 |
| funcs | 200,000 | 119 | 36 |

float64 runs at about 40 % of float32: vectors hold half as many values.

**libmvec** (M = N = 200,000, 192 threads):

| Formula | with libmvec | `KERNOX_CPU_LIBMVEC=0` | Ratio |
| --- | --- | --- | --- |
| gauss, float32 | 207 | 47 | × 4.4 |
| gauss, float64 | 99 | 33 | × 3.0 |
| lse, float32 | 263 | 52 | × 5.1 |
| funcs, float32 | 119 | 10.6 | × 11 |
| funcs, float64 | 36 | 5.9 | × 6.2 |

To reproduce on a compute node:

```sh
export OMP_PROC_BIND=close OMP_PLACES=cores
python tools/bench_pykeops.py --device cpu --threads 192 --size 100000
python tools/bench_cpu.py --threads 1 8 24 48 96 192 --sizes 20000
python tools/bench_cpu.py --threads 192 --sizes 20000 --dtypes float32 --baselines
python tools/bench_cpu.py --threads 192 --sizes 200000 --variants --tile-sweep
KERNOX_CPU_LIBMVEC=0 python tools/bench_cpu.py --threads 192 --sizes 200000 --cases gauss lse funcs
```

## Tests

- `tests/cpu/test_e2e_corpus.py`: the corpus of formulas against the oracle,
  float32 and float64, sizes (97, 301), (300, 1) and (5, 20000).
- `tests/cpu/test_cpu_backend.py`: bit for bit identical results with 1 and 4
  threads for every summation scheme and for sums, LOGSUMEXP,
  SUMSOFTMAXWEIGHT, ARGKMIN and MIN with a split `j` axis; libmvec switch;
  `exp` below libmvec's fast range (subnormal results, special values,
  log-sum-exp); `auto` picks `cpu` for CPU tensors; non-x86-64 hosts are
  refused.
- Conformance: `KP_CONFORMANCE_BACKENDS=cpu python -m pytest tests/conformance`
  (streams are skipped: the backend has none).
