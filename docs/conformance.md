# Conformance suite

The conformance suite (`tests/conformance/`) checks every native backend against
the same FP64 reference, with the same rules. It is shared by the `cuda`,
`hip`, `sycl` and `cpu` backends. A backend without a PyTorch device or without
its module is **skipped**, never simulated. The `cpu` backend has no streams:
the stream tests are skipped for it.

## Running

```sh
# fast profile (CI) on all available backends
python -m pytest -m "not slow" tests/conformance
# long profile
python -m pytest -m slow tests/conformance
# restrict the backends, force fuzzing seeds
KP_CONFORMANCE_BACKENDS=hip KP_FUZZ_SEEDS=0:500 python -m pytest tests/conformance/test_fuzz.py
```

These are correctness tests: they may run on a shared GPU,
but **never during a benchmark on the same card** (they would skew
its measurements).

## Contents

| File | Role |
| --- | --- |
| `_support.py` | backend discovery, inputs, FP64 reference, error bounds, judgment |
| `test_matrix.py` | systematic matrix (reductions × dtypes × dimensions × K × sizes, over="i", acc_dtype) |
| `test_adversarial.py` | NaN, ±inf, −0.0, denormals, huge/tiny values, duplicates, constants, `where` guarding invalid domains |
| `_fuzz.py`, `test_fuzz.py` | seed-based formula generator, minimization, recorded regressions (`regressions/*.json`) |
| `test_keops_port.py` | targeted port of PyKeOps tests (provenance in `THIRD_PARTY.md`) |
| `test_where_cse.py` | region-based CSE: NaN, ±inf, 1/0 in the inactive branch, against the FP64 reference |
| `test_autograd.py` | gradients of SUMs: dense Torch (FP32/FP64), FP64 `gradcheck` and `gradgradcheck` |
| `test_execution.py` | execution rules: non-default stream/queue, zero elements, destruction during execution, no host copy, multi-GPU |
| `test_softmax.py`, `test_reductions_v08.py` | LOGSUMEXP family, MIN_ARGMIN, MAX_ARGMAX, special values |
| `test_reduction_gradients.py` | gradients of the LOGSUMEXP family and of MIN, MAX, KMIN |
| `test_sum_scheme.py` | `direct`, `block` and `kahan` on ill-conditioned sums |
| `test_batch.py`, `test_ranges.py` | batch dimensions and block-sparse ranges |
| `test_fp16.py` | FP16 against the oracle and the FP64 reference |
| `test_fast_math.py` | approximate `exp`/`log` within their error bounds, and actually active |
| `test_flush_denormals.py` | flushing subnormal numbers: within the IEEE bound plus the flushed mass, actually active on cpu and cuda, inherited by gradients |
| `test_complex.py`, `test_ops.py`, `test_linalg.py` | complex formulas, PyKeOps formula operations, `solve` and linear operators |

Matrix: dimensions 1, 2, 3, 7, 16, 64, 65; K = 0, 1, 2, 31, 32, 33, 64, 128,
256, N−1, N, N+1; sizes at tile and block boundaries (1, 31, 32, 33, 255,
256, 257), multiples, primes (1031, 4099, 8191, 12289, 65537), M≪N, M≫N,
M = 0, N = 0; several i and j arguments, scalar and
vector parameters, vector outputs. The fast profile covers each dimension,
each K and each size family; the `slow` profile does the full
product (limited to M·N·D ≤ 8·10⁶ for the host reference).

## Reference and tolerances

The formula is evaluated term by term in FP64 (`evaluate_numpy`, symox's NumPy
lambdify, floating-point constants passed as arguments: lambdify, like
SymPy, prints them with 15 significant digits) on the inputs **rounded to the dtype**, then reduced according to the
contract (NaN ignored, ties to the smallest j, order (value, j), padding
(+inf, −1)). For each term, an evaluation error bound `T_ij`:

- `T_ij = 0` if an independent FP32 evaluation (Torch CPU oracle) reproduces
  the FP64 value exactly: the comparison is then **exact**, ties
  included (integer data of the matrix);
- otherwise `T_ij = 4·(E_ij·s + C_ij) + 4·u·|F_ij| + 4·tiny`, where `E_ij` is the
  measured FP32/FP64 discrepancy, `s = 1` for an FP32 plan and `u64/u32 = 2⁻²⁹` for an
  FP64 plan (error proportional to the unit roundoff), `C_ij` the largest
  FP64 deviation of the term when all inputs are perturbed by a factor `1 ± u`
  (4 fixed sign draws: conditioning, cancellations, function near a
  zero; a single FP32 evaluation may happen to be right by chance), `u` the unit
  roundoff of the dtype and `tiny` the smallest normal (denormals, possible
  flush-to-zero);
- a term whose class (finite, NaN, sign of infinity) differs between the two
  evaluations is **indeterminate**: its row is not judged.

Judgment per reduction:

| Reduction | Criterion |
| --- | --- |
| SUM | `|r − ref| ≤ Σ_j T_ij + γ_N·L1 + 4·tiny`, `γ_N = N·u_acc/(1 − N·u_acc)` (summation in any order), `L1 = Σ_j |F_ij|`; **exact** if all terms are exact, multiples of 1/2 and `2·L1 < 2⁻²/u_acc` |
| COUNT | exact, except for terms whose predicate differs between FP32 and FP64 (at most one per ambiguous term) |
| MIN / MAX | exact if the row is exact or the extremum is infinite; otherwise `|r − ref| ≤ max_j T_ij` |
| ARGMIN / ARGMAX | exact (smallest j) if the row is exact; otherwise the returned index must be ε-optimal (`F − T ≤ ref + T_ref`), and exact if it is the only ε-optimal one; −1 only without candidate |
| KMIN / ARGKMIN | exact (+inf, −1) padding, sorted values; exact if the row is exact; otherwise values within `max T`, distinct indices, non-NaN, ε-equivalent rank by rank |

The dtypes, shapes and devices of the outputs are checked on every call.

## Execution rules (`test_execution.py`)

- **Non-default stream / queue**: inputs written on a stream `s` behind
  ~40 2048² products (explicit data dependency), call under
  `torch.cuda.stream(s)` / `torch.xpu.stream(s)`, output consumed on `s`,
  synchronization of `s` alone. Variant: output read on a second stream
  after `wait_stream`. Sensitivity check (980 Ti): by forcing the
  launch on the default stream, both tests fail (zero inputs read).
- **Zero elements**: M = 0, N = 0, M = N = 0 for the 9 reductions (identities of the
  contract: 0, +inf, −inf, −1, padding (+inf, −1)), K = 0 (shape (M, 0)),
  gradient of a SUM with N = 0 (zero gradients, shapes preserved).
- **Destruction during execution**: `del` of the `Reduction` and of the compiled
  object, `gc.collect()`, new compilation and new launch, while
  a 32768 × 131072 kernel is still queued (checked by an unfinished
  event); the result is compared with the FP64 reference on a sample of
  rows. The test reports whether the kernel was still running and whether the compiled object
  was collected (the backend's caches may keep it alive: allowed).
- **No host copy**: second call (after compilation and selection),
  tensor parameters or Python floats already cached. Two checks:
  a `TorchDispatchMode` that flags any ATen op mixing host and device or
  reading a scalar (`_local_scalar_dense`), on all backends; then the
  Torch profiler with device activity (`CUDA`, including under ROCm, or
  `XPU`) when it is available, without any `Memcpy HtoD/DtoH` event. A
  native copy outside Torch is only seen by the profiler. A **new** Python
  float for a parameter creates a device tensor (input H2D copy,
  see [Concepts](concepts.md#variables)); outside the scope of `launch`.
- **Multi-GPU**: run only if Torch sees at least two devices of the
  same backend (outputs on the right device, reference, rejection of inputs on two
  devices). **Limitation**: the test machines have no two cards from the
  same vendor, and a PyTorch build sees only one vendor, so this test is
  currently skipped, with this reason. Per-device placement (tensor's `device_index`,
  current stream of that device) is only checked at index 0.

## Fuzzing

`_fuzz.Generator(seed)` produces, reproducibly, trees over
`x[c]`, `y[c]`, `p[c]` and Python constants (integers, floats), with
`+ − × /`, unary minus, `abs sin cos tanh atan exp log sqrt rsqrt square
clamp floor minimum maximum atan2`, integer powers (2, 3, −1) and
rational powers (1/2, 3/2, −1/2), nested `where`s and domains guarded by
`where` (`log`, `sqrt`, division), comparisons, `eq`/`ne`, `& | ^ ~`.
Reduction, K, axis, dtype, sizes and data (normal or integer) are drawn
together with the formula. On failure, the tree and the sizes are minimized then
recorded in `tests/conformance/regressions/<backend>-seed<s>-….json`
(frontend text, data regenerated from the seed); `test_regression` replays
each file on all available backends.

## Known limitations

- The matrix accepts an explicit `UnsupportedError` for K > 256 and reports
  it as a warning; for K ≤ 256 a refusal is a failure. The four backends
  currently support K up to 1024.
- The frontend brings the complex artifacts of the symbolic engine back to real
  arithmetic (`builder.realify`): `I`/`zoo` in a branch value of a
  `Piecewise` (e.g. `sqrt(where(c, a, -1.5))` distributed over the branches)
  become `nan`, `re(a)` → `a`, `im(a)` → 0. Outside a branch, they remain
  rejected (`UnsupportedError`).
- The reference and the judgment run on the host: long profile bounded in M·N·D.
- The FP64 error bound is derived from the scaled FP32 discrepancy; it is
  not a rigorous bound for every formula, but an estimate
  proportional to the measured conditioning, inflated by a factor of 4.
