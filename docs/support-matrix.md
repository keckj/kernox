# Support matrix

Status of kernox 1.0.0 per backend, card and feature. A cell is only
marked "tested" after an actual run on the card named; a
function present in the API but not run on a card is not
advertised for it.

## States

| State | Meaning |
| --- | --- |
| implemented | code present, covered by the CPU tests (oracle) and compilation |
| **tested** | correctness verified on the actual GPU (backend test suite and/or conformance suite, [`conformance.md`](conformance.md)) |
| **qualified** | tested, and performance measured on large cases, correctness verified before timing |

"Qualified" says nothing by itself about performance parity with PyKeOps.

## Cards

| Backend | Compilation | Card | Execution |
| --- | --- | --- | --- |
| `cuda` | NVRTC + CUDA Driver API | GTX 980 Ti (sm_52) | `tests/cuda` suite and conformance |
| `cuda` | | RTX 2080 Ti (sm_75) | `tests/cuda` suite and conformance |
| `cuda` | | RTX 5090 (sm_120) | `tests/cuda` suite |
| `hip` | Torch ROCm's hipRTC | Radeon AI PRO R9700 (gfx1201) | `tests/hip` suite and conformance (including long fuzzing) |
| `sycl` | icpx (oneAPI DPC++ 2026.2) | Intel Arc / Battlemage G31 | `tests/sycl` suite and conformance |
| `oracle` | none (blockwise Torch CPU) | CPU | test reference, on explicit request only |

A single card per vendor apart from NVIDIA. There is no native CPU backend:
`backend="auto"` refuses a CPU tensor (`UnsupportedError`), and `oracle` only
serves as a reference.

## Reductions

All reductions are implemented natively by the three backends and
tested on each conformance card above, in FP32 and FP64 (FP16:
see below).

| Reduction | cuda | hip | sycl |
| --- | --- | --- | --- |
| SUM (optional FP64 acc), COUNT | qualified | qualified | qualified |
| MIN, MAX, ARGMIN, ARGMAX | qualified | tested | tested |
| MIN_ARGMIN, MAX_ARGMAX | tested | tested | tested |
| KMIN, ARGKMIN, KMIN_ARGKMIN (K ≤ 1024) | qualified | qualified (ARGKMIN) | qualified (ARGKMIN) |
| LOGSUMEXP, LOGSUMEXPWEIGHT, SUMSOFTMAXWEIGHT | qualified (LOGSUMEXP) | tested | tested |

Beyond K = 1024: `UnsupportedError` before launch. On sycl, ARGKMIN
remains slower than blockwise Torch for some K ≥ 32.

## Options and features

| Feature | cuda | hip | sycl |
| --- | --- | --- | --- |
| `sum_scheme` direct / block / kahan | tested | tested | tested |
| batch dimensions | tested | tested | tested |
| block-sparse ranges (`with_ranges`) | tested | tested | tested |
| gradients (`torch.autograd`): SUM order 2; LSE, SSMW, MIN/MAX/KMIN order 1 | tested | tested | tested |
| FP16 (FP32 accumulation by default) | tested (emulation on sm_52, native `f16` on sm_53+) | tested (`kp_h`) | tested (`sycl::half`) |
| complex formulas | tested | tested | tested |
| `solve`, `linear_operator` | tested | tested | tested |
| `fp_contract=False` | implemented (`--fmad=false`) | tested (`-ffp-contract=off`) | implemented (`-ffp-contract=off`) |
| `fast_math=True` | FP32 (`ex2.approx`/`lg2.approx`); FP64 unchanged; FP16 refused | refused (`UnsupportedError`) | refused (`UnsupportedError`) |
| kernel variants (`measure`/`patient` autotuning) | tested | tested | tested |

`fast_math` is disabled by default and is never enabled implicitly;
a backend that does not implement it refuses it explicitly.

## Not covered

- Multi-GPU (no test machine has two cards from the same vendor);
  per-device placement is only verified at index 0.
- N ≥ 2³¹ in an actual run.
- Native CPU backend (out of scope; the CPU oracle remains a test reference).
- AMD and Intel cards other than those listed above.
