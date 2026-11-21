# Backends and hardware

kernox runs the same formulas on NVIDIA, AMD and Intel GPUs and on x86-64
CPUs, through four native backends. Support for other accelerators may be added
later.

PyTorch is required: inputs are always `torch.Tensor`s, and the backend is the
one of their device (`backend="auto"`). The `oracle` backend is a separate, slow
PyTorch reference used by the tests; it is never selected automatically.

## Requirements

| Backend | GPUs | Compiler and runtime | PyTorch |
| --- | --- | --- | --- |
| `cuda` | NVIDIA | NVRTC (PyPI wheel `nvidia-cuda-nvrtc-cu12` ≥ 12.6) and the CUDA driver | ≥ 2.11, CUDA build (for example the `cu126` index) |
| `hip` | AMD | hipRTC shipped with the ROCm build of PyTorch: no system ROCm needed | ≥ 2.15, ROCm build (`rocm10.0` index) |
| `sycl` | Intel | oneAPI DPC++ compiler `icpx` 2026.2 installed on the host, and a SYCL runtime at least as recent (see [Intel GPUs](getting-started.md#intel-gpus)) | ≥ 2.15, XPU build (`xpu` index) |
| `cpu` | x86-64 CPUs (Linux) | `g++` ≥ 12 with OpenMP; glibc `libmvec` (used when present) | any build; the CPU wheel shares its OpenMP runtime with kernox |

All backends also need Python ≥ 3.10, NumPy ≥ 2.3 and symox ≥ 0.1.1 from
PyPI (installed with kernox, see [Getting started](getting-started.md)). The
`cuda`, `rocm` and `xpu` extras of the kernox package pull the right minimum
versions.

## Tested cards

| Backend | Cards |
| --- | --- |
| `cuda` | GeForce GTX 980 Ti (sm_52), RTX 2080 Ti (sm_75), RTX 5090 (sm_120) |
| `hip` | Radeon AI PRO R9700 (gfx1201) |
| `sycl` | Intel Arc, Battlemage G31 |
| `cpu` | Intel Xeon (Skylake-AVX512), AMD EPYC 9654 (Zen 4) |

Other cards of the same families are expected to work but have not been
tested. Run `python -m kernox.diagnostics --check` to verify a new card.

## Feature support

Every reduction and every formula operation runs natively on all four
backends, in float16, float32 and float64. The differences are in a few
options:

| Feature | `cuda` | `hip` | `sycl` | `cpu` |
| --- | --- | --- | --- | --- |
| all reductions, `K ≤ 1024` | yes | yes | yes | yes |
| `sum_scheme` direct, block, kahan | yes | yes | yes | yes |
| batch dimensions, block-sparse ranges | yes | yes | yes | yes |
| gradients | yes | yes | yes | yes |
| FP16 | yes (emulated before sm_53) | yes | yes | yes (`_Float16`, binary16 rounding per operation) |
| complex formulas, `solve`, linear operators | yes | yes | yes | yes |
| `fp_contract=False` | yes | yes | yes | yes |
| `fast_math=True` | float32 (float64: no effect; float16: refused) | refused | refused | accepted, no effect (IEEE code) |
| `flush_denormals=True` | float32 (`--ftz=true`; float64: no effect) | refused | refused | yes (FTZ + DAZ, every dtype) |
| autotuning variants | yes | yes | yes | yes |
| streams | current stream | current stream | current queue | none: synchronous call |

A feature that a backend does not support raises `UnsupportedError` before
anything runs. kernox never falls back to another backend or to the CPU.

## How each backend works

![How kernox turns a formula into a kernel: symox, a backend compiler, a cache, a launch](images/pipeline.svg)

- **`cuda`** generates CUDA C++ source, compiles it in-process with NVRTC and
  launches it through the CUDA driver API on the current PyTorch stream.
- **`hip`** generates HIP source and compiles it with the hipRTC library that
  ships with PyTorch's ROCm build, so the kernel and PyTorch share the same HIP
  runtime. See [HIP backend](backend-hip.md) for details.
- **`sycl`** generates SYCL source and compiles it with `icpx`, loads it as a shared
  library, and runs it on PyTorch's XPU queue. Kernels are compiled ahead of time
  for known architectures, and to SPIR-V otherwise.
- **`cpu`** generates C++17 and compiles it with `g++ -O3 -march=native
  -fopenmp` into a shared library loaded with `ctypes`. Rows and blocks of `j`
  are spread over `torch.get_num_threads()` threads; the `j` points are
  transposed per tile so that the formula is evaluated with SIMD instructions,
  with transcendental functions from glibc's vector math library `libmvec`
  (≤ 4 ulp, same accuracy class as the GPU libraries; `KERNOX_CPU_LIBMVEC=0`
  falls back to the scalar libm). No fast-math option is used. How the `j`
  axis is split depends on the sizes only, so results are bit for bit
  identical for any number of threads. Details and measurements:
  [CPU backend](backend-cpu.md).

### Vector instruction sets on the CPU

The `cpu` backend compiles for the host with `-march=native`, so it uses
whatever x86-64 vector extension the CPU has, and `libmvec` provides a
matching variant of each math function:

| CPU | Vectors used | `libmvec` variant |
| --- | --- | --- |
| SSE2 only (x86-64 baseline, before 2008) | 128 bits for arithmetic | none: math functions stay scalar |
| SSE4.2 (x86-64-v2) | 128 bits | SSE (`_ZGVb`) |
| AVX, AVX2 (x86-64-v3) | 256 bits | AVX / AVX2 (`_ZGVc`, `_ZGVd`) |
| AVX-512, AMD Zen 4 and Zen 5 | 512 bits (`zmm`) | AVX-512 (`_ZGVe`) |
| AVX-512, Intel Skylake-SP to Granite Rapids | 256 bits, with AVX-512VL instructions: GCC's tuning for these CPUs prefers 256-bit vectors (`-mprefer-vector-width=256`) | AVX2 (`_ZGVd`) |

The vector width is the compiler's choice for the CPU; kernox does not override
it. The compiled kernels are only valid for the CPU model that built them: the
cache key holds the resolved architecture and every target option, so a
cache shared over NFS between different machines never hands a binary to the
wrong CPU.

Other instruction sets (ARM NEON/SVE, POWER VSX, RISC-V V) are not supported:
on a non-x86-64 or non-Linux host, the `cpu` backend raises `UnsupportedError`.

On GPUs, the kernel runs asynchronously on the current stream (or queue) of
the input device, like a PyTorch operation; on the CPU, the call returns when
the result is written. Compiled kernels are cached on disk.

## Caches and environment variables

| Variable | Purpose |
| --- | --- |
| `KERNOX_CACHE_DIR` | root of all caches (compiled kernels and autotuning wisdom); default `~/.cache/kernox` (`$XDG_CACHE_HOME/kernox` for the `cuda` cache and the wisdom when `XDG_CACHE_HOME` is set) |
| `KERNOX_CUDA_CACHE_DIR` | root used by the `cuda` kernel cache instead of `KERNOX_CACHE_DIR` |
| `KERNOX_CUDA_LIBRARY`, `KERNOX_NVRTC_LIBRARY` | explicit paths to the CUDA driver and NVRTC libraries |
| `KERNOX_ONEAPI_ROOT` | oneAPI installation root, to find `icpx` |
| `KERNOX_SYCL_TARGET` | force the SYCL compilation target |
| `KERNOX_CPU_CXX` | C++ compiler of the `cpu` backend (default `g++`) |
| `KERNOX_CPU_LIBMVEC` | `0` disables glibc vector math in the `cpu` backend (scalar libm, ≤ 1 ulp, slower) |
| `KERNOX_SYMOX_COMPAT` | `pinned` forces the compatibility code path for older symox builds (for tests) |

An empty variable counts as unset. `python -m kernox.diagnostics` shows the
cache locations and sizes.

## Known limitations

- `kmin`, `argkmin` and `kmin_argkmin`: scalar formulas only, `K ≤ 1024`.
- `fast_math=True` only on the `cuda` backend (accepted without effect on `cpu`).
- The `cpu` backend targets Linux x86-64 with GCC only (no ARM, POWER or
  RISC-V, no Windows or macOS, no Clang); see
  [Vector instruction sets](#vector-instruction-sets-on-the-cpu).
- The `cpu` backend has no streams: a call returns when its result is written.
- Gradients: second order for sums only; the other differentiable reductions
  are first order.
- Block-sparse ranges cannot be combined with batch dimensions.
- One tested card per vendor apart from NVIDIA. Multi-GPU machines and
  reductions with N ≥ 2³¹ have not been tested on real hardware.
- On `sycl`, `argkmin` can be slower than a dense PyTorch computation for some
  `K ≥ 32`.
