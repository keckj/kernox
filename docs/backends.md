# Backends and hardware

kernox runs the same formulas on NVIDIA, AMD and Intel GPUs, through three
backends. Support for other accelerators may be added later.

PyTorch is required: inputs are always `torch.Tensor`s, on the device of the
GPU. There is no native CPU backend, and none is planned; the CPU `oracle`
backend is a slow reference used by the tests.

## Requirements

| Backend | GPUs | Compiler and runtime | PyTorch |
| --- | --- | --- | --- |
| `cuda` | NVIDIA | NVRTC (PyPI wheel `nvidia-cuda-nvrtc-cu12` ≥ 12.6) and the CUDA driver | ≥ 2.11, CUDA build (for example the `cu126` index) |
| `hip` | AMD | hipRTC shipped with the ROCm build of PyTorch: no system ROCm needed | ≥ 2.15, ROCm build (`rocm10.0` index) |
| `sycl` | Intel | oneAPI DPC++ compiler `icpx` 2026.2 installed on the host, with the matching SYCL runtime | ≥ 2.15, XPU build (`xpu` index) |

All backends also need Python ≥ 3.10, NumPy ≥ 2.3 and symox at the pinned
commit (see [Getting started](getting-started.md)). The `cuda`, `rocm` and
`xpu` extras of the kernox package pull the right minimum versions.

## Tested cards

| Backend | Cards |
| --- | --- |
| `cuda` | GeForce GTX 980 Ti (sm_52), RTX 2080 Ti (sm_75), RTX 5090 (sm_120) |
| `hip` | Radeon AI PRO R9700 (gfx1201) |
| `sycl` | Intel Arc, Battlemage G31 |

Other cards of the same families are expected to work but have not been
tested. Run `python -m kernox.diagnostics --check` to verify a new card.

## Feature support

Every reduction and every formula operation runs natively on all three
backends, in float16, float32 and float64. The differences are in a few
options:

| Feature | `cuda` | `hip` | `sycl` |
| --- | --- | --- | --- |
| all reductions, `K ≤ 1024` | yes | yes | yes |
| `sum_scheme` direct, block, kahan | yes | yes | yes |
| batch dimensions, block-sparse ranges | yes | yes | yes |
| gradients | yes | yes | yes |
| FP16 | yes (emulated before sm_53) | yes | yes |
| complex formulas, `solve`, linear operators | yes | yes | yes |
| `fp_contract=False` | yes | yes | yes |
| `fast_math=True` | float32 (float64: no effect; float16: refused) | refused | refused |
| autotuning variants | yes | yes | yes |

A feature that a backend does not support raises `UnsupportedError` before
anything runs. kernox never falls back to another backend or to the CPU.

## How each backend works

- **`cuda`** generates CUDA C++ source, compiles it in-process with NVRTC and
  launches it through the CUDA driver API on the current PyTorch stream.
- **`hip`** generates HIP source and compiles it with the hipRTC library that
  ships with PyTorch's ROCm build, so the kernel and PyTorch share the same HIP
  runtime. See [HIP backend](backend-hip.md) for details.
- **`sycl`** generates SYCL source and compiles it with `icpx`, loads it as a shared
  library, and runs it on PyTorch's XPU queue. Kernels are compiled ahead of time
  for known architectures, and to SPIR-V otherwise.

In all cases, the kernel runs asynchronously on the current stream (or queue)
of the input device, like a PyTorch operation. Compiled kernels are cached on
disk.

## Caches and environment variables

| Variable | Purpose |
| --- | --- |
| `KERNOX_CACHE_DIR` | root of all caches (compiled kernels and autotuning wisdom); default `~/.cache/kernox` (`$XDG_CACHE_HOME/kernox` for the `cuda` cache and the wisdom when `XDG_CACHE_HOME` is set) |
| `KERNOX_CUDA_CACHE_DIR` | root used by the `cuda` kernel cache instead of `KERNOX_CACHE_DIR` |
| `KERNOX_CUDA_LIBRARY`, `KERNOX_NVRTC_LIBRARY` | explicit paths to the CUDA driver and NVRTC libraries |
| `KERNOX_ONEAPI_ROOT` | oneAPI installation root, to find `icpx` |
| `KERNOX_SYCL_TARGET` | force the SYCL compilation target |
| `KERNOX_SYMOX_COMPAT` | `pinned` forces the code path of the pinned symox commit (for tests) |

An empty variable counts as unset. `python -m kernox.diagnostics` shows the
cache locations and sizes.

## Known limitations

- `kmin`, `argkmin` and `kmin_argkmin`: scalar formulas only, `K ≤ 1024`.
- `fast_math=True` only on the `cuda` backend.
- Gradients: second order for sums only; the other differentiable reductions
  are first order.
- Block-sparse ranges cannot be combined with batch dimensions.
- One tested card per vendor apart from NVIDIA. Multi-GPU machines and
  reductions with N ≥ 2³¹ have not been tested on real hardware.
- On `sycl`, `argkmin` can be slower than a dense PyTorch computation for some
  `K ≥ 32`.
