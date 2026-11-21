# Getting started

This page installs kernox, checks that it can see your GPU, and walks through
a first reduction.

## Requirements

- Python 3.10 or later and NumPy 2.3 or later.
- **PyTorch.** kernox works on `torch.Tensor` inputs only. There is no NumPy
  interface, and none is planned.
- **A supported GPU**: NVIDIA (CUDA), AMD (ROCm) or Intel (XPU), **or an
  x86-64 CPU** with `g++` (see [Running on the CPU](#running-on-the-cpu)).

The [Backends and hardware](backends.md) page lists the tested cards and the
exact versions.

## About symox

symox is the symbolic engine kernox is built on. It is published on
[PyPI](https://pypi.org/project/symox/) as prebuilt wheels and pip installs it
automatically with kernox. No Rust toolchain is needed, unless pip has to build
symox from source on a platform without a wheel.

## Install PyTorch and kernox

Install PyTorch from the index that matches your GPU, then kernox with the
matching extra. The extras only set minimum versions; the PyTorch build is
chosen by the index URL.

```sh
# NVIDIA
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "kernox[cuda]"

# AMD (no system ROCm needed: kernox uses the one shipped with PyTorch)
pip install torch --index-url https://download.pytorch.org/whl/rocm10.0
pip install "kernox[rocm]"
```

Intel GPUs need one more step, described just below.

From a source checkout, use an editable install with the test extra, for
example `pip install -e ".[cuda,test]"`.

### Intel GPUs

The `sycl` backend compiles kernels with the oneAPI DPC++ compiler `icpx`,
which must be installed on the host (tested with 2026.2). The compiled kernels
are loaded by the SYCL runtime that comes with PyTorch, and that runtime must
be **at least as recent as `icpx`**. PyTorch 2.15 for XPU ships the 2026.1
runtime, so with `icpx` 2026.2, upgrade the runtime in the same environment:

```sh
pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install "kernox[xpu]"
pip install --no-deps intel-sycl-rt==2026.2.0 intel-cmplr-lib-rt==2026.2.0 \
    intel-cmplr-lib-ur==2026.2.0 intel-cmplr-lic-rt==2026.2.0 \
    intel-opencl-rt==2026.2.0 intel-openmp==2026.2.0 umf==1.2.0 tcmlib==1.6.0
```

Use the version of your `icpx` (`icpx --version`) in the last command. The
`--no-deps` install is deliberate: PyTorch pins the exact runtime it was built
with, so `pip check` reports a conflict afterwards, but PyTorch runs correctly
on a newer runtime (the opposite direction, a runtime older than the compiler,
is what fails). If the versions do not match, compiling a reduction fails with
an `undefined symbol` error that points to this section.

Do not install the SYCL runtime packages without `--no-deps`, and do not let
pip resolve them together with `kernox[xpu]` from PyPI only: to satisfy the
pin, pip would replace the XPU build of PyTorch with the CUDA build.

## Check the installation

```sh
python -m kernox.diagnostics --check
```

This prints the versions of Python, PyTorch, NumPy and symox, which GPU
backends are available (or exactly why one is not), whether FP64 is supported,
and where the caches live. With `--check`, it also compiles a few small
reductions, runs them on each available device and compares them with an
FP64 reference. The
exit code is 1 if a check fails.

Two more options: `--backend NAME` (`cuda`, `hip`, `sycl` or `cpu`, repeatable)
restricts the report to some backends, and `--json` prints machine-readable
output. The same report is available from Python as
`kernox.diagnostics(check=False, backends=None)`, which returns a dictionary.

## Your first reduction

We compute, for each point $x_i$, a Gaussian-weighted sum over all the points
$y_j$:

$$
\mathrm{out}_i = \sum_j \exp\left(-\frac{\|x_i - y_j\|^2}{2\sigma^2}\right) \cdot w_j
$$

**1. Pick the device.** Use `"cuda"` for NVIDIA and AMD GPUs, `"xpu"` for
Intel GPUs, `"cpu"` for the CPU.

```python
import torch
import kernox as kp

device = "cuda"
```

**2. Declare the variables.** `Vi` is indexed by `i` (one row per output),
`Vj` by `j` (the rows reduced over), and `Param` is the same for every pair.
The second argument is the dimension.

```python
x = kp.Vi("x", 3)        # tensor of shape (M, 3)
y = kp.Vj("y", 3)        # tensor of shape (N, 3)
w = kp.Vj("w", 1)        # tensor of shape (N, 1)
sigma = kp.Param("sigma")
```

**3. Write the formula.** Python operators build a symbolic expression;
nothing is computed yet. `.sum()` adds the components of a vector, and
`.sum_over("j")` is the reduction.

```python
d2 = ((x - y) ** 2).sum()
kernel = kp.exp(-d2 / (2 * sigma**2))
gaussian_sum = (kernel * w).sum_over("j")
```

**4. Compile.** The dtype is mandatory. The GPU kernel itself is generated and
compiled on the first call, then cached on disk.

```python
run = kp.compile(gaussian_sum, dtype="float32")
```

**5. Call it** with keyword arguments named after the variables. Each call
can use different sizes M and N.

```python
X = torch.randn(10_000, 3, device=device)
Y = torch.randn(20_000, 3, device=device)
W = torch.rand(20_000, 1, device=device)

out = run(x=X, y=Y, w=W, sigma=0.5)
print(out.shape)          # torch.Size([10000, 1])
```

The $10{,}000 \times 20{,}000$ matrix of kernel values never exists in memory: each GPU
thread keeps a running sum.

Inputs are checked before anything runs. A wrong shape, a `float64` tensor
passed to a `float32` reduction, tensors on two devices or a non-contiguous
tensor raise `kernox.ValidationError` with a message naming the argument.
kernox never converts or copies your tensors behind your back.

## Running on the CPU

With CPU tensors, `backend="auto"` selects the native `cpu` backend: kernox
generates C++, compiles it with `g++` and OpenMP (once, then from the disk
cache) and runs it on `torch.get_num_threads()` threads. It needs `g++` 12 or
later on Linux x86-64 and a CPU build of PyTorch; no other dependency.

```python
run_cpu = kp.compile(gaussian_sum, dtype="float32")   # "auto": cpu backend
out_cpu = run_cpu(x=X.cpu(), y=Y.cpu(), w=W.cpu(), sigma=0.5)
```

Set the number of threads with `torch.set_num_threads`; results do not depend
on it. On a machine with several sockets, pinning the threads
(`OMP_PROC_BIND=close OMP_PLACES=cores`) gives steadier timings. On a
192-core AMD EPYC node, a Gaussian sum in dimension 3 over 200,000 points
takes 0.13 s in float32 ([measurements](backend-cpu.md#measured-performance)).

The `"oracle"` backend also evaluates the reductions on the CPU, with plain
PyTorch, block by block. It is the slow reference used by the tests and is
never selected automatically: ask for it with `backend="oracle"`.

## Next steps

- [Concepts](concepts.md) explains variables, formulas and reductions in more
  detail.
- The how-to guides start with [kernel sums](guides/kernel-sums.md) and
  [nearest neighbours](guides/nearest-neighbours.md).
- Coming from PyKeOps? Read [Coming from PyKeOps](pykeops.md).
