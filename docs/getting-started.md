# Getting started

This page installs kernox, checks that it can see your GPU, and walks through
a first reduction.

## Requirements

- Python 3.10 or later and NumPy 2.3 or later.
- **PyTorch.** kernox works on `torch.Tensor` inputs only. There is no NumPy
  interface, and none is planned.
- **A supported GPU**: NVIDIA (CUDA), AMD (ROCm) or Intel (XPU). kernox has no
  native CPU backend, and none is planned. A slow CPU reference exists for
  testing (see [Trying kernox without a GPU](#trying-kernox-without-a-gpu)).
- A Rust toolchain, to build symox (for example with
  [rustup](https://rustup.rs/)).

The [Backends and hardware](backends.md) page lists the tested cards and the
exact versions.

## Install symox

symox is the symbolic engine kernox is built on. It is not on PyPI: install it
from GitHub, at the commit kernox is pinned on. pip fetches
[maturin](https://www.maturin.rs/) automatically; the Rust compiler must
already be installed.

```sh
pip install "symox @ git+https://github.com/keckj/symox@53d9576e9f39db16e9ca8b3483d754385ebaf986"
```

## Install PyTorch and kernox

Install PyTorch from the index that matches your GPU, then kernox with the
matching extra. The extras only set minimum versions; the PyTorch build is
chosen by the index URL.

```sh
# NVIDIA
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "kernox[cuda] @ git+https://github.com/keckj/kernox"

# AMD (no system ROCm needed: kernox uses the one shipped with PyTorch)
pip install torch --index-url https://download.pytorch.org/whl/rocm10.0
pip install "kernox[rocm] @ git+https://github.com/keckj/kernox"

# Intel (the oneAPI DPC++ compiler icpx 2026.2 must be installed on the host)
pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install "kernox[xpu] @ git+https://github.com/keckj/kernox"
```

From a source checkout, use an editable install with the test extra, for
example `pip install -e ".[cuda,test]"`.

## Check the installation

```sh
python -m kernox.diagnostics --check
```

This prints the versions of Python, PyTorch, NumPy and symox, which GPU
backends are available (or exactly why one is not), whether FP64 is supported,
and where the caches live. With `--check`, it also compiles a few small
reductions, runs them on the GPU and compares them with an FP64 reference. The
exit code is 1 if a check fails.

Two more options: `--backend NAME` (`cuda`, `hip` or `sycl`, repeatable)
restricts the report to some backends, and `--json` prints machine-readable
output. The same report is available from Python as
`kernox.diagnostics(check=False, backends=None)`, which returns a dictionary.

## Your first reduction

We compute, for each point `x_i`, a Gaussian-weighted sum over all the points
`y_j`:

```
out[i] = Σ_j exp(-|x_i - y_j|² / (2σ²)) · w_j
```

**1. Pick the device.** Use `"cuda"` for NVIDIA and AMD GPUs, `"xpu"` for
Intel GPUs.

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

The 10,000 × 20,000 matrix of kernel values never exists in memory: each GPU
thread keeps a running sum.

Inputs are checked before anything runs. A wrong shape, a `float64` tensor
passed to a `float32` reduction, tensors on two devices or a non-contiguous
tensor raise `kernox.ValidationError` with a message naming the argument.
kernox never converts or copies your tensors behind your back.

## Trying kernox without a GPU

The `"oracle"` backend evaluates the same reductions with plain PyTorch on the
CPU, block by block. It is a reference used by the tests: it is slow and is
never selected automatically. Ask for it explicitly:

```python
run_cpu = kp.compile(gaussian_sum, dtype="float32", backend="oracle")
out_cpu = run_cpu(x=X.cpu(), y=Y.cpu(), w=W.cpu(), sigma=0.5)
```

With the default `backend="auto"`, CPU tensors raise `kernox.UnsupportedError`.

## Next steps

- [Concepts](concepts.md) explains variables, formulas and reductions in more
  detail.
- The how-to guides start with [kernel sums](guides/kernel-sums.md) and
  [nearest neighbours](guides/nearest-neighbours.md).
- Coming from PyKeOps? Read [Coming from PyKeOps](pykeops.md).
