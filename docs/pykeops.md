# Coming from PyKeOps

kernox covers the reductions and formula operations of PyKeOps 2.4 for
PyTorch on GPUs, and adds AMD and Intel GPUs. The main change is in how a
computation is written: in PyKeOps, a `LazyTensor` wraps your data and the
reduction runs immediately; in kernox, you write a formula on named variables,
compile it once, and call it with tensors.

## Side by side

### Gaussian kernel sum

PyKeOps:

<!-- skip-test -->
```python
from pykeops.torch import LazyTensor

x_i = LazyTensor(X[:, None, :])          # (M, 1, 3)
y_j = LazyTensor(Y[None, :, :])          # (1, N, 3)
b_j = LazyTensor(B[None, :, :])          # (1, N, 1)

D_ij = ((x_i - y_j) ** 2).sum(-1)
out = ((-D_ij / (2 * sigma**2)).exp() * b_j).sum(dim=1)
```

kernox:

```python
import torch
import kernox as kp

device = "cuda"
X = torch.randn(1_000, 3, device=device)
Y = torch.randn(2_000, 3, device=device)
B = torch.randn(2_000, 1, device=device)
sigma = 0.5

x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 1)

d2 = ((x - y) ** 2).sum()
run = kp.compile((kp.exp(-d2 / (2 * sigma**2)) * b).sum_over("j"), dtype="float32")
out = run(x=X, y=Y, b=B)
```

What changed:

- `LazyTensor(X[:, None, :])` becomes `Vi("x", 3)`, and `LazyTensor(Y[None, :, :])`
  becomes `Vj("y", 3)`. The tensors are passed at call time, by name.
- `.sum(-1)` (sum of the components) becomes `.sum()`.
- `.sum(dim=1)` (reduction over `j`) becomes `.sum_over("j")`; `dim=0`
  becomes `"i"`.
- Methods such as `.exp()` become functions: `kp.exp(...)`.
- The dtype is given to `compile`, not inferred from the tensors.

### K nearest neighbours

PyKeOps:

<!-- skip-test -->
```python
idx = D_ij.argKmin(K=10, dim=1)
vals, idx = D_ij.Kmin_argKmin(K=10, dim=1)
```

kernox:

```python
knn = kp.compile(d2.argkmin(10, over="j"), dtype="float32")
idx = knn(x=X, y=Y)
vals, idx = kp.compile(d2.kmin_argkmin(10, over="j"), dtype="float32")(x=X, y=Y)
```

### Log-sum-exp and softmax

PyKeOps:

<!-- skip-test -->
```python
lse = (-D_ij).logsumexp(dim=1)
lse_w = (-D_ij).logsumexp(dim=1, weight=b_j)
smw = (-D_ij).sumsoftmaxweight(y_j, axis=1)
```

kernox:

```python
lse = kp.compile((-d2).logsumexp(over="j"), dtype="float32")(x=X, y=Y)
lse_w = kp.compile((-d2).logsumexp(over="j", weight=b), dtype="float32")(x=X, y=Y, b=B.abs())
smw = kp.compile((-d2).sumsoftmaxweight(y, over="j"), dtype="float32")(x=X, y=Y)
```

### Block-sparse ranges

PyKeOps attaches ranges to the `LazyTensor` (`K_ij.ranges = ranges_ij`, often
built with `pykeops.torch.cluster.from_matrix`). In kernox, ranges are passed
to the compiled reduction: `run.with_ranges(r)(x=..., y=...)`, and
`kp.ranges_from_labels(labels_x, labels_y, keep)` builds them from sorted
cluster labels. See [Block-sparse reductions](guides/block-sparse.md).

### Solving a linear system

PyKeOps:

<!-- skip-test -->
```python
a = K_xx.solve(Vi(rhs), alpha=0.1)
```

kernox names the unknown, a `Vj` variable of a compiled sum:

```python
rhs = torch.randn(1_000, 1, device=device)
K = kp.compile((kp.exp(-d2) * b).sum_over("j"), dtype="float32")
a = K.solve("b", rhs, alpha=0.1, x=X, y=X)
```

See [Linear systems](guides/linear-systems.md).

### Options

| PyKeOps | kernox |
| --- | --- |
| `dtype_acc="float64"` | `acc_dtype="float64"` |
| `sum_scheme="direct_sum"`, `"block_sum"`, `"kahan_scheme"` | `sum_scheme="direct"`, `"block"`, `"kahan"` |
| `use_fast_math=True` (the PyKeOps default) | `fast_math=True` (off by default) |
| `backend="GPU"`, `device_id=` | the device of the tensors |
| `backend="CPU"` | CPU tensors (`backend="auto"` or `"cpu"`) |
| `IntCst(3)`, `Var(...)`, formula strings, `Genred` | Python constants and symbolic formulas |
| `ComplexLazyTensor` | `Vi("z", D, complex=True)`, see [Complex formulas](guides/complex.md) |

## Things to watch for

- **Equality.** `x == y` is Python identity on formulas. Use `kp.eq(x, y)`.
- **No truth value.** `if f:` and `and`/`or` on formulas raise `TypeError`; use
  `kp.where`, `&` and `|`.
- **Exact dtypes.** kernox does not convert inputs: a float64 tensor passed to a
  float32 reduction raises `ValidationError`.
- **Contiguous inputs.** Non-contiguous tensors are rejected; slicing columns
  usually requires `.contiguous()`.
- **IEEE by default.** PyKeOps uses approximate `exp` and `log` by default in
  float32; kernox does not, unless `fast_math=True`. Results can therefore differ
  slightly.
- **FP16 sums** accumulate in float32 in kernox, in float16 in PyKeOps. Use
  `acc_dtype="float16"` to reproduce PyKeOps.

## Feature parity

### Supported

| PyKeOps feature | In kernox |
| --- | --- |
| `sum`, `min`, `max`, `argmin`, `argmax` | same names, with `over=` |
| `min_argmin`, `max_argmax` | same names |
| `Kmin`, `argKmin`, `Kmin_argKmin` | `kmin`, `argkmin`, `kmin_argkmin` |
| `logsumexp` (with `weight=`), `sumsoftmaxweight` | same names |
| counting with a boolean formula | `cond.sum_over("j")`, int64 result |
| batch dimensions | yes, see [Batch dimensions](guides/batch.md) |
| block-sparse `ranges` | `run.with_ranges(r)` |
| `sum_scheme` | yes |
| FP16, FP32, FP64 | yes |
| complex formulas | yes, `complex=True` |
| `solve`, linear-operator interface (`matvec`, `rmatvec`, `T`), SciPy compatibility | yes |
| gradients through `torch.autograd` | yes, see the differences below |
| `grad`, `diff`, `grad_matrix`, `divergence`, `laplacian`, `trace_operator` inside a formula | yes |
| formula operations (`relu`, `step`, `round`, `mod`, `sinxdivx`, `normalize`, `weightedsqdist`, `one_hot`, `matvecmult`, `tensorprod`, `keops_tensordot`, `keops_kron`, `bspline`, `softdtw_sqdist`, ...) | yes, see [Formula operations](operations.md) |
| `factorize`, `auto_factorize` | accepted, with no effect (repeated subexpressions are always shared) |

### Differs

| Topic | PyKeOps | kernox |
| --- | --- | --- |
| approximate `exp`/`log` | on by default | off by default (`fast_math=True` to enable) |
| FP16 accumulator | float16 | float32 by default (`acc_dtype="float16"` to match) |
| `"kahan"` scheme | Kahan summation | Kahan-Babuška (Neumaier), slightly more accurate |
| linearity for `solve` | assumed | checked symbolically, `ValidationError` if not linear |
| `solve` default `maxiter` with batched `rhs` | `10 ×` the size of the first dimension | `10·M` |
| gradients of min, max, kmin, log-sum-exp family | supported | first order only (second order: sums only) |
| `kmin`, `argkmin` | vector formulas allowed | scalar formulas, `K ≤ 1024` on GPUs |
| `bspline`, `round`, `one_hot`, `mod` | not all differentiable | differentiable (zero derivative where piecewise constant) |
| `softdtw_sqdist` | not differentiable in `gamma` | differentiable in `x`, `y` and `gamma` |
| `round(x, d)` with `d ≠ 0` | divides by `10^d` | multiplies by `10^-d`: may differ by one unit in the last place |
| `one_hot` near a half-integer, in float64 | rounds `x + 0.5` first | exact comparison: can differ just below a half-integer |
| `clamp`, `clampint` with NaN input | NaN | depends on the GPU's min/max |
| `rsqrt(0)` | 0 | `inf` (`kp.rsqrt` is exactly `1/sqrt`); `normalize` of the zero vector is 0 in both |

### Not supported

| PyKeOps feature | Status |
| --- | --- |
| NumPy interface (`pykeops.numpy`) | not supported, not planned: kernox works with PyTorch only. (The linear operator accepts NumPy arrays for SciPy interoperability.) |
| formula strings and the `Genred` interface | not supported: formulas are Python expressions |

## Performance

M = N = 10⁵, FP32, median of three runs. PyKeOps 2.3 runs through `Genred`
in its best configuration (each kernel compiled alone in a fresh process).
The last column is the speed-up of kernox (PyKeOps time / kernox time).

On an RTX 5090, kernox with `fast_math=True`:

| Case | kernox | PyKeOps | Speed-up |
| --- | --- | --- | --- |
| Log-sum-exp, D = 2 | 3.60 ms | 7.02 ms | × 1.95 |
| Log-sum-exp, D = 3 | 4.36 ms | 8.03 ms | × 1.84 |
| Log-sum-exp, D = 2, block-sparse ranges | 0.249 ms | 0.552 ms | × 2.22 |
| Log-sum-exp, D = 3, block-sparse ranges | 0.359 ms | 0.739 ms | × 2.06 |
| Gaussian kernel sum, D = 3 | 3.49 ms | 5.24 ms | × 1.50 |
| Gaussian kernel sum, D = 3, block-sparse ranges | 0.321 ms | 0.563 ms | × 1.76 |
| 10 nearest neighbours, D = 3 | 5.29 ms | 8.70 ms | × 1.64 |

On a node with 2 × AMD EPYC 9654 (192 cores, 192 threads), against the CPU
backend of PyKeOps, IEEE arithmetic on both sides:

| Case | kernox | PyKeOps | Speed-up |
| --- | --- | --- | --- |
| Log-sum-exp, D = 2 | 35.2 ms | 344 ms | × 9.8 |
| Log-sum-exp, D = 3 | 37.4 ms | 374 ms | × 10.0 |
| Log-sum-exp, D = 2, block-sparse ranges | 1.35 ms | 21.6 ms | × 16.0 |
| Log-sum-exp, D = 3, block-sparse ranges | 1.72 ms | 33.7 ms | × 19.6 |
| Gaussian kernel sum, D = 3 | 47.2 ms | 343 ms | × 7.3 |
| Gaussian kernel sum, D = 3, block-sparse ranges | 1.94 ms | 36.1 ms | × 18.6 |
| 10 nearest neighbours, D = 3 | 13.8 ms | 110 ms | × 7.9 |

To reproduce: `python tools/bench_pykeops.py --device cuda --repeat 3`
(`--device cpu --threads 192` on the CPU). Thread scaling and larger sizes:
[CPU backend](backend-cpu.md).
