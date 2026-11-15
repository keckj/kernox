# Concepts

A kernox computation has three parts: **variables** that stand for your
tensors, a **formula** that combines them for one pair `(i, j)`, and a
**reduction** that combines the formula over all `j` (or all `i`). You
**compile** the reduction once and call it with tensors as often as you like.

```python
import torch
import kernox as kp

device = "cuda"

x, y = kp.Vi("x", 3), kp.Vj("y", 3)       # variables
d2 = ((x - y) ** 2).sum()                  # formula
nearest = d2.argmin(over="j")              # reduction
run = kp.compile(nearest, dtype="float32") # compile once

X = torch.randn(1000, 3, device=device)
Y = torch.randn(5000, 3, device=device)
idx = run(x=X, y=Y)                        # call: (1000, 1) int64
```

## Variables

| Declaration | Tensor passed at call time | Meaning |
| --- | --- | --- |
| `Vi("x", D)` | `(M, D)` | one row per output row `i` |
| `Vj("y", D)` | `(N, D)` | one row per reduced row `j` |
| `Param("p", D)` | `(D,)`, or Python numbers | the same value for every pair |

- The name is the keyword used at call time: `Vi("x", 3)` is passed as
  `x=...`. It must be a plain ASCII Python identifier, and must not start
  with `_kp` or `_sx`.
- The dimension `D` defaults to 1. A name can only have one role and one
  dimension in a formula.
- All `Vi` inputs share the same number of rows M, all `Vj` inputs the same N.
- A `Param` of dimension 1 accepts a Python number (`sigma=0.5`); a `Param` of
  dimension D accepts a list or tuple of D numbers. Python numbers are
  rounded once to the reduction's dtype; pass a tensor instead if you need a
  gradient with respect to the parameter.

Inputs can also carry leading batch dimensions: see
[Batch dimensions](guides/batch.md).

## Formulas

A formula is a small vector of symbolic expressions. Its dimension is the
number of components. Python operators work as you would expect:

```python
a, b = kp.Vi("a", 3), kp.Vj("b", 3)
s = kp.Param("s")

f = (a - b) * s + 1.0         # dimension 3: a dimension-1 operand is broadcast
g = kp.exp(-f) / (1 + a**2)   # functions apply component by component
```

Python numbers (`int`, `float`, `fractions.Fraction`) are kept exact while the
formula is built and rounded once to the dtype at compile time. A constant that
overflows the dtype, or a non-zero constant that would round to zero, raises
`kernox.ConstantConversionError` instead of silently becoming `inf` or `0`.

Working with components:

| Expression | Result |
| --- | --- |
| `f[0]`, `f[-1]`, `f[1:]`, `f[[0, 2]]` | selected components |
| `kp.cat(f, g)` | concatenation |
| `f.sum()` | sum of the components (dimension 1) |
| `f.sqnorm2()`, `f.norm2()`, `f.dot(g)` | squared norm, norm, dot product |

!!! note "`sum()` versus `sum_over()`"
    `f.sum()` adds the **components** of a formula, for one pair. `f.sum_over("j")`
    is the **reduction** over all `j`. They are different on purpose:
    `f.sum("j")` raises a `TypeError`.

### Conditions

Comparisons (`<`, `<=`, `>`, `>=`) build boolean formulas. Combine them with
`&`, `|`, `^` and `~`, and use `kp.eq(a, b)` and `kp.ne(a, b)` for equality,
because `==` and `!=` keep their usual Python meaning (identity).

`kp.where(cond, a, b)` picks `a` where `cond` holds and `b` elsewhere. Only the
active branch counts, so it can guard an invalid operation:

```python
r = kp.where(d2 > 0, kp.sqrt(d2), 0.0)   # never takes sqrt of a negative number
```

A boolean formula cannot be used in arithmetic (`TypeError`); use `where`.
A formula has no truth value either: `if f:` raises a `TypeError`.

The complete list of functions is on the
[Formula operations](operations.md) page.

## Reductions

| Method | Result for M rows | Output dtype |
| --- | --- | --- |
| `f.sum_over("j")` | `(M, D)`: sum over `j` | float (see [dtypes](#dtypes)) |
| `cond.sum_over("j")` (boolean `cond`) | `(M, D)`: count of `j` where `cond` holds | int64 |
| `f.min(over="j")`, `f.max(over="j")` | `(M, D)` | input dtype |
| `f.argmin(over="j")`, `f.argmax(over="j")` | `(M, D)`: index of the extremum | int64 |
| `f.min_argmin(over="j")`, `f.max_argmax(over="j")` | tuple `(values, indices)`, one pass | input dtype, int64 |
| `f.kmin(K, over="j")` | `(M, K)`: the K smallest values, sorted | input dtype |
| `f.argkmin(K, over="j")` | `(M, K)`: their indices | int64 |
| `f.kmin_argkmin(K, over="j")` | tuple `(values, indices)`, one pass | input dtype, int64 |
| `f.logsumexp(over="j")` | `(M, D)`: `log Σ_j exp f` | float |
| `f.logsumexp(over="j", weight=g)` | `(M, E)`: `log Σ_j exp(f) g` | float |
| `f.sumsoftmaxweight(g, over="j")` | `(M, E)`: `Σ_j exp(f) g / Σ_j exp(f)` | float |

- Every reduction except the K-nearest family works component by component.
  `kmin`, `argkmin` and `kmin_argkmin` need a scalar formula, `K` must be a
  Python `int` (0 is allowed), and GPU backends accept `K ≤ 1024`.
- `logsumexp(weight=g)` and `sumsoftmaxweight` take a scalar score `f` and
  weights `g` of any dimension E.
- Ties go to the smallest index `j`. NaN values are ignored by min, max and the
  K-nearest reductions, and propagate through sums. See [Numerics](numerics.md).
- With no `j` at all (N = 0), each reduction returns its neutral value: 0 for
  sums and counts, `+inf` for min, `-inf` for max and log-sum-exp, `-1` for
  indices.

### Reducing over `i`

Every reduction also accepts `over="i"`. The roles swap: the output has one
row per `Vj` row.

```python
col = d2.min(over="i")
print(kp.compile(col, dtype="float32")(x=X, y=Y).shape)   # torch.Size([5000, 1])
```

A reduction needs at least one `Vi` and one `Vj` variable, so that both M and N
are known.

## Compiling and calling

```python
run = kp.compile(nearest, dtype="float32")
```

`compile` takes the reduction and keyword options:

| Option | Default | Purpose |
| --- | --- | --- |
| `dtype` | required | `"float16"`, `"float32"` or `"float64"` |
| `backend` | `"auto"` | which backend runs it, see below |
| `acc_dtype` | depends on `dtype` | accumulator of sums and log-sum-exp, see [dtypes](#dtypes) |
| `sum_scheme` | `"direct"` | `"direct"`, `"block"` or `"kahan"`, see [Numerics](numerics.md#summation-schemes) |
| `fp_contract` | `True` | allow fused multiply-add, see [Numerics](numerics.md#fused-multiply-add) |
| `fast_math` | `False` | approximate `exp` and `log`, see [fast_math](guides/fast-math.md) |
| `tuning` | `"estimate"` | kernel autotuning, see [Autotuning](guides/autotuning.md) |

The same options are available as a method: `nearest.compile("float32")`.

The compiled object is called with keyword arguments only. It returns a tensor
(or a tuple for `min_argmin`, `max_argmax` and `kmin_argkmin`) allocated on the
device of the inputs. `run.input_names` lists the expected names.

The GPU kernel is generated and compiled on the first call for a device, and
cached on disk (see [Backends](backends.md#caches-and-environment-variables)),
so later processes start fast.

## Devices and backends

With `backend="auto"`, the device of the input tensors decides:

| Tensors on | Backend | GPUs |
| --- | --- | --- |
| `cuda` (CUDA build of PyTorch) | `"cuda"` | NVIDIA |
| `cuda` (ROCm build of PyTorch) | `"hip"` | AMD |
| `xpu` | `"sycl"` | Intel |
| `cpu` | none: `UnsupportedError` | |

You can also name a backend explicitly. `backend="oracle"` is the CPU reference
used by the tests; it is never chosen automatically.

kernox never falls back silently. A missing backend raises
`BackendUnavailableError`; an operation, option or dtype a backend does not
support raises `UnsupportedError`. Both are raised before anything runs.

All inputs of one call must be on the same device. The kernel runs
asynchronously on the current PyTorch stream of that device, like a PyTorch
operation.

## dtypes

The `dtype` of `compile` is the dtype of all floating-point inputs: tensors
must have exactly this dtype (kernox does not convert them). The formula is
evaluated in this dtype.

Sums and the log-sum-exp family accumulate in `acc_dtype`:

| `dtype` | default `acc_dtype` | allowed `acc_dtype` | output dtype |
| --- | --- | --- | --- |
| `"float64"` | `"float64"` | `"float64"` | float64 |
| `"float32"` | `"float32"` | `"float32"`, `"float64"` | `acc_dtype` |
| `"float16"` | `"float32"` | `"float16"`, `"float32"` | float16 |

For example, `acc_dtype="float64"` with `dtype="float32"` gives float32
arithmetic with a float64 accumulator and a float64 result. For the other
reductions, an `acc_dtype` different from `dtype` is refused. Indices and counts are always int64. More
on half precision in [FP16](guides/fp16.md).

Some GPUs do not support float64; kernox then refuses a float64 reduction on
them instead of computing in float32.

## Errors

| Exception | When |
| --- | --- |
| `kernox.ValidationError` | invalid input: names, shapes, dtypes, devices, `K`, options |
| `kernox.ConstantConversionError` | a constant does not fit in the dtype |
| `kernox.UnsupportedError` | the operation is not supported (by kernox, or by this backend) |
| `kernox.BackendUnavailableError` | the backend is missing or cannot be loaded |

`ValidationError` and `ConstantConversionError` are also `ValueError`s.
All are raised before any kernel is launched.
