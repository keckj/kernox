# Faster `exp` and `log` with `fast_math`

Formulas dominated by exponentials (Gaussian kernels, log-sum-exp, Sinkhorn)
spend much of their time in `exp`. `fast_math=True` replaces `exp` and `log` by
fast hardware approximations, as PyKeOps does by default with
`use_fast_math`.

```python
import torch
import kernox as kp

device = "cuda"

x, y = kp.Vi("x", 2), kp.Vj("y", 2)
lse = (-x.sqdist(y) / 0.01).logsumexp(over="j")

fast = kp.compile(lse, dtype="float32", fast_math=True)
X = torch.randn(1_000, 2, device=device)
Y = torch.randn(2_000, 2, device=device)
L = fast(x=X, y=Y)
```

## What changes

- `exp` and `log` are approximated: those written in the formula, and those
  used inside the log-sum-exp reductions.
- The relative error of the approximate `exp` is about `2⁻²¹` plus a term
  proportional to the size of its argument; the absolute error of the
  approximate `log` is about `2⁻²¹`. In float32, `2⁻²¹` amounts to a few
  units in the last place.
- Everything else stays exact IEEE arithmetic: divisions, square roots,
  trigonometric functions, the order of operations, and subnormal numbers.
- Special values are unchanged: `exp(-inf) = 0`, `exp(+inf) = +inf`,
  `log(0) = -inf`, `log` of a negative number is NaN, NaN propagates.

## Where it applies

| Backend | float32 | float64 | float16 |
| --- | --- | --- | --- |
| `cuda` | approximate `exp` and `log` | no effect (no fast instruction exists) | `UnsupportedError` |
| `hip`, `sycl` | `UnsupportedError` | `UnsupportedError` | `UnsupportedError` |
| `cpu` | no effect (stays IEEE, already vectorized) | no effect | no effect |
| `oracle` | no effect (stays IEEE) | no effect | no effect |

A backend that does not implement `fast_math` refuses it with an explicit
error rather than ignoring it.

Gradients of a `fast_math` reduction also use the approximations.

`fast_math` does not flush subnormal numbers to zero, even on `cuda`; that is
a separate option, `flush_denormals=True`, see
[Numerics](../numerics.md#flushing-subnormal-numbers).

The [Performance](../index.md#performance) figures of kernox against PyKeOps
use `fast_math=True`.
