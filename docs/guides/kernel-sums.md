# Kernel sums

A kernel sum computes, for every point $x_i$, $\sum_j k(x_i, y_j)\, b_j$. It is the
basic operation behind kernel density estimation, Gaussian processes, kernel
methods and many particle simulations.

## Gaussian kernel

```python
import torch
import kernox as kp

device = "cuda"

x, y = kp.Vi("x", 3), kp.Vj("y", 3)
b = kp.Vj("b", 1)
sigma = kp.Param("sigma")

k = kp.exp(-x.sqdist(y) / (2 * sigma**2))
gauss = kp.compile((k * b).sum_over("j"), dtype="float32")

X = torch.randn(5_000, 3, device=device)
Y = torch.randn(10_000, 3, device=device)
B = torch.randn(10_000, 1, device=device)
out = gauss(x=X, y=Y, b=B, sigma=0.3)     # (5000, 1)
```

The $5{,}000 \times 10{,}000$ kernel matrix is never stored, so the same code works for
millions of points: memory stays proportional to $M + N$.

## Several signals at once

Give `b` more columns to apply the same kernel to several signals in a single
pass. The output has one column per signal:

```python
b5 = kp.Vj("b", 5)
gauss5 = kp.compile((k * b5).sum_over("j"), dtype="float32")
out5 = gauss5(x=X, y=Y, b=torch.randn(10_000, 5, device=device), sigma=0.3)   # (5000, 5)
```

## Other kernels

Any formula works. A few common ones:

```python
r2 = x.sqdist(y)
laplace = kp.exp(-kp.sqrt(r2) / sigma)
cauchy = 1 / (1 + r2 / sigma**2)
matern32 = (1 + kp.sqrt(3 * r2) / sigma) * kp.exp(-kp.sqrt(3 * r2) / sigma)
```

Anisotropic distances use `weightedsqdist`. With a weight `s` of dimension $D$,
the distance is $\sum_k s_k (x_k - y_k)^2$:

```python
s = kp.Param("s", 3)
aniso = kp.exp(-x.weightedsqdist(y, s))
run = kp.compile((aniso * b).sum_over("j"), dtype="float32")
out = run(x=X, y=Y, b=B, s=[1.0, 0.5, 2.0])
```

## Accuracy of large sums

In float32, a sum over millions of terms loses a few digits. Two remedies, both
cheap compared to switching everything to float64:

```python
kahan = kp.compile((k * b).sum_over("j"), dtype="float32", sum_scheme="kahan")
acc64 = kp.compile((k * b).sum_over("j"), dtype="float32", acc_dtype="float64")
```

The first uses compensated summation; the second accumulates in float64 and
returns a float64 result. See [Numerics](../numerics.md#summation-schemes).

## Gradients

The result is differentiable with respect to `X`, `Y`, `B` and a tensor
`sigma`: see [Gradients](gradients.md).
