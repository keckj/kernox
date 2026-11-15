# Gradients

kernox reductions are differentiable with `torch.autograd`. If an input tensor
has `requires_grad=True`, the result is part of the autograd graph, and
`backward()` computes the gradients with fused reductions: no M×N matrix
appears in the backward pass either.

## A first gradient

```python
import torch
import kernox as kp

device = "cuda"

x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 1)
s = kp.Param("s")
energy = kp.compile((kp.exp(-x.sqdist(y) * s) * b).sum_over("j"), dtype="float64")

X = torch.randn(1_000, 3, device=device, dtype=torch.float64, requires_grad=True)
Y = torch.randn(2_000, 3, device=device, dtype=torch.float64, requires_grad=True)
B = torch.randn(2_000, 1, device=device, dtype=torch.float64)
S = torch.tensor([0.5], device=device, dtype=torch.float64, requires_grad=True)

out = energy(x=X, y=Y, b=B, s=S)
out.sum().backward()
print(X.grad.shape, Y.grad.shape, S.grad.shape)   # (1000, 3) (2000, 3) (1)
```

Gradients are available with respect to `Vi` and `Vj` tensors, and to `Param`
values passed as tensors. A `Param` passed as a Python number has no gradient.

## How it works

kernox differentiates the formula symbolically with symox, then runs the
derivatives as new sum reductions on the same backend: one over `j` for the
gradients with respect to `Vi` inputs and parameters, one over `i` for the
`Vj` inputs. These gradient reductions are built on the first `backward()` and
then reused.

Gradients have the dtype of the inputs. With `acc_dtype="float64"`, the
gradient sums also accumulate in float64.

## What can be differentiated

| Reduction | Gradient |
| --- | --- |
| `sum_over` | first and second order |
| `logsumexp`, `logsumexp(weight=)`, `sumsoftmaxweight` | first order |
| `min`, `max`, `kmin`, and the values of `min_argmin`, `max_argmax`, `kmin_argkmin` | first order, through the selected terms |
| `argmin`, `argmax`, `argkmin`, counts | none: integer outputs, as in PyTorch |

For min, max and kmin, the gradient flows through the selected pair only, as in
PyKeOps; kernox evaluates the derivatives on those K pairs per row.

A second derivative of a reduction other than `sum_over` raises an error from
PyTorch.

## Second order

For sums, the backward pass is itself differentiable. Use `create_graph=True`,
for example for a Hessian-vector product:

```python
X2 = X.detach().requires_grad_(True)
(gx,) = torch.autograd.grad(energy(x=X2, y=Y, b=B, s=0.5).sum(), X2, create_graph=True)
v = torch.randn_like(X2)
(hv,) = torch.autograd.grad((gx * v).sum(), X2)  # Hessian-vector product, (1000, 3)
```

`torch.autograd.gradcheck` and `gradgradcheck` work in float64.

## Non-smooth points

| Formula | Derivative used |
| --- | --- |
| `where(c, a, b)` | derivative of the active branch; the condition is not differentiated |
| `abs(f)` | `sign(f)`, so 0 at 0 |
| `minimum`, `maximum`, `clamp` at a tie | 1/2 for each side |
| `floor`, `ceiling`, `sign`, `round`, `step` | 0 |
| `sqrt(f)` at 0 | `inf`, as in PyTorch |

An invalid value in the inactive branch of a `where` does not contaminate the
gradient:

```python
d2 = x.sqdist(y)
r = kp.where(d2 > 1.0, kp.sqrt(d2 - 1.0), 0.0)
run = kp.compile((r * b).sum_over("j"), dtype="float64")
X3 = X.detach().requires_grad_(True)
run(x=X3, y=Y.detach(), b=B).sum().backward()
assert torch.isfinite(X3.grad).all()
```

## Gradients of a formula inside a formula

To use a derivative as part of a formula (a force field, a score function, a
Laplacian), build it with `f.grad(v, e)`, `f.laplacian(v)` and friends: see
[Formula operations](../operations.md#derivatives-inside-a-formula).
