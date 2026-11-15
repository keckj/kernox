# Batch dimensions

To run many independent reductions of the same shape in one launch, add
leading batch dimensions to the inputs. No change to the formula or to
`compile` is needed.

```python
import torch
import kernox as kp

device = "cuda"

x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 1)
gauss = kp.compile((kp.exp(-x.sqdist(y)) * b).sum_over("j"), dtype="float32")

X = torch.randn(8, 500, 3, device=device)     # 8 problems, M = 500
Y = torch.randn(8, 700, 3, device=device)     #             N = 700
B = torch.randn(8, 700, 1, device=device)
out = gauss(x=X, y=Y, b=B)                    # (8, 500, 1)
```

Each batch element is an independent reduction. Indices returned by `argmin`
or `argkmin` are local to their element (between 0 and N - 1).

## Shapes and broadcasting

| Input | Shape |
| --- | --- |
| `Vi` | `(*Bi, M, D)` |
| `Vj` | `(*Bj, N, D)` |
| `Param` | `(D,)`, or `(*Bp, D)`, or Python numbers |
| output | `(*B, M, ...)` |

All `Vi` and `Vj` inputs must have the **same number** of batch dimensions.
The batch shapes broadcast with the usual PyTorch rules, and `B` is the
broadcast shape. Broadcasting is free: an expanded tensor is read in place,
without a copy.

```python
X1 = torch.randn(1, 500, 3, device=device)    # one query set...
out = gauss(x=X1, y=Y, b=B)                   # ...against 8 databases: (8, 500, 1)

Xg = torch.randn(4, 1, 500, 3, device=device) # 4 × 8 grid of problems
out = gauss(x=Xg, y=Y[None], b=B[None])       # (4, 8, 500, 1)
```

A batched parameter gives each element its own value:

```python
s = kp.Param("s")
scaled = kp.compile((kp.exp(-x.sqdist(y) * s) * b).sum_over("j"), dtype="float32")
S = torch.rand(8, 1, device=device)           # one scale per problem
out = scaled(x=X, y=Y, b=B, s=S)              # (8, 500, 1)
```

## Memory layout

The last two dimensions of each input (rows and components) must be
contiguous. The batch dimensions can have any stride, including 0 (from
`expand`). A tensor whose last two dimensions are not contiguous raises
`ValidationError`; call `.contiguous()` on it.

## Gradients

Gradients work as without batch dimensions. For a broadcast input, the
gradient is summed over the broadcast dimensions, as PyTorch does.

## Limitations

- Batch dimensions cannot be combined with [block-sparse ranges](block-sparse.md).
- For problems of different sizes, batching does not apply: call the reduction
  once per problem, or pad the inputs and use a `where` mask.
