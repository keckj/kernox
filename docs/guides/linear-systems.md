# Linear operators and linear systems

A sum reduction that is linear in one of its `Vj` inputs is a matrix-vector
product with an implicit M×N matrix. kernox can expose it as a linear operator,
and solve linear systems with it, as PyKeOps' `solve` does.

## Kernel ridge regression with `solve`

Find `a` such that `(alpha·I + K) a = rhs`, where `K` is a Gaussian kernel
matrix:

```python
import torch
import kernox as kp

device = "cuda"

x, y, b = kp.Vi("x", 3), kp.Vj("y", 3), kp.Vj("b", 1)
K = kp.compile((kp.exp(-x.sqdist(y)) * b).sum_over("j"), dtype="float64")

X = torch.randn(2_000, 3, device=device, dtype=torch.float64)
rhs = torch.randn(2_000, 1, device=device, dtype=torch.float64)

a = K.solve("b", rhs, alpha=0.1, x=X, y=X)           # (2000, 1)
```

The first argument names the unknown, a `Vj` variable of the reduction. The
other inputs are passed by keyword, as for a normal call.

`solve` uses the conjugate gradient method of PyKeOps, with the same
defaults:

| Option | Default | Meaning |
| --- | --- | --- |
| `alpha` | `1e-10` | regularization `alpha·I` |
| `eps` | `1e-6` | stop when the residual norm is below `max(eps·|rhs|, eps)` |
| `maxiter` | `10·M` | iteration limit; a `RuntimeWarning` is emitted if it is reached |
| `x0` | zeros | starting point |

The operator must be square (same number of rows and same dimension on both
sides) and is assumed symmetric positive definite, as in PyKeOps: this is not
checked. If an input is named like one of these options, pass the inputs as a
dictionary: `K.solve("b", rhs, alpha=0.1, inputs={"x": X, "y": X})`.

`solve` is differentiable with respect to `rhs` and to the other tensor inputs.

## Linear operators

`run.linear_operator(var, **inputs)` binds all inputs except `var` and returns
an operator with the interface of `scipy.sparse.linalg.LinearOperator`:

```python
op = K.linear_operator("b", x=X, y=X)
v = torch.randn(2_000, 1, device=device, dtype=torch.float64)

u = op @ v                 # same as K(x=X, y=X, b=v)
u = op.matvec(v)
w = op.rmatvec(v)          # transpose product, also op.T @ v
print(op.shape)            # (2000, 2000)
```

Operands can be `(N, Dv)` tensors (the result is then `(M, Do)`), flat
vectors of length `N·Dv`, or `(N·Dv, k)` matrices of `k` columns. `shape` is
`(M·Do, N·Dv)`. kernox checks symbolically that the reduction really is linear
in `var`, and raises `ValidationError` otherwise.

The operator plugs into SciPy's iterative solvers and eigensolvers. For this
interoperability, `matvec` and `rmatvec` also accept NumPy arrays: they are
copied to the device of the inputs, and the result is returned as a NumPy
array. This is the only place where kernox accepts NumPy arrays.

<!-- skip-test -->
```python
import scipy.sparse.linalg as sla

eigenvalues = sla.eigsh(sla.aslinearoperator(op), k=5, return_eigenvectors=False)
```

## Batches

With [batch dimensions](batch.md) on the bound inputs, `op.batch` gives the
batch shape and `solve` solves one system per batch element (like PyKeOps, a
single conjugate gradient runs on the block-diagonal system). `rhs` can carry
its own batch dimensions, broadcast with those of the inputs.

## The conjugate gradient alone

`kp.conjugate_gradient(linop, b, x0=None, eps=1e-6, maxiter=None)` runs the
solver on any function `linop` and returns `(x, info)`, where `info` reports
convergence, the number of iterations and the final residual.

```python
x_cg, info = kp.conjugate_gradient(lambda v: 0.1 * v + op @ v, rhs)
print(info["converged"], info["niter"])
```
