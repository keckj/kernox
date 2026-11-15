# Formula operations

Every operation on this page is available as a function of the `kernox`
module (`kp.exp(f)`) and, for the PyKeOps-style ones, also as a method
(`f.relu()`). All of them work on any backend. The examples assume:

```python
import torch
import kernox as kp

x, y = kp.Vi("x", 3), kp.Vj("y", 3)
t = kp.Vi("t", 1)
```

## Arithmetic and comparisons

| Operation | Notes |
| --- | --- |
| `f + g`, `f - g`, `f * g`, `f / g`, `f ** g`, `-f`, `abs(f)` | component by component; a dimension-1 operand is broadcast |
| `f < g`, `f <= g`, `f > g`, `f >= g` | boolean formulas |
| `kp.eq(f, g)`, `kp.ne(f, g)` (or `f.eq(g)`, `f.ne(g)`) | equality tests (`==` stays Python identity) |
| `c1 & c2`, `c1 | c2`, `c1 ^ c2`, `~c` | logic on boolean formulas |
| `kp.where(c, f, g)` | `f` where `c` holds, else `g`; only the active branch counts |

## Mathematical functions

All apply component by component:

`exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`,
`atan2(y, x)`, `sinh`, `cosh`, `tanh`, `abs`, `sign`, `floor`, `ceiling`,
`square`, `minimum(f, g)`, `maximum(f, g)`, `clamp(f, lo, hi)`.

`kp.rsqrt(f)` is exactly `1 / sqrt(f)` (no approximate instruction, `inf` at 0).

## Vectors and components

| Operation | Result |
| --- | --- |
| `f[k]`, `f[a:b]`, `f[[0, 2]]` | components |
| `kp.cat(f, g, ...)` | concatenation |
| `f.sum()` | sum of the components |
| `f.sqnorm2()`, `f.norm2()`, `f.dot(g)` | squared norm, norm, dot product |
| `f.normalize()` | `f / |f|`, and 0 for the zero vector |
| `f.sqdist(g)` | `|f - g|²` |
| `f.weightedsqnorm(s)` | `s·|f|²` if `s` has dimension 1, `Σ s_k f_k²` if D, `fᵀ S f` if D² (row-major `S`) |
| `f.weightedsqdist(g, s)` | `(f - g).weightedsqnorm(s)` |
| `f.elem(k)`, `f.extract(k, d)` | component `k`, or `d` components from `k` |
| `t.one_hot(D)` | for a scalar `t`, the vector of dimension D with a 1 at the integer nearest to `t` (zero vector outside `[0, D)`) |

```python
d2 = x.sqdist(y)                        # same as ((x - y) ** 2).sum()
u = (x - y).normalize()
```

## PyKeOps element-wise functions

| Operation | Definition |
| --- | --- |
| `f.relu()` | 0 where `f < 0`, else `f` |
| `f.step()` | 1 if `f >= 0`, else 0 |
| `f.sinxdivx()`, `f.sinc()` | `sin(f)/f` (1 at 0), and `sin(πf)/(πf)` |
| `f.xlogx()` | `f log f`, 0 at 0 |
| `f.inv()` | `1 / f` |
| `f.round(d=0)` | rounding to `d` decimals, halves away from zero |
| `f.mod(n, d=0)` | `f - n·floor((f - d)/n)` |
| `f.clampint(a, b)` | `clamp` with integer bounds |
| `f.ifelse(a, b)` | `a` where `f >= 0`, else `b` |

## Small linear algebra

Matrices are stored row-major in the components, as in PyKeOps.

| Operation | Result |
| --- | --- |
| `A.matvecmult(v)` | matrix-vector product |
| `v.vecmatmult(A)` | vector-matrix product |
| `a.tensorprod(b)` | outer product |
| `a.keops_tensordot(b, dimfa, dimfb, contfa, contfb, permute=None)` (also `kp.tensordot`) | tensor contraction |
| `a.keops_kron(b, dimfa, dimfb)` (also `kp.kron`) | Kronecker product |

```python
A = kp.Vi("A", 9)                       # a 3×3 matrix per row i
Av = A.matvecmult(y)                    # dimension 3
```

## Splines and Soft-DTW

- `knots.bspline(s, k=0)`: the `dim(knots) - k - 1` B-splines of order `k`
  evaluated at the scalar `s` (De Boor recurrence).
- `x.softdtw_sqdist(y, gamma)`: Soft-DTW between the series `x` and `y` with
  squared-difference cost. It is differentiable in `x`, `y` and `gamma`. The
  recurrence is unrolled symbolically, so building the formula grows quickly
  with the series lengths (a few seconds at 16 × 16).

## Derivatives inside a formula

These build the derivative symbolically, as an ordinary formula. `v` must be a
variable (`Vi`, `Vj` or `Param`).

| Operation | Result |
| --- | --- |
| `f.grad(v, e)` | `(∂f/∂v)ᵀ e`, of the dimension of `v` |
| `f.diff(v, u)` (also `kp.diff`) | `(∂f/∂v) u`, of the dimension of `f` |
| `f.grad_matrix(v)` | the Jacobian, flattened row-major |
| `f.divergence(v)`, `f.trace_operator(v)` | divergence of `f` with respect to `v` |
| `f.laplacian(v)` | Laplacian of a scalar `f` |

```python
gauss = kp.exp(-x.sqdist(y))
force = gauss.grad(x, 1.0)              # ∇_x exp(-|x - y|²), dimension 3
F = kp.compile(force.sum_over("j"), dtype="float32", backend="oracle")
print(F(x=torch.randn(5, 3), y=torch.randn(7, 3)).shape)   # torch.Size([5, 3])
```

Gradients of a whole reduction with respect to its inputs go through
`torch.autograd` instead: see [Gradients](guides/gradients.md).

## Factorization hints

`f.factorize(g)` and `f.auto_factorize()` exist for PyKeOps compatibility and
return `f` unchanged. kernox always eliminates repeated subexpressions when it
generates the kernel, so they are not needed.
