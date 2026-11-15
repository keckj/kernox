# Complex formulas

Declare a variable with `complex=True` to work with complex numbers, as with
PyKeOps' `ComplexLazyTensor`.

```python
import torch
import kernox as kp

device = "cuda"

z = kp.Vi("z", 3, complex=True)
w = kp.Vj("w", 3, complex=True)
t = kp.Vi("t", 3)                                 # a real variable

f = kp.exp(z * w.conj()) / (w + 2.0) + t.exp1j() * (1 - 2j)
run = kp.compile(f.sum_over("j"), dtype="float32")

Z = torch.randn(100, 3, dtype=torch.complex64, device=device)
W = torch.randn(200, 3, dtype=torch.complex64, device=device)
T = torch.randn(100, 3, device=device)
out = run(z=Z, w=W, t=T)                          # (100, 3) complex64
```

## Inputs and outputs

- A complex variable of dimension D takes a complex tensor of shape `(L, D)`:
  `complex64` for `dtype="float32"`, `complex128` for `"float64"`. Its real
  view of shape `(L, 2D)` (as returned by `torch.view_as_real(...).flatten(-2)`)
  is accepted too.
- A complex `Param` also accepts Python complex numbers.
- The same name cannot be both real and complex.
- A complex sum returns a complex tensor of shape `(M, D)`.

## Operations

| Operation | Notes |
| --- | --- |
| `z + w`, `z - w`, `z * w`, `z / w`, `-z` | complex arithmetic; real formulas and Python complex numbers are promoted |
| `z.real`, `z.imag` (or `kp.real(z)`, `kp.imag(z)`) | real formulas |
| `z.conj()`, `kp.conj(z)` | conjugate |
| `abs(z)`, `z.abs()` | modulus (a real formula) |
| `z.square_abs()` | squared modulus |
| `z.angle()`, `kp.angle(z)` | argument, `atan2(im, re)` |
| `kp.exp(z)`, `z.exp()` | complex exponential |
| `z.sum()`, `z[k]` | sum of the components, component `k` |
| `t.real2complex()`, `t.imag2complex()`, `t.exp1j()` | from a real formula: `t`, `i·t`, `exp(i·t)` |
| `kp.complex_formula(re, im)` | from two real formulas |

Power (`**`) and real functions such as `kp.sin` are not defined on complex
formulas and raise `TypeError`, as in PyKeOps.

## Reductions

Only `sum_over` reduces a complex formula. For the other reductions, reduce a
real formula built from it, which still takes complex inputs:

```python
closest = kp.compile(abs(z - w).sum().argmin(over="j"), dtype="float32")
idx = closest(z=Z, w=W)                           # (100, 1) int64
```

## Gradients

Gradients work through `torch.autograd` with PyTorch's convention for complex
numbers.

## How it works

Like keopscore, kernox stores a complex formula of dimension D as a real
formula of dimension 2D with interleaved real and imaginary parts, and writes
each complex operation with real arithmetic. Every backend therefore runs
complex formulas without any complex type on the GPU.
