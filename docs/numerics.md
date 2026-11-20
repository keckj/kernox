# Numerics

kernox aims for results you can reason about: by default it follows IEEE
floating-point arithmetic, and anything that trades accuracy for speed is an
explicit option.

## What is guaranteed by default

- No fast-math: `exp`, `log`, divisions and square roots are the correctly
  behaved library versions.
- No reassociation of your formula, no flush of subnormal numbers to zero
  (unless you ask for it, see [below](#flushing-subnormal-numbers)),
  no TF32, and no silent conversion of float64 to float32.
- Every constant is rounded exactly once, to the reduction's dtype.
- Only one freedom is left to the compiler: fusing `a * b + c` into a single
  fused multiply-add (FMA), described below.

The order in which the terms of a sum are added is not specified and depends
on the device and on the kernel variant, so float sums can differ slightly from
one backend or variant to another. On the `cpu` backend it does not depend on
the number of threads.

## Constants

Constants are exact until the reduction is compiled:

- Python `int` and `fractions.Fraction` values stay exact integers and
  rationals; a `float` keeps its exact binary64 value; symox numbers such as
  `pi` stay symbolic. symox folds constant subexpressions exactly.
- At compile time, each floating-point constant is rounded once, to nearest
  (ties to even), to the dtype. Rationals are emitted as `p/q` in the dtype.
- A finite constant that overflows the dtype, or a non-zero constant that
  would round to zero, raises `ConstantConversionError`. `inf` and `nan`
  written explicitly are kept.
- Parameters passed as Python numbers follow the same rule. A `Param` passed
  as a tensor must already have the exact dtype.

## Fused multiply-add

By default (`fp_contract=True`), the compiler may compute `a * b + c` inside
one expression as an FMA, with a single rounding instead of two. This is what
nvcc and PyKeOps do. It is not less accurate, but results can differ in the
last bit between backends.

For bit-for-bit reproducibility across backends, forbid it:

```python
import torch
import kernox as kp

device = "cuda"
x, y = kp.Vi("x", 3), kp.Vj("y", 3)
d2 = x.sqdist(y)

run = kp.compile(kp.exp(-d2).sum_over("j"), dtype="float64", fp_contract=False)
```

## Summation schemes

Sums (and the internal sums of the log-sum-exp family) accept three schemes,
the same as PyKeOps' `sum_scheme`:

| `sum_scheme` | How terms are added | Error, roughly |
| --- | --- | --- |
| `"direct"` (default) | one running sum per row | grows like `N · u · Σ|term|` |
| `"block"` | sums of blocks of `T` terms, then the total | `(N/T + T) · u · Σ|term|` |
| `"kahan"` | compensated summation (Kahan-Babuška, also known as Neumaier) | `2 · u · Σ|term|`, independent of N |

Here `u` is the unit roundoff of the accumulator (about `6e-8` in float32).
`"kahan"` costs a second accumulator per output; it is the scheme to use when
a float32 sum over a million terms must stay accurate.

```python
precise = kp.compile(kp.exp(-d2).sum_over("j"), dtype="float32", sum_scheme="kahan")
```

The other reductions (min, max, indices, counts) ignore `sum_scheme`.
A wider accumulator is the other way to improve a sum: with
`dtype="float32"`, `acc_dtype="float64"` accumulates in float64.

## Special values

| Reduction | Neutral value (no term) | NaN terms | Ties |
| --- | --- | --- | --- |
| sum | 0 | propagate | |
| count | 0 | a comparison with NaN is false | |
| min / max | `+inf` / `-inf` | ignored | |
| argmin / argmax | `-1` | ignored | smallest index |
| kmin / argkmin | padding with `+inf` / `-1` | ignored | smallest index first |
| logsumexp | `-inf` | propagate | |
| sumsoftmaxweight | NaN (0/0) | propagate | |

- Infinities are ordinary values: a `+inf` term can be returned by `min`
  together with its index.
- When `K` is larger than the number of candidates, `kmin` and `argkmin` pad
  with `+inf` and `-1`.
- Log-sum-exp is computed in one pass with the usual shift by the running
  maximum, so large scores (`1e4`) do not overflow. A term with score `-inf`
  has zero weight. A single `+inf` score gives `+inf` for `logsumexp` and NaN
  for `sumsoftmaxweight`.

## `where` and invalid values

`where(cond, a, b)` returns exactly the value of the active branch. An invalid
operation in the other branch, such as `sqrt(-1)`, `log(0)` or `1/0`, does not
leak into the result, and does not leak into gradients either.

```python
safe = kp.where(d2 > 1.0, kp.log(d2 - 1.0), 0.0)
X, Y = torch.randn(100, 3, device=device), torch.randn(50, 3, device=device)
out = kp.compile(safe.sum_over("j"), dtype="float32")(x=X, y=Y)
assert torch.isfinite(out).all()
```

## Simplifications made by symox

Formulas go through the automatic simplifications of symox, which assume exact
real arithmetic. For example `x / x` becomes `1`, and `~(a < b)` becomes
`a >= b`. This matters with NaN: `~(a < b)` is then false when `a` is NaN,
whereas the literal negation would be true. Write the comparison you mean
rather than negating its opposite.

## Approximate `exp` and `log`

`compile(..., fast_math=True)` allows approximate `exp` and `log`, the
counterpart of PyKeOps' `use_fast_math`. It is off by default and never turned
on implicitly. See [fast_math](guides/fast-math.md).

## Flushing subnormal numbers

`compile(..., flush_denormals=True)` allows subnormal numbers (smaller than
about 1.2·10⁻³⁸ in float32, 2.2·10⁻³⁰⁸ in float64) to be replaced by zero,
both as inputs and as results of each operation. It is off by default and
never turned on implicitly. Gradients inherit it.

| Backend | float32 | float64 | float16 |
| --- | --- | --- | --- |
| `cpu` | flushed (FTZ + DAZ) | flushed (FTZ + DAZ) | flushed when computed through float32 |
| `cuda` | flushed (`--ftz=true`) | kept | flushed when computed through float32 |
| `hip`, `sycl` | `UnsupportedError` | `UnsupportedError` | `UnsupportedError` |
| `oracle` | kept | kept | kept |

The only effect on a result is the flushed mass: a sum loses at most its
number of terms times the smallest normal number, scaled by the factors those
terms are multiplied by. A sum or a log-sum-exp whose largest terms are near 1
does not change in practice. It matters for results that are themselves tiny,
for example a Gaussian kernel sum at a point far from every other point.

Speed: on Intel CPUs, each operation that produces a subnormal result costs a
microcode assist, many times slower than a normal operation; AMD CPUs pay less,
but still pay. Narrow Gaussian kernels, where many terms fall in the subnormal
range, are faster with `flush_denormals=True` (4.8× on a Xeon, 1.5× on an AMD EPYC) (see [CPU backend](backend-cpu.md#vector-math-libmvec)).
On the cpu backend the flags are set for the kernel only: PyTorch, which
shares the OpenMP threads, still sees subnormal numbers afterwards.

## Half precision

FP16 has its own rules (float32 accumulator by default, float16 output): see
[FP16](guides/fp16.md).
