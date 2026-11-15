# kernox API (1.0)

API of kernox 1.0. The core (frontend, plans, runtime, CPU oracle) is
shared; the native backends `cuda`, `hip` and `sycl` register themselves with
the runtime. The CPU oracle (`backend="oracle"`) is a test reference, never
a fallback. Status per backend and per card:
[`support-matrix.md`](support-matrix.md).

```python
from kernox import Vi, Vj, Param, exp, where, compile

x = Vi("x", 3)                  # tensor (M, 3)
y = Vj("y", 3)                  # tensor (N, 3)
w = Vj("w", 1)
sigma = Param("sigma")          # tensor (1,) or Python scalar

d2 = ((x - y) ** 2).sum()       # sum of COMPONENTS -> scalar formula
gaussian = exp(-d2 / (2.0 * sigma**2))

weighted = (gaussian * w).sum_over("j")   # reduction over axis j -> (M, 1)
neighbors = d2.argkmin(32, over="j")       # (M, 32) int64
count = (d2 <= 0.25).sum_over("j")         # COUNT -> (M, 1) int64
bounded = where(d2 < 4.5, gaussian, 0.0)

run = compile(weighted, dtype="float64", backend="auto")
result = run(x=x_t, y=y_t, w=w_t, sigma=0.7)
```

## Formulas

- `Vi(name, dim=1)`, `Vj(name, dim=1)`, `Param(name, dim=1)`: variables. The name
  is a Python ASCII identifier (neither a keyword nor prefixed with `_kp`/`_sx`).
  The same name cannot denote two roles or two dimensions (`ValidationError`).
- A `Formula` is a vector of scalar **symox** expressions (`.components`);
  the components of the variables are real `symox.Symbol`s (`x[0]`, …). No
  string is parsed.
- Operators: `+ - * / **`, `-f`, `abs(f)`, broadcasting of a dimension 1 to D.
- Components: `f[0]`, `f[-1]`, `f[1:]`, `f[[0, 2]]`, `cat(f, g)`.
- `f.sum()`: sum of the components (scalar). **Distinct** from `f.sum_over("j")`.
  `f.sum("j")` or `f.sum(over=...)` raises `TypeError` to avoid confusion.
  Also: `f.sqnorm2()`, `f.norm2()`, `f.dot(g)`.
- Functions (component-wise): `exp log sqrt rsqrt sin cos tan asin
  acos atan atan2 sinh cosh tanh abs sign floor ceiling square minimum maximum
  clamp`. `rsqrt(x)` is exactly `1/sqrt(x)` (no approximate instruction).
- PyKeOps parity operations (`kernox/ops.py`), as methods named after the
  `LazyTensor` ones and as `kp.*` functions; exact keopscore semantics, deviations
  listed in `pykeops-parity.md`:
  - element-wise: `relu step sinxdivx sinc xlogx inv`, `round(d=0)` (half away
    from zero), `mod(n, d=0)` (`x − n·floor((x − d)/n)`), `clampint(a, b)`,
    `ifelse(a, b)` (`a` if `x ≥ 0`);
  - vectors: `normalize` (zero vector → 0), `sqdist(y)`,
    `weightedsqnorm(s)` and `weightedsqdist(y, s)` (`s` of dimension 1, D or D²
    row-major), `one_hot(D)` (scalar x), `elem(i)`, `extract(i, d)`;
  - algebra (row-major): `A.matvecmult(v)`, `v.vecmatmult(A)`,
    `a.tensorprod(b)`, `a.keops_tensordot(b, dimfa, dimfb, contfa, contfb[,
    permute])` (`kp.tensordot`), `a.keops_kron(b, dimfa, dimfb)` (`kp.kron`);
  - `knots.bspline(x, k=0)`: the `dim(knots) − k − 1` B-splines of order k at
    scalar x (De Boor, as in keopscore);
  - `f.factorize(g)`, `f.auto_factorize()` (keopscore `Factorize`,
    `AutoFactorize`): return `f` unchanged, since the generator's CSE already
    factors out every repeated subexpression;
  - derivatives inside the formula (keopscore `Grad`, `Diff`, `GradMatrix`,
    `Divergence`, `Laplacian`, `TraceOperator`), `v` being a variable
    (`Vi`, `Vj`, `Param`): `f.grad(v, e)` = `(∂f/∂v)ᵀ e` (dim of v),
    `f.diff(v, u)` (`kp.diff`) = `(∂f/∂v) u` (dim of f), `f.grad_matrix(v)`
    (`∂f_k/∂v_c` at `k·dim(v) + c`), `f.divergence(v)`, `f.laplacian(v)`
    (scalar f), `f.trace_operator(v)` (= divergence); symbolic derivatives
    following the gradient rules (below);
  - `x.softdtw_sqdist(y, gamma)` (`kp.softdtw_sqdist`): Soft-DTW between the
    series x (dim n) and y (dim m), cost `(x_i − y_j)²`, scalar `gamma`;
    keopscore recurrence unrolled into a DAG of O(n·m) nodes, differentiable
    in x, y and `gamma`.
- Predicates: `< <= > >=`, `eq(a, b)`, `ne(a, b)` (or `a.eq(b)`); `==`/`!=`
  remain Python identity. Logic: `&`, `|`, `^`, `~`. A formula has no
  truth value (`if f:` raises `TypeError`).
- `where(cond, a, b)`: symox `Piecewise`; the result is always that of the
  active branch: an out-of-domain value in the inactive branch (`sqrt(-1)`,
  `log(0)`, `1/0`, `inf - inf`) never contaminates the output.
  Subexpressions shared by several branches, or already computed by the
  condition, are computed once (region-based CSE, see
  `kernel-backend-contract.md`); those specific to a single branch stay
  in the ternary.
- Arithmetic on booleans is rejected (`TypeError`): use `where`.
- symox's automatic simplifications apply (exact arithmetic over the
  reals), e.g. `x/x → 1`, `~(a < b) → a >= b`. Consequence: with NaNs,
  `~(a < b)` is evaluated as `a >= b` (false), not as `!(a < b)`.

## Reductions

| Method | Plan | Output(s) for M rows |
| --- | --- | --- |
| `f.sum_over(ax)` (real f) | SUM | `(M, D)` in `acc_dtype` |
| `pred.sum_over(ax)` (boolean f) | COUNT | `(M, D)` int64 |
| `f.min(over=ax)` / `max` | MIN / MAX | `(M, D)` dtype |
| `f.argmin(over=ax)` / `argmax` | ARGMIN / ARGMAX | `(M, D)` int64 |
| `f.kmin(K, over=ax)` | KMIN | `(M, K)` dtype |
| `f.argkmin(K, over=ax)` | ARGKMIN | `(M, K)` int64 |
| `f.kmin_argkmin(K, over=ax)` | KMIN_ARGKMIN | tuple `((M, K), (M, K))` |
| `f.logsumexp(over=ax)` | LOGSUMEXP | `(M, D)` in `acc_dtype`: `log Σ exp f_d` per component |
| `f.sumsoftmaxweight(g, over=ax)` (scalar f, g of dim E) | SUMSOFTMAXWEIGHT | `(M, E)` in `acc_dtype`: `Σ exp(f)·g / Σ exp(f)` |
| `f.min_argmin(over=ax)`, `f.max_argmax(over=ax)` | MIN_ARGMIN, MAX_ARGMAX (amendment v0.8) | tuple `((M, D), (M, D) int64)`, single pass |
| `f.logsumexp(over=ax, weight=g)` (scalar f, g of dim E) | LOGSUMEXPWEIGHT (amendment v0.8) | `(M, E)` in `acc_dtype`: `log Σ exp(f)·g_e` |

`ax` is `"j"` or `"i"`. `over="i"` is normalized by swapping the i/j roles
(`plan.meta["swapped_ij"]`): M is then the number of rows of the `Vj` variables.
Semantics (NaN, ties, K, empty inputs): see
[`kernel-backend-contract.md`](kernel-backend-contract.md). `K` must be a Python
`int` `>= 0` (otherwise `ValidationError` at construction). KMIN/ARGKMIN: scalar
formula only in v0 (`UnsupportedError`).
LOGSUMEXP / SUMSOFTMAXWEIGHT (amendment v0.5, PyKeOps names `logsumexp` and
`sumsoftmaxweight`): stable single-pass computation (shift by the running
maximum); a term `f = −inf` has zero weight; N = 0 or all `f = −inf`
→ `−inf` / NaN; one `f = +inf` → `+inf` / NaN; NaN propagated. The
SUMSOFTMAXWEIGHT plan has outputs `(f, g_1, …, g_E)`. `acc_dtype="float64"` is
allowed in FP32 as for SUM. First-order gradient: see "Gradients". v0 requires at least one i variable
and one j variable (otherwise M or N cannot be inferred: `UnsupportedError`).

## Constants: promotion and conversion

1. **Formula construction** (independent of the dtype):
   - `int` → exact `Integer`; `bool` → `true/false` in a boolean context, 0/1 in
     arithmetic (Python semantics); `Fraction` → exact `Rational`;
   - `float` (and NumPy scalars, including `np.float32`) → symox `Float` with
     53-bit precision carrying **exactly** the binary64 value; `inf` → `oo`, `nan` → `nan`;
   - symox numbers (`Rational(1, 3)`, `pi`, `E`): kept exact;
   - no constant is rounded to FP32 at this stage.
   Numeric subexpressions are folded by symox (exact rationals,
   53-bit `Float`) before any rounding to the dtype.
2. **Plan construction** (`compile(..., dtype=...)`, `builder.convert_constants`):
   - `Float`: rounded **once** to nearest (ties to even) to `dtype`; a `Float`
     with precision > 53 bits is first rounded to binary64;
   - `Rational`/`Integer`: kept exact; emitted as `p/q` in the dtype (correct
     rounding of p/q when p and q are representable, otherwise p and q are rounded
     first); `pi`, `E`: literals with 17 (FP64) or 9 (FP32) digits;
   - overflow (finite value outside the dtype range) or nonzero value rounded to 0:
     `ConstantConversionError` (no silent `inf`/0). Explicit `inf`/`nan`
     are kept.
3. **Parameters** passed as Python scalars: same rule (rounding to `dtype`,
   error on overflow/underflow). A tensor `Param` must already have the
   exact dtype. The device tensor created for Python values is cached
   by exact values (64 entries): a new value costs one host → device
   copy at call time, an already-seen value none; passing a device
   tensor avoids any copy.

## `compile` and runtime

`compile(expr, *, dtype, backend="auto", acc_dtype=None, tuning="estimate", fp_contract=None, sum_scheme=None, fast_math=None) -> Reduction`

(also `formula.compile(dtype, ...)`, `formula.to_plan(dtype, acc_dtype=None, fp_contract=True, sum_scheme=None, fast_math=False)`.)

- `dtype` is **mandatory** (`"float16"`/`"float32"`/`"float64"`). `acc_dtype`
  (SUM and LOGSUMEXP families) ≥ `dtype`: `float32`/`float64` in FP32,
  `float16`/`float32` in FP16 (default `float32`; `float16`: `block` scheme
  by default, as in PyKeOps). An FP16 plan returns FP16 (±inf beyond
  65504; amendment v0.9). In FP16 the formula is computed in FP16, by
  the oracle as well as by the `cuda`, `hip` and `sycl` backends.
- `fp_contract` (contract v0.3): `True` by default (`None` = default, or the value
  of the supplied `Plan`) — the compiler may fuse `a*b + c` into an FMA **within
  an expression** (a single rounding: no fast-math, no
  reassociation, no loss of precision; results may differ in the
  last bit from one backend to another). `False` forbids any contraction on
  all backends, for bit-exact reproducibility; a backend that
  cannot yet guarantee it raises `UnsupportedError` (never silently
  ignored; the Torch oracle never contracts). A boolean is required; the option
  is part of `plan.key()` (cache, wisdom).
- `sum_scheme` (amendment v0.7, equivalent to the PyKeOps option): `"direct"`
  (default; `None` = default or the value of the supplied `Plan`), `"block"` or
  `"kahan"`. Applies to the floating-point accumulations of SUM and to the sums
  `s`, `t_e` of LOGSUMEXP/SUMSOFTMAXWEIGHT; ignored without error by COUNT,
  MIN/MAX, ARG*, KMIN*.
  - `direct`: one `acc_dtype` accumulator per thread, partial sums added up
    (error ≲ N·u·Σ|term|).
  - `block`: each tile of `T` terms is summed into a fresh accumulator
    then added to the total (≲ (N/T + T)·u·Σ|term|); `T` is published
    in `CompileInfo.options` (`sum_block=T`).
  - `kahan`: Kahan–Babuška (Neumaier) compensated summation, state
    `(s, c)`; error ≲ 2u·Σ|term|, independent of N to first order, at the
    cost of a doubled state. A non-finite `s` (±inf, NaN) is returned as is.

  The gradient of a SUM uses the same scheme as the forward pass. A backend
  that does not implement the scheme raises `UnsupportedError` (never a fallback to
  `direct`). The option is part of `plan.key()` (except `direct`, key unchanged).
  The oracle actually simulates each scheme in `acc_dtype` (additions rounded
  one by one), so that direct/kahan differences are visible there.
- `fast_math` (amendment v0.10): `False` by default (`None` = default); the
  computation stays IEEE. `True` allows **approximate** `exp` and `log`, like
  PyKeOps' `use_fast_math`: in CUDA FP32, `ex2.approx`/`lg2.approx`
  (without flush-to-zero); FP64 unchanged. Divisions, roots, trigonometry and
  order of operations stay IEEE; edge cases (`exp(−inf) = 0`,
  `log(0) = −inf`, NaN) are preserved. `hip`, `sycl`, and `cuda` in FP16
  raise `UnsupportedError`; the oracle stays IEEE. The option is part of
  `plan.key()` and gradient plans inherit it.
- `backend`: `"auto"` chooses based on the device of the tensors (`cuda` → `"cuda"`,
  or `"hip"` under ROCm; `xpu` → `"sycl"`); CPU → `UnsupportedError`.
  `"oracle"` is the blockwise Torch reference, only on explicit request.
  A missing native backend raises `BackendUnavailableError`: **never a fallback**.
- Keyword-only calls: `run(x=..., y=..., sigma=...)`.
- Validation before any launch: names (missing/unexpected), `torch.Tensor`
  for i/j, shapes `(M, dim)`/`(N, dim)`/`(dim,)`, exact dtype (no implicit
  conversion), same device, contiguity (non-contiguous rejected), consistency of M and N.
- Autograd: see "Gradients" below.
- Outputs allocated by the core via `plan.output_shapes` on the input device.
  `M = 0` or `K = 0`: no launch (the backend must nevertheless exist and
  accept the plan).
- `Reduction.plan`, `Reduction.compiled(device).info` (`CompileInfo`).

Registry: `register_backend(name, factory)` or a module `kernox.backends.<name>`
exposing `get_backend() -> Backend` (loaded lazily). See the contract.

## Batch dimensions (amendment v0.6)

Inputs may carry leading batch dimensions: `x (*Bi, M, d)`,
`y (*Bj, N, d)`, parameter `(*Bp, d)` or `(d,)` (or a Python scalar). All
i and j inputs have the same number of batch dimensions; shapes
broadcast according to Torch rules, and the output is `(*B, M, out)`. Each
element is an independent reduction (local indices `0 ≤ j < N`).

```python
run = kp.compile(gauss, dtype="float32", backend=backend)
out = run(x=X.expand(4, 1, -1, -1), y=Y4[:, None], b=B4[:, None], s=0.5)  # (4, 1, M, 2)
```

The last two dimensions must be contiguous; batch dimensions
accept any stride (including 0, `expand`), without copying. Gradients:
as without batch, reduced over the broadcast dimensions. Backends:
see `docs/support-matrix.md`; a backend that does not yet handle batching
raises `UnsupportedError` (never an implicit loop: looping is left to the
user). Measured autotuning does not yet apply to batched
calls (`fallback` decision).

## Block-sparse ranges (amendment v0.6)

`run.with_ranges(r)(**inputs)` restricts the reduction to blocks (same
convention as the PyKeOps `ranges` argument). `kp.Ranges(ranges_i,
slices_i, redranges_j, ranges_j=None, slices_j=None, redranges_i=None)`:
int32/int64 tensors on the device of the inputs, user orientation;
for i in block `ranges_i[c] = [a, b)`, the reduction runs over the
j intervals `redranges_j[slices_i[c-1]:slices_i[c]]`. Uncovered rows:
neutral element of the reduction. Returned indices: global; ties go to the
smallest j of the reduced set.

```python
r = kp.ranges_from_labels(labels_x, labels_y, keep)   # sorted labels, boolean keep (Ci, Cj)
near = kp.compile(d2.argkmin(8, over="j"), dtype="float32").with_ranges(r)
idx = near(x=X_sorted, y=Y_sorted)
```

The transposed triplet is required for `over="i"` and for gradients with
respect to j arguments (otherwise an explicit `ValidationError` /
`UnsupportedError`); `ranges_from_labels` builds both. `with_ranges(r,
validate=True)` checks bounds and overlaps (one host synchronization).
Ranges combined with batch dimensions: rejected. Backends: see
`docs/support-matrix.md`.

## Gradients (`torch.autograd`, contract v0.4)

A SUM call in which a tensor input has `requires_grad` (grad mode enabled) is
differentiable with respect to the i and j tensors and to floating-point tensor
parameters; a parameter passed as a Python number has no gradient.

```python
run = kp.compile((kp.exp(-d2 * s) * w).sum_over("j"), dtype="float32")
out = run(x=X, y=Y, w=W, s=S)        # X, Y, W, S requires_grad
out.sum().backward()                 # X.grad, Y.grad, W.grad, S.grad
```

- Derivatives are **symbolic** (`symox.diff`) and executed by the same
  backend as fused SUM reductions, without an M×N matrix: one reduction over
  j for the i args and the parameters (per-row contributions then summed
  by Torch), one reduction over i for the j args. All derivatives on one
  side are the outputs of a single plan (subexpressions shared by the CSE).
  Gradient plans are built on the first `backward` and then cached.
- Order 2 (`create_graph=True`, `gradgradcheck`): the backward pass is
  itself differentiable by the same mechanism.
- dtype: gradients in the dtype of the inputs; with `acc_dtype="float64"`, the
  upstream gradient is brought back to the plan's `dtype` and the gradient sums
  accumulate in `acc_dtype`.
- Rules at non-smooth points: `where` → gradient of the active branch (the
  condition is not differentiated; an out-of-domain value in the inactive branch
  does not contaminate the gradient); `maximum`/`minimum`/`clamp` → 1/2 at ties;
  `abs` → `sign` (0 at 0); `floor`, `ceiling`, `sign` → 0; distributional
  parts (`DiracDelta`) → 0; `sqrt` at 0 → `inf` (as in Torch).
- LOGSUMEXP, `logsumexp(weight=)`, SUMSOFTMAXWEIGHT (order 1): the gradient
  is that of a surrogate SUM plan in which the forward output is held fixed
  (`Σ_j exp(f − L) ∂f`, `Σ_j exp(f − L_e) g_e`, `Σ_j exp(f − S)(g_e − O_e)`),
  hence fused reductions as above (batch and ranges included;
  SUMSOFTMAXWEIGHT recomputes `S = log Σ exp f`). A row without contribution
  (`L = −∞`) has a zero gradient.
- MIN, MAX, KMIN and the values of MIN_ARGMIN, MAX_ARGMAX, KMIN_ARGKMIN
  (order 1): gradient at the selected term, as in PyKeOps: the symbolic
  derivatives are evaluated by Torch on the selected pairs only (O(M·K),
  never M×N); on the j side, accumulation by index (`scatter_add`). On ties, the
  selected term is the smallest j; an index −1 (no candidate) does not
  contribute. Indices are recomputed (ARG* reduction) when the reduction does not
  return them.
- Order 2: SUM only; for the other reductions, the second derivative
  raises a Torch error (`once_differentiable`).
- **Not differentiable**: ARGMIN/ARGMAX/ARGKMIN/COUNT and indices return
  `int64` integers, without gradient (Torch semantics).
- No reuse between forward and backward pass: the upstream gradient
  is only known at `backward`; the forward pass is not recomputed.

## Complex formulas (PyKeOps `ComplexLazyTensor` parity)

```python
z, w, x = Vi("z", 3, complex=True), Vj("w", 3, complex=True), Vi("x", 3)
f = exp(z * w.conj()) / (w + 2.0) + x.exp1j() * (1 - 2j)
run = compile(f.sum_over("j"), dtype="float32")
out = run(z=z_c64, w=w_c64, x=x_f32)          # complex64 (M, 3)
```

- keopscore representation: **interleaved** real formula `(re, im, …)` of
  dimension 2D (`ComplexFormula.formula`); each operation is keopscore's
  definition in real primitives, hence executed by all backends.
- `Vi/Vj/Param(name, D, complex=True)`: complex tensor `(…, L, D)`
  (`complex64` in FP32, `complex128` in FP64; lazy conjugation resolved)
  or real view `(…, L, 2D)`; `Param` also accepts complex Python
  scalars. The same name cannot be both real and complex.
- Operations: `+ − × /` (complex, real promoted by `real2complex`, real of
  dimension 1 as `ComplexRealScal`, complex Python constants; broadcasting
  of a complex dimension 1), `-z`, `z.real`, `z.imag`, `z.conj()`,
  `abs(z)`/`kp.abs` (modulus `sqrt(re² + im²)`), `z.square_abs()`,
  `z.angle()` (`atan2`), `kp.exp(z)`, `z.sum()` (sum of the components),
  `z[k]`; from a real: `x.real2complex()`, `x.imag2complex()`,
  `x.exp1j()`, `kp.complex_formula(re, im)`. Functions `kp.conj`, `kp.angle`,
  `kp.real`, `kp.imag`.
- Reduction: `z.sum_over(ax)` only (SUM, complex output `(…, M, D)`);
  the other reductions raise `TypeError` (reduce `.real`, `.abs()`…).
  Real formulas derived from complex ones (`abs(z).min(over="j")`) accept
  complex inputs.
- Rejected as in PyKeOps: `real2complex`/`imag2complex`/`exp1j` of a complex,
  `**`, real functions (`sin(z)`…) on a complex (`TypeError`).
- Gradients via `torch.view_as_real` (Torch convention for complex numbers).

## Linear operator and `solve` (PyKeOps parity)

```python
x, y, b = Vi("x", 3), Vj("y", 3), Vj("b", 1)
run = compile((exp(-(x - y).sqnorm2()) * b).sum_over("j"), dtype="float64")
op = run.linear_operator("b", x=x_t, y=y_t)   # K : (N, 1) -> (M, 1)
op @ v, op.matvec(v), op.rmatvec(u), op.T @ u, op.shape   # (M·Do, N·Dv)
scipy.sparse.linalg.eigsh(scipy.sparse.linalg.aslinearoperator(op), k=5)
a = run.solve("b", rhs, alpha=1e-2, x=x_t, y=x_t)        # (alpha·I + K) a = rhs
```

- `run.linear_operator(var, **inputs)`: SUM reduction, `var` a j variable, the
  outputs **linear** in `var` (checked symbolically: zero at `var = 0`,
  derivatives free of `var`, otherwise `ValidationError`). Inputs bound at
  construction (except `var`). Batch (amendment v0.6): inputs `(*B, L, D)`;
  `op.batch` = `B`, `op.shape` = `(*B, M·Do, N·Dv)` (SciPy: without batch
  only); products on `(*B', N, Dv)`, `B'` broadcast with `B`.
- `matvec`/`matmat`/`@`: `(N, Dv) -> (M, Do)`, or vector `(N·Dv,)`, or
  columns `(N·Dv, k)`; NumPy accepted (converted to the device of the inputs,
  returned as NumPy); same dtype and device as the inputs, otherwise
  `ValidationError`. `rmatvec`/`rmatmat`/`T`/`t()`: transpose via the gradient
  plan (vector-Jacobian product). Differentiable products.
- `run.solve(var, rhs, *, alpha=1e-10, eps=1e-6, maxiter=None, x0=None,
  inputs=None, **named_inputs)` (or `op.solve(rhs, ..., return_info=False)`):
  PyKeOps conjugate gradient on `alpha·I + K`, `K` square
  (`M = N`, `Do = Dv`), assumed symmetric positive definite (not checked,
  as in PyKeOps); stops when `‖r‖ < max(eps‖rhs‖, eps)`, `maxiter = 10·M` by
  default, `RuntimeWarning` if reached. `rhs` `(*Bb, M, Do)`, result
  `(*broadcast(B, Bb), N, Dv)`: as in PyKeOps, a single conjugate gradient
  on the block-diagonal system (norms over all coefficients). Differentiable with respect to `rhs` and
  to the tensor inputs (any order). `inputs={...}` avoids a name clash
  with `alpha`, `eps`, `maxiter`, `x0`. Also on `run.with_ranges(r)`.
- `kp.conjugate_gradient(linop, b, x0=None, eps=1e-6, maxiter=None)`:
  the solver alone, returns `(x, info)`.

## Diagnostics

`kernox.diagnostics(check=False, backends=None) -> dict` and
`python -m kernox.diagnostics [--check] [--backend NAME] [--json]`:
versions (Python, Torch and CUDA/ROCm/XPU runtime, NumPy, symox with its revision
identified by the fingerprint of its native library and a warning if it
is not `PINNED_SYMOX`, symox capabilities), available native backends or
exact reason for unavailability, device, arch, FP64 accepted, `fp_contract=False`
supported, compilation caches and wisdom. `--check` compiles small
reductions (Gaussian SUM, ARGKMIN, SUM gradient) in FP32/FP64, publishes the
exact compiler and driver and compares against the FP64 oracle; exit code 1
on failure. `--check` launches kernels: not during a benchmark on
the card.

## Errors

`ValidationError` (invalid input), `ConstantConversionError` (non-representable
constant), `UnsupportedError` (operation explicitly not supported),
`BackendUnavailableError` (missing backend). All are raised before launch.

## Optional autotuning (`tuning=`, contract v0.2)

`compile(expr, dtype=..., backend=..., tuning="estimate")`

| `tuning` | Behavior |
| --- | --- |
| `"estimate"` (default) | `backend.default_variant(plan, device, Sizes(M, N))` at each call; no measurement, no wisdom read; reproducible for a given device |
| `"wisdom_only"` | variant from the wisdom for the key if it exists and is valid, otherwise `estimate`; no measurement |
| `"measure"` | wisdom if present; otherwise measures **at most 4** candidates `variants(..., "measure")`, keeps the fastest, persists it if the measurement is stable |
| `"patient"` | same with **at most 32** candidates `variants(..., "patient")` |

- **Key**: `plan.key()`, backend, device identity (optional method
  `backend.device_identity(device)`, otherwise name/arch/compiler from the `CompileInfo`
  of the default variant), Torch fingerprint of the device (CUDA/ROCm UUID if
  available, runtime version), generator version, and size class
  in half-octaves `(ceil(2 log2 M), ceil(2 log2 N), ceil(2 log2 nb),
  ceil(2 log2 1/ρ))` (wisdom v3: `nb` batch elements, `ρ` density of the
  ranges; at most a factor √2 per axis within a class; K, D and dtype are
  in the plan). Batched plans and plans with
  ranges are measured like the others (no more `fallback` decision). Within the same class: neither
  new measurement nor recompilation. An entry is only reused by an effort level
  lower than or equal to the one that produced it (`patient` redoes the
  search instead of reusing a `measure` decision, then replaces it;
  `wisdom_only` reuses any entry). Format `kernox-wisdom-v3` (files in another
  format are ignored).
- **Measurement**: only at the first call of a class without wisdom, a
  **blocking** call; real inputs read-only, temporary outputs;
  compilation excluded from timing but published; 1 warm-up then 5
  launches timed with GPU events on the current stream (host clock
  on CPU); total budget `BUDGET_SECONDS` = 2 s (`measure`) / 30 s (`patient`),
  compilation included; at least the default variant is measured.
- **Bounds**: a candidate list longer than 4/32 is **truncated**
  (default variant first) and the overflow is logged
  (`logging`: `kernox.tuning`).
- **Instability**: if `(max − min)/median > 0.25` for either of the two
  best variants, the decision is `stable=False`, **not persisted**, and
  kept only in memory for the current compiled object.
- **Result**: always recomputed with the selected variant, in the outputs
  of the call; never that of a discarded variant.
- **Backends without variants** (no `default_variant`/`variants`, or `compile`
  without a `variant` parameter, e.g. `oracle`): explicit fallback
  to `estimate`, decision `source="fallback"`, warning logged once.
- **Diagnostics**: `run.last_decision` (`TuningDecision`: level, source
  `estimate|wisdom|measured|fallback`, variant, size class, stability,
  persistence, candidates, truncation, raw measurements in ms, tuning time,
  budget) and `run.tuning_decisions`.

Wisdom: one JSON file per device under `$KERNOX_CACHE_DIR/wisdom` (otherwise
`$XDG_CACHE_HOME/kernox/wisdom`, or `~/.cache/kernox/wisdom`), written
atomically (temporary file + `os.replace`) under an `fcntl` lock. An entry with a different
format, key or identity (driver, compiler, generator...)
is ignored. `export_wisdom(path=None) -> dict`, `import_wisdom(dict | path) -> int`
(valid entries imported), `forget_wisdom() -> int` (files deleted).

Operating rule: persisted measurements and benchmarks are only done
on an idle GPU; correctness tests (including those of the tuner) may
run on a shared GPU, since an unstable measurement is not persisted there.
