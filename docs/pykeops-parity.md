# Feature parity with PyKeOps — inventory and tracking

Inventory of the features of PyKeOps 2.4 (`pykeops` LazyTensor,
`keopscore/formulas/{maths,reductions,complex,LinearOperators,factorization}`)
and of their equivalent in kernox. A row is only "done" when implemented
**and** tested on cuda, hip and sycl (conformance), with its evidence in
`docs/support-matrix.md`.

Legend: ✅ done · 🔶 partial / in progress · ⬜ to do.

**kernox 1.0: all rows are ✅ on cuda, hip and sycl.** Remaining outside
parity: the native CPU backend, out of scope. Performance: see the
"Performance" section of the README.

## Reductions

| PyKeOps | Kernox | Status |
| --- | --- | --- |
| `sum` (Sum_Reduction) | SUM | ✅ |
| `min`, `max`, `argmin`, `argmax` | MIN, MAX, ARGMIN, ARGMAX | ✅ |
| `min_argmin`, `max_argmax` (single pass, two outputs) | MIN_ARGMIN, MAX_ARGMAX (amendment v0.8, adopted) | ✅ native cuda, hip, sycl |
| `Kmin`, `argKmin`, `Kmin_argKmin` | KMIN, ARGKMIN, KMIN_ARGKMIN | ✅ K ≤ 1024 on cuda, hip, sycl (cooperative top-K) |
| `logsumexp` (with `weight`) | LOGSUMEXP; LOGSUMEXPWEIGHT (amendment v0.8, adopted) | ✅ native cuda, hip, sycl |
| `sumsoftmaxweight` | SUMSOFTMAXWEIGHT | ✅ native cuda, hip, sycl |
| `sum_scheme` direct / block / kahan | `sum_scheme` (amendment v0.7) | ✅ cuda, hip, sycl (kahan = Neumaier, more accurate than PyKeOps' plain Kahan) |
| counting | COUNT (Kernox only) | ✅ |

## Calling, shapes, execution

| PyKeOps | Status |
| --- | --- |
| batch dimensions (`nbatchdims`) | ✅ amendment v0.6; native cuda, hip, sycl |
| `ranges` (block-sparse) | ✅ amendment v0.6 (`run.with_ranges`); native cuda, hip, sycl (ranges + batch combined: rejected) |
| `solve` (conjugate gradient, `alpha`) | ✅ `run.solve(var, b, alpha=, eps=, maxiter=, x0=, **inputs)`; PyKeOps CG and stopping criterion, differentiable (b, bound inputs, any order); oracle and ranges (`tests/test_linalg.py`), backend conformance (`tests/conformance/test_linalg.py`) |
| linear operator interface: `matvec`, `rmatvec`, `T`/`t`, `trace_operator` | ✅ `run.linear_operator(var, **inputs)`: `shape`, `dtype`, `matvec`, `rmatvec` (VJP of the gradient plan), `matmat`, `T`/`t()`, `@`, compatible with `scipy.sparse.linalg.aslinearoperator`; linearity checked symbolically; `trace_operator` inside the formula (Gradients section) |
| FP16 (`dtype=float16`, FP32 accumulation) | ✅ amendment v0.9 (adopted): hip (`kp_h` emulation), cuda (emulation on sm_52, `add/sub/mul.rn.f16` on sm_53+), sycl (`sycl::half`, symox 53d9576e); FP16 outputs as in PyKeOps; deliberate deviation: FP32 accumulation by default (PyKeOps: automatic `dtype_acc` = FP16); `acc_dtype="float16"` reproduces PyKeOps exactly (half accumulation, `block_sum` by default) |
| complex numbers (`Real2Complex`, `ComplexMult`, `conj`, `angle`, `exp1j`, …) | ✅ `Vi/Vj/Param(..., complex=True)`, `ComplexFormula` (`kernox/complex.py`): the operations of `keopscore/formulas/complex` as interleaved reals (`ComplexSumT`, internal to gradients, unnecessary: symbolic differentiation), complex SUM, gradients; oracle against complex Torch (`tests/test_complex.py`), conformance (`tests/conformance/test_complex.py`, cuda, hip and sycl 8/8) |
| `auto_factorize` / `factorize` | ✅ methods `f.factorize(g)`, `f.auto_factorize()`: semantic identities, symox's CSE (region-based under `where`) is always applied by the generator; `tests/test_ops.py` checks that a repeated subexpression is evaluated only once on cuda, hip and sycl |
| native CPU backend | out of scope for now; the CPU oracle remains a test reference |

## Gradients

| PyKeOps | Status |
| --- | --- |
| gradients of sums, order 2 | ✅ (cuda, hip, sycl) |
| gradients of min/max/kmin and of the values of min_argmin/max_argmax/kmin_argkmin (at the selected term, order 1) | ✅ Torch on the selected pairs + backend ARG* reduction; conformance cuda, hip, sycl |
| gradients of logsumexp, `logsumexp(weight=)`, sumsoftmaxweight (surrogate SUM plan, order 1) | ✅ fused SUM reductions; conformance cuda, hip, sycl |
| `grad`, `grad_matrix`, `diff`, `laplacian`, `divergence`, `trace_operator` **inside the formula** | ✅ ordinary formulas (`tests/test_derivative_ops.py`, against `torch.func`), executed by all backends |

## Formula operations

Already present: `+ - * / **`, `abs acos asin atan atan2 cos sin tan cosh sinh tanh
exp log sqrt rsqrt square sign floor ceiling clamp minimum maximum where`,
comparisons, logic, `sum` (components), `norm2 sqnorm2 dot`, indexing,
`cat`.

Added (`kernox/ops.py`, `Formula` methods with PyKeOps names and
`kp.*` functions): keopscore semantics reproduced exactly, expressed in
symox primitives (`Piecewise`, `floor`, arithmetic), hence without any codegen
extension and printed identically by the three backends.

| PyKeOps | Status |
| --- | --- |
| `round`, `mod`, `relu`, `step`, `sinxdivx`/`sinc`, `xlogx`, `clampint`, `inv`, `ifelse` | ✅ oracle, cuda, hip, sycl |
| `normalize`, `sqdist`, `weightedsqnorm`, `weightedsqdist` (iso, diag, sym) | ✅ same |
| `one_hot`, `elem`, `extract` (gradients by symbolic differentiation) | ✅ same |
| `matvecmult`, `vecmatmult`, `tensorprod`, `keops_tensordot`, `keops_kron` | ✅ same |
| `bspline` (differentiable here, keopscore's is not) | ✅ same |
| `softdtw_sqdist` (differentiable in x, y **and** γ; keopscore rejects γ) | ✅ oracle, cuda, hip, sycl (symox ≥ b8d7adda, see below) |

Evidence: `tests/test_ops.py` (oracle against an independent NumPy
transcription of keopscore's C, plus Cox-de Boor for `bspline` and
`torch.autograd` for the gradients); `tests/conformance/test_ops.py`
(cuda 980 Ti, hip R9700 and sycl G31: 20 passed each).

Comparison with PyKeOps 2.4b0 (980 Ti, `use_fast_math=False`, one j row, 31
cases × float32/float64, 2000 i rows): **bit-for-bit identical** results
for `relu step round(0) mod clampint one_hot elem extract tensorprod
keops_kron keops_tensordot` (without contraction), order-0 `bspline`, and for
`sinxdivx xlogx inv` (including in float32); ≤ 3 ulp on expressions without
long sums (`sinc normalize sqdist weightedsqnorm` iso/diag, `bspline`
1 and 3); contractions (`matvecmult`, `vecmatmult`, `tensordot`,
symmetric `weightedsqnorm`) differ in the order of additions, with a
relative error with respect to FP64 of the same order on both sides. With the PyKeOps
default (`use_fast_math=True`), PyKeOps float32 returns `sinxdivx(1e-30) = 0`
and an error of 9e-4 on `xlogx`; Kernox stays IEEE by default
(`fast_math=False`). PyKeOps fails to compile `bspline` in float64 ("subtraction of two
c_variable is only possible with same dtype").

Deliberate deviations from keopscore:

- `round(x, d ≠ 0)`: symox rewrites `/10^d` as `×10^-d` (rounded constant),
  hence a deviation of ≤ 1 ulp (12 to 36 % of the test values, float32 and float64).
  `round(x)` (d = 0) is exact and returns `−0` as C does.
- `one_hot`: the bound `(int)(x + .5)` is evaluated **exactly**
  (`k − 1/2 ≤ x < k + 1/2`); keopscore computes `x + .5` in double, which
  amounts to the same in float32 but rounds in float64 just below a
  half-integer (`0.49999999999999994` → 1 in keopscore, 0 here). Outside
  `[0, D)` and for NaN: zero vector (undefined behavior in keopscore).
- `clamp`/`clampint` remain `min(max(x, a), b)`: identical except for NaN
  (keopscore propagates NaN; here the result depends on `fmin`/`fmax`).
- `normalize` uses `rsqrt(0) = 0` as keopscore does, with an exact `1/sqrt`;
  `kp.rsqrt` remains `1/sqrt(x)` (inf at 0), unlike keopscore's `Rsqrt`.
- `bspline`, `one_hot`, `round`, `mod` are differentiable (symbolic
  derivatives, piecewise zero for piecewise-constant
  operations); keopscore rejects the gradient of `bspline`.

`softdtw_sqdist`: keopscore recurrence unrolled into a DAG of O(n·m) shared
nodes (three-argument `min`, shift by the minimum as in
keopscore), with no hand-written function; the gradient is the symbolic
derivative of this DAG (the contribution of `min` cancels out analytically).
Values equal to a NumPy transcription of keopscore's C to within 1e-13
(`tests/test_ops.py`, n×m up to 8×5), gradients equal to
`torch.autograd` on the full recurrence. Construction (symox
main, commits 22104451, 4aa32743 and b8d7adda):
8×8 0.6 s (plan 0.3 s, oracle 0.7 s), 16×16 5.2 s (plan 5.5 s, oracle
12.6 s); polynomial growth, roughly (n·m)^2.3. With symox b02db0ca, 4×5 took 48 s and each additional cell
multiplied the time. The nested form `min(min(a, b), c)` goes through
SymPy's Python algorithm and remains slower (12×12: 23 s, versus 1.5 s).
On the Kernox side, `builder.fold_constants` and `builder.realify` also
traversed the unfolded tree (memoized), and so did the plan key (`srepr`):
beyond 200,000 unfolded nodes, the output is serialized as a graph
(existing keys unchanged).
The oracle now evaluates formulas containing `Piecewise` (where symox disables
Torch CSE) with the region-based CSE of the GPU backends, by dependency
level.

`solve` and linear operator (`kernox/linalg.py`): PyKeOps writes
`K.solve(Vj(b), alpha=)` on a `LazyTensor`; Kernox names the unknown,
a j variable of the SUM reduction: `run.solve("b", rhs, alpha=, eps=,
maxiter=, x0=, x=..., y=...)` solves `(alpha·I + K) b = rhs`, `K` being
`v ↦ run(..., b=v)`. Conjugate gradient transcribed from `ConjugateGradientSolver`
(same defaults: `alpha = 1e-10`, `eps = 1e-6`, `maxiter = 10·M`, stop when
`‖r‖ < max(eps‖b‖, eps)` over all coefficients; warning if
`maxiter` is reached); backward pass as in `KernelSolveAutograd`
(solve with `K` assumed symmetric, then `−∂⟨K⁻¹g, K(θ) a⟩/∂θ` by the
reduction itself), differentiable to any order. Deviations: linearity in
`b` is **checked symbolically** (PyKeOps assumes it); batch dimensions
accepted as in PyKeOps (single CG on the block-diagonal system), and
`rhs` may carry its own broadcast batch; default `maxiter` is `10·M`
(PyKeOps uses `10·b.shape[0]`, i.e. 10·B when `b` is batched); `run.linear_operator(var, ...)`
exposes `shape = (M·Do, N·Dv)`, `matvec`/`matmat` (shapes `(N, Dv)`,
`(N·Dv,)`, `(N·Dv, k)`, NumPy or Torch), `rmatvec` via the gradient plan
(vector-Jacobian product, same result as PyKeOps' `T` which
transposes the formula), `T`, `t()` and `@`. Tests against Torch dense
algebra, `scipy.sparse.linalg.eigsh` and `cg` (`tests/test_linalg.py`).
