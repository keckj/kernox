"""Linear operators and ``solve`` (PyKeOps parity).

A SUM reduction that is **linear** in a ``j`` variable (``var``) defines
the operator ``K : v ↦ run(..., var=v)`` from ``(N, Dv)`` to ``(M, Do)``:
``(K v)_i = Σ_j J(x_i, y_j) v_j`` with ``J = ∂F/∂var`` (Do × Dv), constant in
``var``. :class:`LinearOperator` exposes it with the interface of
:mod:`scipy.sparse.linalg` (``shape``, ``dtype``, ``matvec``, ``rmatvec``,
``matmat``, ``T``, ``@``), like ``LazyTensor`` on the PyKeOps side;
:meth:`LinearOperator.solve` solves ``(α I + K) a = b`` by conjugate gradient
(transcription of PyKeOps' ``ConjugateGradientSolver``, see THIRD_PARTY.md),
differentiable with respect to ``b`` and to the other bound inputs.

Linearity is **checked symbolically** (zero outputs at ``var = 0``,
derivatives free of ``var``) before any launch; never assumed.
"""

from __future__ import annotations

import warnings

from .errors import UnsupportedError, ValidationError
from .plan import ReductionKind

#: Valeurs par défaut de PyKeOps (``LazyTensor.solve``, ``ConjugateGradientSolver``).
DEFAULT_ALPHA = 1e-10
DEFAULT_EPS = 1e-6


def conjugate_gradient(linop, b, *, x0=None, eps: float = DEFAULT_EPS, maxiter: int | None = None):
    """PyKeOps conjugate gradient (``pykeops.common.operations.ConjugateGradientSolver``).

    ``linop`` symmetric positive definite; stops when ``‖r‖ < max(eps‖b‖, eps)``
    (norm over all coefficients, like PyKeOps); ``maxiter`` defaults to
    ``10 · b.shape[0]``. Returns ``(x, info)``; warns
    (``RuntimeWarning``) if ``maxiter`` is reached.
    """
    import torch

    atol = max(eps, eps * float(torch.linalg.vector_norm(b)))
    maxiter = 10 * b.shape[0] if maxiter is None else int(maxiter)
    if maxiter < 0:
        raise ValidationError(f"maxiter >= 0 attendu, reçu {maxiter}")
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b.clone() if x0 is None else b - linop(x0)
    nr2 = (r * r).sum()
    niter, converged = 0, bool(nr2 <= atol * atol)
    if not converged:
        p = r.clone()
        for niter in range(1, maxiter + 1):
            mp = linop(p)
            step = nr2 / (p * mp).sum()
            x += step * p
            r -= step * mp
            nr2new = (r * r).sum()
            if nr2new < atol * atol:
                converged, nr2 = True, nr2new
                break
            p = r + (nr2new / nr2) * p
            nr2 = nr2new
        if not converged:
            warnings.warn(f"kernox solve : {maxiter} itérations de gradient conjugué sans convergence "
                          f"(résidu {float(nr2.sqrt()):.3e} > {atol:.3e})", RuntimeWarning, stacklevel=3)
    info = {"converged": converged, "niter": niter, "residual_norm": float(nr2.sqrt()), "atol": atol,
            "maxiter": maxiter}
    return x, info


def check_linear(plan, var: str):
    """Raises :class:`ValidationError` if the plan is not a SUM linear in ``var`` (``j`` argument)."""
    import symox

    from .autograd import derivative

    if plan.reduction.kind is not ReductionKind.SUM:
        raise ValidationError(f"opérateur linéaire : réduction SUM attendue, reçu {plan.reduction.kind.value}")
    if plan.batch_rank:
        raise UnsupportedError("opérateur linéaire : plan déjà batché (batch_rank) ; compiler sans batch, "
                               "le batch vient des entrées")
    arg = next((a for a in plan.args if a.name == var), None)
    if arg is None:
        raise ValidationError(f"opérateur linéaire : {var!r} n'est pas une entrée ({[a.name for a in plan.args]})")
    if arg.role != "j":
        raise ValidationError(f"opérateur linéaire : {var!r} doit être une variable j (réduite), "
                              f"rôle {arg.role!r}")
    syms = set(arg.symbols)
    zero = {s: symox.Integer(0) for s in arg.symbols}
    for k, out in enumerate(plan.outputs):
        if not syms & out.free_symbols:
            if out != 0:
                raise ValidationError(f"opérateur linéaire : composante {k} indépendante de {var!r} et non nulle "
                                      "(application affine, pas linéaire)")
            continue
        if out.xreplace(zero) != 0:
            raise ValidationError(f"opérateur linéaire : composante {k} non nulle en {var}=0 ; "
                                  "linéarité non démontrée")
        for s in arg.symbols:
            if syms & derivative(out, s).free_symbols:
                raise ValidationError(f"opérateur linéaire : composante {k} non linéaire en {var!r}")
    return arg


class LinearOperator:
    """Operator ``v ↦ run(**inputs, var=v)`` bound to inputs (``run.linear_operator``).

    ``shape = (M·Do, N·Dv)`` and ``dtype`` (NumPy) for
    :func:`scipy.sparse.linalg.aslinearoperator`. ``matvec(v)`` accepts a
    ``(N, Dv)`` tensor (returned as ``(M, Do)``), a ``(N·Dv,)`` vector or a
    ``(N·Dv, k)`` matrix (flattened columns); NumPy arrays are converted to the
    device of the inputs and returned as NumPy. ``matvec`` is differentiable (it is
    the reduction itself); so is ``rmatvec`` (vector-Jacobian product).
    """

    def __init__(self, red, var: str, inputs: dict, *, _transposed: bool = False):
        import torch

        plan = red.plan
        self._arg = check_linear(plan, var)
        if var in inputs:
            raise ValidationError(f"opérateur linéaire : {var!r} est l'inconnue, ne pas la fournir")
        self.red, self.var, self.plan = red, var, plan
        self._raw = dict(inputs)
        n = rank = None
        for a in plan.args:
            v = inputs.get(a.name)
            if a.role != "param" and a.name != var and isinstance(v, torch.Tensor):
                rank = v.dim() - 2 if rank is None else rank
                if a.role == "j" and n is None:
                    n = v.shape[-2]
        if n is None:
            raise ValidationError(f"opérateur linéaire : N indéterminé ({var!r} est la seule variable j)")
        dt = _torch_dtype(plan.dtype)
        dev = next((v.device for v in inputs.values() if isinstance(v, torch.Tensor)), None)
        if dev is None:
            raise ValidationError("opérateur linéaire : aucune entrée tenseur pour fixer le device")
        self.rank = max(rank, 0)
        probe = torch.zeros((1,) * self.rank + (n, self._arg.dim), dtype=dt, device=dev)
        detached = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        base = getattr(red, "red", red)  # RangedReduction -> Reduction
        tensors, device, m, n2, batch = base.validate({**detached, var: probe})
        self.batch = tuple(batch) if batch is not None else ()
        self._tensors, self.device, self.m, self.n = tensors, device, m, n2
        self.dv, self.do = self._arg.dim, plan.out_dim
        self._dt = dt
        self._transposed = _transposed

    # ---- interface SciPy
    @property
    def shape(self):
        s = (self.m * self.do, self.n * self.dv)
        return self.batch + (s[::-1] if self._transposed else s)

    @property
    def dtype(self):
        import numpy as np

        return np.dtype(self.plan.dtype)

    @property
    def T(self):  # noqa: N802 (nom NumPy / PyKeOps)
        t = object.__new__(LinearOperator)
        t.__dict__.update(self.__dict__)
        t._transposed = not self._transposed
        return t

    def t(self):
        return self.T

    def matvec(self, v):
        return self._apply(v, self._transposed)

    def rmatvec(self, v):
        return self._apply(v, not self._transposed)

    def matmat(self, v):
        return self._apply(v, self._transposed)

    def rmatmat(self, v):
        return self._apply(v, not self._transposed)

    def __matmul__(self, v):
        return self.matvec(v)

    def __repr__(self):
        return (f"kernox.LinearOperator({self.var!r}, shape={self.shape}, dtype={self.plan.dtype}"
                f"{', transposé' if self._transposed else ''})")

    # ---- produits
    def _apply(self, v, adjoint: bool):
        """``K v`` (``adjoint=False``) ou ``Kᵀ v``, formes acceptées selon la classe."""
        import numpy as np
        import torch

        rows, cols = (self.n, self.dv) if adjoint else (self.m, self.do)  # sortie
        in_rows, in_cols = (self.m, self.do) if adjoint else (self.n, self.dv)
        as_numpy = isinstance(v, np.ndarray)
        if as_numpy:
            v = torch.from_numpy(np.ascontiguousarray(v)).to(device=self.device, dtype=self._dt)
        elif not isinstance(v, torch.Tensor):
            raise ValidationError(f"opérateur linéaire : torch.Tensor ou numpy.ndarray attendu, "
                                  f"reçu {type(v).__name__}")
        if v.dim() >= 2 and tuple(v.shape[-2:]) == (in_rows, in_cols):
            out = self._product(v, adjoint)
        elif self.rank and v.dim() <= 2:
            raise ValidationError(f"opérateur linéaire batché : forme (*B, {in_rows}, {in_cols}) attendue, "
                                  f"reçu {tuple(v.shape)} (formes aplaties réservées à l'opérateur sans batch)")
        elif v.dim() == 1 and v.shape[0] == in_rows * in_cols:
            out = self._product(v.reshape(in_rows, in_cols), adjoint).reshape(rows * cols)
        elif v.dim() == 2 and v.shape[0] == in_rows * in_cols:
            cols_out = [self._product(v[:, c].reshape(in_rows, in_cols), adjoint).reshape(rows * cols)
                        for c in range(v.shape[1])]
            out = torch.stack(cols_out, dim=1) if cols_out else v.new_zeros(rows * cols, 0)
        else:
            raise ValidationError(f"opérateur linéaire : forme {tuple(v.shape)} incompatible "
                                  f"(attendu (*B, {in_rows}, {in_cols}), ({in_rows * in_cols},) "
                                  f"ou ({in_rows * in_cols}, k))")
        return out.detach().cpu().numpy() if as_numpy else out

    def _check_operand(self, v):
        if v.device != self.device:
            raise ValidationError(f"opérateur linéaire : device {v.device} différent de {self.device}")
        if v.dtype != self._dt:
            raise ValidationError(f"opérateur linéaire : dtype {v.dtype} différent de {self.plan.dtype}")

    def _product(self, v, adjoint: bool):
        self._check_operand(v)
        import torch

        v = v.contiguous()  # colonnes de matmat, tableaux NumPy : l'opérateur rend contigu
        r = max(self.rank, v.dim() - 2)
        v = self._lift_one(v, v.dim() - 2, r)
        if not adjoint:
            return self.red(**self._lift(self._raw, r), **{self.var: v})
        # Kᵀ u = ∂⟨u, K z⟩/∂z, indépendant de z par linéarité : plan de gradient de la
        # réduction, lui-même différentiable (entrées liées et u) quand Torch le demande.
        batch = tuple(torch.broadcast_shapes(self.batch, v.shape[:-2])) if r else ()
        z = self._tensors[self.var].new_zeros(batch + (self.n, self.dv))
        bound = self._lift(self._bound(), r)
        return self.red._backward({**bound, self.var: z}, v, frozenset((self.var,)))[self.var]

    @staticmethod
    def _lift_one(t, have: int, r: int):
        return t.reshape((1,) * (r - have) + tuple(t.shape)) if r > have else t

    def _lift(self, values: dict, r: int) -> dict:
        """Entrées liées vues au rang de batch ``r`` (dimensions 1 en tête, sans copie)."""
        import torch

        if r == self.rank:
            return dict(values)
        out = {}
        for a in self.plan.args:
            t = values.get(a.name)
            if a.name == self.var or not isinstance(t, torch.Tensor):
                if a.name in values:
                    out[a.name] = t
            elif a.role != "param":
                out[a.name] = self._lift_one(t, t.dim() - 2, r)
            else:
                out[a.name] = self._lift_one(t, t.dim() - 1, r) if t.dim() > 1 else t
        return out

    def _bound(self, replace=None):
        """Entrées validées, les tenseurs fournis par l'appelant gardant leur graphe Torch."""
        import torch

        out = dict(self._tensors)
        out.update((k, v) for k, v in self._raw.items() if isinstance(v, torch.Tensor))
        if replace:
            out.update(replace)
        return out

    # ---- solve
    def solve(self, b, *, alpha: float = DEFAULT_ALPHA, eps: float = DEFAULT_EPS, maxiter: int | None = None,
              x0=None, return_info: bool = False):
        """Solves ``(alpha·I + K) a = b`` by conjugate gradient (PyKeOps' ``LazyTensor.solve``).

        ``K`` must be square (``M = N``, ``Do = Dv``) and, for the conjugate
        gradient to converge, symmetric positive definite (``alpha`` regularizes);
        symmetry is not checked (like PyKeOps) and is used by the backward
        pass (``Kᵀ = K``). ``b``: ``(*Bb, M, Do)``; result ``(*B, N, Dv)``,
        ``B`` being the broadcast of the batch of the bound inputs and of ``Bb`` (one system per
        element; like PyKeOps, the conjugate gradient operates on the whole
        block-diagonal system, with norms over all coefficients). ``maxiter``
        defaults to ``10·M``. Differentiable with respect to ``b`` and to the bound tensor inputs,
        at any order (``create_graph``).
        """
        import torch

        if self._transposed:
            raise UnsupportedError("solve sur l'opérateur transposé : résoudre sur l'opérateur direct "
                                   "(symétrique par hypothèse)")
        if self.m != self.n or self.do != self.dv:
            raise ValidationError(f"solve : opérateur carré requis, reçu ({self.m}, {self.do}) <- "
                                  f"({self.n}, {self.dv})")
        if not isinstance(b, torch.Tensor):
            raise ValidationError(f"solve : torch.Tensor ({self.m}, {self.do}) attendu, reçu {type(b).__name__}")
        if b.dim() < 2 or tuple(b.shape[-2:]) != (self.m, self.do):
            raise ValidationError(f"solve : b de forme (*B, {self.m}, {self.do}) attendu, reçu {tuple(b.shape)}")
        self._check_operand(b)
        try:
            out_shape = tuple(torch.broadcast_shapes(self.batch, b.shape[:-2])) + (self.n, self.dv)
        except RuntimeError:
            raise ValidationError(f"solve : batch de b {tuple(b.shape[:-2])} non diffusable avec {self.batch}") \
                from None
        if x0 is not None:
            if not isinstance(x0, torch.Tensor) or tuple(x0.shape) != out_shape:
                raise ValidationError(f"solve : x0 de forme {out_shape} attendu")
            self._check_operand(x0)
            x0 = x0.detach()
        alpha = float(alpha)
        if not alpha >= 0.0:
            raise ValidationError(f"solve : alpha >= 0 attendu, reçu {alpha}")
        names = tuple(k for k, v in self._raw.items() if isinstance(v, torch.Tensor))
        opts = (alpha, float(eps), 10 * self.m if maxiter is None else maxiter, x0, out_shape)
        info = {}
        x = _solve_function().apply(self, names, opts, info, b, *(self._raw[k] for k in names))
        return (x, info) if return_info else x

    def _linop(self, alpha: float, r: int):
        """``v ↦ (alpha I + K) v`` sur tenseurs détachés validés, au rang de batch ``r``."""
        tensors, var, red = self._lift(self._tensors, r), self.var, self.red

        def linop(v):
            out = red._forward({**tensors, var: v.contiguous()})
            return out + alpha * v if alpha else out

        return linop


_SOLVE = None


def _solve_function():
    global _SOLVE
    if _SOLVE is not None:
        return _SOLVE
    import torch

    class KpSolve(torch.autograd.Function):
        """``a = (αI + K)⁻¹ b`` ; arrière : ``ḃ = (αI + K)⁻¹ g``, ``θ̄ = -∂⟨ḃ, K(θ) a⟩/∂θ``."""

        @staticmethod
        def forward(ctx, op, names, opts, info, b, *tensors):
            alpha, eps, maxiter, x0, out_shape = opts
            r = len(out_shape) - 2
            bb = op._lift_one(b.detach(), b.dim() - 2, r).expand(out_shape).contiguous()
            x, i = conjugate_gradient(op._linop(alpha, r), bb, x0=x0, eps=eps, maxiter=maxiter)
            info.update(i)
            ctx.op, ctx.names, ctx.opts, ctx.b_shape = op, names, opts, b.shape
            ctx.save_for_backward(x, *tensors)
            return x

        @staticmethod
        def backward(ctx, grad):
            op, names, (alpha, eps, maxiter, _, out_shape) = ctx.op, ctx.names, ctx.opts
            r = len(out_shape) - 2
            x, *tensors = ctx.saved_tensors
            create = torch.is_grad_enabled()
            needs = ctx.needs_input_grad[5:]
            kinv_g = KpSolve.apply(op, names, (alpha, eps, maxiter, None, out_shape), {}, grad.contiguous(),
                                   *tensors)
            grads = [None] * len(tensors)
            want = [names[k] for k, n in enumerate(needs) if n]
            if want:
                # θ̄ = -∂⟨K⁻¹g, K(θ) a⟩/∂θ : plans de gradient de la réduction en (θ, var = a),
                # appelés directement (ni autograd.grad, qui retraverserait KpSolve) ;
                # avec create_graph, ces réductions sont elles-mêmes différentiables.
                bound = op._lift(op._bound(dict(zip(names, tensors))), r)
                bound[op.var] = x
                if not create:
                    bound = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in bound.items()}
                    kinv_g = kinv_g.detach()
                g = op.red._backward(bound, -kinv_g, frozenset(want))
                grads = [g.get(k) if needs[i] else None for i, k in enumerate(names)]
            kinv_g = kinv_g.sum_to_size(ctx.b_shape) if ctx.needs_input_grad[4] else None
            return (None, None, None, None, kinv_g, *grads)

    _SOLVE = KpSolve
    return _SOLVE


def _torch_dtype(name):
    from .runtime import torch_dtype

    return torch_dtype(name)


__all__ = ["LinearOperator", "conjugate_gradient", "check_linear", "DEFAULT_ALPHA", "DEFAULT_EPS"]
