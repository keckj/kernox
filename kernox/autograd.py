"""Gradients of reductions (``torch.autograd`` integration, proposed contract v0.4).

SUM (below): fused gradient reductions, at any order. The other floating-point
reductions reduce to this mechanism or to the selected term (first order):

- LOGSUMEXP, LOGSUMEXPWEIGHT, SUMSOFTMAXWEIGHT: surrogate SUM plan, with the
  forward outputs held fixed (`surrogate`);
- MIN, MAX, KMIN and the values of MIN_ARGMIN, MAX_ARGMAX, KMIN_ARGKMIN:
  derivative at the selected pair, evaluated on O(M·K) pairs
  (`SelectGradient`).

For a normalized SUM plan (reduction over j)::

    out[i, e] = Σ_j F_e(x_i, y_j, p)          (x: "i" args, y: "j" args, p: params)

and an upstream gradient ``G[i, e]``, the gradients are themselves fused SUM
reductions, differentiated **symbolically** (``symox.diff``) and executed by
the same Plan → backend pipeline, with no M×N matrix:

- i side (one plan, reduction over j): for each component ``c`` of an i arg,
  ``∂L/∂x_i[c] = Σ_j Σ_e G[i,e] ∂F_e/∂x[c]``; for a parameter, the per-row
  contribution ``Σ_j Σ_e G[i,e] ∂F_e/∂p[c]`` is one more output of the same
  plan, then summed over i by Torch (M × P vector);
- j side (one plan, i/j roles swapped, hence a reduction over i):
  ``∂L/∂y_j[c] = Σ_i Σ_e G[i,e] ∂F_e/∂y[c]``.

``G`` enters as an extra i variable of dimension E. All the derivatives of a
side are outputs of the **same** plan: the generator's CSE (including the
region-based one with ``where``) shares the common subexpressions
(``exp(-d²)``, ``d²``, differences ``x - y``...).

Differentiation rules (after ``symox.diff``):

- ``where(c, a, b)``: gradient of the active branch (``Piecewise``
  differentiated branch by branch, conditions unchanged); the condition is not
  differentiated;
- ``Heaviside(z)`` (derivative of ``max``/``min``/``clamp``) becomes 1 if z > 0,
  0 if z < 0 and ``1/2`` at z = 0 (symox/SymPy convention): on ties,
  ``max(x, y)`` splits the gradient half and half;
- ``abs``: ``sign(x)`` (0 at 0); ``floor``, ``ceiling``, ``sign``: 0;
- ``DiracDelta`` (distributional parts, e.g. the second derivative of
  ``abs``): 0;
- a derivative that symox leaves unevaluated (``Derivative``): ``UnsupportedError``.

Away from non-smooth points, the gradient is exact with respect to the formula
(including ``sqrt`` at 0: ``inf``, as in Torch).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .builder import component_symbol, make_plan
from .errors import UnsupportedError
from .plan import Arg, Plan, ReductionKind
from .plan import Reduction as PlanReduction

_ROLE_ORDER = {"i": 0, "j": 1, "param": 2}


def _smooth(expr):
    """Replaces the distributions and step functions produced by ``diff``."""
    import symox

    def heaviside(*args):
        z = args[0]
        h0 = args[1] if len(args) > 1 else symox.Rational(1, 2)
        return symox.Piecewise((1, z > 0), (0, z < 0), (h0, True))

    out = expr.replace(lambda e: isinstance(e, symox.DiracDelta), lambda e: symox.S.Zero)
    out = out.replace(lambda e: isinstance(e, symox.Heaviside), lambda e: heaviside(*e.args))
    if out.has(symox.Derivative):
        bad = sorted({str(d) for d in out.atoms(symox.Derivative)})
        raise UnsupportedError(f"derivative cannot be evaluated symbolically: {', '.join(bad)}")
    return out


def _factor_pow2(expr):
    """Factors a common 2^k coefficient (k ≥ 1) back out of the terms of a sum.

    symox distributes numbers over sums: ``∂(a − b)²/∂a`` becomes
    ``2a − 2b``. The difference ``a − b`` is then no longer shared with the
    forward formula, which costs two operations per component and per pair in
    the gradient. Factoring ``2(a − b)`` back is exact (multiplying by 2^k
    commutes with rounding), barring overflow or subnormal results. Other
    coefficients are left alone: factoring them out would change the rounding.
    """
    import symox

    def rule(x):
        cs = [t.as_coeff_Mul() for t in x.args]
        c0 = abs(cs[0][0])
        if not (c0.is_Integer and c0 > 1 and (int(c0) & (int(c0) - 1)) == 0):
            return x
        if any(abs(c) != c0 for c, _ in cs):
            return x
        inner = symox.Add(*[(c / c0) * r for c, r in cs])
        if inner.could_extract_minus_sign():
            return symox.Mul(-c0, -inner, evaluate=False)
        return symox.Mul(c0, inner, evaluate=False)

    return expr.replace(lambda x: isinstance(x, symox.Add), rule)


def derivative(expr, symbol):
    """``∂expr/∂symbol`` following the rules of this module."""
    import symox

    return _factor_pow2(_smooth(symox.diff(expr, symbol)))


def _fresh_name(plan: Plan, base: str) -> str:
    names = {a.name for a in plan.args}
    name, n = base, 0
    while name in names:
        n += 1
        name = f"{base}{n}"
    return name


@dataclass(frozen=True)
class GradientPlans:
    """Gradient plans of a SUM plan.

    ``grad_name``: name of the ``G`` input (M, E). ``i_plan`` (or None): outputs
    ``i_layout`` = ((arg name, role, dim), ...) in column order; args with role
    ``param`` yield per-row (i) contributions that must be summed.
    ``j_plan`` (or None): reduction over i (roles swapped), outputs
    ``j_layout`` for the j args.
    """

    grad_name: str
    i_plan: Plan | None
    i_layout: tuple
    j_plan: Plan | None
    j_layout: tuple


def gradient_plans(plan: Plan, wrt) -> GradientPlans:
    """Builds the gradient plans of ``plan`` (SUM) with respect to the args ``wrt``."""
    import symox

    if plan.reduction.kind is not ReductionKind.SUM:
        raise UnsupportedError(f"{plan.reduction.kind.value} is not differentiable (only SUM is)")
    wrt = set(wrt)
    gname = _fresh_name(plan, "kp_grad")
    E = len(plan.outputs)
    gsyms = tuple(component_symbol(gname, e) for e in range(E))

    def side(roles, targets):
        """Plan of one pass: ``roles`` maps each original role to its new role."""
        if not targets:
            return None, ()
        outputs, layout = [], []
        for a in targets:
            for s in a.symbols:
                outputs.append(symox.Add(*(g * derivative(F, s) for g, F in zip(gsyms, plan.outputs))))
            layout.append((a.name, a.role, a.dim))
        args = [Arg(a.name, roles[a.role], a.dim, a.symbols) for a in plan.args]
        args.append(Arg(gname, roles["i"], E, gsyms))
        args.sort(key=lambda a: (_ROLE_ORDER[a.role], a.name))
        meta = {"gradient_of": plan.key(), "gradient_side": "i" if roles["i"] == "i" else "j"}
        p = make_plan(outputs, ReductionKind.SUM, plan.dtype, args=args, acc_dtype=plan.acc_dtype,
                      fp_contract=plan.fp_contract, sum_scheme=plan.sum_scheme, meta=meta, fast_math=plan.fast_math,
                      flush_denormals=plan.flush_denormals)
        return p, tuple(layout)

    i_targets = [a for a in plan.args if a.name in wrt and a.role in ("i", "param")]
    j_targets = [a for a in plan.args if a.name in wrt and a.role == "j"]
    i_plan, i_layout = side({"i": "i", "j": "j", "param": "param"}, i_targets)
    j_plan, j_layout = side({"i": "j", "j": "i", "param": "param"}, j_targets)
    return GradientPlans(gname, i_plan, i_layout, j_plan, j_layout)


# --------------------------------------------------------------------------
# LOGSUMEXP, LOGSUMEXPWEIGHT, SUMSOFTMAXWEIGHT: surrogate SUM plan
# --------------------------------------------------------------------------
SURROGATE_KINDS = (ReductionKind.LOGSUMEXP, ReductionKind.LOGSUMEXPWEIGHT, ReductionKind.SUMSOFTMAXWEIGHT)


@dataclass
class Surrogate:
    """SUM plan whose gradient, with the forward outputs held fixed, is that of
    ``plan``; ``fixed``: ((name, dim, source), ...), source ``"out"`` (forward
    output) or ``"lse"`` (log Σ_j exp f, LOGSUMEXP reduction ``lse_plan``)."""

    plan: Plan
    fixed: tuple
    lse_plan: Plan | None


def surrogate(plan: Plan) -> Surrogate:
    """Gradients of softmax-type reductions (forward output ``L``, ``O`` held fixed):

    - LOGSUMEXP ``L_e = log Σ_j exp f_e``: ``∂L_e = Σ_j exp(f_e − L_e) ∂f_e``,
      the gradient of ``Σ_j exp(f_e − L_e)`` with ``L`` fixed;
    - LOGSUMEXPWEIGHT ``L_e = log Σ_j exp(f) g_e``: the gradient of
      ``Σ_j exp(f − L_e) g_e``;
    - SUMSOFTMAXWEIGHT ``O_e = Σ_j w_j g_e``, ``w = exp(f − S)``, ``S = log Σ exp f``:
      ``∂O_e = Σ_j w (∂g_e + (g_e − O_e) ∂f)``, the gradient of ``Σ_j exp(f − S)(g_e − O_e)``.

    A row with no contribution (``L = −∞``) has a zero gradient (not NaN).
    """
    import symox

    kind, outs = plan.reduction.kind, plan.outputs

    def fixed(base, dim):
        name = _fresh_name(plan, base)
        return name, tuple(component_symbol(name, c) for c in range(dim))

    def w(f, lse):
        return symox.Piecewise((0, symox.Eq(lse, -symox.oo)), (symox.exp(f - lse), True))

    lse_plan = None
    if kind is ReductionKind.LOGSUMEXP:
        lname, L = fixed("kp_lse", len(outs))
        new = [w(f, l_) for f, l_ in zip(outs, L)]
        extra = [(lname, L, "out")]
    elif kind is ReductionKind.LOGSUMEXPWEIGHT:
        f, g = outs[0], outs[1:]
        lname, L = fixed("kp_lse", len(g))
        new = [w(f, l_) * ge for ge, l_ in zip(g, L)]
        extra = [(lname, L, "out")]
    elif kind is ReductionKind.SUMSOFTMAXWEIGHT:
        f, g = outs[0], outs[1:]
        sname, S = fixed("kp_lse", 1)
        oname, O = fixed("kp_out", len(g))
        new = [w(f, S[0]) * (ge - oe) for ge, oe in zip(g, O)]
        extra = [(sname, S, "lse"), (oname, O, "out")]
        lse_plan = dataclasses.replace(plan, outputs=(f,), reduction=PlanReduction(ReductionKind.LOGSUMEXP))
    else:
        raise UnsupportedError(f"{kind.value}: no surrogate plan")
    args = list(plan.args) + [Arg(name, "i", len(syms), syms) for name, syms, _ in extra]
    args.sort(key=lambda a: (_ROLE_ORDER[a.role], a.name))
    p = make_plan(new, ReductionKind.SUM, plan.dtype, args=args, acc_dtype=plan.acc_dtype,
                  fp_contract=plan.fp_contract, sum_scheme=plan.sum_scheme, fast_math=plan.fast_math,
                  flush_denormals=plan.flush_denormals, meta={"surrogate_of": plan.key()})
    p = dataclasses.replace(p, batch_rank=plan.batch_rank, ranges=plan.ranges)
    return Surrogate(p, tuple((name, len(syms), src) for name, syms, src in extra), lse_plan)


# --------------------------------------------------------------------------
# MIN, MAX, KMIN (and values of MIN_ARGMIN, MAX_ARGMAX, KMIN_ARGKMIN): selected index
# --------------------------------------------------------------------------
SELECT_KINDS = {
    ReductionKind.MIN: ReductionKind.ARGMIN, ReductionKind.MAX: ReductionKind.ARGMAX,
    ReductionKind.KMIN: ReductionKind.ARGKMIN, ReductionKind.MIN_ARGMIN: None,
    ReductionKind.MAX_ARGMAX: None, ReductionKind.KMIN_ARGKMIN: None,
}


def index_plan(plan: Plan) -> Plan:
    """Plan of the indices selected by a MIN/MAX/KMIN reduction (forward pass of the gradient)."""
    kind = SELECT_KINDS[plan.reduction.kind]
    return dataclasses.replace(plan, reduction=PlanReduction(kind, plan.reduction.k))


class SelectGradient:
    """Gradient of a selection reduction: only the selected term depends on the inputs.

    ``v[i, c] = F_e(x_i, y_{j*(i, c)}, p)`` (``e = c`` for MIN/MAX, ``e = 0``
    and ``c = k`` for KMIN): ``∂L/∂x_i = Σ_c G[i, c] ∂F_e/∂x`` at the selected
    pair, ``∂L/∂y_j = Σ_{(i, c): j*(i, c) = j} G[i, c] ∂F_e/∂y``. The symbolic
    derivatives are evaluated by Torch **on the selected pairs only**
    (O(M·C), never M×N); an index of −1 (no candidate) does not contribute.
    On ties, the gradient goes to the selected index (the smallest j), as in PyKeOps.
    """

    def __init__(self, plan: Plan, wrt):
        from .oracle import _torch_function

        self.plan = plan
        kinds = (ReductionKind.KMIN, ReductionKind.KMIN_ARGKMIN)
        self.per_k = plan.reduction.kind in kinds
        self.targets = [a for a in plan.args if a.name in wrt]
        self.symbols = [s for a in plan.args for s in a.symbols]
        self.fns = []
        for f in plan.outputs:
            exprs = [derivative(f, s) for a in self.targets for s in a.symbols]
            self.fns.append(_torch_function(self.symbols, exprs))

    def __call__(self, tensors: dict, index, grad):
        import torch

        from .runtime import torch_dtype

        plan = self.plan
        dt = torch_dtype(plan.dtype)
        E = len(plan.outputs)
        grad = grad.to(dt)
        valid = index >= 0
        safe = index.clamp(min=0)
        bshape = index.shape[:-2]
        M, C = index.shape[-2:]
        out = {a.name: None for a in self.targets}
        for e in range(E):
            cols = list(range(C)) if self.per_k else [e]
            idx = safe[..., cols]                       # (*B, M, C_e)
            ok = valid[..., cols]
            g = torch.where(ok, grad[..., cols], torch.zeros((), dtype=dt, device=grad.device))
            values = []
            for a in plan.args:
                t = tensors[a.name].to(dt)
                for c in range(a.dim):
                    if a.role == "i":
                        values.append(t[..., :, c].unsqueeze(-1))          # (*B, M, 1)
                    elif a.role == "j":
                        col = t[..., :, c]
                        col = col.expand(*bshape, col.shape[-1]) if bshape else col
                        values.append(torch.gather(col, -1, idx.reshape(*bshape, -1)).reshape(idx.shape))
                    else:
                        values.append(t[c])
            ders = self.fns[e](*values, dtype=dt)
            k = 0
            for a in self.targets:
                parts = []
                for _ in range(a.dim):
                    d = torch.as_tensor(ders[k], dtype=dt, device=grad.device).expand(idx.shape)
                    parts.append(torch.where(ok, d, torch.zeros((), dtype=dt, device=grad.device)) * g)
                    k += 1
                contrib = torch.stack(parts, dim=-1)            # (*B, M, C_e, dim)
                if a.role == "i":
                    r = contrib.sum(dim=-2)                     # (*B, M, dim)
                elif a.role == "param":
                    r = contrib.reshape(*bshape, -1, a.dim).sum(dim=-2)
                else:
                    n = tensors[a.name].shape[-2]
                    r = torch.zeros(*bshape, n, a.dim, dtype=dt, device=grad.device)
                    flat = idx.reshape(*bshape, -1)
                    r.scatter_add_(-2, flat.unsqueeze(-1).expand(*flat.shape, a.dim),
                                   contrib.reshape(*bshape, -1, a.dim))
                out[a.name] = r if out[a.name] is None else out[a.name] + r
        grads = {}
        for a in self.targets:
            t = tensors[a.name]
            grads[a.name] = out[a.name].to(t.dtype).sum_to_size(t.shape)
        return grads


def _make_function():
    import torch

    class KpSum(torch.autograd.Function):
        """Forward pass: SUM reduction; backward pass: fused gradient reductions.

        The backward pass calls `Reduction` objects, which
        are differentiable themselves: with ``create_graph=True``, second-order
        derivatives go through the same mechanism.
        """

        @staticmethod
        def forward(ctx, red, names, *tensors):
            inputs = dict(zip(names, (t.detach() for t in tensors)))
            out = red._forward(inputs)
            ctx.red, ctx.names = red, names
            ctx.save_for_backward(*tensors)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            red, names = ctx.red, ctx.names
            tensors = ctx.saved_tensors
            wrt = {n for n, t, need in zip(names, tensors, ctx.needs_input_grad[2:]) if need}
            grads = red._backward(dict(zip(names, tensors)), grad_out, wrt)
            return (None, None, *(grads.get(n) for n in names))

    class KpNonDifferentiable(torch.autograd.Function):
        """Reductions with no defined gradient (integer outputs: COUNT, ARG*): explicit refusal."""

        @staticmethod
        def forward(ctx, red, names, *tensors):
            ctx.kind = red.plan.reduction.kind.value
            out = red._forward(dict(zip(names, (t.detach() for t in tensors))))
            if isinstance(out, tuple):
                ctx.mark_non_differentiable(out[1])
            return out

        @staticmethod
        def backward(ctx, *grads):
            raise UnsupportedError(
                f"{ctx.kind} is not differentiable in Kernox: "
                "detach the inputs or reformulate the loss")

    class KpSurrogate(torch.autograd.Function):
        """LOGSUMEXP, LOGSUMEXPWEIGHT, SUMSOFTMAXWEIGHT: gradient through the surrogate
        SUM plan (`surrogate`), fused reductions. First order only."""

        @staticmethod
        def forward(ctx, red, names, *tensors):
            out = red._forward(dict(zip(names, (t.detach() for t in tensors))))
            ctx.red, ctx.names = red, names
            ctx.save_for_backward(*tensors, out)
            return out

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_out):
            red, names = ctx.red, ctx.names
            *tensors, out = ctx.saved_tensors
            wrt = {n for n, need in zip(names, ctx.needs_input_grad[2:]) if need}
            grads = red._backward_surrogate(dict(zip(names, tensors)), out, grad_out, wrt)
            return (None, None, *(grads.get(n) for n in names))

    class KpSelect(torch.autograd.Function):
        """MIN, MAX, KMIN (and values of MIN_ARGMIN, MAX_ARGMAX, KMIN_ARGKMIN):
        gradient at the selected term (`SelectGradient`). First order only;
        the indices are not differentiable."""

        @staticmethod
        def forward(ctx, red, names, *tensors):
            out = red._forward(dict(zip(names, (t.detach() for t in tensors))))
            ctx.red, ctx.names = red, names
            index = out[1] if isinstance(out, tuple) else None
            if index is not None:
                ctx.mark_non_differentiable(index)
                ctx.save_for_backward(*tensors, index)
            else:
                ctx.save_for_backward(*tensors)
            ctx.has_index = index is not None
            return out

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_value, *_):
            red, names = ctx.red, ctx.names
            saved = ctx.saved_tensors
            tensors, index = (saved[:-1], saved[-1]) if ctx.has_index else (saved, None)
            wrt = {n for n, need in zip(names, ctx.needs_input_grad[2:]) if need}
            grads = red._backward_select(dict(zip(names, tensors)), index, grad_value, wrt)
            return (None, None, *(grads.get(n) for n in names))

    return KpSum, KpNonDifferentiable, KpSurrogate, KpSelect


_FUNCTIONS = None


def functions():
    """``(KpSum, KpNonDifferentiable)`` (compatibility)."""
    return _all()[:2]


def _all():
    global _FUNCTIONS
    if _FUNCTIONS is None:
        _FUNCTIONS = _make_function()
    return _FUNCTIONS


def function_for(kind: ReductionKind):
    """``torch.autograd`` function of the reduction ``kind``."""
    kp_sum, kp_nondiff, kp_surrogate, kp_select = _all()
    if kind is ReductionKind.SUM:
        return kp_sum
    if kind in SURROGATE_KINDS:
        return kp_surrogate
    if kind in SELECT_KINDS:
        return kp_select
    return kp_nondiff


__all__ = ["GradientPlans", "gradient_plans", "derivative", "functions", "function_for", "surrogate",
           "SelectGradient"]
