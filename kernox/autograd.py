"""Gradients des réductions (intégration ``torch.autograd``, contrat v0.4 proposé).

SUM (ci-dessous) : réductions de gradient fusionnées, toutes ordres. Les autres
réductions flottantes se ramènent à ce mécanisme ou au terme retenu (ordre 1) :

- LOGSUMEXP, LOGSUMEXPWEIGHT, SUMSOFTMAXWEIGHT : plan SUM substitut, sorties
  avant tenues fixes (:func:`surrogate`) ;
- MIN, MAX, KMIN et valeurs de MIN_ARGMIN, MAX_ARGMAX, KMIN_ARGKMIN : dérivée au
  couple retenu, évaluée sur O(M·K) couples (:class:`SelectGradient`).

Pour un plan SUM normalisé (réduction sur j) ::

    out[i, e] = Σ_j F_e(x_i, y_j, p)          (x : args « i », y : args « j », p : params)

et un gradient amont ``G[i, e]``, les gradients sont eux-mêmes des réductions
SUM fusionnées, dérivées **symboliquement** (``symox.diff``) et exécutées par
le même pipeline Plan → backend, sans matrice M×N :

- côté i (un plan, réduction sur j) : pour chaque composante ``c`` d'un arg i
  ``∂L/∂x_i[c] = Σ_j Σ_e G[i,e] ∂F_e/∂x[c]`` ; pour un paramètre, la
  contribution par ligne ``Σ_j Σ_e G[i,e] ∂F_e/∂p[c]`` est une sortie de plus du
  même plan, sommée ensuite sur i par Torch (vecteur M × P) ;
- côté j (un plan, rôles i/j échangés, donc réduction sur i) :
  ``∂L/∂y_j[c] = Σ_i Σ_e G[i,e] ∂F_e/∂y[c]``.

``G`` entre comme une variable i supplémentaire de dimension E. Toutes les
dérivées d'un côté sont des sorties d'un **même** plan : la CSE du générateur
(y compris par régions avec ``where``) partage les sous-expressions communes
(``exp(-d²)``, ``d²``, différences ``x - y``...).

Règles de dérivation (après ``symox.diff``) :

- ``where(c, a, b)`` : gradient de la branche active (``Piecewise`` dérivé
  branche par branche, conditions inchangées) ; la condition n'est pas dérivée ;
- ``Heaviside(z)`` (dérivée de ``max``/``min``/``clamp``) devient 1 si z > 0,
  0 si z < 0 et ``1/2`` en z = 0 (convention de symox/SymPy) : à égalité,
  ``max(x, y)`` répartit le gradient moitié-moitié ;
- ``abs`` : ``sign(x)`` (0 en 0) ; ``floor``, ``ceiling``, ``sign`` : 0 ;
- ``DiracDelta`` (parties distributionnelles, p. ex. dérivée seconde de
  ``abs``) : 0 ;
- une dérivée que symox laisse non évaluée (``Derivative``) : ``UnsupportedError``.

Hors des points non lisses, le gradient est exact au sens de la formule
(y compris ``sqrt`` en 0 : ``inf``, comme Torch).
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
    """Remplace les distributions et fonctions de saut issues de ``diff``."""
    import symox

    def heaviside(*args):
        z = args[0]
        h0 = args[1] if len(args) > 1 else symox.Rational(1, 2)
        return symox.Piecewise((1, z > 0), (0, z < 0), (h0, True))

    out = expr.replace(lambda e: isinstance(e, symox.DiracDelta), lambda e: symox.S.Zero)
    out = out.replace(lambda e: isinstance(e, symox.Heaviside), lambda e: heaviside(*e.args))
    if out.has(symox.Derivative):
        bad = sorted({str(d) for d in out.atoms(symox.Derivative)})
        raise UnsupportedError(f"dérivée non évaluable symboliquement : {', '.join(bad)}")
    return out


def _factor_pow2(expr):
    """Remet en facteur un coefficient commun 2^k (k ≥ 1) des termes d'une somme.

    symox distribue les nombres sur les sommes : ``∂(a − b)²/∂a`` devient
    ``2a − 2b``. La différence ``a − b`` n'est alors plus partagée avec la forme
    avant, ce qui coûte deux opérations par composante et par paire dans le
    gradient. Remettre ``2(a − b)`` en facteur est exact (multiplier par 2^k
    commute avec l'arrondi), hors dépassement de capacité ou résultats
    sous-normaux. Les autres coefficients ne sont pas touchés : leur mise en
    facteur changerait l'arrondi.
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
    """``∂expr/∂symbol`` selon les règles du module."""
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
    """Plans de gradient d'un plan SUM.

    ``grad_name`` : nom de l'entrée ``G`` (M, E). ``i_plan`` (ou None) : sorties
    ``i_layout`` = ((nom d'arg, rôle, dim), ...) dans l'ordre des colonnes ;
    les args de rôle ``param`` donnent des contributions par ligne i à sommer.
    ``j_plan`` (ou None) : réduction sur i (rôles échangés), sorties
    ``j_layout`` pour les args j.
    """

    grad_name: str
    i_plan: Plan | None
    i_layout: tuple
    j_plan: Plan | None
    j_layout: tuple


def gradient_plans(plan: Plan, wrt) -> GradientPlans:
    """Construit les plans de gradient de ``plan`` (SUM) par rapport aux args ``wrt``."""
    import symox

    if plan.reduction.kind is not ReductionKind.SUM:
        raise UnsupportedError(f"{plan.reduction.kind.value} n'est pas différentiable (seule SUM l'est)")
    wrt = set(wrt)
    gname = _fresh_name(plan, "kp_grad")
    E = len(plan.outputs)
    gsyms = tuple(component_symbol(gname, e) for e in range(E))

    def side(roles, targets):
        """Plan d'une passe : ``roles`` donne le nouveau rôle de chaque rôle d'origine."""
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
                      fp_contract=plan.fp_contract, sum_scheme=plan.sum_scheme, meta=meta, fast_math=plan.fast_math)
        return p, tuple(layout)

    i_targets = [a for a in plan.args if a.name in wrt and a.role in ("i", "param")]
    j_targets = [a for a in plan.args if a.name in wrt and a.role == "j"]
    i_plan, i_layout = side({"i": "i", "j": "j", "param": "param"}, i_targets)
    j_plan, j_layout = side({"i": "j", "j": "i", "param": "param"}, j_targets)
    return GradientPlans(gname, i_plan, i_layout, j_plan, j_layout)


# --------------------------------------------------------------------------
# LOGSUMEXP, LOGSUMEXPWEIGHT, SUMSOFTMAXWEIGHT : plan SUM substitut
# --------------------------------------------------------------------------
SURROGATE_KINDS = (ReductionKind.LOGSUMEXP, ReductionKind.LOGSUMEXPWEIGHT, ReductionKind.SUMSOFTMAXWEIGHT)


@dataclass
class Surrogate:
    """Plan SUM dont le gradient, sorties de la passe avant tenues fixes, est
    celui de ``plan`` ; ``fixed`` : ((nom, dim, source), ...), source ``"out"``
    (sortie avant) ou ``"lse"`` (log Σ_j exp f, réduction LOGSUMEXP ``lse_plan``)."""

    plan: Plan
    fixed: tuple
    lse_plan: Plan | None


def surrogate(plan: Plan) -> Surrogate:
    """Gradients des réductions de type softmax (sortie avant ``L``, ``O`` fixe) :

    - LOGSUMEXP ``L_e = log Σ_j exp f_e`` : ``∂L_e = Σ_j exp(f_e − L_e) ∂f_e``,
      gradient de ``Σ_j exp(f_e − L_e)`` à ``L`` fixé ;
    - LOGSUMEXPWEIGHT ``L_e = log Σ_j exp(f) g_e`` : gradient de
      ``Σ_j exp(f − L_e) g_e`` ;
    - SUMSOFTMAXWEIGHT ``O_e = Σ_j w_j g_e``, ``w = exp(f − S)``, ``S = log Σ exp f`` :
      ``∂O_e = Σ_j w (∂g_e + (g_e − O_e) ∂f)``, gradient de ``Σ_j exp(f − S)(g_e − O_e)``.

    Une ligne sans contribution (``L = −∞``) a un gradient nul (et non NaN).
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
        raise UnsupportedError(f"{kind.value} : pas de plan substitut")
    args = list(plan.args) + [Arg(name, "i", len(syms), syms) for name, syms, _ in extra]
    args.sort(key=lambda a: (_ROLE_ORDER[a.role], a.name))
    p = make_plan(new, ReductionKind.SUM, plan.dtype, args=args, acc_dtype=plan.acc_dtype,
                  fp_contract=plan.fp_contract, sum_scheme=plan.sum_scheme, fast_math=plan.fast_math,
                  meta={"surrogate_of": plan.key()})
    p = dataclasses.replace(p, batch_rank=plan.batch_rank, ranges=plan.ranges)
    return Surrogate(p, tuple((name, len(syms), src) for name, syms, src in extra), lse_plan)


# --------------------------------------------------------------------------
# MIN, MAX, KMIN (et valeurs de MIN_ARGMIN, MAX_ARGMAX, KMIN_ARGKMIN) : indice retenu
# --------------------------------------------------------------------------
SELECT_KINDS = {
    ReductionKind.MIN: ReductionKind.ARGMIN, ReductionKind.MAX: ReductionKind.ARGMAX,
    ReductionKind.KMIN: ReductionKind.ARGKMIN, ReductionKind.MIN_ARGMIN: None,
    ReductionKind.MAX_ARGMAX: None, ReductionKind.KMIN_ARGKMIN: None,
}


def index_plan(plan: Plan) -> Plan:
    """Plan des indices retenus par une réduction MIN/MAX/KMIN (passe avant du gradient)."""
    kind = SELECT_KINDS[plan.reduction.kind]
    return dataclasses.replace(plan, reduction=PlanReduction(kind, plan.reduction.k))


class SelectGradient:
    """Gradient d'une réduction par sélection : seul le terme retenu dépend des entrées.

    ``v[i, c] = F_e(x_i, y_{j*(i, c)}, p)`` (``e = c`` pour MIN/MAX, ``e = 0``
    et ``c = k`` pour KMIN) : ``∂L/∂x_i = Σ_c G[i, c] ∂F_e/∂x`` au couple
    retenu, ``∂L/∂y_j = Σ_{(i, c) : j*(i, c) = j} G[i, c] ∂F_e/∂y``. Les dérivées
    symboliques sont évaluées par Torch **sur les seuls couples retenus**
    (O(M·C), jamais M×N) ; un indice −1 (aucun candidat) ne contribue pas.
    À égalité, le gradient va à l'indice retenu (le plus petit j), comme PyKeOps.
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
        """Passe avant : réduction SUM ; passe arrière : réductions de gradient fusionnées.

        La passe arrière appelle des :class:`~kernox.runtime.Reduction`, elles-mêmes
        différentiables : avec ``create_graph=True``, les dérivées d'ordre 2
        passent par le même mécanisme.
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
        """Réductions sans gradient défini (sorties entières : COUNT, ARG*) : refus explicite."""

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
                f"{ctx.kind} n'est pas différentiable dans Kernox : "
                "détacher les entrées ou reformuler la perte")

    class KpSurrogate(torch.autograd.Function):
        """LOGSUMEXP, LOGSUMEXPWEIGHT, SUMSOFTMAXWEIGHT : gradient par le plan SUM
        substitut (:func:`surrogate`), réductions fusionnées. Ordre 1 seulement."""

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
        """MIN, MAX, KMIN (et valeurs de MIN_ARGMIN, MAX_ARGMAX, KMIN_ARGKMIN) :
        gradient au terme retenu (:class:`SelectGradient`). Ordre 1 seulement ;
        les indices ne sont pas différentiables."""

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
    """``(KpSum, KpNonDifferentiable)`` (compatibilité)."""
    return _all()[:2]


def _all():
    global _FUNCTIONS
    if _FUNCTIONS is None:
        _FUNCTIONS = _make_function()
    return _FUNCTIONS


def function_for(kind: ReductionKind):
    """Fonction ``torch.autograd`` de la réduction ``kind``."""
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
