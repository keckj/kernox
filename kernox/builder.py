"""Constructeur de :class:`kernox.plan.Plan` (noyau, utilisable sans frontend).

Ce module est la seule voie recommandée pour fabriquer un ``Plan`` : il valide
le plan (noms, rôles, symboles liés, types booléens, K, dtypes) et applique les
règles de conversion des constantes vers ``dtype`` (cf. ``docs/api.md``,
section « Constantes »).

Exemple minimal (backends, tests) ::

    from kernox.builder import make_arg, make_plan
    from kernox.plan import ReductionKind
    import symox

    x, y = make_arg("x", "i", 3), make_arg("y", "j", 3)
    d2 = sum((a - b) ** 2 for a, b in zip(x.symbols, y.symbols))
    plan = make_plan([symox.exp(-d2)], ReductionKind.SUM, "float32", args=[x, y])
"""

from __future__ import annotations

import keyword
import math
from fractions import Fraction

from .errors import ConstantConversionError, UnsupportedError, ValidationError
from .plan import SUM_SCHEMES, Arg, DType, Plan, Reduction, ReductionKind

_DTYPES = ("float16", "float32", "float64")
_RANK = {"float16": 0, "float32": 1, "float64": 2}
#: accumulateurs permis par dtype (comme PyKeOps : FP16 -> FP16 ou FP32, FP32 -> FP32 ou FP64)
_ACC_ALLOWED = {"float16": ("float16", "float32"), "float32": ("float32", "float64"), "float64": ("float64",)}
_ROLES = ("i", "j", "param")
_K_KINDS = (ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN)
_ACC_KINDS = (ReductionKind.SUM, ReductionKind.LOGSUMEXP, ReductionKind.SUMSOFTMAXWEIGHT,
              ReductionKind.LOGSUMEXPWEIGHT)
# Plus grande valeur finie et plus petit sous-normal positif par dtype.
_FMAX = {"float16": 65504.0, "float32": float.fromhex("0x1.fffffep+127"), "float64": float.fromhex("0x1.fffffffffffffp+1023")}


def component_symbol(name: str, index: int):
    """Symbole symox scalaire réel de la composante ``index`` de l'argument ``name``."""
    import symox

    return symox.Symbol(f"{name}[{index}]", real=True)


def check_name(name) -> str:
    if not isinstance(name, str) or not name.isidentifier() or not name.isascii():
        raise ValidationError(f"nom d'argument invalide {name!r} : identifiant ASCII requis")
    if keyword.iskeyword(name) or name.startswith(("_kp", "_sx")):
        raise ValidationError(f"nom d'argument réservé : {name!r}")
    return name


def make_arg(name: str, role: str, dim: int = 1) -> Arg:
    """Fabrique un :class:`Arg` et ses symboles de composantes."""
    check_name(name)
    if role not in _ROLES:
        raise ValidationError(f"rôle invalide {role!r} (attendu : {_ROLES})")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValidationError(f"dimension invalide {dim!r} pour {name!r} : entier >= 1 requis")
    return Arg(name, role, dim, tuple(component_symbol(name, c) for c in range(dim)))


def is_boolean(expr) -> bool:
    """Vrai si l'expression symox est booléenne (même règle que les printers symox)."""
    from symox import _core

    return _core._sycl_type(expr) == "bool"


def check_k(k) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValidationError(f"K doit être un entier Python, reçu {k!r}")
    if k < 0:
        raise ValidationError(f"K doit être >= 0, reçu {k}")
    return k


# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------


def _round_to(value: float, dtype: DType) -> float:
    if dtype == "float64":
        return value
    import numpy as np

    with np.errstate(over="ignore", under="ignore"):
        return float(np.float16(value) if dtype == "float16" else np.float32(value))


def convert_float(value: float, dtype: DType, what: str = "constante") -> float:
    """Arrondi au plus proche (pair) d'un ``float`` Python vers ``dtype``.

    Erreur si le résultat déborde (|x| fini > max du dtype) ou si une valeur
    non nulle s'arrondit à zéro. ``inf``/``nan`` explicites sont conservés.
    """
    if math.isnan(value) or math.isinf(value):
        return value
    rounded = _round_to(value, dtype)
    if math.isinf(rounded):
        raise ConstantConversionError(f"{what} {value!r} non représentable en {dtype} (dépassement)")
    if rounded == 0.0 and value != 0.0:
        raise ConstantConversionError(f"{what} {value!r} non représentable en {dtype} (s'arrondit à 0)")
    return rounded


def _exact_to_float(fr: Fraction, what) -> float:
    try:
        return float(fr)  # arrondi correct en binary64
    except OverflowError:
        raise ConstantConversionError(f"{what} non représentable en float64 (dépassement)") from None


def convert_constants(expr, dtype: DType):
    """Applique la politique des constantes du plan à une expression symox.

    - ``Float`` : arrondi une fois vers ``dtype`` (au plus proche, pair), le
      ``Float`` résultant (précision 53 bits) porte exactement la valeur du
      dtype ; le printer l'émet avec assez de chiffres pour l'identifier.
      Un ``Float`` de précision > 53 bits est d'abord arrondi en binary64.
    - ``Rational``/``Integer`` exacts : conservés exacts (les printers émettent
      ``p/q`` dans le dtype, ce qui donne l'arrondi correct de p/q quand p et q
      sont représentables) ; seule leur représentabilité est vérifiée.
    - Dépassement ou underflow d'une valeur non nulle vers 0 :
      :class:`ConstantConversionError`.
    """
    import symox

    if dtype not in _DTYPES:
        raise ValidationError(f"dtype invalide {dtype!r}")
    replacements = {}
    for atom in expr.atoms(symox.Number):
        if atom.is_Float:
            value = float(atom)
            what = f"constante {atom}"
            new = convert_float(value, dtype, what)
            if new != value or getattr(atom, "_prec", 53) != 53:
                replacements[atom] = symox.Float(new)
        elif atom.is_Rational:
            p, q = int(atom.p), int(atom.q)
            what = f"constante {atom}"
            for part in {p, q} - {0, 1, -1}:
                convert_float(_exact_to_float(Fraction(part), what), dtype, what)
            convert_float(_exact_to_float(Fraction(p, q), what), dtype, what)
        elif atom.is_finite is False or atom is symox.nan:
            continue
        else:  # nombres non réels (I...) ou inconnus
            raise UnsupportedError(f"constante non réelle ou non prise en charge : {atom}")
    return expr.xreplace(replacements) if replacements else expr


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


def realify(expr):
    """Ramène à l'arithmétique réelle IEEE les artefacts complexes du moteur.

    Toutes les variables Kernox sont réelles, mais le moteur symbolique (comme
    SymPy) distribue les fonctions sur les branches d'un ``Piecewise`` et ne
    sait pas toujours prouver qu'une branche gardée est réelle :

    - ``sqrt(where(c, a, -1.5))`` devient ``Piecewise((sqrt(a), c), (1.22*I, True))``
      et ``1/where(c, x, 0)`` fait apparaître ``zoo`` ; dans les valeurs de
      branche, ``I`` et ``zoo`` sont remplacés par ``nan``, qui est le résultat
      réel IEEE (``sqrt(-1.5)``, ``0/0``) ; une branche ainsi atteinte est hors
      domaine et c'est le ``where`` englobant qui l'écarte ;
    - ``abs(exp(z))`` devient ``exp(re(z))`` : ``re(a)`` vaut ``a`` et ``im(a)``
      vaut 0 pour une expression réelle (sans ``I``).

    Un ``I`` ou ``zoo`` hors de toute branche reste refusé par :func:`make_plan`.
    """
    import symox

    bad = {symox.I: symox.nan, symox.zoo: symox.nan}
    memo = {}  # DAG partagé : chaque sous-expression une seule fois (sinon exponentiel)

    def walk(e):
        if not e.args:
            return e
        hit = memo.get(e)
        if hit is not None:
            return hit
        if isinstance(e, symox.Piecewise):
            out = symox.Piecewise(
                *((walk(v).xreplace(bad), walk(c)) for v, c in e.args)
            )
        else:
            out = e.func(*(walk(a) for a in e.args))
        memo[e] = out
        return out

    if not expr.has(symox.Piecewise, symox.re, symox.im):
        return expr
    out = walk(expr)
    for _ in range(64):
        new = out.replace(lambda n: isinstance(n, symox.re) and not n.args[0].has(symox.I),
                          lambda n: n.args[0])
        new = new.replace(lambda n: isinstance(n, symox.im) and not n.args[0].has(symox.I),
                          lambda n: symox.S.Zero)
        if new == out:
            break
        out = walk(new)
    return out


def fold_constants(expr):
    """Évalue numériquement les sous-expressions constantes non atomiques.

    ``cos(1)``, ``sqrt(2)``, ``log(3)`` restent symboliques dans le moteur et
    seraient imprimés tels quels ; ``cos(1)`` en double est ambigu pour NVRTC
    (surcharges ``cos(float)``/``cos(double)`` pour un argument entier). La
    valeur est calculée en précision étendue puis arrondie au dtype par
    :func:`convert_constants`.
    """
    import symox

    memo = {}  # DAG partagé : chaque sous-expression une seule fois (sinon exponentiel)

    def walk(e):
        if not e.args:
            return e
        hit = memo.get(e)
        if hit is not None:
            return hit
        boolean = isinstance(e, symox.logic.boolalg.Boolean)
        if not boolean and getattr(e, "is_number", False) and not e.free_symbols:
            v = e.evalf(30)
            if v.is_Float or v.is_Rational:
                out = symox.Float(v, 30) if v.is_Float else v
            else:
                out = e
        else:
            out = e.func(*(walk(a) for a in e.args))
        memo[e] = out
        return out

    return walk(expr)


def make_plan(
    outputs,
    reduction,
    dtype: DType,
    *,
    args,
    k: int | None = None,
    acc_dtype: DType | None = None,
    fp_contract: bool = True,
    sum_scheme: str | None = "direct",
    meta: dict | None = None,
    fast_math: bool = False,
) -> Plan:
    """Construit et valide un :class:`Plan`.

    ``outputs`` : séquence d'expressions symox scalaires (ou Python/booléens).
    ``reduction`` : :class:`ReductionKind`, sa valeur (``"sum"``...) ou
    :class:`Reduction`. ``args`` : séquence d':class:`Arg`, conservée dans
    l'ordre donné (le frontend fournit l'ordre canonique i, j, param puis nom).
    ``fp_contract`` : contraction FMA intra-expression permise (contrat v0.3).
    ``fast_math`` (avenant v0.10) : exp/log approchés permis (opt-in, défaut IEEE).
    ``sum_scheme`` : ``"direct"``, ``"block"`` ou ``"kahan"`` (avenant v0.7) ; ``None`` :
    défaut, ``"block"`` pour FP16 accumulé en FP16 (comme PyKeOps), ``"direct"`` sinon.
    """
    import symox

    if not isinstance(fast_math, bool):
        raise ValidationError(f"fast_math doit être un booléen, reçu {fast_math!r}")
    if not isinstance(fp_contract, bool):
        raise ValidationError(f"fp_contract doit être un booléen, reçu {fp_contract!r}")
    if sum_scheme is not None and sum_scheme not in SUM_SCHEMES:
        raise ValidationError(f"sum_scheme invalide {sum_scheme!r} (attendu : {SUM_SCHEMES})")
    if dtype not in _DTYPES:
        raise ValidationError(f"dtype invalide {dtype!r} (attendu : {_DTYPES})")

    if isinstance(reduction, Reduction):
        if k is not None and k != reduction.k:
            raise ValidationError("K donné deux fois avec des valeurs différentes")
        kind, k = reduction.kind, reduction.k
    else:
        try:
            kind = ReductionKind(reduction)
        except ValueError:
            raise ValidationError(f"réduction inconnue {reduction!r}") from None
    if kind in _K_KINDS:
        if k is None:
            raise ValidationError(f"K requis pour {kind.value}")
        k = check_k(k)
    elif k is not None:
        raise ValidationError(f"K n'a pas de sens pour {kind.value}")
    if acc_dtype is None:
        # avenant v0.9 : FP16 accumule en FP32 par défaut (SUM et familles LOGSUMEXP)
        acc_dtype = "float32" if dtype == "float16" and kind in _ACC_KINDS else dtype
    if acc_dtype not in _DTYPES:
        raise ValidationError(f"acc_dtype invalide {acc_dtype!r}")
    if _RANK[acc_dtype] < _RANK[dtype]:
        raise ValidationError(f"acc_dtype {acc_dtype} inférieur à dtype {dtype}")
    if acc_dtype not in _ACC_ALLOWED[dtype]:
        raise ValidationError(f"acc_dtype {acc_dtype} non permis pour dtype {dtype} "
                              f"(attendu : {_ACC_ALLOWED[dtype]}, comme PyKeOps)")
    if sum_scheme is None:
        # avenant v0.9 (c) : acc_dtype="float16" reproduit PyKeOps (half, block_sum)
        sum_scheme = "block" if acc_dtype == "float16" and kind in _ACC_KINDS else "direct"
    if acc_dtype != dtype and kind not in _ACC_KINDS:
        raise ValidationError("acc_dtype distinct de dtype n'est défini que pour SUM, LOGSUMEXP, "
                              "SUMSOFTMAXWEIGHT et LOGSUMEXPWEIGHT")

    args = tuple(args)
    seen_names, seen_symbols = set(), set()
    for a in args:
        if not isinstance(a, Arg):
            raise ValidationError(f"argument {a!r} n'est pas un kernox.plan.Arg")
        check_name(a.name)
        if a.role not in _ROLES:
            raise ValidationError(f"rôle invalide {a.role!r} pour {a.name!r}")
        if isinstance(a.dim, bool) or not isinstance(a.dim, int) or a.dim < 1 or len(a.symbols) != a.dim:
            raise ValidationError(f"dimension/symboles incohérents pour {a.name!r}")
        if a.name in seen_names:
            raise ValidationError(f"nom d'argument dupliqué : {a.name!r}")
        seen_names.add(a.name)
        for s in a.symbols:
            if not isinstance(s, symox.Symbol) or s in seen_symbols:
                raise ValidationError(f"symbole invalide ou dupliqué {s!r} dans {a.name!r}")
            seen_symbols.add(s)
    if not any(a.role == "i" for a in args) or not any(a.role == "j" for a in args):
        raise UnsupportedError(
            "v0 : la formule doit dépendre d'au moins une variable i et une variable j "
            "(sinon M ou N ne peut être déduit des entrées)"
        )

    exprs = tuple(symox.sympify(e) for e in outputs)
    if not exprs:
        raise ValidationError("au moins une sortie est requise")
    exprs = tuple(realify(e) for e in exprs)
    for e in exprs:
        unbound = e.free_symbols - seen_symbols
        if unbound:
            raise ValidationError("symboles non déclarés : " + ", ".join(sorted(map(str, unbound))))
        if e.has(symox.I) or e.has(symox.zoo):
            raise UnsupportedError(f"expression non réelle : {e}")
    booleans = [is_boolean(e) for e in exprs]
    if kind is ReductionKind.COUNT:
        if not all(booleans):
            raise ValidationError("COUNT exige des sorties booléennes")
    elif any(booleans):
        raise ValidationError(
            f"{kind.value} exige des sorties réelles ; utiliser where(...) ou sum_over sur un "
            "prédicat (COUNT)"
        )
    if kind in (ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT) and len(exprs) < 2:
        raise ValidationError(f"{kind.name} : sorties (f, g_1, ..., g_E) avec E >= 1")
    if kind in _K_KINDS and len(exprs) != 1:
        raise UnsupportedError(f"{kind.value} : formule scalaire uniquement en v0 (out_dim={len(exprs)})")

    exprs = tuple(convert_constants(fold_constants(e), dtype) for e in exprs)
    return Plan(
        args=args,
        outputs=exprs,
        reduction=Reduction(kind, k),
        dtype=dtype,
        acc_dtype=acc_dtype,
        fp_contract=fp_contract,
        sum_scheme=sum_scheme,
        fast_math=fast_math,
        meta=dict(meta or {}),
    )
