"""Fonction de paire générée par symox, par dialecte (contrat v0).

``pair_function(plan, dialect)`` produit une fonction device scalaire,
indépendante de la stratégie de kernel ::

    // cuda/hip : __device__ __forceinline__ ; sycl : inline
    struct kp_out { T o0; ...; };                    // bool pour COUNT
    kp_out kp_pair(const T* __restrict__ xi_0, ...,   // un pointeur par arg i
                   const T* __restrict__ yj_0, ...,   // un pointeur par arg j
                   const T* __restrict__ p_0, ...);   // un pointeur par param

Les pointeurs suivent l'ordre de ``plan.args`` ; leurs noms sont ``xi_<n>``,
``yj_<n>``, ``p_<n>`` où ``<n>`` est le rang de l'argument parmi ceux de même
rôle. ``PairSource.arg_layout`` donne la correspondance exacte.

Politique numérique (cf. ``docs/api.md``) :

- jamais de fast-math (``plan.fast_math=True`` → ``UnsupportedError``) ;
- ``1/sqrt(x)`` n'est PAS remplacé par ``rsqrt`` (précision réduite en CUDA) ;
- aucune conversion FP64 → FP32 : tous les temporaires sont de type ``T`` ;
- CSE par régions en présence de ``Piecewise`` (``where``), cf.
  :func:`_region_cse` : une sous-expression commune est hissée (évaluée
  inconditionnellement) seulement si elle est déjà évaluée hors de toute
  branche (condition du ``where`` le plus externe, partie non conditionnelle)
  ou si elle apparaît dans au moins deux régions distinctes ; sinon elle reste
  dans sa branche. C'est sûr car le code généré n'a aucun effet de bord et
  les GPU visés n'ont pas de trap flottant : une valeur calculée hors de son
  domaine (``sqrt(-1)``, ``1/0``, ``log(0)``) donne NaN/inf sans interrompre le
  kernel, et elle n'est lue que par les branches qui la consomment ; le
  ternaire C retient la valeur de la branche active, la même qu'une
  évaluation non hissée (au choix de contraction FMA près, déjà permis par
  ``fp_contract``). Les fonctions utilisateur (coût ou effets inconnus) ne
  sont jamais hissées.

Dépendances de la source : CUDA/NVRTC aucune ; HIP : ``hip/hip_runtime.h``
(inclus automatiquement par hipcc/hiprtc) ; SYCL : ``<sycl/sycl.hpp>`` doit
être inclus par le backend avant la source (fonctions ``sycl::``).

« À proposer à symox » : symox expose déjà ``cuda_device_function`` (fonction
``__device__`` à arguments scalaires) et ``syclcode_function`` (fonction
``inline`` à arguments scalaires). Il manque, de façon générale et réutilisable :
(1) un dialecte HIP explicite, (2) des arguments vectoriels par pointeur
``const T* __restrict__`` avec chargement des seules composantes utilisées,
(3) une struct de sorties nommée par l'appelant et typée bool/T, (4) le
contrôle ``use_rsqrt=False`` exposé dans les fonctions device, (5) une
politique d'arrondi des constantes ``Float`` vers le dtype. Ce module
implémente ces points côté Kernox en réutilisant les printers et helpers symox.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass

from .. import _symox_compat
from ..builder import is_boolean
from ..errors import UnsupportedError, ValidationError
from ..plan import Plan

GENERATOR_VERSION = "kernox-pair-v2"
DIALECTS = ("cuda", "hip", "sycl")
_PREFIX = {"i": "xi", "j": "yj", "param": "p"}
_CTYPE = {"float32": "float", "float64": "double"}


@dataclass(frozen=True)
class ArgLayout:
    """Paramètre C ``param`` ↔ argument ``arg`` du plan (rôle, dim)."""

    param: str
    arg: str
    role: str
    dim: int


@dataclass(frozen=True)
class PairSource:
    """Source générée ; champs ``source``, ``name``, ``out_type``, ``arg_layout``
    fixés par le contrat v0, les autres sont des ajouts rétrocompatibles."""

    source: str
    name: str
    out_type: str
    arg_layout: tuple[ArgLayout, ...]
    dialect: str = "cuda"
    scalar_type: str = "float"                 # T
    out_fields: tuple[str, ...] = ()           # "o0", "o1"...
    out_ctypes: tuple[str, ...] = ()           # "float" | "double" | "bool"
    generator_version: str = GENERATOR_VERSION

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()


_POW_HELPERS = {f"kp_pow{n}": f"kp_pow{n}" for n in (2, 3, 4)}

# --------------------------------------------------------------------- FP16 (avenant v0.9)
# Type FP16 du dialecte hip : ``kp_h`` enveloppe ``_Float16`` pour que chaque opération
# soit arrondie en binary16 (sémantique de l'oracle Torch). Avec ``_Float16`` nu, un
# littéral ``1.5F`` ou un appel ``expf`` promouvrait le calcul en FP32. Conversions
# vers kp_h implicites (constantes, entiers ; arrondi au plus proche), conversions
# depuis kp_h explicites : aucun opérateur intégré float n'est candidat. Fonctions
# mathématiques : calcul en float puis arrondi en FP16 (comme Torch sur CPU).
# « À proposer à symox » : un dtype float16 pour ``cudacode``/``cudacode_function``.
HALF_UNARY = ("expf", "logf", "sqrtf", "fabsf", "sinf", "cosf", "tanf", "asinf", "acosf", "atanf",
              "sinhf", "coshf", "tanhf", "floorf", "ceilf", "expm1f", "log1pf", "exp2f", "log2f",
              "log10f", "erff", "erfcf", "cbrtf", "roundf", "truncf", "rintf")
HALF_BINARY = ("powf", "atan2f", "fminf", "fmaxf", "copysignf", "fmodf", "hypotf")
HALF_PRELUDE = ("""#ifndef KP_HALF_DEFINED
#define KP_HALF_DEFINED
struct kp_h {
  _Float16 v;
  kp_h() = default;
  __device__ __forceinline__ kp_h(float x) : v((_Float16)x) {}
  __device__ __forceinline__ kp_h(double x) : v((_Float16)x) {}
  __device__ __forceinline__ kp_h(int x) : v((_Float16)x) {}
  __device__ __forceinline__ kp_h(long long x) : v((_Float16)x) {}
  __device__ __forceinline__ explicit operator float() const { return (float)v; }
  __device__ __forceinline__ explicit operator double() const { return (double)v; }
};
__device__ __forceinline__ kp_h kp_hraw(_Float16 x) { kp_h r; r.v = x; return r; }
#define KP_HBIN(op) __device__ __forceinline__ kp_h operator op(kp_h a, kp_h b) { return kp_hraw(a.v op b.v); }
KP_HBIN(+) KP_HBIN(-) KP_HBIN(*) KP_HBIN(/)
#undef KP_HBIN
#define KP_HCMP(op) __device__ __forceinline__ bool operator op(kp_h a, kp_h b) { return a.v op b.v; }
KP_HCMP(<) KP_HCMP(>) KP_HCMP(<=) KP_HCMP(>=) KP_HCMP(==) KP_HCMP(!=)
#undef KP_HCMP
#define KP_HSET(op) __device__ __forceinline__ kp_h& operator op##=(kp_h& a, kp_h b) { a.v = a.v op b.v; return a; }
KP_HSET(+) KP_HSET(-) KP_HSET(*) KP_HSET(/)
#undef KP_HSET
__device__ __forceinline__ kp_h operator-(kp_h a) { return kp_hraw(-a.v); }
__device__ __forceinline__ kp_h operator+(kp_h a) { return a; }
"""
    + "".join(f"__device__ __forceinline__ kp_h {f}(kp_h x) {{ return kp_h({f}((float)x)); }}\n" for f in HALF_UNARY)
    + "".join(f"__device__ __forceinline__ kp_h {f}(kp_h x, kp_h y) {{ return kp_h({f}((float)x, (float)y)); }}\n"
              for f in HALF_BINARY)
    + "#endif\n")
_HALF_CALLS_OK = set(HALF_UNARY + HALF_BINARY) | {"kp_pow2", "kp_pow3", "kp_pow4", "kp_h", "__int_as_float",
                                                  "if", "return", "sizeof", "defined"}


def _half_source(ps: "PairSource", plan: Plan) -> "PairSource":
    """Source FP32 → FP16 (``kp_h``) ; refus explicite si une fonction n'a pas de surcharge."""
    lines = ps.source.split("\n")
    lines[0] = re.sub(r"plan \S+", f"plan {plan.key()}", lines[0]) + " [float16 : kp_h]"
    body = re.sub(r"\bfloat\b", "kp_h", "\n".join(lines))
    code = re.sub(r"(?m)//[^\n]*|^\s*#[^\n]*", "", body)   # commentaires, préprocesseur
    for fn in set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", code)):
        if fn not in _HALF_CALLS_OK and fn not in (ps.name, "kp_out", ps.out_type) and not fn.startswith("_kp"):
            raise UnsupportedError(f"dtype float16 : fonction {fn} sans surcharge FP16 (backend hip)")
    return dataclasses.replace(
        ps, source=HALF_PRELUDE + body, scalar_type="kp_h",
        out_ctypes=tuple("kp_h" if c == "float" else c for c in ps.out_ctypes))


def _emit_factory(dialect: str, dtype: str):
    if dialect in ("cuda", "hip"):
        from symox.extra import cudacode

        ctype = _CTYPE[dtype]

        def base(expr):
            # math_macros=False : constantes (pi, e...) en littéraux, M_PI absent sous NVRTC
            return cudacode(expr, dtype=ctype, fast_math=False, use_rsqrt=False,
                            math_macros=False, user_functions=_POW_HELPERS)

        # symox récent : opérateurs logiques natifs ; commit épinglé : contournement
        return base if _symox_compat.cuda_logic() else _with_boolean_operators(base)
    from symox import _core

    def emit(expr):
        return _core._syclcode(expr, dtype=dtype, user_functions=_POW_HELPERS)

    return emit


_LOGIC = {"And": "&&", "Or": "||", "Xor": "!="}


def _with_boolean_operators(base):
    """Étend un printer sans ``And/Or/Xor/Not`` (``cudacode`` de symox à 2e623e6).

    Chaque sous-arbre logique maximal est remplacé par un symbole substitut,
    le reste est imprimé par ``base``, puis les substituts sont remplacés par
    ``(a) && (b)``, ``(a) || (b)``, ``(a) != (b)`` (xor booléen), ``!(a)``.
    Les opérandes sont des booléens purs : l'absence de court-circuit
    n'a pas d'effet observable. « À proposer à symox » : support des
    opérateurs logiques dans ``cudacode``.
    """
    import re

    import symox

    def emit(expr):
        name = type(expr).__name__
        if name in _LOGIC:
            parts = [f"({emit(a)})" for a in expr.args]
            text = parts[0]
            for part in parts[1:]:
                text = f"({text} {_LOGIC[name]} {part})"
            return text
        if name == "Not":
            return f"(!({emit(expr.args[0])}))"
        mapping = {}

        def collect(node):
            if type(node).__name__ in _LOGIC or type(node).__name__ == "Not":
                mapping.setdefault(node, symox.Symbol(f"_kp_ph{len(mapping)}_"))
                return
            for arg in node.args:
                collect(arg)

        collect(expr)
        if not mapping:
            return base(expr)
        text = base(expr.xreplace(mapping))
        for node, ph in mapping.items():
            code = "(" + emit(node) + ")"
            text = re.sub(rf"\b{ph.name}\b", lambda _m, c=code: c, text)
        return text

    return emit


def _rewrite_integer_powers(expr):
    """``b**n`` (n entier, 2 <= |n| <= 4) → ``kp_pow{|n|}(b)`` (produits explicites).

    Évite ``powf(a - b, 2)`` émis par le printer CUDA pour une base composée.
    Les produits ``v*v`` sont correctement arrondis ; ``1/kp_pow{n}`` pour n < 0.
    « À proposer à symox » : inlining des petites puissances entières pour
    toute base dans ``cudacode``.
    """
    import symox

    helpers = {n: symox.Function(f"kp_pow{n}") for n in (2, 3, 4)}

    def query(e):
        return e.is_Pow and e.exp.is_Integer and 2 <= abs(int(e.exp)) <= 4

    def value(e):
        n = int(e.exp)
        r = helpers[abs(n)](e.base)
        return 1 / r if n < 0 else r

    return expr.replace(query, value)


_HELPERS_TEMPLATE = """#ifndef KP_PAIR_HELPERS
#define KP_PAIR_HELPERS
{q} float kp_pow2(float v) {{ return v * v; }}
{q} float kp_pow3(float v) {{ return v * v * v; }}
{q} float kp_pow4(float v) {{ const float t = v * v; return t * t; }}
{q} double kp_pow2(double v) {{ return v * v; }}
{q} double kp_pow3(double v) {{ return v * v * v; }}
{q} double kp_pow4(double v) {{ const double t = v * v; return t * t; }}
#endif
"""

# Surcharges binary16 (sycl::half op sycl::half reste en half : chaque produit
# est arrondi en FP16, comme l'oracle).
_HALF_HELPERS = """#ifndef KP_PAIR_HELPERS_HALF
#define KP_PAIR_HELPERS_HALF
inline sycl::half kp_pow2(sycl::half v) { return v * v; }
inline sycl::half kp_pow3(sycl::half v) { return v * v * v; }
inline sycl::half kp_pow4(sycl::half v) { const sycl::half t = v * v; return t * t; }
#endif
"""

_CUDA_PREAMBLE = """#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
"""


def _expand_ite(expr):
    """Remplace ``ITE(c, a, b)`` (booléen) par ``(c & a) | (~c & b)``.

    symox crée des ``ITE`` en simplifiant un ``where`` placé dans une condition
    (``where(where(c, u, v) > 0, ...)``) ; ni ``cudacode`` ni ``_syclcode`` ne les
    impriment. ``a`` et ``b`` sont des booléens purs (comparaisons, logique) :
    les évaluer tous deux n'a pas d'effet observable, une comparaison avec NaN
    étant simplement fausse. « À proposer à symox » : imprimer ``ITE`` en ternaire.
    """
    from symox.logic.boolalg import ITE

    import symox

    for _ in range(64):
        found = [a for a in expr.atoms(ITE)]
        if not found:
            return expr
        expr = expr.xreplace({
            a: symox.Or(symox.And(a.args[0], a.args[1]), symox.And(symox.Not(a.args[0]), a.args[2]))
            for a in found
        })
    raise UnsupportedError("ITE imbriqués trop profondément")


def _regions(exprs, temps):
    """Ensemble des régions où apparaît chaque symbole de ``temps``.

    La région racine ``()`` est évaluée inconditionnellement (hors ``Piecewise``,
    première condition d'un ``Piecewise`` non conditionnel). La branche ``k``
    d'un ``Piecewise`` et ses conditions suivantes sont des régions filles.
    """
    import symox

    found = {t: set() for t in temps}
    counter = [0]

    def walk(e, region):
        if e in found:
            found[e].add(region)
            return
        if isinstance(e, symox.Piecewise):
            counter[0] += 1
            pid = counter[0]
            for k, (value, cond) in enumerate(e.args):
                walk(cond, region if k == 0 else region + ((pid, "c", k),))
                walk(value, region + ((pid, k),))
            return
        for a in e.args:
            walk(a, region)

    for e in exprs:
        walk(e, ())
    return found


def _region_cse(exprs, prefix="_kp_t"):
    """CSE valide en présence de ``where`` (voir la politique du module).

    On part de la CSE complète de symox, puis chaque temporaire, du dernier au
    premier, est conservé (hissé, évalué inconditionnellement) s'il apparaît
    dans la région racine (y compris dans la valeur d'un temporaire déjà
    hissé) ou dans au moins deux régions distinctes ; sinon il est réinjecté
    dans son unique branche. Les appels de fonctions non définies autres que
    les aides ``kp_pow*`` (fonctions utilisateur, coût ou effets inconnus) ne
    sont jamais hissés.
    """
    import symox
    from symox.extra.codegen.pytorch import _cse_preserving_assumptions

    replacements, exprs = _cse_preserving_assumptions(exprs, prefix)
    kept = []  # (symbole, valeur), ordre inverse
    for symbol, value in reversed(replacements):
        values = [v for _, v in kept]
        regions = _regions(list(exprs) + values, {symbol})[symbol]
        in_kept = any(symbol in v.free_symbols for v in values)
        opaque = any(isinstance(f, symox.core.function.AppliedUndef) and not str(f.func).startswith("kp_pow")
                     for f in value.atoms(symox.Function))
        if not opaque and (in_kept or () in regions or len(regions) >= 2):
            kept.append((symbol, value))
        else:
            sub = {symbol: value}
            exprs = [e.xreplace(sub) for e in exprs]
            kept = [(s_, v.xreplace(sub)) for s_, v in kept]
    return list(reversed(kept)), exprs


def pair_function(
    plan: Plan,
    dialect: str,
    *,
    name: str = "kp_pair",
    out_type: str = "kp_out",
    cse: bool = True,
) -> PairSource:
    """Génère la fonction de paire de ``plan`` dans ``dialect`` (cuda|hip|sycl).

    ``name``/``out_type`` peuvent être changés pour inclure plusieurs plans
    dans une même unité de compilation.
    """
    import symox
    from symox.extra.codegen.pytorch import _contains_conditionals, _cse_preserving_assumptions

    if dialect not in DIALECTS:
        raise ValidationError(f"dialecte inconnu {dialect!r} (attendu : {DIALECTS})")
    for ident in (name, out_type):
        if not (isinstance(ident, str) and ident.isidentifier() and ident.isascii()) or ident.startswith("_kp"):
            raise ValidationError(f"identifiant C invalide : {ident!r}")
    if plan.fast_math:
        raise UnsupportedError("fast_math n'est pris en charge par aucun générateur Kernox")
    if plan.dtype == "float16":
        # avenant v0.9 : hip émule half (paire FP32 réécrite en kp_h, arrondi
        # binary16 par opération) ; sycl utilise sycl::half natif (printer half
        # de symox requis). cuda réécrit lui-même sa paire FP32 (backend).
        if dialect == "hip":
            p32 = dataclasses.replace(plan, dtype="float32", acc_dtype="float32", sum_scheme="direct")
            return _half_source(pair_function(p32, dialect, name=name, out_type=out_type, cse=cse), plan)
        if dialect != "sycl":
            raise UnsupportedError(f"dtype float16 : fonction de paire non disponible en dialecte {dialect}")
        if not _symox_compat.sycl_half():
            raise UnsupportedError("dtype float16 : le printer SYCL de symox ne connaît pas sycl::half "
                                   "(symox avec SyclDtype::Float16 requis)")
        T = "sycl::half"
    elif plan.dtype not in _CTYPE:
        raise ValidationError(f"dtype invalide {plan.dtype!r}")
    else:
        T = _CTYPE[plan.dtype]

    # Noms C des paramètres et des composantes (symboles renommés de façon sûre).
    layout, counters, rename, loads = [], {"i": 0, "j": 0, "param": 0}, {}, {}
    for a in plan.args:
        param = f"{_PREFIX[a.role]}_{counters[a.role]}"
        counters[a.role] += 1
        layout.append(ArgLayout(param, a.name, a.role, a.dim))
        for c, s in enumerate(a.symbols):
            local = symox.Symbol(f"_kp_{param}_{c}", **s.assumptions0)
            rename[s] = local
            loads[local] = f"{param}[{c}]"

    qualifier = "__device__ __forceinline__" if dialect in ("cuda", "hip") else "inline"
    fields = tuple(f"o{n}" for n in range(len(plan.outputs)))
    header = f"// Généré par kernox.codegen.pair ({GENERATOR_VERSION}), dialecte {dialect}, plan {plan.key()}"
    out_ctypes = tuple("bool" if is_boolean(e) else T for e in plan.outputs)
    conditional = _contains_conditionals(list(plan.outputs))
    # cudacode_function de symox n'applique aucune CSE en présence de Piecewise :
    # le chemin local porte la CSE par régions.
    if dialect in ("cuda", "hip") and _symox_compat.cudacode_function() and not (cse and conditional):
        source = _via_symox_device_function(plan, dialect, name, out_type, fields, layout, cse, header, qualifier)
        return PairSource(source=source, name=name, out_type=out_type, arg_layout=tuple(layout),
                          dialect=dialect, scalar_type=T, out_fields=fields, out_ctypes=out_ctypes)

    exprs = [e.xreplace(rename) for e in plan.outputs]
    exprs = [_rewrite_integer_powers(_expand_ite(e)) for e in exprs]
    replacements = []
    if cse and exprs:
        if conditional:
            replacements, exprs = _region_cse(exprs, "_kp_t")
        else:
            replacements, exprs = _cse_preserving_assumptions(exprs, "_kp_t")

    emit = _emit_factory(dialect, plan.dtype)
    used = set()
    for _, value in replacements:
        used |= value.free_symbols
    for e in exprs:
        used |= e.free_symbols

    lines = [header]
    if dialect in ("cuda", "hip"):
        lines.append(_CUDA_PREAMBLE.rstrip())
    lines.append(_HELPERS_TEMPLATE.format(q=qualifier).rstrip())
    if T == "sycl::half":
        lines.append(_HALF_HELPERS.rstrip())
    lines.append(f"struct {out_type} {{")
    lines += [f"    {ct} {f};" for ct, f in zip(out_ctypes, fields)]
    lines.append("};")
    params = ", ".join(f"const {T}* __restrict__ {l.param}" for l in layout)
    lines.append(f"{qualifier} {out_type} {name}({params}) {{")
    for l in layout:
        if not any(rename[s] in used for s in next(a for a in plan.args if a.name == l.arg).symbols):
            lines.append(f"    (void){l.param};")
    for local, load in loads.items():
        if local in used:
            lines.append(f"    const {T} {local.name} = {load};")
    for symbol, value in replacements:
        ctype = "bool" if is_boolean(value) else T
        lines.append(f"    const {ctype} {symbol.name} = {emit(value)};")
    lines.append(f"    {out_type} _kp_r;")
    for f, ct, e in zip(fields, out_ctypes, exprs):
        lines.append(f"    _kp_r.{f} = ({ct})({emit(e)});")
    lines.append("    return _kp_r;")
    lines.append("}")
    return PairSource(
        source="\n".join(lines) + "\n",
        name=name,
        out_type=out_type,
        arg_layout=tuple(layout),
        dialect=dialect,
        scalar_type=T,
        out_fields=fields,
        out_ctypes=out_ctypes,
    )


def _via_symox_device_function(plan, dialect, name, out_type, fields, layout, cse, header, qualifier):
    """Chemin symox récent : ``symox.extra.cudacode_function`` (pointeurs, struct nommée).

    Même ABI que le chemin du commit épinglé : paramètres ``xi_<n>``...,
    ``struct out_type`` toujours présente (sorties passées en liste), champs
    ``o<n>``, aides ``kp_pow*`` (aucune base n'est répétée textuellement),
    pas de CSE en présence de ``Piecewise``, pas de rsqrt ni fast-math.
    """
    from symox.extra import cudacode_function

    exprs = [_rewrite_integer_powers(_expand_ite(e)) for e in plan.outputs]
    body = cudacode_function(
        name,
        [list(a.symbols) for a in plan.args],
        exprs,
        dtype=plan.dtype,
        dialect=dialect,
        cse=cse,
        arg_names=[l.param for l in layout],
        result_type=out_type,
        result_fields=list(fields),
        use_rsqrt=False,
        reciprocal_sqrt=False,
        user_functions=_POW_HELPERS,
    )
    return "\n".join([header + " (symox cudacode_function)", _HELPERS_TEMPLATE.format(q=qualifier).rstrip(), body.rstrip()]) + "\n"


__all__ = ["ArgLayout", "PairSource", "pair_function", "DIALECTS", "GENERATOR_VERSION", "HALF_PRELUDE"]
