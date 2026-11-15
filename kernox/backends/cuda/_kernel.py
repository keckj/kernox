"""Génération du kernel CUDA de réduction fusionnée (enveloppe autour de ``kp_pair``).

Stratégie v0 (inspirée de KeOps ``GpuReduc1D`` @ ef72bc9, MIT ; code réécrit,
voir ``THIRD_PARTY.md``) :

- un thread par ligne ``i`` ; variables ``i`` et paramètres en registres ;
- variables ``j`` chargées par tuiles de ``tj`` lignes en mémoire partagée
  (copie coalescée de la plage contiguë par tout le bloc) ; ``tj`` est
  indépendant de ``blockDim.x`` pour que les grandes dimensions ne
  contraignent pas la taille de bloc ;
- accumulation en registres, aucune matrice M×N ;
- partition optionnelle de l'axe ``j`` (``gridDim.y = S`` tranches) : chaque
  tranche écrit un état partiel borné (M × S × état) dans un temporaire Torch,
  fusionné ensuite dans l'ordre des tranches par ``kp_merge`` (déterministe,
  préserve le départage des égalités par indice).

LOGSUMEXP / SUMSOFTMAXWEIGHT (contrat v0.5) : état ``(m, s[, t_e])`` en ``ACC``
par composante, décalé par le maximum courant, mis à jour terme par terme en une
seule passe (pas d'état de tuile : un seul jeu de registres, ce qui compte pour
SSMW à grand E) ; états partiels fusionnés par ``kp_merge``. Un terme ``f = −inf``
est ignoré (poids nul, ses ``g`` aussi) ; à égalité de maxima le facteur vaut
exactement 1, d'où ``exp(−inf − (−inf))`` jamais évalué ; ``m`` n'est jamais NaN
(les NaN vivent dans ``s``/``t``). ``exp``/``log`` exacts (``expf``/``exp``,
``logf``/``log`` : NVRTC sans ``--use_fast_math``).

Schémas de sommation (contrat v0.7, ``Plan.sum_scheme``, lu par ``getattr`` en
attendant le champ) : ``direct`` (SUM : somme par tuile puis total, comme avant ;
LSE/SSMW : un seul niveau), ``block`` (SUM : identique, T = tuile j du lancement ;
LSE/SSMW : état de tuile neuf fusionné en fin de tuile), ``kahan`` (Neumaier :
correctif par somme, redimensionné avec ``s`` pour LSE/SSMW ; fusion TwoSum).

Sémantique : ``docs/kernel-backend-contract.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...plan import Plan, ReductionKind as R

GENERATOR_VERSION = "kernox-cuda-reduce-17"

CTYPE = {"float16": "kp_h", "float32": "float", "float64": "double"}

# Avenant v0.9 (FP16) : type kp_h, binary16 en mémoire (2 o), chaque opération évaluée en
# FP32 puis arrondie au plus proche en FP16 (cvt.rn.f16.f32, PTX disponible dès sm_20) :
# même sémantique que l'oracle (chaque opération arrondie en FP16) sur toutes les
# architectures, y compris sm_52 sans arithmétique half native. L'arrondi FP32 puis FP16
# d'un +, −, ×, / ou sqrt est l'arrondi direct (24 >= 2·11 + 2 bits). Aucune contraction
# FMA possible entre opérations (conversion explicite). Conversion vers float explicite :
# les fonctions mathématiques float ne s'appliquent jamais sans arrondi.
HALF_TYPE = r"""
#ifndef KP_HALF_TYPE
#define KP_HALF_TYPE
struct kp_h {
  unsigned short b;
  kp_h() = default;
  __device__ __forceinline__ kp_h(float f) { asm("cvt.rn.f16.f32 %0, %1;" : "=h"(b) : "f"(f)); }
  __device__ __forceinline__ kp_h(double d) { asm("cvt.rn.f16.f64 %0, %1;" : "=h"(b) : "d"(d)); }
  __device__ __forceinline__ kp_h(int i) : kp_h((float)i) {}
  __device__ __forceinline__ kp_h(long long i) : kp_h((float)i) {}
  __device__ __forceinline__ explicit operator float() const { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(b)); return f; }
  __device__ __forceinline__ explicit operator double() const { return (double)(float)*this; }
};
static_assert(sizeof(kp_h) == 2, "kp_h : 2 octets");
__device__ __forceinline__ float kp_f(const kp_h a) { return (float)a; }
__device__ __forceinline__ kp_h kp_bits(unsigned short b) { kp_h r; r.b = b; return r; }
__device__ __forceinline__ kp_h operator-(const kp_h a) { return kp_bits((unsigned short)(a.b ^ 0x8000u)); }
__device__ __forceinline__ kp_h operator+(const kp_h a) { return a; }
#define KP_H_BIN(op) \
  __device__ __forceinline__ kp_h operator op(const kp_h a, const kp_h b) { return kp_h(kp_f(a) op kp_f(b)); }
#if __CUDA_ARCH__ >= 530
// sm_53+ : +, −, × natifs en FP16, arrondi .rn explicite (aucune contraction FMA) ;
// résultat identique au passage par FP32 (arrondi correct des deux côtés)
#define KP_H_NAT(op, ins) \
  __device__ __forceinline__ kp_h operator op(const kp_h a, const kp_h b) { \
    kp_h r; asm(ins " %0, %1, %2;" : "=h"(r.b) : "h"(a.b), "h"(b.b)); return r; }
KP_H_NAT(+, "add.rn.f16") KP_H_NAT(-, "sub.rn.f16") KP_H_NAT(*, "mul.rn.f16") KP_H_BIN(/)
#else
KP_H_BIN(+) KP_H_BIN(-) KP_H_BIN(*) KP_H_BIN(/)
#endif
#define KP_H_CMP(op) \
  __device__ __forceinline__ bool operator op(const kp_h a, const kp_h b) { return kp_f(a) op kp_f(b); }
KP_H_CMP(<) KP_H_CMP(>) KP_H_CMP(<=) KP_H_CMP(>=) KP_H_CMP(==) KP_H_CMP(!=)
#define KP_H_SET(op) \
  __device__ __forceinline__ kp_h& operator op##=(kp_h& a, const kp_h b) { a = a op b; return a; }
KP_H_SET(+) KP_H_SET(-) KP_H_SET(*) KP_H_SET(/)
#define KP_H_F1(name, fn) \
  __device__ __forceinline__ kp_h name(const kp_h a) { return kp_h(fn(kp_f(a))); }
#define KP_H_M1(base) KP_H_F1(base, base##f) KP_H_F1(base##f, base##f)
KP_H_M1(exp) KP_H_M1(exp2) KP_H_M1(expm1) KP_H_M1(log) KP_H_M1(log2) KP_H_M1(log10) KP_H_M1(log1p)
KP_H_M1(sqrt) KP_H_M1(cbrt) KP_H_M1(rsqrt) KP_H_M1(sin) KP_H_M1(cos) KP_H_M1(tan) KP_H_M1(asin)
KP_H_M1(acos) KP_H_M1(atan) KP_H_M1(sinh) KP_H_M1(cosh) KP_H_M1(tanh) KP_H_M1(asinh) KP_H_M1(acosh)
KP_H_M1(atanh) KP_H_M1(erf) KP_H_M1(erfc) KP_H_M1(floor) KP_H_M1(ceil) KP_H_M1(round) KP_H_M1(rint)
KP_H_M1(trunc) KP_H_M1(fabs) KP_H_M1(lgamma) KP_H_M1(tgamma)
KP_H_F1(abs, fabsf)
#define KP_H_F2(name, fn) \
  __device__ __forceinline__ kp_h name(const kp_h a, const kp_h b) { return kp_h(fn(kp_f(a), kp_f(b))); }
#define KP_H_M2(base) KP_H_F2(base, base##f) KP_H_F2(base##f, base##f)
KP_H_M2(pow) KP_H_M2(atan2) KP_H_M2(fmod) KP_H_M2(fmax) KP_H_M2(fmin) KP_H_M2(copysign) KP_H_M2(hypot)
KP_H_M2(remainder)
__device__ __forceinline__ bool isnan(const kp_h a) { return (a.b & 0x7fffu) > 0x7c00u; }
__device__ __forceinline__ bool kp_num(const kp_h a) { return (a.b & 0x7fffu) <= 0x7c00u; }
__device__ __forceinline__ bool isinf(const kp_h a) { return (a.b & 0x7fffu) == 0x7c00u; }
__device__ __forceinline__ bool isfinite(const kp_h a) { return (a.b & 0x7c00u) != 0x7c00u; }
__device__ __forceinline__ bool signbit(const kp_h a) { return (a.b & 0x8000u) != 0; }
__device__ __forceinline__ kp_h __shfl_sync(unsigned m, const kp_h v, int l, int w = 32) {
  return kp_bits((unsigned short)__shfl_sync(m, (unsigned)v.b, l, w)); }
__device__ __forceinline__ kp_h __shfl_up_sync(unsigned m, const kp_h v, unsigned d, int w = 32) {
  return kp_bits((unsigned short)__shfl_up_sync(m, (unsigned)v.b, d, w)); }
__device__ __forceinline__ kp_h kp_exp(kp_h x) { return exp(x); }
__device__ __forceinline__ kp_h kp_log(kp_h x) { return log(x); }
__device__ __forceinline__ kp_h kp_lowest(kp_h) { return kp_bits((unsigned short)0xfbffu); }
__device__ __forceinline__ kp_h kp_fabs(kp_h x) { return kp_bits((unsigned short)(x.b & 0x7fffu)); }
#endif
"""


def fast_pair_source(source: str) -> str:
    """Avenant v0.10 : ``expf``/``logf`` de la paire FP32 → ``kp_fexp``/``kp_flog`` (approchés)."""
    import re

    return re.sub(r"\blogf\(", "kp_flog(", re.sub(r"\bexpf\(", "kp_fexp(", source))


def half_pair_source(source: str) -> str:
    """Source de paire générée en FP32 (plan à dtype float32, constantes déjà arrondies
    en FP16 par le builder) réécrite en kp_h : chaque mot « float » devient « kp_h »."""
    import re

    return HALF_TYPE + re.sub(r"\bfloat\b", "kp_h", source)

_PRELUDE = r"""
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
typedef long long kp_idx;
// Avenant v0.10 (fast_math, FP32 seulement) : exp/log approchés comme PyKeOps
// use_fast_math (ex2.approx/lg2.approx, sans ftz : sous-normaux conservés).
// exp(x) = 2^(x·log2 e) ; log(x) = lg2(x)·ln 2. ±inf, NaN, 0 : mêmes cas limites qu'IEEE.
__device__ __forceinline__ float kp_fexp(float x) {
  float y; asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f)); return y; }
__device__ __forceinline__ float kp_flog(float x) {
  float y; asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y * 0.6931471805599453f; }
// v non NaN, en arithmétique entière : hors du pipeline FP64 (DSETP), goulot des
// GeForce en double ; garde de liste vide des ARG*/K* (passage critique en j).
__device__ __forceinline__ float kp_lowest(float) { return -3.402823466e38f; }
__device__ __forceinline__ double kp_lowest(double) { return -1.7976931348623157e308; }
__device__ __forceinline__ bool kp_num(float x) { return (__float_as_uint(x) & 0x7fffffffu) <= 0x7f800000u; }
__device__ __forceinline__ bool kp_num(double x) {
  return ((unsigned long long)__double_as_longlong(x) & 0x7fffffffffffffffull) <= 0x7ff0000000000000ull; }
"""


@dataclass(frozen=True)
class StateLayout:
    """État partiel par ligne et par tranche (pour la partition de j)."""

    n_val: int        # nombre de valeurs (type ``val_type``)
    val_type: str     # "T", "ACC" ou "kp_idx"
    n_idx: int        # nombre d'indices int64

    def bytes(self, dtype_bytes: int, acc_bytes: int) -> int:
        vb = {"T": dtype_bytes, "ACC": acc_bytes, "kp_idx": 8}[self.val_type]
        return self.n_val * vb + self.n_idx * 8


SUM_SCHEMES = ("direct", "block", "kahan")
SOFT = (R.LOGSUMEXP, R.SUMSOFTMAXWEIGHT, R.LOGSUMEXPWEIGHT)
# Avenant v0.8 : MIN_ARGMIN / MAX_ARGMAX = état et algorithme d'ARGMIN / ARGMAX (valeur
# écrite en plus de l'indice) ; LOGSUMEXPWEIGHT = état de SUMSOFTMAXWEIGHT, fin m + log t_e.
BASE_KIND = {R.MIN_ARGMIN: R.ARGMIN, R.MAX_ARGMAX: R.ARGMAX, R.LOGSUMEXPWEIGHT: R.SUMSOFTMAXWEIGHT}


def sum_scheme(plan: Plan) -> str:
    """Schéma de sommation effectif (contrat v0.7) : celui du plan pour SUM, LOGSUMEXP
    et SUMSOFTMAXWEIGHT, "direct" (sans objet) pour les autres réductions."""
    if plan.reduction.kind not in (R.SUM,) + SOFT:
        return "direct"
    return getattr(plan, "sum_scheme", "direct")


def state_layout(plan: Plan) -> StateLayout:
    k, d = BASE_KIND.get(plan.reduction.kind, plan.reduction.kind), plan.out_dim
    K = plan.reduction.k or 0
    kah = sum_scheme(plan) == "kahan"   # compensation : un terme correctif par somme
    if k is R.SUM:
        return StateLayout(2 * d if kah else d, "ACC", 0)
    if k is R.COUNT:
        return StateLayout(d, "kp_idx", 0)
    if k in (R.MIN, R.MAX):
        return StateLayout(d, "T", 0)
    if k in (R.ARGMIN, R.ARGMAX):
        return StateLayout(d, "T", d)
    if k in (R.KMIN, R.ARGKMIN, R.KMIN_ARGKMIN):
        return StateLayout(K, "T", K)
    if k is R.LOGSUMEXP:
        return StateLayout((3 if kah else 2) * d, "ACC", 0)     # (m_c, s_c[, ks_c]) par composante
    if k is R.SUMSOFTMAXWEIGHT:
        return StateLayout(2 * d + 1 if kah else 1 + d, "ACC", 0)   # (m, s, t_1..t_E[, ks, kt_1..kt_E])
    raise ValueError(k)


_SOFT_HELPERS = r"""
__device__ __forceinline__ float kp_exp(float x) { return expf(x); }
__device__ __forceinline__ double kp_exp(double x) { return exp(x); }
__device__ __forceinline__ float kp_log(float x) { return logf(x); }
__device__ __forceinline__ float kp_fabs(float x) { return fabsf(x); }
__device__ __forceinline__ double kp_fabs(double x) { return fabs(x); }
__device__ __forceinline__ double kp_log(double x) { return log(x); }
"""

# Kahan–Babuška (Neumaier), contrat v0.7 : additions et soustractions seulement (aucune
# contraction FMA possible) ; le test de grandeur choisit l'ordre exact du TwoSum rapide.
# Ajouter un état (s2, c2) : kp_neu(s, c, s2) puis c += c2 (TwoSum(s1, s2) + c1 + c2).
# Résultat : s + c si s est fini ; sinon s (±inf, NaN : le correctif vaudrait NaN).
_KAHAN_HELPERS = r"""
__device__ __forceinline__ void kp_neu(ACC& s, ACC& c, const ACC x) {
  const ACC t = s + x;
  const ACC as = s < (ACC)0 ? -s : s, ax = x < (ACC)0 ? -x : x;
  c += (as >= ax) ? ((s - t) + x) : ((x - t) + s);
  s = t;
}
__device__ __forceinline__ ACC kp_neu_fin(const ACC s, const ACC c) { return (s - s == (ACC)0) ? s + c : s; }
"""


def _soft_comps(kind, d):
    """Composantes d'état : [(m, s, [t...], indice du score, [indices des poids])]."""
    if kind is R.LOGSUMEXP:
        return [(f"m{c}", f"s{c}", [], c, []) for c in range(d)]
    return [("m0", "s0", [f"t{e}" for e in range(d - 1)], 0, list(range(1, d)))]


def _soft_state_names(comp, kahan):
    """Ordre des valeurs d'un état partiel : m, s, t..., puis ks, kt... (Kahan)."""
    m, s, ts = comp[0], comp[1], comp[2]
    return [m, s] + ts + ([f"k{x}" for x in [s] + ts] if kahan else [])


def _add(var, x, kahan):
    return f" kp_neu({var}, k{var}, {x});" if kahan else f" {var} += {x};"


def _soft_insert(m, s, ts, v, gs, kahan=False):
    """Insertion du terme v (poids g_e) ; v == −inf ignoré (ses g aussi)."""
    g = "".join(f" const ACC g{e} = (ACC)({x});" for e, x in enumerate(gs))
    if kahan:  # redimensionnement de s, t et de leurs correctifs, puis addition compensée
        up = "".join(f" {x} *= c; k{x} *= c;" for x in [s] + ts) + _add(s, "(ACC)1", True) + "".join(
            _add(t, f"g{e}", True) for e, t in enumerate(ts))
    else:
        up = f" {s} = {s} * c + (ACC)1;" + "".join(f" {t} = {t} * c + g{e};" for e, t in enumerate(ts))
    if not kahan:
        # Sans branche sur v > m : les deux cas calculent exp(−|v − m|) (négation exacte),
        # un seul exp par j et des sélections ; mêmes opérations, donc même résultat au bit
        # près que la forme à branches, sans divergence de warp (les lignes i d'un warp
        # changent de maximum à des j différents ; mesuré sur LSE D = 2, 2080 Ti).
        # d = 0 à égalité (évite inf − inf) : exp(−0) = 1 exactement ; v = −inf : état
        # inchangé (sélection, pas de branche autour de l'exp).
        # m part du plus petit fini (decl_soft) : v = −inf ⇒ d = −inf, e = 0, s + 0 = s, sans
        # test. Le test ok ne reste que pour les poids g (0·g serait NaN si g est infini).
        sel = "".join(f" {t} = ok ? (gt ? {t} * e + g{q} : {t} + e * g{q}) : {t};" for q, t in enumerate(ts))
        okd = "const bool ok = v != -(ACC)INFINITY; " if ts else ""
        return (f"{{ const ACC v = (ACC)({v});{g}\n"
                f"  {okd}const bool gt = v > {m};\n"
                f"  const ACC d = (v == {m}) ? (ACC)0 : v - {m};\n"
                f"  const ACC e = kp_exp(-kp_fabs(d));\n"
                f"  {s} = gt ? {s} * e + (ACC)1 : {s} + e;{sel} {m} = gt ? v : {m};\n"
                f"  }}")
    add = _add(s, "q", kahan) + "".join(_add(t, f"q * g{e}", kahan) for e, t in enumerate(ts))
    return (f"{{ const ACC v = (ACC)({v});{g}\n"
            f"  if (v != -(ACC)INFINITY) {{\n"
            f"    if (v > {m}) {{ const ACC c = kp_exp({m} - v);{up} {m} = v; }}\n"
            f"    else {{ const ACC q = (v == {m}) ? (ACC)1 : kp_exp(v - {m});{add} }}\n"
            f"  }} }}")


def _soft_merge(names, src, kahan=False):
    """Fusion de l'état ``src`` (expressions, même ordre que ``names``) dans l'état
    ``names`` ; facteur exactement 1 à égalité des maxima."""
    m = names[0]
    n = (len(names) - 1) // 2 if kahan else len(names) - 1
    sums, ks = names[1:1 + n], names[1 + n:]
    bs = [f"b{q}" for q in range(len(names))]
    decl = "const ACC " + ", ".join(f"{b} = {x}" for b, x in zip(bs, src)) + ";"
    bsum, bk = bs[1:1 + n], bs[1 + n:]
    if kahan:
        up = "".join(f" {x} *= c; {k} *= c;" for x, k in zip(sums, ks)) + "".join(
            f" kp_neu({x}, {k}, {b}); {k} += {bkk};" for x, k, b, bkk in zip(sums, ks, bsum, bk))
        add = "".join(f" kp_neu({x}, {k}, {b} * q); {k} += {bkk} * q;" for x, k, b, bkk in zip(sums, ks, bsum, bk))
    else:
        up = "".join(f" {x} = {x} * c + {b};" for x, b in zip(sums, bsum))
        add = "".join(f" {x} += {b} * q;" for x, b in zip(sums, bsum))
    return (f"{{ {decl}\n"
            f"  if ({bs[0]} > {m}) {{ const ACC c = kp_exp({m} - {bs[0]});{up} {m} = {bs[0]}; }}\n"
            f"  else {{ const ACC q = ({bs[0]} == {m}) ? (ACC)1 : kp_exp({bs[0]} - {m});{add} }} }}")


def _soft_finish(kind, m, s, ts, dst, kahan=False, lsew=False):
    """LSE : ``s`` NaN → NaN ; ``m = +inf`` → +inf ; sinon ``m + log s`` (vide : −inf).
    SSMW : ``m = +inf`` → NaN ; sinon ``t / s`` (vide : 0/0 = NaN).
    LSEW (avenant v0.8) : ``m + log t_e`` sans cas particulier (vide : −inf ; m = +inf :
    +inf si t_e > 0, NaN sinon ; t_e < 0 : NaN)."""
    if lsew:
        return "{ " + " ".join(
            f"{dst(e)} = {m} + kp_log({f'kp_neu_fin({t}, k{t})' if kahan else t});" for e, t in enumerate(ts)) + " }"
    S = f"kp_neu_fin({s}, k{s})" if kahan else s
    if kind is R.LOGSUMEXP:
        return (f"{{ const ACC S = {S}; "
                f"{dst(0)} = (S != S) ? S : (({m} == (ACC)INFINITY) ? {m} : {m} + kp_log(S)); }}")
    return f"{{ const ACC S = {S};\n" + "\n".join(
        f"{dst(e)} = ({m} == (ACC)INFINITY) ? (ACC)NAN : "
        f"{f'kp_neu_fin({t}, k{t})' if kahan else t} / S;" for e, t in enumerate(ts)) + " }"


def output_names(plan: Plan) -> tuple[str, ...]:
    k = plan.reduction.kind
    if k in (R.ARGMIN, R.ARGMAX, R.ARGKMIN):
        return ("index",)
    if k in (R.KMIN_ARGKMIN, R.MIN_ARGMIN, R.MAX_ARGMAX):
        return ("value", "index")
    return ("value",)


def ranges_tie_break(kind, coop) -> bool:
    """Le noyau ranges départage-t-il les égalités par j ? (ARG*/K* hors coopératif)."""
    return not coop and kind in (R.ARGMIN, R.ARGMAX, R.KMIN, R.ARGKMIN, R.KMIN_ARGKMIN)


def _ranges_kernel(lines, kind, K, d, coop, tie=True):
    """Avenant v0.6 §2 : ``kp_reduce_ranges``, dérivé du texte de ``kp_reduce``. Un bloc par
    tuile de travail ``kp_work[blockIdx.x] = (i0, i1, k0, k1)`` (hauteur <= blockDim.x / G),
    réduction sur ``∪ kp_red[k0:k1]`` dans l'ordre des intervalles, sans partition de j
    (gridDim.y = 1 : écriture finale directe). Les intervalles pouvant arriver dans le
    désordre, ARGMIN/ARGMAX/K* départagent les égalités par j (plus petit j de J(i)) ;
    le top-K coopératif l'est déjà (ordre (valeur, j) total). Les lignes non couvertes
    reçoivent le neutre par un lancement de ``kp_reduce`` à N = 0 (backend).

    ``tie=False`` : variante ``kp_reduce_ranges_mono`` sans départage, pour les
    tuiles dont les intervalles sont triés et disjoints (j strictement croissant :
    la comparaison stricte garde déjà le plus petit j). Le test d'égalité coûte
    jusqu'à 30 % sur sm_52 (KNN FP32)."""
    head, body = lines[0], "\n".join(lines[1:])
    assert head.startswith('extern "C" __global__ void kp_reduce(') and head.endswith(") {"), head
    name = "kp_reduce_ranges" if tie else "kp_reduce_ranges_mono"
    head = (head.replace("void kp_reduce(", f"void {name}(")[:-3]
            + ", const kp_idx* __restrict__ kp_work, const kp_idx* __restrict__ kp_red) {")
    wk = ("  const kp_idx* kp_wk = kp_work + 4 * (kp_idx)blockIdx.x;\n"
          "  const kp_idx kp_i1 = kp_wk[1];\n")

    def sub(old, new, count=1):
        nonlocal body
        assert body.count(old) == count, (old, body.count(old))
        body = body.replace(old, new)

    if coop:
        sub(f"  const kp_idx i = ((kp_idx)blockIdx.x * blockDim.x + threadIdx.x) / {coop};",
            wk + f"  const kp_idx i = kp_wk[0] + (kp_idx)(threadIdx.x / {coop});")
    else:
        sub("  const kp_idx i = (kp_idx)blockIdx.x * blockDim.x + threadIdx.x;",
            wk + "  const kp_idx i = kp_wk[0] + (kp_idx)threadIdx.x;")
    sub("  const kp_idx jb = (kp_idx)blockIdx.y * jchunk;\n", "")
    sub("  const kp_idx je = (jb + jchunk < N) ? (jb + jchunk) : N;\n", "")
    sub("  for (kp_idx js = jb; js < je; js += tj) {",
        "  for (kp_idx kr = kp_wk[2]; kr < kp_wk[3]; ++kr) {\n"
        "  const kp_idx jb = kp_red[2 * kr], je = kp_red[2 * kr + 1];\n"
        "  for (kp_idx js = jb; js < je; js += tj) {")
    sub("  }\n  if (i < M) {\n    if (gridDim.y == 1) {", "  }\n  }\n  if (i < M) {\n    if (gridDim.y == 1) {")
    import re
    body = re.sub(r"\bi < M\b", "i < kp_i1", body)
    if not tie:
        pass
    elif not coop and kind in (R.ARGMIN, R.ARGMAX):
        cmp = ">" if kind is R.ARGMAX else "<"
        for c in range(d):
            sub(f"(v {cmp} acc{c}) || (arg{c} < 0 && kp_num(v))",
                f"(v {cmp} acc{c}) || (v == acc{c} && jj < arg{c}) || (arg{c} < 0 && kp_num(v))")
    elif not coop and kind in (R.KMIN, R.ARGKMIN, R.KMIN_ARGKMIN):
        sub(f"(v < kv[{K - 1}]) || (ki[{K - 1}] < 0 && kp_num(v))",
            f"(v < kv[{K - 1}]) || (v == kv[{K - 1}] && jj < ki[{K - 1}]) || (ki[{K - 1}] < 0 && kp_num(v))")
        sub("b[k] = (v < kv[k]) || (ki[k] < 0);", "b[k] = (v < kv[k]) || (ki[k] < 0) || (v == kv[k] && jj < ki[k]);")
    return ["", head, body]


def _better(kind):
    # Pour ARGMIN/ARGMAX/KMIN : candidat (v, j) meilleur que (lv, lj) ?
    # lj < 0 : emplacement vide. Égalité de valeur : plus petit indice.
    cmp = ">" if kind is R.ARGMAX else "<"
    return (
        "__device__ __forceinline__ bool kp_better(T v, kp_lidx j, T lv, kp_lidx lj) {\n"
        f"  return (lj < 0) || (v {cmp} lv) || (v == lv && j < lj);\n}}\n"
    )


# Top-K coopératif (K > 256 : la liste triée ne tient plus dans les registres d'un
# thread). G lanes consécutives d'un warp par ligne i ; la lane t détient les rangs
# [t·LC, (t+1)·LC) de la liste triée des G·LC >= K meilleurs (valeur, j). Chaque
# lane évalue un j différent ; le lot des G candidats est inséré par mélanges de
# warp : une insertion décale la liste d'un rang, la lane qui reçoit le candidat
# l'insère et chaque lane suivante reçoit l'ancien dernier élément de la
# précédente (__shfl_up_sync). Flot uniforme dans le warp (lignes inactives sans
# candidat). Ordre lexicographique total (valeur, j) : le résultat ne dépend pas
# de l'ordre d'insertion, donc identique à la liste par thread. Même principe que
# la variante ``coop`` des backends sycl et hip et que WarpSelect (FAISS), réécrit.
COOP_GROUPS = (2, 4, 8, 16, 32)

_COOP_HELPER = r"""
__device__ __forceinline__ bool kp_less(T v, kp_lidx j, T lv, kp_lidx lj) {
  return (lj < 0) || (v < lv) || (v == lv && j < lj);
}
// Insère le lot (v, jv) des G lanes de chaque groupe (jv < 0 : pas de candidat).
// Renvoie false (uniforme dans le warp) si aucun candidat du warp ne bat le seuil
// de sa ligne : le seuil ne fait que baisser, un candidat refusé le reste.
__device__ __forceinline__ bool kp_coop_batch(const int lane, const int gb, const T v, const kp_lidx jv,
                                              T (&kv)[KP_LC], kp_lidx (&ki)[KP_LC]) {
  const unsigned FULL = 0xffffffffu;
  const T lv0 = __shfl_sync(FULL, kv[KP_LC - 1], gb + KP_G - 1);
  const kp_lidx lj0 = __shfl_sync(FULL, ki[KP_LC - 1], gb + KP_G - 1);
  const unsigned bits = __ballot_sync(FULL, jv >= 0 && kp_less(v, jv, lv0, lj0));
  if (bits == 0u) return false;
  unsigned todo = 0u;
  #pragma unroll
  for (int g0 = 0; g0 < 32; g0 += KP_G) todo |= bits >> g0;
  if (KP_G < 32) todo &= (1u << (KP_G & 31)) - 1u;
  #pragma unroll 1
  while (todo) {
    const int q = __ffs(todo) - 1;
    todo &= todo - 1u;
    const T cv = __shfl_sync(FULL, v, gb + q);
    const kp_lidx cj = __shfl_sync(FULL, jv, gb + q);
    const T lv = __shfl_sync(FULL, kv[KP_LC - 1], gb + KP_G - 1);
    const kp_lidx lj = __shfl_sync(FULL, ki[KP_LC - 1], gb + KP_G - 1);
    const bool aff = cj >= 0 && kp_less(cv, cj, lv, lj) && kp_less(cv, cj, kv[KP_LC - 1], ki[KP_LC - 1]);
    const T pv = __shfl_up_sync(FULL, kv[KP_LC - 1], 1);
    const kp_lidx pj = __shfl_up_sync(FULL, ki[KP_LC - 1], 1);
    const int paff = __shfl_up_sync(FULL, (int)aff, 1);
    if (aff) {
      const bool fromprev = paff != 0 && lane > 0;
      const T xv = fromprev ? pv : cv;
      const kp_lidx xj = fromprev ? pj : cj;
      bool c[KP_LC];
      #pragma unroll
      for (int k = 0; k < KP_LC; ++k) c[k] = kp_less(xv, xj, kv[k], ki[k]);
      #pragma unroll
      for (int k = KP_LC - 1; k > 0; --k) {
        kv[k] = c[k] ? (c[k - 1] ? kv[k - 1] : xv) : kv[k];
        ki[k] = c[k] ? (c[k - 1] ? ki[k - 1] : xj) : ki[k];
      }
      kv[0] = c[0] ? xv : kv[0];
      ki[0] = c[0] ? xj : ki[0];
    }
  }
  return true;
}
"""


def generate(plan: Plan, pair_source: str, pair_name: str, local_index: str = "int",
             staging: str = "smem", coop: int = 0) -> str:
    """Source CUDA complète (fonction de paire + kernels ``kp_reduce`` / ``kp_merge``).

    ``local_index`` : type des indices j conservés en registres (``int`` si
    N < 2^31, sinon ``long long``) ; les sorties et partiels restent int64.

    ``staging`` : ``"smem"`` (tuiles j copiées en mémoire partagée, défaut) ou
    ``"global"`` (lecture directe des lignes j en mémoire globale, sans copie ni
    barrière) quand une ligne j est trop large pour la mémoire partagée (grand D).
    Tous les threads d'un bloc lisent la même ligne j au même moment : diffusion par
    le cache L1/L2. Même ordre d'évaluation, donc même valeur que ``"smem"``.

    ``coop`` : G > 0, top-K coopératif (KMIN/ARGKMIN/KMIN_ARGKMIN) à G lanes par
    ligne i (``_COOP_HELPER``) ; un bloc couvre ``blockDim.x / G`` lignes.
    """
    if staging not in ("smem", "global"):
        raise ValueError(f"staging inconnu {staging!r}")
    kind0 = plan.reduction.kind
    kind = BASE_KIND.get(kind0, kind0)
    lsew = kind0 is R.LOGSUMEXPWEIGHT
    if coop and (coop not in COOP_GROUPS or kind not in (R.KMIN, R.ARGKMIN, R.KMIN_ARGKMIN)):
        raise ValueError(f"top-K coopératif : G = {coop} invalide pour {kind}")
    d = plan.out_dim
    K = plan.reduction.k or 0
    T = CTYPE[plan.dtype]
    ACC = CTYPE[plan.acc_dtype]
    args = plan.args
    jargs = [(n, a) for n, a in enumerate(args) if a.role == "j"]
    st = state_layout(plan)
    outs = output_names(plan)

    half = plan.dtype == "float16"
    # OUT : type des sorties accumulées (avenant v0.9 (b) : FP16 pour un plan float16)
    OUT = "kp_h" if half else ACC
    L = ([HALF_TYPE] if half else []) + [f"typedef {T} T;", f"typedef {ACC} ACC;", f"typedef {OUT} OUT;",
                                          _PRELUDE, f"typedef {local_index} kp_lidx;", pair_source, ""]
    if coop:
        LC = -(-K // coop)
        L += [f"#define KP_G {coop}", f"#define KP_LC {LC}", _COOP_HELPER]
    elif kind in (R.ARGMIN, R.ARGMAX, R.KMIN, R.ARGKMIN, R.KMIN_ARGKMIN):
        L.append(_better(kind))
    soft = kind in SOFT
    scheme = sum_scheme(plan)
    if scheme not in SUM_SCHEMES:
        raise ValueError(f"sum_scheme inconnu {scheme!r}")
    kahan, block = scheme == "kahan", scheme == "block"
    if soft:
        helpers = _SOFT_HELPERS
        if plan.fast_math and plan.dtype == "float32":  # avenant v0.10 : état max-shift et log final
            helpers = (helpers.replace("return expf(x);", "return kp_fexp(x);")
                       .replace("return logf(x);", "return kp_flog(x);"))
        L.append(helpers)
        comps = _soft_comps(kind, d)
    if kahan:
        L.append(_KAHAN_HELPERS)

    # ---------------------------------------------------------------- état
    def decl_soft(prefix=""):
        out = []
        for comp in comps:
            nm = _soft_state_names(comp, kahan)
            # m initial : plus petit fini (pas −inf) ; v = −inf donne alors exp(−inf) = 0 sans
            # test dédié (LOGSUMEXP), et une ligne vide finit en m + log 0 = −inf comme avant.
            out.append(f"ACC {prefix}{nm[0]} = kp_lowest((ACC)0);"
                       + "".join(f" ACC {prefix}{x} = (ACC)0;" for x in nm[1:]))
        return "\n".join(out)

    def decl_state():
        s = []
        if kind is R.SUM:
            s += [f"ACC acc{c} = (ACC)0;" + (f" ACC kacc{c} = (ACC)0;" if kahan else "") for c in range(d)]
        elif kind is R.COUNT:
            s += [f"kp_idx acc{c} = 0;" for c in range(d)]
        elif kind in (R.MIN, R.MAX):
            init = "(T)INFINITY" if kind is R.MIN else "-(T)INFINITY"
            s += [f"T acc{c} = {init};" for c in range(d)]
        elif kind in (R.ARGMIN, R.ARGMAX):
            init = "(T)INFINITY" if kind is R.ARGMIN else "-(T)INFINITY"
            s += [f"T acc{c} = {init}; kp_lidx arg{c} = -1;" for c in range(d)]
        elif soft:
            s.append(decl_soft())
        else:
            n = LC if coop else K
            s += [f"T kv[{n}]; kp_lidx ki[{n}];",
                  f"#pragma unroll\nfor (int k = 0; k < {n}; ++k) {{ kv[k] = (T)INFINITY; ki[k] = -1; }}"]
        return "\n".join(s)

    # Insertion triée (K) entièrement déroulée (indices statiques -> registres) :
    # b[k] = candidat meilleur que l'entrée k (monotone sur une liste triée) ;
    # décalage sans chaîne de dépendance entre positions.
    def insert_k(better):
        # b[K-1] vrai par construction (test d'entrée) : l'écrire évite que nvcc
        # recalcule la comparaison flottante (DSETP en double) sur le chemin critique.
        return ("{ bool b[%d];\n  #pragma unroll\n  for (int k = 0; k < %d; ++k) b[k] = %s;\n"
                "  b[%d] = true;\n"
                "  #pragma unroll\n  for (int k = %d; k > 0; --k) if (b[k]) "
                "{ kv[k] = b[k - 1] ? kv[k - 1] : v; ki[k] = b[k - 1] ? ki[k - 1] : jj; }\n"
                "  if (b[0]) { kv[0] = v; ki[0] = jj; } }") % (K, K, better, K - 1, K - 1)

    def accumulate(src_val, src_idx, c, merge=False):
        """Mise à jour de l'état avec la valeur ``src_val`` d'indice ``src_idx``.

        Boucle principale : les j arrivent croissants, donc une égalité de valeur
        garde l'entrée existante (plus petit j) et un seul test flottant suffit
        dans le cas courant ; la fusion des partiels utilise l'ordre (valeur, j).
        """
        if kind is R.SUM:
            if kahan:
                return f"kp_neu(acc{c}, kacc{c}, (ACC)({src_val}));"
            return f"tmp{c} += (ACC)({src_val});" if not merge else f"acc{c} += {src_val};"
        if kind is R.COUNT:
            return f"acc{c} += ({src_val}) ? 1 : 0;" if not merge else f"acc{c} += {src_val};"
        if kind is R.MIN:
            return f"{{ const T v = {src_val}; if (v < acc{c}) acc{c} = v; }}"
        if kind is R.MAX:
            return f"{{ const T v = {src_val}; if (v > acc{c}) acc{c} = v; }}"
        cmp = ">" if kind is R.ARGMAX else "<"
        if kind in (R.ARGMIN, R.ARGMAX):
            if merge:
                test = f"jj >= 0 && kp_better(v, jj, acc{c}, arg{c})"
            else:  # NaN : v < acc et kp_num(v) faux ; +-inf valide accepté si vide
                test = f"(v {cmp} acc{c}) || (arg{c} < 0 && kp_num(v))"
            return (f"{{ const T v = {src_val}; const kp_lidx jj = (kp_lidx)({src_idx}); "
                    f"if ({test}) {{ acc{c} = v; arg{c} = jj; }} }}")
        # K*
        if merge:
            test = f"jj >= 0 && kp_better(v, jj, kv[{K - 1}], ki[{K - 1}])"
            body = insert_k("kp_better(v, jj, kv[k], ki[k])")
        else:
            test = f"(v < kv[{K - 1}]) || (ki[{K - 1}] < 0 && kp_num(v))"
            body = insert_k("(v < kv[k]) || (ki[k] < 0)")
        return (f"{{ const T v = {src_val}; const kp_lidx jj = (kp_lidx)({src_idx}); "
                f"if ({test}) {body} }}")

    def write_final(row):
        w = []
        if soft:
            od = d if kind is R.LOGSUMEXP else d - 1
            for c, (m_, s_, ts, _v, _g) in enumerate(comps):
                if kind is R.LOGSUMEXP:
                    w.append(_soft_finish(kind, m_, s_, ts, lambda e, c=c: f"out_value[{row} * {od} + {c}]", kahan))
                else:
                    w.append(_soft_finish(kind, m_, s_, ts, lambda e: f"out_value[{row} * {od} + {e}]", kahan, lsew))
        elif kind is R.SUM and kahan:
            w += [f"out_value[{row} * {d} + {c}] = kp_neu_fin(acc{c}, kacc{c});" for c in range(d)]
        elif kind in (R.SUM, R.COUNT, R.MIN, R.MAX):
            w += [f"out_value[{row} * {d} + {c}] = acc{c};" for c in range(d)]
        elif kind in (R.ARGMIN, R.ARGMAX):
            w += [f"out_index[{row} * {d} + {c}] = (kp_idx)arg{c};" for c in range(d)]
            if "value" in outs:  # MIN_ARGMIN / MAX_ARGMAX : valeur du même passage
                w += [f"out_value[{row} * {d} + {c}] = acc{c};" for c in range(d)]
        elif coop:  # lane t : rangs [t·LC, (t+1)·LC) ∩ [0, K)
            loop = f"#pragma unroll\nfor (int k = 0; k < {LC}; ++k) {{ const int r = lane * {LC} + k; if (r < {K}) {{ "
            if "value" in outs:
                loop += f"out_value[{row} * {K} + r] = kv[k]; "
            if "index" in outs:
                loop += f"out_index[{row} * {K} + r] = (kp_idx)ki[k]; "
            w.append(loop + "} }")
        else:
            loop = f"#pragma unroll\nfor (int k = 0; k < {K}; ++k) {{ "
            if "value" in outs:
                loop += f"out_value[{row} * {K} + k] = kv[k]; "
            if "index" in outs:
                loop += f"out_index[{row} * {K} + k] = (kp_idx)ki[k]; "
            w.append(loop + "}")
        return "\n".join(w)

    vt = {"T": "T", "ACC": "ACC", "kp_idx": "kp_idx"}[st.val_type]

    def write_partial():
        w = [f"const kp_idx base = (i * (kp_idx)gridDim.y + blockIdx.y);"]
        if kind in (R.SUM, R.COUNT, R.MIN, R.MAX):
            w += [f"part_val[base * {st.n_val} + {c}] = acc{c};" for c in range(d)]
            if kind is R.SUM and kahan:
                w += [f"part_val[base * {st.n_val} + {d + c}] = kacc{c};" for c in range(d)]
        elif soft:
            q = 0
            for comp in comps:
                for x in _soft_state_names(comp, kahan):
                    w.append(f"part_val[base * {st.n_val} + {q}] = {x};")
                    q += 1
        elif kind in (R.ARGMIN, R.ARGMAX):
            w += [f"part_val[base * {d} + {c}] = acc{c}; part_idx[base * {d} + {c}] = (kp_idx)arg{c};"
                  for c in range(d)]
        elif coop:
            w.append(f"#pragma unroll\nfor (int k = 0; k < {LC}; ++k) {{ const int r = lane * {LC} + k; "
                     f"if (r < {K}) {{ part_val[base * {K} + r] = kv[k]; part_idx[base * {K} + r] = (kp_idx)ki[k]; }} }}")
        else:
            w.append(f"#pragma unroll\nfor (int k = 0; k < {K}; ++k) {{ "
                     f"part_val[base * {K} + k] = kv[k]; part_idx[base * {K} + k] = (kp_idx)ki[k]; }}")
        return "\n".join(w)

    out_params = []
    if "value" in outs:
        vtype = {R.SUM: "OUT", R.COUNT: "kp_idx", R.LOGSUMEXP: "OUT",
                 R.SUMSOFTMAXWEIGHT: "OUT"}.get(kind, "T")
        out_params.append(f"{vtype}* __restrict__ out_value")
    if "index" in outs:
        out_params.append("kp_idx* __restrict__ out_index")
    part_params = [f"{vt}* __restrict__ part_val", "kp_idx* __restrict__ part_idx"]

    in_params = [f"const T* __restrict__ in{n}" for n in range(len(args))]

    # Batch natif (avenant v0.6 §1) : pointeurs de base suffixés « _ », indice de lot
    # bz = b0 + blockIdx.z ; entrées décalées par la table boff (nargs, nbt) en éléments,
    # sorties (nb, M, ·) et partiels (nb, M, S, ·) contigus.
    batched = bool(getattr(plan, "batch_rank", 0))
    out_w = {}
    if batched:
        from ...plan import output_shapes as _oshapes
        for name, (shape, _dt) in _oshapes(plan, 1).items():
            w_ = 1
            for x in shape[1:]:
                w_ *= x
            out_w[name] = w_
        in_params = [p_ + "_" for p_ in in_params]
        out_params = [p_ + "_" for p_ in out_params]
        part_params = [p_ + "_" for p_ in part_params]
    tail = ["const kp_idx* __restrict__ boff", "const kp_idx nbt", "const kp_idx b0"] if batched else []

    def batch_prologue(rows):
        """Décalages du lot ``bz`` ; ``rows`` : tranches de partiels par ligne (expression C),
        None pour kp_merge (S)."""
        if not batched:
            return []
        b = ["  const kp_idx bz = b0 + (kp_idx)blockIdx.z;"]
        if rows is not None:
            b += [f"  const T* __restrict__ in{n} = in{n}_ + boff[{n} * nbt + bz];" for n in range(len(args))]
        if "value" in outs:
            vtype_ = {R.SUM: "OUT", R.COUNT: "kp_idx", R.LOGSUMEXP: "OUT",
                      R.SUMSOFTMAXWEIGHT: "OUT"}.get(kind, "T")
            b.append(f"  {vtype_}* __restrict__ out_value = out_value_ + bz * M * {out_w['value']};")
        if "index" in outs:
            b.append(f"  kp_idx* __restrict__ out_index = out_index_ + bz * M * {out_w['index']};")
        cq = "const " if rows is None else ""
        r_ = "S" if rows is None else rows
        b.append(f"  {cq}{vt}* __restrict__ part_val = part_val_ + bz * M * (kp_idx){r_} * {st.n_val};")
        b.append(f"  {cq}kp_idx* __restrict__ part_idx = part_idx_ + bz * M * (kp_idx){r_} * {st.n_idx};")
        return b

    # ---------------------------------------------------------------- kp_reduce
    red_start = len(L)
    L.append(
        'extern "C" __global__ void kp_reduce(const kp_idx M, const kp_idx N, const kp_idx jchunk, const int tj, '
        + ", ".join(in_params + out_params + part_params + tail) + ") {"
    )
    L += batch_prologue("gridDim.y")
    if coop:
        L.append(f"  const int lane = (int)(threadIdx.x % {coop});")
        L.append(f"  const int gb = (int)(threadIdx.x & 31u) - (int)(threadIdx.x & 31u) % {coop};  // 1re lane du groupe")
        L.append(f"  const kp_idx i = ((kp_idx)blockIdx.x * blockDim.x + threadIdx.x) / {coop};")
    else:
        L.append("  const kp_idx i = (kp_idx)blockIdx.x * blockDim.x + threadIdx.x;")
    L.append("  const kp_idx jb = (kp_idx)blockIdx.y * jchunk;")
    L.append("  const kp_idx je = (jb + jchunk < N) ? (jb + jchunk) : N;")
    L.append("  extern __shared__ __align__(16) unsigned char kp_smem[];")
    # Disposition des tuiles j : d'abord les tuiles dont la ligne n'est pas un multiple
    # de 16 o (la plus large en tête, à l'origine de kp_smem), puis les autres dans
    # l'ordre des arguments. ptxas ne vectorise les lectures partagées (LDS.128) d'une
    # telle tuile déroulée que si son adresse dérive de kp_smem sans décalage
    # d'exécution ; __builtin_assume_aligned sur la base décalée n'y suffit pas
    # (SASS vérifié sm_52/75/120 : gauss D = 3 FP32, 23 -> 13 LDS pour 7 couples).
    # Les lignes multiples de 16 o sont déjà vectorisées (assume_aligned ci-dessous) :
    # leur ordre est conservé (gauss D = 16 : source inchangée). Sans effet sur la
    # valeur : seules les adresses changent.
    tb = {"float64": 8, "float32": 4, "float16": 2}[plan.dtype]

    # FP32 seulement : en FP64 (gauss D = 3), le réordonnancement fait passer le kernel
    # de 39 à 46 registres sur sm_52 et coûte x1,065 sur le 980 Ti (10^4) pour un gain
    # de x0,96-0,98 sur le 5090 ; l'ordre des arguments est conservé en FP64.
    def _layout_key(t):
        aligned = plan.dtype == "float64" or (t[1].dim * tb) % 16 == 0
        return (aligned, 0 if aligned else -t[1].dim)

    off = 0
    for n, a in (sorted(jargs, key=_layout_key) if staging == "smem" else ()):
        L.append(f"  T* sh{n} = ((T*)kp_smem) + {off} * (kp_idx)tj;")
        off += a.dim
    for n, a in enumerate(args):
        if a.role == "param":
            L.append(f"  T r{n}[{a.dim}];")
            L.append(f"  #pragma unroll\n  for (int c = 0; c < {a.dim}; ++c) r{n}[c] = in{n}[c];")
        elif a.role == "i":
            L.append(f"  T r{n}[{a.dim}];")
            L.append(f"  if (i < M) {{\n  #pragma unroll\n  for (int c = 0; c < {a.dim}; ++c) "
                     f"r{n}[c] = in{n}[i * {a.dim} + c]; }}")
    L.append(decl_state())
    L.append("  for (kp_idx js = jb; js < je; js += tj) {")
    L.append("    const int nt = (int)((je - js) < (kp_idx)tj ? (je - js) : (kp_idx)tj);")
    if staging == "smem":
        for n, a in jargs:
            L.append(f"    for (int e = threadIdx.x; e < nt * {a.dim}; e += blockDim.x) "
                     f"sh{n}[e] = in{n}[js * {a.dim} + e];")
        L.append("    __syncthreads();")
    else:
        for n, a in jargs:
            L.append(f"    const T* sh{n} = in{n} + js * {a.dim};")
    if coop:  # flot uniforme dans le warp : pas de garde i < M autour des mélanges
        L.append(f"    for (int jr = 0; jr < nt; jr += {coop}) {{")
        L.append("      const int jl = jr + lane;")
        L.append("      T v = (T)0; kp_lidx jv = -1;")
        L.append("      if (i < M && jl < nt) {")
    L.append("    if (i < M) {" if not coop else "")
    if kind is R.SUM and not kahan:
        L += [f"      ACC tmp{c} = (ACC)0;" for c in range(d)]
    if soft and block:  # schéma « block » : état de tuile neuf, fusionné en fin de tuile
        L.append(decl_soft("l"))
    # Tuiles j : la région de l'argument n commence à off_n * tj éléments, tj multiple
    # de 4 (garanti par le backend) : base alignée sur 16 o. On le dit au compilateur
    # quand la ligne a une taille multiple de 16 (resp. 8) o, pour des chargements
    # vectoriels (ld.shared.v4/v2) quel que soit l'ordre des arguments (tb défini plus haut).
    call_args = []
    for n, a in enumerate(args):
        if a.role != "j":
            call_args.append(f"r{n}")
            continue
        row = a.dim * tb
        al = 16 if row % 16 == 0 else (8 if row % 8 == 0 else 0)
        if half:  # régions de tuile alignées sur 8 o seulement (tj multiple de 4, 2 o par élément)
            al = 0
        ptr = f"sh{n} + {'jl' if coop else 'jr'} * {a.dim}"
        if staging == "global":  # alignement des lignes en mémoire globale non garanti
            al = 0
        call_args.append(f"(const T*)__builtin_assume_aligned({ptr}, {al})" if al else ptr)
    if coop:
        L.append(f"        const auto o = {pair_name}({', '.join(call_args)});")
        L.append("        v = o.o0; if (v == v) jv = (kp_lidx)(js + jl);  // NaN ignoré")
        L.append("      }")
        L.append("      kp_coop_batch(lane, gb, v, jv, kv, ki);")
        L.append("    }")
    else:
        L.append("      for (int jr = 0; jr < nt; ++jr) {")
        L.append(f"        const auto o = {pair_name}({', '.join(call_args)});")
    if coop:
        pass
    elif soft:
        pf = "l" if block else ""
        for m_, s_, ts, v, gs in comps:
            L.append("        " + _soft_insert(pf + m_, pf + s_, [pf + t for t in ts], f"o.o{v}",
                                               [f"o.o{g}" for g in gs], kahan))
    else:
        for c in range(d):
            L.append("        " + accumulate(f"o.o{c}", "js + jr", c))
    if not coop:
        L.append("      }")
    if kind is R.SUM and not kahan:  # somme partielle par tuile (schémas direct et block)
        L += [f"      acc{c} += tmp{c};" for c in range(d)]
    if soft and block:
        for comp in comps:
            nm = _soft_state_names(comp, False)
            L.append("      " + _soft_merge(nm, ["l" + x for x in nm]))
    if not coop:
        L.append("    }")
    if staging == "smem":
        L.append("    __syncthreads();")
    L.append("  }")
    L.append("  if (i < M) {")
    L.append("    if (gridDim.y == 1) {")
    L.append(write_final("i"))
    L.append("    } else {")
    L.append(write_partial())
    L.append("    }")
    L.append("  }")
    L.append("}")
    red_end = len(L)
    if getattr(plan, "ranges", False):
        L += _ranges_kernel(L[red_start:], kind, K, d, coop)
        if ranges_tie_break(kind, coop):
            L += _ranges_kernel(L[red_start:red_end], kind, K, d, coop, tie=False)

    # ---------------------------------------------------------------- kp_merge
    L.append(
        'extern "C" __global__ void kp_merge(const kp_idx M, const int S, '
        + ", ".join(out_params + [f"const {vt}* __restrict__ part_val" + ("_" if batched else ""),
                                  "const kp_idx* __restrict__ part_idx" + ("_" if batched else "")]
                    + tail) + ") {"
    )
    L += batch_prologue(None)
    if coop:
        L.append(f"  const int lane = (int)(threadIdx.x % {coop});")
        L.append(f"  const int gb = (int)(threadIdx.x & 31u) - (int)(threadIdx.x & 31u) % {coop};  // 1re lane du groupe")
        L.append(f"  const kp_idx i = ((kp_idx)blockIdx.x * blockDim.x + threadIdx.x) / {coop};")
        L.append(decl_state())
        L.append("  for (int s = 0; s < S; ++s) {")
        L.append("    const kp_idx base = i * S + s;")
        L.append(f"    for (int q0 = 0; q0 < {K}; q0 += {coop}) {{  // liste triée : arrêt au premier lot refusé")
        L.append("      const int r = q0 + lane;")
        L.append("      T v = (T)0; kp_lidx jv = -1;")
        L.append(f"      if (i < M && r < {K}) {{ v = part_val[base * {K} + r]; jv = (kp_lidx)part_idx[base * {K} + r]; }}")
        L.append("      if (!kp_coop_batch(lane, gb, v, jv, kv, ki)) break;")
        L.append("    }")
        L.append("  }")
        L.append("  if (i < M) {")
        L.append(write_final("i"))
        L.append("  }")
        L.append("}")
        return "\n".join(L) + "\n"
    L.append("  const kp_idx i = (kp_idx)blockIdx.x * blockDim.x + threadIdx.x;")
    L.append("  if (i >= M) return;")
    L.append(decl_state())
    L.append("  for (int s = 0; s < S; ++s) {")
    L.append("    const kp_idx base = i * S + s;")
    if soft:
        q = 0
        for comp in comps:
            nm = _soft_state_names(comp, kahan)
            L.append("    " + _soft_merge(nm, [f"part_val[base * {st.n_val} + {q + e}]" for e in range(len(nm))],
                                         kahan))
            q += len(nm)
    elif kind is R.SUM and kahan:
        for c in range(d):
            L.append(f"    kp_neu(acc{c}, kacc{c}, part_val[base * {st.n_val} + {c}]); "
                     f"kacc{c} += part_val[base * {st.n_val} + {d + c}];")
    elif kind in (R.SUM, R.COUNT, R.MIN, R.MAX, R.ARGMIN, R.ARGMAX):
        for c in range(d):
            L.append("    " + accumulate(f"part_val[base * {st.n_val} + {c}]",
                                         f"part_idx[base * {d} + {c}]", c, merge=True))
    else:
        L.append(f"    for (int q = 0; q < {K}; ++q) "
                 + accumulate(f"part_val[base * {K} + q]", f"part_idx[base * {K} + q]", 0, merge=True))
    L.append("  }")
    L.append(write_final("i"))
    L.append("}")
    return "\n".join(L) + "\n"
