"""Génération du kernel SYCL de réduction fusionnée (stratégie 1D tuilée).

Stratégie v0 (inspirée de KeOps ``GpuReduc1D`` @ef72bc9, réécrite ; voir
``THIRD_PARTY.md``) :

- un work-item par ligne ``i`` ; ses arguments ``i`` et les paramètres sont
  copiés en registres (tableaux privés de taille statique) ;
- l'axe ``j`` est parcouru par tuiles de ``TJ`` lignes chargées coopérativement
  (accès coalescés) en mémoire locale de work-group, puis barrière ;
- chaque work-item parcourt la tuile dans l'ordre croissant de ``j`` : l'ordre
  de visite est donc strictement croissant, ce qui donne directement la règle
  « plus petit indice j » pour les égalités ;
- SUM : somme par blocs (accumulateur partiel par tuile ajouté ensuite à
  l'accumulateur principal), erreur d'arrondi en O((TJ + N/TJ)·ε) au lieu de
  O(N·ε) ; même idée que ``sum_scheme="block_sum"`` de KeOps ;
- partition de l'axe j (« split-j ») quand M est petit devant la taille du
  GPU : S tranches de j en parallèle, états partiels dans un temporaire
  alloué par Torch (borné, voir ``__init__.py``), puis kernel de fusion qui
  parcourt les tranches par j croissant (règle du plus petit indice conservée).

- top-K coopératif (variante ``coop=G``, KMIN/ARGKMIN) : G lanes consécutives
  d'un sous-groupe de taille imposée par ligne i ; la liste triée des G·L
  meilleurs (valeur, j) est répartie (L = ceil(K/G) rangs par lane, en
  registres) ; chaque lane évalue un j différent puis les candidats sont
  insérés par mélanges de sous-groupe (``kp_coop_batch``). La fusion des
  tranches j est coopérative elle aussi (lots coalescés de G rangs, arrêt
  d'une tranche dès qu'aucun candidat ne bat le seuil). Même principe que la
  variante ``coop`` du backend HIP (``docs/backend-hip.md``) et que la
  sélection par warp de FAISS (WarpSelect), réécrit ; aucun code repris.

Sémantique (contrat v0) : NaN ignorés par MIN/MAX/ARG*/K*, ±inf valides,
neutres +inf/−inf/−1, COUNT en int64, SUM accumulé en ``acc_dtype``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...plan import Plan, ReductionKind, accumulated_output_dtype, output_shapes

_CTYPE = {"float16": "sycl::half", "float32": "float", "float64": "double"}
_SIZE = {"sycl::half": 2, "float": 4, "double": 8, "int64_t": 8}

WORK_GROUP = 256
COOP_GROUPS = (2, 4, 8, 16, 32)  # lanes par ligne i (top-K coopératif)
LOCAL_BUDGET_BYTES = 32 * 1024  # par work-group : laisse plusieurs WG par Xe-core


@dataclass(frozen=True)
class KernelLayout:
    work_group: int
    tile_j: int
    local_bytes: int
    in_names: tuple[str, ...]      # ordre des pointeurs d'entrée (= plan.args)
    out_names: tuple[str, ...]     # ordre des pointeurs de sortie
    partial_bytes: int             # état partiel par (ligne i, tranche j)
    rows_per_group: int = 0        # lignes i par work-group (0 = work_group ; coop : WG / G)

    @property
    def rows(self) -> int:
        return self.rows_per_group or self.work_group


LOCAL_MAX_BYTES = 64 * 1024  # plafond absolu des variantes (SLM du G31 : 128 Kio)


def _tile_j(plan: Plan, tj_req: int | None = None, budget: int = LOCAL_BUDGET_BYTES) -> tuple[int, int]:
    """Tuile j effective : ``tj_req`` (défaut : WORK_GROUP) bornée par ``budget`` octets."""
    size = _SIZE[_CTYPE[plan.dtype]]
    dj = sum(a.dim for a in plan.args_of("j"))
    want = tj_req or WORK_GROUP
    if dj == 0:
        return want, size  # aucune donnée j : tampon local minimal
    tj = min(want, max(1, budget // (dj * size)))
    if tj >= 16:
        tj -= tj % 16
    return tj, tj * dj * size


def out_names(plan: Plan) -> tuple[str, ...]:
    from ...plan import output_shapes

    return tuple(output_shapes(plan, 0).keys())


SOFT_KINDS = (ReductionKind.LOGSUMEXP, ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT)
ARG_KINDS = (ReductionKind.ARGMIN, ReductionKind.ARGMAX, ReductionKind.MIN_ARGMIN, ReductionKind.MAX_ARGMAX)
FLOAT_ACC_KINDS = (ReductionKind.SUM,) + SOFT_KINDS


def _kahan(plan: Plan) -> bool:
    return getattr(plan, "sum_scheme", "direct") == "kahan" and plan.reduction.kind in FLOAT_ACC_KINDS


def _soft_groups(plan: Plan):
    """Groupes d'état LOGSUMEXP / SUMSOFTMAXWEIGHT (contrat v0.5) : liste de
    (indice de m, expression du score, [(indice de la somme, indice de la
    compensation ou None, poids)]). LSE : un groupe (m, s[, c]) par composante ;
    SSMW : un groupe (m, s[, c], t_1[, c_1], …, t_E[, c_E]) ; poids 1 pour s, g_e pour t_e."""
    k = 2 if _kahan(plan) else 1
    D = plan.out_dim
    if plan.reduction.kind is ReductionKind.LOGSUMEXP:
        out = []
        for c in range(D):
            b = c * (1 + k)
            out.append((b, f"o.o{c}", [(b + 1, b + 2 if k == 2 else None, None)]))
        return out
    sums = []
    for e in range(D):  # e = 0 : s ; e >= 1 : t_e, poids g_e = o.o{e}
        si = 1 + e * k
        sums.append((si, si + 1 if k == 2 else None, None if e == 0 else f"o.o{e}"))
    return [(0, "o.o0", sums)]


def _soft_add(V, si, ci, x):
    return f"kp_add({V}[{si}], {V}[{ci}], {x});" if ci is not None else f"{V}[{si}] += {x};"


def _soft_insert_lines(V, group):
    """Insertion d'un terme (score v, poids g) ; v = −inf ignoré (poids nul, g NaN compris)."""
    m, vexpr, sums = group
    L = [f"{{ const ACC v = static_cast<ACC>({vexpr});",
         "  if (v != -std::numeric_limits<ACC>::infinity()) {",
         f"    if (v > {V}[{m}]) {{ const ACC cf = sycl::exp({V}[{m}] - v);"]
    for si, ci, _w in sums:
        L.append(f"      {V}[{si}] *= cf;" + (f" {V}[{ci}] *= cf;" if ci is not None else ""))
    for si, ci, w in sums:
        L.append("      " + _soft_add(V, si, ci, "ACC(1)" if w is None else f"static_cast<ACC>({w})"))
    L.append(f"      {V}[{m}] = v; }}")
    L.append(f"    else {{ const ACC q = (v == {V}[{m}]) ? ACC(1) : sycl::exp(v - {V}[{m}]);")
    for si, ci, w in sums:
        L.append("      " + _soft_add(V, si, ci, "q" if w is None else f"q * static_cast<ACC>({w})"))
    L.append("    }")
    L.append("  }")
    L.append("}")
    return L


def _soft_merge_lines(V, P, group):
    """Fusion de l'état P dans V (facteur exactement 1 à égalité des maxima ;
    état vide neutre, sans NaN)."""
    m, _v, sums = group
    L = [f"{{ const ACC mb = {P}[{m}];",
         f"  if (mb > {V}[{m}]) {{ const ACC cf = sycl::exp({V}[{m}] - mb);"]
    for si, ci, _w in sums:
        L.append(f"    {V}[{si}] *= cf;" + (f" {V}[{ci}] *= cf;" if ci is not None else ""))
        L.append("    " + (f"kp_kmerge({V}[{si}], {V}[{ci}], {P}[{si}], {P}[{ci}]);" if ci is not None
                           else f"{V}[{si}] += {P}[{si}];"))
    L.append(f"    {V}[{m}] = mb; }}")
    L.append(f"  else {{ const ACC q = (mb == {V}[{m}]) ? ACC(1) : sycl::exp(mb - {V}[{m}]);")
    for si, ci, _w in sums:
        L.append("    " + (f"kp_kmerge({V}[{si}], {V}[{ci}], {P}[{si}] * q, {P}[{ci}] * q);" if ci is not None
                           else f"{V}[{si}] += {P}[{si}] * q;"))
    L.append("  }")
    L.append("}")
    return L


def _soft_finish_lines(plan: Plan, V):
    """LSE : ``s NaN -> NaN ; m = +inf -> +inf ; sinon m + log s`` (vide : −inf) ;
    SSMW : ``m = +inf -> NaN ; sinon t / s`` (vide : 0/0 = NaN)."""
    L = []
    inf = "std::numeric_limits<ACC>::infinity()"

    def tot(si, ci):
        return f"kp_kfin({V}[{si}], {V}[{ci}])" if ci is not None else f"{V}[{si}]"

    groups = _soft_groups(plan)
    if plan.reduction.kind is ReductionKind.LOGSUMEXP:
        for c, (m, _v, sums) in enumerate(groups):
            si, ci, _w = sums[0]
            L.append(f"{{ const ACC s_ = {tot(si, ci)}; out_value[i * D + {c}] = (s_ != s_) ? s_ : "
                     f"(({V}[{m}] == {inf}) ? {V}[{m}] : {V}[{m}] + sycl::log(s_)); }}")
    elif plan.reduction.kind is ReductionKind.SUMSOFTMAXWEIGHT:
        m, _v, sums = groups[0]
        E = len(sums) - 1
        L.append(f"{{ const ACC s_ = {tot(*sums[0][:2])};")
        for e in range(1, E + 1):
            si, ci, _w = sums[e]
            L.append(f"  out_value[i * {E} + {e - 1}] = ({V}[{m}] == {inf}) ? "
                     f"std::numeric_limits<ACC>::quiet_NaN() : {tot(si, ci)} / s_;")
        L.append("}")
    elif plan.reduction.kind is ReductionKind.LOGSUMEXPWEIGHT:
        # avenant v0.8 : m + log(t_e) sans cas particulier (vide : −inf ; f = +inf :
        # +inf si t_e > 0, NaN sinon ; t_e < 0 : NaN)
        m, _v, sums = groups[0]
        E = len(sums) - 1
        for e in range(1, E + 1):
            si, ci, _w = sums[e]
            L.append(f"out_value[i * {E} + {e - 1}] = {V}[{m}] + sycl::log({tot(si, ci)});")
    return L


KAHAN_HELPERS = """
// Sommation compensée de Kahan–Babuška (Neumaier), contrat v0.7 : additions et
// soustractions seulement (-fp-model=precise : ni réassociation ni fast-math).
inline void kp_add(ACC& s, ACC& c, ACC x) {
    const ACC t = s + x;
    c += (sycl::fabs(s) >= sycl::fabs(x)) ? (s - t) + x : (x - t) + s;
    s = t;
}
inline void kp_kmerge(ACC& s, ACC& c, ACC s2, ACC c2) {  // TwoSum(s, s2) + c + c2
    kp_add(s, c, s2);
    c += c2;
}
inline ACC kp_kfin(ACC s, ACC c) { return sycl::isfinite(s) ? s + c : s; }
"""


def state_spec(plan: Plan) -> tuple[int, str, bool]:
    """(largeur W de l'état par ligne i, type C++ des valeurs d'état, présence d'indices)."""
    kind = plan.reduction.kind
    if kind in (ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN):
        return max(plan.reduction.k or 0, 1), _CTYPE[plan.dtype], True
    kahan = _kahan(plan)
    if kind is ReductionKind.SUM:
        return plan.out_dim * (2 if kahan else 1), _CTYPE[plan.acc_dtype], False
    if kind is ReductionKind.LOGSUMEXP:
        return plan.out_dim * (3 if kahan else 2), _CTYPE[plan.acc_dtype], False
    if kind in (ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT):
        return 1 + plan.out_dim * (2 if kahan else 1), _CTYPE[plan.acc_dtype], False
    if kind is ReductionKind.COUNT:
        return plan.out_dim, "int64_t", False
    return plan.out_dim, _CTYPE[plan.dtype], kind in ARG_KINDS


def partial_bytes_per_row(plan: Plan) -> int:
    """Octets d'état partiel par ligne i et par tranche j (partition de l'axe j)."""
    w, vt, has_idx = state_spec(plan)
    size = _SIZE[vt]
    return w * (size + (8 if has_idx else 0))


# Au-delà de ROW_REG_MAX_BYTES, un argument i (ou un paramètre) n'est pas copié
# en registres : la fonction de paire lit directement la ligne en mémoire
# globale (cache L1). Sans cela, D = 990 dépasse l'espace scratch par thread
# du G31 (conformance test_min_argmin_chunked).
ROW_REG_MAX_BYTES = 1024  # D <= 256 (FP32) / 128 (FP64) : registres, comme mesuré sur G31


def _row_lines(arg, n: int, T: str) -> list[str]:
    size = _SIZE[T]
    src = f"in{n} + (active ? i : 0) * {arg.dim}" if arg.role == "i" else f"in{n}"
    if arg.dim * size > ROW_REG_MAX_BYTES:
        return [f"const T* __restrict__ r{n} = {src};  // ligne lue en mémoire globale (grand D)"]
    first = f"active ? in{n}[i * {arg.dim} + c] : T(0)" if arg.role == "i" else f"in{n}[c]"
    return [f"T r{n}[{arg.dim}];", "#pragma unroll", f"for (int c = 0; c < {arg.dim}; ++c) r{n}[c] = {first};"]


COOP_HELPER = r"""
// Top-K coopératif (de type WarpSelect, réécrit) : la liste triée des G·L
// meilleurs (valeur, j) d'une ligne i est répartie sur G lanes consécutives d'un
// sous-groupe, la lane t détenant les rangs [t·L, (t+1)·L) en registres.
// kp_coop_batch insère le lot des G candidats (v, jv) des lanes d'un groupe
// (jv < 0 : pas de candidat). Une insertion décale la liste d'un rang : la
// lane qui reçoit le candidat l'insère, chaque lane suivante reçoit le dernier
// élément de la précédente (shift_group_right). Toutes les opérations de
// sous-groupe sont exécutées par le sous-groupe entier (flot uniforme) ; les
// lignes inactives n'apportent aucun candidat. Ordre lexicographique total
// (valeur, j) : le résultat ne dépend pas de l'ordre d'insertion.
// Renvoie false si aucun candidat du sous-groupe ne bat le seuil de sa ligne.
template <class IT>
inline bool kp_coop_batch(const sycl::sub_group& sg, int lane, int gb, T v, IT jv,
                          T (&kv)[L], IT (&ki)[L]) {
    const T lv0 = sycl::select_from_group(sg, kv[L - 1], gb + G - 1);
    const IT lj0 = sycl::select_from_group(sg, ki[L - 1], gb + G - 1);
    const bool pass = jv >= 0 && kp_less(v, jv, lv0, lj0);
    const sycl::ext::oneapi::sub_group_mask mk = sycl::ext::oneapi::group_ballot(sg, pass);
    uint32_t bits = 0;
    mk.extract_bits(bits);
    if (bits == 0) return false;
    // rangs q (dans le groupe) portant un candidat dans au moins un groupe du
    // sous-groupe : le seuil ne fait que baisser, un candidat refusé au départ
    // le reste
    uint32_t todo = 0;
    #pragma unroll
    for (int g0 = 0; g0 < 32; g0 += G) todo |= bits >> g0;
    if (G < 32) todo &= (1u << (G & 31)) - 1u;
    #pragma unroll 1
    while (todo) {
        const int q = sycl::ctz(todo);
        todo &= todo - 1u;
        const T cv = sycl::select_from_group(sg, v, gb + q);
        const IT cj = sycl::select_from_group(sg, jv, gb + q);
        const T lv = sycl::select_from_group(sg, kv[L - 1], gb + G - 1);
        const IT lj = sycl::select_from_group(sg, ki[L - 1], gb + G - 1);
        const bool ins = cj >= 0 && kp_less(cv, cj, lv, lj);  // uniforme dans le groupe
        const bool aff = ins && kp_less(cv, cj, kv[L - 1], ki[L - 1]);
        const T pv = sycl::shift_group_right(sg, kv[L - 1], 1);
        const IT pj = sycl::shift_group_right(sg, ki[L - 1], 1);
        const int paff = sycl::shift_group_right(sg, static_cast<int>(aff), 1);
        if (aff) {
            const bool fromprev = paff != 0 && lane > 0;
            const T xv = fromprev ? pv : cv;
            const IT xj = fromprev ? pj : cj;
            // insertion sans branchement, indices statiques (état en registres) :
            // c[k] = x < liste[k] est monotone ; liste'[k] = c[k] ? (c[k-1] ? liste[k-1] : x) : liste[k]
            bool c[L];
            #pragma unroll
            for (int k = 0; k < L; ++k) c[k] = kp_less(xv, xj, kv[k], ki[k]);
            #pragma unroll
            for (int k = L - 1; k > 0; --k) {
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


def _out_width(plan: Plan, name: str) -> int:
    shape = output_shapes(plan, 1)[name][0]
    w = 1
    for s in shape[1:]:
        w *= s
    return w


def _batch_lines(plan: Plan, outs, per_group: str, inputs: bool = True) -> list[str]:
    """Plan batché (avenant v0.6) : élément ``kb_`` déduit du groupe, pointeurs
    d'entrée décalés par les tables d'offsets (en éléments), sorties et
    temporaires contigus ``(NB, M, ...)`` / ``(NB, M, S, W)``. ``g`` devient
    l'indice du groupe dans l'élément."""
    if not plan.batch_rank:
        return ["const size_t g = it.get_group(0);"]
    L = ["const size_t bg_ = it.get_group(0);",
         f"const size_t kb_ = bg_ / ({per_group});  // élément de batch",
         f"const size_t g = bg_ % ({per_group});"]
    for n in range(len(plan.args) if inputs else 0):  # fusion : entrées non lues
        L.append(f"const T* __restrict__ in{n} = in{n}_g + bo{n}[kb_];")
    for name in outs:
        L.append(f"auto* __restrict__ out_{name} = out_{name}_g + kb_ * static_cast<size_t>(M) * {_out_width(plan, name)};")
    L.append("VT* __restrict__ tv = tv_g ? tv_g + kb_ * nstate : nullptr;")
    L.append("int64_t* __restrict__ tix = tix_g ? tix_g + kb_ * nstate : nullptr;")
    L.append("(void)tv; (void)tix;")
    return L


RANGES_ROWS = ("const int64_t* __restrict__ wk = kp_work + 4 * g;  // tuile (i0, i1, k0, k1)",)


def _j_loop_open(rng: bool) -> list[str]:
    """Ouverture de la boucle des tuiles de j : tranche [jbeg, jend) ou, avec
    ranges (avenant v0.6 §2), intervalles redranges[k0:k1] de la tuile."""
    if not rng:
        return ["for (int64_t jb = jbeg; jb < jend; jb += TJ) {"]
    return ["for (int64_t kr = wk[2]; kr < wk[3]; ++kr) {  // uniforme dans le work-group",
            "const int64_t jend = kp_red[2 * kr + 1];",
            "for (int64_t jb = kp_red[2 * kr]; jb < jend; jb += TJ) {"]


def _emit_coop(a, plan: Plan, kernel_tag: str, outs, grf: int, sg: int, tj: int, local_bytes: int) -> None:
    """Kernels principal et de fusion du top-K coopératif (G lanes par ligne i)."""
    props = (", sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<"
             + str(grf) + ">}") if grf else ""
    attr = f" [[sycl::reqd_sub_group_size({sg})]]"
    joff, off = {}, 0
    for n, arg in enumerate(plan.args):
        if arg.role == "j":
            joff[n] = off
            off += tj * arg.dim
    T = _CTYPE[plan.dtype]

    rng = bool(plan.ranges)

    def ids(p, main=False):
        a(p + "const auto sgp = it.get_sub_group();")
        a(p + "const int sgl = static_cast<int>(sgp.get_local_linear_id());")
        a(p + "const int lane = sgl % G;")
        a(p + "const int gb = sgl - lane;")
        a(p + "const int lid = static_cast<int>(it.get_local_id(0));")
        if rng and main:
            for line in RANGES_ROWS:
                a(p + line)
            a(p + "const int64_t i = wk[0] + lid / G;")
            a(p + "const bool active = i < wk[1];")
        else:
            a(p + "const int64_t i = static_cast<int64_t>((g % ngi) * R + lid / G);")
            a(p + "const bool active = i < M;")

    def write(p, dst, idx):
        a(p + "#pragma unroll")
        a(p + "for (int k = 0; k < L; ++k) {")
        a(p + "    const int rk = lane * L + k;")
        a(p + "    if (rk < W) {")
        for line in dst(idx):
            a(p + "        " + line)
        a(p + "    }")
        a(p + "}")

    def final_lines(idx):
        r = []
        if "value" in outs:
            r.append("out_value[i * W + rk] = kv[k];")
        if "index" in outs:
            r.append(f"out_index[i * W + rk] = {idx};")
        return r

    glob = "(ki[k] < 0 ? int64_t(-1) : jbeg + int64_t(ki[k]))"
    a("        q.submit([&](sycl::handler& h) {")
    a(f"            sycl::local_accessor<T, 1> tile(sycl::range<1>({max(1, local_bytes // _SIZE[T])}), h);")
    a(f"            h.parallel_for<kp_kernel_{kernel_tag}_main>(sycl::nd_range<1>(GMAIN * WG, WG){props}, [=](sycl::nd_item<1> it){attr} {{")
    p = " " * 16
    for line in _batch_lines(plan, outs, "ngi * S"):
        a(p + line)
    if rng:
        a(p + "const int64_t s = 0;")
        ids(p, main=True)
        a(p + "const int64_t jbeg = 0;  // ranges : indices j globaux (N < 2^31)")
    else:
        a(p + "const int64_t s = static_cast<int64_t>(g / ngi);")
        ids(p)
        a(p + "const int64_t jbeg = s * chunk;")
        a(p + "const int64_t jend = (N - jbeg < chunk) ? N : jbeg + chunk;")
    a(p + "T* tp = tile.get_multi_ptr<sycl::access::decorated::no>().get();")
    for n, arg in enumerate(plan.args):
        if arg.role in ("i", "param"):
            for line in _row_lines(arg, n, T):
                a(p + line)
    a(p + "T kv[L]; int32_t ki[L];  // indices relatifs à la tranche j")
    a(p + "#pragma unroll")
    a(p + "for (int k = 0; k < L; ++k) { kv[k] = std::numeric_limits<T>::infinity(); ki[k] = -1; }")
    for line in _j_loop_open(rng):
        a(p + line)
    a(p + "    const int nt = (jend - jb < TJ) ? static_cast<int>(jend - jb) : TJ;")
    for n, arg in enumerate(plan.args):
        if arg.role == "j":
            a(p + f"    for (int t = lid; t < nt * {arg.dim}; t += WG) tp[{joff[n]} + t] = in{n}[jb * {arg.dim} + t];")
    a(p + "    sycl::group_barrier(it.get_group());")
    a(p + "    #pragma unroll 1  // un seul exemplaire de l'insertion (pression de registres)")
    a(p + "    for (int b0 = 0; b0 < nt; b0 += G) {  // uniforme dans le work-group")
    a(p + "        const int t = b0 + lane;")
    a(p + "        T v = std::numeric_limits<T>::infinity();")
    a(p + "        int32_t jv = -1;")
    call = []
    for n, arg in enumerate(plan.args):
        call.append(f"tp + {joff[n]} + t * {arg.dim}" if arg.role == "j" else f"r{n}")
    a(p + "        if (active && t < nt) {")
    a(p + f"            const kp_out o = kp_pair({', '.join(call)});")
    a(p + "            if (!sycl::isnan(o.o0)) { v = o.o0; jv = static_cast<int32_t>(jb - jbeg) + t; }")
    a(p + "        }")
    a(p + "        kp_coop_batch<int32_t>(sgp, lane, gb, v, jv, kv, ki);")
    a(p + "    }")
    a(p + "    sycl::group_barrier(it.get_group());")
    a(p + "}")
    if rng:
        a(p + "}  // intervalles")
    a(p + "if (!active) return;")
    a(p + "if (S == 1) {")
    write(p + "    ", final_lines, glob)
    a(p + "} else {")
    a(p + "    const size_t base = (static_cast<size_t>(i) * S + s) * W;")
    write(p + "    ", lambda idx: ["tv[base + rk] = kv[k];", f"tix[base + rk] = {idx};"], glob)
    a(p + "}")
    a("            });")
    a("        });")
    # fusion coopérative : les S listes partielles (triées, vides en fin) sont lues
    # par lots coalescés de G rangs et insérées ; une tranche est abandonnée dès
    # qu'aucun candidat du lot ne bat le seuil (listes triées, seuil décroissant).
    a("        if (S > 1) {")
    a(f"            q.parallel_for<kp_kernel_{kernel_tag}_merge>(sycl::nd_range<1>(NB * ngi * WG, WG){props}, [=](sycl::nd_item<1> it){attr} {{")
    p = " " * 16
    for line in _batch_lines(plan, outs, "ngi", inputs=False):
        a(p + line)
    ids(p)
    # indices de l'état de fusion : position int32 p = s·W + rang dans le
    # temporaire de la ligne. À valeur égale, l'ordre des p est celui des j
    # (tranches par j croissant, listes partielles triées par (valeur, j)) ;
    # j est relu dans le temporaire à l'écriture. S·W < 2^31 (temporaire borné).
    a(p + "T kv[L]; int32_t ki[L];  // positions p = s*W + rang (ordre de j à valeur égale)")
    a(p + "#pragma unroll")
    a(p + "for (int k = 0; k < L; ++k) { kv[k] = std::numeric_limits<T>::infinity(); ki[k] = -1; }")
    a(p + "for (int64_t s = 0; s < S; ++s) {")
    a(p + "    const size_t base = (static_cast<size_t>(active ? i : 0) * S + s) * W;")
    a(p + "    const int32_t p0 = static_cast<int32_t>(s * W);")
    a(p + "    #pragma unroll 1  // sinon déroulée (W/G constant) : code et registres multipliés")
    a(p + "    for (int m0 = 0; m0 < W; m0 += G) {")
    a(p + "        const int rk = m0 + lane;")
    a(p + "        T v = std::numeric_limits<T>::infinity();")
    a(p + "        int32_t jv = -1;")
    a(p + "        if (active && rk < W && tix[base + rk] >= 0) { jv = p0 + rk; v = tv[base + rk]; }")
    a(p + "        if (!kp_coop_batch<int32_t>(sgp, lane, gb, v, jv, kv, ki)) break;  // uniforme (sous-groupe)")
    a(p + "    }")
    a(p + "}")
    a(p + "if (!active) return;")
    write(p, final_lines, "(ki[k] < 0 ? int64_t(-1) : tix[static_cast<size_t>(i) * S * W + ki[k]])")
    a("            });")
    a("        }")


def generate(plan: Plan, pair_source: str, kernel_tag: str, *, wg: int = WORK_GROUP,
             tj: int | None = None, sg: int = 0, grf: int = 0, unroll: bool = True,
             coop: int = 0, variant_note: str = "") -> tuple[str, KernelLayout]:
    """Deux kernels dans un module :

    - ``main`` : grille (tranches j) × (groupes i). La tranche ``s`` couvre
      ``[s*chunk, min(N, (s+1)*chunk))``. Si ``S == 1``, écrit directement les
      sorties ; sinon écrit l'état partiel (valeurs, indices) dans ``temp`` ;
    - ``merge`` (si ``S > 1``) : un work-item par ligne i fusionne les S états
      dans l'ordre croissant des tranches (donc des j), puis écrit les sorties.
    """
    kind = plan.reduction.kind
    T = _CTYPE[plan.dtype]
    ACC = _CTYPE[plan.acc_dtype]
    D = plan.out_dim
    tj, local_bytes = _tile_j(plan, tj, LOCAL_MAX_BYTES if tj else LOCAL_BUDGET_BYTES)
    outs = out_names(plan)
    W, VT, has_idx = state_spec(plan)
    is_k = kind in (ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN)
    neg = kind in (ReductionKind.MAX, ReductionKind.ARGMAX, ReductionKind.MAX_ARGMAX)
    soft = kind in SOFT_KINDS
    kahan = _kahan(plan)
    rng = bool(getattr(plan, "ranges", False))
    if rng and plan.batch_rank:
        raise ValueError("ranges et batch combinés : refusé (avenant v0.6)")
    if coop:
        if not is_k:
            raise ValueError("top-K coopératif : KMIN/ARGKMIN uniquement")
        if not sg:
            raise ValueError("top-K coopératif : taille de sous-groupe imposée requise")
        if coop not in COOP_GROUPS or sg % coop or wg % sg:
            raise ValueError(f"groupe coopératif invalide : G={coop}, sg={sg}, wg={wg}")
    G = coop or 1
    Ls = -(-W // G)  # rangs par lane (coop)

    L: list[str] = []
    a = L.append
    a("// Généré par kernox.backends.sycl.codegen, ne pas éditer.")
    a(f"// variante : {variant_note}")
    a("#include <sycl/sycl.hpp>")
    if grf:
        a("#include <sycl/ext/intel/experimental/grf_size_properties.hpp>")
    if coop:
        a("#include <sycl/ext/oneapi/sub_group_mask.hpp>")
    a("#include <cstdint>")
    a("#include <vector>")
    a("#include <cstdio>")
    a("#include <limits>")
    a("")
    a(pair_source)
    a("")
    a("namespace {")
    a(f"using T = {T};")
    a(f"using ACC = {ACC};")
    # sorties accumulées : acc_dtype, sauf FP16 (avenant v0.9 b : conversion finale
    # au plus proche vers binary16, ±inf au-delà de 65504)
    a(f"using OUT = {_CTYPE[accumulated_output_dtype(plan)]};")
    a(f"using VT = {VT};")
    a(f"constexpr int WG = {wg};")
    a(f"constexpr int TJ = {tj};")
    a(f"constexpr int D = {D};")
    a(f"constexpr int W = {W};")
    a(f"constexpr int G = {G};   // lanes par ligne i (1 : un work-item par ligne)")
    a(f"constexpr int L = {Ls};  // rangs top-K par lane")
    a("constexpr int R = WG / G; // lignes i par work-group")
    a(f"class kp_kernel_{kernel_tag}_main;")
    a(f"class kp_kernel_{kernel_tag}_merge;")
    a("template <class IT>")
    a("inline bool kp_less(T a, IT ia, T b, IT ib) {")
    a("    // ordre lexicographique (valeur, j) ; un emplacement vide (ib < 0) est le plus grand")
    a("    return ib < 0 || a < b || (a == b && ia < ib);")
    a("}")
    if coop:
        a(COOP_HELPER)
    if kind in FLOAT_ACC_KINDS:
        a(KAHAN_HELPERS)
    # --- état : initialisation
    a("// IT : type d'indice de l'état. Kernel principal : int32 relatif au début de la")
    a("// tranche j (tranche < 2^31, garanti par la partition) -> moins de registres ;")
    a("// fusion : int64 global. Les sorties sont toujours int64.")
    a("template <class IT>")
    a("struct kp_state {")
    a("    VT v[W];")
    if has_idx:
        a("    IT ix[W];")
    a("    inline void init() {")
    a("        #pragma unroll")
    a("        for (int c = 0; c < W; ++c) {")
    if kind in (ReductionKind.SUM, ReductionKind.COUNT) or soft:
        a("            v[c] = VT(0);")
    else:
        a(f"            v[c] = {'-' if neg else ''}std::numeric_limits<T>::infinity();")
    if has_idx:
        a("            ix[c] = -1;")
    a("        }")
    if soft:
        for m, _v, _s in _soft_groups(plan):
            a(f"        v[{m}] = -std::numeric_limits<VT>::infinity();  // m : état vide")
    a("    }")
    # insertion top-K (utilisée par update et merge)
    if is_k:
        a("    inline void insert(T cv, IT ci) {")
        a("        if (!kp_less(cv, ci, v[W - 1], ix[W - 1])) return;")
        a("        #pragma unroll")
        a("        for (int k = 0; k < W; ++k) {")
        a("            if (kp_less(cv, ci, v[k], ix[k])) {")
        a("                const T tv = v[k]; const IT ti = ix[k];")
        a("                v[k] = cv; ix[k] = ci; cv = tv; ci = ti;")
        a("            }")
        a("        }")
        a("    }")
    # fusion d'un état partiel (valeurs pv, indices pi), tranches traitées par j croissant
    a("    inline void merge(const VT* pv, const int64_t* pi) {")
    a("        (void)pi;")
    if is_k:
        a("        for (int k = 0; k < W; ++k) { if (pi[k] < 0) break; insert(pv[k], pi[k]); }")
    elif soft:
        for grp in _soft_groups(plan):
            for line in _soft_merge_lines("v", "pv", grp):
                a("        " + line)
    elif kind is ReductionKind.SUM and kahan:
        a("        #pragma unroll")
        a("        for (int c = 0; c < D; ++c) kp_kmerge(v[c], v[D + c], pv[c], pv[D + c]);")
    elif kind in (ReductionKind.SUM, ReductionKind.COUNT):
        a("        #pragma unroll")
        a("        for (int c = 0; c < W; ++c) v[c] += pv[c];")
    elif kind in (ReductionKind.MIN, ReductionKind.MAX):
        op = ">" if neg else "<"
        a("        #pragma unroll")
        a(f"        for (int c = 0; c < W; ++c) if (pv[c] {op} v[c]) v[c] = pv[c];")
    else:
        op = ">" if neg else "<"
        a("        #pragma unroll")
        a(f"        for (int c = 0; c < W; ++c) if (pi[c] >= 0 && (ix[c] < 0 || pv[c] {op} v[c])) {{ v[c] = pv[c]; ix[c] = pi[c]; }}")
    a("    }")
    a("};")
    a("}  // namespace")
    a("")
    a('extern "C" int kp_launch(void* kp_queue, int64_t M, int64_t N, void* const* kp_in,')
    a("                         void* const* kp_out_ptrs, int64_t S, int64_t chunk, void* kp_temp,")
    a("                         int64_t NB, const int64_t* const* kp_boff,")
    a("                         int64_t NW, const int64_t* kp_work, const int64_t* kp_red,")
    a("                         char* kp_err, size_t kp_errlen)")
    a("{")
    a("    try {")
    a("        sycl::queue& q = *static_cast<sycl::queue*>(kp_queue);")
    a("        if (!q.is_in_order()) {")
    a('            std::snprintf(kp_err, kp_errlen, "la queue Torch XPU doit être in-order");')
    a("            return 2;")
    a("        }")
    if "sycl::half" in (T, ACC):
        a("        if (!q.get_device().has(sycl::aspect::fp16)) {")
        a('            std::snprintf(kp_err, kp_errlen, "float16 demandé sur un device sans aspect::fp16");')
        a("            return 3;")
        a("        }")
    if "double" in (T, ACC):
        a("        if (!q.get_device().has(sycl::aspect::fp64)) {")
        a('            std::snprintf(kp_err, kp_errlen, "float64 demandé sur un device sans aspect::fp64");')
        a("            return 3;")
        a("        }")
    bsuf = "_g" if plan.batch_rank else ""
    if plan.batch_rank:
        a("        if (NB < 1 || kp_boff == nullptr) {")
        a('            std::snprintf(kp_err, kp_errlen, "batch invalide (NB, tables d\'offsets)");')
        a("            return 5;")
        a("        }")
        for n in range(len(plan.args)):
            a(f"        const int64_t* bo{n} = kp_boff[{n}];")
    else:
        a("        if (NB != 1) {")
        a('            std::snprintf(kp_err, kp_errlen, "plan sans batch : NB = 1 attendu");')
        a("            return 5;")
        a("        }")
    a("        if (S < 1 || (S > 1 && kp_temp == nullptr) || chunk < 1) {")
    a('            std::snprintf(kp_err, kp_errlen, "partition j invalide");')
    a("            return 4;")
    a("        }")
    for n, arg in enumerate(plan.args):
        a(f"        const T* __restrict__ in{n}{bsuf} = static_cast<const T*>(kp_in[{n}]);")
    for n, name in enumerate(outs):
        ctype = "int64_t" if name == "index" or kind is ReductionKind.COUNT else (
            "OUT" if kind in FLOAT_ACC_KINDS else "T")
        a(f"        {ctype}* __restrict__ out_{name}{bsuf} = static_cast<{ctype}*>(kp_out_ptrs[{n}]);")
    # partiels : indices (int64) d'abord puis valeurs, tous deux alignés sur 8
    a("        const size_t nstate = static_cast<size_t>(M) * static_cast<size_t>(S) * W;")
    # batch : temporaires (NB, M, S, W) ; nstate reste la taille d'un élément
    if has_idx:
        a(f"        int64_t* __restrict__ tix{bsuf} = static_cast<int64_t*>(kp_temp);")
        a(f"        VT* __restrict__ tv{bsuf} = reinterpret_cast<VT*>(tix{bsuf} + (kp_temp ? NB * nstate : 0));")
    else:
        a(f"        int64_t* __restrict__ tix{bsuf} = nullptr;")
        a(f"        VT* __restrict__ tv{bsuf} = static_cast<VT*>(kp_temp);")
    a("        (void)nstate;")
    a("        if (M <= 0) return 0;")
    a("        const size_t ngi = (static_cast<size_t>(M) + R - 1) / R;  // groupes de lignes i")
    if rng:
        a("        if (S != 1 || NW < 0 || (NW > 0 && (kp_work == nullptr || kp_red == nullptr))) {")
        a('            std::snprintf(kp_err, kp_errlen, "ranges : liste de travail invalide");')
        a("            return 6;")
        a("        }")
        a("        if (NW == 0) return 0;")
        a("        const size_t GMAIN = static_cast<size_t>(NW);  // un work-group par tuile de la liste")
    else:
        a("        (void)NW; (void)kp_work; (void)kp_red;")
        a("        const size_t GMAIN = static_cast<size_t>(NB) * ngi * static_cast<size_t>(S);")

    def write_final(indent, idx):
        p = " " * indent
        if soft:  # indices déjà statiques : pas de boucle
            for line in _soft_finish_lines(plan, "st.v"):
                a(p + line)
            return
        a(p + "#pragma unroll  // indices statiques : l'état reste en registres")
        if kind is ReductionKind.SUM and kahan:
            a(p + "for (int c = 0; c < D; ++c) out_value[i * D + c] = kp_kfin(st.v[c], st.v[D + c]);")
        elif kind in (ReductionKind.SUM, ReductionKind.COUNT, ReductionKind.MIN, ReductionKind.MAX):
            a(p + "for (int c = 0; c < D; ++c) out_value[i * D + c] = st.v[c];")
        elif kind in ARG_KINDS:
            a(p + "for (int c = 0; c < D; ++c) {")
            if "value" in outs:  # MIN_ARGMIN / MAX_ARGMAX (avenant v0.8) : même passage
                a(p + "    out_value[i * D + c] = st.v[c];")
            a(p + f"    out_index[i * D + c] = {idx('c')};")
            a(p + "}")
        else:
            a(p + "for (int k = 0; k < W; ++k) {")
            if "value" in outs:
                a(p + "    out_value[i * W + k] = st.v[k];")
            if "index" in outs:
                a(p + f"    out_index[i * W + k] = {idx('k')};")
            a(p + "}")

    if coop:
        _emit_coop(a, plan, kernel_tag, outs, grf, sg, tj, local_bytes)
    else:
        a("        q.submit([&](sycl::handler& h) {")
        a(f"            sycl::local_accessor<T, 1> tile(sycl::range<1>({max(1, local_bytes // _SIZE[T])}), h);")
        a(f"            h.parallel_for<kp_kernel_{kernel_tag}_main>(sycl::nd_range<1>(GMAIN * WG, WG){', sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<' + str(grf) + '>}' if grf else ''}, [=](sycl::nd_item<1> it){' [[sycl::reqd_sub_group_size(' + str(sg) + ')]]' if sg else ''} {{")
        for line in _batch_lines(plan, outs, "ngi * S"):
            a("                " + line)
        if rng:
            a("                const int64_t s = 0;")
            for line in RANGES_ROWS:
                a("                " + line)
            a("                const int64_t i = wk[0] + static_cast<int64_t>(it.get_local_id(0));")
        else:
            a("                const int64_t s = static_cast<int64_t>(g / ngi);")
            a("                const int64_t i = static_cast<int64_t>((g % ngi) * WG + it.get_local_id(0));")
        a("                const int lid = static_cast<int>(it.get_local_id(0));")
        if rng:
            a("                const bool active = i < wk[1];")
            a("                const int64_t jbeg = 0;  // ranges : indices j globaux (N < 2^31)")
        else:
            a("                const bool active = i < M;")
            a("                const int64_t jbeg = s * chunk;")
            a("                const int64_t jend = (N - jbeg < chunk) ? N : jbeg + chunk;")
        a("                T* tp = tile.get_multi_ptr<sycl::access::decorated::no>().get();")
        for n, arg in enumerate(plan.args):
            if arg.role in ("i", "param"):
                for line in _row_lines(arg, n, T):
                    a("                " + line)
        a(f"                kp_state<{'int32_t' if has_idx else 'int64_t'}> st; st.init();")
        off = 0
        joff = {}
        for n, arg in enumerate(plan.args):
            if arg.role == "j":
                joff[n] = off
                off += tj * arg.dim
        for line in _j_loop_open(rng):
            a("                " + line)
        a("                    const int nt = (jend - jb < TJ) ? static_cast<int>(jend - jb) : TJ;")
        for n, arg in enumerate(plan.args):
            if arg.role == "j":
                a(f"                    for (int t = lid; t < nt * {arg.dim}; t += WG) tp[{joff[n]} + t] = in{n}[jb * {arg.dim} + t];")
        a("                    sycl::group_barrier(it.get_group());")
        a("                    if (active) {")
        if kind is ReductionKind.SUM and not kahan:
            a("                        ACC tacc[D];  // somme par blocs (par tuile)")
            a("                        #pragma unroll")
            a("                        for (int c = 0; c < D; ++c) tacc[c] = ACC(0);")
        if soft and not kahan:
            a("                        kp_state<int64_t> tst; tst.init();  // état de tuile (sommation par blocs)")
        a("                        for (int t = 0; t < nt; ++t) {")
        a("                            const int32_t jl = static_cast<int32_t>(jb - jbeg) + t;  // relatif à la tranche")
        a("                            (void)jl;")
        call = []
        for n, arg in enumerate(plan.args):
            call.append(f"tp + {joff[n]} + t * {arg.dim}" if arg.role == "j" else f"r{n}")
        a(f"                            const kp_out o = kp_pair({', '.join(call)});")
        if kind is ReductionKind.SUM and kahan:
            for c in range(D):
                a(f"                            kp_add(st.v[{c}], st.v[{D + c}], static_cast<ACC>(o.o{c}));")
        elif kind is ReductionKind.SUM:
            for c in range(D):
                a(f"                            tacc[{c}] += static_cast<ACC>(o.o{c});")
        elif soft:
            for grp in _soft_groups(plan):
                for line in _soft_insert_lines("st.v" if kahan else "tst.v", grp):
                    a("                            " + line)
        elif kind is ReductionKind.COUNT:
            for c in range(D):
                a(f"                            st.v[{c}] += o.o{c} ? 1 : 0;")
        elif kind in (ReductionKind.MIN, ReductionKind.MAX):
            op = ">" if neg else "<"
            for c in range(D):
                a(f"                            if (o.o{c} {op} st.v[{c}]) st.v[{c}] = o.o{c};  // NaN : comparaison fausse, ignoré")
        elif kind in ARG_KINDS:
            op = ">" if neg else "<"
            for c in range(D):
                # ranges : intervalles dans un ordre quelconque -> égalité départagée par le plus petit j
                tie = f" || (o.o{c} == st.v[{c}] && jl < st.ix[{c}])" if rng else ""
                a(f"                            if (!sycl::isnan(o.o{c}) && (st.ix[{c}] < 0 || o.o{c} {op} st.v[{c}]{tie})) {{ st.v[{c}] = o.o{c}; st.ix[{c}] = jl; }}")
        else:
            a("                            if (!sycl::isnan(o.o0)) st.insert(o.o0, jl);")
        a("                        }")
        if kind is ReductionKind.SUM and not kahan:
            a("                        #pragma unroll")
            a("                        for (int c = 0; c < D; ++c) st.v[c] += tacc[c];")
        if soft and not kahan:
            a("                        st.merge(tst.v, nullptr);")
        a("                    }")
        a("                    sycl::group_barrier(it.get_group());")
        a("                }")
        if rng:
            a("                }  // intervalles")
        a("                if (!active) return;")
        def glob(c):
            return f"(st.ix[{c}] < 0 ? int64_t(-1) : jbeg + int64_t(st.ix[{c}]))"

        a("                if (S == 1) {")
        write_final(20, glob)
        a("                } else {")
        a("                    const size_t base = (static_cast<size_t>(i) * S + s) * W;")
        a("                    #pragma unroll")
        a("                    for (int c = 0; c < W; ++c) tv[base + c] = st.v[c];")
        if has_idx:
            a("                    #pragma unroll")
            a(f"                    for (int c = 0; c < W; ++c) tix[base + c] = {glob('c')};")
        a("                }")
        a("            });")
        a("        });")
        a("        if (S > 1) {")
        a(f"            q.parallel_for<kp_kernel_{kernel_tag}_merge>(sycl::nd_range<1>(NB * ngi * WG, WG), [=](sycl::nd_item<1> it) {{")
        for line in _batch_lines(plan, outs, "ngi", inputs=False):
            a("                " + line)
        a("                const int64_t i = static_cast<int64_t>(g * WG + it.get_local_id(0));")
        a("                if (i >= M) return;")
        a("                kp_state<int64_t> st; st.init();")
        a("                for (int64_t s = 0; s < S; ++s) {")
        a("                    const size_t base = (static_cast<size_t>(i) * S + s) * W;")
        a(f"                    st.merge(tv + base, {'tix + base' if has_idx else 'nullptr'});")
        a("                }")
        write_final(16, lambda c: f"st.ix[{c}]")
        a("            });")
        a("        }")
    a("        return 0;")
    a("    } catch (const sycl::exception& e) {")
    a('        std::snprintf(kp_err, kp_errlen, "sycl::exception: %s", e.what());')
    a("        return 1;")
    a("    } catch (const std::exception& e) {")
    a('        std::snprintf(kp_err, kp_errlen, "std::exception: %s", e.what());')
    a("        return 1;")
    a("    }")
    a("}")
    a("")
    a("// Construction JIT des kernels avant tout lancement : un échec de")
    a("// compilation du pilote (p. ex. espace scratch par thread dépassé) est")
    a("// remonté à compile() ; renvoie la mémoire privée par work-item.")
    a('extern "C" int kp_probe(void* kp_queue, int64_t* kp_priv, char* kp_err, size_t kp_errlen)')
    a("{")
    a("    try {")
    a("        sycl::queue& q = *static_cast<sycl::queue*>(kp_queue);")
    a("        const sycl::device dev = q.get_device();")
    a(f"        const std::vector<sycl::kernel_id> ids{{sycl::get_kernel_id<kp_kernel_{kernel_tag}_main>(),")
    a(f"                                               sycl::get_kernel_id<kp_kernel_{kernel_tag}_merge>()}};")
    a("        auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {dev}, ids);")
    a("        for (int n = 0; n < 2; ++n)")
    a("            kp_priv[n] = static_cast<int64_t>(kb.get_kernel(ids[n]).get_info<")
    a("                sycl::info::kernel_device_specific::private_mem_size>(dev));")
    a("        return 0;")
    a("    } catch (const std::exception& e) {")
    a('        std::snprintf(kp_err, kp_errlen, "%s", e.what());')
    a("        return 1;")
    a("    }")
    a("}")
    if not unroll:
        # boucles sur l'état de largeur W non déroulées : code compact, état en
        # mémoire privée (scratch) ; préférable à très grand K (mesuré sur le G31)
        for n in range(len(L) - 1):
            if L[n].strip() == "#pragma unroll" and "< W" in L[n + 1]:
                L[n] = L[n].replace("#pragma unroll", "#pragma unroll 1")
    layout = KernelLayout(wg, tj, local_bytes, tuple(x.name for x in plan.args), outs,
                          partial_bytes_per_row(plan), wg // G)
    return "\n".join(L) + "\n", layout
