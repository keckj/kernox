"""Backend ``sycl`` (Intel XPU via Torch XPU) — contrat v0.

Interop : le kernel est soumis sur la ``sycl::queue`` du stream Torch XPU
courant (``torch.xpu.current_stream(device).sycl_queue``), queue in-order, avec
les pointeurs USM device des tenseurs Torch. Aucune synchronisation hôte,
aucune copie, aucun temporaire global (``temp_bytes_per_launch == 0``).

Le module natif est chargé par ``ctypes`` après ``import torch`` : il partage
l'unique ``libsycl.so.9`` du processus (celle de Torch XPU).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import threading
from ...plan import Plan, ReductionKind, output_shapes
from ..base import CompileInfo, Sizes, Support, UnsupportedError, Variant
from . import codegen, toolchain

_KMIN_KINDS = (ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN)
_ERRLEN = 1024
VARIANT_NAME = "tiled1d_jsplit"


# ``CompileInfo.variant`` est désormais dans base.py ; alias conservé.
SyclCompileInfo = CompileInfo


def _variant(wg=None, tj=0, sg=0, groups=None, grf=0, unroll=1, coop=0) -> Variant:
    """Paramètres : ``wg`` taille de work-group, ``tj`` tuile j demandée (0 =
    heuristique, budget local 32 Kio), ``sg`` taille de sous-groupe imposée
    (0 = choix du compilateur ; 16 ou 32 sur le G31), ``groups`` cible de
    work-groups pour la répartition de l'axe j (paramètre de lancement),
    ``grf`` taille du fichier de registres (0 = défaut, 128, 256), ``unroll``
    déroulage des boucles sur l'état top-K, ``coop`` top-K coopératif : G lanes
    d'un sous-groupe par ligne i (0 = un work-item par ligne)."""
    from .codegen import WORK_GROUP

    params = {"wg": wg or WORK_GROUP, "tj": tj, "sg": sg, "groups": groups or TARGET_GROUPS, "grf": grf,
              "unroll": unroll, "coop": coop}
    return Variant(VARIANT_NAME, tuple(sorted(params.items())))

# Partition de l'axe j : viser au moins TARGET_GROUPS work-groups, chaque tranche
# contenant au moins MIN_TILES_PER_SLICE tuiles ; temporaire borné par TEMP_CAP_BYTES.
TARGET_GROUPS = 512
MIN_TILES_PER_SLICE = 4
TEMP_CAP_BYTES = 256 << 20
BATCH_MIN_CHUNK = 1024  # j par tranche au moins pour suivre le S d'un élément (batch)
INT32_CHUNK = (1 << 31) - 1

# Top-K coopératif (variante ``coop``). Mesures G31 :
# G tel que ceil(K/G) <= COOP_TARGET_SLOTS (K=32 -> 8, 64 -> 16, 128 -> 16) est
# le meilleur ou à moins de 10 % du meilleur dans les 22 cas ; GRF 128 suffit
# tant que L <= COOP_SLOTS (0 spill mesuré), y compris en D = 64.
COOP_K_MIN = 32          # FP32 (mesuré : à K = 16, un work-item par ligne reste meilleur)
COOP_K_MIN_FP64 = 16     # FP64 (mesuré : coop G = 4 à K = 16, ×0,42 à ×0,76)
COOP_TARGET_SLOTS = 4
COOP_SLOTS = 8   # au-delà : GRF 256 (débordements mesurés en GRF 128)


def coop_group(k: int) -> int:
    """G (lanes par ligne, sous-groupe 16) : plus petit G tel que ceil(K/G) <= COOP_TARGET_SLOTS, G <= 16."""
    g = 2
    while g < 16 and -(-k // g) > COOP_TARGET_SLOTS:
        g *= 2
    return g


def coop_grf(plan: Plan, g: int) -> int:
    """GRF 256 si l'état par lane dépasse COOP_SLOTS rangs, sinon défaut (128)."""
    k = plan.reduction.k or 1
    return 256 if -(-k // g) > COOP_SLOTS else 0


def split_j(m: int, n: int, layout, target_groups: int = TARGET_GROUPS, nb: int = 1) -> tuple[int, int]:
    """(S, chunk) : nombre de tranches j et longueur d'une tranche (multiple de TJ).

    Batch (``nb`` > 1) : la cible de groupes porte sur NB·M lignes, mais S ne
    descend pas sous celui d'un élément seul tant que les tranches gardent au
    moins ``BATCH_MIN_CHUNK`` j. Sur NB·M seul, S = 1 dès que NB·M remplit la
    cible : tranches j trop longues, batch natif 8 à 12 % plus lent que la
    boucle d'appels quand chaque élément remplit la carte (G31) ; au
    S d'un élément seul, les petits problèmes paient la fusion (tranches courtes).
    """
    tj = layout.tile_j

    def target_s(rows):
        ngi = -(-rows // layout.rows)
        return min(max(1, -(-target_groups // ngi)), max(1, n // (MIN_TILES_PER_SLICE * tj)))

    s = target_s(nb * m)
    if nb > 1:
        s = max(s, min(target_s(m), max(1, n // BATCH_MIN_CHUNK)))
    if layout.partial_bytes:
        s = min(s, max(1, TEMP_CAP_BYTES // (nb * m * layout.partial_bytes)))
    # l'état du kernel principal stocke les indices en int32 relatifs à la tranche
    s = max(s, -(-n // (INT32_CHUNK - tj)))
    if s <= 1:
        return 1, max(n, 1)
    chunk = -(-n // s)
    chunk = -(-chunk // tj) * tj
    s = -(-n // chunk)
    return s, chunk


def _pair(plan: Plan):
    """Fonction de paire du noyau si livrée, sinon adaptateur local temporaire."""
    try:
        from ...codegen.pair import pair_function  # noyau
    except ImportError:
        from ._pair_local import pair_function
    return pair_function(plan, "sycl")


def _scheme_options(plan: Plan, layout) -> tuple[str, ...]:
    """Contrat v0.7 : schéma de sommation effectif et taille de bloc T publiés.
    ``direct`` et ``block`` partagent l'implémentation (somme par tuile de j)."""
    if plan.reduction.kind not in codegen.FLOAT_ACC_KINDS:
        return ()
    scheme = plan.sum_scheme
    out = (f"sum_scheme={scheme}",)
    if scheme in ("direct", "block"):  # direct : même somme par tuile, T publié aussi
        out += (f"sum_block={layout.tile_j}",)
    return out


def _probe(lib, device_index: int) -> tuple[int, int]:
    """Construit les kernels (JIT du pilote) avant tout lancement ; un refus du
    pilote (p. ex. « total scratch space exceeds HW supported limit ») devient
    ``UnsupportedError`` à la compilation. Renvoie la mémoire privée par
    work-item (principal, fusion), en octets."""
    import torch

    fn = lib.kp_probe
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    priv = (ctypes.c_int64 * 2)()
    err = ctypes.create_string_buffer(_ERRLEN)
    rc = fn(torch.xpu.current_stream(device_index).sycl_queue, priv, err, _ERRLEN)
    if rc != 0:
        raise UnsupportedError("backend sycl : kernel refusé par le pilote : "
                               + err.value.decode(errors="replace"))
    return priv[0], priv[1]


def _device_index(device) -> int:
    import torch

    d = torch.device(device)
    if d.type != "xpu":
        raise UnsupportedError(f"backend sycl : device XPU attendu, reçu {d}")
    return d.index if d.index is not None else torch.xpu.current_device()


class CompiledSyclReduction:
    def __init__(self, plan: Plan, device_index: int, lib, layout, info: CompileInfo,
                 target_groups: int = TARGET_GROUPS):
        self.target_groups = target_groups
        self.plan = plan
        self.device_index = device_index
        self.layout = layout
        # avenant v0.6 §2 : hauteur des tuiles de la liste de travail = lignes i d'un work-group
        self.ranges_tile_m = layout.rows
        self.info = info
        self._lib = lib  # jamais déchargé : des kernels peuvent encore être en vol
        self._fn = lib.kp_launch
        self._fn.restype = ctypes.c_int
        self._fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                             ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                             ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_char_p, ctypes.c_size_t]
        self._nin = len(layout.in_names)
        self._nout = len(layout.out_names)

    def _check(self, inputs: dict, outputs: dict, batch=None) -> tuple[int, int, int]:
        """Revalidation à bas coût de ce dont dépend la sûreté mémoire native.

        ``batch`` (avenant v0.6) : entrées ``(*B, L, d)`` dont seules les deux
        dernières dimensions doivent être contiguës ; l'étendue atteinte par
        les tables d'offsets (lues sur l'hôte) est bornée par le stockage."""
        import torch

        plan = self.plan
        tdt = getattr(torch, plan.dtype)
        dev = torch.device("xpu", self.device_index)
        r = plan.batch_rank
        if bool(r) != (batch is not None):
            raise ValueError(f"plan batch_rank={r} : BatchLayout {'requis' if r else 'inattendu'}")
        nb = 1 if batch is None else int(batch.nb)
        m = n = None
        for arg in plan.args:
            t = inputs[arg.name]
            if t.device != dev or t.dtype != tdt:
                raise ValueError(f"entrée {arg.name} : device/dtype invalides")
            tail = 1 if arg.role == "param" else 2
            if batch is None:
                if not t.is_contiguous():
                    raise ValueError(f"entrée {arg.name} : non contiguë")
            else:
                if t.dim() not in (tail, tail + r) or (arg.role != "param" and t.dim() != tail + r):
                    raise ValueError(f"entrée {arg.name} : rang de batch {t.dim() - tail} != {r}")
                st = t.stride()
                if (arg.dim > 1 and st[-1] != 1) or (tail == 2 and t.shape[-2] > 1 and st[-2] != arg.dim):
                    raise ValueError(f"entrée {arg.name} : deux dernières dimensions non contiguës")
                off = batch.host_offsets[arg.name]
                if off.dtype != torch.int64 or tuple(off.shape) != (nb,):
                    raise ValueError(f"offsets de {arg.name} : int64 ({nb},) attendu")
                dv = batch.offsets[arg.name]
                if dv.device != dev or dv.dtype != torch.int64 or not dv.is_contiguous() or tuple(dv.shape) != (nb,):
                    raise ValueError(f"offsets device de {arg.name} : int64 ({nb},) contigu sur {dev} attendu")
                rows = 1 if arg.role == "param" else t.shape[-2]
                if nb and t.numel():
                    lo, hi = int(off.min()), int(off.max())
                    room = t.untyped_storage().nbytes() // t.element_size() - t.storage_offset()
                    if lo < -t.storage_offset() or hi + rows * arg.dim > room:
                        raise ValueError(f"offsets de {arg.name} hors du stockage")
            shape = tuple(t.shape[-tail:])
            if arg.role == "param":
                if shape != (arg.dim,):
                    raise ValueError(f"param {arg.name} : forme {tuple(t.shape)}, ({arg.dim},) attendu")
                continue
            if shape[1] != arg.dim:
                raise ValueError(f"entrée {arg.name} : forme {tuple(t.shape)}, dim {arg.dim} attendue")
            if arg.role == "i":
                if m is not None and shape[0] != m:
                    raise ValueError("arguments i de longueurs différentes")
                m = shape[0]
            else:
                if n is not None and shape[0] != n:
                    raise ValueError("arguments j de longueurs différentes")
                n = shape[0]
        if m is None:
            m = plan.meta.get("m")
            if m is None:
                raise UnsupportedError("plan sans argument i : M doit être fourni par plan.meta['m']")
        n = 0 if n is None else n
        expected = output_shapes(plan, m, () if batch is None else (nb,))
        for name, (shape, dt) in expected.items():
            t = outputs[name]
            if (tuple(t.shape) != tuple(shape) or t.dtype != getattr(torch, dt)
                    or t.device != dev or not t.is_contiguous()):
                raise ValueError(f"sortie {name} : attendu {shape} {dt} sur {dev} contigu")
        return m, n, nb

    def _check_ranges(self, ranges, m: int, n: int) -> int:
        """Liste de travail (avenant v0.6 §2) : formes, dtypes, device et bornes
        vérifiés sur les copies hôte (sûreté mémoire). Renvoie le nombre de
        lignes couvertes (tuiles disjointes)."""
        import torch

        if not self.plan.ranges:
            raise ValueError("plan sans ranges : RangesLayout inattendu")
        if int(ranges.tile_m) != self.ranges_tile_m:
            raise ValueError(f"ranges : tile_m {ranges.tile_m} != {self.ranges_tile_m}")
        if n >= INT32_CHUNK:
            raise UnsupportedError("ranges : N >= 2^31 non pris en charge (indices j int32)")
        dev = torch.device("xpu", self.device_index)
        hw, hr = ranges.host_work, ranges.host_redranges
        for name, d, h, w in (("work", ranges.work, hw, 4), ("redranges", ranges.redranges, hr, 2)):
            if (d.device != dev or d.dtype != torch.int64 or not d.is_contiguous() or d.dim() != 2
                    or d.shape[1] != w or tuple(h.shape) != tuple(d.shape) or h.dtype != torch.int64):
                raise ValueError(f"ranges : {name} int64 (·, {w}) contigu sur {dev} attendu")
        s = ranges.summary  # calculé une fois par layout (copies hôte) : O(1) par appel
        if s.nw == 0:
            return 0
        if (s.i0_min < 0 or s.i1_max > m or s.h_min < 0 or s.h_max > self.ranges_tile_m
                or s.k0_min < 0 or s.k1_max > s.nr or s.dk_min < 0):
            raise ValueError("ranges : liste de travail hors bornes")
        if s.nr and (s.c_min < 0 or s.d_max > n):
            raise ValueError("ranges : redranges hors de [0, N]")
        return s.covered

    def _fill_neutral(self, outputs: dict) -> None:
        """Neutres de réduction (lignes couvertes par aucune tuile)."""
        kind = self.plan.reduction.kind
        inf = float("inf")
        for name, t in outputs.items():
            if name == "index":
                t.fill_(-1)
            elif kind in (ReductionKind.SUM, ReductionKind.COUNT):
                t.zero_()
            elif kind in (ReductionKind.MIN, ReductionKind.KMIN, ReductionKind.KMIN_ARGKMIN,
                          ReductionKind.MIN_ARGMIN):
                t.fill_(inf)
            elif kind in (ReductionKind.MAX, ReductionKind.LOGSUMEXP, ReductionKind.MAX_ARGMAX,
                          ReductionKind.LOGSUMEXPWEIGHT):
                t.fill_(-inf)
            else:  # SUMSOFTMAXWEIGHT : 0/0
                t.fill_(float("nan"))

    def launch(self, inputs: dict, outputs: dict, batch=None, ranges=None) -> None:
        import torch

        if batch is not None and ranges is not None:
            raise UnsupportedError("ranges et batch combinés : refusé (avenant v0.6)")
        if self.plan.ranges and ranges is None:
            raise ValueError("plan ranges=True : RangesLayout requis")
        m, n, nb = self._check(inputs, outputs, batch)
        if nb == 0 or m == 0 or (self.plan.reduction.kind in _KMIN_KINDS and self.plan.reduction.k == 0):
            return
        if ranges is not None:
            covered = self._check_ranges(ranges, m, n)
            if covered < m:
                self._fill_neutral(outputs)
            nw = int(ranges.work.shape[0])
            if nw == 0 or ranges.redranges.shape[0] == 0:
                if covered and ranges.redranges.shape[0] == 0:
                    self._fill_neutral(outputs)  # aucun intervalle : neutre partout
                return
        queue = torch.xpu.current_stream(self.device_index).sycl_queue
        ins = (ctypes.c_void_p * self._nin)(*[inputs[k].data_ptr() for k in self.layout.in_names])
        outs = (ctypes.c_void_p * self._nout)(*[outputs[k].data_ptr() for k in self.layout.out_names])
        # batch : partition de j d'un élément, temporaire NB·M·S (voir split_j) ;
        # ranges : un work-group par tuile, sans partition de j
        if ranges is not None:
            nsplit, chunk = 1, max(n, 1)
        else:
            nsplit, chunk = split_j(m, n, self.layout, self.target_groups, nb=max(nb, 1))
        if nsplit > 1 and nb * m * nsplit * self.layout.partial_bytes > TEMP_CAP_BYTES:
            raise UnsupportedError("N >= 2^31 avec NB·M trop grand : temporaire au-delà de la borne publiée")
        temp = None
        if nsplit > 1:
            # temporaire alloué par l'allocateur Torch sur le stream courant : sa
            # libération en fin de fonction est ordonnée après les kernels.
            temp = torch.empty(nb * m * nsplit * self.layout.partial_bytes, dtype=torch.uint8,
                               device=torch.device("xpu", self.device_index))
        boff = None
        if batch is not None:
            boff = (ctypes.c_void_p * self._nin)(*[batch.offsets[k].data_ptr() for k in self.layout.in_names])
        self.last_split = (nsplit, chunk, 0 if temp is None else temp.numel())
        err = ctypes.create_string_buffer(_ERRLEN)
        rc = self._fn(queue, m, n, ins, outs, nsplit, chunk,
                      None if temp is None else temp.data_ptr(), nb, boff,
                      0 if ranges is None else nw,
                      None if ranges is None else ranges.work.data_ptr(),
                      None if ranges is None else ranges.redranges.data_ptr(), err, _ERRLEN)
        if rc != 0:
            raise RuntimeError(f"kp_launch (sycl) a échoué [{rc}] : {err.value.decode(errors='replace')}")


class SyclBackend:
    name = "sycl"
    fp_contract_off = True  # contrat v0.3 : fp_contract=False pris en charge (-ffp-contract=off)
    batch = True            # avenant v0.6 §1 : dimensions de batch natives (élément = groupe de la grille)
    ranges = True           # avenant v0.6 §2 : liste de travail, un work-group par tuile (i0, i1, k0, k1)
    sum_schemes = ("direct", "block", "kahan")  # avenant v0.7 (lu par le runtime)

    def __init__(self):
        self._mem: dict = {}
        self._lock = threading.Lock()

    def supports(self, plan: Plan, device) -> Support:
        try:
            import torch
        except ImportError:
            return Support(False, "torch indisponible")
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return Support(False, "torch.xpu indisponible")
        try:
            idx = _device_index(device)
        except UnsupportedError as e:
            return Support(False, str(e))
        if plan.fast_math:
            return Support(False, "fast_math non pris en charge (jamais activé par le backend sycl)")
        if plan.sum_scheme not in ("direct", "block", "kahan"):
            return Support(False, f"sum_scheme inconnu : {plan.sum_scheme!r}")
        if plan.ranges and plan.batch_rank:
            return Support(False, "ranges et dimensions de batch combinés : refusé (avenant v0.6)")
        rank = {"float16": 0, "float32": 1, "float64": 2}
        if plan.dtype not in rank or plan.acc_dtype not in rank:
            return Support(False, f"dtype {plan.dtype}/{plan.acc_dtype} non pris en charge")
        if rank[plan.acc_dtype] < rank[plan.dtype]:
            return Support(False, "acc_dtype inférieur à dtype interdit")
        if plan.dtype == "float16":
            # avenant v0.9 : formule en sycl::half (chaque opération arrondie en
            # binary16), accumulateur float16 ou float32 (float64 refusé)
            if plan.acc_dtype == "float64":
                return Support(False, "dtype float16 : acc_dtype float16 ou float32 uniquement (avenant v0.9)")
            if not getattr(torch.xpu.get_device_properties(idx), "has_fp16", True):
                return Support(False, "float16 refusé : le device ne déclare pas aspect::fp16")
            from ... import _symox_compat

            if not _symox_compat.sycl_half():
                return Support(False, "dtype float16 : symox sans printer sycl::half (SyclDtype::Float16 requis)")
        if plan.acc_dtype != plan.dtype and plan.reduction.kind not in codegen.FLOAT_ACC_KINDS:
            return Support(False, "acc_dtype distinct de dtype : SUM et famille LOGSUMEXP uniquement")
        if "float64" in (plan.dtype, plan.acc_dtype):
            if not torch.xpu.get_device_properties(idx).has_fp64:
                return Support(False, "float64 refusé : le device ne déclare pas aspect::fp64")
        kind, k = plan.reduction.kind, plan.reduction.k
        if kind in (ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT) and plan.out_dim < 2:
            return Support(False, f"{kind.value} : outputs = (f, g_1..g_E) avec E >= 1")
        if kind in _KMIN_KINDS:
            if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                return Support(False, f"K invalide : {k!r}")
            if plan.out_dim != 1:
                return Support(False, "KMIN/ARGKMIN : formule scalaire uniquement (v0)")
            if k > 1024:
                return Support(False, "K > 1024 non pris en charge par la stratégie registres v0")
        elif k is not None:
            return Support(False, "K fourni pour une réduction sans K")
        if plan.out_dim == 0:
            return Support(False, "formule sans sortie")
        if plan.index_dtype != "int64":
            return Support(False, "indices int64 uniquement")
        return Support(True)

    # ------------------------------------------------------------ variantes (v0.2)

    def default_variant(self, plan: Plan, device, sizes: Sizes | None = None) -> Variant:
        """Heuristique déterministe (effort ESTIMATE), indépendante des tailles :
        WG 256, tuile j bornée à 32 Kio locaux, sous-groupe laissé au compilateur,
        répartition de j visant 512 work-groups.

        Top-K de K >= COOP_K_MIN (32 ; 16 en FP64, COOP_K_MIN_FP64) : variante coopérative (sous-groupe 16, G lanes
        par ligne, au plus COOP_TARGET_SLOTS rangs par lane jusqu'à K = 64, sans débordement de
        registres). Seuil, G et GRF mesurés sur G31.
        Top-K de 16 < K < COOP_K_MIN : état déroulé en registres + GRF 256 (choisi
        par mesure A/B sur G31).
        """
        k = plan.reduction.k or 0
        kmin = COOP_K_MIN_FP64 if plan.dtype == "float64" else COOP_K_MIN
        if plan.reduction.kind in _KMIN_KINDS and k >= kmin:
            if plan.dtype == "float32" and 256 < k <= 512 and self._has_sg32(device):
                # mesuré : coop G = 32 (sous-groupe 32, L = 16) ×1,8 à ×2,9 plus rapide que
                # G = 16 à K = 512 en FP32 ; stress de 200 appels sans blocage. Pas en FP64 (×3
                # plus lent) ni à K = 1024 en FP32 (débordements massifs, ×38).
                return _variant(sg=32, coop=32, grf=256)
            return _variant(sg=16, coop=(g := coop_group(k)), grf=coop_grf(plan, g))
        if plan.reduction.kind in _KMIN_KINDS and k > 16:  # 16 < K < COOP_K_MIN
            di = sum(a.dim for a in plan.args_of("i"))
            if (plan.dtype == "float32" and di <= 16) or (plan.dtype == "float64" and di <= 3):
                return _variant(grf=256, unroll=1)
            return _variant(grf=0, unroll=0)
        return _variant()

    @staticmethod
    def _has_sg32(device) -> bool:
        try:
            import torch

            props = torch.xpu.get_device_properties(_device_index(device))
            return 32 in set(int(x) for x in props.sub_group_sizes)
        except Exception:
            return False

    def variants(self, plan: Plan, device, sizes: Sizes | None, effort: str) -> list[Variant]:
        import torch

        idx = _device_index(device)
        props = torch.xpu.get_device_properties(idx)
        max_wg = int(props.max_work_group_size)
        sgs = set(int(x) for x in props.sub_group_sizes)
        size = {"float16": 2, "float32": 4, "float64": 8}[plan.dtype]
        dj = sum(a.dim for a in plan.args_of("j"))
        k = plan.reduction.k or 0
        is_k = plan.reduction.kind in _KMIN_KINDS
        heavy_k = is_k and k > 8
        out = [self.default_variant(plan, device, sizes)]
        if effort not in ("measure", "patient"):
            raise ValueError(f"effort inconnu : {effort!r}")
        cands: list[Variant] = []
        if is_k and k >= 16:  # candidats coop dès K = 16 (seuil par défaut : 16 en FP64, 32 en FP32)
            g0 = coop_group(k)
            r0 = coop_grf(plan, g0)
            g2 = min(16, g0 * 2)
            gh = max(2, g0 // 2)
            if effort == "measure":  # ≤ 4 : coop G/2, 2G, G avec l'autre GRF ; registres
                cands += [_variant(sg=16, coop=gh, grf=coop_grf(plan, gh)),
                          _variant(sg=16, coop=g2, grf=coop_grf(plan, g2)),
                          _variant(sg=16, coop=g0, grf=0 if r0 else 256),
                          _variant(grf=256, unroll=1)]
            else:  # ≤ 32
                cands += [_variant(wg=wg, sg=16, coop=g, grf=gr) for g in (2, 4, 8, 16)
                          for wg in (128, 256) for gr in (0, 256)]
                cands += [_variant(sg=16, coop=g, tj=128) for g in (4, 8, 16)]
                # coop G = 32 (sous-groupe 32) : candidat en FP32 et K <= 512 seulement
                # (mesuré : K = 1024 FP32 en 11,9 s, ×38 ; FP64 K = 512 ×3 plus lent)
                if 32 in sgs and plan.dtype == "float32" and k <= 512:
                    cands += [_variant(wg=256, sg=32, coop=32, grf=256)]
                cands += [_variant(grf=g, unroll=u) for g in (0, 256) for u in (0, 1)]
        else:
            if effort == "measure":
                cands += [_variant(128, 0, 0, 512), _variant(256, 0, 0, 2048)]
                if heavy_k:
                    cands += [_variant(grf=256, unroll=1)]
            else:
                cands += [_variant(wg, tj, sg, 512) for wg in (128, 256, 512) for tj in (128, 256)
                          for sg in (0, 16, 32)]
                cands += [_variant(wg, tj, 0, 2048) for wg in (128, 256, 512) for tj in (128, 256)]
        cap = 4 if effort == "measure" else 32
        for v in cands:
            prm = dict(v.params)
            wg, tj, sg = prm["wg"], prm["tj"], prm["sg"]
            if wg > max_wg or (sg and sg not in sgs):
                continue
            if sg == 32 and heavy_k and not prm["coop"]:  # état top-K volumineux : SIMD32 déborde
                continue
            if tj and dj * size * tj > codegen.LOCAL_MAX_BYTES:
                continue
            if v not in out:
                out.append(v)
        return out[:cap]

    def compile(self, plan: Plan, device, variant: Variant | None = None) -> CompiledSyclReduction:
        import time

        import torch

        sup = self.supports(plan, device)
        if not sup.ok:
            raise UnsupportedError(sup.reason)
        if variant is None:
            variant = self.default_variant(plan, device, None)
        if variant.name != VARIANT_NAME:
            raise UnsupportedError(f"variante sycl inconnue : {variant.name!r}")
        prm = dict(variant.params)
        unknown = set(prm) - {"wg", "tj", "sg", "groups", "grf", "unroll", "coop"}
        if unknown:
            raise UnsupportedError(f"paramètres de variante inconnus : {sorted(unknown)}")
        wg, tj, sg, groups = int(prm.get("wg", 256)), int(prm.get("tj", 0)), int(prm.get("sg", 0)), int(
            prm.get("groups", TARGET_GROUPS))
        idx = _device_index(device)
        props = torch.xpu.get_device_properties(idx)
        if wg < 16 or wg > int(props.max_work_group_size) or wg % 16:
            raise UnsupportedError(f"taille de work-group invalide : {wg}")
        if sg and sg not in set(int(x) for x in props.sub_group_sizes):
            raise UnsupportedError(f"sous-groupe {sg} non supporté par le device")
        grf = int(prm.get("grf", 0))
        if grf not in (0, 128, 256):
            raise UnsupportedError(f"taille de GRF invalide : {grf}")
        if groups < 1 or tj < 0:
            raise UnsupportedError("paramètres de variante invalides")
        target = toolchain.target_for_device_id(props.device_id)
        t0 = time.perf_counter()
        pair = _pair(plan)
        note = json.dumps(variant.as_json(), sort_keys=True)
        # Le runtime SYCL indexe les images device par NOM de kernel pour tout le
        # processus : deux modules chargés avec le même nom de kernel se
        # confondent (constaté sur le G31). Le nom dérive donc du hash de la
        # source complète du kernel (réduction, K, tuiles, variante comprises).
        unroll = int(prm.get("unroll", 1))
        if unroll not in (0, 1):
            raise UnsupportedError(f"unroll invalide : {unroll}")
        coop = int(prm.get("coop", 0))
        if coop:
            if plan.reduction.kind not in _KMIN_KINDS:
                raise UnsupportedError("top-K coopératif : KMIN/ARGKMIN uniquement")
            if not sg:
                raise UnsupportedError("top-K coopératif : sous-groupe imposé (sg) requis")
            if coop not in codegen.COOP_GROUPS or sg % coop or wg % sg:
                raise UnsupportedError(f"groupe coopératif invalide : G={coop}, sg={sg}, wg={wg}")
        kw = dict(wg=wg, tj=tj or None, sg=sg, grf=grf, unroll=bool(unroll), coop=coop, variant_note=note)
        draft, _ = codegen.generate(plan, pair.source, "TAG", **kw)
        tag = hashlib.sha256(draft.encode()).hexdigest()[:24]
        source, layout = codegen.generate(plan, pair.source, tag, **kw)
        mem_key = (hashlib.sha256(source.encode()).hexdigest(), target, bool(plan.fp_contract))
        with self._lock:
            hit = self._mem.get(mem_key)
        if hit is not None:
            lib, build = hit
            cache_hit, secs = True, 0.0
        else:
            build = toolchain.build_shared_object(source, target, bool(plan.fp_contract))
            lib = ctypes.CDLL(str(build.path), mode=ctypes.RTLD_LOCAL)
            _probe(lib, idx)  # UnsupportedError si le pilote refuse les kernels
            with self._lock:
                self._mem[mem_key] = (lib, build)
            cache_hit, secs = build.cache_hit, time.perf_counter() - t0
        info = CompileInfo(
            backend="sycl",
            device_name=f"{props.name} (0x{props.device_id:x})",
            arch=target,
            compiler=build.compiler,
            options=build.options + _scheme_options(plan, layout),
            cache_hit=cache_hit,
            compile_seconds=secs,
            temp_bytes_per_launch=TEMP_CAP_BYTES,  # borne ; valeur exacte : last_split[2]
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            variant=variant,
        )
        return CompiledSyclReduction(plan, idx, lib, layout, info, target_groups=groups)


_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


def get_backend() -> SyclBackend:
    """Point d'entrée du registre de ``kernox.runtime`` (instance partagée)."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = SyclBackend()
        return _INSTANCE


__all__ = ["SyclBackend", "CompiledSyclReduction", "get_backend"]
