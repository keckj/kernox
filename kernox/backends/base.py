"""Common backend interface (contract v0).

A backend only receives Torch tensors already validated by the core
(``kernox.runtime``): right device, right dtype, contiguous, conforming shapes.
It cheaply re-validates whatever native memory safety depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

from ..plan import Plan


class UnsupportedError(NotImplementedError):
    """Operation explicitly not supported (never a silent fallback)."""


@dataclass(frozen=True)
class Support:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class CompileInfo:
    """Publié pour diagnostics et benchmarks."""

    backend: str              # "cuda" | "hip" | "sycl"
    device_name: str
    arch: str                 # sm_75, gfx1201, intel_gpu_bmg_g31...
    compiler: str             # nom + version exacte
    options: tuple[str, ...]
    cache_hit: bool
    compile_seconds: float
    temp_bytes_per_launch: int  # buffers temporaires bornés, publiés
    source_sha256: str
    variant: "Variant | None" = None  # variante effective (contrat v0.2)


class CompiledReduction(Protocol):
    info: CompileInfo

    def launch(self, inputs: dict, outputs: dict) -> None:
        """Lance la réduction de façon asynchrone sur le stream/queue Torch courant
        du device des tenseurs.

        ``inputs`` : nom d'argument -> torch.Tensor (i: (M,d), j: (N,d),
        param: (d,)), tous sur le même device, dtype ``plan.dtype``, contigus.
        ``outputs`` : clés de ``plan.output_shapes`` -> tenseurs préalloués par le
        noyau. Aucune synchronisation hôte, aucune copie CPU. Les temporaires
        sont alloués via l'allocateur Torch du même device et stream.
        Cas M == 0 : retour immédiat. Cas N == 0 : sorties = éléments neutres.

        Avenant v0.6 (plans ``batch_rank > 0``, backends déclarant ``batch``) :
        appel ``launch(inputs, outputs, batch=BatchLayout)``. Les entrées gardent
        leur forme ``(*Ba, L, d)`` / ``(*Bp, d)`` et leurs strides de batch
        (quelconques, 0 compris) ; leurs deux dernières dimensions sont
        contiguës. L'élément ``b`` d'un argument commence à
        ``data_ptr + batch.offsets[nom][b] * itemsize``. Les sorties sont
        contiguës ``(nb, M, …)``. Sans batch, ``batch`` n'est pas passé.
        """


@dataclass(frozen=True)
class Sizes:
    """Tailles d'un appel, utilisées pour choisir une variante (contrat v0.2).

    ``nb`` : nombre d'éléments de batch (avenant v0.6), 1 sans batch.
    """

    m: int
    n: int
    nb: int = 1


@dataclass(frozen=True, eq=False)
class BatchLayout:
    """Batch aplati d'un appel (avenant v0.6), construit et mis en cache par le runtime.

    ``shape`` : forme de batch diffusée ``B`` ; ``nb = prod(B)``.
    ``offsets`` : nom d'argument -> tenseur int64 ``(nb,)`` sur le device des
    entrées, offset en éléments du début de l'élément ``b`` relativement à
    ``data_ptr`` (diffusion = offsets répétés, param sans batch = zéros).
    ``host_offsets`` : les mêmes tables sur l'hôte (CPU), pour les backends de
    référence ; un backend natif ne les lit pas pendant ``launch``.
    """

    shape: tuple[int, ...]
    nb: int
    offsets: dict
    host_offsets: dict


class RangesLayout:
    """Ranges block-sparse d'un appel (avenant v0.6 §2), normalisés par le runtime.

    Orientation du plan (réduction sur j). ``work`` : int64 ``(W, 4)`` sur le
    device, tuiles ``(i0, i1, k0, k1)`` de hauteur ``i1 − i0 ≤ tile_m`` ;
    ``redranges`` : int64 ``(R, 2)`` sur le device, intervalles ``[c, d)`` de j ;
    la tuile réduit ses lignes sur ``∪ redranges[k0:k1]``. Les lignes
    couvertes par aucune tuile reçoivent le neutre (à la charge de ``launch``).
    ``host_work`` / ``host_redranges`` : copies hôte pour les backends de
    référence, faites à la première lecture si elles ne sont pas fournies
    (les backends GPU n'en ont pas besoin : une copie de R × 2 entiers coûte
    plusieurs ms à R ~ 10⁶). ``tile_m`` : valeur publiée par l'objet compilé
    (``ranges_tile_m``). Égalité et hachage par identité.
    """

    def __init__(self, work, redranges, tile_m, host_work=None, host_redranges=None):
        self.work, self.redranges, self.tile_m = work, redranges, tile_m
        self._host_work, self._host_redranges = host_work, host_redranges

    @property
    def host_work(self):
        if self._host_work is None:
            self._host_work = self.work.cpu()
        return self._host_work

    @property
    def host_redranges(self):
        if self._host_redranges is None:
            self._host_redranges = self.redranges.cpu()
        return self._host_redranges

    @cached_property
    def summary(self) -> "RangesSummary":
        """Résumé des bornes, calculé une fois par layout (sur le device, un seul
        transfert de 15 entiers ; sur l'hôte pour XPU) : les contrôles de sûreté
        mémoire des backends et du runtime sont en O(1) par appel."""
        import torch

        if self._host_work is not None or torch.device(self.work.device).type in ("cpu", "xpu"):
            return RangesSummary.of(self.host_work, self.host_redranges)
        return RangesSummary.of(self.work, self.redranges)


@dataclass(frozen=True)
class RangesSummary:
    """Bornes d'un :class:`RangesLayout` (toutes les entrées, puis entrées non vides).

    ``i0_min``, ``i1_max``, ``h_min``/``h_max`` (hauteur ``i1 − i0``), ``k0_min``,
    ``k1_max``, ``dk_min`` (``k1 − k0``), ``c_min``, ``d_max`` : sur toutes les
    entrées ; ``live_*`` : tuiles ``i1 > i0`` et intervalles ``d > c`` seulement
    (ceux que parcourt un noyau). ``covered`` : Σ (i1 − i0).
    """

    nw: int
    nr: int
    i0_min: int = 0
    i1_max: int = 0
    h_min: int = 0
    h_max: int = 0
    k0_min: int = 0
    k1_max: int = 0
    dk_min: int = 0
    c_min: int = 0
    d_max: int = 0
    live_i0_min: int = 0
    live_i1_max: int = 0
    live_c_min: int = 0
    live_d_max: int = 0
    covered: int = 0

    @staticmethod
    def of(work, red) -> "RangesSummary":
        import torch

        nw, nr = int(work.shape[0]), int(red.shape[0])
        names, vals = [], []
        if nw:
            i0, i1, k0, k1 = work.to(torch.int64).unbind(1)
            h = i1 - i0
            live = h > 0
            z = torch.zeros_like(i0)
            names += ["i0_min", "i1_max", "h_min", "h_max", "k0_min", "k1_max", "dk_min",
                      "live_i0_min", "live_i1_max", "covered"]
            vals += [i0.min(), i1.max(), h.min(), h.max(), k0.min(), k1.max(), (k1 - k0).min(),
                     torch.where(live, i0, z).min(), torch.where(live, i1, z).max(), h.sum()]
        if nr:
            c, d = red.to(torch.int64).unbind(1)
            live = d > c
            z = torch.zeros_like(c)
            names += ["c_min", "d_max", "live_c_min", "live_d_max"]
            vals += [c.min(), d.max(), torch.where(live, c, z).min(), torch.where(live, d, z).max()]
        kw = dict(zip(names, torch.stack(vals).tolist())) if vals else {}  # un seul transfert
        return RangesSummary(nw, nr, **kw)


@dataclass(frozen=True)
class Variant:
    """Variante de kernel d'un backend (stratégie + paramètres), contrat v0.2.

    ``name`` : stratégie (ex. ``"row"``, ``"jsplit"``, ``"coop_topk"``).
    ``params`` : paires (clé, valeur int|str) triées, sérialisables en JSON.
    Toutes les variantes d'un plan respectent la même sémantique ; seules les
    sommes flottantes peuvent différer par l'ordre de sommation (dans la
    tolérance des tests). Une variante ne change jamais dtype, accumulateur,
    fast-math ni contraction FMA.
    """

    name: str
    params: tuple[tuple[str, int | str], ...] = ()

    def as_json(self) -> dict:
        return {"name": self.name, "params": dict(self.params)}


class Backend(Protocol):
    name: str
    #: contrat v0.3 : True si le backend sait compiler sans aucune contraction
    #: FMA (``plan.fp_contract == False``). Attribut optionnel, absent = False :
    #: le runtime refuse alors un plan ``fp_contract=False`` (UnsupportedError).
    fp_contract_off: bool
    #: avenant v0.7 : schémas de sommation implémentés (``"direct"``, ``"block"``,
    #: ``"kahan"``). Attribut optionnel, absent = ``("direct",)`` : le runtime
    #: refuse alors tout autre schéma pour SUM/LOGSUMEXP/SUMSOFTMAXWEIGHT
    #: (UnsupportedError), en plus de ``supports``. Un backend qui implémente
    #: ``block`` publie sa taille de tuile ``T`` dans ``CompileInfo.options``
    #: sous la forme ``"sum_block=T"``.
    sum_schemes: tuple
    #: avenant v0.6 : True si ``launch`` accepte ``batch=BatchLayout`` pour les
    #: plans ``batch_rank > 0``. Attribut optionnel, absent = False : le runtime
    #: refuse alors tout appel batché (UnsupportedError), en plus de ``supports``.
    batch: bool
    #: avenant v0.6 §2 : True si ``launch`` accepte ``ranges=RangesLayout`` pour
    #: les plans ``ranges=True`` ; l'objet compilé publie alors ``ranges_tile_m``.
    #: Absent = False : le runtime refuse tout appel avec ranges.
    ranges: bool

    def supports(self, plan: Plan, device) -> Support: ...

    def default_variant(self, plan: Plan, device, sizes: Sizes) -> Variant:
        """Choix heuristique sans mesure (effort ESTIMATE). Déterministe."""

    def variants(self, plan: Plan, device, sizes: Sizes, effort: str) -> list[Variant]:
        """Candidats valides pour l'autotuning, ``default_variant`` en tête.

        ``effort == "measure"`` : petit ensemble ciblé (≤ 4) ;
        ``effort == "patient"`` : espace plus large (≤ 32). Les candidats
        invalides pour le device (mémoire partagée, registres...) sont exclus.
        """

    def compile(self, plan: Plan, device, variant: Variant | None = None) -> CompiledReduction:
        """Compile (ou charge depuis le cache atomique) le kernel du plan.

        ``variant=None`` équivaut à la variante par défaut du backend ; la
        variante fait partie de la clé de cache. ``CompiledReduction.info``
        expose la variante effective (attribut ``variant``).
        """
