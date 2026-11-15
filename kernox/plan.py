"""Plan typé : contrat commun entre le frontend kernox et les backends.

Statut : contrat v0 (implémenté comme structure de données, sémantique décrite
dans ``docs/kernel-backend-contract.md``). Les backends ne consomment QUE ce
module ; ils ne dépendent jamais des classes du frontend (``Vi``, ``Vj``...).

Normalisation : un plan réduit toujours sur l'axe ``j``. Une réduction
« over i » est convertie par le frontend en échangeant les rôles i/j.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

DType = Literal["float16", "float32", "float64"]
SumScheme = Literal["direct", "block", "kahan"]
SUM_SCHEMES = ("direct", "block", "kahan")
Role = Literal["i", "j", "param"]


class ReductionKind(str, Enum):
    SUM = "sum"          # somme sur j de chaque composante de la formule
    COUNT = "count"      # somme sur j d'une formule booléenne, accumulateur int64
    MIN = "min"          # minimum par composante
    MAX = "max"
    ARGMIN = "argmin"    # indice j (int64) du minimum, par composante
    ARGMAX = "argmax"
    KMIN = "kmin"        # K plus petites valeurs (formule scalaire uniquement)
    ARGKMIN = "argkmin"  # indices int64 des K plus petites valeurs
    KMIN_ARGKMIN = "kmin_argkmin"  # les deux sorties, un seul passage
    LOGSUMEXP = "logsumexp"  # log Σ_j exp f_d, par composante (contrat v0.5)
    SUMSOFTMAXWEIGHT = "sumsoftmaxweight"  # Σ_j softmax_j(f) g_e ; outputs = (f, g_1..g_E)
    # avenant v0.8 (parité PyKeOps) :
    MIN_ARGMIN = "min_argmin"  # MIN et ARGMIN, un seul passage, deux sorties
    MAX_ARGMAX = "max_argmax"
    LOGSUMEXPWEIGHT = "logsumexpweight"  # log Σ_j exp(f) g_e ; outputs = (f, g_1..g_E)


@dataclass(frozen=True)
class Arg:
    """Argument déclaré, dans l'ordre stable de la signature du kernel.

    ``i`` : tenseur (M, dim) ; ``j`` : tenseur (N, dim) ; ``param`` : (dim,).
    ``symbols`` : symboles symox scalaires des composantes, longueur ``dim``.
    """

    name: str
    role: Role
    dim: int
    symbols: tuple  # tuple[symox.Symbol, ...]


@dataclass(frozen=True)
class Reduction:
    kind: ReductionKind
    k: int | None = None  # requis pour KMIN / ARGKMIN / KMIN_ARGKMIN, k >= 0


@dataclass(frozen=True)
class Plan:
    """Description complète et immuable d'une réduction fusionnée.

    ``outputs`` : expressions symox scalaires (composantes de la formule, ordre
    row-major), fonctions des symboles de ``args`` uniquement. Pour COUNT, ce
    sont des expressions booléennes. ``out_dim == len(outputs)``.

    ``dtype`` : type des entrées flottantes et du calcul de la formule.
    ``acc_dtype`` : type de l'accumulateur de SUM, LOGSUMEXP, SUMSOFTMAXWEIGHT et LOGSUMEXPWEIGHT
    (état (m, s[, t]) des deux dernières ; par défaut ``dtype`` ;
    ``float64`` possible en FP32 si le backend le déclare). Jamais inférieur à
    ``dtype``. COUNT accumule toujours en int64.
    """

    args: tuple[Arg, ...]
    outputs: tuple  # tuple[symox.Expr, ...]
    reduction: Reduction
    dtype: DType
    acc_dtype: DType
    index_dtype: Literal["int64"] = "int64"
    fast_math: bool = False  # interdit par défaut ; aucun backend ne l'active seul
    fp_contract: bool = True  # contraction FMA intra-expression (contrat v0.3)
    #: avenant v0.7 : schéma de sommation des accumulations flottantes (SUM, et
    #: s, t_e de LOGSUMEXP/SUMSOFTMAXWEIGHT) ; sans effet sur les autres réductions.
    sum_scheme: SumScheme = "direct"
    #: avenant v0.6 : nombre de dimensions de batch des entrées (0 = sans batch).
    #: Les tailles de batch sont des données d'appel (``BatchLayout``), hors clé.
    batch_rank: int = 0
    #: avenant v0.6 §2 : réduction restreinte par des ranges block-sparse
    #: (``RangesLayout`` à l'appel) ; les tenseurs de ranges sont hors clé.
    ranges: bool = False
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def out_dim(self) -> int:
        return len(self.outputs)

    def args_of(self, role: Role) -> tuple[Arg, ...]:
        return tuple(a for a in self.args if a.role == role)

    def key(self) -> str:
        """Hash stable (sha256 hex) : srepr des sorties, args, réduction, dtypes.

        Implémenté par le noyau ; sert de clé de cache de compilation, avec
        en plus la version du générateur, le compilateur, les options et
        l'architecture cible (responsabilité du backend).
        """
        from ._plan_key import plan_key  # fourni par le noyau

        return plan_key(self)


def output_shapes(plan: Plan, m: int, batch: tuple[int, ...] = ()) -> dict[str, tuple[tuple[int, ...], str]]:
    """Formes et dtypes des sorties pour M lignes i (contrat, cf. doc).

    ``batch`` (avenant v0.6) : dimensions de batch préfixées à chaque forme,
    ``(*batch, M, …)`` ; le runtime passe ``(nb,)`` (batch aplati) au backend.
    """
    batch = tuple(int(b) for b in batch)
    return {key: (batch + shape, dt) for key, (shape, dt) in _output_shapes(plan, m).items()}


def accumulated_output_dtype(plan: Plan) -> str:
    """dtype des sorties accumulées (SUM, familles LOGSUMEXP) : ``acc_dtype``, sauf en FP16.

    Avenant v0.9 (b) : un plan float16 rend du float16 (parité de type avec
    PyKeOps), conversion finale de l'accumulateur (±inf au-delà de 65504).
    """
    return "float16" if plan.dtype == "float16" else plan.acc_dtype


def _output_shapes(plan: Plan, m: int) -> dict[str, tuple[tuple[int, ...], str]]:
    kind, d, k = plan.reduction.kind, plan.out_dim, plan.reduction.k
    idx = plan.index_dtype
    acc_out = accumulated_output_dtype(plan)
    if kind is ReductionKind.SUM:
        return {"value": ((m, d), acc_out)}
    if kind is ReductionKind.COUNT:
        return {"value": ((m, d), "int64")}
    if kind in (ReductionKind.MIN, ReductionKind.MAX):
        return {"value": ((m, d), plan.dtype)}
    if kind in (ReductionKind.ARGMIN, ReductionKind.ARGMAX):
        return {"index": ((m, d), idx)}
    if kind in (ReductionKind.MIN_ARGMIN, ReductionKind.MAX_ARGMAX):
        return {"value": ((m, d), plan.dtype), "index": ((m, d), idx)}
    if kind is ReductionKind.KMIN:
        return {"value": ((m, k), plan.dtype)}
    if kind is ReductionKind.ARGKMIN:
        return {"index": ((m, k), idx)}
    if kind is ReductionKind.KMIN_ARGKMIN:
        return {"value": ((m, k), plan.dtype), "index": ((m, k), idx)}
    if kind is ReductionKind.LOGSUMEXP:
        return {"value": ((m, d), acc_out)}
    if kind in (ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT):
        return {"value": ((m, d - 1), acc_out)}
    raise ValueError(kind)
