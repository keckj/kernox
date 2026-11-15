"""Cas de test du backend hip : Plans construits à la main + références Torch CPU.

Tant que ``codegen/pair.py`` (noyau) n'est pas livré, les fonctions de paire de
ces cas sont écrites à la main au format du contrat (``kp_out``/``kp_pair``)
et injectées via ``pair_provider``. Chaque cas fournit une fonction Torch qui
calcule la même formule dans le même ordre d'opérations (référence bit à bit
pour + − × en IEEE, ``-ffp-contract=off``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from kernox.plan import Arg, Plan, Reduction, ReductionKind


@dataclass
class PairSrc:
    source: str
    name: str = "kp_pair"
    out_type: str = "kp_out"
    arg_layout: tuple = ()


@dataclass
class Case:
    name: str
    args: tuple  # Arg
    out_dim: int
    pair_body: str  # corps C++ : remplit r.o0.. à partir des pointeurs
    torch_f: Callable  # (dict nom -> tenseur CPU élargi) -> (M, N, d)
    boolean: bool = False


def _pair_source(case: Case, ctype: str) -> str:
    otype = "bool" if case.boolean else ctype
    fields = " ".join(f"{otype} o{c};" for c in range(case.out_dim))
    params = ", ".join(f"const {ctype}* __restrict__ {a.name}" for a in case.args)
    return (f"struct kp_out {{ {fields} }};\n"
            f"__device__ __forceinline__ kp_out kp_pair({params}) {{\n"
            f"  kp_out r;\n{case.pair_body}\n  return r;\n}}\n")


def make_plan(case: Case, kind: ReductionKind, dtype: str, k=None, acc_dtype=None,
              fp_contract: bool = False) -> Plan:
    import symox

    # symboles factices (des Expr : la clé de plan parcourt leurs ``args``)
    outs = tuple(symox.Symbol(f"{case.name}_o{c}") for c in range(case.out_dim))
    # fp_contract=False : références Torch bit à bit (même ordre d'opérations, sans FMA) ;
    # le mode par défaut (contraction autorisée, v0.3) est couvert par test_fp_contract.py.
    return Plan(args=case.args, outputs=outs, reduction=Reduction(kind, k), dtype=dtype,
                acc_dtype=acc_dtype or dtype, fp_contract=fp_contract, meta={"case": case})


def pair_provider(plan: Plan) -> PairSrc:
    case = plan.meta["case"]
    ctype = {"float32": "float", "float64": "double"}[plan.dtype]
    return PairSrc(_pair_source(case, ctype))


A = lambda n, r, d: Arg(n, r, d, ())  # noqa: E731

SQDIST = Case(
    "sqdist", (A("x", "i", 3), A("y", "j", 3)), 1,
    "  kp_t d0 = x[0] - y[0]; kp_t d1 = x[1] - y[1]; kp_t d2 = x[2] - y[2];\n"
    "  r.o0 = (d0 * d0 + d1 * d1) + d2 * d2;",
    lambda t: (((t["x"][:, None, 0] - t["y"][None, :, 0]) ** 2
                + (t["x"][:, None, 1] - t["y"][None, :, 1]) ** 2)
               + (t["x"][:, None, 2] - t["y"][None, :, 2]) ** 2)[..., None],
)

# Gaussienne pondérée à sortie vectorielle, paramètre placé entre i et j.
GAUSS = Case(
    "gauss", (A("x", "i", 3), A("s", "param", 1), A("y", "j", 3), A("b", "j", 2)), 2,
    "  kp_t d0 = x[0] - y[0]; kp_t d1 = x[1] - y[1]; kp_t d2 = x[2] - y[2];\n"
    "  kp_t q = (d0 * d0 + d1 * d1) + d2 * d2;\n"
    "  kp_t g = exp(-q / (s[0] * s[0]));\n"
    "  r.o0 = g * b[0]; r.o1 = g * b[1];",
    lambda t: (torch.exp(-SQDIST.torch_f(t)[..., 0] / (t["s"][0] * t["s"][0]))[..., None]
               * t["b"][None, :, :]),
)

COUNT_BALL = Case(
    "ball", (A("x", "i", 3), A("y", "j", 3), A("rr", "param", 1)), 1,
    "  kp_t d0 = x[0] - y[0]; kp_t d1 = x[1] - y[1]; kp_t d2 = x[2] - y[2];\n"
    "  r.o0 = ((d0 * d0 + d1 * d1) + d2 * d2) <= rr[0];",
    lambda t: (SQDIST.torch_f(t)[..., 0] <= t["rr"][0])[..., None],
    boolean=True,
)

# Formule vectorielle signée (min/max/argmin/argmax par composante).
VEC = Case(
    "vec", (A("x", "i", 2), A("y", "j", 2)), 2,
    "  r.o0 = x[0] - y[0]; r.o1 = x[1] * y[1];",
    lambda t: torch.stack([t["x"][:, None, 0] - t["y"][None, :, 0],
                           t["x"][:, None, 1] * t["y"][None, :, 1]], -1),
)

# Scalaire 1D : sert aux égalités et NaN (valeurs entières répétées).
ABS1 = Case(
    "abs1", (A("x", "i", 1), A("y", "j", 1)), 1,
    "  r.o0 = fabs(x[0] - y[0]);",
    lambda t: (t["x"][:, None, 0] - t["y"][None, :, 0]).abs()[..., None],
)


# ---------------------------------------------------------------- références
def ref_reduce(F: torch.Tensor, kind: ReductionKind, k=None, acc=torch.float64):
    """Sémantique du contrat v0 sur F (M, N, d), CPU. Retourne dict de sorties."""
    M, N, d = F.shape
    if kind is ReductionKind.SUM:
        return {"value": F.to(acc).sum(1) if N else torch.zeros(M, d, dtype=acc)}
    if kind is ReductionKind.COUNT:
        return {"value": F.to(torch.int64).sum(1)}
    nan = torch.isnan(F)
    if kind in (ReductionKind.MIN, ReductionKind.ARGMIN):
        G = torch.where(nan, torch.full_like(F, math.inf), F)
        val = G.amin(1) if N else torch.full((M, d), math.inf, dtype=F.dtype)
        if kind is ReductionKind.MIN:
            return {"value": val}
        valid = ~nan & (G == val[:, None, :])
    elif kind in (ReductionKind.MAX, ReductionKind.ARGMAX):
        G = torch.where(nan, torch.full_like(F, -math.inf), F)
        val = G.amax(1) if N else torch.full((M, d), -math.inf, dtype=F.dtype)
        if kind is ReductionKind.MAX:
            return {"value": val}
        valid = ~nan & (G == val[:, None, :])
    else:
        assert d == 1
        vals = torch.full((M, k), math.inf, dtype=F.dtype)
        idx = torch.full((M, k), -1, dtype=torch.int64)
        for i in range(M):
            cand = [(F[i, j, 0].item(), j) for j in range(N) if not math.isnan(F[i, j, 0].item())]
            cand.sort()
            for r, (v, j) in enumerate(cand[:k]):
                vals[i, r], idx[i, r] = v, j
        out = {}
        if kind in (ReductionKind.KMIN, ReductionKind.KMIN_ARGKMIN):
            out["value"] = vals
        if kind in (ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN):
            out["index"] = idx
        return out
    # ARGMIN/ARGMAX : plus petit j valide atteignant l'extremum, -1 sinon
    ar = torch.arange(N, dtype=torch.int64)[None, :, None].expand(M, N, d)
    big = torch.full_like(ar, 2**62)
    j = torch.where(valid, ar, big).amin(1) if N else torch.full((M, d), -1, dtype=torch.int64)
    return {"index": torch.where(j == 2**62, torch.full_like(j, -1), j)}
