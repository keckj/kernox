"""Outils de test du backend sycl : plans construits à la main et référence.

Indépendants du frontend (``kernox.expr``) et de l'oracle
(``kernox.oracle``) du noyau. La référence
ci-dessous est indépendante du codegen : la formule est réécrite en Torch et
évaluée en FP64 sur CPU (matrice dense, petits cas uniquement).
"""

from __future__ import annotations

import torch

from kernox.plan import Arg, Plan, Reduction, ReductionKind, output_shapes

DEV = "xpu"

# Règle : aucune allocation hôte en O(M·N) (mémoire hôte bornée).
# Les références denses (M, N, D) sont réservées aux petits cas et plafonnées ;
# au-delà, la référence se calcule par blocs de lignes de HOST_BLOCK_ELEMS.
HOST_DENSE_MAX = 1 << 23   # éléments (64 Mio en FP64)
HOST_BLOCK_ELEMS = 1 << 22


def arg(name: str, role: str, dim: int) -> Arg:
    import symox

    syms = tuple(symox.Symbol(f"{name}_{c}", real=True) for c in range(dim))
    return Arg(name, role, dim, syms)


def plan(args, outputs, kind, dtype="float32", acc_dtype=None, k=None, fp_contract=True, **meta) -> Plan:
    return Plan(
        args=tuple(args),
        outputs=tuple(outputs),
        reduction=Reduction(ReductionKind(kind), k),
        dtype=dtype,
        acc_dtype=acc_dtype or dtype,
        fp_contract=fp_contract,
        meta=dict(meta),
    )


_BACKEND = None


def backend():
    global _BACKEND
    if _BACKEND is None:
        from kernox.backends.sycl import SyclBackend

        _BACKEND = SyclBackend()
    return _BACKEND


def alloc_outputs(p: Plan, m: int, device=DEV) -> dict:
    return {
        name: torch.empty(shape, dtype=getattr(torch, dt), device=device)
        for name, (shape, dt) in output_shapes(p, m).items()
    }


def run(p: Plan, inputs: dict, m: int | None = None, compiled=None):
    c = compiled or backend().compile(p, DEV)
    if m is None:
        m = next(inputs[a.name].shape[0] for a in p.args if a.role == "i")
    outs = alloc_outputs(p, m)
    c.launch(inputs, outs)
    torch.xpu.synchronize()
    return {k: v.cpu() for k, v in outs.items()}, c


# ---------------------------------------------------------------- référence

def reduce_ref(kind: str, F: torch.Tensor, k: int | None = None):
    """F : (M, N, D) en float64 CPU (bool pour count). Sémantique du contrat v0."""
    kind = ReductionKind(kind)
    M, N, D = F.shape
    assert F.numel() <= HOST_DENSE_MAX, f"référence dense hôte {tuple(F.shape)} : au-delà du plafond, calculer par blocs"
    if kind is ReductionKind.SUM:
        return {"value": F.sum(1)}
    if kind is ReductionKind.COUNT:
        return {"value": F.to(torch.int64).sum(1)}
    nan = torch.isnan(F)
    if kind in (ReductionKind.MIN, ReductionKind.ARGMIN):
        # premier minimum non-NaN ; tri stable pour la règle du plus petit j,
        # torch.sort range les NaN en dernier
        vals, idx = torch.sort(torch.where(nan, torch.full_like(F, float("nan")), F), dim=1, stable=True)
    if kind in (ReductionKind.MAX, ReductionKind.ARGMAX):
        vals, idx = torch.sort(torch.where(nan, torch.full_like(F, float("nan")), -F), dim=1, stable=True)
        vals = -vals
    if kind in (ReductionKind.MIN, ReductionKind.MAX, ReductionKind.ARGMIN, ReductionKind.ARGMAX):
        if N == 0:
            neutral = float("inf") if kind in (ReductionKind.MIN, ReductionKind.ARGMIN) else float("-inf")
            return {"value": torch.full((M, D), neutral, dtype=F.dtype),
                    "index": torch.full((M, D), -1, dtype=torch.int64)}
        v0, i0 = vals[:, 0, :], idx[:, 0, :]
        valid = ~torch.isnan(v0)
        neutral = float("inf") if kind in (ReductionKind.MIN, ReductionKind.ARGMIN) else float("-inf")
        return {"value": torch.where(valid, v0, torch.full_like(v0, neutral)),
                "index": torch.where(valid, i0, torch.full_like(i0, -1))}
    # K-min (formule scalaire)
    f = F[:, :, 0]
    vals, idx = torch.sort(f, dim=1, stable=True)  # NaN rangés en dernier
    kk = min(k, N)
    out_v = torch.full((M, k), float("inf"), dtype=F.dtype)
    out_i = torch.full((M, k), -1, dtype=torch.int64)
    v, ii = vals[:, :kk], idx[:, :kk]
    ok = ~torch.isnan(v)
    out_v[:, :kk] = torch.where(ok, v, torch.full_like(v, float("inf")))
    out_i[:, :kk] = torch.where(ok, ii, torch.full_like(ii, -1))
    return {"value": out_v, "index": out_i}
