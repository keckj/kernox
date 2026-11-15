"""Plans construits à la main + fonctions de paire C++ écrites à la main.

Indépendant du frontend et de la fonction de paire du noyau (pas encore livrés) :
chaque formule de test a une source ``kp_pair`` et une référence Torch.
"""

from __future__ import annotations

import math

import torch

from kernox.codegen.pair import ArgLayout, PairSource
from kernox.plan import Arg, Plan, Reduction, ReductionKind as R


def make_plan(args, out_dim, kind, dtype="float64", k=None, acc_dtype=None, tag=""):
    # ``outputs`` : marqueurs opaques (la fonction de paire est fournie à la main)
    return Plan(
        args=tuple(Arg(n, role, dim, ()) for n, role, dim in args),
        outputs=tuple(f"{tag}{c}" for c in range(out_dim)),
        reduction=Reduction(R(kind), k),
        dtype=dtype,
        acc_dtype=acc_dtype or dtype,
    )


def pair_from(body_by_plan):
    """Fabrique une ``pair_function`` à partir d'une fonction plan -> corps C++."""

    def pair_function(plan, dialect="cuda"):
        T = "double" if plan.dtype == "float64" else "float"
        ot = "bool" if plan.reduction.kind is R.COUNT else T
        params = ", ".join(f"const {T}* __restrict__ a{n}" for n in range(len(plan.args)))
        src = ("struct kp_out { " + " ".join(f"{ot} o{c};" for c in range(plan.out_dim)) + " };\n"
               f"__device__ __forceinline__ kp_out kp_pair({params}) {{\n kp_out r;\n"
               + body_by_plan(plan, T) + "\n return r;\n}\n")
        layout = tuple(ArgLayout(f"a{n}", a.name, a.role, a.dim) for n, a in enumerate(plan.args))
        return PairSource(src, "kp_pair", "kp_out", layout)

    return pair_function


# ------------------------------------------------------------------ formules


def sqdist_body(plan, T):
    D = plan.args[0].dim
    return (f" {T} s = ({T})0;\n for (int c = 0; c < {D}; ++c) {{ {T} t = a0[c] - a1[c]; s += t * t; }}\n"
            " r.o0 = s;")


def sqdist_ref(x, y):
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)


def gauss_body(plan, T):
    # args : x(i,D), y(j,D), b(j,E), s(param,1) ; o_e = exp(-d2 / (2 s^2)) * b_e
    D, E = plan.args[0].dim, plan.args[2].dim
    lit = "2.0" if T == "double" else "2.0f"
    lines = [f" {T} d2 = ({T})0;",
             f" for (int c = 0; c < {D}; ++c) {{ {T} t = a0[c] - a1[c]; d2 += t * t; }}",
             f" const {T} g = exp(-d2 / ({lit} * a3[0] * a3[0]));"]
    lines += [f" r.o{e} = g * a2[{e}];" for e in range(E)]
    return "\n".join(lines)


def gauss_ref(x, y, b, s):
    x, y, b, s = (t.double() for t in (x, y, b, s))
    return torch.exp(-sqdist_ref(x, y) / (2 * s[0] ** 2)) @ b


def count_body(plan, T):
    D = plan.args[0].dim
    return (f" {T} s = ({T})0;\n for (int c = 0; c < {D}; ++c) {{ {T} t = a0[c] - a1[c]; s += t * t; }}\n"
            " r.o0 = s <= a2[0];")


def two_comp_body(plan, T):
    # o0 = a0[0] - a1[0] ; o1 = a0[1] * a1[1]  (D = 2)
    return " r.o0 = a0[0] - a1[0];\n r.o1 = a0[1] * a1[1];"


def two_comp_ref(x, y):
    return torch.stack([x[:, None, 0] - y[None, :, 0], x[:, None, 1] * y[None, :, 1]], -1)


def ident_body(plan, T):
    # o0 = y_j : permet d'injecter NaN/inf/égalités exactement
    return " r.o0 = a1[0] + ((" + T + ")0) * a0[0];"


# ------------------------------------------------------------------ références


def ref_minmax(vals, kind):
    """vals (M, N, D) float64 CPU -> sorties selon contrat (NaN ignorés)."""
    M, N, D = vals.shape
    is_min = kind in ("min", "argmin")
    out_v = torch.full((M, D), math.inf if is_min else -math.inf, dtype=torch.float64)
    out_i = torch.full((M, D), -1, dtype=torch.int64)
    for i in range(M):
        for c in range(D):
            best, bj = None, -1
            for j in range(N):
                v = float(vals[i, j, c])
                if v != v:
                    continue
                if bj < 0 or (v < best if is_min else v > best):
                    best, bj = v, j
            if bj >= 0:
                out_v[i, c], out_i[i, c] = best, bj
    return out_v, out_i


def ref_kmin(vals, K):
    """vals (M, N) -> (values, indices) (M, K) : ordre (valeur, j), NaN ignorés."""
    M, N = vals.shape
    ov = torch.full((M, K), math.inf, dtype=torch.float64)
    oi = torch.full((M, K), -1, dtype=torch.int64)
    for i in range(M):
        cands = sorted((float(vals[i, j]), j) for j in range(N) if vals[i, j] == vals[i, j])[:K]
        for k, (v, j) in enumerate(cands):
            ov[i, k], oi[i, k] = v, j
    return ov, oi
