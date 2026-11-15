"""Typed plan: the shared contract between the kernox frontend and the backends.

Status: contract v0 (implemented as a data structure, semantics described
in ``docs/kernel-backend-contract.md``). Backends consume ONLY this
module; they never depend on frontend classes (``Vi``, ``Vj``...).

Normalization: a plan always reduces over the ``j`` axis. A reduction
"over i" is converted by the frontend by swapping the i/j roles.
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
    SUM = "sum"          # sum over j of each formula component
    COUNT = "count"      # sum over j of a boolean formula, int64 accumulator
    MIN = "min"          # per-component minimum
    MAX = "max"
    ARGMIN = "argmin"    # j index (int64) of the minimum, per component
    ARGMAX = "argmax"
    KMIN = "kmin"        # K smallest values (scalar formula only)
    ARGKMIN = "argkmin"  # int64 indices of the K smallest values
    KMIN_ARGKMIN = "kmin_argkmin"  # both outputs, single pass
    LOGSUMEXP = "logsumexp"  # log Σ_j exp f_d, per component (contract v0.5)
    SUMSOFTMAXWEIGHT = "sumsoftmaxweight"  # Σ_j softmax_j(f) g_e; outputs = (f, g_1..g_E)
    # amendment v0.8 (PyKeOps parity):
    MIN_ARGMIN = "min_argmin"  # MIN and ARGMIN, single pass, two outputs
    MAX_ARGMAX = "max_argmax"
    LOGSUMEXPWEIGHT = "logsumexpweight"  # log Σ_j exp(f) g_e; outputs = (f, g_1..g_E)


@dataclass(frozen=True)
class Arg:
    """Declared argument, in the stable order of the kernel signature.

    ``i``: tensor (M, dim); ``j``: tensor (N, dim); ``param``: (dim,).
    ``symbols``: scalar symox symbols of the components, of length ``dim``.
    """

    name: str
    role: Role
    dim: int
    symbols: tuple  # tuple[symox.Symbol, ...]


@dataclass(frozen=True)
class Reduction:
    kind: ReductionKind
    k: int | None = None  # required for KMIN / ARGKMIN / KMIN_ARGKMIN, k >= 0


@dataclass(frozen=True)
class Plan:
    """Complete, immutable description of a fused reduction.

    ``outputs``: scalar symox expressions (formula components, row-major
    order), functions of the ``args`` symbols only. For COUNT, they are
    boolean expressions. ``out_dim == len(outputs)``.

    ``dtype``: type of the floating-point inputs and of the formula evaluation.
    ``acc_dtype``: accumulator type of SUM, LOGSUMEXP, SUMSOFTMAXWEIGHT and LOGSUMEXPWEIGHT
    (the (m, s[, t]) state for the last two; defaults to ``dtype``;
    ``float64`` is allowed in FP32 if the backend declares it). Never narrower
    than ``dtype``. COUNT always accumulates in int64.
    """

    args: tuple[Arg, ...]
    outputs: tuple  # tuple[symox.Expr, ...]
    reduction: Reduction
    dtype: DType
    acc_dtype: DType
    index_dtype: Literal["int64"] = "int64"
    fast_math: bool = False  # off by default; no backend enables it on its own
    fp_contract: bool = True  # FMA contraction within expressions (contract v0.3)
    #: amendment v0.7: summation scheme of floating-point accumulations (SUM, and
    #: s, t_e of LOGSUMEXP/SUMSOFTMAXWEIGHT); no effect on other reductions.
    sum_scheme: SumScheme = "direct"
    #: amendment v0.6: number of batch dimensions of the inputs (0 = no batch).
    #: Batch sizes are call-time data (``BatchLayout``), not part of the key.
    batch_rank: int = 0
    #: amendment v0.6 §2: reduction restricted by block-sparse ranges
    #: (``RangesLayout`` at call time); range tensors are not part of the key.
    ranges: bool = False
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def out_dim(self) -> int:
        return len(self.outputs)

    def args_of(self, role: Role) -> tuple[Arg, ...]:
        return tuple(a for a in self.args if a.role == role)

    def key(self) -> str:
        """Stable hash (hex sha256): srepr of outputs, args, reduction, dtypes.

        Implemented by the core; used as the compilation cache key, together
        with the generator version, compiler, options and target
        architecture (the backend's responsibility).
        """
        from ._plan_key import plan_key  # provided by the core

        return plan_key(self)


def output_shapes(plan: Plan, m: int, batch: tuple[int, ...] = ()) -> dict[str, tuple[tuple[int, ...], str]]:
    """Output shapes and dtypes for M rows i (contract, see the docs).

    ``batch`` (amendment v0.6): batch dimensions prepended to each shape,
    ``(*batch, M, …)``; the runtime passes ``(nb,)`` (flattened batch) to the backend.
    """
    batch = tuple(int(b) for b in batch)
    return {key: (batch + shape, dt) for key, (shape, dt) in _output_shapes(plan, m).items()}


def accumulated_output_dtype(plan: Plan) -> str:
    """dtype of accumulated outputs (SUM, LOGSUMEXP families): ``acc_dtype``, except in FP16.

    Amendment v0.9 (b): a float16 plan returns float16 (type parity with
    PyKeOps), with a final conversion of the accumulator (±inf beyond 65504).
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
