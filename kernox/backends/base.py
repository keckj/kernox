"""Common backend interface (see ``docs/kernel-backend-contract.md``).

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
    """Published for diagnostics and benchmarks."""

    backend: str              # "cuda" | "hip" | "sycl" | "cpu"
    device_name: str
    arch: str                 # sm_75, gfx1201, intel_gpu_bmg_g31...
    compiler: str             # name + exact version
    options: tuple[str, ...]
    cache_hit: bool
    compile_seconds: float
    temp_bytes_per_launch: int  # bounded temporary buffers, published
    source_sha256: str
    variant: "Variant | None" = None  # effective variant (contract v0.2)


class CompiledReduction(Protocol):
    info: CompileInfo

    def launch(self, inputs: dict, outputs: dict) -> None:
        """Launches the reduction asynchronously on the current Torch
        stream/queue of the tensors' device.

        ``inputs``: argument name -> torch.Tensor (i: (M,d), j: (N,d),
        param: (d,)), all on the same device, dtype ``plan.dtype``, contiguous.
        ``outputs``: keys of ``plan.output_shapes`` -> tensors preallocated by
        the core. No host synchronization, no CPU copy. Temporaries are
        allocated through the Torch allocator of the same device and stream.
        M == 0: return immediately. N == 0: outputs = neutral elements.

        Amendment v0.6 (plans with ``batch_rank > 0``, backends declaring
        ``batch``): call ``launch(inputs, outputs, batch=BatchLayout)``. Inputs
        keep their shape ``(*Ba, L, d)`` / ``(*Bp, d)`` and their batch strides
        (arbitrary, 0 included); their last two dimensions are contiguous.
        Element ``b`` of an argument starts at
        ``data_ptr + batch.offsets[name][b] * itemsize``. Outputs are
        contiguous ``(nb, M, …)``. Without batching, ``batch`` is not passed.
        """


@dataclass(frozen=True)
class Sizes:
    """Sizes of a call, used to choose a variant (contract v0.2).

    ``nb``: number of batch elements (amendment v0.6), 1 without batching.
    """

    m: int
    n: int
    nb: int = 1


@dataclass(frozen=True, eq=False)
class BatchLayout:
    """Flattened batch of a call (amendment v0.6), built and cached by the runtime.

    ``shape``: broadcast batch shape ``B``; ``nb = prod(B)``.
    ``offsets``: argument name -> int64 tensor ``(nb,)`` on the inputs'
    device, offset in elements of the start of element ``b`` relative to
    ``data_ptr`` (broadcasting = repeated offsets, unbatched param = zeros).
    ``host_offsets``: the same tables on the host (CPU), for the reference
    backends; a native backend does not read them during ``launch``.
    """

    shape: tuple[int, ...]
    nb: int
    offsets: dict
    host_offsets: dict


class RangesLayout:
    """Block-sparse ranges of a call (amendment v0.6 §2), normalized by the runtime.

    Plan orientation (reduction over j). ``work``: int64 ``(W, 4)`` on the
    device, tiles ``(i0, i1, k0, k1)`` of height ``i1 − i0 ≤ tile_m``;
    ``redranges``: int64 ``(R, 2)`` on the device, j intervals ``[c, d)``;
    a tile reduces its rows over ``∪ redranges[k0:k1]``. Rows covered by no
    tile receive the neutral value (``launch`` is responsible for this).
    ``host_work`` / ``host_redranges``: host copies for the reference
    backends, made on first access if they are not provided (GPU backends do
    not need them: copying R × 2 integers costs several ms at R ~ 10⁶).
    ``tile_m``: value published by the compiled object (``ranges_tile_m``).
    Equality and hashing by identity.
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
        """Summary of the bounds, computed once per layout (on the device, with a
        single transfer of 15 integers; on the host for XPU): the memory-safety
        checks of the backends and of the runtime are O(1) per call."""
        import torch

        if self._host_work is not None or torch.device(self.work.device).type in ("cpu", "xpu"):
            return RangesSummary.of(self.host_work, self.host_redranges)
        return RangesSummary.of(self.work, self.redranges)


@dataclass(frozen=True)
class RangesSummary:
    """Bounds of a :class:`RangesLayout` (all entries, then non-empty entries).

    ``i0_min``, ``i1_max``, ``h_min``/``h_max`` (height ``i1 − i0``), ``k0_min``,
    ``k1_max``, ``dk_min`` (``k1 − k0``), ``c_min``, ``d_max``: over all
    entries; ``live_*``: tiles with ``i1 > i0`` and intervals with ``d > c`` only
    (those a kernel actually visits). ``covered``: Σ (i1 − i0).
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
        kw = dict(zip(names, torch.stack(vals).tolist())) if vals else {}  # a single transfer
        return RangesSummary(nw, nr, **kw)


@dataclass(frozen=True)
class Variant:
    """Kernel variant of a backend (strategy + parameters), contract v0.2.

    ``name``: strategy (e.g. ``"row"``, ``"jsplit"``, ``"coop_topk"``).
    ``params``: sorted (key, int|str value) pairs, JSON-serializable.
    All variants of a plan follow the same semantics; only floating-point
    sums may differ, through the summation order (within the test
    tolerances). A variant never changes the dtype, the accumulator,
    fast-math or FMA contraction.
    """

    name: str
    params: tuple[tuple[str, int | str], ...] = ()

    def as_json(self) -> dict:
        return {"name": self.name, "params": dict(self.params)}


class Backend(Protocol):
    name: str
    #: contract v0.3: True if the backend can compile without any FMA
    #: contraction (``plan.fp_contract == False``). Optional attribute, missing =
    #: False: the runtime then rejects a ``fp_contract=False`` plan (UnsupportedError).
    fp_contract_off: bool
    #: amendment v0.11: True if the backend accepts ``plan.flush_denormals``
    #: (subnormal values may be flushed to zero). Optional attribute, missing =
    #: False: the runtime then rejects such a plan (UnsupportedError).
    flush_denormals: bool
    #: amendment v0.7: implemented summation schemes (``"direct"``, ``"block"``,
    #: ``"kahan"``). Optional attribute, missing = ``("direct",)``: the runtime
    #: then rejects any other scheme for SUM/LOGSUMEXP/SUMSOFTMAXWEIGHT
    #: (UnsupportedError), on top of ``supports``. A backend implementing
    #: ``block`` publishes its tile size ``T`` in ``CompileInfo.options``
    #: as ``"sum_block=T"``.
    sum_schemes: tuple
    #: amendment v0.6: True if ``launch`` accepts ``batch=BatchLayout`` for
    #: plans with ``batch_rank > 0``. Optional attribute, missing = False: the
    #: runtime then rejects any batched call (UnsupportedError), on top of ``supports``.
    batch: bool
    #: amendment v0.6 §2: True if ``launch`` accepts ``ranges=RangesLayout`` for
    #: plans with ``ranges=True``; the compiled object then publishes ``ranges_tile_m``.
    #: Missing = False: the runtime rejects any call with ranges.
    ranges: bool

    def supports(self, plan: Plan, device) -> Support: ...

    def default_variant(self, plan: Plan, device, sizes: Sizes) -> Variant:
        """Heuristic choice without measurement (ESTIMATE effort). Deterministic."""

    def variants(self, plan: Plan, device, sizes: Sizes, effort: str) -> list[Variant]:
        """Valid candidates for autotuning, ``default_variant`` first.

        ``effort == "measure"``: small targeted set (≤ 4);
        ``effort == "patient"``: wider space (≤ 32). Candidates that are
        invalid for the device (shared memory, registers...) are excluded.
        """

    def compile(self, plan: Plan, device, variant: Variant | None = None) -> CompiledReduction:
        """Compiles (or loads from the atomic cache) the plan's kernel.

        ``variant=None`` is equivalent to the backend's default variant; the
        variant is part of the cache key. ``CompiledReduction.info`` exposes
        the effective variant (``variant`` attribute).
        """
