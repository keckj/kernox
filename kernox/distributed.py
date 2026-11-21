r"""Reductions split across the processes of a ``torch.distributed`` group.

Model: every rank of the group passes the same full inputs (replicated
data, as in the rest of a PyTorch program), and every rank receives the
full result. Each rank only computes its share of the $M \times N$ pairs:

- ``split="output"``: rank r owns a contiguous span $[a, b)$ of the output
  rows; its outputs are exact and gathered with ``all_gather``;
- ``split="reduced"``: rank r owns a span $[c, d)$ of the reduced axis; the
  partial results are merged (sum, min with index, K smallest, log-sum-exp
  states...), with the same semantics as a single call.

Autograd works through both splits (Megatron-style operators): an input
that requires grad gets the sum ``all_reduce`` of its partial gradients in
the backward pass, the gathered output sends each rank the rows it owns.
The output gradient must be the same on all ranks, which holds when the
loss is computed from replicated results.

Collectives are plain ``all_gather``/``all_reduce``: results are identical
on all ranks and reproducible for a fixed group size (NCCL, Gloo), but
not bitwise identical to a single-process call (another summation order).

Launch with ``torchrun``, or ``mpirun``/``srun``/OAR and
`init_from_env`. Without an initialized group, or with a single rank,
`reduce` is a plain call.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from collections import OrderedDict
from types import SimpleNamespace

from .errors import ValidationError
from .plan import Reduction as PlanReduction
from .plan import ReductionKind as K

SPLITS = ("output", "reduced")

# --------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------

_ENV_RANK = (
    ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"),  # torchrun
    ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK",
     "OMPI_COMM_WORLD_LOCAL_SIZE"),  # Open MPI
    ("PMIX_RANK", "PMIX_SIZE", "PMIX_LOCAL_RANK", "PMIX_LOCAL_SIZE"),
    ("PMI_RANK", "PMI_SIZE", "MPI_LOCALRANKID", "MPI_LOCALNRANKS"),  # MPICH, Intel MPI
    ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID", "SLURM_NTASKS_PER_NODE"),
)


def _env_ranks(env):
    for rank, size, local, local_size in _ENV_RANK:
        if rank in env and size in env:
            loc = env.get(local)
            lsz = env.get(local_size, "")
            lsz = lsz.split("(")[0].split(",")[0]  # SLURM: "4(x2),2"
            return int(env[rank]), int(env[size]), (int(loc) if loc else None), (int(lsz) if lsz.isdigit() else None)
    return None


def _master_addr(env):
    if env.get("MASTER_ADDR"):
        return env["MASTER_ADDR"]
    nodefile = env.get("OAR_NODEFILE") or env.get("OAR_FILE_NODES") or env.get("PBS_NODEFILE")
    if nodefile and os.path.exists(nodefile):
        with open(nodefile) as f:
            for line in f:
                if line.strip():
                    return line.strip()
    if env.get("SLURM_LAUNCH_NODE_IPADDR"):
        return env["SLURM_LAUNCH_NODE_IPADDR"]
    return "127.0.0.1"


def init_from_env(backend=None, *, threads="auto", timeout=None):
    """Initializes the default process group from the launcher's environment.

    Reads the rank and world size of ``torchrun`` (``RANK``...), Open MPI
    (``OMPI_COMM_WORLD_*``), PMIx, MPICH/Intel MPI (``PMI_*``) or SLURM
    (``SLURM_PROCID``...). ``MASTER_ADDR`` defaults to the first host of
    ``$OAR_NODEFILE`` (or ``$PBS_NODEFILE``, then the SLURM launch node, then
    127.0.0.1), ``MASTER_PORT`` to 29500. ``backend``: ``"nccl"`` when CUDA
    is available, ``"gloo"`` otherwise. With CUDA, selects the device
    ``local_rank % device_count``. ``threads="auto"``: on the CPU, sets
    ``torch.set_num_threads`` to the cores allowed to this process, divided
    by the ranks of the node allowed on the same cores: a rank bound to its
    own cores by the launcher uses them all, unbound ranks share the node
    (an int sets it, ``None`` leaves it).

    Returns ``(rank, world_size, local_rank, device)`` as a namespace. Does
    nothing but return it if the group is already initialized.
    """
    import torch
    import torch.distributed as dist

    env = os.environ
    found = _env_ranks(env)
    if found is None:
        rank, size, local, local_size = 0, 1, 0, 1
    else:
        rank, size, local, local_size = found
    if local is None:
        local = 0 if size == 1 else rank
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = torch.device("cpu")
    if backend == "nccl" or (backend != "gloo" and torch.cuda.is_available()):
        device = torch.device("cuda", local % max(1, torch.cuda.device_count()))
        torch.cuda.set_device(device)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", _master_addr(env))
        os.environ.setdefault("MASTER_PORT", "29500")
        kw = {}
        if timeout is not None:
            import datetime

            kw["timeout"] = timeout if isinstance(timeout, datetime.timedelta) else datetime.timedelta(seconds=timeout)
        if device.type == "cuda" and backend == "nccl":
            kw["device_id"] = device
        dist.init_process_group(backend, rank=rank, world_size=size, init_method="env://", **kw)
    if device.type == "cpu" and threads is not None:
        if threads == "auto":
            threads = _auto_threads(dist)
        torch.set_num_threads(int(threads))
    return SimpleNamespace(rank=dist.get_rank(), world_size=dist.get_world_size(), local_rank=local,
                           device=device)


def _affinity():
    try:
        return frozenset(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return frozenset(range(os.cpu_count() or 1))


def _omp_places():
    """``omp_get_num_places()`` of the OpenMP runtime loaded in the process
    (libgomp, libomp or libiomp5), or 0."""
    import ctypes

    try:
        with open("/proc/self/maps") as f:
            paths = {line.split()[-1] for line in f if line.rstrip().endswith(".so") or ".so." in line}
    except OSError:
        return 0
    for path in sorted(paths):
        name = os.path.basename(path)
        if name.startswith(("libgomp", "libomp", "libiomp5")):
            try:
                return int(ctypes.CDLL(path).omp_get_num_places())
            except (OSError, AttributeError):
                continue
    return 0


def _cores(mine):
    """Cores allowed to this process. With ``OMP_PROC_BIND``, the OpenMP runtime
    pins the main thread to its first place when torch starts, so the affinity
    seen afterwards is one core; the OpenMP places, computed from the affinity
    the process started with (the binding of the launcher), count instead."""
    bind = os.environ.get("OMP_PROC_BIND", "").strip().lower()
    if bind and bind != "false":
        return max(len(mine), _omp_places())
    return len(mine)


def _share(mine, cores, host, gathered):
    """``cores`` divided by the ranks of ``host`` whose affinities overlap ``mine``."""
    sharing = sum(1 for h, other in gathered if h == host and other & mine)
    return max(1, cores // max(1, sharing))


def _auto_threads(dist):
    """Threads of this rank: its cores (affinity), shared with the ranks of the
    same host bound to overlapping cores (one all_gather of host names and
    affinities). Ranks bound to disjoint cores by the launcher keep all of
    theirs; unbound ranks divide the node."""
    import socket

    mine, host = _affinity(), socket.gethostname()
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (host, mine))
    return _share(mine, _cores(mine), host, gathered)


# --------------------------------------------------------------------------
# Spans
# --------------------------------------------------------------------------


def _uniform(n, size):
    base, rem = divmod(int(n), size)
    out, a = [], 0
    for r in range(size):
        b = a + base + (r < rem)
        out.append((a, b))
        a = b
    return tuple(out)


def _from_cumulative(cum, size):
    """Spans of balanced work, ``cum[t]`` = work of rows [0, t+1); each span ≥ 1 row."""
    import torch

    n = cum.numel()
    total = int(cum[-1])
    if total <= 0:
        return _uniform(n, size)
    targets = torch.tensor([total * r / size for r in range(1, size)], dtype=torch.float64)
    cuts = (torch.searchsorted(cum.to(torch.float64), targets, side="left") + 1).tolist()
    bounds, prev = [0], 0
    for r, t in enumerate(cuts, start=1):
        t = min(max(int(t), prev + 1), n - (size - r))
        bounds.append(t)
        prev = t
    bounds.append(n)
    return tuple((bounds[r], bounds[r + 1]) for r in range(size))


def spans(n: int, group=None):
    """Balanced contiguous spans ``((a_0, b_0), ...)`` of ``range(n)``, one per rank."""
    import torch.distributed as dist

    return _uniform(n, dist.get_world_size(group))


def _ranges_cumulative(tr, length, axis):
    """Cumulative pair count along ``axis`` ("i": rows, "j": reduced axis); host tensor."""
    import torch

    rows, slices, red = (t.to(torch.int64).cpu() for t in (tr.rows, tr.slices, tr.red))
    diff = torch.zeros(length + 1, dtype=torch.int64)
    if rows.shape[0] and red.shape[0]:
        counts = torch.diff(slices, prepend=slices.new_zeros(1))
        blk = torch.repeat_interleave(torch.arange(rows.shape[0]), counts)
        if axis == "i":  # rows of block b each have Σ (d - c) pairs
            w = torch.zeros(rows.shape[0], dtype=torch.int64).index_add_(0, blk, red[:, 1] - red[:, 0])
            diff.index_add_(0, rows[:, 0], w)
            diff.index_add_(0, rows[:, 1], -w)
        else:  # columns [c, d) each meet the rows of their block
            h = (rows[:, 1] - rows[:, 0])[blk]
            diff.index_add_(0, red[:, 0], h)
            diff.index_add_(0, red[:, 1], -h)
    return torch.cumsum(torch.cumsum(diff[:length], 0), 0)


# --------------------------------------------------------------------------
# Caches (by identity of the range tensors, like the runtime)
# --------------------------------------------------------------------------

_LOCK = threading.Lock()
_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 64


def _cached(key, refs, build):
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and all(a is b for a, b in zip(hit[0], refs)):
            _CACHE.move_to_end(key)
            return hit[1]
    value = build()
    with _LOCK:
        _CACHE[key] = (refs, value)
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return value


def _refs(tr):
    return () if tr is None else (tr.rows, tr.slices, tr.red)


def _ids(refs):
    return tuple(id(t) for t in refs)


def _range_spans(tr, length, axis, size):
    refs = _refs(tr)
    return _cached(("spans", _ids(refs), length, axis, size), refs,
                   lambda: _from_cumulative(_ranges_cumulative(tr, length, axis), size))


def _clip_intervals(t, lo, hi):
    return (t.clamp(lo, hi) - lo).to(t.dtype)


def _clip(tr, lo, hi, part):
    """Triplet restricted to [lo, hi) and shifted: ``part="rows"`` or ``"red"``."""
    from .ranges import Triplet

    if tr is None:
        return None
    refs = _refs(tr)

    def build():
        if part == "rows":
            return Triplet(_clip_intervals(tr.rows, lo, hi), tr.slices, tr.red)
        return Triplet(tr.rows, tr.slices, _clip_intervals(tr.red, lo, hi))

    return _cached(("clip", _ids(refs), lo, hi, part), refs, build)


# --------------------------------------------------------------------------
# Collectives and autograd operators
# --------------------------------------------------------------------------


def _all_gather(t, group):
    import torch
    import torch.distributed as dist

    cplx = t.is_complex()
    src = torch.view_as_real(t) if cplx else t
    src = src.contiguous()
    out = [torch.empty_like(src) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, src, group=group)
    return [torch.view_as_complex(o) for o in out] if cplx else out


def _functions():
    import torch
    import torch.distributed as dist

    class CopyToGroup(torch.autograd.Function):
        """Identity; backward: sum ``all_reduce`` of the gradient (replicated input)."""

        @staticmethod
        def forward(ctx, t, group):
            ctx.group = group
            return t.view_as(t)

        @staticmethod
        def backward(ctx, grad):
            g = grad.contiguous().clone()
            if g.is_complex():
                dist.all_reduce(torch.view_as_real(g), group=ctx.group)
            else:
                dist.all_reduce(g, group=ctx.group)
            return g, None

    class GatherRows(torch.autograd.Function):
        """Rows of every rank concatenated along dim -2; backward: own rows."""

        @staticmethod
        def forward(ctx, t, rank, row_spans, group):
            ctx.span = row_spans[rank]
            return _gather_rows(t, row_spans, group)

        @staticmethod
        def backward(ctx, grad):
            a, b = ctx.span
            return grad.narrow(-2, a, b - a), None, None, None

    return CopyToGroup, GatherRows


_FN = None


def _fn():
    global _FN
    if _FN is None:
        _FN = _functions()
    return _FN


def _gather_rows(t, row_spans, group):
    import torch

    width = max(b - a for a, b in row_spans)
    pad = width - t.shape[-2]
    if pad:
        shape = list(t.shape)
        shape[-2] = pad
        t = torch.cat([t, t.new_zeros(shape)], dim=-2)
    parts = _all_gather(t, group)
    return torch.cat([p.narrow(-2, 0, b - a) for p, (a, b) in zip(parts, row_spans)], dim=-2)


def _stack_live(p, rank, group):
    """Partials of all ranks stacked on dim 0, slot ``rank`` being the live ``p``.

    The merge is then made of ordinary torch operations: autograd gives the
    derivative with respect to the local partial only, the other ranks
    handle theirs.
    """
    import torch

    parts = _all_gather(p.detach(), group)
    parts[rank] = p
    return torch.stack(parts)


# --------------------------------------------------------------------------
# Merges of the reduced-axis split
# --------------------------------------------------------------------------


def _shift(idx, c):
    import torch

    return torch.where(idx >= 0, idx + c, idx) if c else idx


def _merge_select(v, idx, rank, group, largest):
    """(value, index) of the best candidate over ranks; ties: smallest index ≥ 0."""
    import torch

    V = _stack_live(v, rank, group)
    I = torch.stack(_all_gather(idx, group))
    Vd = V.detach()
    best = Vd.amax(0) if largest else Vd.amin(0)
    cand = Vd == best
    P = V.shape[0]
    order = torch.arange(P, device=V.device).view((P,) + (1,) * (V.dim() - 1))
    score = torch.where(cand & (I >= 0), 0, torch.where(cand, 1, 2)) * P + order
    pick = score.argmin(0, keepdim=True)
    return V.gather(0, pick).squeeze(0), I.gather(0, pick).squeeze(0)


def _merge_kmin(v, idx, k, rank, group):
    """K smallest (value, index) pairs over ranks, lexicographic order, (+inf, -1) padding."""
    import torch

    V = _stack_live(v, rank, group)          # (P, ..., M, K)
    I = torch.stack(_all_gather(idx, group))
    V = V.movedim(0, -2).flatten(-2)          # (..., M, P*K)
    I = I.movedim(0, -2).flatten(-2)
    key = torch.where(I >= 0, I, torch.iinfo(torch.int64).max)
    o1 = torch.sort(key, dim=-1, stable=True).indices
    o2 = torch.sort(V.detach().take_along_dim(o1, -1), dim=-1, stable=True).indices
    order = o1.take_along_dim(o2, -1)[..., :k]
    return V.take_along_dim(order, -1), I.take_along_dim(order, -1)


def _merge_lse(lse_parts):
    import torch

    return torch.logsumexp(lse_parts, dim=0)


def _softmax_combine(lse_r, s_r, rank, group):
    """(LSE, Σ_r softmax_r(LSE_r) S_r) with empty shards (LSE_r = -inf) ignored."""
    import torch

    L = _stack_live(lse_r, rank, group)
    S = _stack_live(s_r, rank, group)
    lse = torch.logsumexp(L, dim=0)
    finite = torch.isfinite(lse)
    w = torch.exp(L - torch.where(finite, lse, torch.zeros_like(lse)))
    w = torch.where(finite, w, torch.zeros_like(w))
    keep = w > 0
    S = torch.where(keep, S, torch.zeros_like(S))
    return lse, (w * S).sum(0)


# --------------------------------------------------------------------------
# reduce
# --------------------------------------------------------------------------


def _companion(red, kind, outputs=None, k=None):
    """Reduction of the same plan with another kind (cached on ``red``)."""
    from .runtime import Reduction

    cache = red.__dict__.setdefault("_dist_companions", {})
    key = (kind, outputs is not None, k)
    hit = cache.get(key)
    if hit is None:
        plan = red.plan
        changes = {"reduction": PlanReduction(kind, plan.reduction.k if k is None else k)}
        if outputs is not None:
            changes["outputs"] = outputs
        hit = cache[key] = Reduction(dataclasses.replace(plan, **changes), red.backend, red.tuning)
    return hit


def _resolve_split(split, m, n, size):
    if split is None:
        return "output" if (m is not None and (m >= n or m >= 32 * size)) or n is None else "reduced"
    if split not in SPLITS:
        raise ValidationError(f"split: expected one of {SPLITS} or None, got {split!r}")
    return split


def reduce(red, /, *, group=None, split=None, ranges=None, **inputs):  # noqa: A001
    """Calls the reduction ``red`` with its work split across ``group``.

    ``red``: a `kernox.Reduction` or a ``RangedReduction``
    (``run.with_ranges(r)``); ``ranges`` (a `kernox.Ranges`) is the
    same as passing ``run.with_ranges(ranges)``. ``group``: a
    ``torch.distributed`` process group (default: the world group).
    ``split``: ``"output"`` (each rank computes a span of output rows),
    ``"reduced"`` (each rank reduces a span of the reduced axis, partial
    results merged) or ``None`` (``"output"`` unless the output axis is
    short compared to the reduced axis). Every rank must call it with the
    same inputs; every rank returns the full result, with autograd.

    Falls back to a plain call without an initialized group, with a single
    rank, or when the split axis is shorter than the number of ranks.
    """
    import torch
    import torch.distributed as dist

    from .runtime import RangedReduction, Reduction

    validate = False
    if isinstance(red, RangedReduction):
        if ranges is not None:
            raise ValidationError("reduce: ranges given twice (RangedReduction and ranges=)")
        base, fwd, bwd, validate = red.red, red._fwd, red._bwd, red._validate
    elif isinstance(red, Reduction):
        base, fwd, bwd = red, None, None
        if ranges is not None:
            r = base.with_ranges(ranges)
            fwd, bwd = r._fwd, r._bwd
    else:
        raise TypeError(f"reduce expects a kernox Reduction, got {type(red).__name__}")

    def local(reduction, kw, f=fwd, b=bwd):
        if f is None:
            return reduction(**kw)
        return RangedReduction(reduction, f, b, validate=validate)(**kw)

    if not (dist.is_available() and dist.is_initialized()):
        return local(base, inputs)
    size = dist.get_world_size(group)
    if size == 1:
        return local(base, inputs)
    rank = dist.get_rank(group)

    plan = base.plan
    roles = {a.name: a.role for a in plan.args}
    if inputs.keys() != roles.keys():
        missing = sorted(set(roles) - set(inputs))
        unexpected = sorted(set(inputs) - set(roles))
        raise ValidationError(f"missing arguments {missing}, unexpected {unexpected}")
    m = n = None
    for name, role in roles.items():
        v = inputs[name]
        if role == "param" or not torch.is_tensor(v) or v.dim() < 2:
            continue
        if role == "i" and m is None:
            m = v.shape[-2]
        elif role == "j" and n is None:
            n = v.shape[-2]
    split = _resolve_split(split, m, n, size)
    length = m if split == "output" else n
    if length is None or length < size:
        return local(base, inputs)

    side = "i" if split == "output" else "j"
    if fwd is None:
        span_list = _uniform(length, size)
    else:
        span_list = _range_spans(fwd, length, side, size)
    lo, hi = span_list[rank]

    copy, gather_rows = _fn()
    grad = torch.is_grad_enabled()
    kw = {}
    for name, v in inputs.items():
        if torch.is_tensor(v):
            if grad and v.requires_grad:
                v = copy.apply(v, group)
            if roles[name] == side:
                v = v.narrow(-2, lo, hi - lo)
        kw[name] = v

    if split == "output":
        f = _clip(fwd, lo, hi, "rows")
        b = _clip(bwd, lo, hi, "red")
        out = local(base, kw, f, b)

        def g(t):
            if t.requires_grad:
                return gather_rows.apply(t, rank, span_list, group)
            return _gather_rows(t, span_list, group)

        return tuple(g(t) for t in out) if isinstance(out, tuple) else g(out)

    f = _clip(fwd, lo, hi, "red")
    b = _clip(bwd, lo, hi, "rows")
    kind = plan.reduction.kind
    if kind in (K.SUM, K.COUNT):
        return _stack_live(local(base, kw, f, b), rank, group).sum(0)
    if kind is K.LOGSUMEXP:
        return _merge_lse(_stack_live(local(base, kw, f, b), rank, group))
    if kind in (K.MIN, K.ARGMIN, K.MIN_ARGMIN, K.MAX, K.ARGMAX, K.MAX_ARGMAX):
        largest = kind in (K.MAX, K.ARGMAX, K.MAX_ARGMAX)
        pair = K.MAX_ARGMAX if largest else K.MIN_ARGMIN
        v, idx = local(_companion(base, pair) if kind is not pair else base, kw, f, b)
        v, idx = _merge_select(v, _shift(idx, lo), rank, group, largest)
        if kind in (K.MIN, K.MAX):
            return v
        if kind in (K.ARGMIN, K.ARGMAX):
            return idx
        return v, idx
    if kind in (K.KMIN, K.ARGKMIN, K.KMIN_ARGKMIN):
        k = plan.reduction.k
        comp = base if kind is K.KMIN_ARGKMIN else _companion(base, K.KMIN_ARGKMIN)
        v, idx = local(comp, kw, f, b)
        v, idx = _merge_kmin(v, _shift(idx, lo), k, rank, group)
        if kind is K.KMIN:
            return v
        if kind is K.ARGKMIN:
            return idx
        return v, idx
    if kind in (K.SUMSOFTMAXWEIGHT, K.LOGSUMEXPWEIGHT):
        lse_r = local(_companion(base, K.LOGSUMEXP, outputs=plan.outputs[:1]), kw, f, b)
        s_r = local(base if kind is K.SUMSOFTMAXWEIGHT else _companion(base, K.SUMSOFTMAXWEIGHT), kw, f, b)
        lse, s = _softmax_combine(lse_r, s_r, rank, group)
        if kind is K.SUMSOFTMAXWEIGHT:
            return torch.where(torch.isfinite(lse) | torch.isnan(lse), s, torch.full_like(s, float("nan")))
        return lse + torch.log(s)
    raise ValidationError(f"reduce: unsupported reduction {kind.value}")


__all__ = ["init_from_env", "reduce", "spans", "SPLITS"]
