"""Block-sparse ranges, in the PyKeOps sense.

A `Ranges` holds, on the device of the inputs, the six integer tensors
of PyKeOps, in the **user's** orientation (i = ``Vi``, j = ``Vj``):

- ``ranges_i`` (Ci, 2): blocks of i rows ``[a, b)``;
- ``slices_i`` (Ci,): cumulative end, in ``redranges_j``, of the intervals of
  each block;
- ``redranges_j`` (Ri, 2): intervals ``[c, d)`` of j to reduce;
- ``ranges_j``, ``slices_j``, ``redranges_i``: the transposed triplet
  (reduction over i; gradients with respect to the j arguments). Optional.

For i in block ``c``: reduction over ``J(i) = ∪ redranges_j[slices_i[c−1]
: slices_i[c]]``; a row covered by no block receives the neutral element of the
reduction; ties go to the smallest j of ``J(i)``. Disjoint blocks and
disjoint intervals within a block: checked by ``with_ranges(r,
validate=True)``, otherwise an overlap is undefined behavior.

Call: ``run.with_ranges(r)(**inputs)`` (see `kernox.runtime`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ValidationError

_INT = ("torch.int32", "torch.int64")


@dataclass(frozen=True, eq=False)
class Triplet:
    """(rows, cumulative ends, reduced intervals), in the plan orientation."""

    rows: object
    slices: object
    red: object


@dataclass(frozen=True, eq=False)
class Ranges:
    ranges_i: object
    slices_i: object
    redranges_j: object
    ranges_j: object = None
    slices_j: object = None
    redranges_i: object = None

    def __post_init__(self):
        import torch

        tr = (self.ranges_j, self.slices_j, self.redranges_i)
        if any(t is None for t in tr) and not all(t is None for t in tr):
            raise ValidationError("ranges: the transposed triplet (ranges_j, slices_j, redranges_i) must be complete or absent")
        for name in ("ranges_i", "slices_i", "redranges_j", "ranges_j", "slices_j", "redranges_i"):
            t = getattr(self, name)
            if t is None:
                continue
            if not isinstance(t, torch.Tensor) or str(t.dtype) not in _INT:
                raise ValidationError(f"ranges.{name}: expected an int32 or int64 tensor")
        _check_triplet_shapes("i", self.ranges_i, self.slices_i, self.redranges_j)
        if self.ranges_j is not None:
            _check_triplet_shapes("j", self.ranges_j, self.slices_j, self.redranges_i)

    @property
    def has_transpose(self) -> bool:
        return self.ranges_j is not None

    @property
    def device(self):
        return self.ranges_i.device

    def triplet_i(self) -> Triplet:
        return Triplet(self.ranges_i, self.slices_i, self.redranges_j)

    def triplet_j(self) -> Triplet | None:
        return Triplet(self.ranges_j, self.slices_j, self.redranges_i) if self.has_transpose else None

    def oriented(self, swapped: bool):
        """(forward-pass triplet, transposed triplet or None) in the plan orientation.

        The plan always reduces over its j rows; ``swapped`` (``over="i"``)
        exchanges the roles, hence the triplets.
        """
        if not swapped:
            return self.triplet_i(), self.triplet_j()
        if not self.has_transpose:
            raise ValidationError(
                "ranges with over='i': the triplet (ranges_j, slices_j, redranges_i) is required "
                "(reduction over i)")
        return self.triplet_j(), self.triplet_i()

    @classmethod
    def dense(cls, m: int, n: int, device=None) -> "Ranges":
        """A single block, all rows, the whole other axis (equivalent to the call without ranges)."""
        import torch

        def t(v, shape):
            return torch.tensor(v, dtype=torch.int64, device=device).reshape(shape)

        return cls(t([0, m], (1, 2)), t([1], (1,)), t([0, n], (1, 2)),
                   t([0, n], (1, 2)), t([1], (1,)), t([0, m], (1, 2)))


def _check_triplet_shapes(side, rows, slices, red):
    if rows.dim() != 2 or rows.shape[1] != 2:
        raise ValidationError(f"ranges_{side}: expected shape (C, 2), got {tuple(rows.shape)}")
    if slices.dim() != 1 or slices.shape[0] != rows.shape[0]:
        raise ValidationError(f"slices_{side}: expected shape ({rows.shape[0]},), got {tuple(slices.shape)}")
    if red.dim() != 2 or red.shape[1] != 2:
        other = "j" if side == "i" else "i"
        raise ValidationError(f"redranges_{other}: expected shape (R, 2), got {tuple(red.shape)}")
    if not (rows.device == slices.device == red.device):
        raise ValidationError(f"ranges ({side}): tensors on different devices")


def _scan_unsafe(device) -> bool:
    """torch-xpu 2.15 on the G31 (Battlemage, driver 26.05): ``nonzero``,
    single-argument ``torch.where``, ``bincount``, ``searchsorted`` and
    ``repeat_interleave`` return wrong results (checked with both the
    stock and an overridden SYCL runtime; ``cumsum`` and ``scatter_add_``
    are correct). The small structural computations on ranges are therefore
    done on the host for an XPU device, then copied back."""
    import torch

    return torch.device(device).type == "xpu"


def ranges_from_labels(labels_i, labels_j, keep) -> Ranges:
    """Block ranges from **sorted** labels and a matrix of kept blocks.

    ``labels_i`` (M,), ``labels_j`` (N,): non-decreasing integers in
    ``[0, Ci)`` / ``[0, Cj)`` (points sorted by cluster); ``keep`` (Ci, Cj)
    boolean. Functional equivalent of ``pykeops.torch.cluster.from_matrix``
    applied to the cluster bounds; both triplets are built.
    """
    import torch

    keep = torch.as_tensor(keep)
    if _scan_unsafe(keep.device):
        dev = keep.device
        r = ranges_from_labels(torch.as_tensor(labels_i).cpu(), torch.as_tensor(labels_j).cpu(), keep.cpu())
        return Ranges(*(t.to(dev) for t in (r.ranges_i, r.slices_i, r.redranges_j,
                                             r.ranges_j, r.slices_j, r.redranges_i)))
    if keep.dim() != 2 or keep.dtype != torch.bool:
        raise ValidationError("keep: expected a boolean (Ci, Cj) matrix")
    ci, cj = keep.shape

    def bounds(labels, c, name):
        labels = torch.as_tensor(labels).to(device=keep.device, dtype=torch.int64)
        if labels.dim() != 1:
            raise ValidationError(f"{name}: expected a vector")
        counts = torch.zeros(c, dtype=torch.int64, device=keep.device)
        if labels.numel():
            # Indices outside [0, c) are clamped into range for counting; the
            # validation below rejects them (a single synchronization).
            counts.index_add_(0, labels.clamp(0, max(c - 1, 0)), torch.ones_like(labels))
        ends = torch.cumsum(counts, 0)
        return labels, torch.stack([ends - counts, ends], dim=1)

    (li, ri), (lj, rj) = bounds(labels_i, ci, "labels_i"), bounds(labels_j, cj, "labels_j")

    def bad(labels, c):
        if not labels.numel():
            return torch.zeros((), dtype=torch.bool, device=keep.device)
        return (labels[1:] < labels[:-1]).any() | (labels.min() < 0) | (labels.max() >= c)

    # Validate both labelings in a single host transfer (2 booleans).
    flags = torch.stack([bad(li, ci), bad(lj, cj)]).tolist()
    for flag, name, c in zip(flags, ("labels_i", "labels_j"), (ci, cj)):
        if flag:
            raise ValidationError(f"{name}: expected non-decreasing labels in [0, {c})")
    # Kept blocks: everything on the device (the host path, for XPU, is above).
    cols = keep.nonzero()[:, 1]
    rows = keep.t().nonzero()[:, 1]
    return Ranges(
        ri, torch.cumsum(keep.sum(1), 0), rj[cols].reshape(-1, 2),
        rj, torch.cumsum(keep.sum(0), 0), ri[rows].reshape(-1, 2),
    )


def validate_triplet(tr: Triplet, m: int, n: int, side: str) -> None:
    """Values of a triplet (plan: ``m`` rows reduced over ``n``); one host synchronization."""
    import torch

    rows, slices, red = (t.to(torch.int64) for t in (tr.rows, tr.slices, tr.red))
    if _scan_unsafe(rows.device):
        rows, slices, red = rows.cpu(), slices.cpu(), red.cpu()
    c, r = rows.shape[0], red.shape[0]
    problems = []
    if c:
        a, b = rows[:, 0], rows[:, 1]
        if bool(((a < 0) | (b < a) | (b > m)).any()):
            problems.append(f"row block outside [0, {m}] or reversed")
        live = b > a  # empty blocks overlap nothing
        order = torch.argsort(a[live])
        al, bl = a[live][order], b[live][order]
        if al.numel() > 1 and bool((al[1:] < bl[:-1]).any()):
            problems.append("overlapping row blocks")
        if bool((slices[1:] < slices[:-1]).any()) or int(slices[0]) < 0 or int(slices[-1]) != r:
            problems.append(f"decreasing slices or last end different from {r}")
    elif r:
        problems.append("intervals without any row block")
    if r and not problems:
        cc, d = red[:, 0], red[:, 1]
        if bool(((cc < 0) | (d < cc) | (d > n)).any()):
            problems.append(f"reduced interval outside [0, {n}]")
        else:
            block = torch.repeat_interleave(torch.arange(c, device=red.device), torch.diff(slices, prepend=slices.new_zeros(1)))
            live = d > cc  # empty intervals overlap nothing
            block, cc, d = block[live], cc[live], d[live]
            key = block * (n + 1) + cc
            order = torch.argsort(key)
            same = block[order][1:] == block[order][:-1]
            if bool((same & (cc[order][1:] < d[order][:-1])).any()):
                problems.append("overlapping intervals within a block")
    if problems:
        raise ValidationError(f"invalid ranges ({side}): " + "; ".join(problems))


def work_list(tr: Triplet, tile_m: int):
    """Tiles ``(i0, i1, k0, k1)`` int64 (W, 4), height ≤ ``tile_m``, on the device of the triplet.

    ``[k0, k1)``: intervals of ``tr.red`` to traverse. Torch computation; the
    size W requires a host synchronization (the runtime caches the result
    by identity of the ranges tensors).
    """
    import torch

    rows = tr.rows.to(torch.int64)
    slices = tr.slices.to(torch.int64)
    dev = rows.device
    if _scan_unsafe(dev):  # host computation (repeat_interleave is wrong on XPU), copied to the device
        return work_list(Triplet(rows.cpu(), slices.cpu(), tr.red.cpu()), tile_m).to(dev)
    c = rows.shape[0]
    if c == 0:
        return torch.zeros((0, 4), dtype=torch.int64, device=dev)
    a, b = rows[:, 0], rows[:, 1]
    ntiles = torch.clamp((b - a + tile_m - 1) // tile_m, min=0)
    blk = torch.repeat_interleave(torch.arange(c, device=dev), ntiles)
    first = torch.cumsum(ntiles, 0) - ntiles
    t = torch.arange(blk.numel(), device=dev) - first[blk]
    i0 = a[blk] + t * tile_m
    i1 = torch.minimum(i0 + tile_m, b[blk])
    k1 = slices[blk]
    k0 = torch.where(blk > 0, slices[(blk - 1).clamp_min(0)], torch.zeros_like(k1))
    return torch.stack([i0, i1, k0, k1], dim=1).contiguous()


__all__ = ["Ranges", "Triplet", "ranges_from_labels", "validate_triplet", "work_list"]
