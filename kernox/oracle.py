"""Reference oracles (core), **not** an optimized GPU execution path.

1. :class:`OracleBackend` (name ``"oracle"``): blockwise Torch reference with
   bounded blocks, on the tensors' device (CPU or GPU). It is never selected
   by ``backend="auto"``; it must be requested explicitly.
2. :func:`reference_numpy`: small FP64 NumPy oracle (full M×N grid, Python
   loops), an independent implementation of the contract semantics, meant
   for small test cases only.

Semantics (``docs/kernel-backend-contract.md``): SUM propagates NaN;
COUNT is int64; MIN/MAX ignore NaN candidates (neutral ±inf); ARG*: smallest
index on ties, −1 when there is no candidate; KMIN/ARGKMIN: lexicographic
order (value, j), NaN ignored, padded with (+inf, −1);
LOGSUMEXP / SUMSOFTMAXWEIGHT (amendment v0.5): state (m, s[, t]) shifted by
the running maximum, terms with ``f = −inf`` have zero weight, NaN propagated,
``+inf`` → ``+inf`` / NaN, empty → ``−inf`` / NaN.
"""

from __future__ import annotations

import math
import time

from . import _symox_compat
from .backends.base import CompileInfo, Support, UnsupportedError
from .plan import Plan, ReductionKind, output_shapes

_K_KINDS = (ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN)
_SOFT_KINDS = (ReductionKind.LOGSUMEXP, ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT)
_WEIGHTED_KINDS = (ReductionKind.SUMSOFTMAXWEIGHT, ReductionKind.LOGSUMEXPWEIGHT)  # outputs (f, g_1..g_E)
_MIN_KINDS = (ReductionKind.MIN, ReductionKind.ARGMIN, ReductionKind.MIN_ARGMIN)
_MAX_KINDS = (ReductionKind.MAX, ReductionKind.ARGMAX, ReductionKind.MAX_ARGMAX)


def _soft_finish(kind, m, s, t):
    import torch

    inf = float("inf")
    if kind is ReductionKind.LOGSUMEXP:
        r = m + torch.log(s)
        r = torch.where(m == inf, torch.full_like(r, inf), r)
        return torch.where(torch.isnan(s), torch.full_like(r, float("nan")), r)
    if kind is ReductionKind.LOGSUMEXPWEIGHT:
        # m + log t_e with no special case: empty → −inf; f = +inf → +inf if t_e > 0, NaN otherwise
        return m.unsqueeze(-1) + torch.log(t)
    r = t / s.unsqueeze(-1)
    return torch.where((m == inf).unsqueeze(-1), torch.full_like(r, float("nan")), r)
DEFAULT_BLOCK_PAIRS = 1 << 20
#: amendment v0.7: tile size T of the oracle's ``block`` scheme (published in
#: ``CompileInfo.options``) and segment size of the ``kahan`` summation.
SUM_BLOCK = 256


class _Accumulator:
    """Per-row, per-component sums following ``sum_scheme`` (amendment v0.7).

    Faithful simulation in ``acc_dtype`` (NumPy, additions rounded one by one):

    - ``direct``: one sequential accumulator over j (``np.add.accumulate`` does
      not reassociate, unlike ``sum`` or ``torch.cumsum`` on CPU);
    - ``block``: each tile of ``SUM_BLOCK`` consecutive terms (aligned on j
      from 0) is summed into a fresh accumulator, then added to the total;
    - ``kahan``: Kahan-Babuška (Neumaier), state ``(s, c)``; segments of
      ``SUM_BLOCK`` terms summed in parallel, then merged in j order with
      ``TwoSum(s1, s2)`` and ``c1 + c2`` (the contract's merge rule).

    Result ``s + c`` (``s`` alone if ``s`` is not finite: a ``±inf`` or a NaN
    is not spoiled by an ``inf − inf`` compensation).
    """

    def __init__(self, scheme: str, shape, dtype):
        import numpy as np

        self.scheme, self.dtype = scheme, dtype
        self.s = np.zeros(shape, dtype)
        self.c = np.zeros(shape, dtype) if scheme == "kahan" else None

    def scale(self, f):
        self.s = (self.s * f).astype(self.dtype)
        if self.c is not None:
            self.c = (self.c * f).astype(self.dtype)

    @staticmethod
    def _pad(v, size):
        import numpy as np

        cols = v.shape[1]
        if cols % size == 0:
            return v
        pad = np.zeros((v.shape[0], size - cols % size) + v.shape[2:], v.dtype)
        return np.concatenate([v, pad], axis=1)

    def add(self, v):
        """Adds the terms ``v`` (rows, cols, …), in increasing j order."""
        import numpy as np

        v = np.asarray(v, dtype=self.dtype)
        if v.shape[1] == 0:
            return
        with np.errstate(all="ignore"):
            if self.scheme == "direct":
                seq = np.concatenate([self.s[:, None], v], axis=1)
                self.s = np.add.accumulate(seq, axis=1)[:, -1]
            elif self.scheme == "block":
                t = self._pad(v, SUM_BLOCK)
                t = t.reshape((t.shape[0], -1, SUM_BLOCK) + t.shape[2:])
                partial = np.add.accumulate(t, axis=2)[:, :, -1]
                seq = np.concatenate([self.s[:, None], partial], axis=1)
                self.s = np.add.accumulate(seq, axis=1)[:, -1]
            else:
                t = self._pad(v, SUM_BLOCK)
                t = t.reshape((t.shape[0], -1, SUM_BLOCK) + t.shape[2:])
                s = np.zeros(t[:, :, 0].shape, self.dtype)
                c = np.zeros_like(s)
                for k in range(SUM_BLOCK):
                    x = t[:, :, k]
                    u = s + x
                    c = c + np.where(np.abs(s) >= np.abs(x), (s - u) + x, (x - u) + s)
                    s = u
                for k in range(s.shape[1]):
                    self.s, self.c = _two_sum_merge(self.s, self.c, s[:, k], c[:, k])

    def result(self):
        import numpy as np

        if self.c is None:
            return self.s
        with np.errstate(all="ignore"):
            return np.where(np.isfinite(self.s), self.s + self.c, self.s)


def _two_sum_merge(s1, c1, s2, c2):
    """Merges two Neumaier states: ``TwoSum(s1, s2)``, then ``c1 + c2 + error``."""
    t = s1 + s2
    bp = t - s1
    err = (s1 - (t - bp)) + (s2 - bp)
    return t, c1 + c2 + err


def _flat_symbols(plan: Plan):
    return [s for a in plan.args for s in a.symbols]


def _torch_function(symbols, outputs):
    """Torch function of ``outputs`` (same call as ``torch_lambdify``).

    Without ``Piecewise``, ``torch_lambdify`` performs the CSE itself. With a
    ``Piecewise`` (``where``, gradients of ``min``/``max``), symox disables it
    (a shared temporary could evaluate an inactive branch before its mask):
    a deep DAG would then be printed as an unfolded tree, of exponential size.
    We apply the region-based CSE of the GPU backends
    (``codegen.pair._region_cse``, same hoisting policy), then the temporaries
    are evaluated by dependency level, one ``torch_lambdify`` function per
    level (small values; inputs: arguments and temporaries of the previous
    levels).
    """
    from symox.extra import torch_lambdify
    from symox.extra.codegen.pytorch import _contains_conditionals

    if not _contains_conditionals(outputs):
        return torch_lambdify(symbols, outputs, cse=True)
    from .codegen.pair import _region_cse

    temps, outs = _region_cse(outputs, prefix="_kp_o")
    if not temps:
        return torch_lambdify(symbols, outputs, cse=True)
    level = {}
    for sym, value in temps:
        level[sym] = 1 + max((level[t] for t in value.free_symbols if t in level), default=-1)
    depth = max(level.values()) + 1
    stages, known = [], list(symbols)
    for d in range(depth):
        group = [(sym, value) for sym, value in temps if level[sym] == d]
        stages.append(torch_lambdify(list(known), [value for _, value in group], cse=False))
        known += [sym for sym, _ in group]
    final = torch_lambdify(list(known), list(outs), cse=True)

    def run(*values, dtype=None):
        vals = list(values)
        for fn in stages:
            vals += list(fn(*vals, dtype=dtype))
        return final(*vals, dtype=dtype)

    return run


def _numpy_dtype(name):
    import numpy as np

    return {"float16": np.float16, "float32": np.float32, "float64": np.float64}[name]


def _torch_dtype(name):
    import torch

    return {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64, "int64": torch.int64}[name]


# --------------------------------------------------------------------------
# Blockwise Torch oracle
# --------------------------------------------------------------------------


class OracleReduction:
    """Compiled reference reduction (Torch function generated by symox)."""

    def __init__(self, plan: Plan, device, block_pairs: int = DEFAULT_BLOCK_PAIRS):
        import torch

        t0 = time.perf_counter()
        self.plan = plan
        self.block_pairs = int(block_pairs)
        self._fn = _torch_function(_flat_symbols(plan), list(plan.outputs))
        dev = torch.device(device)
        itemsize = {"float16": 2, "float32": 4, "float64": 8}[plan.dtype]
        nsym = len(_flat_symbols(plan))
        self.info = CompileInfo(
            backend="oracle",
            device_name=str(dev),
            arch=f"torch-{dev.type}",
            compiler=f"torch {torch.__version__} (symox torch_lambdify)",
            options=(f"block_pairs={self.block_pairs}", f"sum_scheme={plan.sum_scheme}")
            + ((f"sum_block={SUM_BLOCK}",) if plan.sum_scheme == "block" else ()),
            cache_hit=False,
            compile_seconds=time.perf_counter() - t0,
            # estimate: block values + broadcast inputs, excluding Torch temporaries
            temp_bytes_per_launch=self.block_pairs * (plan.out_dim + nsym) * itemsize,
            source_sha256="",
        )

    def _evaluate(self, inputs, i0, i1, j0, j1):
        """Formula values (bi, bj, D) on the block, in the plan dtype (or bool)."""
        import torch

        plan, values = self.plan, []
        for a in plan.args:
            t = inputs[a.name]
            for c in range(a.dim):
                if a.role == "i":
                    values.append(t[i0:i1, c].unsqueeze(1))
                elif a.role == "j":
                    values.append(t[j0:j1, c].unsqueeze(0))
                else:
                    values.append(t[c])
        dtype = _torch_dtype(plan.dtype)
        outs = self._fn(*values, dtype=dtype)
        shape = (i1 - i0, j1 - j0)
        cols = []
        for o in outs:
            o = torch.as_tensor(o)
            if o.dtype != torch.bool:
                o = o.to(dtype)
            cols.append(o.expand(shape))
        return torch.stack(cols, dim=-1)

    #: maximum tile height of the ranges work list (amendment v0.6 §2)
    ranges_tile_m = 1024

    def launch(self, inputs: dict, outputs: dict, batch=None, ranges=None) -> None:
        """Reference reduction; ``batch`` (amendment v0.6): loop over the elements.

        Each element ``b`` is an ``as_strided`` view of the input at the offset
        published by the runtime (``BatchLayout.host_offsets``); the output
        ``(nb, M, …)`` is written element by element.
        """
        if ranges is not None:
            if batch is not None:
                raise UnsupportedError("ranges combined with batch: rejected (amendment v0.6)")
            self._launch_ranged(inputs, outputs, ranges)
            return
        if batch is None:
            self._launch_one(inputs, outputs)
            return
        plan = self.plan
        offsets = {name: off.tolist() for name, off in batch.host_offsets.items()}
        for b in range(batch.nb):
            sub = {}
            for a in plan.args:
                t = inputs[a.name]
                o = t.storage_offset() + offsets[a.name][b]
                if a.role == "param":
                    sub[a.name] = t.as_strided((a.dim,), (1,), o)
                else:
                    sub[a.name] = t.as_strided((t.shape[-2], a.dim), (a.dim, 1), o)
            self._launch_one(sub, {key: out[b] for key, out in outputs.items()})

    def _launch_ranged(self, inputs: dict, outputs: dict, ranges) -> None:
        """Ranges reference: neutral values everywhere, then each tile reduced over sorted ``J``.

        ``J`` is the sorted union of the tile's intervals: ties go to the
        smallest j of ``J(i)`` whatever the order of the intervals, and the
        returned indices are global.
        """
        import torch

        plan = self.plan
        # N = 0 on every row: contract neutral values (rows not covered)
        self._launch_one({a.name: (inputs[a.name][:0] if a.role == "j" else inputs[a.name]) for a in plan.args},
                         outputs)
        device = next(iter(inputs.values())).device
        red = ranges.host_redranges.tolist()
        for i0, i1, k0, k1 in ranges.host_work.tolist():
            if i1 <= i0:
                continue
            parts = [torch.arange(c, d, dtype=torch.int64) for c, d in red[k0:k1] if d > c]
            jidx = (torch.sort(torch.cat(parts)).values if parts else torch.zeros(0, dtype=torch.int64)).to(device)
            sub = {}
            for a in plan.args:
                t = inputs[a.name]
                sub[a.name] = t[i0:i1] if a.role == "i" else (t.index_select(0, jidx) if a.role == "j" else t)
            tmp = {key: torch.empty((i1 - i0,) + tuple(o.shape[1:]), dtype=o.dtype, device=o.device)
                   for key, o in outputs.items()}
            self._launch_one(sub, tmp)
            for key, v in tmp.items():
                if key == "index" and jidx.numel():
                    v = torch.where(v >= 0, jidx[v.clamp_min(0)], v)
                outputs[key][i0:i1] = v

    def _launch_one(self, inputs: dict, outputs: dict) -> None:
        import torch

        plan = self.plan
        kind, k = plan.reduction.kind, plan.reduction.k
        m = next(inputs[a.name].shape[0] for a in plan.args if a.role == "i")
        n = next(inputs[a.name].shape[0] for a in plan.args if a.role == "j")
        if m == 0:
            return
        if kind in _K_KINDS and k == 0:
            return
        device = next(iter(inputs.values())).device
        bi = max(1, min(m, 1024))
        bj = max(1, min(max(n, 1), self.block_pairs // bi))
        if bj >= SUM_BLOCK:
            bj -= bj % SUM_BLOCK  # block-scheme tiles / kahan segments aligned on j
        np_adt = _numpy_dtype(plan.acc_dtype)
        d = plan.out_dim
        dt = _torch_dtype(plan.dtype)
        inf = float("inf")
        for i0 in range(0, m, bi):
            i1 = min(m, i0 + bi)
            rows = i1 - i0
            # accumulators initialized to the neutral values
            if kind is ReductionKind.SUM:
                adt = _torch_dtype(plan.acc_dtype)
                acc = _Accumulator(plan.sum_scheme, (rows, d), np_adt)
            elif kind in _SOFT_KINDS:
                adt = _torch_dtype(plan.acc_dtype)
                dm = d if kind is ReductionKind.LOGSUMEXP else 1
                sm = torch.full((rows, dm), -inf, dtype=adt, device=device)
                ss = _Accumulator(plan.sum_scheme, (rows, dm), np_adt)
                st = (_Accumulator(plan.sum_scheme, (rows, d - 1), np_adt)
                      if kind in _WEIGHTED_KINDS else None)
            elif kind is ReductionKind.COUNT:
                acc = torch.zeros((rows, d), dtype=torch.int64, device=device)
            elif kind in _MIN_KINDS:
                best = torch.full((rows, d), inf, dtype=dt, device=device)
                bidx = torch.full((rows, d), -1, dtype=torch.int64, device=device)
            elif kind in _MAX_KINDS:
                best = torch.full((rows, d), -inf, dtype=dt, device=device)
                bidx = torch.full((rows, d), -1, dtype=torch.int64, device=device)
            else:
                kval = torch.full((rows, k), inf, dtype=dt, device=device)
                kidx = torch.full((rows, k), -1, dtype=torch.int64, device=device)
                kabs = torch.ones((rows, k), dtype=torch.bool, device=device)  # missing candidate
            for j0 in range(0, n, bj):
                j1 = min(n, j0 + bj)
                v = self._evaluate(inputs, i0, i1, j0, j1)  # (rows, cols, d)
                if kind is ReductionKind.SUM:
                    acc.add(v.to(adt).cpu().numpy())
                elif kind in _SOFT_KINDS:
                    # running maximum; the states (s, c) and (t, c_t) are rescaled by
                    # exp(m_old − m_new) before adding the block's terms (amendment v0.7)
                    va = v.to(adt)
                    f, g = (va, None) if kind is ReductionKind.LOGSUMEXP else (va[..., :1], va[..., 1:])
                    live = f != -inf  # NaN included: propagated
                    bm = torch.where(live & ~torch.isnan(f), f, torch.full_like(f, -inf)).amax(dim=1)
                    mn = torch.maximum(sm, bm)
                    rescale = torch.where(sm == mn, torch.ones_like(mn), torch.exp(sm - mn)).cpu().numpy()
                    ss.scale(rescale)
                    w = torch.where(f == mn.unsqueeze(1), torch.ones_like(f), torch.exp(f - mn.unsqueeze(1)))
                    w = torch.where(live, w, torch.zeros_like(w))
                    ss.add(w.cpu().numpy())
                    if st is not None:
                        st.scale(rescale)
                        st.add(torch.where(live, w * g, torch.zeros_like(g)).cpu().numpy())
                    sm = mn
                elif kind is ReductionKind.COUNT:
                    acc += v.sum(dim=1, dtype=torch.int64)
                elif kind in _MIN_KINDS + _MAX_KINDS:
                    is_min = kind in _MIN_KINDS
                    valid = ~torch.isnan(v)
                    fill = inf if is_min else -inf
                    masked = torch.where(valid, v, torch.full_like(v, fill))
                    bv = masked.amin(dim=1) if is_min else masked.amax(dim=1)  # (rows, d)
                    cand = valid & (v == bv.unsqueeze(1))
                    has = cand.any(dim=1)
                    first = cand.to(torch.int8).argmax(dim=1) + j0  # first True
                    better = bv < best if is_min else bv > best
                    take = has & ((bidx < 0) | better)
                    best = torch.where(take, bv, best)
                    bidx = torch.where(take, first, bidx)
                else:
                    vv = v[..., 0]
                    cols = j1 - j0
                    absent = torch.isnan(vv)
                    vals = torch.where(absent, torch.full_like(vv, inf), vv)
                    idx = torch.arange(j0, j1, device=device, dtype=torch.int64).expand(rows, cols)
                    allv = torch.cat([kval, vals], dim=1)
                    alli = torch.cat([kidx, idx], dim=1)
                    alla = torch.cat([kabs, absent], dim=1)
                    # lexicographic sort (missing, value, index) via successive stable sorts
                    order = torch.sort(alli, dim=1, stable=True).indices
                    allv, alli, alla = (t.gather(1, order) for t in (allv, alli, alla))
                    # normalized key: -0.0 + 0.0 = +0.0, so that signed zeros tie
                    # (GPU radix sorts would tell them apart otherwise)
                    order = torch.sort(allv + 0.0, dim=1, stable=True).indices
                    allv, alli, alla = (t.gather(1, order) for t in (allv, alli, alla))
                    order = torch.sort(alla.to(torch.int8), dim=1, stable=True).indices
                    allv, alli, alla = (t.gather(1, order) for t in (allv, alli, alla))
                    kval, kidx, kabs = allv[:, :k], alli[:, :k], alla[:, :k]
            # write the outputs
            if kind is ReductionKind.SUM:
                outputs["value"][i0:i1] = torch.from_numpy(acc.result()).to(device)
            elif kind is ReductionKind.COUNT:
                outputs["value"][i0:i1] = acc
            elif kind is ReductionKind.LOGSUMEXP:
                outputs["value"][i0:i1] = _soft_finish(kind, sm, torch.from_numpy(ss.result()).to(device), None)
            elif kind is ReductionKind.SUMSOFTMAXWEIGHT:
                s_, t_ = (torch.from_numpy(a.result()).to(device) for a in (ss, st))
                outputs["value"][i0:i1] = _soft_finish(kind, sm, s_, t_.unsqueeze(1))[:, 0, :]
            elif kind is ReductionKind.LOGSUMEXPWEIGHT:
                t_ = torch.from_numpy(st.result()).to(device)
                outputs["value"][i0:i1] = _soft_finish(kind, sm[:, 0], None, t_)
            elif kind in _MIN_KINDS + _MAX_KINDS:
                if "value" in outputs:
                    outputs["value"][i0:i1] = best
                if "index" in outputs:
                    outputs["index"][i0:i1] = bidx
            else:
                kval = torch.where(kabs, torch.full_like(kval, inf), kval)
                kidx = torch.where(kabs, torch.full_like(kidx, -1), kidx)
                if "value" in outputs:
                    outputs["value"][i0:i1] = kval
                if "index" in outputs:
                    outputs["index"][i0:i1] = kidx


class OracleBackend:
    """Explicit reference backend: ``backend="oracle"`` only."""

    name = "oracle"
    # Torch CPU rounds every elementary operation: no FMA contraction.
    fp_contract_off = True
    # amendment v0.7: all three schemes, simulated in acc_dtype (see _Accumulator)
    sum_schemes = ("direct", "block", "kahan")
    # amendment v0.6: batch dimensions (reference loop over the elements)
    batch = True
    # amendment v0.6 §2: block-sparse ranges (masked reference reduction)
    ranges = True

    def __init__(self, block_pairs: int = DEFAULT_BLOCK_PAIRS):
        self.block_pairs = block_pairs

    def supports(self, plan: Plan, device) -> Support:
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover
            return Support(False, "torch unavailable")
        return Support(True)

    def compile(self, plan: Plan, device) -> OracleReduction:
        return OracleReduction(plan, device, self.block_pairs)


# --------------------------------------------------------------------------
# FP64 NumPy oracle (small cases)
# --------------------------------------------------------------------------


def evaluate_numpy(plan: Plan, inputs: dict):
    """Formula values (M, N, D) in FP64 (booleans for COUNT)."""
    import numpy as np
    import symox

    m = next(np.asarray(inputs[a.name]).shape[0] for a in plan.args if a.role == "i")
    n = next(np.asarray(inputs[a.name]).shape[0] for a in plan.args if a.role == "j")
    # Floats are passed as arguments: lambdify (like SymPy) prints them with
    # 15 significant digits, which corrupts the last FP64 ulp.
    floats = sorted({f for e in plan.outputs for f in e.atoms(symox.Float)}, key=float)
    holders = [symox.Dummy(f"c{i}", real=True) for i in range(len(floats))]
    outputs = [e.xreplace(dict(zip(floats, holders))) for e in plan.outputs]
    fn = symox.lambdify(list(_flat_symbols(plan)) + holders, outputs, modules="numpy")
    values = []
    for a in plan.args:
        t = np.asarray(_to_numpy(inputs[a.name]), dtype=np.float64)
        for c in range(a.dim):
            if a.role == "i":
                col = t[:, c].reshape(m, 1)
            elif a.role == "j":
                col = t[:, c].reshape(1, n)
            else:
                col = np.float64(t.reshape(-1)[c])
            if _symox_compat.lambdify_broadcasts():
                values.append(col)
            else:
                # native lambdify of the pinned commit: no broadcasting, explicit (M, N) grids
                values.append(np.ascontiguousarray(np.broadcast_to(col, (m, n))))
    values += [np.float64(float(f)) for f in floats]
    with np.errstate(all="ignore"):
        outs = fn(*values)
    cols = [np.broadcast_to(np.asarray(o), (m, n)) for o in outs]
    boolean = plan.reduction.kind is ReductionKind.COUNT
    return np.stack([c.astype(bool if boolean else np.float64) for c in cols], axis=-1)


def _to_numpy(t):
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return t


def reduce_numpy(plan: Plan, vals):
    """Applies the contract semantics to values (M, N, D), in pure Python."""
    import numpy as np

    kind, k = plan.reduction.kind, plan.reduction.k
    m, n, d = vals.shape
    inf = math.inf
    if kind is ReductionKind.SUM:
        return {"value": vals.sum(axis=1) if n else np.zeros((m, d))}
    if kind is ReductionKind.COUNT:
        return {"value": vals.sum(axis=1, dtype=np.int64) if n else np.zeros((m, d), np.int64)}
    if kind in _SOFT_KINDS:
        return {"value": _soft_numpy(kind, vals)}
    if kind in _MIN_KINDS + _MAX_KINDS:
        is_min = kind in _MIN_KINDS
        value = np.full((m, d), inf if is_min else -inf)
        index = np.full((m, d), -1, dtype=np.int64)
        for i in range(m):
            for c in range(d):
                for j in range(n):
                    x = float(vals[i, j, c])
                    if math.isnan(x):
                        continue
                    if index[i, c] < 0 or (x < value[i, c] if is_min else x > value[i, c]):
                        value[i, c], index[i, c] = x, j
        out = {"value": value, "index": index}
        return {key: out[key] for key in output_shapes(plan, m)}
    value = np.full((m, k), inf)
    index = np.full((m, k), -1, dtype=np.int64)
    for i in range(m):
        cands = sorted((float(vals[i, j, 0]), j) for j in range(n) if not math.isnan(vals[i, j, 0]))
        for r, (x, j) in enumerate(cands[:k]):
            value[i, r], index[i, r] = x, j
    out = {"value": value, "index": index}
    return {key: out[key] for key in output_shapes(plan, m)}


def _soft_numpy(kind, vals):
    """LOGSUMEXP / SUMSOFTMAXWEIGHT in pure Python (``math.fsum``), independent of the Torch oracle."""
    import numpy as np

    m, n, d = vals.shape
    if kind is ReductionKind.LOGSUMEXPWEIGHT:
        return _lsew_numpy(vals)
    lse = kind is ReductionKind.LOGSUMEXP
    out = np.empty((m, d if lse else d - 1))
    for i in range(m):
        for c in range(d if lse else 1):
            f = [float(vals[i, j, c]) for j in range(n)]
            if any(math.isnan(x) for x in f):
                out[i, c if lse else slice(None)] = math.nan
                continue
            if any(x == math.inf for x in f):
                out[i, c if lse else slice(None)] = math.inf if lse else math.nan
                continue
            live = [j for j in range(n) if f[j] != -math.inf]
            if not live:
                out[i, c if lse else slice(None)] = -math.inf if lse else math.nan
                continue
            mx = max(f[j] for j in live)
            w = {j: math.exp(f[j] - mx) for j in live}
            s = math.fsum(w.values())
            if lse:
                out[i, c] = mx + math.log(s)
            else:
                for e in range(d - 1):
                    out[i, e] = math.fsum(w[j] * float(vals[i, j, 1 + e]) for j in live) / s
    return out


def _lsew_numpy(vals):
    """LOGSUMEXPWEIGHT: ``m + log Σ_j exp(f_j − m) g_je`` (``math.fsum``), m = max f.

    No special case after the shift: f NaN → NaN; f = +inf →
    ``+inf + log Σ_{f=+inf} g`` (+inf if > 0, NaN otherwise); empty → −inf.
    """
    import numpy as np

    m, n, d = vals.shape
    out = np.empty((m, d - 1))
    for i in range(m):
        f = [float(vals[i, j, 0]) for j in range(n)]
        if any(math.isnan(x) for x in f):
            out[i] = math.nan
            continue
        live = [j for j in range(n) if f[j] != -math.inf]
        mx = max((f[j] for j in live), default=-math.inf)
        w = {j: (1.0 if f[j] == mx else math.exp(f[j] - mx)) for j in live}
        for e in range(d - 1):
            t = math.fsum(w[j] * float(vals[i, j, 1 + e]) for j in live) if live else 0.0
            if math.isnan(t):
                out[i, e] = math.nan
            elif t < 0:
                out[i, e] = math.nan
            else:
                out[i, e] = mx + (math.log(t) if t > 0 else -math.inf)
    return out


def reference_numpy(plan: Plan, inputs: dict) -> dict:
    """Full FP64 oracle (small M, N): dict of the contract outputs (NumPy)."""
    return reduce_numpy(plan, evaluate_numpy(plan, inputs))


__all__ = ["OracleBackend", "OracleReduction", "reference_numpy", "evaluate_numpy", "reduce_numpy"]
