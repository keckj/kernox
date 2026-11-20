"""Stable key of a :class:`kernox.plan.Plan` (contract v0).

The key is the hexadecimal sha256 of a canonical JSON serialization of:

- the key format version (``PLAN_KEY_FORMAT``);
- for each argument, in ``plan.args`` order: name, role, dimension and the
  symox ``srepr`` of each component symbol (name + assumptions);
- the symox ``srepr`` of each output, in order; if the unfolded tree of an
  output exceeds ``DAG_THRESHOLD`` nodes (deep shared DAG: ``srepr`` would be
  exponential), the output is serialized as a graph: ``["dag", nodes]``, the
  list of distinct nodes in postfix order, each one being ``srepr`` for an atom
  or ``[class, head, child indices]`` otherwise. Keys of existing plans
  (smaller trees) are unchanged;
- reduction (kind, K), ``dtype``, ``acc_dtype``, ``index_dtype``, ``fast_math``, ``fp_contract``, ``flush_denormals`` (only when true);
- ``sum_scheme`` (amendment v0.7) only if it differs from ``"direct"``: the
  key of an existing plan is unchanged.
- ``batch_rank`` (amendment v0.6) only if nonzero: the key of a plan
  without batch is unchanged;
- ``ranges`` (amendment v0.6 §2) only if true.

``plan.meta`` is not part of the key (diagnostic information only).
The key depends neither on ``PYTHONHASHSEED`` nor on object creation order
(tested in ``tests/test_plan_key.py``). Backends add the generator version,
compiler, options and architecture to it (see the contract).
"""

from __future__ import annotations

import hashlib
import json

PLAN_KEY_FORMAT = "kernox-plan-key-v1"


#: unfolded tree size beyond which an output is serialized as a graph
DAG_THRESHOLD = 200_000


def _srepr(expr) -> str:
    from symox import srepr

    return srepr(expr)


def _tree_size(expr, limit):
    """Node count of the unfolded tree (memoized per subexpression), capped at ``limit``."""
    memo = {}

    def size(e):
        hit = memo.get(e)
        if hit is None:
            hit = min(limit, 1 + sum(size(a) for a in e.args))
            memo[e] = hit
        return hit

    return size(expr)


def _dag(expr) -> list:
    """Canonical graph serialization: distinct nodes in postfix order."""
    from symox import srepr

    index, nodes = {}, []

    def visit(e):
        hit = index.get(e)
        if hit is not None:
            return hit
        if not e.args:
            node = srepr(e)
        else:
            children = [visit(a) for a in e.args]
            node = [type(e).__name__, str(e.func), children]
        index[e] = len(nodes)
        nodes.append(node)
        return index[e]

    visit(expr)
    return ["dag", nodes]


def _output(expr):
    if _tree_size(expr, DAG_THRESHOLD + 1) > DAG_THRESHOLD:
        return _dag(expr)
    return _srepr(expr)


def plan_payload(plan) -> dict:
    """Canonical JSON structure hashed by :func:`plan_key` (useful for debugging)."""
    payload = {
        "format": PLAN_KEY_FORMAT,
        "args": [
            [a.name, a.role, int(a.dim), [_srepr(s) for s in a.symbols]]
            for a in plan.args
        ],
        "outputs": [_output(e) for e in plan.outputs],
        "reduction": [plan.reduction.kind.value, plan.reduction.k],
        "dtype": plan.dtype,
        "acc_dtype": plan.acc_dtype,
        "index_dtype": plan.index_dtype,
        "fast_math": bool(plan.fast_math),
        "fp_contract": bool(plan.fp_contract),
    }
    if getattr(plan, "flush_denormals", False):  # absent when off: earlier keys unchanged
        payload["flush_denormals"] = True
    scheme = getattr(plan, "sum_scheme", "direct")
    if scheme != "direct":
        payload["sum_scheme"] = scheme
    rank = int(getattr(plan, "batch_rank", 0))
    if rank:
        payload["batch_rank"] = rank
    if getattr(plan, "ranges", False):
        payload["ranges"] = True
    return payload


def plan_key(plan) -> str:
    """Hex sha256 of :func:`plan_payload`, canonically serialized."""
    text = json.dumps(plan_payload(plan), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()
