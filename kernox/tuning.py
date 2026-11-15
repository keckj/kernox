"""Optional autotuning of kernel variants.

Levels (``compile(..., tuning=...)``):

- ``"estimate"`` (default): ``backend.default_variant``; no measurement, no
  wisdom read; deterministic;
- ``"wisdom_only"``: variant from the wisdom if a valid entry exists for the
  key, otherwise ``estimate``; no measurement;
- ``"measure"`` / ``"patient"``: wisdom if present; otherwise measurement of at most
  4 / 32 candidates from ``backend.variants(...)`` (list truncated and overflow
  logged), on the actual inputs of the call, and recording if the
  measurement is stable.

Measurement: **blocking** call, only once per size class
``(ceil(2 log2 M), ceil(2 log2 N), ceil(2 log2 nb), ceil(2 log2 1/ρ))``
(half-octaves; wisdom v3: ``nb`` batch elements, density ``ρ`` of the block-sparse
ranges, 0 and 0 without batch or ranges) and per compiled object; compilation excluded from
timing (published separately); 1 warm-up then ``REPEATS`` launches
timed with GPU events on the current stream (host clock on CPU);
bounded total time budget (``BUDGET_SECONDS``, compilation included); at
least the default variant is measured. If the relative spread
``(max - min) / median`` of one of the two best variants exceeds
``MAX_SPREAD``, the decision is marked unstable and **is not persisted**.
The result of the call is always recomputed with the selected variant.

Wisdom: one JSON file per device under ``$KERNOX_CACHE_DIR/wisdom`` (otherwise
``$XDG_CACHE_HOME/kernox/wisdom`` or ``~/.cache/kernox/wisdom``), written
atomically (temporary file + ``os.replace``) under an ``fcntl`` lock. An
entry with a different format or different versions (device, driver, compiler, generator)
is ignored, as is an entry produced by a lower effort level
than the one requested (``patient`` does not reuse a ``measure`` decision).

Backends without variants (``default_variant``/``variants`` missing, or
``compile`` without a ``variant`` parameter): explicit fallback to ``estimate``
(decision ``source="fallback"``, message logged once).
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import logging
import math
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ._env import getenv
from .backends.base import Sizes, Variant

log = logging.getLogger("kernox.tuning")

LEVELS = ("estimate", "wisdom_only", "measure", "patient")
MAX_CANDIDATES = {"measure": 4, "patient": 32}
BUDGET_SECONDS = {"measure": 2.0, "patient": 30.0}
REPEATS = 5
WARMUP = 1
MAX_SPREAD = 0.25
WISDOM_FORMAT = "kernox-wisdom-v3"  # class (M, N, nb, 1/ρ)
#: Rank of the measurement levels: an entry is only reused by an effort level
#: lower than or equal to the one that produced it (``wisdom_only``: all of them).
_EFFORT = {"wisdom_only": 0, "measure": 1, "patient": 2}


def check_level(level) -> str:
    if level not in LEVELS:
        from .errors import ValidationError

        raise ValidationError(f"invalid tuning {level!r} (expected one of {LEVELS})")
    return level


def _half_octave(x: float) -> int:
    return math.ceil(2 * math.log2(x)) if x > 1 else 0


def size_class(m: int, n: int, nb: int = 1, density: float = 1.0) -> tuple[int, int, int, int]:
    """Size class in half-octaves (wisdom v3), ``log2(x <= 1) := 0``:
    ``(ceil(2 log2 M), ceil(2 log2 N), ceil(2 log2 nb), ceil(2 log2 1/ρ))``.

    Two sizes in the same class differ by at most a factor sqrt(2) per
    axis (v1, in octaves, grouped sizes up to 2x apart and could reuse a
    worse decision). ``nb``: batch elements; ``density``: fraction ρ of the pairs (i, j) covered by the block-sparse
    ranges (1 without ranges; ρ = 0 counted as one pair). K, the
    dimension and the dtype are in the plan key.
    """
    rho = _half_octave(1.0 / max(float(density), 1.0 / max(m * n, 1))) if density < 1.0 else 0
    return (_half_octave(m), _half_octave(n), _half_octave(nb), rho)


def variant_from_json(data) -> Variant:
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        raise ValueError("invalid JSON variant")
    params = data.get("params", {})
    if not isinstance(params, dict) or not all(
        isinstance(k, str) and isinstance(v, (int, str)) and not isinstance(v, bool) for k, v in params.items()
    ):
        raise ValueError("invalid variant parameters")
    return Variant(data["name"], tuple(sorted(params.items())))


@dataclass
class TuningDecision:
    """Published decision (``Reduction.tuning_decisions``) for a device and a size class."""

    level: str
    source: str                      # "estimate" | "wisdom" | "measured" | "fallback"
    variant: dict | None
    size_class: tuple[int, int]
    stable: bool = True
    persisted: bool = False
    candidates: int = 0
    truncated_from: int | None = None
    measurements: list = field(default_factory=list)  # [{variant, median_ms, spread, samples_ms, compile_s}]
    tuning_seconds: float = 0.0
    budget_seconds: float | None = None
    reason: str = ""


# --------------------------------------------------------------------------
# backend capabilities
# --------------------------------------------------------------------------


def backend_has_variants(backend) -> bool:
    if not (callable(getattr(backend, "default_variant", None)) and callable(getattr(backend, "variants", None))):
        return False
    try:
        return "variant" in inspect.signature(backend.compile).parameters
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# wisdom
# --------------------------------------------------------------------------


def wisdom_dir() -> Path:
    root = getenv("KERNOX_CACHE_DIR")
    if root:
        return Path(root) / "wisdom"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "kernox" / "wisdom"


def _canonical(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identity_id(identity: dict) -> str:
    return hashlib.sha256(_canonical(identity).encode()).hexdigest()[:24]


def entry_key(plan_key: str, backend: str, identity: dict, sclass, fmt: str = WISDOM_FORMAT) -> str:
    return hashlib.sha256(_canonical([fmt, plan_key, backend, identity, list(sclass)]).encode()).hexdigest()


def _wisdom_file(identity: dict) -> Path:
    return wisdom_dir() / f"wisdom-{identity_id(identity)}.json"


@contextlib.contextmanager
def _locked(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read(path: Path, identity: dict | None = None) -> dict:
    """Valid file contents, or an empty file; any invalid content is ignored."""
    empty = {"format": WISDOM_FORMAT, "identity": identity, "entries": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return empty
    if not isinstance(data, dict) or data.get("format") != WISDOM_FORMAT or not isinstance(data.get("entries"), dict):
        return empty
    if identity is not None and data.get("identity") != identity:
        return empty
    return data


def _write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", delete=False) as f:
            tmp = Path(f.name)
            json.dump(data, f, sort_keys=True, indent=1)
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _valid_entry(key: str, entry, identity: dict) -> bool:
    try:
        variant_from_json(entry["variant"])
        return (
            isinstance(entry, dict)
            and entry.get("format") == WISDOM_FORMAT
            and entry.get("identity") == identity
            and entry_key(entry["plan_key"], entry["backend"], identity, entry["size_class"]) == key
        )
    except (KeyError, TypeError, ValueError):
        return False


def lookup(plan_key: str, backend: str, identity: dict, sclass, level: str = "wisdom_only") -> Variant | None:
    """Recorded variant, or None. An entry produced by a lower
    effort than ``level`` (``measure`` read by ``patient``) is ignored: the
    search is redone and the entry replaced."""
    key = entry_key(plan_key, backend, identity, sclass)
    path = _wisdom_file(identity)
    if not path.exists():
        return None
    with _locked(path):
        entry = _read(path, identity)["entries"].get(key)
    if entry is None or not _valid_entry(key, entry, identity):
        return None
    if _EFFORT.get(entry.get("level"), 0) < _EFFORT.get(level, 0):
        return None
    return variant_from_json(entry["variant"])


def store(plan_key: str, backend: str, identity: dict, sclass, variant: Variant, extra: dict) -> None:
    key = entry_key(plan_key, backend, identity, sclass)
    path = _wisdom_file(identity)
    with _locked(path):
        data = _read(path, identity)
        data["identity"] = identity
        data["entries"][key] = {
            "format": WISDOM_FORMAT,
            "identity": identity,
            "plan_key": plan_key,
            "backend": backend,
            "size_class": list(sclass),
            "variant": variant.as_json(),
            **extra,
        }
        _write_atomic(path, data)


def export_wisdom(path=None) -> dict:
    """All valid wisdom ``{format, devices: {id: {identity, entries}}}``; written to ``path`` if given."""
    out = {"format": WISDOM_FORMAT, "devices": {}}
    folder = wisdom_dir()
    for file in sorted(folder.glob("wisdom-*.json")) if folder.exists() else ():
        with _locked(file):
            data = _read(file)
        identity = data.get("identity")
        if not isinstance(identity, dict):
            continue
        entries = {k: e for k, e in data["entries"].items() if _valid_entry(k, e, identity)}
        if entries:
            out["devices"][identity_id(identity)] = {"identity": identity, "entries": entries}
    if path is not None:
        _write_atomic(Path(path), out)
    return out


def import_wisdom(source) -> int:
    """Merges an exported wisdom (dict or path). Returns the number of valid entries imported."""
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    if not isinstance(data, dict) or data.get("format") != WISDOM_FORMAT:
        raise ValueError(f"wisdom has an unknown format (expected {WISDOM_FORMAT})")
    count = 0
    for block in (data.get("devices") or {}).values():
        identity = block.get("identity") if isinstance(block, dict) else None
        if not isinstance(identity, dict):
            continue
        entries = block.get("entries") or {}
        valid = {k: e for k, e in entries.items() if _valid_entry(k, e, identity)}
        if not valid:
            continue
        path = _wisdom_file(identity)
        with _locked(path):
            current = _read(path, identity)
            current["identity"] = identity
            current["entries"].update(valid)
            _write_atomic(path, current)
        count += len(valid)
    return count


def forget_wisdom() -> int:
    """Deletes all wisdom. Returns the number of files deleted."""
    folder = wisdom_dir()
    removed = 0
    for file in sorted(folder.glob("wisdom-*.json")) if folder.exists() else ():
        with _locked(file):
            file.unlink(missing_ok=True)
        removed += 1
    return removed


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


class _Timer:
    """Timing with GPU events on the current stream; host clock on CPU."""

    def __init__(self, device):
        import torch

        self.device = device
        mod = {"cuda": getattr(torch, "cuda", None), "xpu": getattr(torch, "xpu", None)}.get(device.type)
        self.mod = mod if mod is not None and hasattr(mod, "Event") else None

    def run(self, fn) -> float:
        """Duration of ``fn()`` in milliseconds."""
        if self.mod is None:
            t0 = time.perf_counter()
            fn()
            return (time.perf_counter() - t0) * 1e3
        with self.mod.device(self.device):
            start, end = self.mod.Event(enable_timing=True), self.mod.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end))

    def synchronize(self):
        if self.mod is not None:
            self.mod.synchronize(self.device)


def candidates(backend, plan, device, sizes: Sizes, level: str, default: Variant):
    """Deduplicated candidates, default first, truncated to ``MAX_CANDIDATES[level]``."""
    raw = list(backend.variants(plan, device, sizes, level))
    seen, out = set(), []
    for v in [default] + raw:
        if not isinstance(v, Variant):
            raise TypeError(f"backend {backend.name!r} returned an invalid variant: {v!r}")
        if v not in seen:
            seen.add(v)
            out.append(v)
    limit = MAX_CANDIDATES[level]
    truncated_from = None
    if len(out) > limit:
        truncated_from = len(out)
        log.warning(
            "backend %r: %d candidates for tuning=%r, truncated to %d (contract v0.2)",
            backend.name, len(out), level, limit,
        )
        out = out[:limit]
    return out, truncated_from


def measure(level, compile_variant, candidates_list, inputs, output_shapes_fn, device, budget=None,
            launch_kwargs=None):
    """Measures the candidates; returns ``(winner, measurements, stable, seconds)``.

    ``compile_variant(v)`` -> compiled object; ``output_shapes_fn()`` -> fresh
    temporary outputs. The inputs are only read. ``launch_kwargs(compiled)``
    -> keyword arguments of ``launch`` (``batch=``, ``ranges=``; wisdom v3).
    """
    budget = BUDGET_SECONDS[level] if budget is None else budget
    timer = _Timer(device)
    t0 = time.perf_counter()
    results = []
    for index, variant in enumerate(candidates_list):
        if index > 0 and time.perf_counter() - t0 > budget:
            log.info("tuning=%r: budget of %.1fs exhausted after %d candidates", level, budget, index)
            break
        c0 = time.perf_counter()
        compiled = compile_variant(variant)
        compile_s = time.perf_counter() - c0
        outs = output_shapes_fn()
        kw = launch_kwargs(compiled) if launch_kwargs is not None else {}
        for _ in range(WARMUP):
            compiled.launch(inputs, outs, **kw)
        samples = [timer.run(lambda: compiled.launch(inputs, outs, **kw)) for _ in range(REPEATS)]
        median = statistics.median(samples)
        spread = (max(samples) - min(samples)) / median if median > 0 else 0.0
        results.append({"variant": variant.as_json(), "median_ms": median, "spread": spread,
                        "samples_ms": samples, "compile_s": compile_s, "_variant": variant})
    timer.synchronize()
    ranked = sorted(results, key=lambda r: r["median_ms"])
    stable = all(r["spread"] <= MAX_SPREAD for r in ranked[:2])
    winner = ranked[0]["_variant"]
    for r in results:
        del r["_variant"]
    return winner, results, stable, time.perf_counter() - t0


def decision_dict(d: TuningDecision) -> dict:
    return asdict(d)
