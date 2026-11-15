"""Formula fuzzing against the FP64 reference (see ``_fuzz.py``).

Quick profile: seeds ``0..FUZZ_QUICK-1``; long profile (``slow``):
``FUZZ_SLOW`` seeds; ``KP_FUZZ_SEEDS=a:b`` forces the range. A failure is
minimized (tree and sizes) then recorded in ``regressions/``; recorded
regressions are replayed on **all** available backends.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from . import _fuzz
from ._support import run_case

FUZZ_QUICK = 40
FUZZ_SLOW = 1000
REG_DIR = Path(__file__).parent / "regressions"


def _seeds(slow):
    spec = os.environ.get("KP_FUZZ_SEEDS")
    if spec:
        a, b = (int(v) for v in spec.split(":"))
        return range(a, b)
    return range(FUZZ_QUICK, FUZZ_QUICK + FUZZ_SLOW) if slow else range(FUZZ_QUICK)


def _run(backend, case):
    name, dev = backend
    expr, _ = _fuzz.build(case)
    return run_case(name, dev, expr, _fuzz.arrays(case), case["dtype"])


def _record(backend, case, exc):
    name, _ = backend

    signature = _signature(exc)

    def fails(c):
        # same failure class only (exception type and violated criterion)
        try:
            _run(backend, c)
        except Exception as e:  # noqa: BLE001
            return _signature(e) == signature
        return False

    small = _fuzz.minimize(case, fails)
    try:
        _run(backend, small)
        small_exc = exc
    except Exception as e:  # noqa: BLE001
        small_exc = e
    expr_texts = [_fuzz.source(t) for t in small["trees"]]
    rec = {k: v for k, v in small.items() if k != "trees"}
    rec.update(sources=expr_texts, backend=name, error=f"{type(small_exc).__name__}: {small_exc}"[:2000],
               original_seed=case["seed"])
    REG_DIR.mkdir(exist_ok=True)
    path = REG_DIR / f"{name}-seed{case['seed']}-{case['kind']}-{case['dtype']}.json"
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return path, rec


def _signature(exc):
    text = str(exc)
    label = text.split(" (", 1)[0] if type(exc).__name__ == "Mismatch" else ""
    return type(exc).__name__, label


def _fuzz_one(backend, seed):
    case = _fuzz.Generator(seed).case()
    try:
        _run(backend, case)
    except Exception as exc:  # noqa: BLE001
        path, rec = _record(backend, case, exc)
        pytest.fail(f"seed {seed}: {type(exc).__name__}: {exc}\n"
                    f"minimal case recorded in {path}:\n{json.dumps(rec, indent=1, ensure_ascii=False)}")


@pytest.mark.parametrize("seed", _seeds(False))
def test_fuzz_quick(backend, seed):
    _fuzz_one(backend, seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", _seeds(True))
def test_fuzz_slow(backend, seed):
    _fuzz_one(backend, seed)


def _regressions():
    return sorted(REG_DIR.glob("*.json")) if REG_DIR.exists() else []


@pytest.mark.parametrize("path", _regressions(), ids=lambda p: p.stem)
def test_regression(backend, path):
    case = json.loads(path.read_text())
    _run(backend, case)


def test_generator_is_reproducible():
    a, b = _fuzz.Generator(123).case(), _fuzz.Generator(123).case()
    assert [_fuzz.source(t) for t in a["trees"]] == [_fuzz.source(t) for t in b["trees"]]
    for seed in range(200):  # every generated formula builds
        _fuzz.build(_fuzz.Generator(seed).case())
