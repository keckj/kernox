"""Avenant v0.10 côté CUDA : source générée (sans GPU) et refus FP16."""

import dataclasses

import kernox as kp
from kernox.backends.cuda import _kernel
from kernox.codegen.pair import pair_function


def _lse(fast, dt="float32"):
    x, y, h = kp.Vi("x", 2), kp.Vj("y", 2), kp.Vj("h", 1)
    return (h - kp.exp(((x - y) ** 2).sum()) + kp.log(h * h + 1)).logsumexp(over="j").to_plan(dt, fast_math=fast)


def _src(plan):
    pair = pair_function(dataclasses.replace(plan, fast_math=False), "cuda")
    src = pair.source
    if plan.fast_math and plan.dtype == "float32":
        src = _kernel.fast_pair_source(src)
    return _kernel.generate(plan, src, pair.name)


def test_fast_source_uses_approx_everywhere():
    s = _src(_lse(True))
    body = s.split("kp_flog(float x)")[1]  # après les définitions du prélude
    assert "kp_fexp(" in body and "kp_flog(" in body
    assert "expf(" not in body.replace("kp_fexp(", "") and "logf(" not in body.replace("kp_flog(", "")


def test_ieee_source_untouched():
    s = _src(_lse(False))
    body = s.split("kp_flog(float x)")[1]
    assert "kp_fexp(" not in body and "kp_flog(" not in body
    assert "expf(" in body and "logf(" in body


def test_fp64_fast_is_ieee():
    a, b = _src(_lse(True, "float64")), _src(_lse(False, "float64"))
    assert a == b  # pas d'instruction approchée en FP64 : source IEEE identique
