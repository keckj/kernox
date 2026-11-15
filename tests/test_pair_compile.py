"""La source de paire compile pour chaque dialecte (et s'exécute sur NVIDIA si disponible).

Sauté proprement si l'outil n'est pas présent sur l'hôte :
NVRTC (wheel nvidia-cuda-nvrtc-cu12 ou CUDA toolkit), hipcc (ROCm), icpx (oneAPI).
"""
import os

import numpy as np
import pytest

from kernox.oracle import evaluate_numpy

from corpus import reductions
import pair_harness as H

CASES = [(n, d) for n in reductions() for d in ("float32", "float64")]


def _nvrtc_available():
    try:
        from symox.extra.codegen.cuda import _library
        _library("nvrtc")
        return True
    except Exception:
        return False


def _cuda_device_available():
    try:
        from symox.extra.codegen.cuda import _driver
        _driver()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _nvrtc_available(), reason="NVRTC absent")
@pytest.mark.parametrize("name,dtype", CASES)
def test_nvrtc(name, dtype):
    _, src = H.eval_source(reductions()[name].to_plan(dtype), "cuda")
    assert b".entry kp_eval" in H.nvrtc_compile(src, os.environ.get("KP_NVRTC_ARCH"))


@pytest.mark.skipif(H.find_hipcc() is None, reason="hipcc absent")
@pytest.mark.parametrize("name,dtype", CASES)
def test_hipcc(name, dtype, tmp_path):
    _, src = H.eval_source(reductions()[name].to_plan(dtype), "hip")
    H.hipcc_compile(src, tmp_path, os.environ.get("KP_HIP_ARCH", "gfx1201"))


@pytest.mark.skipif(H.find_icpx() is None, reason="icpx absent")
@pytest.mark.parametrize("name,dtype", CASES)
def test_icpx(name, dtype, tmp_path):
    _, src = H.eval_source(reductions()[name].to_plan(dtype), "sycl")
    H.icpx_check(src, tmp_path, os.environ.get("KP_ICPX_MODE", "-fsyntax-only"))


@pytest.mark.gpu
@pytest.mark.skipif(not (_nvrtc_available() and _cuda_device_available()), reason="GPU NVIDIA absent")
@pytest.mark.parametrize("name,dtype", CASES)
def test_cuda_pair_values(name, dtype):
    """Valeurs de la fonction de paire sur GPU NVIDIA vs oracle NumPy FP64."""
    from conftest import make_inputs

    plan = reductions()[name].to_plan(dtype)
    inputs = {k: v.numpy() for k, v in make_inputs(plan, 7, 11, seed=3).items()}
    got = H.cuda_eval(plan, inputs)
    ref = evaluate_numpy(plan, inputs)
    if plan.reduction.kind.value == "count":
        np.testing.assert_array_equal(got.astype(bool), ref)
    else:
        tol = 2e-5 if dtype == "float32" else 1e-13
        np.testing.assert_allclose(got, ref, rtol=tol, atol=tol)
