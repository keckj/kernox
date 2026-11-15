"""Test harness: wraps ``kp_pair`` in a "one pair per thread" kernel.

Only used to check that the pair source compiles (NVRTC, hipcc, icpx)
and, on NVIDIA GPUs, that it computes the right values. This is not a
reduction kernel.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from kernox.codegen.pair import pair_function


def _call(ps, plan):
    parts = []
    for lay in ps.arg_layout:
        if lay.role == "i":
            parts.append(f"{lay.param} + _kp_i * {lay.dim}")
        elif lay.role == "j":
            parts.append(f"{lay.param} + _kp_j * {lay.dim}")
        else:
            parts.append(lay.param)
    return f"{ps.name}({', '.join(parts)})"


def eval_source(plan, dialect):
    """Complete compilation unit: pair + ``kp_eval`` kernel (output (M, N, D) of type T)."""
    ps = pair_function(plan, dialect)
    T = ps.scalar_type
    params = ", ".join(f"const {T}* __restrict__ {l.param}" for l in ps.arg_layout)
    stores = "\n".join(
        f"    out[_kp_idx * {plan.out_dim} + {c}] = ({T})_kp_r.{f};" for c, f in enumerate(ps.out_fields)
    )
    body = f"""    const long long _kp_i = _kp_idx / N, _kp_j = _kp_idx % N;
    const {ps.out_type} _kp_r = {_call(ps, plan)};
{stores}"""
    if dialect in ("cuda", "hip"):
        head = "#include <hip/hip_runtime.h>\n" if dialect == "hip" else ""
        return ps, head + ps.source + f"""
extern "C" __global__ void kp_eval(long long M, long long N, {params}, {T}* __restrict__ out) {{
    const long long _kp_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (_kp_idx >= M * N) return;
{body}
}}
"""
    return ps, "#include <sycl/sycl.hpp>\n" + ps.source + f"""
sycl::event kp_eval(sycl::queue& q, long long M, long long N, {params}, {T}* __restrict__ out) {{
    return q.parallel_for(sycl::range<1>((size_t)(M * N)), [=](sycl::id<1> _kp_id) {{
        const long long _kp_idx = (long long)_kp_id[0];
{body}
    }});
}}
"""


# ---------------------------------------------------------------- compilers
def nvrtc_default_arch(preferred=52):
    """Test architecture: ``preferred`` if the loaded NVRTC can still compile
    for it, otherwise the oldest one it supports (CUDA 13 dropped sm_52,
    hence "invalid value for --gpu-architecture")."""
    import ctypes as ct

    from symox.extra.codegen.cuda import _library

    nvrtc = _library("nvrtc")
    n = ct.c_int()
    try:
        nvrtc.nvrtcGetNumSupportedArchs.argtypes = (ct.POINTER(ct.c_int),)
        nvrtc.nvrtcGetSupportedArchs.argtypes = (ct.POINTER(ct.c_int),)
        if nvrtc.nvrtcGetNumSupportedArchs(ct.byref(n)) or n.value <= 0:
            return str(preferred)
        archs = (ct.c_int * n.value)()
        if nvrtc.nvrtcGetSupportedArchs(archs):
            return str(preferred)
    except AttributeError:  # NVRTC < 11.2: no introspection, sm_52 still supported
        return str(preferred)
    supported = sorted(archs)
    return str(preferred if preferred in supported else supported[0])


def nvrtc_compile(source, arch=None):
    """PTX via NVRTC (symox.extra.codegen.cuda._compile), uncached."""
    from symox.extra.codegen.cuda import _compile

    ptx, _, _ = _compile(source, (), arch or nvrtc_default_arch(), False)
    return ptx


def find_hipcc():
    for c in (shutil.which("hipcc"), "/opt/rocm/bin/hipcc"):
        if c and Path(c).exists():
            return c
    return None


def hipcc_compile(source, tmp_path, arch="gfx1201"):
    src = Path(tmp_path) / "pair.hip"
    src.write_text(source)
    cmd = [find_hipcc(), f"--offload-arch={arch}", "-O2", "-std=c++17", "-c", str(src), "-o", str(src.with_suffix(".o"))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-4000:]


def find_icpx():
    for c in (shutil.which("icpx"), "/opt/intel/oneapi/compiler/latest/bin/icpx"):
        if c and Path(c).exists():
            return c
    return None


def icpx_check(source, tmp_path, mode="-fsyntax-only"):
    src = Path(tmp_path) / "pair.cpp"
    src.write_text(source)
    cmd = [find_icpx(), "-fsycl", "-std=c++17", mode, str(src)]
    if mode == "-c":
        cmd += ["-o", str(src.with_suffix(".o"))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-4000:]


# ---------------------------------------------------------------- CUDA execution
def cuda_eval(plan, inputs, device=0):
    """Evaluates all pairs on an NVIDIA GPU through the CUDA driver (symox), returns (M, N, D)."""
    import ctypes as ct

    import numpy as np
    from symox.extra.codegen.cuda import _cleanup, compile_cuda

    _, source = eval_source(plan, "cuda")
    np_dt = np.float64 if plan.dtype == "float64" else np.float32
    arrays = [np.ascontiguousarray(np.asarray(inputs[a.name], dtype=np_dt)) for a in plan.args]
    m = next(np.asarray(inputs[a.name]).shape[0] for a in plan.args if a.role == "i")
    n = next(np.asarray(inputs[a.name]).shape[0] for a in plan.args if a.role == "j")
    out = np.zeros((m, n, plan.out_dim), dtype=np_dt)
    kernel = compile_cuda(source, "kp_eval", device=device, cache_dir=False)
    drv, allocs = kernel._driver, []
    try:
        with kernel._current():
            try:
                for arr in arrays + [out]:
                    ptr = ct.c_uint64()
                    drv.call("cuMemAlloc_v2", ct.byref(ptr), max(arr.nbytes, 8))
                    allocs.append(ptr)
                for arr, ptr in zip(arrays, allocs):
                    drv.call("cuMemcpyHtoD_v2", ptr, ct.c_void_p(arr.ctypes.data), arr.nbytes)
                total = m * n
                kernel.launch([ct.c_longlong(m), ct.c_longlong(n), *allocs],
                              grid=max(1, (total + 127) // 128), block=128)
                drv.call("cuCtxSynchronize")
                drv.call("cuMemcpyDtoH_v2", ct.c_void_p(out.ctypes.data), allocs[-1], out.nbytes)
            finally:
                _cleanup([(drv.call, ("cuMemFree_v2", p)) for p in reversed(allocs)])
    finally:
        kernel.close()
    return out
