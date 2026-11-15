"""Diagnostics de ressources des kernels hip (registres, spills, LDS, scratch).

Deux sources complémentaires :

1. ``HipCompiledReduction.resources()`` : attributs du code object **réellement
   chargé** (hipRTC) via ``hipFuncGetAttribute`` (VGPR, scratch par thread, LDS).
2. ``resource_remarks(source, flags)`` : remarques
   ``-Rpass-analysis=kernel-resource-usage`` de l'``amdclang++`` embarqué dans
   le même paquet ROCm que Torch (même LLVM que hipRTC), sur la même source et
   les mêmes options : SGPR/VGPR, spills, scratch, occupation.

Usage : ``python -m kernox.backends.hip.diagnostics`` (compilation seule,
aucun kernel lancé : utilisable même quand le GPU est occupé).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile


def _sdk_root() -> str:
    from . import _hip

    return os.path.dirname(os.path.dirname(os.path.realpath(_hip.hip_runtime().path)))


def resource_remarks(source: str, flags: list[str], kernel: str = "kp_reduce") -> dict:
    root = _sdk_root()
    clang = os.path.join(root, "lib", "llvm", "bin", "amdclang++")
    devlib = os.path.join(root, "lib", "llvm", "amdgcn", "bitcode")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "k.hip")
        with open(src, "w") as f:
            f.write(source)
        # hipRTC inclut hip_runtime.h d'office ; on le reproduit pour amdclang++.
        cmd = [clang, "-x", "hip", f"--rocm-path={root}", f"--rocm-device-lib-path={devlib}",
               "-I", os.path.join(root, "include"), "-include", "hip/hip_runtime.h",
               "--offload-device-only", *flags, "-Rpass-analysis=kernel-resource-usage",
               "-c", src, "-o", os.path.join(d, "k.o")]
        p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-4000:])
    per_fn: dict = {}
    cur = None
    for line in p.stderr.splitlines():
        m = re.search(r"remark:\s+(.+?):\s+(\S+) \[-Rpass-analysis", line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2)
        if key == "Function Name":
            cur = per_fn.setdefault(val, {})
            continue
        if cur is not None:
            cur[key] = int(val) if val.lstrip("-").isdigit() else val
    return per_fn.get(kernel, {}) if kernel else per_fn


def kmin_table(ks=(1, 4, 8, 16, 32, 64, 128, 256), dtypes=("float32", "float64"), dim=3,
               strategies=("thread", "coop")):
    import torch

    import kernox as kp

    from .backend import HipBackend, HipOptions

    rows = []
    x, y = kp.Vi("x", dim), kp.Vj("y", dim)
    d2 = ((x - y) ** 2).sum()
    for strat in strategies:
        be = HipBackend(HipOptions(strategy=strat, split="off"))
        for dt in dtypes:
            for k in ks:
                if strat == "coop" and k < 16:
                    continue
                plan = d2.kmin_argkmin(k, over="j").to_plan(dt)
                cr = be.compile(plan, torch.device("cuda"))
                rem = resource_remarks(cr.source, list(cr.info.options), "kp_reduce")
                rem_m = resource_remarks(cr.source, list(cr.info.options), "kp_merge")
                rows.append({"strategy": strat, "group": cr.config.group, "dtype": dt, "K": k,
                             **cr.resources(), "compile_s": round(cr.info.compile_seconds, 2),
                             **{k2: rem.get(k2) for k2 in ("VGPRs", "TotalSGPRs", "VGPRs Spill", "SGPRs Spill",
                                                            "ScratchSize [bytes/lane]", "Occupancy [waves/SIMD]",
                                                            "LDS Size [bytes/block]")},
                             "merge_VGPRs": rem_m.get("VGPRs"), "merge_scratch": rem_m.get("ScratchSize [bytes/lane]")})
    return rows


if __name__ == "__main__":
    import json
    import sys

    strategies = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("thread", "coop")
    for r in kmin_table(strategies=strategies):
        print(json.dumps(r))
        sys.stdout.flush()
