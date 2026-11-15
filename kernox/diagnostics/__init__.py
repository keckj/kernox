"""Kernox installation diagnostics: ``kernox.diagnostics()`` and ``python -m kernox.diagnostics``.

Reports, without installing or modifying anything:

- versions: Python, Torch (CUDA/ROCm/XPU runtime), NumPy, symox (revision
  identified by the fingerprint of its native library, warning if it
  is not ``PINNED_SYMOX``) and detected symox capabilities;
- native backends (cuda, hip, sycl): available, or exact reason
  for unavailability (module, device, rejected dtype); device, arch, actual FP64;
- caches: compilation and wisdom directories (number of valid entries);
- option ``check=True``: compilation of a small plan (exact compiler and driver
  via ``CompileInfo``) and a mini correctness check against
  the FP64 oracle (Gaussian SUM, ARGKMIN, SUM gradient), per backend and dtype.

``check=True`` launches small kernels: do not use it on a card where
a benchmark is running.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
import sys
from pathlib import Path

# Empreintes (sha256 RECORD de symox/_core.abi3.so) des wheels de référence
# construits par le projet ; les builds ne sont pas reproductibles
# d'une machine à l'autre : un autre build est « révision non identifiée ».
_SYMOX_CORE_FINGERPRINTS = {
    "3LfAlFu09abIJM7NAM6Gvlf9Wc-j4FiuVDsNbHPEs3A": "53d9576e9f39db16e9ca8b3483d754385ebaf986",
    "i1Jmdslg76QrkR3k1NH9ZVCbIWCqLtKMRAWBQv4F2kM": "b8d7adda11b40e027ac6880f5f82bdc39f6a1082",
    "PVH6Zvhv7vXgoKMJzREfi75XmPE1O88A-_fWT1igNJw": "b02db0cad3f3fe9c9135772bbe44b940ebd97b86",
    "Su1RIcj-ZqG2MbkBIA31k-6QA2OiCRda6xTPOVueilo": "45179a226f7bc7a6e9366be4ec8389ec15f774ad",
    "keR8KNEilzeEMGdCOtFeVIQgMUxRtcvjdFbs-ugI3hY": "482434bf",
    "udhqSC_5649pK63V-SiA-LiMwcU6SNG6H4zb6rVjVTQ": "2e623e66bb31882e31f6dbfff6f77ae4bd945408",
}


def _symox_info() -> dict:
    from .. import _symox_compat as compat

    out = {"pinned": compat.PINNED_SYMOX, "min_supported": compat.MIN_SUPPORTED_SYMOX, "warnings": []}
    try:
        import symox
    except ImportError as exc:
        out.update(available=False, error=str(exc))
        out["warnings"].append("symox absent : voir l'installation de symox dans le README")
        return out
    out.update(available=True, version=getattr(symox, "__version__", "?"), path=os.path.dirname(symox.__file__))
    fingerprint, source = None, None
    try:
        dist = md.distribution("symox")
        record = dist.read_text("RECORD") or ""
        for line in record.splitlines():
            if line.startswith("symox/_core"):
                fingerprint = line.split(",")[1].removeprefix("sha256=")
        direct = dist.read_text("direct_url.json")
        source = json.loads(direct).get("url") if direct else None
    except md.PackageNotFoundError:
        pass
    revision = _SYMOX_CORE_FINGERPRINTS.get(fingerprint)
    out.update(core_fingerprint=fingerprint, installed_from=source, revision=revision)
    if revision is None:
        out["warnings"].append(
            f"révision symox non identifiée (empreinte {fingerprint}) : Kernox est épinglé sur "
            f"{compat.PINNED_SYMOX[:8]} (installation : voir le README)")
    elif revision != compat.PINNED_SYMOX:
        out["warnings"].append(
            f"symox {revision[:8]} n'est pas la révision épinglée {compat.PINNED_SYMOX[:8]} "
            "(réinstaller symox au commit épinglé : voir le README)")
    caps = {}
    for name in ("cuda_logic", "cudacode_function", "lambdify_broadcasts"):
        fn = getattr(compat, name, None)
        if callable(fn):
            try:
                caps[name] = bool(fn())
            except Exception as exc:  # noqa: BLE001 - sonde
                caps[name] = f"erreur : {type(exc).__name__}: {exc}"
    out["capabilities"] = caps
    return out


def _torch_info() -> dict:
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "error": str(exc)}
    info = {"available": True, "version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "hip_runtime": getattr(torch.version, "hip", None),
            "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0}
    xpu = getattr(torch, "xpu", None)
    info["xpu_devices"] = xpu.device_count() if xpu is not None and xpu.is_available() else 0
    return info


def _device_for(name: str):
    import torch

    if name in ("cuda", "hip"):
        if not torch.cuda.is_available():
            return None, "aucun device Torch CUDA/ROCm visible"
        is_hip = getattr(torch.version, "hip", None) is not None
        if (name == "hip") != is_hip:
            return None, f"Torch construit pour {'ROCm' if is_hip else 'CUDA'}, pas pour {name}"
        return torch.device("cuda:0"), ""
    if name == "sycl":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            return None, "aucun device Torch XPU visible"
        return torch.device("xpu:0"), ""
    raise ValueError(name)


def _device_props(device) -> dict:
    import torch

    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        arch = p.gcnArchName if getattr(torch.version, "hip", None) else f"sm_{p.major}{p.minor}"
        return {"name": p.name, "arch": arch, "memory_bytes": int(p.total_memory),
                "multiprocessors": int(p.multi_processor_count)}
    p = torch.xpu.get_device_properties(device)
    return {"name": p.name, "driver": str(getattr(p, "driver_version", "")),
            "memory_bytes": int(getattr(p, "total_memory", 0)),
            "has_fp64": bool(getattr(p, "has_fp64", False))}


def _probe_plan(dtype):
    from ..expr import Vi, Vj, exp

    x, y = Vi("x", 3), Vj("y", 3)
    return (exp(-((x - y) ** 2).sum()) * Vj("w", 2)).sum_over("j").to_plan(dtype)


def _backend_info(name: str, check: bool) -> dict:
    from ..errors import BackendUnavailableError
    from ..runtime import get_backend

    out = {"name": name}
    try:
        backend = get_backend(name)
    except BackendUnavailableError as exc:
        out.update(available=False, reason=str(exc))
        return out
    device, why = _device_for(name)
    if device is None:
        out.update(available=False, reason=why)
        return out
    out.update(available=True, device=str(device), properties=_device_props(device),
               fp_contract_off=bool(getattr(backend, "fp_contract_off", False)))
    dtypes = {}
    for dt in ("float32", "float64"):
        s = backend.supports(_probe_plan(dt), device)
        dtypes[dt] = {"supported": s.ok, "reason": s.reason}
    out["dtypes"] = dtypes
    out["fp64"] = dtypes["float64"]["supported"]
    if check:
        out["check"] = _check_backend(name, device, [dt for dt, v in dtypes.items() if v["supported"]])
    return out


def _check_backend(name, device, dtypes) -> dict:
    """Petites réductions comparées à la référence FP64 de l'oracle."""
    import torch

    from .. import compile as kp_compile
    from ..expr import Vi, Vj, exp

    x, y, w = Vi("x", 3), Vj("y", 3), Vj("w", 2)
    d2 = ((x - y) ** 2).sum()
    cases = {"sum_gauss": (exp(-d2) * w).sum_over("j"), "argkmin": d2.argkmin(4, over="j")}
    g = torch.Generator().manual_seed(0)
    host = {"x": torch.randn(57, 3, generator=g, dtype=torch.float64),
            "y": torch.randn(83, 3, generator=g, dtype=torch.float64),
            "w": torch.randn(83, 2, generator=g, dtype=torch.float64)}
    results = {}
    for dt in dtypes:
        tdt = getattr(torch, dt)
        tol = 1e-5 if dt == "float32" else 1e-12
        for cname, red in cases.items():
            key = f"{cname}[{dt}]"
            try:
                run = kp_compile(red, dtype=dt, backend=name)
                inp = {k: v.to(tdt) for k, v in host.items() if k in run.input_names}
                ref = kp_compile(red, dtype="float64", backend="oracle")(
                    **{k: v.to(torch.float64) for k, v in inp.items()})
                got = run(**{k: v.to(device) for k, v in inp.items()})
                if hasattr(torch, device.type) and hasattr(getattr(torch, device.type), "synchronize"):
                    getattr(torch, device.type).synchronize()
                got = got.cpu()
                if cname == "argkmin":
                    ok = bool(torch.equal(got, ref))
                    err = int((got != ref).sum())
                else:
                    err = float(((got.double() - ref).abs() / ref.abs().max().clamp_min(1e-300)).max())
                    ok = err <= tol
                info = run.compiled(device, 57, 83).info
                results[key] = {"ok": ok, "error": err, "compiler": info.compiler, "arch": info.arch}
            except Exception as exc:  # noqa: BLE001 - diagnostic
                results[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        key = f"grad_sum[{dt}]"
        try:
            run = kp_compile(cases["sum_gauss"], dtype=dt, backend=name)
            X = host["x"].to(tdt).to(device).requires_grad_(True)
            out = run(x=X, y=host["y"].to(tdt).to(device), w=host["w"].to(tdt).to(device))
            (gx,) = torch.autograd.grad(out.sum(), (X,))
            Xr = host["x"].clone().requires_grad_(True)
            dense = torch.exp(-((Xr[:, None] - host["y"][None]) ** 2).sum(-1)) @ host["w"]
            (want,) = torch.autograd.grad(dense.sum(), (Xr,))
            err = float(((gx.cpu().double() - want).abs() / want.abs().max()).max())
            results[key] = {"ok": err <= (1e-4 if dt == "float32" else 1e-11), "error": err}
        except Exception as exc:  # noqa: BLE001
            results[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return results


def _cache_info() -> dict:
    from .. import tuning

    out = {name: os.environ.get(name)
           for name in ("KERNOX_CACHE_DIR", "KERNOX_CUDA_CACHE_DIR")}
    wdir = tuning.wisdom_dir()
    out["wisdom_dir"] = str(wdir)
    try:
        data = tuning.export_wisdom()
        out["wisdom_devices"] = len(data["devices"])
        out["wisdom_entries"] = sum(len(d["entries"]) for d in data["devices"].values())
        out["wisdom_format"] = data["format"]
    except Exception as exc:  # noqa: BLE001
        out["wisdom_error"] = f"{type(exc).__name__}: {exc}"
    dirs = {}
    for name, mod, fn in (("cuda", "kernox.backends.cuda.backend", "_cache_dir"),
                          ("hip", "kernox.backends.hip._cache", "cache_dir"),
                          ("sycl", "kernox.backends.sycl.toolchain", "cache_dir")):
        try:
            d = Path(getattr(importlib.import_module(mod), fn)())
        except Exception as exc:  # noqa: BLE001 - backend absent
            dirs[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        files = [p for p in d.rglob("*") if p.is_file()] if d.exists() else []
        dirs[name] = {"path": str(d), "files": len(files), "bytes": sum(p.stat().st_size for p in files)}
    out["compile_caches"] = dirs
    return out


def diagnostics(*, check: bool = False, backends=None) -> dict:
    """Diagnostic report (JSON-serializable dict). ``check=True`` launches small kernels."""
    from ..runtime import NATIVE_BACKENDS

    names = tuple(backends) if backends else NATIVE_BACKENDS
    report = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "kernox": {"version": _version(), "path": os.path.dirname(os.path.dirname(__file__))},
        "numpy": _dist_version("numpy"),
        "torch": _torch_info(),
        "symox": _symox_info(),
    }
    report["backends"] = {n: _backend_info(n, check) if report["torch"]["available"]
                          else {"name": n, "available": False, "reason": "Torch absent"} for n in names}
    report["cache"] = _cache_info()
    warnings = list(report["symox"].get("warnings", []))
    for n, b in report["backends"].items():
        for k, r in (b.get("check") or {}).items():
            if not r["ok"]:
                warnings.append(f"vérification {n} {k} en échec : {r['error']}")
    report["warnings"] = warnings
    return report


def _version():
    try:
        return md.version("kernox")
    except md.PackageNotFoundError:
        return "non installé (arbre source)"


def _dist_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def format_report(r: dict) -> str:
    lines = ["Kernox — diagnostic", ""]
    lines.append(f"Python {r['python']} ({r['platform']})")
    lines.append(f"kernox {r['kernox']['version']} : {r['kernox']['path']}")
    t = r["torch"]
    if t["available"]:
        lines.append(f"Torch {t['version']} (CUDA {t['cuda_runtime']}, ROCm {t['hip_runtime']}) ; "
                     f"devices cuda/rocm {t['cuda_devices']}, xpu {t['xpu_devices']}")
    else:
        lines.append(f"Torch absent : {t['error']}")
    s = r["symox"]
    if s["available"]:
        rev = s["revision"][:8] if s["revision"] else "non identifiée"
        lines.append(f"symox {s['version']} révision {rev} (épinglée {s['pinned'][:8]}) : {s['path']}")
        lines.append("  capacités : " + ", ".join(f"{k}={v}" for k, v in s.get("capabilities", {}).items()))
    lines.append("")
    for name, b in r["backends"].items():
        if not b["available"]:
            lines.append(f"[{name}] indisponible : {b['reason']}")
            continue
        p = b["properties"]
        lines.append(f"[{name}] {b['device']} : {p.get('name')} {p.get('arch', '')} ; "
                     f"FP64 {'accepté' if b['fp64'] else 'refusé (' + b['dtypes']['float64']['reason'] + ')'} ; "
                     f"fp_contract=False {'pris en charge' if b['fp_contract_off'] else 'refusé'}")
        for k, c in (b.get("check") or {}).items():
            extra = f" ; {c['compiler']} {c['arch']}" if "compiler" in c else ""
            lines.append(f"    {'OK ' if c['ok'] else 'ÉCHEC'} {k} (écart {c['error']}){extra}")
    c = r["cache"]
    lines.append("")
    lines.append(f"wisdom : {c['wisdom_dir']} ({c.get('wisdom_entries', '?')} entrées, "
                 f"{c.get('wisdom_devices', '?')} devices)")
    for name, d in c["compile_caches"].items():
        if "path" in d:
            lines.append(f"cache {name} : {d['path']} ({d['files']} fichiers, {d['bytes']} octets)")
    if r["warnings"]:
        lines.append("")
        lines += [f"AVERTISSEMENT : {w}" for w in r["warnings"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kernox.diagnostics", description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="compile et vérifie de petites réductions par backend")
    ap.add_argument("--backend", action="append", choices=("cuda", "hip", "sycl"), help="restreindre (répétable)")
    ap.add_argument("--json", action="store_true", help="sortie JSON")
    a = ap.parse_args(argv)
    r = diagnostics(check=a.check, backends=a.backend)
    print(json.dumps(r, indent=2, default=str) if a.json else format_report(r))
    failed = any(not c["ok"] for b in r["backends"].values() for c in (b.get("check") or {}).values())
    return 1 if failed else 0

