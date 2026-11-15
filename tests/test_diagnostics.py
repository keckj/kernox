import json

import kernox as kp
import kernox.diagnostics as diag_mod  # noqa: F401  (package; kp.diagnostics stays the function)
from kernox.diagnostics import _SYMOX_CORE_FINGERPRINTS, format_report, main


def test_report_is_json_and_mentions_every_backend(capsys):
    assert callable(kp.diagnostics)
    r = kp.diagnostics()
    json.dumps(r, default=str)
    assert set(r["backends"]) == {"cuda", "hip", "sycl"}
    for b in r["backends"].values():
        assert b["available"] or b["reason"]
    assert "wisdom_dir" in r["cache"]
    assert format_report(r).startswith("Kernox")
    assert main(["--json", "--backend", "sycl"]) == 0
    assert json.loads(capsys.readouterr().out)["backends"].keys() == {"sycl"}


def test_symox_capability_warning(monkeypatch):
    from kernox import _symox_compat as compat

    r = kp.diagnostics(backends=())
    s = r["symox"]
    assert s["revision"] in (None, *_SYMOX_CORE_FINGERPRINTS.values())
    if all(v is True for v in s["capabilities"].values()):
        assert not s["warnings"]
    monkeypatch.setattr(compat, "cuda_logic", lambda: False)
    assert any("cuda_logic" in w for w in kp.diagnostics(backends=())["warnings"])
