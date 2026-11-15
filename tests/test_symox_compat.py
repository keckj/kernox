"""symox capability detection: the pinned path (2e623e66) and the recent path are equivalent."""
import pytest

from kernox import _symox_compat as compat
from kernox.codegen.pair import DIALECTS, pair_function

from corpus import reductions


def test_capabilities_and_forced_pinned(monkeypatch):
    caps = compat.capabilities()
    assert set(caps) >= {"cuda_logic", "cuda_reciprocal_sqrt", "cudacode_function", "lambdify_broadcasts"}
    monkeypatch.setenv("KERNOX_SYMOX_COMPAT", "pinned")
    assert not any(v for k, v in compat.capabilities().items() if k not in ("symox_version", "forced_pinned"))


@pytest.mark.parametrize("name", list(reductions()))
@pytest.mark.parametrize("dialect", DIALECTS)
def test_same_abi_on_both_paths(name, dialect, monkeypatch):
    plan = reductions()[name].to_plan("float32")
    from symox.extra.codegen.pytorch import _contains_conditionals
    # Conditional formulas go through the region-based CSE of pair.py
    # (cudacode_function applies no CSE in the presence of Piecewise).
    conditional = _contains_conditionals(list(plan.outputs))
    use_recent = compat.cudacode_function() and dialect != "sycl" and not conditional
    recent = pair_function(plan, dialect)
    monkeypatch.setenv("KERNOX_SYMOX_COMPAT", "pinned")
    pinned = pair_function(plan, dialect)
    for field in ("name", "out_type", "arg_layout", "out_fields", "out_ctypes", "scalar_type"):
        assert getattr(recent, field) == getattr(pinned, field)
    if use_recent:
        assert "symox cudacode_function" in recent.source
    else:
        # Logical operators: recent cudacode prints them natively, the pinned path
        # goes through a workaround; only the parentheses differ.
        def tokens(src):
            return "".join(c for c in src if c not in "() \n\t")
        assert tokens(recent.source) == tokens(pinned.source)
