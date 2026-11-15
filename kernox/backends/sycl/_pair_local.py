"""TEMPORARY pair function for the sycl backend.

Superseded by ``kernox.codegen.pair.pair_function(plan, "sycl")`` (core)
as soon as it is available; the backend uses it automatically when it exists.

Produces the contract v0 signature (``docs/kernel-backend-contract.md``):

    struct kp_out { T o0; ...; };            // bool for a boolean output
    inline kp_out kp_pair(const T* __restrict__ a0, ...);   // plan.args order

by wrapping ``symox.extra.codegen.sycl.syclcode_function`` (scalar arguments
passed by value). symox itself disables global CSE in the presence of
conditionals; symox emits no fast-math option.
"""

from __future__ import annotations

from dataclasses import dataclass

_CTYPE = {"float32": "float", "float64": "double"}


@dataclass(frozen=True)
class PairSource:
    source: str
    name: str
    out_type: str
    arg_layout: tuple  # ((name, role, dim), ...)
    out_types: tuple  # C++ type of each field o0..o{D-1}


def pair_function(plan, dialect: str = "sycl") -> PairSource:
    if dialect != "sycl":
        raise ValueError("local pair function: sycl dialect only")
    from symox import _core
    from symox.extra.codegen.sycl import syclcode_function

    T = _CTYPE[plan.dtype]
    symbols = [s for a in plan.args for s in a.symbols]
    outputs = list(plan.outputs)
    inner = "kp_pair_scalar"
    body = syclcode_function(inner, symbols, outputs if len(outputs) > 1 else outputs[0],
                             dtype=plan.dtype)
    out_types = tuple(
        "bool" if _core._sycl_type(o) == "bool" else T for o in outputs
    )
    lines = [body, "struct kp_out {"]
    lines += [f"    {t} o{c};" for c, t in enumerate(out_types)]
    lines.append("};")
    params = ", ".join(f"const {T}* __restrict__ kp_a{n}" for n in range(len(plan.args)))
    lines.append(f"inline kp_out kp_pair({params})")
    lines.append("{")
    call = ", ".join(f"kp_a{n}[{c}]" for n, a in enumerate(plan.args) for c in range(a.dim))
    if len(outputs) == 1:
        lines.append(f"    return kp_out{{ {inner}({call}) }};")
    else:
        lines.append(f"    const auto r = {inner}({call});")
        fields = ", ".join(f"r._sx_o_{c}" for c in range(len(outputs)))
        lines.append(f"    return kp_out{{ {fields} }};")
    lines.append("}")
    return PairSource(
        source="\n".join(lines) + "\n",
        name="kp_pair",
        out_type="kp_out",
        arg_layout=tuple((a.name, a.role, a.dim) for a in plan.args),
        out_types=out_types,
    )
