"""Formula corpus shared by the code generation / compilation tests."""
import symox

import kernox as kp
from kernox import Param, Vi, Vj, where


def reductions():
    x, y, w, s = Vi("x", 3), Vj("y", 3), Vj("w", 2), Param("sigma")
    d2 = ((x - y) ** 2).sum()
    funcs = (kp.sin(x[0]) * kp.cos(y[1]) + kp.atan2(x[1], y[0]) + kp.log(abs(x[2]) + 1)
             + kp.sign(x[0] - y[0]) + kp.floor(y[2]) + kp.minimum(x[0], y[0]) + symox.pi * x[1]
             + symox.Rational(1, 3) * y[1] + kp.tanh(x[2] * y[2]) + kp.exp(-abs(x[0]))
             + kp.sqrt(abs(y[1])) + (x[0] - y[2]) ** 3 / (1 + (x[1] - y[1]) ** 4) + 0.1 * s)
    logic = ((d2 < 1.5) & ~(x[0] > y[0])) | (y[1] >= 0.5) ^ kp.eq(x[2], y[2])
    return {
        "gauss_vec": (kp.exp(-d2 / (2.0 * s ** 2)) * w).sum_over("j"),
        "count": (d2 <= 2.5).sum_over("j"),
        "logic": logic.sum_over("j"),
        "min_vec": (x - y).min(over="j"),
        "kmin": d2.kmin(3, over="j"),
        "argkmin_over_i": d2.argkmin(2, over="i"),
        "where_sqrt": where(d2 > 1.0, 1 / kp.sqrt(d2 - 1.0), 0.0).sum_over("j"),
        "where_inf": where(x[0] < y[0], x[1], float("-inf")).max(over="j"),
        "funcs": funcs.sum_over("j"),
        # where inside a condition: symox builds a boolean ITE (conformance regression)
        "where_in_cond": where(where(x[0] <= y[1], x[0], y[0]) > 0,
                               kp.log(where(x[0] <= y[1], x[0], y[0])), y[0]).max(over="j"),
        # complex-valued engine artifacts in guarded branches (builder.realify)
        "guarded_sqrt_branch": where(where(y[1] >= 0, kp.sqrt(y[1]), -1.5) >= 0,
                                     kp.sqrt(where(y[1] >= 0, kp.sqrt(y[1]), -1.5)), x[1]).max(over="j"),
        "guarded_div_branch": where(kp.ne(where(kp.ne(x[0], y[0]), 0, y[2]), 0),
                                    x[2] / where(kp.ne(x[0], y[0]), 0, y[2]), 0).sum_over("j"),
        # non-atomic symbolic constants (cos(1) is ambiguous for NVRTC in double)
        "const_calls": kp.maximum(kp.cos(-1), where(x[2] <= y[1], y[2], x[1] * kp.log(3) + kp.sqrt(2))).max(over="j"),
        # region-based CSE: subexpressions hoisted out of where, NaN/inf/1/0 in the inactive branch
        "where_cse_inactive": (where(d2 > 1.0, kp.log(d2 - 1.0) * kp.exp(-d2), kp.exp(-d2) * kp.sqrt(1.0 - d2))
                               + where(kp.eq(x[0], y[0]), 0.0, 1 / (x[0] - y[0]))
                               + where(x[0] > y[0], 1 / (x[0] - y[0]), 2.0)).sum_over("j"),
        "abs_exp_re": (kp.abs(kp.exp(kp.sin(where(x[1] >= 0, kp.sqrt(x[1]), 0.5)) * 3)) * y[0]).sum_over("j"),
    }
