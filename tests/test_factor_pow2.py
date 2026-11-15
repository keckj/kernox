"""Mise en facteur exacte des coefficients 2^k dans les dérivées (``autograd._factor_pow2``)."""

from __future__ import annotations

import symox

from kernox.autograd import _factor_pow2, derivative

a, b, c, d, s = symox.symbols("a b c d s")


def test_square_difference_keeps_shared_difference():
    # ∂(a − b)²/∂a : 2·(a − b), et non 2a − 2b
    out = derivative((a - b) ** 2, a)
    assert (a - b) in out.args
    assert symox.expand(out - 2 * (a - b)) == 0


def test_negative_coefficient_sign_extracted():
    out = derivative((a - b) ** 2, b)
    assert (a - b) in out.args
    assert symox.expand(out + 2 * (a - b)) == 0


def test_gradient_shares_difference_with_forward():
    f = s * symox.exp(-s * ((a - b) ** 2 + (c - d) ** 2))
    out = derivative(f, a)
    assert (a - b) in symox.Mul.make_args(out)
    assert symox.simplify(out - symox.diff(f, a)) == 0


def test_non_power_of_two_left_alone():
    e = 3 * a - 3 * b
    assert _factor_pow2(e) == e
    e = 2 * a - 4 * b
    assert _factor_pow2(e) == e
    e = a / 2 - b / 2
    assert _factor_pow2(e) == e
