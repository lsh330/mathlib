# cython: language_level=3
# cython: embedsignature=True
#
# lambert_w.pyx — Lambert W 함수 (기존 알고리즘 유지, .py -> .pyx 변환)

from typing import Optional, Union
import cmath as _cmath
import math as _math

_DEFAULT_TOL = 2.220446049250313e-16
_E_VAL = _math.e


def _validate_number_system(number_system: str) -> str:
    if not isinstance(number_system, str):
        raise TypeError("number_system must be a string.")
    value = number_system.lower()
    if value not in ("real", "complex"):
        raise ValueError("number_system must be either 'real' or 'complex'.")
    return value


def _validate_tol(tol) -> float:
    if tol is None:
        return _DEFAULT_TOL
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError("tol must be a positive real number.")
    return float(tol)


def _validate_max_iter(max_iter: int) -> int:
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    return max_iter


def _validate_branch(branch) -> int:
    if isinstance(branch, bool) or not isinstance(branch, int):
        raise TypeError("branch must be an integer.")
    return branch


def _validate_number(name: str, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise TypeError(f"{name} must be an int, float, or complex number.")
    return value


def _validate_real_number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float in real mode.")
    return float(value)


def _normalize(value: complex, tol: float):
    if abs(value.imag) <= tol * max(1.0, abs(value.real)):
        return float(value.real)
    return value


def _initial_guess_complex(x: complex, branch: int, tol: float) -> complex:
    if x == 0:
        return 0.0 + 0.0j
    if branch == 0:
        if abs(x) < 1.0:
            return x
        ln_x = _cmath.log(x)
        if abs(ln_x) > tol:
            ln_ln_x = _cmath.log(ln_x)
            return ln_x - ln_ln_x
        return ln_x
    ln_x = _cmath.log(x)
    w = ln_x + complex(0.0, 2.0 * _cmath.pi * branch)
    if abs(w) <= tol:
        return w
    ln_w = _cmath.log(w)
    return w - ln_w


def _iterate_lambert_w(x: complex, branch: int, tol: float, max_iter: int,
                        initial_guess=None) -> complex:
    if initial_guess is None:
        w = _initial_guess_complex(x, branch, tol)
    else:
        w = initial_guess

    for _ in range(max_iter):
        exp_w = _cmath.exp(w)
        f = w * exp_w - x
        w_plus_1 = w + 1.0
        if abs(w_plus_1) <= tol:
            w_plus_1 += tol
        denominator = exp_w * w_plus_1 - ((w + 2.0) * f) / (2.0 * w_plus_1)
        if abs(denominator) <= tol:
            raise RuntimeError("Lambert W iteration encountered a near-zero denominator.")
        delta = f / denominator
        w_next = w - delta
        if abs(w_next - w) <= tol * max(1.0, abs(w_next)):
            return w_next
        w = w_next

    raise RuntimeError("Lambert W iteration did not converge within max_iter.")


def _lambert_w_real_branch_minus_one(x_real: float, tol: float, max_iter: int) -> float:
    if x_real >= 0.0:
        raise ValueError("branch -1 requires x in [-1/e, 0).")
    minus_inv_e = -1.0 / _E_VAL
    if x_real < minus_inv_e - tol:
        raise ValueError("real branch -1 is undefined for x < -1/e.")
    if abs(x_real - minus_inv_e) <= tol:
        return -1.0

    left = -50.0
    right = -1.0

    def exp_real(value: float) -> float:
        return _math.exp(value)

    def f(ww: float) -> float:
        return ww * exp_real(ww) - x_real

    f_left = f(left)
    f_right = f(right)
    if f_left * f_right > 0:
        raise RuntimeError("Failed to bracket branch -1 root.")

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = f(mid)
        if abs(f_mid) <= tol:
            return mid
        if abs(right - left) <= tol * max(1.0, abs(mid)):
            return mid
        if f_left * f_mid > 0:
            left = mid
            f_left = f_mid
        else:
            right = mid
            f_right = f_mid

    return 0.5 * (left + right)


def lambert_w(
    x,
    branch: int = 0,
    tol=None,
    max_iter: int = 100,
    number_system: str = "real",
):
    """
    Compute Lambert W such that W(x) * exp(W(x)) = x.
    """
    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)
    max_iter_value = _validate_max_iter(max_iter)
    branch_value = _validate_branch(branch)

    minus_inv_e = -1.0 / _E_VAL

    if number_system_value == "real":
        x_real = _validate_real_number("x", x)
        if branch_value not in (0, -1):
            raise ValueError("real mode supports only branch 0 and -1.")
        if x_real < minus_inv_e - tol_value:
            raise ValueError("real Lambert W is undefined for x < -1/e.")
        if branch_value == -1 and x_real > tol_value:
            raise ValueError("branch -1 in real mode is defined on [-1/e, 0].")
        if abs(x_real - minus_inv_e) <= tol_value:
            return -1.0
        if abs(x_real) <= tol_value:
            if branch_value == 0:
                return 0.0
            raise ValueError("branch -1 is not finite at x=0.")
        if branch_value == -1:
            return _lambert_w_real_branch_minus_one(x_real, tol_value, max_iter_value)
        result = _iterate_lambert_w(complex(x_real, 0.0), branch_value, tol_value, max_iter_value)
        normalized = _normalize(result, tol_value)
        if isinstance(normalized, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized)

    x_value = _validate_number("x", x)
    x_complex = complex(x_value)
    if abs(x_complex) <= tol_value:
        if branch_value == 0:
            return 0.0
        raise ValueError("only principal branch is finite at x=0.")
    result_complex = _iterate_lambert_w(x_complex, branch_value, tol_value, max_iter_value)
    return _normalize(result_complex, tol_value)
