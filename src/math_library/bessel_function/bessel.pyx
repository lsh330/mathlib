# cython: language_level=3
# cython: embedsignature=True
#
# bessel.pyx — Bessel 함수 (기존 알고리즘 유지, .py -> .pyx 변환)

from typing import Optional, Union

_DEFAULT_TOL = 2.220446049250313e-16


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


def _validate_max_terms(max_terms: int) -> int:
    if isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms <= 0:
        raise ValueError("max_terms must be a positive integer.")
    return max_terms


def _validate_order(n) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer.")
    if n < 0:
        raise ValueError("n must be non-negative.")
    return n


def _validate_number(name: str, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise TypeError(f"{name} must be an int, float, or complex number.")
    return value


def _validate_real_number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float in real mode.")
    return float(value)


def _factorial(n: int) -> int:
    if n < 2:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _normalize(value: complex, tol: float):
    if abs(value.imag) <= tol * max(1.0, abs(value.real)):
        return float(value.real)
    return value


def _power_complex(base, exp_val, tol: float):
    """내부용 복소수 거듭제곱"""
    import cmath
    if base == 0:
        return 0.0
    return base ** exp_val


def bessel_j(
    n,
    x,
    tol=None,
    max_terms=200,
    number_system="real",
):
    """
    Bessel function of the first kind J_n(x) for integer n >= 0.
    """
    n_value = _validate_order(n)
    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    if number_system_value == "real":
        x_value = _validate_real_number("x", x)
    else:
        x_value = _validate_number("x", x)

    x_complex = complex(x_value)
    half_x = x_complex / 2.0

    if abs(half_x) <= tol_value:
        if n_value == 0:
            term = 1.0 + 0.0j
        else:
            term = 0.0 + 0.0j
    else:
        term = _power_complex(half_x, n_value, tol_value)
        term /= _factorial(n_value)

    result = term

    for m in range(1, max_terms_value):
        term *= -(half_x * half_x) / (m * (m + n_value))
        result += term

        if abs(term) <= tol_value * max(1.0, abs(result)):
            normalized = _normalize(result, tol_value)
            if number_system_value == "real" and isinstance(normalized, complex):
                raise RuntimeError("real-mode result unexpectedly became complex.")
            return normalized

    raise RuntimeError("Bessel series did not converge within max_terms.")


def bessel_j0(x, tol=None, max_terms: int = 200, number_system: str = "real"):
    return bessel_j(0, x, tol=tol, max_terms=max_terms, number_system=number_system)


def bessel_j1(x, tol=None, max_terms: int = 200, number_system: str = "real"):
    return bessel_j(1, x, tol=tol, max_terms=max_terms, number_system=number_system)
