# cython: language_level=3
# cython: embedsignature=True
#
# legendre.pyx — 르장드르 다항식 (기존 알고리즘 유지, .py -> .pyx 변환)

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


def _validate_order(n) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer.")
    if n < 0:
        raise ValueError("n must be non-negative.")
    return n


def _validate_input(x, number_system: str):
    if number_system == "real":
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError("x must be an int or float in real mode.")
        return float(x)
    if isinstance(x, bool) or not isinstance(x, (int, float, complex)):
        raise TypeError("x must be an int, float, or complex number.")
    return x


def _normalize(value: complex, tol: float):
    if abs(value.imag) <= tol * max(1.0, abs(value.real)):
        return float(value.real)
    return value


def legendre_polynomial(
    n,
    x,
    tol=None,
    number_system="real",
):
    """
    Compute Legendre polynomial P_n(x) using the three-term recurrence.
    """
    n_value = _validate_order(n)
    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)
    x_value = _validate_input(x, number_system_value)
    x_complex = complex(x_value)

    if n_value == 0:
        return 1.0
    if n_value == 1:
        value = _normalize(x_complex, tol_value)
        if number_system_value == "real" and isinstance(value, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return value

    p_nm2 = 1.0 + 0.0j
    p_nm1 = x_complex

    for k in range(2, n_value + 1):
        p_n = ((2 * k - 1) * x_complex * p_nm1 - (k - 1) * p_nm2) / k
        p_nm2 = p_nm1
        p_nm1 = p_n

    normalized = _normalize(p_nm1, tol_value)
    if number_system_value == "real" and isinstance(normalized, complex):
        raise RuntimeError("real-mode result unexpectedly became complex.")
    return normalized
