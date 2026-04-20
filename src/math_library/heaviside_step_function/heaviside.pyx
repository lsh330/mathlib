# cython: language_level=3
# cython: embedsignature=True
#
# heaviside.pyx — Heaviside 계단 함수 (기존 알고리즘 유지)

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


def _validate_at_zero(at_zero) -> float:
    if isinstance(at_zero, bool) or not isinstance(at_zero, (int, float)):
        raise TypeError("at_zero must be an int or float.")
    return float(at_zero)


def _validate_real_number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float in real mode.")
    return float(value)


def _validate_number(name: str, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise TypeError(f"{name} must be an int, float, or complex number.")
    return value


def heaviside(
    x,
    at_zero: float = 0.5,
    tol=None,
    number_system: str = "real",
) -> float:
    """
    Heaviside step function H(x).
    """
    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)
    at_zero_value = _validate_at_zero(at_zero)

    if number_system_value == "real":
        x_real = _validate_real_number("x", x)
    else:
        x_value = _validate_number("x", x)
        x_complex = complex(x_value)
        if abs(x_complex.imag) > tol_value:
            raise ValueError("Heaviside ordering is undefined for non-real complex values.")
        x_real = float(x_complex.real)

    if x_real > tol_value:
        return 1.0
    if x_real < -tol_value:
        return 0.0
    return at_zero_value
