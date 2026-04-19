from typing import Optional, Union

from ..constant.epsilon import epsilon
from ..gamma_function import gamma

_DEFAULT_TOL = epsilon()


def _validate_number_system(number_system: str) -> str:
    if not isinstance(number_system, str):
        raise TypeError("number_system must be a string.")
    value = number_system.lower()
    if value not in ("real", "complex"):
        raise ValueError("number_system must be either 'real' or 'complex'.")
    return value


def _validate_tol(tol: Optional[float]) -> float:
    if tol is None:
        return _DEFAULT_TOL
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError("tol must be a positive real number.")
    return float(tol)


def _validate_number(name: str, value: object) -> Union[int, float, complex]:
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise TypeError(f"{name} must be an int, float, or complex number.")
    return value


def _validate_real_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float in real mode.")
    return float(value)


def _normalize(value: complex, tol: float) -> Union[float, complex]:
    if abs(value.imag) <= tol * max(1.0, abs(value.real)):
        return float(value.real)
    return value


def beta(
    x: Union[int, float, complex],
    y: Union[int, float, complex],
    tol: Optional[float] = None,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute Euler beta function:
        B(x, y) = Gamma(x) * Gamma(y) / Gamma(x + y)
    """
    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)

    if number_system_value == "real":
        x_real = _validate_real_number("x", x)
        y_real = _validate_real_number("y", y)

        numerator = gamma(x_real, tol=tol_value, number_system="real")
        numerator *= gamma(y_real, tol=tol_value, number_system="real")
        denominator = gamma(x_real + y_real, tol=tol_value, number_system="real")

        if denominator == 0:
            raise ZeroDivisionError("beta denominator is zero.")

        result = complex(numerator, 0.0) / complex(denominator, 0.0)
        normalized = _normalize(result, tol_value)
        if isinstance(normalized, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized)

    x_value = _validate_number("x", x)
    y_value = _validate_number("y", y)

    numerator_complex = complex(gamma(x_value, tol=tol_value, number_system="complex"))
    numerator_complex *= complex(gamma(y_value, tol=tol_value, number_system="complex"))
    denominator_complex = complex(gamma(complex(x_value) + complex(y_value), tol=tol_value, number_system="complex"))

    if abs(denominator_complex) <= tol_value:
        raise ZeroDivisionError("beta denominator is approximately zero.")

    return _normalize(numerator_complex / denominator_complex, tol_value)
