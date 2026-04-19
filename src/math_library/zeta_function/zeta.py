from typing import Optional, Union

from ..constant.e import e
from ..constant.epsilon import epsilon
from ..constant.pi import pi
from ..exponential_function.power import power
from ..gamma_function import gamma
from ..logarithmic_function.log import log
from ..trigonometric_function.sin import sin

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


def _validate_max_terms(max_terms: int) -> int:
    if isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms <= 0:
        raise ValueError("max_terms must be a positive integer.")
    return max_terms


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


def _is_integer_like(value: float, tol: float) -> bool:
    nearest = round(value)
    return abs(value - nearest) <= tol * max(1.0, abs(value))


def _eta_series(s: complex, tol: float, max_terms: int) -> complex:
    total = 0.0 + 0.0j
    convergence_tol = max(tol, 1e-10)

    for n in range(1, max_terms + 1):
        term = complex(power(float(n), -s, tol=tol, max_terms=200, number_system="complex"))
        if n % 2 == 0:
            term = -term
        total += term

        if n >= 20 and abs(term) <= convergence_tol * max(1.0, abs(total)):
            return total

    return total


def _zeta_via_eta(s: complex, tol: float, max_terms: int) -> complex:
    denominator = 1.0 - complex(power(2.0, 1.0 - s, tol=tol, max_terms=200, number_system="complex"))
    if abs(denominator) <= tol:
        raise ValueError("zeta is undefined at this input (pole or singular denominator).")

    eta_value = _eta_series(s, tol, max_terms)
    return eta_value / denominator


def _zeta_complex(s: complex, tol: float, max_terms: int, depth: int = 0) -> complex:
    if depth > 12:
        raise RuntimeError("zeta continuation recursion depth exceeded.")

    if abs(s - 1.0) <= tol:
        raise ValueError("zeta has a pole at s=1.")

    if abs(s) <= tol:
        return -0.5 + 0.0j

    if abs(s.imag) <= tol and s.real < 0.0 and _is_integer_like(s.real, tol):
        integer_value = int(round(s.real))
        if integer_value % 2 == 0:
            return 0.0 + 0.0j

    if s.real > 0.0:
        return _zeta_via_eta(s, tol, max_terms)

    reflected_argument = 1.0 - s
    reflected = _zeta_complex(reflected_argument, tol, max_terms, depth + 1)

    factor_1 = complex(power(2.0, s, tol=tol, max_terms=200, number_system="complex"))
    factor_2 = complex(power(pi(), s - 1.0, tol=tol, max_terms=200, number_system="complex"))
    factor_3 = complex(sin((pi() * s) / 2.0, tol=tol, number_system="complex"))
    factor_4 = complex(gamma(1.0 - s, tol=tol, number_system="complex"))

    return factor_1 * factor_2 * factor_3 * factor_4 * reflected


def zeta(
    s: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 5000,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute the Riemann zeta function using eta-series + analytic continuation.
    """
    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    if number_system_value == "real":
        s_real = _validate_real_number("s", s)
        result = _zeta_complex(complex(s_real, 0.0), tol_value, max_terms_value)
        normalized = _normalize(result, tol_value)
        if isinstance(normalized, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized)

    s_value = _validate_number("s", s)
    result_complex = _zeta_complex(complex(s_value), tol_value, max_terms_value)
    return _normalize(result_complex, tol_value)
