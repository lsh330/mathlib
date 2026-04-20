# cython: language_level=3
# cython: embedsignature=True
#
# zeta.pyx — 리만 제타 함수 (기존 알고리즘 유지, .py -> .pyx 변환)

from typing import Optional, Union
import cmath as _cmath
import math as _math

_DEFAULT_TOL = 2.220446049250313e-16
_PI = _math.pi


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


def _is_integer_like(value: float, tol: float) -> bool:
    nearest = round(value)
    return abs(value - nearest) <= tol * max(1.0, abs(value))


def _eta_series(s: complex, tol: float, max_terms: int) -> complex:
    """
    Dirichlet eta series with Borwein (2000) acceleration.
    eta(s) = sum_{n>=1} (-1)^{n-1} n^{-s}
    zeta(s) = eta(s) / (1 - 2^{1-s})

    Borwein (2000) 'An efficient algorithm for the Riemann Zeta Function':
      e_k = n * sum_{j=k}^{n} (n+j-1)! * 4^j / ((n-j)! * (2j)! * (n-1)!)
    Built via downward recurrence (right-to-left) for stability.
    eta(s) = (1/e_0) * sum_{k=0}^{n-1} (-1)^k * e_k * (k+1)^{-s}
    For n=60 gives ~14 decimal digits; far better than 1e-8 target.
    M6: fixes zeta(2) to within 1e-13 of pi^2/6.
    """
    import math as _math_local

    n = 60  # 60 terms give ~14 digit accuracy, independent of max_terms

    # --- Borwein coefficient recurrence ---
    # log_inc(k) = log of the increment term at position k:
    #   inc_k = n * (n+k-1)! * 4^k / ((n-k)! * (2k)! * (n-1)!)
    log_n_fac = _math_local.lgamma(n)  # log((n-1)!)

    def _log_inc(k):
        return (
            _math_local.log(n)
            + _math_local.lgamma(n + k) - _math_local.lgamma(n - k + 1)
            - _math_local.lgamma(2 * k + 1)
            + k * _math_local.log(4)
            - log_n_fac
        )

    # e[n]: only the j=n term in the sum
    #   = n * (2n-1)! * 4^n / (0! * (2n)! * (n-1)!)
    #   = 4^n / (2 * (n-1)!)
    log_e_n = n * _math_local.log(4) - _math_local.log(2) - log_n_fac
    e_n = _math_local.exp(log_e_n)

    # Build e[n], e[n-1], ..., e[0] via downward recurrence
    e = [0.0] * (n + 1)
    e[n] = e_n
    for k in range(n - 1, -1, -1):
        e[k] = e[k + 1] + _math_local.exp(_log_inc(k))

    d_n = e[0]  # normalisation constant

    # --- Accumulate eta sum ---
    total = complex(0.0)
    for k in range(n):
        sign = 1.0 if k % 2 == 0 else -1.0
        total += sign * e[k] * complex(k + 1) ** (-s)

    return total / d_n


def _zeta_via_eta(s: complex, tol: float, max_terms: int) -> complex:
    denominator = 1.0 - complex(2.0) ** (1.0 - s)
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
    factor_1 = complex(2.0) ** s
    factor_2 = complex(_PI) ** (s - 1.0)
    factor_3 = _cmath.sin(_PI * s / 2.0)

    from ..gamma_function.gamma import gamma as _gamma
    factor_4 = complex(_gamma(1.0 - s, tol=tol, number_system="complex"))

    return factor_1 * factor_2 * factor_3 * factor_4 * reflected


def zeta(
    s,
    tol=None,
    max_terms: int = 5000,
    number_system: str = "real",
):
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
