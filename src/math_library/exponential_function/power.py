from typing import Optional, Union

from ..constant.epsilon import epsilon
from ..constant.pi import pi

_PI = pi()
_HALF_PI = _PI / 2.0
_QUARTER_PI = _PI / 4.0
_LN2 = 0.6931471805599453
_DEFAULT_TOL = epsilon()


def _validate_number(name: str, value: object) -> Union[int, float, complex]:
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise TypeError(f"{name} must be an int, float, or complex number.")
    return value


def _validate_real_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float in real mode.")
    return float(value)


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


def _normalize(value: complex, tol: float) -> Union[float, complex]:
    if abs(value.imag) <= tol * max(1.0, abs(value.real)):
        return float(value.real)
    return value


def _is_integer_like(value: float, tol: float) -> bool:
    nearest = round(value)
    return abs(value - nearest) <= tol * max(1.0, abs(value))


def _power_integer_real(base: float, exponent: int) -> float:
    if exponent == 0:
        return 1.0

    negative_power = exponent < 0
    n = abs(exponent)
    result = 1.0
    b = base

    while n > 0:
        if n % 2 == 1:
            result *= b
        b *= b
        n //= 2

    if negative_power:
        return 1.0 / result
    return result


def _exp_series(z: complex, tol: float, max_terms: int) -> complex:
    term = 1.0 + 0.0j
    result = 1.0 + 0.0j

    for n in range(1, max_terms):
        term *= z / n
        result += term
        if abs(term) <= tol * max(1.0, abs(result)):
            return result

    raise RuntimeError("Exponential series did not converge within max_terms.")


def _ln_real_positive(x: float, tol: float, max_terms: int) -> float:
    if x <= 0:
        raise ValueError("x must be positive.")

    if x == 1.0:
        return 0.0

    reduction = 0.0
    value = x

    while value > 2.0:
        value *= 0.5
        reduction += _LN2

    while value < 0.5:
        value *= 2.0
        reduction -= _LN2

    y = (value - 1.0) / (value + 1.0)
    y_squared = y * y

    term = y
    result = y

    for n in range(1, max_terms):
        term *= y_squared
        addend = term / (2 * n + 1)
        result += addend
        if abs(addend) <= tol * max(1.0, abs(result)):
            return 2.0 * result + reduction

    raise RuntimeError("Natural logarithm series did not converge within max_terms.")


def _arctan_real(x: float, tol: float, max_terms: int) -> float:
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return _QUARTER_PI
    if x == -1.0:
        return -_QUARTER_PI

    if abs(x) <= 1.0:
        term = x
        result = x
        x_squared = x * x

        for n in range(1, max_terms):
            term *= -x_squared
            addend = term / (2 * n + 1)
            result += addend
            if abs(addend) <= tol * max(1.0, abs(result)):
                return result

        raise RuntimeError("Arctangent series did not converge within max_terms.")

    if x > 0.0:
        return _HALF_PI - _arctan_real(1.0 / x, tol, max_terms)
    return -_HALF_PI - _arctan_real(1.0 / x, tol, max_terms)


def _arg_principal(z: complex, tol: float, max_terms: int) -> float:
    if z == 0:
        raise ValueError("Argument is undefined for zero.")

    x = z.real
    y = z.imag

    if abs(y) <= tol:
        if x > 0:
            return 0.0
        return _PI

    if abs(x) <= tol:
        return _HALF_PI if y > 0 else -_HALF_PI

    angle = _arctan_real(y / x, tol, max_terms)
    if x > 0:
        return angle
    if y >= 0:
        return angle + _PI
    return angle - _PI


def _complex_log(z: complex, tol: float, max_terms: int) -> complex:
    if z == 0:
        raise ValueError("Logarithm is undefined for zero.")

    r_squared = z.real * z.real + z.imag * z.imag
    real_part = 0.5 * _ln_real_positive(r_squared, tol, max_terms)
    imag_part = _arg_principal(z, tol, max_terms)
    return complex(real_part, imag_part)


def power(
    base: Union[int, float, complex],
    exponent: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute base**exponent.

    Modes:
        number_system="real" (default):
            Only real inputs are accepted, and outputs are real.
            Undefined real-domain cases raise ValueError.
        number_system="complex":
            Complex principal branch is used.
    """
    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    if number_system_value == "real":
        base_real = _validate_real_number("base", base)
        exponent_real = _validate_real_number("exponent", exponent)

        if base_real == 0.0:
            if exponent_real == 0.0:
                raise ValueError("0**0 is undefined.")
            if exponent_real < 0.0:
                raise ValueError("0 cannot be raised to a negative exponent.")
            return 0.0

        if exponent_real == 0.0:
            return 1.0

        if base_real < 0.0:
            if not _is_integer_like(exponent_real, tol_value):
                raise ValueError("negative base with non-integer exponent is undefined in real mode.")
            integer_exponent = int(round(exponent_real))
            return _power_integer_real(base_real, integer_exponent)

        exponent_argument_real = exponent_real * _ln_real_positive(base_real, tol_value, max_terms_value)
        result_real = _exp_series(complex(exponent_argument_real, 0.0), tol_value, max_terms_value)
        normalized = _normalize(result_real, tol_value)
        if isinstance(normalized, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized)

    base_value = _validate_number("base", base)
    exponent_value = _validate_number("exponent", exponent)
    base_complex = complex(base_value)
    exponent_complex = complex(exponent_value)

    if base_complex == 0:
        if exponent_complex == 0:
            raise ValueError("0**0 is undefined.")
        if abs(exponent_complex.imag) > tol_value:
            raise ValueError("0 cannot be raised to a complex exponent with non-zero imaginary part.")
        if exponent_complex.real < 0:
            raise ValueError("0 cannot be raised to a negative exponent.")
        return 0.0

    if exponent_complex == 0:
        return 1.0

    exponent_argument = exponent_complex * _complex_log(base_complex, tol_value, max_terms_value)
    result = _exp_series(exponent_argument, tol_value, max_terms_value)
    return _normalize(result, tol_value)

