from typing import Optional, Union

from ..constant.e import e
from ..constant.epsilon import epsilon
from ..constant.pi import pi
from ..exponential_function.power import power
from ..trigonometric_function.sin import sin

_DEFAULT_TOL = epsilon()
_LANCZOS_G = 7.0
_LANCZOS_COEFFS = [
    0.9999999999998099,
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    0.000009984369578019572,
    0.00000015056327351493116,
]


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


def _is_integer_like(value: float, tol: float) -> bool:
    nearest = round(value)
    return abs(value - nearest) <= tol * max(1.0, abs(value))


def _check_pole(z: complex, tol: float) -> None:
    if abs(z.imag) <= tol and z.real <= 0.0 and _is_integer_like(z.real, tol):
        raise ValueError("gamma has poles at non-positive integers.")


def _gamma_lanczos(z: complex, tol: float, depth: int = 0) -> complex:
    if depth > 30:
        raise RuntimeError("gamma recursion depth exceeded.")

    _check_pole(z, tol)

    if z.real < 0.5:
        sin_term = sin(pi() * z, tol=tol, number_system="complex")
        if abs(complex(sin_term)) <= tol:
            raise ValueError("gamma is undefined at this input (pole).")
        reflected = _gamma_lanczos(1.0 - z, tol, depth + 1)
        return pi() / (complex(sin_term) * reflected)

    z_shifted = z - 1.0
    acc = complex(_LANCZOS_COEFFS[0], 0.0)
    for i in range(1, len(_LANCZOS_COEFFS)):
        acc += _LANCZOS_COEFFS[i] / (z_shifted + i)

    t = z_shifted + _LANCZOS_G + 0.5
    sqrt_two_pi = power(2.0 * pi(), 0.5, tol=tol, number_system="complex")
    t_power = power(t, z_shifted + 0.5, tol=tol, number_system="complex")
    exp_part = power(e(), -t, tol=tol, number_system="complex")

    return complex(sqrt_two_pi) * complex(t_power) * complex(exp_part) * acc


def gamma(
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute Gamma(x) using Lanczos approximation + reflection formula.
    """
    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)

    if number_system_value == "real":
        x_real = _validate_real_number("x", x)
        _check_pole(complex(x_real, 0.0), tol_value)
        result = _gamma_lanczos(complex(x_real, 0.0), tol_value)
        normalized = _normalize(result, tol_value)
        if isinstance(normalized, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized)

    x_value = _validate_number("x", x)
    result_complex = _gamma_lanczos(complex(x_value), tol_value)
    return _normalize(result_complex, tol_value)
