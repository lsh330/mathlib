from typing import Optional, Union

from ..constant.pi import pi
from ..exponential_function.power import (
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)
from .._core.trigonometric import tan as _tan_core
from .cos import cos
from .sin import sin

_PI = pi()
_DEG2RAD = _PI / 180.0


def tan(
    x: Union[int, float, complex],
    unit: str = "rad",
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute tan(x).
    real mode: Cython _core 구현 (musl fdlibm, 1 ULP 정밀도)
    complex mode: sin(z) / cos(z)
    """
    # fast-path
    if type(x) is float and unit == "rad" and number_system == "real":
        return _tan_core(x)

    number_system_value = _validate_number_system(number_system)

    if not isinstance(unit, str):
        raise TypeError("unit must be a string.")
    unit_lower = unit.lower()
    if unit_lower not in ("rad", "deg"):
        raise ValueError("unit must be either 'rad' or 'deg'.")

    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    if number_system_value == "real":
        x_value = _validate_real_number("x", x)
        x_rad = x_value * _DEG2RAD if unit_lower == "deg" else x_value
        return _tan_core(x_rad)

    # complex mode
    _validate_number("x", x)
    sin_value = complex(
        sin(x, unit=unit, tol=tol_value, max_terms=max_terms_value, number_system=number_system_value)
    )
    cos_value = complex(
        cos(x, unit=unit, tol=tol_value, max_terms=max_terms_value, number_system=number_system_value)
    )

    if abs(cos_value) <= tol_value:
        raise ZeroDivisionError("cos(x) is zero, tan(x) is undefined.")

    result = sin_value / cos_value
    return _normalize(result, tol_value)
