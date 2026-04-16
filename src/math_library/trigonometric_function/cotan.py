from typing import Optional, Union

from ..exponential_function.power import (
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)
from .cos import cos
from .sin import sin


def cotan(
    x: Union[int, float, complex],
    unit: str = "rad",
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute cotan(x) = cos(x) / sin(x).
    """
    number_system_value = _validate_number_system(number_system)
    if number_system_value == "real":
        _validate_real_number("x", x)
    else:
        _validate_number("x", x)
    if not isinstance(unit, str):
        raise TypeError("unit must be a string.")
    if unit.lower() not in ("rad", "deg"):
        raise ValueError("unit must be either 'rad' or 'deg'.")

    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    sin_value = complex(
        sin(x, unit=unit, tol=tol_value, max_terms=max_terms_value, number_system=number_system_value)
    )
    cos_value = complex(
        cos(x, unit=unit, tol=tol_value, max_terms=max_terms_value, number_system=number_system_value)
    )

    if abs(sin_value) <= tol_value:
        raise ZeroDivisionError("sin(x) is zero, cotan(x) is undefined.")

    normalized = _normalize(cos_value / sin_value, tol_value)

    if number_system_value == "real":
        if isinstance(normalized, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized)

    return normalized
