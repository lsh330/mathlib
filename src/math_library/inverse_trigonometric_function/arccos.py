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
from .arcsin import arcsin

_HALF_PI = 0.5 * pi()


def arccos(
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute principal arccos(z) = pi/2 - arcsin(z).
    """
    number_system_value = _validate_number_system(number_system)
    if number_system_value == "real":
        x_real = _validate_real_number("x", x)
        if x_real < -1.0 or x_real > 1.0:
            raise ValueError("x must be in [-1, 1] in real mode.")
    else:
        _validate_number("x", x)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    result = complex(_HALF_PI, 0.0) - complex(
        arcsin(x, tol=tol_value, max_terms=max_terms_value, number_system=number_system_value)
    )
    normalized = _normalize(result, tol_value)

    if number_system_value == "real":
        if isinstance(normalized, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized)

    return normalized
