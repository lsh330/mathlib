from typing import Optional, Union

from ..constant.e import e
from ..constant.pi import pi
from ..exponential_function import power
from ..exponential_function.power import (
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)

_PI = pi()


def cos(
    x: Union[int, float, complex],
    unit: str = "rad",
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute cos(x) with Euler's identity:
        cos(x) = (e^(ix) + e^(-ix)) / 2
    """
    number_system_value = _validate_number_system(number_system)
    if number_system_value == "real":
        x_value: Union[int, float, complex] = _validate_real_number("x", x)
    else:
        x_value = _validate_number("x", x)

    if not isinstance(unit, str):
        raise TypeError("unit must be a string.")

    unit_lower = unit.lower()
    if unit_lower not in ("rad", "deg"):
        raise ValueError("unit must be either 'rad' or 'deg'.")

    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    x_complex = complex(x_value)
    if unit_lower == "deg":
        x_complex = x_complex * _PI / 180.0

    e_value = e()
    exp_pos = complex(
        power(e_value, 1j * x_complex, tol=tol_value, max_terms=max_terms_value, number_system="complex")
    )
    exp_neg = complex(
        power(e_value, -1j * x_complex, tol=tol_value, max_terms=max_terms_value, number_system="complex")
    )
    result = (exp_pos + exp_neg) / 2.0
    normalized = _normalize(result, tol_value)

    if number_system_value == "real":
        if isinstance(normalized, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized)

    return normalized
