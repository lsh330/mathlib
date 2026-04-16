from typing import Optional, Union

from ..exponential_function.power import (
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)
from .arcsin import arcsin


def arccosec(
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute principal arccosec(z) = arcsin(1/z).
    """
    number_system_value = _validate_number_system(number_system)
    if number_system_value == "real":
        x_value: Union[int, float, complex] = _validate_real_number("x", x)
        if x_value == 0.0:
            raise ZeroDivisionError("x is zero, arccosec(x) is undefined.")
        if abs(x_value) < 1.0:
            raise ValueError("x must satisfy |x| >= 1 in real mode.")
    else:
        x_value = _validate_number("x", x)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    if number_system_value == "real":
        reciprocal_real = 1.0 / float(x_value)
        return arcsin(reciprocal_real, tol=tol_value, max_terms=max_terms_value, number_system="real")

    z = complex(x_value)
    if z == 0:
        raise ZeroDivisionError("x is zero, arccosec(x) is undefined.")
    return arcsin(1.0 / z, tol=tol_value, max_terms=max_terms_value, number_system="complex")
