from typing import Optional, Union

from ..exponential_function.power import (
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)
from .hypercos import hypercos


def hypersec(
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute sech(x) = 1 / cosh(x).
    """
    number_system_value = _validate_number_system(number_system)
    if number_system_value == "real":
        _validate_real_number("x", x)
    else:
        _validate_number("x", x)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    cosh_value = complex(
        hypercos(x, tol=tol_value, max_terms=max_terms_value, number_system=number_system_value)
    )
    if abs(cosh_value) <= tol_value:
        raise ZeroDivisionError("cosh(x) is zero, sech(x) is undefined.")

    normalized = _normalize(1.0 / cosh_value, tol_value)
    if number_system_value == "real":
        if isinstance(normalized, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized)
    return normalized
