from typing import Optional, Union

from ..constant.e import e
from ..exponential_function import power
from ..exponential_function.power import (
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)


def hypersin(
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute sinh(x) = (e^x - e^(-x)) / 2.
    """
    number_system_value = _validate_number_system(number_system)
    if number_system_value == "real":
        x_value: Union[int, float, complex] = _validate_real_number("x", x)
    else:
        x_value = _validate_number("x", x)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    e_value = e()
    exponent_pos: Union[float, complex]
    exponent_neg: Union[float, complex]
    if number_system_value == "real":
        exponent_pos = float(x_value)
        exponent_neg = -float(x_value)
    else:
        exponent_pos = complex(x_value)
        exponent_neg = -complex(x_value)

    exp_pos = complex(
        power(e_value, exponent_pos, tol=tol_value, max_terms=max_terms_value, number_system=number_system_value)
    )
    exp_neg = complex(
        power(e_value, exponent_neg, tol=tol_value, max_terms=max_terms_value, number_system=number_system_value)
    )
    result = (exp_pos - exp_neg) / 2.0

    normalized = _normalize(result, tol_value)
    if number_system_value == "real":
        if isinstance(normalized, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized)
    return normalized
