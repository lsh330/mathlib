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
from ..logarithmic_function import log


def arcsin(
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute principal arcsin(z):
        arcsin(z) = -i * Log(i*z + sqrt(1 - z^2))
    """
    number_system_value = _validate_number_system(number_system)
    if number_system_value == "real":
        x_value: Union[int, float, complex] = _validate_real_number("x", x)
        if x_value < -1.0 or x_value > 1.0:
            raise ValueError("x must be in [-1, 1] in real mode.")
    else:
        x_value = _validate_number("x", x)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    z = complex(x_value)
    root_value = complex(
        power(1.0 - z * z, 0.5, tol=tol_value, max_terms=max_terms_value, number_system="complex")
    )
    inside = 1j * z + root_value
    ln_inside = complex(log(e(), inside, tol=tol_value, max_terms=max_terms_value, number_system="complex"))
    result = -1j * ln_inside
    normalized = _normalize(result, tol_value)

    if number_system_value == "real":
        if isinstance(normalized, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized)

    return normalized
