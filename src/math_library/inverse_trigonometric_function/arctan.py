from typing import Optional, Union

from ..constant.e import e
from ..exponential_function.power import (
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)
from ..logarithmic_function import log


def arctan(
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute arctan(z).

    real mode: real input, real output
    complex mode: principal complex branch
    """
    number_system_value = _validate_number_system(number_system)
    if number_system_value == "real":
        x_value: Union[int, float, complex] = _validate_real_number("x", x)
    else:
        x_value = _validate_number("x", x)

    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    z = complex(x_value)
    e_value = e()
    log_minus = complex(
        log(e_value, 1.0 - 1j * z, tol=tol_value, max_terms=max_terms_value, number_system="complex")
    )
    log_plus = complex(
        log(e_value, 1.0 + 1j * z, tol=tol_value, max_terms=max_terms_value, number_system="complex")
    )
    result = 0.5j * (log_minus - log_plus)
    normalized = _normalize(result, tol_value)

    if number_system_value == "real":
        if isinstance(normalized, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized)

    return normalized
