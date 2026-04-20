from typing import Optional, Union

from ..exponential_function.power import (
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)
from .._core.hyperbolic import hypertan as _hypertan_core
from .hypercos import hypercos
from .hypersin import hypersin


def hypertan(
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute tanh(x).
    real mode: Cython _core 구현 (musl fdlibm, 2 ULP 정밀도)
    complex mode: sinh(z) / cosh(z)
    """
    # fast-path
    if type(x) is float and number_system == "real":
        return _hypertan_core(x)

    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    if number_system_value == "real":
        x_value = _validate_real_number("x", x)
        return _hypertan_core(x_value)

    # complex mode
    _validate_number("x", x)
    sinh_value = complex(hypersin(x, tol=tol_value, max_terms=max_terms_value, number_system="complex"))
    cosh_value = complex(hypercos(x, tol=tol_value, max_terms=max_terms_value, number_system="complex"))

    if abs(cosh_value) <= tol_value:
        raise ZeroDivisionError("cosh(x) is zero, tanh(x) is undefined.")

    result = sinh_value / cosh_value
    return _normalize(result, tol_value)
