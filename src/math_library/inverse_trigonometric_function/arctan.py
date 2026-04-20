from typing import Optional, Union

from ..exponential_function.power import (
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)
from .._core.inverse_trig import arctan as _arctan_core


def arctan(
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute arctan(z).
    real mode: Cython _core 구현 (fdlibm, 1 ULP 정밀도)
    complex mode: 0.5j * (log(1-iz) - log(1+iz))
    """
    # fast-path
    if type(x) is float and number_system == "real":
        return _arctan_core(x)

    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    if number_system_value == "real":
        x_value = _validate_real_number("x", x)
        return _arctan_core(x_value)

    # complex mode
    from ..constant.e import e
    from ..logarithmic_function import log

    x_value = _validate_number("x", x)
    z = complex(x_value)
    e_value = e()
    log_minus = complex(
        log(e_value, 1.0 - 1j * z, tol=tol_value, max_terms=max_terms_value, number_system="complex")
    )
    log_plus = complex(
        log(e_value, 1.0 + 1j * z, tol=tol_value, max_terms=max_terms_value, number_system="complex")
    )
    result = 0.5j * (log_minus - log_plus)
    return _normalize(result, tol_value)
