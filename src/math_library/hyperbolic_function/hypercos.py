from typing import Optional, Union

from ..exponential_function.power import (
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)
from .._core.hyperbolic import hypercos as _hypercos_core


def hypercos(
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute cosh(x).
    real mode: Cython _core 구현 (musl fdlibm, 2 ULP 정밀도)
    complex mode: (e^z + e^-z) / 2
    """
    # fast-path
    if type(x) is float and number_system == "real":
        return _hypercos_core(x)

    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    if number_system_value == "real":
        x_value = _validate_real_number("x", x)
        return _hypercos_core(x_value)

    # complex mode
    from ..constant.e import e
    from ..exponential_function import power

    x_value = _validate_number("x", x)
    e_value = e()
    x_complex = complex(x_value)
    exp_pos = complex(
        power(e_value, x_complex, tol=tol_value, max_terms=max_terms_value, number_system="complex")
    )
    exp_neg = complex(
        power(e_value, -x_complex, tol=tol_value, max_terms=max_terms_value, number_system="complex")
    )
    result = (exp_pos + exp_neg) / 2.0
    return _normalize(result, tol_value)
