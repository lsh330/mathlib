from typing import Optional, Union

from ..exponential_function.power import (
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)
from .._core.inverse_trig import arcsin as _arcsin_core


def arcsin(
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute principal arcsin(z).
    real mode: Cython _core 구현 (fdlibm, 1 ULP 정밀도)
    complex mode: arcsin(z) = -i * Log(i*z + sqrt(1 - z^2))
    """
    # fast-path
    if type(x) is float and number_system == "real":
        if x < -1.0 or x > 1.0:
            raise ValueError("x must be in [-1, 1] in real mode.")
        return _arcsin_core(x)

    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    if number_system_value == "real":
        x_value = _validate_real_number("x", x)
        if x_value < -1.0 or x_value > 1.0:
            raise ValueError("x must be in [-1, 1] in real mode.")
        return _arcsin_core(x_value)

    # complex mode
    from ..constant.e import e
    from ..exponential_function import power
    from ..logarithmic_function import log

    x_value = _validate_number("x", x)
    z = complex(x_value)
    root_value = complex(
        power(1.0 - z * z, 0.5, tol=tol_value, max_terms=max_terms_value, number_system="complex")
    )
    inside = 1j * z + root_value
    ln_inside = complex(log(e(), inside, tol=tol_value, max_terms=max_terms_value, number_system="complex"))
    result = -1j * ln_inside
    return _normalize(result, tol_value)
