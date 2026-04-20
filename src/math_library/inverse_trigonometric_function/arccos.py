from typing import Optional, Union

from ..exponential_function.power import (
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)
from .._core.inverse_trig import arccos as _arccos_core
from .arcsin import arcsin

from ..constant.pi import pi
_HALF_PI = 0.5 * pi()


def arccos(
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute principal arccos(z) = pi/2 - arcsin(z).
    real mode: Cython _core 구현 (fdlibm, 1 ULP 정밀도)
    complex mode: pi/2 - arcsin(z)
    """
    # fast-path
    if type(x) is float and number_system == "real":
        if x < -1.0 or x > 1.0:
            raise ValueError("x must be in [-1, 1] in real mode.")
        return _arccos_core(x)

    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    if number_system_value == "real":
        x_real = _validate_real_number("x", x)
        if x_real < -1.0 or x_real > 1.0:
            raise ValueError("x must be in [-1, 1] in real mode.")
        return _arccos_core(x_real)

    # complex mode
    _validate_number("x", x)
    result = complex(_HALF_PI, 0.0) - complex(
        arcsin(x, tol=tol_value, max_terms=max_terms_value, number_system=number_system_value)
    )
    return _normalize(result, tol_value)
