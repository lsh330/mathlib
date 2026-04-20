from typing import Optional, Union

from ..constant.pi import pi
from ..exponential_function.power import (
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)
from .._core.trigonometric import cos as _cos_core

_PI = pi()
_DEG2RAD = _PI / 180.0


def cos(
    x: Union[int, float, complex],
    unit: str = "rad",
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute cos(x).
    real mode: Cython _core 구현 (musl fdlibm, 1 ULP 정밀도)
    complex mode: Euler's identity cos(z) = (e^iz + e^-iz) / 2
    """
    # fast-path
    if type(x) is float and unit == "rad" and number_system == "real":
        return _cos_core(x)

    number_system_value = _validate_number_system(number_system)

    if not isinstance(unit, str):
        raise TypeError("unit must be a string.")
    unit_lower = unit.lower()
    if unit_lower not in ("rad", "deg"):
        raise ValueError("unit must be either 'rad' or 'deg'.")

    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    if number_system_value == "real":
        x_value = _validate_real_number("x", x)
        x_rad = x_value * _DEG2RAD if unit_lower == "deg" else x_value
        return _cos_core(x_rad)

    # complex mode
    from ..constant.e import e
    from ..exponential_function import power

    x_value = _validate_number("x", x)
    x_complex = complex(x_value)
    if unit_lower == "deg":
        x_complex = x_complex * _DEG2RAD

    e_value = e()
    exp_pos = complex(
        power(e_value, 1j * x_complex, tol=tol_value, max_terms=max_terms_value, number_system="complex")
    )
    exp_neg = complex(
        power(e_value, -1j * x_complex, tol=tol_value, max_terms=max_terms_value, number_system="complex")
    )
    result = (exp_pos + exp_neg) / 2.0
    return _normalize(result, tol_value)
