from typing import Optional, Union

from ..exponential_function.power import (
    _complex_log,
    _normalize,
    _validate_max_terms,
    _validate_number,
    _validate_number_system,
    _validate_real_number,
    _validate_tol,
)


def log(
    base: Union[int, float, complex],
    x: Union[int, float, complex],
    tol: Optional[float] = None,
    max_terms: int = 100,
    number_system: str = "real",
) -> Union[float, complex]:
    """
    Compute log_base(x) on the principal complex branch:
        log_base(x) = Log(x) / Log(base)
    """
    number_system_value = _validate_number_system(number_system)
    tol_value = _validate_tol(tol)
    max_terms_value = _validate_max_terms(max_terms)

    if number_system_value == "real":
        base_real = _validate_real_number("base", base)
        x_real = _validate_real_number("x", x)

        if x_real <= 0.0:
            raise ValueError("x must be positive in real mode.")
        if base_real <= 0.0:
            raise ValueError("base must be positive in real mode.")
        if base_real == 1.0:
            raise ValueError("base must not be 1.")

        log_base_real = _complex_log(complex(base_real, 0.0), tol_value, max_terms_value)
        log_x_real = _complex_log(complex(x_real, 0.0), tol_value, max_terms_value)
        result_real = log_x_real / log_base_real
        normalized_real = _normalize(result_real, tol_value)
        if isinstance(normalized_real, complex):
            raise RuntimeError("real-mode result unexpectedly became complex.")
        return float(normalized_real)

    base_value = _validate_number("base", base)
    x_value = _validate_number("x", x)
    base_complex = complex(base_value)
    x_complex = complex(x_value)

    if x_complex == 0:
        raise ValueError("x must be non-zero.")
    if base_complex == 0:
        raise ValueError("base must be non-zero.")
    if base_complex == 1:
        raise ValueError("base must not be 1.")

    log_base = _complex_log(base_complex, tol_value, max_terms_value)
    if abs(log_base) <= tol_value:
        raise ValueError("base has zero logarithm on the principal branch.")
    result = _complex_log(x_complex, tol_value, max_terms_value) / log_base
    return _normalize(result, tol_value)

