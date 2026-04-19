def _validate_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def gcd(a: int, b: int) -> int:
    """
    Compute the greatest common divisor of two integers.

    By convention, gcd(0, 0) returns 0.
    """
    a_value = _validate_integer("a", a)
    b_value = _validate_integer("b", b)

    x = abs(a_value)
    y = abs(b_value)

    while y != 0:
        x, y = y, x % y

    return x
