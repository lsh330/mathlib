from ..gcd import gcd


def _validate_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def lcm(a: int, b: int) -> int:
    """
    Compute the least common multiple of two integers.

    By convention, lcm(0, b) and lcm(a, 0) return 0.
    """
    a_value = _validate_integer("a", a)
    b_value = _validate_integer("b", b)

    if a_value == 0 or b_value == 0:
        return 0

    return abs((a_value // gcd(a_value, b_value)) * b_value)
