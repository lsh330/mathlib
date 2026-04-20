# cython: language_level=3
# cython: cdivision=True
# cython: embedsignature=True
#
# lcm.pyx — 최소공배수 (기존 알고리즘 유지, .py -> .pyx 변환)

from ..gcd.gcd import gcd as _gcd_py


def lcm(a, b):
    """
    Compute the least common multiple of two integers.
    By convention, lcm(0, b) and lcm(a, 0) return 0.
    """
    if isinstance(a, bool) or not isinstance(a, int):
        raise TypeError("a must be an integer.")
    if isinstance(b, bool) or not isinstance(b, int):
        raise TypeError("b must be an integer.")

    a_val = int(a)
    b_val = int(b)

    if a_val == 0 or b_val == 0:
        return 0

    return abs((a_val // _gcd_py(a_val, b_val)) * b_val)
