# cython: language_level=3
# cython: cdivision=True
# cython: embedsignature=True
#
# gcd.pyx — 최대공약수 (기존 알고리즘 유지, .py -> .pyx 변환)


cdef int _gcd_core(int a, int b) noexcept nogil:
    """내부 GCD 계산 (타입 검사 없이)"""
    cdef int x, y, tmp
    x = a if a >= 0 else -a
    y = b if b >= 0 else -b
    while y != 0:
        tmp = y
        y = x % y
        x = tmp
    return x


def gcd(a, b):
    """
    Compute the greatest common divisor of two integers.
    By convention, gcd(0, 0) returns 0.
    """
    if isinstance(a, bool) or not isinstance(a, int):
        raise TypeError("a must be an integer.")
    if isinstance(b, bool) or not isinstance(b, int):
        raise TypeError("b must be an integer.")
    return _gcd_core(<int>a, <int>b)
