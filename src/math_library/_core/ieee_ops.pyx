# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# ieee_ops.pyx
#
# IEEE 754 비트 조작 primitive — libc.math 허용 헬퍼 래핑
# ceil, floor, trunc, fmod, copysign, remainder, modf, nextafter, ulp

from libc.math cimport (
    ceil     as _lceil,
    floor    as _lfloor,
    trunc    as _ltrunc,
    fmod     as _lfmod,
    copysign as _lcopysign,
    remainder as _lremainder,
    nextafter as _lnextafter,
    INFINITY,
)
from ._helpers cimport double_to_bits, bits_to_double
from libc.stdint cimport uint64_t


cpdef double ceil(double x) noexcept:
    """ceil(x): x 이상의 최소 정수 (float 반환)."""
    return _lceil(x)


cpdef double floor(double x) noexcept:
    """floor(x): x 이하의 최대 정수 (float 반환)."""
    return _lfloor(x)


cpdef double trunc(double x) noexcept:
    """trunc(x): 0 방향으로 내림 (float 반환)."""
    return _ltrunc(x)


cpdef double fmod(double x, double y) noexcept:
    """fmod(x, y): C 방식 나머지 (부호는 x와 동일)."""
    return _lfmod(x, y)


cpdef double copysign(double x, double y) noexcept:
    """copysign(x, y): |x|에 y의 부호를 붙인 값."""
    return _lcopysign(x, y)


cpdef double remainder(double x, double y) noexcept:
    """remainder(x, y): IEEE 754 나머지 (가장 가까운 짝수 반올림 기반)."""
    return _lremainder(x, y)


cpdef object modf(double x):
    """
    modf(x): (소수부, 정수부) 튜플 반환.
    부호는 양쪽 모두 x와 동일.
    """
    cdef double int_part
    int_part = _ltrunc(x)
    cdef double frac = x - int_part
    return (frac, int_part)


cpdef double nextafter(double x, double y) noexcept:
    """nextafter(x, y): x에서 y 방향으로 다음 float."""
    return _lnextafter(x, y)


cpdef double ulp(double x) noexcept:
    """
    ulp(x): x에서의 ULP (unit in the last place).
    ulp(x) = nextafter(|x|, inf) - |x|.
    특수값: ulp(inf) = inf, ulp(nan) = nan.
    """
    cdef double ax = x if x >= 0.0 else -x
    cdef uint64_t bits
    cdef double nx

    if ax != ax:   # NaN
        return x
    if ax == INFINITY:
        return INFINITY

    nx = _lnextafter(ax, INFINITY)
    return nx - ax
