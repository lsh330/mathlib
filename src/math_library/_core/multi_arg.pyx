# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# multi_arg.pyx
#
# atan2 (실수 전용), hypot (실수 + 복소수), dist (n차원)

from libc.math cimport fabs
from ._helpers cimport _atan2_real, _hypot_real, _make_complex
from .power_sqrt cimport _sqrt_c


# ------------------------------------------------------------------ atan2

cpdef double atan2(double y, double x) noexcept:
    """
    atan2(y, x): 결과 범위 (-π, π].
    IEEE 754-2019 특수값:
      atan2(±0, x>0) = ±0
      atan2(±0, x<0) = ±π
      atan2(y>0, ±0) = +π/2
      atan2(±∞, +∞)  = ±π/4
      atan2(±∞, -∞)  = ±3π/4
    """
    return _atan2_real(y, x)


# ------------------------------------------------------------------ hypot

cdef double _hypot_complex_vals(double complex a, double complex b) noexcept nogil:
    """hypot(z1, z2) = sqrt(|z1|^2 + |z2|^2) — overflow safe"""
    cdef double r1 = a.real * a.real + a.imag * a.imag
    cdef double r2 = b.real * b.real + b.imag * b.imag
    return _sqrt_c(r1 + r2)


cpdef object hypot(object a, object b):
    """
    hypot(a, b):
      실수 a, b: sqrt(a^2 + b^2) overflow-safe
      복소수 a 또는 b: sqrt(|a|^2 + |b|^2)
    """
    cdef double complex ca, cb
    if type(a) is complex or type(b) is complex:
        ca = <double complex>a if type(a) is complex else _make_complex(<double>a, 0.0)
        cb = <double complex>b if type(b) is complex else _make_complex(<double>b, 0.0)
        return _hypot_complex_vals(ca, cb)
    return _hypot_real(<double>a, <double>b)


# ------------------------------------------------------------------ dist

cpdef double dist(object p, object q):
    """
    dist(p, q): n차원 유클리드 거리. p, q는 동일 길이 sequence.
    sqrt(sum((pi - qi)^2)) — overflow 방지 (|max_diff| 정규화).
    """
    cdef Py_ssize_t n = len(p)
    if n != len(q):
        raise ValueError("dist: p and q must have equal length")
    if n == 0:
        return 0.0

    # overflow-safe: max 절댓값으로 정규화
    cdef double max_d = 0.0, d, acc = 0.0
    cdef Py_ssize_t i
    cdef list diffs = [0.0] * n

    for i in range(n):
        d = <double>(p[i]) - <double>(q[i])
        if d < 0.0:
            d = -d
        diffs[i] = d
        if d > max_d:
            max_d = d

    if max_d == 0.0:
        return 0.0

    for i in range(n):
        d = diffs[i] / max_d
        acc += d * d

    return max_d * _sqrt_c(acc)
