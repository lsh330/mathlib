# cython: language_level=3
# _helpers.pxd — cdef inline 유틸리티: DoubleUnion, poly_n, FMA chain, bit ops
# 다른 .pyx 모듈에서 cimport로 C 수준 인라인 호출 가능

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.math cimport fma

# IEEE 754 double <-> uint64 union (C 수준 bit manipulation)
cdef extern from *:
    """
    typedef union {
        double d;
        unsigned long long u;
    } DoubleUnion;
    """
    ctypedef struct DoubleUnion:
        double d
        unsigned long long u


# ------------------------------------------------------------------ bit ops
cdef inline uint64_t double_to_bits(double x) noexcept nogil:
    cdef DoubleUnion du
    du.d = x
    return du.u


cdef inline double bits_to_double(uint64_t bits) noexcept nogil:
    cdef DoubleUnion du
    du.u = bits
    return du.d


cdef inline uint32_t high_word(double x) noexcept nogil:
    """상위 32비트 (부호 + 지수 + 가수 상위 20비트)"""
    return <uint32_t>(double_to_bits(x) >> 32)


cdef inline uint32_t low_word(double x) noexcept nogil:
    """하위 32비트 (가수 하위 32비트)"""
    return <uint32_t>(double_to_bits(x) & 0xFFFFFFFFULL)


# ------------------------------------------------------------------ Horner + FMA
cdef inline double poly2(double x,
                          double c0, double c1, double c2) noexcept nogil:
    return fma(fma(c2, x, c1), x, c0)


cdef inline double poly3(double x,
                          double c0, double c1, double c2,
                          double c3) noexcept nogil:
    return fma(fma(fma(c3, x, c2), x, c1), x, c0)


cdef inline double poly4(double x,
                          double c0, double c1, double c2,
                          double c3, double c4) noexcept nogil:
    return fma(fma(fma(fma(c4, x, c3), x, c2), x, c1), x, c0)


cdef inline double poly5(double x,
                          double c0, double c1, double c2,
                          double c3, double c4, double c5) noexcept nogil:
    return fma(fma(fma(fma(fma(c5, x, c4), x, c3), x, c2), x, c1), x, c0)


cdef inline double poly6(double x,
                          double c0, double c1, double c2,
                          double c3, double c4, double c5,
                          double c6) noexcept nogil:
    return fma(fma(fma(fma(fma(fma(c6, x, c5), x, c4), x, c3), x, c2), x, c1), x, c0)
