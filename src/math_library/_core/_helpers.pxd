# cython: language_level=3
# _helpers.pxd — cdef inline 유틸리티: DoubleUnion, poly_n, FMA chain, bit ops
# 다른 .pyx 모듈에서 cimport로 C 수준 인라인 호출 가능

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.math cimport fma, sqrt as _libc_sqrt_h

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


# nogil 안전한 복소수 생성 — Cython-native 구현
# (이전 GCC 확장 __complex__/__real__/__imag__ 사용을 제거, MSVC/GCC/Clang 공통 호환)
cdef inline double complex _make_complex(double re, double im) noexcept nogil:
    return <double complex>re + <double complex>im * 1j


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


# ------------------------------------------------------------------ 복소수 헬퍼
# _atan_inner: musl atan.c 5구간 argument reduction + 11계수 Horner
# _atan2_real: 사분면 판별 포함 2인수 arctan (실수 전용)
cdef inline double _atan_inner(double x) noexcept nogil:
    """
    musl atan.c 포팅: 5구간 arg reduction + 11계수 Horner.
    algorithm_reference.md 섹션 6 계수 사용.
    """
    cdef double ATANHI0 = 4.63647609000806093515e-01
    cdef double ATANHI1 = 7.85398163397448278999e-01
    cdef double ATANHI2 = 9.82793723247329054082e-01
    cdef double ATANHI3 = 1.57079632679489655800e+00
    cdef double ATANLO0 = 2.26987774529616870924e-17
    cdef double ATANLO1 = 3.06161699786838301793e-17
    cdef double ATANLO2 = 1.39033110312309984516e-17
    cdef double ATANLO3 = 6.12323399573676603587e-17

    cdef double absx, z, w, s1, s2, result
    cdef int id, neg
    cdef uint32_t hx, ix

    hx = <uint32_t>((<uint64_t>0 | double_to_bits(x)) >> 32)
    ix = hx & 0x7FFFFFFFU
    neg = 1 if (hx >> 31) else 0
    absx = -x if neg else x

    # |x| >= 2^66 → ±π/2
    if ix >= 0x44100000U:
        if ix > 0x7FF00000U or (ix == 0x7FF00000U):
            return x + x  # NaN → NaN, inf → signed inf (near π/2)
        if neg:
            return -(ATANHI3 + ATANLO3)
        return ATANHI3 + ATANLO3

    if ix < 0x3FDC0000U:
        # |x| < 7/16
        if ix < 0x3E400000U:
            return x  # |x| < 2^-27
        id = -1
    else:
        if ix < 0x3FF30000U:
            if ix < 0x3FE60000U:
                id = 0
                absx = (2.0 * absx - 1.0) / (2.0 + absx)
            else:
                id = 1
                absx = (absx - 1.0) / (absx + 1.0)
        else:
            if ix < 0x40038000U:
                id = 2
                absx = (absx - 1.5) / (1.0 + 1.5 * absx)
            else:
                id = 3
                absx = -1.0 / absx

    z  = absx * absx
    w  = z * z
    s1 = z * (3.33333333333329318027e-01 + w * (1.42857142725034663711e-01
           + w * (9.09088713343650656196e-02 + w * (6.66107313738753120669e-02
           + w * (4.97687799461593236017e-02 + w * 1.62858201153657823623e-02)))))
    s2 = w * (-1.99999999998764832476e-01 + w * (-1.11111104054623557880e-01
           + w * (-7.69187620504482999495e-02 + w * (-5.83357013379057348645e-02
           + w * (-3.65315727442169155270e-02)))))

    if id < 0:
        result = absx - absx * (s1 + s2)
    elif id == 0:
        result = ATANHI0 - ((absx*(s1+s2) - ATANLO0) - absx)
    elif id == 1:
        result = ATANHI1 - ((absx*(s1+s2) - ATANLO1) - absx)
    elif id == 2:
        result = ATANHI2 - ((absx*(s1+s2) - ATANLO2) - absx)
    else:
        result = ATANHI3 - ((absx*(s1+s2) - ATANLO3) - absx)

    return -result if neg else result


cdef inline double _atan2_real(double y, double x) noexcept nogil:
    """
    atan2(y, x): 결과 범위 (-π, π].
    _atan_inner (musl 5구간 full arg reduction) 사용.
    """
    cdef double PI   = 3.14159265358979323846e+00
    cdef double PIO2 = 1.57079632679489655800e+00
    cdef double r

    if x == 0.0 and y == 0.0:
        return 0.0
    if y == 0.0:
        if x > 0.0:
            return 0.0
        return PI
    if x == 0.0:
        if y > 0.0:
            return PIO2
        return -PIO2

    r = _atan_inner(y / x)

    if x > 0.0:
        return r
    if y >= 0.0:
        return r + PI
    return r - PI


# _hypot_real: overflow-safe hypot(x, y)
cdef inline double _hypot_real(double x, double y) noexcept nogil:
    """
    sqrt(x^2 + y^2) — overflow 방지.
    |x| >= |y| 정렬 후 |x|*sqrt(1+(y/x)^2).
    """
    cdef double ax, ay, t
    ax = -x if x < 0.0 else x
    ay = -y if y < 0.0 else y
    if ax == 0.0 and ay == 0.0:
        return 0.0
    if ax < ay:
        t = ax; ax = ay; ay = t
    t = ay / ax
    return ax * _libc_sqrt_h(1.0 + t * t)
