# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# inverse_trig.pyx
#
# arcsin, arccos, arctan, arcsec, arccosec, arccotan
# 참조: algorithm_reference.md 섹션 4~6

from libc.math cimport fma
from libc.stdint cimport uint32_t, uint64_t
from ._helpers cimport high_word, low_word, double_to_bits, bits_to_double
from .power_sqrt cimport _sqrt_c

import cmath as _cmath

# ------------------------------------------------------------------ asin/acos 공유 상수
# musl asin.c/acos.c, algorithm_reference.md 섹션 4~5
cdef double ASIN_PIO2_HI = 1.57079632679489655800e+00  # 0x3FF921FB54442D18
cdef double ASIN_PIO2_LO = 6.12323399573676603587e-17  # 0x3C91A62633145C07
cdef double ASIN_PI_HI   = 3.14159265358979311600e+00
cdef double ASIN_PI_LO   = 1.22464679914735317199e-16

# pS 계수 (분자 P(z))
cdef double pS0 =  1.66666666666666657415e-01
cdef double pS1 = -3.25565818622400915405e-01
cdef double pS2 =  2.01212532134862925881e-01
cdef double pS3 = -4.00555345006794114027e-02
cdef double pS4 =  7.91534994289814532176e-04
cdef double pS5 =  3.47933107596021167570e-05

# qS 계수 (분모 Q(z), 선행계수 1 생략)
cdef double qS1 = -2.40339491173441421878e+00
cdef double qS2 =  2.02094576023350569471e+00
cdef double qS3 = -6.88283971605453293030e-01
cdef double qS4 =  7.70381505559019352791e-02

# ------------------------------------------------------------------ atan 상수
# musl atan.c, algorithm_reference.md 섹션 6
cdef double ATANHI0 = 4.63647609000806093515e-01  # atan(0.5)
cdef double ATANHI1 = 7.85398163397448278999e-01  # atan(1) = π/4
cdef double ATANHI2 = 9.82793723247329054082e-01  # atan(1.5)
cdef double ATANHI3 = 1.57079632679489655800e+00  # atan(∞) = π/2

cdef double ATANLO0 = 2.26987774529616870924e-17
cdef double ATANLO1 = 3.06161699786838301793e-17
cdef double ATANLO2 = 1.39033110312309984516e-17
cdef double ATANLO3 = 6.12323399573676603587e-17

# aT 계수 (11개)
cdef double AT0  =  3.33333333333329318027e-01
cdef double AT1  = -1.99999999998764832476e-01
cdef double AT2  =  1.42857142725034663711e-01
cdef double AT3  = -1.11111104054623557880e-01
cdef double AT4  =  9.09088713343650656196e-02
cdef double AT5  = -7.69187620504482999495e-02
cdef double AT6  =  6.66107313738753120669e-02
cdef double AT7  = -5.83357013379057348645e-02
cdef double AT8  =  4.97687799461593236017e-02
cdef double AT9  = -3.65315727442169155270e-02
cdef double AT10 =  1.62858201153657823623e-02


# ------------------------------------------------------------------ 내부 함수
cdef inline double _asin_R(double z) noexcept nogil:
    """
    유리 근사 R(z) = P(z)/Q(z), z = x² ∈ [0, 0.25]
    musl asin.c rational approximation
    """
    cdef double p, q
    p = z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5))))
    p += pS0
    q = 1.0 + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)))
    return p / q


# ------------------------------------------------------------------ arcsin
cpdef double arcsin(double x) noexcept:
    """
    arcsin(x), asin(x).
    ULP: <= 1
    special: asin(+-1)=+-pi/2, asin(|x|>1)=NaN, asin(+-0)=+-0
    """
    cdef uint32_t hx, ix
    cdef double z, r, s, t, w, p, fx, df, c
    cdef uint64_t sbits
    cdef double result

    hx = high_word(x)
    ix = hx & 0x7FFFFFFFU

    # |x| >= 1
    if ix >= 0x3FF00000U:
        if ix == 0x3FF00000U and low_word(x) == 0U:
            # |x| == 1: +-π/2
            if hx >> 31:
                return -(ASIN_PIO2_HI + ASIN_PIO2_LO)
            return ASIN_PIO2_HI + ASIN_PIO2_LO
        return (x - x) / (x - x)  # NaN

    # |x| < 0.5
    if ix < 0x3FE00000U:
        if ix < 0x3E500000U:  # |x| < 2^-26
            return x
        z = x * x
        r = _asin_R(z)
        return x + x * z * r

    # 0.5 <= |x| < 1: asin(x) = pi/2 - 2*asin(sqrt((1-|x|)/2))
    if hx >> 31:
        fx = -x
    else:
        fx = x
    w = 1.0 - fx
    t = w * 0.5
    p = t * _asin_R(t)
    s = _sqrt_c(t)
    # 정밀 sqrt: s의 상위 32비트만
    sbits = double_to_bits(s) & 0xFFFFFFFF00000000ULL
    df = bits_to_double(sbits)
    c = (t - df * df) / (s + df)
    w = p * s + c
    result = 2.0 * df + 2.0 * w
    result = ASIN_PIO2_HI - (result - ASIN_PIO2_LO)

    if hx >> 31:
        return -result
    return result


# ------------------------------------------------------------------ arccos
cpdef double arccos(double x) noexcept:
    """
    arccos(x), acos(x).
    ULP: <= 1
    special: acos(1)=0, acos(-1)=pi, acos(|x|>1)=NaN
    """
    cdef uint32_t hx, ix
    cdef double z, s, w, r, p, c, df
    cdef uint64_t sbits

    hx = high_word(x)
    ix = hx & 0x7FFFFFFFU

    # |x| >= 1
    if ix >= 0x3FF00000U:
        if ix == 0x3FF00000U and low_word(x) == 0U:
            if hx >> 31:
                return 2.0 * ASIN_PIO2_HI + 2.0 * ASIN_PIO2_LO  # acos(-1) = π
            return 0.0  # acos(1) = 0
        return (x - x) / (x - x)  # NaN

    # |x| < 0.5
    if ix < 0x3FE00000U:
        if ix <= 0x3C600000U:  # |x| < 2^-57
            return ASIN_PIO2_HI + ASIN_PIO2_LO
        z = x * x
        p = z * _asin_R(z)
        return ASIN_PIO2_HI - (x - (ASIN_PIO2_LO - x * p))

    # x <= -0.5
    if hx >> 31:
        z = (1.0 + x) * 0.5
        p = z * _asin_R(z)
        s = _sqrt_c(z)
        w = p * s - ASIN_PIO2_LO
        return 2.0 * (ASIN_PIO2_HI - (s + w))

    # x >= 0.5
    z = (1.0 - x) * 0.5
    s = _sqrt_c(z)
    sbits = double_to_bits(s) & 0xFFFFFFFF00000000ULL
    df = bits_to_double(sbits)
    c = (z - df * df) / (s + df)
    p = z * _asin_R(z)
    w = p * s + c
    return 2.0 * (df + w)


# ------------------------------------------------------------------ arctan
cpdef double arctan(double x) noexcept:
    """
    arctan(x), atan(x).
    ULP: <= 1
    5구간 argument reduction + 11계수 다항식.
    special: atan(+-0)=+-0, atan(+-inf)=+-pi/2, atan(NaN)=NaN
    """
    cdef uint32_t hx, ix
    cdef double z, w, s1, s2, absx, result
    cdef int id

    hx = high_word(x)
    ix = hx & 0x7FFFFFFFU

    # |x| >= 2^66
    if ix >= 0x44100000U:
        if ix > 0x7FF00000U or (ix == 0x7FF00000U and low_word(x) != 0):
            return x + x  # NaN
        if hx >> 31:
            return -(ATANHI3 + ATANLO3)
        return ATANHI3 + ATANLO3

    # 5구간 분기
    if ix < 0x3FDC0000U:
        # |x| < 7/16
        if ix < 0x3E400000U:  # |x| < 2^-27
            return x
        id = -1
        # x는 원본 그대로 유지
        absx = x if not (hx >> 31) else -x
    else:
        absx = x if not (hx >> 31) else -x

        if ix < 0x3FF30000U:
            if ix < 0x3FE60000U:
                # 7/16 <= |x| < 11/16
                id = 0
                absx = (2.0 * absx - 1.0) / (2.0 + absx)
            else:
                # 11/16 <= |x| < 19/16
                id = 1
                absx = (absx - 1.0) / (absx + 1.0)
        else:
            if ix < 0x40038000U:
                # 19/16 <= |x| < 39/16
                id = 2
                absx = (absx - 1.5) / (1.0 + 1.5 * absx)
            else:
                # 39/16 <= |x|
                id = 3
                absx = -1.0 / absx
        x = absx

    # 다항식 평가 (z = x^2)
    z = x * x
    w = z * z
    s1 = z * (AT0 + w * (AT2 + w * (AT4 + w * (AT6 + w * (AT8 + w * AT10)))))
    s2 = w * (AT1 + w * (AT3 + w * (AT5 + w * (AT7 + w * AT9))))

    if id < 0:
        # x는 원본 값 그대로 (부호 포함) → 부호 반전 불필요
        return x - x * (s1 + s2)

    if id == 0:
        result = ATANHI0 - ((x * (s1 + s2) - ATANLO0) - x)
    elif id == 1:
        result = ATANHI1 - ((x * (s1 + s2) - ATANLO1) - x)
    elif id == 2:
        result = ATANHI2 - ((x * (s1 + s2) - ATANLO2) - x)
    else:
        result = ATANHI3 - ((x * (s1 + s2) - ATANLO3) - x)

    if hx >> 31:
        return -result
    return result


# ------------------------------------------------------------------ 파생 함수
cpdef double arcsec(double x) noexcept:
    """arcsec(x) = arccos(1/x)"""
    if x == 0.0:
        return (x - x) / (x - x)  # NaN
    return arccos(1.0 / x)


cpdef double arccosec(double x) noexcept:
    """arccosec(x) = arccsc(x) = arcsin(1/x)"""
    if x == 0.0:
        return (x - x) / (x - x)  # NaN
    return arcsin(1.0 / x)


cpdef double arccotan(double x) noexcept:
    """arccotan(x) = arccot(x) = π/2 - arctan(x)"""
    return (ASIN_PIO2_HI + ASIN_PIO2_LO) - arctan(x)


# ------------------------------------------------------------------ 복소수 auto-dispatch (방향 A)
cpdef object arcsin_dispatch(object x):
    """arcsin: 실수 → cpdef double arcsin, 복소수 → cmath.asin"""
    if type(x) is complex:
        return _cmath.asin(x)
    return arcsin(<double>x)

cpdef object arccos_dispatch(object x):
    """arccos: 실수 → cpdef double arccos, 복소수 → cmath.acos"""
    if type(x) is complex:
        return _cmath.acos(x)
    return arccos(<double>x)

cpdef object arctan_dispatch(object x):
    """arctan: 실수 → cpdef double arctan, 복소수 → cmath.atan"""
    if type(x) is complex:
        return _cmath.atan(x)
    return arctan(<double>x)

cpdef object arcsec_dispatch(object x):
    """arcsec: 실수 → cpdef double arcsec, 복소수 → cmath.acos(1/x)"""
    if type(x) is complex:
        return _cmath.acos(1.0 / x)
    return arcsec(<double>x)

cpdef object arccosec_dispatch(object x):
    """arccosec: 실수 → cpdef double arccosec, 복소수 → cmath.asin(1/x)"""
    if type(x) is complex:
        return _cmath.asin(1.0 / x)
    return arccosec(<double>x)

cpdef object arccotan_dispatch(object x):
    """arccotan: 실수 → cpdef double arccotan, 복소수 → cmath.atan(1/x)"""
    if type(x) is complex:
        return _cmath.atan(1.0 / x)
    return arccotan(<double>x)
