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
# 복소수 자체 구현 (cmath 사용 금지)
# 참조: algorithm_reference.md 섹션 4~6

from libc.math cimport fma, INFINITY
from libc.stdint cimport uint32_t, uint64_t
from ._helpers cimport high_word, low_word, double_to_bits, bits_to_double
from .power_sqrt cimport _sqrt_c, _sqrt_complex
from .logarithmic cimport _ln_complex
from .exponential cimport _exp_complex
from ._helpers cimport _make_complex

# ------------------------------------------------------------------ asin/acos 공유 상수
cdef double ASIN_PIO2_HI = 1.57079632679489655800e+00
cdef double ASIN_PIO2_LO = 6.12323399573676603587e-17
cdef double ASIN_PI_HI   = 3.14159265358979311600e+00
cdef double ASIN_PI_LO   = 1.22464679914735317199e-16

# pS 계수
cdef double pS0 =  1.66666666666666657415e-01
cdef double pS1 = -3.25565818622400915405e-01
cdef double pS2 =  2.01212532134862925881e-01
cdef double pS3 = -4.00555345006794114027e-02
cdef double pS4 =  7.91534994289814532176e-04
cdef double pS5 =  3.47933107596021167570e-05

# qS 계수
cdef double qS1 = -2.40339491173441421878e+00
cdef double qS2 =  2.02094576023350569471e+00
cdef double qS3 = -6.88283971605453293030e-01
cdef double qS4 =  7.70381505559019352791e-02

# ------------------------------------------------------------------ atan 상수
cdef double ATANHI0 = 4.63647609000806093515e-01
cdef double ATANHI1 = 7.85398163397448278999e-01
cdef double ATANHI2 = 9.82793723247329054082e-01
cdef double ATANHI3 = 1.57079632679489655800e+00

cdef double ATANLO0 = 2.26987774529616870924e-17
cdef double ATANLO1 = 3.06161699786838301793e-17
cdef double ATANLO2 = 1.39033110312309984516e-17
cdef double ATANLO3 = 6.12323399573676603587e-17

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
    cdef double p, q
    p = z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5))))
    p += pS0
    q = 1.0 + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)))
    return p / q


# ------------------------------------------------------------------ arcsin (실수)
cpdef double arcsin(double x) noexcept:
    """arcsin(x). ULP<=1. special: +-1=+-pi/2, |x|>1=NaN"""
    cdef uint32_t hx, ix
    cdef double z, r, s, t, w, p, fx, df, c
    cdef uint64_t sbits
    cdef double result

    hx = high_word(x)
    ix = hx & 0x7FFFFFFFU

    if ix >= 0x3FF00000U:
        if ix == 0x3FF00000U and low_word(x) == 0U:
            if hx >> 31:
                return -(ASIN_PIO2_HI + ASIN_PIO2_LO)
            return ASIN_PIO2_HI + ASIN_PIO2_LO
        return (x - x) / (x - x)

    if ix < 0x3FE00000U:
        if ix < 0x3E500000U:
            return x
        z = x * x
        r = _asin_R(z)
        return x + x * z * r

    if hx >> 31:
        fx = -x
    else:
        fx = x
    w = 1.0 - fx
    t = w * 0.5
    p = t * _asin_R(t)
    s = _sqrt_c(t)
    sbits = double_to_bits(s) & 0xFFFFFFFF00000000ULL
    df = bits_to_double(sbits)
    c = (t - df * df) / (s + df)
    w = p * s + c
    result = 2.0 * df + 2.0 * w
    result = ASIN_PIO2_HI - (result - ASIN_PIO2_LO)

    if hx >> 31:
        return -result
    return result


# ------------------------------------------------------------------ arccos (실수)
cpdef double arccos(double x) noexcept:
    """arccos(x). ULP<=1. special: acos(1)=0, acos(-1)=pi, |x|>1=NaN"""
    cdef uint32_t hx, ix
    cdef double z, s, w, r, p, c, df
    cdef uint64_t sbits

    hx = high_word(x)
    ix = hx & 0x7FFFFFFFU

    if ix >= 0x3FF00000U:
        if ix == 0x3FF00000U and low_word(x) == 0U:
            if hx >> 31:
                return 2.0 * ASIN_PIO2_HI + 2.0 * ASIN_PIO2_LO
            return 0.0
        return (x - x) / (x - x)

    if ix < 0x3FE00000U:
        if ix <= 0x3C600000U:
            return ASIN_PIO2_HI + ASIN_PIO2_LO
        z = x * x
        p = z * _asin_R(z)
        return ASIN_PIO2_HI - (x - (ASIN_PIO2_LO - x * p))

    if hx >> 31:
        z = (1.0 + x) * 0.5
        p = z * _asin_R(z)
        s = _sqrt_c(z)
        w = p * s - ASIN_PIO2_LO
        return 2.0 * (ASIN_PIO2_HI - (s + w))

    z = (1.0 - x) * 0.5
    s = _sqrt_c(z)
    sbits = double_to_bits(s) & 0xFFFFFFFF00000000ULL
    df = bits_to_double(sbits)
    c = (z - df * df) / (s + df)
    p = z * _asin_R(z)
    w = p * s + c
    return 2.0 * (df + w)


# ------------------------------------------------------------------ arctan (실수)
cpdef double arctan(double x) noexcept:
    """arctan(x). ULP<=1. 5구간 arg reduction + 11계수 poly."""
    cdef uint32_t hx, ix
    cdef double z, w, s1, s2, absx, result
    cdef int id

    hx = high_word(x)
    ix = hx & 0x7FFFFFFFU

    if ix >= 0x44100000U:
        if ix > 0x7FF00000U or (ix == 0x7FF00000U and low_word(x) != 0):
            return x + x
        if hx >> 31:
            return -(ATANHI3 + ATANLO3)
        return ATANHI3 + ATANLO3

    if ix < 0x3FDC0000U:
        if ix < 0x3E400000U:
            return x
        id = -1
        absx = x if not (hx >> 31) else -x
    else:
        absx = x if not (hx >> 31) else -x

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
        x = absx

    z = x * x
    w = z * z
    s1 = z * (AT0 + w * (AT2 + w * (AT4 + w * (AT6 + w * (AT8 + w * AT10)))))
    s2 = w * (AT1 + w * (AT3 + w * (AT5 + w * (AT7 + w * AT9))))

    if id < 0:
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


# ------------------------------------------------------------------ 파생 함수 (실수)
cpdef double arcsec(double x) noexcept:
    """arcsec(x) = arccos(1/x)"""
    if x == 0.0:
        return (x - x) / (x - x)
    return arccos(1.0 / x)


cpdef double arccosec(double x) noexcept:
    """arccosec(x) = arcsin(1/x)"""
    if x == 0.0:
        return (x - x) / (x - x)
    return arcsin(1.0 / x)


cpdef double arccotan(double x) noexcept:
    """arccotan(x) = π/2 - arctan(x)"""
    return (ASIN_PIO2_HI + ASIN_PIO2_LO) - arctan(x)


# ------------------------------------------------------------------ 복소수 커널 (자체 구현)
# 상수
cdef double _PI    = 3.14159265358979323846e+00
cdef double _PIO2  = 1.5707963267948966192313e+00


cdef double complex _arcsin_complex(double complex z) noexcept nogil:
    """
    arcsin(z) = -i * ln(i*z + sqrt(1 - z^2))
    = -i * ln( iz + sqrt(1-z^2) )
    """
    # 1 - z^2
    cdef double re = z.real, im = z.imag
    cdef double one_minus_z2_re = 1.0 - (re*re - im*im)
    cdef double one_minus_z2_im = -(2.0 * re * im)
    cdef double complex one_minus_z2 = _make_complex(one_minus_z2_re, one_minus_z2_im)
    cdef double complex sq = _sqrt_complex(one_minus_z2)
    cdef double complex iz = _make_complex(-im, re)
    cdef double complex w  = _make_complex(iz.real + sq.real, iz.imag + sq.imag)
    cdef double complex lw = _ln_complex(w)
    return _make_complex(lw.imag, -lw.real)


cdef double complex _arccos_complex(double complex z) noexcept nogil:
    """arccos(z) = π/2 - arcsin(z)"""
    cdef double complex as_ = _arcsin_complex(z)
    return _make_complex(_PIO2 - as_.real, -as_.imag)


cdef double complex _arctan_complex(double complex z) noexcept nogil:
    """
    arctan(z) = -i/2 * ln((1 + iz) / (1 - iz))
    표준 공식 (위키피디아): arctan(z) = -i/2 * ln((1+iz)/(1-iz))
    등가식:  = (i/2) * (ln(1-iz) - ln(1+iz))

    계산 순서:
      iz = i*z = (-im, re)
      num = 1 + iz = (1 - im, re)
      den = 1 - iz = (1 + im, -re)
      ratio = num / den
      arctan = -i/2 * ln(ratio)  →  real = (1/2)*imag(ln),  imag = -(1/2)*real(ln)
    """
    cdef double re = z.real, im = z.imag
    # iz = i*(re+i*im) = -im + i*re
    cdef double num_re = 1.0 - im   # 1 + iz: real part  = 1 + (-im)
    cdef double num_im = re          # 1 + iz: imag part  = re
    cdef double den_re = 1.0 + im   # 1 - iz: real part  = 1 - (-im)
    cdef double den_im = -re         # 1 - iz: imag part  = -re

    # 특이점: den = 0  ↔  z = i
    cdef double denom2 = den_re * den_re + den_im * den_im
    if denom2 == 0.0:
        return _make_complex(0.0, INFINITY)

    # ratio = num / den  (복소수 나눗셈)
    cdef double ratio_re = (num_re * den_re + num_im * den_im) / denom2
    cdef double ratio_im = (num_im * den_re - num_re * den_im) / denom2

    cdef double complex lr = _ln_complex(_make_complex(ratio_re, ratio_im))
    # arctan = -i/2 * lr  →  real = lr.imag / 2,  imag = -lr.real / 2
    return _make_complex(0.5 * lr.imag, -0.5 * lr.real)


cdef double complex _arcsec_complex(double complex z) noexcept nogil:
    """arcsec(z) = arccos(1/z)"""
    cdef double denom = z.real*z.real + z.imag*z.imag
    if denom == 0.0:
        return _make_complex(INFINITY, 0.0)
    cdef double complex inv_z = _make_complex(z.real / denom, -z.imag / denom)
    return _arccos_complex(inv_z)


cdef double complex _arccosec_complex(double complex z) noexcept nogil:
    """arccosec(z) = arcsin(1/z)"""
    cdef double denom = z.real*z.real + z.imag*z.imag
    if denom == 0.0:
        return _make_complex(INFINITY, 0.0)
    cdef double complex inv_z = _make_complex(z.real / denom, -z.imag / denom)
    return _arcsin_complex(inv_z)


cdef double complex _arccotan_complex(double complex z) noexcept nogil:
    """arccotan(z) = π/2 - arctan(z)"""
    cdef double complex at = _arctan_complex(z)
    return _make_complex(_PIO2 - at.real, -at.imag)


# ------------------------------------------------------------------ 복소수 auto-dispatch + 자동 승격 — cmath 없이 자체 구현
cpdef object arcsin_dispatch(object x):
    """
    arcsin: 실수 |x|<=1 → cpdef double arcsin
            실수 |x|>1  → 자동 승격 _arcsin_complex
            복소수      → _arcsin_complex
    """
    if type(x) is complex:
        return _arcsin_complex(<double complex>x)
    cdef double xd = <double>x
    if xd > 1.0 or xd < -1.0:
        return _arcsin_complex(complex(xd, 0.0))
    return arcsin(xd)

cpdef object arccos_dispatch(object x):
    """
    arccos: 실수 |x|<=1 → cpdef double arccos
            실수 |x|>1  → 자동 승격 _arccos_complex
            복소수      → _arccos_complex
    """
    if type(x) is complex:
        return _arccos_complex(<double complex>x)
    cdef double xd = <double>x
    if xd > 1.0 or xd < -1.0:
        return _arccos_complex(complex(xd, 0.0))
    return arccos(xd)

cpdef object arctan_dispatch(object x):
    """arctan: 실수 → cpdef double arctan (항상 실수 결과), 복소수 → _arctan_complex"""
    if type(x) is complex:
        return _arctan_complex(<double complex>x)
    return arctan(<double>x)

cpdef object arcsec_dispatch(object x):
    """
    arcsec: 실수 |x|>=1 → cpdef double arcsec
            실수 |x|<1  → 자동 승격 _arcsec_complex
            복소수      → _arcsec_complex
    """
    if type(x) is complex:
        return _arcsec_complex(<double complex>x)
    cdef double xd = <double>x
    cdef double axd = -xd if xd < 0.0 else xd
    if axd < 1.0:
        return _arcsec_complex(complex(xd, 0.0))
    return arcsec(xd)

cpdef object arccosec_dispatch(object x):
    """
    arccosec: 실수 |x|>=1 → cpdef double arccosec
              실수 |x|<1  → 자동 승격 _arccosec_complex
              복소수      → _arccosec_complex
    """
    if type(x) is complex:
        return _arccosec_complex(<double complex>x)
    cdef double xd = <double>x
    cdef double axd = -xd if xd < 0.0 else xd
    if axd < 1.0:
        return _arccosec_complex(complex(xd, 0.0))
    return arccosec(xd)

cpdef object arccotan_dispatch(object x):
    """arccotan: 실수 → cpdef double arccotan, 복소수 → _arccotan_complex"""
    if type(x) is complex:
        return _arccotan_complex(<double complex>x)
    return arccotan(<double>x)
