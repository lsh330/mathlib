# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# inverse_hyperbolic.pyx
#
# arc_hypersin (asinh), arc_hypercos (acosh), arc_hypertan (atanh)
# arc_hypersec, arc_hypercosec, arc_hypercotan
# 복소수 자체 구현, 자동 승격 포함

from libc.math cimport fabs, log1p as _log1p_c
from libc.stdint cimport uint32_t
from ._helpers cimport double_to_bits, high_word, _make_complex
from .exponential cimport _expm1_inline, _exp_inline
from .logarithmic cimport _ln_inline, _ln_complex
from .power_sqrt cimport _sqrt_c, _sqrt_complex

# ------------------------------------------------------------------ 상수
cdef double _LN2   = 6.93147180369123816490e-01
cdef double _INF   = 1.0 / 0.0
cdef double _NAN_V = 0.0 / 0.0


# ------------------------------------------------------------------ arc_hypersin 실수 커널 (asinh)
cdef double _arc_hypersin_real(double x) noexcept nogil:
    """
    asinh(x) = ln(x + sqrt(x^2 + 1)), 전 정의역 수치 안정.
    소|x|:  log1p(x + x^3/(2*(1+sqrt(1+x^2)))) — cancellation 방지
    대|x|:  sign(x) * (ln(|x|) + ln(2))  [ln(|x|+sqrt(x^2+1)) ≈ ln(2|x|) for large |x|]
    """
    cdef double ax, t
    cdef int neg
    cdef uint32_t hx, ix

    hx  = high_word(x)
    ix  = hx & 0x7FFFFFFFU
    neg = 1 if (hx >> 31) else 0
    ax  = -x if neg else x

    # NaN / +Inf
    if ix >= 0x7FF00000U:
        return x + x

    # 매우 작은 값: |x| < 2^-26
    if ix < 0x3E500000U:
        return x

    cdef double result

    # 대|x|: |x| >= 2^28 → asinh(x) ≈ sign(x)*(ln|x| + ln2)
    if ix >= 0x41B00000U:
        result = _ln_inline(ax) + _LN2
    # 중간범위
    elif ix >= 0x3FE00000U:
        # ln(x + sqrt(x^2+1))  — 직접 계산, cancellation 없음
        result = _ln_inline(ax + _sqrt_c(ax * ax + 1.0))
    else:
        # 소|x|: log1p(x + x^2/(1+sqrt(1+x^2)))
        t = ax * ax
        result = _ln_inline(1.0 + ax + t / (1.0 + _sqrt_c(1.0 + t)))

    return -result if neg else result


# ------------------------------------------------------------------ arc_hypercos 실수 커널 (acosh)
cdef double _arc_hypercos_real(double x) noexcept nogil:
    """
    acosh(x) = ln(x + sqrt(x^2 - 1)),  x >= 1.
    x < 1: NaN (실수 결과 없음 — 자동 승격으로 복소수 반환)
    """
    cdef uint32_t hx, ix
    hx = high_word(x)
    ix = hx & 0x7FFFFFFFU

    # NaN
    if ix > 0x7FF00000U:
        return x + x
    # +Inf
    if ix == 0x7FF00000U:
        return x

    if x < 1.0:
        # x < 1: 실수 결과 없음 — NaN 반환 (호출측에서 승격)
        return (x - x) / (x - x)

    if x == 1.0:
        return 0.0

    # 대x: acosh(x) ≈ ln(2x) for x >> 1
    if ix >= 0x41B00000U:
        return _ln_inline(x) + _LN2

    # x > 2: ln(x + sqrt(x^2-1))  — 직접
    if ix >= 0x40000000U:
        return _ln_inline(x + _sqrt_c(x * x - 1.0))

    # 1 < x < 2: log1p 기반
    return _ln_inline(1.0 + (x - 1.0) + _sqrt_c((x - 1.0) * (x + 1.0)))


# ------------------------------------------------------------------ arc_hypertan 실수 커널 (atanh)
cdef double _arc_hypertan_real(double x) noexcept nogil:
    """
    atanh(x) = 0.5 * ln((1+x)/(1-x)),  |x| < 1.
    |x| = 1: ±inf
    |x| > 1: NaN (실수 결과 없음 — 승격)
    작은 x: log1p 기반으로 cancellation 방지
    """
    cdef uint32_t hx, ix
    cdef double ax, t
    cdef int neg

    hx  = high_word(x)
    ix  = hx & 0x7FFFFFFFU
    neg = 1 if (hx >> 31) else 0
    ax  = -x if neg else x

    # NaN
    if ix > 0x7FF00000U:
        return x + x

    # |x| > 1: NaN (호출측에서 복소수 승격)
    if ix > 0x3FF00000U:
        return (x - x) / (x - x)

    # |x| = 1: ±inf
    if ix == 0x3FF00000U:
        if neg:
            return -_INF
        return _INF

    cdef double result

    # 소|x|: |x| < 2^-28
    if ix < 0x3E300000U:
        return x

    # |x| < 0.5: 0.5 * log1p(2x/(1-x))  — log1p 기반 정밀
    if ix < 0x3FE00000U:
        t = ax + ax
        result = 0.5 * _log1p_c(t + t * ax / (1.0 - ax))
    else:
        # 0.5 <= |x| < 1: log1p(+x) - log1p(-(1-x)-1+x) = log1p 조합
        # atanh(x) = 0.5*(log1p(x) - log1p(-x))
        result = 0.5 * (_log1p_c(ax) - _log1p_c(-ax))

    return -result if neg else result


# ------------------------------------------------------------------ 복소수 커널
cdef double complex _arc_hypersin_complex(double complex z) noexcept nogil:
    """asinh(z) = ln(z + sqrt(z^2 + 1))"""
    cdef double re = z.real, im = z.imag
    # z^2 + 1
    cdef double zz_re = re * re - im * im + 1.0
    cdef double zz_im = 2.0 * re * im
    cdef double complex sq = _sqrt_complex(_make_complex(zz_re, zz_im))
    cdef double complex w  = _make_complex(re + sq.real, im + sq.imag)
    return _ln_complex(w)


cdef double complex _arc_hypercos_complex(double complex z) noexcept nogil:
    """acosh(z) = ln(z + sqrt(z^2 - 1))"""
    cdef double re = z.real, im = z.imag
    # z^2 - 1
    cdef double zz_re = re * re - im * im - 1.0
    cdef double zz_im = 2.0 * re * im
    cdef double complex sq = _sqrt_complex(_make_complex(zz_re, zz_im))
    cdef double complex w  = _make_complex(re + sq.real, im + sq.imag)
    return _ln_complex(w)


cdef double complex _arc_hypertan_complex(double complex z) noexcept nogil:
    """atanh(z) = 0.5 * ln((1+z)/(1-z))"""
    cdef double re = z.real, im = z.imag
    # num = 1+z,  den = 1-z
    cdef double num_re = 1.0 + re, num_im = im
    cdef double den_re = 1.0 - re, den_im = -im
    cdef double denom2 = den_re * den_re + den_im * den_im
    if denom2 == 0.0:
        return _make_complex(1.0 / 0.0, 0.0)
    cdef double ratio_re = (num_re * den_re + num_im * den_im) / denom2
    cdef double ratio_im = (num_im * den_re - num_re * den_im) / denom2
    cdef double complex lr = _ln_complex(_make_complex(ratio_re, ratio_im))
    return _make_complex(0.5 * lr.real, 0.5 * lr.imag)


# ------------------------------------------------------------------ 공개 API (auto-dispatch + 자동 승격)

cpdef object arc_hypersin(object x):
    """
    arc_hypersin(x) = asinh(x). 전 실수 정의역.
    복소수 입력도 처리.
    """
    if type(x) is complex:
        return _arc_hypersin_complex(<double complex>x)
    return _arc_hypersin_real(<double>x)


cpdef object arc_hypercos(object x):
    """
    arc_hypercos(x) = acosh(x).
    x < 1: 자동 승격 → 복소수 반환.
    """
    if type(x) is complex:
        return _arc_hypercos_complex(<double complex>x)
    cdef double xd = <double>x
    if xd < 1.0:
        return _arc_hypercos_complex(_make_complex(xd, 0.0))
    return _arc_hypercos_real(xd)


cpdef object arc_hypertan(object x):
    """
    arc_hypertan(x) = atanh(x).
    |x| >= 1 (비inf): 자동 승격 → 복소수 반환 (단, |x|=1 → ±inf).
    """
    cdef double xd
    cdef uint32_t ix
    if type(x) is complex:
        return _arc_hypertan_complex(<double complex>x)
    xd = <double>x
    ix = high_word(xd) & 0x7FFFFFFFU
    if ix > 0x3FF00000U:
        # |x| > 1: 복소수 승격
        return _arc_hypertan_complex(_make_complex(xd, 0.0))
    return _arc_hypertan_real(xd)


cpdef object arc_hypersec(object x):
    """arc_hypersec(x) = arc_hypercos(1/x)"""
    cdef double complex zc
    cdef double denom, xd
    if type(x) is complex:
        zc = <double complex>x
        denom = zc.real * zc.real + zc.imag * zc.imag
        if denom == 0.0:
            return complex(1.0 / 0.0, 0.0)
        return _arc_hypercos_complex(_make_complex(zc.real / denom, -zc.imag / denom))
    xd = <double>x
    if xd == 0.0:
        return 1.0 / 0.0
    return arc_hypercos(1.0 / xd)


cpdef object arc_hypercosec(object x):
    """arc_hypercosec(x) = arc_hypersin(1/x)"""
    cdef double complex zc
    cdef double denom, xd
    if type(x) is complex:
        zc = <double complex>x
        denom = zc.real * zc.real + zc.imag * zc.imag
        if denom == 0.0:
            return complex(1.0 / 0.0, 0.0)
        return _arc_hypersin_complex(_make_complex(zc.real / denom, -zc.imag / denom))
    xd = <double>x
    if xd == 0.0:
        return 1.0 / 0.0
    return arc_hypersin(1.0 / xd)


cpdef object arc_hypercotan(object x):
    """arc_hypercotan(x) = arc_hypertan(1/x)"""
    cdef double complex zc
    cdef double denom, xd
    if type(x) is complex:
        zc = <double complex>x
        denom = zc.real * zc.real + zc.imag * zc.imag
        if denom == 0.0:
            return complex(0.0, 0.0)
        return _arc_hypertan_complex(_make_complex(zc.real / denom, -zc.imag / denom))
    xd = <double>x
    if xd == 0.0:
        return 0.0 / 0.0
    return arc_hypertan(1.0 / xd)
