# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# hyperbolic.pyx
#
# hypersin(sinh), hypercos(cosh), hypertan(tanh) + 파생
# 참조: algorithm_reference.md 섹션 7~9

from libc.stdint cimport uint32_t
from ._helpers cimport high_word, low_word
from .exponential cimport _exp_inline, _expm1_inline

import cmath as _cmath

# log(DBL_MAX) 근방 임계값
cdef double LN_DBL_MAX = 7.09782712893383973096e+02

# ------------------------------------------------------------------ sinh
cpdef double hypersin(double x) noexcept:
    """
    sinh(x) = (e^x - e^(-x)) / 2.
    musl sinh.c 알고리즘: expm1 기반.
    ULP: <= 2
    special: sinh(NaN)=NaN, sinh(+-0)=+-0, sinh(+-inf)=+-inf
    """
    cdef uint32_t ix
    cdef double h, t, absx

    ix = high_word(x) & 0x7FFFFFFFU

    # NaN, +-inf
    if ix >= 0x7FF00000U:
        return x + x

    # |x| < 2^-26: x 그대로
    if ix < 0x3E500000U:
        return x

    h = 0.5 if x >= 0.0 else -0.5
    absx = x if x >= 0.0 else -x

    if ix < 0x40862E42U:
        # |x| < log(DBL_MAX)
        t = _expm1_inline(absx)
        if ix < 0x3FF00000U:
            # |x| < 1: t/(t+1) 경로
            return h * (2.0 * t - t * t / (t + 1.0))
        return h * (t + t / (t + 1.0))

    # |x| >= log(DBL_MAX): exp(x/2)^2 분해로 overflow 방지 (musl expo2 방식)
    # sinh(x) ≈ sign * exp(|x|/2)^2 / 2
    cdef double t2 = _exp_inline(absx * 0.5)
    return h * t2 * t2


# ------------------------------------------------------------------ cosh
cpdef double hypercos(double x) noexcept:
    """
    cosh(x) = (e^x + e^(-x)) / 2.
    musl cosh.c 알고리즘.
    ULP: <= 2
    special: cosh(NaN)=NaN, cosh(+-0)=1.0, cosh(+-inf)=+inf
    """
    cdef uint32_t ix
    cdef double t, absx

    ix = high_word(x) & 0x7FFFFFFFU

    if ix >= 0x7FF00000U:
        return x * x  # NaN -> NaN, inf -> +inf

    absx = x if x >= 0.0 else -x

    # |x| < 2^-26: 1.0
    if ix < 0x3E500000U:
        return 1.0

    if ix < 0x3FE62E42U:
        # |x| < log(2)
        t = _expm1_inline(absx)
        return 1.0 + t * t / (2.0 * (1.0 + t))

    if ix < 0x40862E42U:
        # log(2) <= |x| < log(DBL_MAX)
        t = _exp_inline(absx)
        return 0.5 * (t + 1.0 / t)

    # |x| >= log(DBL_MAX): exp(x/2)^2 분해로 overflow 방지 (musl expo2 방식)
    # cosh(x) ≈ exp(|x|/2)^2 / 2
    cdef double t3 = _exp_inline(absx * 0.5)
    return 0.5 * t3 * t3


# ------------------------------------------------------------------ tanh
cpdef double hypertan(double x) noexcept:
    """
    tanh(x) = (e^2x - 1) / (e^2x + 1).
    musl tanh.c 알고리즘: expm1 기반.
    ULP: <= 2
    special: tanh(NaN)=NaN, tanh(+-0)=+-0, tanh(+-inf)=+-1
    """
    cdef uint32_t ix
    cdef double t, absx

    ix = high_word(x) & 0x7FFFFFFFU

    if ix >= 0x7FF00000U:
        # NaN -> NaN, +-inf -> +-1
        if x != x:  # NaN
            return x + x
        if x > 0.0:
            return 1.0
        return -1.0

    absx = x if x >= 0.0 else -x

    # |x| > 22: tanh(x) 는 1 (또는 -1) 로 수렴, tiny 보정 포함
    # tanh(x) = 1 - 2*exp(-2x) for large x
    # |x| > 22: 2*exp(-44) < 2 ULP, so just return +-1 directly
    if ix >= 0x40360000U:   # |x| > 22
        if x >= 0.0:
            return 1.0
        return -1.0

    # subnormal
    if ix < 0x3E300000U:
        return x

    t = _expm1_inline(2.0 * absx)

    if ix >= 0x3FE193EAU:
        # log(3)/2 < |x|
        t = 1.0 - 2.0 / (t + 2.0)
    elif ix >= 0x3FD058AEU:
        # log(5/3)/2 < |x| <= log(3)/2
        t = t / (t + 2.0)
    else:
        # t2 = expm1(-2|x|) (음수값), tanh(x) = -t2 / (t2 + 2)
        t = _expm1_inline(-2.0 * absx)
        t = -t / (t + 2.0)

    if x >= 0.0:
        return t
    return -t


# ------------------------------------------------------------------ 파생 함수
cpdef double hypersec(double x) noexcept:
    """hypersec(x) = sech(x) = 1 / cosh(x)"""
    return 1.0 / hypercos(x)


cpdef double hypercosec(double x) noexcept:
    """hypercosec(x) = csch(x) = 1 / sinh(x)"""
    cdef double s
    s = hypersin(x)
    return 1.0 / s


cpdef double hypercotan(double x) noexcept:
    """hypercotan(x) = coth(x) = cosh(x) / sinh(x)"""
    cdef double s
    s = hypersin(x)
    return hypercos(x) / s


# ------------------------------------------------------------------ 복소수 auto-dispatch (방향 A)
cpdef object hypersin_dispatch(object x):
    """hypersin: 실수 → cpdef double hypersin, 복소수 → cmath.sinh"""
    if type(x) is complex:
        return _cmath.sinh(x)
    return hypersin(<double>x)

cpdef object hypercos_dispatch(object x):
    """hypercos: 실수 → cpdef double hypercos, 복소수 → cmath.cosh"""
    if type(x) is complex:
        return _cmath.cosh(x)
    return hypercos(<double>x)

cpdef object hypertan_dispatch(object x):
    """hypertan: 실수 → cpdef double hypertan, 복소수 → cmath.tanh"""
    if type(x) is complex:
        return _cmath.tanh(x)
    return hypertan(<double>x)

cpdef object hypersec_dispatch(object x):
    """hypersec: 실수 → cpdef double hypersec, 복소수 → 1/cmath.cosh"""
    if type(x) is complex:
        return 1.0 / _cmath.cosh(x)
    return hypersec(<double>x)

cpdef object hypercosec_dispatch(object x):
    """hypercosec: 실수 → cpdef double hypercosec, 복소수 → 1/cmath.sinh"""
    if type(x) is complex:
        return 1.0 / _cmath.sinh(x)
    return hypercosec(<double>x)

cpdef object hypercotan_dispatch(object x):
    """hypercotan: 실수 → cpdef double hypercotan, 복소수 → cmath.cosh/cmath.sinh"""
    if type(x) is complex:
        return _cmath.cosh(x) / _cmath.sinh(x)
    return hypercotan(<double>x)
