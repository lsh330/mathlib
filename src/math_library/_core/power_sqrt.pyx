# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# power_sqrt.pyx
#
# sqrt: 하드웨어 sqrtsd (libc sqrt 허용)
# power: exp(y * ln(x)) + 정수 지수 이진 거듭제곱
# 복소수 자체 구현, 자동 실수→복소수 승격

from libc.math cimport sqrt as _libc_sqrt, INFINITY
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
from ._helpers cimport high_word, low_word, double_to_bits, bits_to_double, _hypot_real, _make_complex
from .exponential cimport _exp_inline, _exp_complex
from .logarithmic cimport _ln_inline, _ln_complex


# ------------------------------------------------------------------ sqrt
cdef double _sqrt_c(double x) noexcept nogil:
    """
    sqrt(x): IEEE 754 하드웨어 sqrtsd 직접 호출.
    ULP: <= 0.5 (IEEE 754 correctly rounded).
    """
    return _libc_sqrt(x)


cpdef double sqrt(double x) noexcept:
    """sqrt(x). special: NaN, +inf, +-0, x<0=NaN."""
    return _libc_sqrt(x)


# ------------------------------------------------------------------ 복소수 sqrt 커널
cdef double complex _sqrt_complex(double complex z) noexcept nogil:
    """
    sqrt(a+bi) — cancellation-safe 구현.
    r = hypot(a, b)
    두 경우에 따라 분기:
      a >= 0: sqrt_re = sqrt((r+a)/2),  sqrt_im = b / (2*sqrt_re)  [cancellation 없음]
      a <  0: sqrt_im_abs = sqrt((r-a)/2),  sqrt_re = |b| / (2*sqrt_im_abs)
    """
    cdef double a = z.real
    cdef double b = z.imag
    cdef double r, sqrt_re, sqrt_im, t

    # b = 0: 실수 sqrt
    if b == 0.0:
        if a >= 0.0:
            return _make_complex(_libc_sqrt(a), 0.0)
        # a < 0: 순허수 결과
        return _make_complex(0.0, _libc_sqrt(-a))

    r = _hypot_real(a, b)

    if a >= 0.0:
        # sqrt_re = sqrt((r+a)/2), sqrt_im = b / (2*sqrt_re)  → no cancellation
        sqrt_re = _libc_sqrt((r + a) * 0.5)
        if sqrt_re == 0.0:
            sqrt_im = 0.0
        else:
            sqrt_im = b * 0.5 / sqrt_re
    else:
        # a < 0: (r-a)는 (r + |a|) ≥ r > 0, no cancellation
        t = _libc_sqrt((r - a) * 0.5)
        sqrt_re = (b if b >= 0.0 else -b) * 0.5 / t
        sqrt_im = t if b >= 0.0 else -t

    return _make_complex(sqrt_re, sqrt_im)


# ------------------------------------------------------------------ 정수 지수 이진 거듭제곱
cdef double _power_integer(double base, int n) noexcept nogil:
    """이진 거듭제곱법"""
    cdef int neg = (n < 0)
    cdef int k = -n if neg else n
    cdef double result = 1.0
    cdef double b = base

    while k > 0:
        if k & 1:
            result *= b
        b *= b
        k >>= 1

    if neg:
        return 1.0 / result
    return result


# ------------------------------------------------------------------ power
cpdef double power(double base, double exponent) noexcept:
    """
    power(base, exponent) = base^exponent.
    특수값: pow(x,0)=1, pow(1,y)=1, NaN 전파,
            pow(-1,±inf)=1, pow(+0,y<0)=+inf, pow(x<0,비정수)=NaN
    """
    cdef uint32_t base_hx, exp_hx
    cdef int int_exp
    cdef double abs_exp
    cdef double ln_base, result

    base_hx = high_word(base) & 0x7FFFFFFFU
    exp_hx  = high_word(exponent) & 0x7FFFFFFFU

    if exponent == 0.0:
        return 1.0
    if base == 1.0:
        return 1.0
    if base != base or exponent != exponent:
        return base + exponent

    if base == -1.0 and exp_hx == 0x7FF00000U:
        return 1.0

    if exp_hx < 0x43300000U:
        int_exp = <int>exponent
        if <double>int_exp == exponent:
            if base < 0.0:
                result = _power_integer(-base, int_exp)
                if int_exp & 1:
                    return -result
                return result
            return _power_integer(base, int_exp)

    if base < 0.0:
        return (base - base) / (base - base)  # NaN

    if base == 0.0:
        if exponent < 0.0:
            return INFINITY  # +inf
        return 0.0

    if exponent == 0.5:
        return _libc_sqrt(base)

    ln_base = _ln_inline(base)
    return _exp_inline(exponent * ln_base)


# ------------------------------------------------------------------ 복소수 power 커널
cdef double complex _power_complex(double complex base, double complex exp) noexcept nogil:
    """
    power(base, z) = exp(z * ln(base))
    """
    cdef double complex ln_b = _ln_complex(base)
    # z * ln_b
    cdef double re = exp.real * ln_b.real - exp.imag * ln_b.imag
    cdef double im = exp.real * ln_b.imag + exp.imag * ln_b.real
    return _exp_complex(_make_complex(re, im))


# ------------------------------------------------------------------ 자동 승격 헬퍼
cdef inline bint _is_integer(double x) noexcept nogil:
    if x != x:   # NaN
        return 0
    cdef int64_t ix = <int64_t>x
    return <double>ix == x


# ------------------------------------------------------------------ cbrt (세제곱근)
cpdef double cbrt(double x) noexcept:
    """
    cbrt(x) = sign(x) * |x|^(1/3).
    실수 음수 입력도 실수 결과 반환 (복소수 주 분기 아님).
    """
    cdef double ax, result
    cdef int neg
    cdef uint32_t hx, ix
    hx  = high_word(x)
    ix  = hx & 0x7FFFFFFFU
    neg = 1 if (hx >> 31) else 0
    ax  = -x if neg else x

    # NaN, Inf, 0
    if ix >= 0x7FF00000U or ax == 0.0:
        return x

    # ax^(1/3) = exp(ln(ax)/3)
    result = _exp_inline(_ln_inline(ax) * (1.0 / 3.0))
    return -result if neg else result


cdef double complex _cbrt_complex(double complex z) noexcept nogil:
    """cbrt(z) = exp((1/3)*ln(z)) — 복소수 주 분기"""
    cdef double complex lz = _ln_complex(z)
    return _exp_complex(_make_complex(lz.real / 3.0, lz.imag / 3.0))


cpdef object cbrt_dispatch(object x):
    """cbrt: 실수 → 실수 (음수도 허용), 복소수 → 복소 주 분기"""
    if type(x) is complex:
        return _cbrt_complex(<double complex>x)
    return cbrt(<double>x)


# ------------------------------------------------------------------ 복소수 auto-dispatch + 자동 승격
cpdef object sqrt_dispatch(object x):
    """
    sqrt: 실수 x>=0 → cpdef double sqrt
          실수 x<0  → 자동 승격 _sqrt_complex(x+0j)
          복소수    → _sqrt_complex
    """
    if type(x) is complex:
        return _sqrt_complex(<double complex>x)
    cdef double xd = <double>x
    if xd < 0.0:
        return _sqrt_complex(complex(xd, 0.0))
    return _libc_sqrt(xd)


cpdef object power_dispatch(object base, object exponent):
    """
    power: 실수 base<0 and non-integer exp → 자동 승격
           복소수 → _power_complex
           실수   → power
    """
    cdef double bd, ed
    cdef double complex cb, ce
    if type(base) is complex or type(exponent) is complex:
        cb = <double complex>base     if type(base)     is complex else complex(base)
        ce = <double complex>exponent if type(exponent) is complex else complex(exponent)
        return _power_complex(cb, ce)
    bd = <double>base
    ed = <double>exponent
    if bd < 0.0 and not _is_integer(ed):
        cb = complex(bd, 0.0)
        ce = complex(ed, 0.0)
        return _power_complex(cb, ce)
    return power(bd, ed)
