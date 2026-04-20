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
# sqrt: algorithm_reference.md 섹션 14 — 하드웨어 sqrtsd 직접 호출
# power: exp(y * ln(x)) 공식 + 정수 지수 이진 거듭제곱 최적화

from libc.math cimport sqrt as _libc_sqrt
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
from ._helpers cimport high_word, low_word, double_to_bits, bits_to_double
from .exponential cimport _exp_inline
from .logarithmic cimport _ln_inline

import cmath as _cmath

# ------------------------------------------------------------------ sqrt
cdef double _sqrt_c(double x) noexcept nogil:
    """
    sqrt(x): IEEE 754 하드웨어 sqrtsd 직접 호출 (libc sqrt 허용).
    algorithm_reference.md 섹션 14: 하드웨어 sqrt 사용 권장.
    ULP: <= 0.5 (IEEE 754 필수 correctly rounded).
    """
    return _libc_sqrt(x)


cpdef double sqrt(double x) noexcept:
    """
    sqrt(x).
    special cases:
      sqrt(NaN)   = NaN
      sqrt(+inf)  = +inf
      sqrt(+/-0)  = +/-0
      sqrt(x < 0) = NaN
    """
    return _libc_sqrt(x)


# ------------------------------------------------------------------ 정수 지수 이진 거듭제곱
cdef double _power_integer(double base, int n) noexcept nogil:
    """이진 거듭제곱법 (기수가 음수인 정수 지수 포함)"""
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

    algorithm_reference.md 섹션 13: pow = exp(y*log(x)).
    정수 지수는 이진 거듭제곱으로 분기.

    special cases (IEEE 754-2019):
      pow(x, 0)      = 1  (모든 x)
      pow(1, y)      = 1  (모든 y)
      pow(NaN, y!=0) = NaN
      pow(x, NaN)    = NaN (x!=1)
      pow(-1, +-inf) = 1
      pow(+0, y<0)   = +inf
      pow(x<0, 비정수) = NaN
    """
    cdef uint32_t base_hx, exp_hx
    cdef int int_exp
    cdef double abs_exp
    cdef double ln_base, result

    base_hx = high_word(base) & 0x7FFFFFFFU
    exp_hx  = high_word(exponent) & 0x7FFFFFFFU

    # 지수 == 0: 결과 1
    if exponent == 0.0:
        return 1.0

    # 밑 == 1: 결과 1
    if base == 1.0:
        return 1.0

    # NaN 처리
    if base != base or exponent != exponent:
        return base + exponent  # NaN 전파

    # M4: pow(-1, ±inf) = 1 (IEEE 754-2019) — 음수 base 체크보다 먼저 처리
    if base == -1.0 and exp_hx == 0x7FF00000U:  # exponent is +-inf
        return 1.0

    # 지수가 정수인지 확인 (|exponent| < 2^53 이면 체크 가능)
    if exp_hx < 0x43300000U:  # |exponent| < 2^52
        int_exp = <int>exponent
        if <double>int_exp == exponent:
            # 정수 지수: 이진 거듭제곱
            if base < 0.0:
                # 음수 밑 + 정수 지수: 부호 처리
                result = _power_integer(-base, int_exp)
                if int_exp & 1:
                    return -result
                return result
            return _power_integer(base, int_exp)

    # 밑 < 0, 비정수 지수: NaN
    if base < 0.0:
        return (base - base) / (base - base)

    # 밑 == 0 처리
    if base == 0.0:
        if exponent < 0.0:
            # +0^y<0 = +inf
            return 1.0 / 0.0
        return 0.0

    # M5: exponent == 0.5 → sqrt 고속 경로 (correctly rounded)
    if exponent == 0.5:
        return _libc_sqrt(base)

    # 일반 경로: exp(exponent * ln(base))
    ln_base = _ln_inline(base)
    return _exp_inline(exponent * ln_base)


# ------------------------------------------------------------------ 복소수 auto-dispatch (방향 A)
cpdef object sqrt_dispatch(object x):
    """sqrt: 실수 → cpdef double sqrt, 복소수 → cmath.sqrt"""
    if type(x) is complex:
        return _cmath.sqrt(x)
    return _libc_sqrt(<double>x)

cpdef object power_dispatch(object base, object exponent):
    """power: 실수 → cpdef double power, 복소수 → cmath.exp(exponent*cmath.log(base))"""
    if type(base) is complex or type(exponent) is complex:
        return _cmath.exp(complex(exponent) * _cmath.log(complex(base)))
    return power(<double>base, <double>exponent)
