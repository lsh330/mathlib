# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# exponential.pyx
#
# fdlibm e_exp.c / s_expm1.c 포팅
# Pade (2,3) 근사: exp(r) = 1 - (lo - x*c/(2-c) - hi)
# x = k*ln2 + r  으로 argument reduction, |r| <= 0.5*ln2
# 2^k scaling: int64_t 부호 확장 비트 덧셈
#
# 참조: http://www.netlib.org/fdlibm/e_exp.c
#       algorithm_reference.md 섹션 10

from libc.math cimport fma, fabs, ldexp
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
from ._helpers cimport high_word, low_word, double_to_bits, bits_to_double

import cmath as _cmath

# ------------------------------------------------------------------ fdlibm exp 상수
# argument reduction: x = k*ln2 + r,  k = round(x / ln2)
cdef double LN2HI   =  6.93147180369123816490e-01  # 0x3FE62E42 FEE00000
cdef double LN2LO   =  1.90821492927058770002e-10  # 0x3DEA39EF 35793C76
cdef double INVLN2  =  1.44269504088896338700e+00  # 0x3FF71547 652B82FE

# Pade(2,3) denominator polynomial: c = x - t^2 * P(t^2)
# exp(x) ~ 1 - (lo - x*c/(2-c) - hi)  after argument reduction
cdef double P1  =  1.66666666666666019037e-01  # 0x3FC55555 5555553E
cdef double P2  = -2.77777777770155933842e-03  # 0xBF66C16C 16BEBD93
cdef double P3  =  6.61375632143793436117e-05  # 0x3F11566A AF25DE2C
cdef double P4  = -1.65339022054652515390e-06  # 0xBEBBBD41 C5D26BF1
cdef double P5  =  4.13813679705723846039e-08  # 0x3E663769 720B0542

# overflow / underflow threshold
cdef double EXP_OVERFLOW  =  7.09782712893383973096e+02   # ln(DBL_MAX)
cdef double EXP_UNDERFLOW = -7.45133219101941108420e+02   # ln(DBL_MIN subnormal)
cdef double HUGE_VAL_SQ   =  1.7976931348623157e+308      # DBL_MAX
cdef double TINY_VAL_SQ   =  5.0e-324                     # DBL_MIN * 0.5

# ------------------------------------------------------------------ expm1 계수 (fdlibm s_expm1.c)
# expm1(r) 유리 근사의 분자/분모 Q 계수
cdef double EM_Q1 = -3.33333333333331316428e-02  # 0xBFA11111 111110F4
cdef double EM_Q2 =  1.58730158725481460165e-03  # 0x3F5A01A0 19FE5585
cdef double EM_Q3 = -7.93650757867487942473e-05  # 0xBF14CE19 9EAADBB7
cdef double EM_Q4 =  4.00821782732936239552e-06  # 0x3ED0CFCA 86E65239
cdef double EM_Q5 = -2.01099218183624371326e-07  # 0xBE8AFDB7 6E09C32D


# ------------------------------------------------------------------ 내부 함수
cdef double _exp_inline(double x) noexcept nogil:
    """
    fdlibm e_exp.c 포팅.
    exp(x) = 2^k * y  where y = 1 - (lo - x*c/(2-c) - hi)
    """
    cdef double hi, lo, r, t, c, y
    cdef int k
    cdef uint32_t hx
    cdef uint64_t y_bits
    cdef int64_t k64

    hx = high_word(x)

    # 특수값 처리
    if (hx & 0x7FFFFFFFU) >= 0x40862E43U:   # |x| >= 709.78
        if (hx & 0x7FFFFFFFU) >= 0x7FF00000U:
            # NaN 또는 infinity
            if x != x:           # NaN
                return x
            if x > 0.0:          # +inf
                return x
            return 0.0           # -inf -> 0
        if x >= EXP_OVERFLOW:
            return HUGE_VAL_SQ * HUGE_VAL_SQ   # -> +inf
        if x <= EXP_UNDERFLOW:
            return TINY_VAL_SQ * TINY_VAL_SQ   # -> +0

    # argument reduction: x = k*ln2 + r,  |r| <= 0.5*ln2
    if x >= 0.0:
        k = <int>(x * INVLN2 + 0.5)
    else:
        k = <int>(x * INVLN2 - 0.5)

    hi = x - <double>k * LN2HI   # x - k * (ln2 - lo)
    lo = <double>k * LN2LO       # 하위 보정항
    r  = hi - lo                  # |r| <= 0.5*ln2

    # Pade 분자 다항식: c = r - r^2*P(r^2)
    t = r * r
    c = r - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))))

    # exp(r) = 1 - (lo - r*c/(2-c) - hi)
    y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi)

    # 2^k 스케일링
    # B4: k <= -1022 이면 subnormal 결과 → 두 단계 스케일링으로 언더플로우 방지
    if k <= -1022:
        # y * 2^(k+54) * 2^(-54): k+54 은 정규화 범위(-968 ~ -1022+54=-968 이하도 처리)
        y_bits = double_to_bits(y)
        k64    = <int64_t>(k + 54)
        y = bits_to_double(y_bits + <uint64_t>(k64 << 52))
        return y * 5.551115123125783e-17   # 2^-54
    # 정규 스케일링: 지수 필드에 k 더하기 (int64_t 부호 안전)
    y_bits = double_to_bits(y)
    k64    = <int64_t>k
    return bits_to_double(y_bits + <uint64_t>(k64 << 52))


cdef double _expm1_inline(double x) noexcept nogil:
    """
    fdlibm s_expm1.c 포팅.
    expm1(x) = e^x - 1  고정밀 (|x| 소에서 취소 오차 없음)

    핵심 알고리즘:
      x = k*ln2 + x_red  (|x_red| <= 0.5*ln2)
      c = 반올림 오차: c = (hi - x_red) - lo  (아주 작은 값)
      Q 유리근사로 e = expm1 보정항 계산
      e = x_red*(e-c) - c - hxs  (보정 반영)
    """
    cdef double hi, lo, c, e, y, t
    cdef double hfx, hxs, r1
    cdef uint32_t hx, abshx
    cdef uint64_t y_bits, t_bits
    cdef int64_t k64
    cdef int k, xsb

    hx    = high_word(x)
    abshx = hx & 0x7FFFFFFFU
    xsb   = <int>(hx >> 31)

    # |x| >= 56*ln2 (~38.8): overflow / underflow
    if abshx >= 0x4043687AU:
        if abshx >= 0x7FF00000U:
            return x               # NaN / +-inf
        if xsb:
            return -1.0            # x <= -56*ln2: expm1 -> -1
        return _exp_inline(x) - 1.0   # x >= 56*ln2: overflow 방지용

    # |x| < 2^-54: expm1(x) ~= x
    if abshx < 0x3C900000U:
        return x

    # argument reduction
    k = 0
    c = 0.0
    hi = x
    lo = 0.0

    if abshx > 0x3FD62E42U:   # |x| > 0.5 * ln2
        if abshx < 0x3FF0A2B2U:   # |x| < 1.5 * ln2: k = +/-1 직접 설정
            if xsb == 0:
                hi = x - LN2HI
                lo = LN2LO
                k  = 1
            else:
                hi = x + LN2HI
                lo = -LN2LO
                k  = -1
        else:
            if xsb == 0:
                k = <int>(INVLN2 * x + 0.5)
            else:
                k = <int>(INVLN2 * x - 0.5)
            t  = <double>k
            hi = x - t * LN2HI
            lo = t * LN2LO

        # x_red = hi - lo, c = 반올림 오차
        x  = hi - lo
        c  = (hi - x) - lo   # 아주 작은 floating-point 반올림 오차

    # Q 다항식으로 expm1 보정항 계산
    hfx = 0.5 * x
    hxs = x * hfx
    r1  = 1.0 + hxs * (EM_Q1 + hxs * (EM_Q2 + hxs * (EM_Q3 + hxs * (EM_Q4 + hxs * EM_Q5))))
    t   = 3.0 - r1 * hfx
    e   = hxs * ((r1 - t) / (6.0 - x * t))

    if k == 0:
        # expm1(x) = x - (x*e - hxs)
        return x - (x * e - hxs)

    # 보정항 반영: e = x*(e-c) - c - hxs
    e = (x * (e - c) - c)
    e = e - hxs

    if k == -1:
        return 0.5 * (x - e) - 0.5

    if k == 1:
        if x < -0.25:
            return -2.0 * (e - (x + 0.5))
        return 1.0 + 2.0 * (x - e)

    # |k| >= 2
    if k <= -2 or k > 56:
        # 2^k * (1 - (e-x)) - 1
        y = 1.0 - (e - x)
        y_bits = double_to_bits(y)
        k64    = <int64_t>k
        return bits_to_double(y_bits + <uint64_t>(k64 << 52)) - 1.0

    if k < 20:
        # t = 1 - 2^-k  (비트 연산으로 정확하게)
        t_bits = (<uint64_t>0x3FF00000U - (<uint64_t>0x200000U >> k)) << 32
        t = bits_to_double(t_bits)
        y = t - (e - x)
        y_bits = double_to_bits(y)
        k64    = <int64_t>k
        return bits_to_double(y_bits + <uint64_t>(k64 << 52))
    else:
        # t = 2^-k
        t_bits = <uint64_t>(<int64_t>(0x3FF - k) << 52)
        t = bits_to_double(t_bits)
        y = x - (e + t)
        y = y + 1.0
        y_bits = double_to_bits(y)
        k64    = <int64_t>k
        return bits_to_double(y_bits + <uint64_t>(k64 << 52))


# ------------------------------------------------------------------ 공개 API
cpdef double exp(double x) noexcept:
    """
    자체 구현 exp(x) — fdlibm e_exp.c 알고리즘.
    math.exp 대비 <= 1.5x 이내 속도 목표.
    ULP 오차: <= 1 ULP
    """
    return _exp_inline(x)


# ------------------------------------------------------------------ 복소수 auto-dispatch (방향 A)
cpdef object exp_dispatch(object x):
    """exp: 실수 → cpdef double exp, 복소수 → cmath.exp"""
    if type(x) is complex:
        return _cmath.exp(x)
    return _exp_inline(<double>x)
