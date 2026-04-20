# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# trigonometric.pyx
#
# sin, cos, tan, sec, cosec, cotan — musl fdlibm 포팅
# 참조: algorithm_reference.md 섹션 1~3

from libc.math cimport fma
from libc.stdint cimport uint32_t, int32_t
from ._helpers cimport high_word, low_word, double_to_bits
from .argument_reduction cimport rem_pio2

import cmath as _cmath

# ------------------------------------------------------------------ sin 계수
# musl __sin.c, algorithm_reference.md 섹션 1
cdef double S1 = -1.66666666666666324348e-01  # 0xBFC5555555555549
cdef double S2 =  8.33333333332248946124e-03  # 0x3F8111111110F8A6
cdef double S3 = -1.98412698298579493134e-04  # 0xBF2A01A019C161D5
cdef double S4 =  2.75573137070700676789e-06  # 0x3EC71DE357B1FE7D
cdef double S5 = -2.50507602534068634195e-08  # 0xBE5AE5E68A2B9CEB
cdef double S6 =  1.58969099521155010221e-10  # 0x3DE5D93A5ACFD57C

# ------------------------------------------------------------------ cos 계수
# musl __cos.c, algorithm_reference.md 섹션 2
cdef double C1 =  4.16666666666666019037e-02  # 0x3FA555555555554C
cdef double C2 = -1.38888888888741095749e-03  # 0xBF56C16C16C15177
cdef double C3 =  2.48015872894767294178e-05  # 0x3EFA01A019CB1590
cdef double C4 = -2.75573143513906633035e-07  # 0xBE927E4F809C52AD
cdef double C5 =  2.08757232129817482790e-09  # 0x3E21EE9EBDB4B1C4
cdef double C6 = -1.13596475577881948265e-11  # 0xBDA8FAE9BE8838D4

# ------------------------------------------------------------------ tan 계수
# musl __tan.c, algorithm_reference.md 섹션 3 (13개)
cdef double T0  =  3.33333333333334091986e-01
cdef double T1  =  1.33333333333201242699e-01
cdef double T2  =  5.39682539762260521377e-02
cdef double T3  =  2.18694882948595424599e-02
cdef double T4  =  8.86323982359930005737e-03
cdef double T5  =  3.59207910759131235356e-03
cdef double T6  =  1.45620945432529025516e-03
cdef double T7  =  5.88041240820264096874e-04
cdef double T8  =  2.46463134818469906812e-04
cdef double T9  =  7.81794442939557092300e-05
cdef double T10 =  7.14072491382608190305e-05
cdef double T11 = -1.85586374855275456654e-05
cdef double T12 =  2.59073051863633712884e-05

# π/4 분할 상수 (_tan_kernel에서 큰 x 재매핑용)
cdef double PIO4_HI = 7.85398163397448278999e-01
cdef double PIO4_LO = 3.06161699786838301793e-17


# ------------------------------------------------------------------ 커널 함수
cdef double _sin_kernel(double x, double y) noexcept nogil:
    """
    [-π/4, π/4] 구간 sin 커널 (musl __sin.c).
    y: rem_pio2의 low-word 보정 (없으면 0.0 전달)
    """
    cdef double z, w, r, v
    z = x * x
    w = z * z
    r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6)
    v = z * x
    if y == 0.0:
        return x + v * (S1 + z * r)
    return x - ((z * (0.5 * y - v * r) - y) - v * S1)


cdef double _cos_kernel(double x, double y) noexcept nogil:
    """
    [-π/4, π/4] 구간 cos 커널 (musl __cos.c).
    y: rem_pio2의 low-word 보정
    """
    cdef double z, w, r, hz, w1
    z = x * x
    w = z * z
    r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6))
    hz = 0.5 * z
    w1 = 1.0 - hz
    return w1 + (((1.0 - w1) - hz) + (z * r - x * y))


cdef double _tan_kernel(double x, double y, int iy) noexcept nogil:
    """
    [-π/4, π/4] 구간 tan 커널 (musl/fdlibm __tan.c 포팅).
    iy == 0: tan(x+y) 반환
    iy == 1: -1/tan(x+y) 반환 (cotan용)

    |x| >= 0.6744 인 경우 (rem_pio2 후 x ≈ ±π/4):
    tan(π/4 - δ) = (1 - tan(δ))/(1 + tan(δ)) 항등식 사용.
    """
    cdef uint32_t hx, ix
    cdef double z, r, v, w, s
    cdef int sign, big_x

    hx = high_word(x)
    ix = hx & 0x7FFFFFFFU

    # |x| < 2^-28: 직접 반환 (선형 근사)
    # iy=1: tan_kernel은 -cot(x+y) = -(cos/sin) 반환
    # x ≈ 0이면 cot(x) ≈ 1/x, 따라서 -cot(x) ≈ -1/x
    if ix < 0x3E300000U:
        if iy == 1:
            return -1.0 / x
        return x + y

    # |x| >= 0.6744 여부를 기억 (재매핑 후 ix가 변경되므로 플래그 사용)
    big_x = 1 if ix >= 0x3FE59428U else 0
    sign = 1

    if big_x:
        # x < 0이면 먼저 반전
        if hx >> 31:
            x = -x
            y = -y
            sign = -1
        # π/4 - x로 재매핑: tan(π/4 - δ) 계산용
        z = PIO4_HI - x
        w = PIO4_LO - y
        x = z + w
        y = 0.0

    # Chebyshev 다항식 근사 (musl __tan.c 계수)
    z = x * x
    w = z * z
    r = T1 + w * (T3 + w * (T5 + w * (T7 + w * (T9 + w * T11))))
    v = z * (T2 + w * (T4 + w * (T6 + w * (T8 + w * (T10 + w * T12)))))
    s = z * x
    r = y + z * (s * (r + v) + y)
    r += T0 * s
    w = x + r    # w ≈ tan(x_remapped)

    if big_x:
        # tan(π/4 - δ) = (1 - tan(δ)) / (1 + tan(δ))
        # iy=0: tan  ->  (1-w)/(1+w)
        # iy=1: -cot -> -(1+w)/(1-w)
        if iy == 0:
            r = (1.0 - w) / (1.0 + w)
        else:
            r = -(1.0 + w) / (1.0 - w)
        return <double>sign * r

    if iy:
        return -1.0 / w
    return w


# ------------------------------------------------------------------ 공개 API
cpdef double sin(double x) noexcept:
    """
    자체 구현 sin(x) (라디안).
    ULP: <= 1 (musl 기준)
    special: sin(NaN)=NaN, sin(+-0)=+-0, sin(+-inf)=NaN
    """
    cdef double y[2]
    cdef int n, q
    cdef uint32_t ix

    ix = high_word(x) & 0x7FFFFFFFU

    # |x| <= π/4: 직접 커널
    if ix <= 0x3FE921FBU:
        # |x| < 2^-26: x 그대로
        if ix < 0x3E500000U:
            return x
        return _sin_kernel(x, 0.0)

    # NaN, +/-Inf
    if ix >= 0x7FF00000U:
        return x - x  # NaN

    # argument reduction
    n = rem_pio2(x, y)
    q = n & 3
    if q == 0:
        return  _sin_kernel(y[0], y[1])
    elif q == 1:
        return  _cos_kernel(y[0], y[1])
    elif q == 2:
        return -_sin_kernel(y[0], y[1])
    else:
        return -_cos_kernel(y[0], y[1])


cpdef double cos(double x) noexcept:
    """
    자체 구현 cos(x) (라디안).
    ULP: <= 1
    special: cos(NaN)=NaN, cos(+-0)=1.0, cos(+-inf)=NaN
    """
    cdef double y[2]
    cdef int n, q
    cdef uint32_t ix

    ix = high_word(x) & 0x7FFFFFFFU

    if ix <= 0x3FE921FBU:
        # |x| < 2^-27: 1.0 반환
        if ix < 0x3E46A09EU:
            return 1.0
        return _cos_kernel(x, 0.0)

    if ix >= 0x7FF00000U:
        return x - x  # NaN

    n = rem_pio2(x, y)
    q = n & 3
    if q == 0:
        return  _cos_kernel(y[0], y[1])
    elif q == 1:
        return -_sin_kernel(y[0], y[1])
    elif q == 2:
        return -_cos_kernel(y[0], y[1])
    else:
        return  _sin_kernel(y[0], y[1])


cpdef double tan(double x) noexcept:
    """
    자체 구현 tan(x) (라디안).
    ULP: <= 1 (π/2 근방 제외)
    special: tan(NaN)=NaN, tan(+-0)=+-0, tan(+-inf)=NaN
    """
    cdef double y[2]
    cdef int n
    cdef uint32_t ix

    ix = high_word(x) & 0x7FFFFFFFU

    if ix <= 0x3FE921FBU:
        if ix < 0x3E400000U:  # |x| < 2^-27
            return x
        return _tan_kernel(x, 0.0, 0)

    if ix >= 0x7FF00000U:
        return x - x  # NaN

    n = rem_pio2(x, y)
    return _tan_kernel(y[0], y[1], n & 1)


cpdef double sec(double x) noexcept:
    """sec(x) = 1 / cos(x)"""
    cdef double c
    c = cos(x)
    # cos(x) == 0이면 inf (정의되지 않음)
    return 1.0 / c


cpdef double cosec(double x) noexcept:
    """cosec(x) = csc(x) = 1 / sin(x)"""
    cdef double s
    s = sin(x)
    return 1.0 / s


cpdef double cotan(double x) noexcept:
    """cotan(x) = cot(x) = cos(x) / sin(x)"""
    cdef double s, c
    s = sin(x)
    c = cos(x)
    return c / s


# ------------------------------------------------------------------ 복소수 auto-dispatch (방향 A)
# isinstance 체크를 Cython 레이어에서 수행 → Python def 트램펄린 ~30ns 제거
cpdef object sin_dispatch(object x):
    """sin: 실수 → cpdef double sin, 복소수 → cmath.sin"""
    if type(x) is complex:
        return _cmath.sin(x)
    return sin(<double>x)

cpdef object cos_dispatch(object x):
    """cos: 실수 → cpdef double cos, 복소수 → cmath.cos"""
    if type(x) is complex:
        return _cmath.cos(x)
    return cos(<double>x)

cpdef object tan_dispatch(object x):
    """tan: 실수 → cpdef double tan, 복소수 → cmath.tan"""
    if type(x) is complex:
        return _cmath.tan(x)
    return tan(<double>x)

cpdef object sec_dispatch(object x):
    """sec: 실수 → cpdef double sec, 복소수 → 1/cmath.cos"""
    if type(x) is complex:
        return 1.0 / _cmath.cos(x)
    return sec(<double>x)

cpdef object cosec_dispatch(object x):
    """cosec: 실수 → cpdef double cosec, 복소수 → 1/cmath.sin"""
    if type(x) is complex:
        return 1.0 / _cmath.sin(x)
    return cosec(<double>x)

cpdef object cotan_dispatch(object x):
    """cotan: 실수 → cpdef double cotan, 복소수 → cmath.cos/cmath.sin"""
    if type(x) is complex:
        return _cmath.cos(x) / _cmath.sin(x)
    return cotan(<double>x)
