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
# 복소수 exp 자체 구현 (cmath 사용 금지)
#
# 참조: algorithm_reference.md 섹션 10

from libc.math cimport fma, fabs, ldexp
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
from ._helpers cimport high_word, low_word, double_to_bits, bits_to_double

# ------------------------------------------------------------------ fdlibm exp 상수
cdef double LN2HI   =  6.93147180369123816490e-01
cdef double LN2LO   =  1.90821492927058770002e-10
cdef double INVLN2  =  1.44269504088896338700e+00

cdef double P1  =  1.66666666666666019037e-01
cdef double P2  = -2.77777777770155933842e-03
cdef double P3  =  6.61375632143793436117e-05
cdef double P4  = -1.65339022054652515390e-06
cdef double P5  =  4.13813679705723846039e-08

cdef double EXP_OVERFLOW  =  7.09782712893383973096e+02
cdef double EXP_UNDERFLOW = -7.45133219101941108420e+02
cdef double HUGE_VAL_SQ   =  1.7976931348623157e+308
cdef double TINY_VAL_SQ   =  5.0e-324

# ------------------------------------------------------------------ expm1 계수
cdef double EM_Q1 = -3.33333333333331316428e-02
cdef double EM_Q2 =  1.58730158725481460165e-03
cdef double EM_Q3 = -7.93650757867487942473e-05
cdef double EM_Q4 =  4.00821782732936239552e-06
cdef double EM_Q5 = -2.01099218183624371326e-07


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

    if (hx & 0x7FFFFFFFU) >= 0x40862E43U:
        if (hx & 0x7FFFFFFFU) >= 0x7FF00000U:
            if x != x:
                return x
            if x > 0.0:
                return x
            return 0.0
        if x >= EXP_OVERFLOW:
            return HUGE_VAL_SQ * HUGE_VAL_SQ
        if x <= EXP_UNDERFLOW:
            return TINY_VAL_SQ * TINY_VAL_SQ

    if x >= 0.0:
        k = <int>(x * INVLN2 + 0.5)
    else:
        k = <int>(x * INVLN2 - 0.5)

    hi = x - <double>k * LN2HI
    lo = <double>k * LN2LO
    r  = hi - lo

    t = r * r
    c = r - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))))
    y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi)

    if k <= -1022:
        y_bits = double_to_bits(y)
        k64    = <int64_t>(k + 54)
        y = bits_to_double(y_bits + <uint64_t>(k64 << 52))
        return y * 5.551115123125783e-17
    y_bits = double_to_bits(y)
    k64    = <int64_t>k
    return bits_to_double(y_bits + <uint64_t>(k64 << 52))


cdef double _expm1_inline(double x) noexcept nogil:
    """
    fdlibm s_expm1.c 포팅.
    expm1(x) = e^x - 1  고정밀
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

    if abshx >= 0x4043687AU:
        if abshx >= 0x7FF00000U:
            return x
        if xsb:
            return -1.0
        return _exp_inline(x) - 1.0

    if abshx < 0x3C900000U:
        return x

    k = 0
    c = 0.0
    hi = x
    lo = 0.0

    if abshx > 0x3FD62E42U:
        if abshx < 0x3FF0A2B2U:
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

        x  = hi - lo
        c  = (hi - x) - lo

    hfx = 0.5 * x
    hxs = x * hfx
    r1  = 1.0 + hxs * (EM_Q1 + hxs * (EM_Q2 + hxs * (EM_Q3 + hxs * (EM_Q4 + hxs * EM_Q5))))
    t   = 3.0 - r1 * hfx
    e   = hxs * ((r1 - t) / (6.0 - x * t))

    if k == 0:
        return x - (x * e - hxs)

    e = (x * (e - c) - c)
    e = e - hxs

    if k == -1:
        return 0.5 * (x - e) - 0.5

    if k == 1:
        if x < -0.25:
            return -2.0 * (e - (x + 0.5))
        return 1.0 + 2.0 * (x - e)

    if k <= -2 or k > 56:
        y = 1.0 - (e - x)
        y_bits = double_to_bits(y)
        k64    = <int64_t>k
        return bits_to_double(y_bits + <uint64_t>(k64 << 52)) - 1.0

    if k < 20:
        t_bits = (<uint64_t>0x3FF00000U - (<uint64_t>0x200000U >> k)) << 32
        t = bits_to_double(t_bits)
        y = t - (e - x)
        y_bits = double_to_bits(y)
        k64    = <int64_t>k
        return bits_to_double(y_bits + <uint64_t>(k64 << 52))
    else:
        t_bits = <uint64_t>(<int64_t>(0x3FF - k) << 52)
        t = bits_to_double(t_bits)
        y = x - (e + t)
        y = y + 1.0
        y_bits = double_to_bits(y)
        k64    = <int64_t>k
        return bits_to_double(y_bits + <uint64_t>(k64 << 52))


# ------------------------------------------------------------------ 복소수 exp 커널 (자체 구현)
# exp(a+bi) = exp(a) * (cos(b) + i*sin(b))
# sin(b), cos(b)는 trigonometric 커널에서 직접 가져오면 순환 cimport 발생하므로
# 여기서 간단한 폴리노미얼로 직접 계산 (sin_kernel 복제 방식 또는 arg reduction 포함)
# → trigonometric 커널을 cimport 하지 않고, argument_reduction을 통해 자체 계산

from .argument_reduction cimport rem_pio2
from ._helpers cimport high_word as _hw, _make_complex

# sin/cos 커널 상수 (exponential.pyx 내부 복제 — trigonometric cimport 대신)
cdef double _ES1 = -1.66666666666666324348e-01
cdef double _ES2 =  8.33333333332248946124e-03
cdef double _ES3 = -1.98412698298579493134e-04
cdef double _ES4 =  2.75573137070700676789e-06
cdef double _ES5 = -2.50507602534068634195e-08
cdef double _ES6 =  1.58969099521155010221e-10

cdef double _EC1 =  4.16666666666666019037e-02
cdef double _EC2 = -1.38888888888741095749e-03
cdef double _EC3 =  2.48015872894767294178e-05
cdef double _EC4 = -2.75573143513906633035e-07
cdef double _EC5 =  2.08757232129817482790e-09
cdef double _EC6 = -1.13596475577881948265e-11


cdef inline double _esin(double x) noexcept nogil:
    cdef double z, w, r, v
    z = x * x; w = z * z
    r = _ES2 + z*(_ES3 + z*_ES4) + z*w*(_ES5 + z*_ES6)
    v = z * x
    return x + v * (_ES1 + z * r)


cdef inline double _ecos(double x) noexcept nogil:
    cdef double z, w, r, hz, w1
    z = x * x; w = z * z
    r = z*(_EC1 + z*(_EC2 + z*_EC3)) + w*w*(_EC4 + z*(_EC5 + z*_EC6))
    hz = 0.5 * z; w1 = 1.0 - hz
    return w1 + (((1.0 - w1) - hz) + z * r)


cdef inline void _sincos_real(double x, double* sin_out, double* cos_out) noexcept nogil:
    """실수 x의 sin/cos 동시 계산 (복소수 커널 내부 전용)"""
    cdef double y[2]
    cdef int n, q
    cdef uint32_t ix
    ix = _hw(x) & 0x7FFFFFFFU
    if ix <= 0x3FE921FBU:
        if ix < 0x3E500000U:
            sin_out[0] = x; cos_out[0] = 1.0
        else:
            sin_out[0] = _esin(x); cos_out[0] = _ecos(x)
        return
    if ix >= 0x7FF00000U:
        sin_out[0] = x - x; cos_out[0] = x - x; return
    n = rem_pio2(x, y)
    q = n & 3
    if q == 0:
        sin_out[0] =  _esin(y[0]); cos_out[0] =  _ecos(y[0])
    elif q == 1:
        sin_out[0] =  _ecos(y[0]); cos_out[0] = -_esin(y[0])
    elif q == 2:
        sin_out[0] = -_esin(y[0]); cos_out[0] = -_ecos(y[0])
    else:
        sin_out[0] = -_ecos(y[0]); cos_out[0] =  _esin(y[0])


cdef double complex _exp_complex(double complex z) noexcept nogil:
    """
    exp(a+bi) = exp(a) * (cos(b) + i*sin(b))
    """
    cdef double a = z.real
    cdef double b = z.imag
    cdef double ea = _exp_inline(a)
    cdef double sb, cb
    _sincos_real(b, &sb, &cb)
    return _make_complex(ea * cb, ea * sb)


# ------------------------------------------------------------------ 공개 API
cpdef double exp(double x) noexcept:
    """
    자체 구현 exp(x) — fdlibm e_exp.c 알고리즘.
    ULP 오차: <= 1 ULP
    """
    return _exp_inline(x)


cpdef double expm1(double x) noexcept:
    """
    expm1(x) = e^x - 1.
    소 x에서 수치 안정.  fdlibm s_expm1.c 알고리즘.
    """
    return _expm1_inline(x)


# 복소수 expm1
cdef double complex _expm1_complex(double complex z) noexcept nogil:
    """expm1(z) = exp(z) - 1 — 복소수"""
    cdef double complex ez = _exp_complex(z)
    return _make_complex(ez.real - 1.0, ez.imag)


# ------------------------------------------------------------------ 복소수 auto-dispatch (방향 A) — cmath 없이 자체 구현
cpdef object exp_dispatch(object x):
    """exp: 실수 → cpdef double exp, 복소수 → 자체 _exp_complex"""
    if type(x) is complex:
        return _exp_complex(<double complex>x)
    return _exp_inline(<double>x)


cpdef object expm1_dispatch(object x):
    """expm1: 실수 → expm1, 복소수 → _expm1_complex"""
    if type(x) is complex:
        return _expm1_complex(<double complex>x)
    return _expm1_inline(<double>x)
