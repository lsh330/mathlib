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
# 복소수 커널 자체 구현 (cmath 사용 금지)
# 참조: algorithm_reference.md 섹션 7~9

from libc.stdint cimport uint32_t
from libc.math cimport INFINITY
from ._helpers cimport high_word, low_word
from .exponential cimport _exp_inline, _expm1_inline

# log(DBL_MAX) 근방 임계값
cdef double LN_DBL_MAX = 7.09782712893383973096e+02


# ------------------------------------------------------------------ 내부 nogil 버전 (복소수 커널에서 호출)
cdef double _sinh_r(double x) noexcept nogil:
    cdef uint32_t ix
    cdef double h, t, absx
    ix = high_word(x) & 0x7FFFFFFFU
    if ix >= 0x7FF00000U:
        return x + x
    if ix < 0x3E500000U:
        return x
    h = 0.5 if x >= 0.0 else -0.5
    absx = x if x >= 0.0 else -x
    if ix < 0x40862E42U:
        t = _expm1_inline(absx)
        if ix < 0x3FF00000U:
            return h * (2.0 * t - t * t / (t + 1.0))
        return h * (t + t / (t + 1.0))
    cdef double t2 = _exp_inline(absx * 0.5)
    return h * t2 * t2


cdef double _cosh_r(double x) noexcept nogil:
    cdef uint32_t ix
    cdef double t, absx
    ix = high_word(x) & 0x7FFFFFFFU
    if ix >= 0x7FF00000U:
        return x * x
    absx = x if x >= 0.0 else -x
    if ix < 0x3E500000U:
        return 1.0
    if ix < 0x3FE62E42U:
        t = _expm1_inline(absx)
        return 1.0 + t * t / (2.0 * (1.0 + t))
    if ix < 0x40862E42U:
        t = _exp_inline(absx)
        return 0.5 * (t + 1.0 / t)
    cdef double t3 = _exp_inline(absx * 0.5)
    return 0.5 * t3 * t3


# ------------------------------------------------------------------ sinh (실수)
cpdef double hypersin(double x) noexcept:
    """
    sinh(x). expm1 기반. ULP<=2.
    special: NaN=NaN, +-0=+-0, +-inf=+-inf
    """
    cdef uint32_t ix
    cdef double h, t, absx

    ix = high_word(x) & 0x7FFFFFFFU

    if ix >= 0x7FF00000U:
        return x + x

    if ix < 0x3E500000U:
        return x

    h = 0.5 if x >= 0.0 else -0.5
    absx = x if x >= 0.0 else -x

    if ix < 0x40862E42U:
        t = _expm1_inline(absx)
        if ix < 0x3FF00000U:
            return h * (2.0 * t - t * t / (t + 1.0))
        return h * (t + t / (t + 1.0))

    cdef double t2 = _exp_inline(absx * 0.5)
    return h * t2 * t2


# ------------------------------------------------------------------ cosh (실수)
cpdef double hypercos(double x) noexcept:
    """
    cosh(x). ULP<=2.
    special: NaN=NaN, +-0=1.0, +-inf=+inf
    """
    cdef uint32_t ix
    cdef double t, absx

    ix = high_word(x) & 0x7FFFFFFFU

    if ix >= 0x7FF00000U:
        return x * x

    absx = x if x >= 0.0 else -x

    if ix < 0x3E500000U:
        return 1.0

    if ix < 0x3FE62E42U:
        t = _expm1_inline(absx)
        return 1.0 + t * t / (2.0 * (1.0 + t))

    if ix < 0x40862E42U:
        t = _exp_inline(absx)
        return 0.5 * (t + 1.0 / t)

    cdef double t3 = _exp_inline(absx * 0.5)
    return 0.5 * t3 * t3


# ------------------------------------------------------------------ tanh (실수)
cpdef double hypertan(double x) noexcept:
    """
    tanh(x). expm1 기반. ULP<=2.
    special: NaN=NaN, +-0=+-0, +-inf=+-1
    """
    cdef uint32_t ix
    cdef double t, absx

    ix = high_word(x) & 0x7FFFFFFFU

    if ix >= 0x7FF00000U:
        if x != x:
            return x + x
        if x > 0.0:
            return 1.0
        return -1.0

    absx = x if x >= 0.0 else -x

    if ix >= 0x40360000U:
        if x >= 0.0:
            return 1.0
        return -1.0

    if ix < 0x3E300000U:
        return x

    t = _expm1_inline(2.0 * absx)

    if ix >= 0x3FE193EAU:
        t = 1.0 - 2.0 / (t + 2.0)
    elif ix >= 0x3FD058AEU:
        t = t / (t + 2.0)
    else:
        t = _expm1_inline(-2.0 * absx)
        t = -t / (t + 2.0)

    if x >= 0.0:
        return t
    return -t


# ------------------------------------------------------------------ 파생 함수 (실수)
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


# ------------------------------------------------------------------ 복소수 커널 (자체 구현)
# sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
# cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
# tanh(z)    = sinh(z)/cosh(z) (overflow 처리 포함)

# sin/cos 커널 상수 (hyperbolic.pyx 내부 전용 복제)
cdef double _HS1 = -1.66666666666666324348e-01
cdef double _HS2 =  8.33333333332248946124e-03
cdef double _HS3 = -1.98412698298579493134e-04
cdef double _HS4 =  2.75573137070700676789e-06
cdef double _HS5 = -2.50507602534068634195e-08
cdef double _HS6 =  1.58969099521155010221e-10

cdef double _HC1 =  4.16666666666666019037e-02
cdef double _HC2 = -1.38888888888741095749e-03
cdef double _HC3 =  2.48015872894767294178e-05
cdef double _HC4 = -2.75573143513906633035e-07
cdef double _HC5 =  2.08757232129817482790e-09
cdef double _HC6 = -1.13596475577881948265e-11

from .argument_reduction cimport rem_pio2
from ._helpers cimport high_word as _hw_h, _make_complex


cdef inline double _hsin(double x) noexcept nogil:
    cdef double z, w, r, v
    z = x*x; w = z*z
    r = _HS2 + z*(_HS3 + z*_HS4) + z*w*(_HS5 + z*_HS6)
    v = z*x
    return x + v*(_HS1 + z*r)


cdef inline double _hcos(double x) noexcept nogil:
    cdef double z, w, r, hz, w1
    z = x*x; w = z*z
    r = z*(_HC1 + z*(_HC2 + z*_HC3)) + w*w*(_HC4 + z*(_HC5 + z*_HC6))
    hz = 0.5*z; w1 = 1.0 - hz
    return w1 + (((1.0 - w1) - hz) + z*r)


cdef inline void _sincos_b(double b, double* s, double* c) noexcept nogil:
    """실수 b의 sin/cos 계산 (복소수 쌍곡 커널 내부 전용)"""
    cdef double y[2]
    cdef int n, q
    cdef uint32_t ix
    ix = _hw_h(b) & 0x7FFFFFFFU
    if ix <= 0x3FE921FBU:
        if ix < 0x3E500000U:
            s[0] = b; c[0] = 1.0
        else:
            s[0] = _hsin(b); c[0] = _hcos(b)
        return
    if ix >= 0x7FF00000U:
        s[0] = b-b; c[0] = b-b; return
    n = rem_pio2(b, y)
    q = n & 3
    if q == 0:
        s[0] =  _hsin(y[0]); c[0] =  _hcos(y[0])
    elif q == 1:
        s[0] =  _hcos(y[0]); c[0] = -_hsin(y[0])
    elif q == 2:
        s[0] = -_hsin(y[0]); c[0] = -_hcos(y[0])
    else:
        s[0] = -_hcos(y[0]); c[0] =  _hsin(y[0])


cdef double complex _hypersin_complex(double complex z) noexcept nogil:
    """sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)"""
    cdef double a = z.real, b = z.imag
    cdef double sinh_a = _sinh_r(a)
    cdef double cosh_a = _cosh_r(a)
    cdef double sin_b, cos_b
    _sincos_b(b, &sin_b, &cos_b)
    return _make_complex(sinh_a * cos_b, cosh_a * sin_b)


cdef double complex _hypercos_complex(double complex z) noexcept nogil:
    """cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)"""
    cdef double a = z.real, b = z.imag
    cdef double sinh_a = _sinh_r(a)
    cdef double cosh_a = _cosh_r(a)
    cdef double sin_b, cos_b
    _sincos_b(b, &sin_b, &cos_b)
    return _make_complex(cosh_a * cos_b, sinh_a * sin_b)


cdef double complex _hypertan_complex(double complex z) noexcept nogil:
    """
    tanh(a+bi) = (sinh(2a) + i*sin(2b)) / (cosh(2a) + cos(2b))
    더 안정적인 공식 (cmath 내부와 동일한 접근).
    overflow 처리: |a| > 20 이면 ±1 근사.
    """
    cdef double a = z.real, b = z.imag
    cdef double absa = a if a >= 0.0 else -a
    cdef double sin2b, cos2b, sinh2a, cosh2a, denom

    if absa > 20.0:
        if a > 0.0:
            return _make_complex(1.0, 0.0)
        return _make_complex(-1.0, 0.0)

    # 2b의 sin/cos 계산
    _sincos_b(2.0 * b, &sin2b, &cos2b)

    # 2a의 sinh/cosh 계산 (nogil 버전)
    sinh2a = _sinh_r(2.0 * a)
    cosh2a = _cosh_r(2.0 * a)

    denom = cosh2a + cos2b
    if denom == 0.0:
        return _make_complex(INFINITY, 0.0)

    return _make_complex(sinh2a / denom, sin2b / denom)


cdef double complex _hypersec_complex(double complex z) noexcept nogil:
    """hypersec(z) = 1/cosh(z)"""
    cdef double complex c = _hypercos_complex(z)
    cdef double denom = c.real*c.real + c.imag*c.imag
    if denom == 0.0:
        return _make_complex(INFINITY, 0.0)
    return _make_complex(c.real / denom, -c.imag / denom)


cdef double complex _hypercosec_complex(double complex z) noexcept nogil:
    """hypercosec(z) = 1/sinh(z)"""
    cdef double complex s = _hypersin_complex(z)
    cdef double denom = s.real*s.real + s.imag*s.imag
    if denom == 0.0:
        return _make_complex(INFINITY, 0.0)
    return _make_complex(s.real / denom, -s.imag / denom)


cdef double complex _hypercotan_complex(double complex z) noexcept nogil:
    """hypercotan(z) = cosh(z)/sinh(z)"""
    cdef double complex s = _hypersin_complex(z)
    cdef double complex c = _hypercos_complex(z)
    cdef double denom = s.real*s.real + s.imag*s.imag
    if denom == 0.0:
        return _make_complex(INFINITY, 0.0)
    return _make_complex(
        (c.real*s.real + c.imag*s.imag) / denom,
        (c.imag*s.real - c.real*s.imag) / denom
    )


# ------------------------------------------------------------------ 복소수 auto-dispatch (방향 A) — cmath 없이 자체 구현
# 쌍곡함수는 실수 입력 시 항상 실수 결과 (자동 승격 없음)
cpdef object hypersin_dispatch(object x):
    """hypersin: 실수 → cpdef double hypersin, 복소수 → 자체 _hypersin_complex"""
    if type(x) is complex:
        return _hypersin_complex(<double complex>x)
    return hypersin(<double>x)

cpdef object hypercos_dispatch(object x):
    """hypercos: 실수 → cpdef double hypercos, 복소수 → 자체 _hypercos_complex"""
    if type(x) is complex:
        return _hypercos_complex(<double complex>x)
    return hypercos(<double>x)

cpdef object hypertan_dispatch(object x):
    """hypertan: 실수 → cpdef double hypertan, 복소수 → 자체 _hypertan_complex"""
    if type(x) is complex:
        return _hypertan_complex(<double complex>x)
    return hypertan(<double>x)

cpdef object hypersec_dispatch(object x):
    """hypersec: 실수 → cpdef double hypersec, 복소수 → 자체 _hypersec_complex"""
    if type(x) is complex:
        return _hypersec_complex(<double complex>x)
    return hypersec(<double>x)

cpdef object hypercosec_dispatch(object x):
    """hypercosec: 실수 → cpdef double hypercosec, 복소수 → 자체 _hypercosec_complex"""
    if type(x) is complex:
        return _hypercosec_complex(<double complex>x)
    return hypercosec(<double>x)

cpdef object hypercotan_dispatch(object x):
    """hypercotan: 실수 → cpdef double hypercotan, 복소수 → 자체 _hypercotan_complex"""
    if type(x) is complex:
        return _hypercotan_complex(<double complex>x)
    return hypercotan(<double>x)
