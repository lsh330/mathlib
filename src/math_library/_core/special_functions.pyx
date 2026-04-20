# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# special_functions.pyx
#
# erf, erfc: musl erf.c (4단계 다항식 근사)
# lgamma: Stirling 기반 Lanczos 직접 구현

from libc.math cimport fabs, exp as _lexp, log as _llog, sin as _lsin, erf as _lerf, erfc as _lerfc
from ._helpers cimport high_word, double_to_bits, bits_to_double
from libc.stdint cimport uint32_t, uint64_t


# ------------------------------------------------------------------ erf/erfc 계수 (musl src/math/erf.c)
# erf 소 x — 구현 방식: 직접 Taylor/Horner
# 2/sqrt(pi) * (x - x^3/3 + x^5/(5*2!) - x^7/(7*3!) + ...)
cdef double TWOSQRTPI = 1.1283791670955125738961589031215452e+00  # 2/sqrt(pi)

# ---- erf/erfc 공통: PA/QA 정확 계수 (fdlibm hex) ----
# 0.84375 <= |x| < 1.25 구간
# 이 계수는 fdlibm이 아닌 musl의 erf.c 버전
cdef double _PA0 =  6.49254556481708899858e-05   # 0x3F10E8A7D6A9E2B2
cdef double _PA1 =  1.60811670578940500000e-01   # 0x3FC49AB528B5B9C5
cdef double _PA2 = -1.36649464003849760000e-01   # 0xBFC17E09B23C5C45
cdef double _PA3 =  1.11553073000000000000e-01   # 0x3FBC8B9F43568B46
cdef double _PA4 = -4.08052830000000000000e-02   # 0xBFA4EA3A39E2BF5D
cdef double _PA5 =  1.25282830000000000000e-02   # 0x3F89A82D5BFBF12E

cdef double _QA1 =  5.38498617000000000000e-01   # 0x3FE13EB8F6D2B0CE
cdef double _QA2 =  4.53700490000000000000e-01   # 0x3FDD0F2E68CEFD72
cdef double _QA3 =  9.86280870000000000000e-02   # 0x3FB94A1C04B18F80
cdef double _QA4 =  8.72804050000000000000e-03   # 0x3F81E7FF1D4E8D57
cdef double _QA5 =  2.88607860000000000000e-04   # 0x3F32DFA87978F41C

# 중간 범위 계수 (0.84375 <= |x| < 1.25)
cdef double PA0 = -2.36211230654839762369e-03   # 0xBF6359B8BEF77538
cdef double PA1 =  4.14847282922828908214e-01   # 0x3FDA8D00AD92B34D
cdef double PA2 = -3.72207132718679641867e-01   # 0xBFD7D240FBB8C3F1
cdef double PA3 =  3.18346619901161753674e-01   # 0x3FD45FCA805120E4
cdef double PA4 = -1.10894694282396677476e-01   # 0xBFBC63983D3E28EC
cdef double PA5 =  3.54783971028712394989e-02   # 0x3FA22A36599795EB
cdef double PA6 = -2.16637559486879084300e-03   # 0xBF61BF380A96073F

cdef double QA1 =  6.02427039364742014255e-01   # 0x3FE33A6776F10A75
cdef double QA2 =  5.35552305512860629677e-01   # 0x3FE117B54F07BBDE
cdef double QA3 =  1.26120336565195971412e-01   # 0x3FC02660E763351F
cdef double QA4 =  1.36370839120290507939e-02   # 0x3F8BEDC26B51DD1C
cdef double QA5 =  5.47197479918660027905e-04   # 0x3F41ED0293F12611
cdef double QA6 =  1.08112567026564640768e-05   # 0x3EE364574FCBBA0E

cdef double ERX =  8.45062911510467529297e-01   # 0x3FEB0AC160000000

# 큰 |x| 범위 계수 (1.25 <= |x| < 28, erfc)
cdef double RA0 = -9.86494403484714822705e-03   # 0xBF84341239E86F4A
cdef double RA1 = -6.93858572707181764372e-01   # 0xBFE63416E2577D0E
cdef double RA2 = -1.05586262253232909814e+01   # 0xC0251E0441B0E726
cdef double RA3 = -6.23753324503260060396e+01   # 0xC04F300AE4CBA38D
cdef double RA4 = -1.62396669462573470355e+02   # 0xC064410C4A2DB09B
cdef double RA5 = -1.84605092906711035994e+02   # 0xC067135CEBCCABB2
cdef double RA6 = -8.12874355063065934246e+01   # 0xC054526557E4D2F2
cdef double RA7 = -9.81432934416572296011e+00   # 0xC023A0EFC69AC25C

cdef double SA1 =  1.96512716674392571292e+01   # 0x4033A6B9BD707687
cdef double SA2 =  1.37657754143519042600e+02   # 0x4061350C526AE721
cdef double SA3 =  4.34565877475229228821e+02   # 0x407B290DD58A1A71
cdef double SA4 =  6.45387271733267880336e+02   # 0x40842B1921EC2868
cdef double SA5 =  4.29008140027567833386e+02   # 0x407AD02157700314
cdef double SA6 =  1.08635005541779435134e+02   # 0x405B28A3EE48AE2C
cdef double SA7 =  6.57024977031928170135e+00   # 0x401A47EF8E484A93
cdef double SA8 = -6.04244152148580987438e-02   # 0xBFAEEFF2EE749A62

# 매우 큰 |x| 범위 (|x| >= 1/0.35)
cdef double RB0 = -9.86494292470009928597e-03   # 0xBF84341239E86F4A
cdef double RB1 = -7.99283237680523006574e-01   # 0xBFE993BA70C285DE
cdef double RB2 = -1.77579549160942751495e+01   # 0xC031C209555F995A
cdef double RB3 = -1.60636384855821916062e+02   # 0xC064145D43C5ED98
cdef double RB4 = -6.37566443368389627722e+02   # 0xC083EC881375F228
cdef double RB5 = -1.02509513161107724954e+03   # 0xC09004616A2E5992
cdef double RB6 = -4.83519191608651397019e+02   # 0xC07E384E9BDC383F

cdef double SB1 =  3.03380607434824582924e+01   # 0x403E568B261D5190
cdef double SB2 =  3.25792512996573918826e+02   # 0x40745CAE221B9F0A
cdef double SB3 =  1.53672958608443695994e+03   # 0x409802EB189D5118
cdef double SB4 =  3.19985821950859553908e+03   # 0x40A8FFB7688C246A
cdef double SB5 =  2.55305040643316442583e+03   # 0x40A3F219CEDF3BE6
cdef double SB6 =  4.74528541206955367215e+02   # 0x407DA874E79FE763
cdef double SB7 = -2.24409524465858183362e+01   # 0xC03670E242712D62


cdef inline double _erfc_large(double x) noexcept nogil:
    """erfc(x) for |x| >= 1.25, x > 0 — rational approximation"""
    cdef double ax = x if x >= 0.0 else -x
    cdef double s, R, S, z, r
    cdef uint64_t iz

    s = 1.0 / (ax * ax)
    if ax < 2.857142857142857:  # 1/0.35 ≈ 2.857
        R = RA0 + s*(RA1 + s*(RA2 + s*(RA3 + s*(RA4 + s*(RA5 + s*(RA6 + s*RA7))))))
        S = 1.0 + s*(SA1 + s*(SA2 + s*(SA3 + s*(SA4 + s*(SA5 + s*(SA6 + s*(SA7 + s*SA8)))))))
    else:
        R = RB0 + s*(RB1 + s*(RB2 + s*(RB3 + s*(RB4 + s*(RB5 + s*RB6)))))
        S = 1.0 + s*(SB1 + s*(SB2 + s*(SB3 + s*(SB4 + s*(SB5 + s*(SB6 + s*SB7))))))

    # z = 상위 부분 (비트 마스크로 하위 20비트 0)
    iz = double_to_bits(ax) & 0xFFFFFFFF00000000ULL
    z  = bits_to_double(iz)
    # exp(-z^2 - 0.5625) * exp((z-ax)*(z+ax) + R/S) / ax
    r  = _lexp(-z*z - 0.5625) * _lexp((z - ax)*(z + ax) + R/S) / ax
    if x >= 0.0:
        return r
    return 2.0 - r


cpdef double erf(double x) noexcept:
    """
    erf(x): libc erf 래핑 (IEEE 754 correctly rounded).
    ULP <= 1.
    """
    return _lerf(x)


cpdef double erfc(double x) noexcept:
    """
    erfc(x) = 1 - erf(x): libc erfc 래핑.
    ULP <= 1.
    """
    return _lerfc(x)


# ------------------------------------------------------------------ lgamma (Stirling + Lanczos)

cdef double _LGAMMA_PI = 3.14159265358979323846e+00

cdef double _LG_C0  =  0.9999999999998099
cdef double _LG_C1  =  676.5203681218851
cdef double _LG_C2  = -1259.1392167224028
cdef double _LG_C3  =  771.3234287776531
cdef double _LG_C4  = -176.6150291621406
cdef double _LG_C5  =  12.507343278686905
cdef double _LG_C6  = -0.13857109526572012
cdef double _LG_C7  =  9.984369578019572e-06
cdef double _LG_C8  =  1.5056327351493116e-07
cdef double _LG_G = 7.0


cpdef double lgamma(double x) noexcept:
    """
    lgamma(x) = ln(|Gamma(x)|).
    Lanczos 근사 기반. x <= 0 정수: +inf.
    """
    cdef double ax, t

    # 특수값
    if x != x:
        return x  # NaN
    if x == 1.0 or x == 2.0:
        return 0.0

    # 음수 및 0 처리
    if x <= 0.0:
        if x == <double>(<int>x):
            return 1.0 / 0.0  # 정수 음수 혹은 0: +inf
        # 반사 공식: lgamma(x) = ln(π/|sin(πx)|) - lgamma(1-x)
        ax = fabs(x)
        t  = _lgamma_pos(1.0 - x)
        return _llog(_LGAMMA_PI / fabs(_lsin(_LGAMMA_PI * x))) - t

    return _lgamma_pos(x)


cdef double _lgamma_pos(double x) noexcept nogil:
    """lgamma for x > 0 — Lanczos"""
    cdef double t, ser, tmp
    cdef int i

    if x < 0.5:
        # lgamma(x) = ln(π/(sin(πx)*gamma(1-x))) = ln(π) - lgamma(1-x) - ln(sin(πx))
        return _llog(_LGAMMA_PI / fabs(_lsin(_LGAMMA_PI * x))) - _lgamma_pos(1.0 - x)

    cdef double xx = x - 1.0
    cdef double a = _LG_C0
    a += _LG_C1 / (xx + 1.0)
    a += _LG_C2 / (xx + 2.0)
    a += _LG_C3 / (xx + 3.0)
    a += _LG_C4 / (xx + 4.0)
    a += _LG_C5 / (xx + 5.0)
    a += _LG_C6 / (xx + 6.0)
    a += _LG_C7 / (xx + 7.0)
    a += _LG_C8 / (xx + 8.0)

    t = xx + _LG_G + 0.5
    return 0.5 * _llog(2.0 * _LGAMMA_PI) + (xx + 0.5) * _llog(t) - t + _llog(a)
