# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# logarithmic.pyx — M1: 128-entry table-based ln (musl log.c style)
# B2: negative input -> NaN  B3: subnormal lx reload (2^52 scale)
#
# 참조: algorithm_reference.md 섹션 11

from libc.math cimport fma, fabs
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
from ._helpers cimport high_word, low_word, double_to_bits, bits_to_double

import cmath as _cmath


cdef extern from *:
    """
#include <stdint.h>
#include <string.h>

#define LOG_TABLE_BITS 7
#define LOG_N 128
#define LOG_OFF 0x3fe6000000000000ULL

static const struct { double invc, logc, logctail; } LOG_T[128] = {
    {0x1.734f0c541fe8dp+0, -0x1.7cc7f7db46a0ep-2, 0x0p+0},
    {0x1.713786d9c7c09p+0, -0x1.76feecb947175p-2, -0x1.0000000000000p-54},
    {0x1.6f26016f26017p+0, -0x1.713e33a46a17cp-2, 0x0p+0},
    {0x1.6d1a62681c861p+0, -0x1.6b85b4cffa3fdp-2, 0x0p+0},
    {0x1.6b1490aa31a3dp+0, -0x1.65d558d4ce00bp-2, 0x0p+0},
    {0x1.691473a88d0c0p+0, -0x1.602d08af091ecp-2, 0x0p+0},
    {0x1.6719f3601671ap+0, -0x1.5a8cadbbedfa1p-2, 0x0p+0},
    {0x1.6524f853b4aa3p+0, -0x1.54f431b7be1a9p-2, 0x1.0000000000000p-54},
    {0x1.63356b88ac0dep+0, -0x1.4f637ebba9810p-2, 0x0p+0},
    {0x1.614b36831ae94p+0, -0x1.49da7f3bcc41fp-2, -0x1.0000000000000p-54},
    {0x1.5f66434292dfcp+0, -0x1.44591e0539f49p-2, 0x0p+0},
    {0x1.5d867c3ece2a5p+0, -0x1.3edf463c1683ep-2, 0x0p+0},
    {0x1.5babcc647fa91p+0, -0x1.396ce359bbf54p-2, 0x1.0000000000000p-54},
    {0x1.59d61f123ccaap+0, -0x1.3401e12aecba1p-2, 0x1.0000000000000p-54},
    {0x1.5805601580560p+0, -0x1.2e9e2bce12286p-2, 0x0p+0},
    {0x1.56397ba7c52e2p+0, -0x1.2941afb186b7cp-2, 0x0p+0},
    {0x1.54725e6bb82fep+0, -0x1.23ec5991eba49p-2, 0x0p+0},
    {0x1.52aff56a8054bp+0, -0x1.1e9e1678899f4p-2, -0x1.0000000000000p-54},
    {0x1.50f22e111c4c5p+0, -0x1.1956d3b9bc2fap-2, 0x1.0000000000000p-54},
    {0x1.4f38f62dd4c9bp+0, -0x1.14167ef367783p-2, -0x1.0000000000000p-54},
    {0x1.4d843bedc2c4cp+0, -0x1.0edd060b78081p-2, -0x1.0000000000000p-54},
    {0x1.4bd3edda68fe1p+0, -0x1.09aa572e6c6d4p-2, 0x0p+0},
    {0x1.4a27fad76014ap+0, -0x1.047e60cde83b8p-2, 0x1.0000000000000p-54},
    {0x1.4880522014880p+0, -0x1.feb2233ea07cdp-3, 0x1.0000000000000p-54},
    {0x1.46dce34596066p+0, -0x1.f474b134df229p-3, 0x1.0000000000000p-55},
    {0x1.453d9e2c776cap+0, -0x1.ea4449f04aaf5p-3, 0x0p+0},
    {0x1.43a2730abee4dp+0, -0x1.e020cc6235ab5p-3, 0x0p+0},
    {0x1.420b5265e5951p+0, -0x1.d60a17f903515p-3, 0x1.0000000000000p-55},
    {0x1.40782d10e6566p+0, -0x1.cc000c9db3c52p-3, 0x0p+0},
    {0x1.3ee8f42a5af07p+0, -0x1.c2028ab17f9b4p-3, -0x1.0000000000000p-55},
    {0x1.3d5d991aa75c6p+0, -0x1.b811730b823d2p-3, -0x1.0000000000000p-54},
    {0x1.3bd60d9232955p+0, -0x1.ae2ca6f672bd4p-3, -0x1.0000000000000p-53},
    {0x1.3a524387ac822p+0, -0x1.a454082e6ab05p-3, 0x1.0000000000000p-54},
    {0x1.38d22d366088ep+0, -0x1.9a8778debaa38p-3, -0x1.0000000000000p-54},
    {0x1.3755bd1c945eep+0, -0x1.90c6db9fcbcd9p-3, -0x1.0000000000000p-54},
    {0x1.35dce5f9f2af8p+0, -0x1.871213750e994p-3, 0x0p+0},
    {0x1.34679ace01346p+0, -0x1.7d6903caf5ad0p-3, 0x1.8000000000000p-54},
    {0x1.32f5ced6a1dfap+0, -0x1.73cb9074fd14dp-3, 0x0p+0},
    {0x1.3187758e9ebb6p+0, -0x1.6a399dabbd383p-3, 0x0p+0},
    {0x1.301c82ac40260p+0, -0x1.60b3100b09476p-3, 0x1.0000000000000p-54},
    {0x1.2eb4ea1fed14bp+0, -0x1.5737cc9018cddp-3, 0x0p+0},
    {0x1.2d50a012d50a0p+0, -0x1.4dc7b897bc1c8p-3, 0x1.0000000000000p-55},
    {0x1.2bef98e5a3711p+0, -0x1.4462b9dc9b3dcp-3, 0x0p+0},
    {0x1.2a91c92f3c105p+0, -0x1.3b08b6757f2a9p-3, 0x1.0000000000000p-54},
    {0x1.293725bb804a5p+0, -0x1.31b994d3a4f85p-3, -0x1.0000000000000p-55},
    {0x1.27dfa38a1ce4dp+0, -0x1.28753bc11aba5p-3, 0x1.8000000000000p-54},
    {0x1.268b37cd60127p+0, -0x1.1f3b925f25d41p-3, -0x1.8000000000000p-54},
    {0x1.2539d7e9177b2p+0, -0x1.160c8024b27b1p-3, 0x1.0000000000000p-55},
    {0x1.23eb79717605bp+0, -0x1.0ce7ecdccc28dp-3, 0x1.0000000000000p-54},
    {0x1.22a0122a0122ap+0, -0x1.03cdc0a51ec0dp-3, 0x0p+0},
    {0x1.21579804855e6p+0, -0x1.f57bc7d9005dbp-4, 0x0p+0},
    {0x1.2012012012012p+0, -0x1.e3707ee30487bp-4, 0x0p+0},
    {0x1.1ecf43c7fb84cp+0, -0x1.d179788219364p-4, 0x1.0000000000000p-55},
    {0x1.1d8f5672e4abdp+0, -0x1.bf968769fca11p-4, -0x1.c000000000000p-54},
    {0x1.1c522fc1ce059p+0, -0x1.adc77ee5aea8cp-4, -0x1.0000000000000p-55},
    {0x1.1b17c67f2bae3p+0, -0x1.9c0c32d4d2548p-4, -0x1.4000000000000p-54},
    {0x1.19e0119e0119ep+0, -0x1.8a6477a91dc29p-4, 0x0p+0},
    {0x1.18ab083902bdbp+0, -0x1.78d02263d82d3p-4, -0x1.0000000000000p-54},
    {0x1.1778a191bd684p+0, -0x1.674f089365a7ap-4, 0x1.0000000000000p-55},
    {0x1.1648d50fc3201p+0, -0x1.55e10050e0384p-4, 0x1.0000000000000p-55},
    {0x1.151b9a3fdd5c9p+0, -0x1.4485e03dbdfadp-4, -0x1.8000000000000p-55},
    {0x1.13f0e8d344724p+0, -0x1.333d7f8183f4bp-4, 0x1.0000000000000p-56},
    {0x1.12c8b89edc0acp+0, -0x1.2207b5c78549ep-4, -0x1.8000000000000p-55},
    {0x1.11a3019a74826p+0, -0x1.10e45b3cae831p-4, 0x1.0000000000000p-53},
    {0x1.107fbbe011080p+0, -0x1.ffa6911ab9301p-5, -0x1.0000000000000p-54},
    {0x1.0f5edfab325a2p+0, -0x1.dda8adc67ee4ep-5, -0x1.6000000000000p-54},
    {0x1.0e40655826011p+0, -0x1.bbcebfc68f420p-5, -0x1.0000000000000p-55},
    {0x1.0d24456359e3ap+0, -0x1.9a187b573de7cp-5, -0x1.4000000000000p-55},
    {0x1.0c0a7868b4171p+0, -0x1.788595a3577bap-5, -0x1.c000000000000p-54},
    {0x1.0af2f722eecb5p+0, -0x1.5715c4c03ceefp-5, 0x1.c000000000000p-54},
    {0x1.09ddba6af8360p+0, -0x1.35c8bfaa1306bp-5, 0x1.0000000000000p-56},
    {0x1.08cabb37565e2p+0, -0x1.149e3e4005a8dp-5, 0x0p+0},
    {0x1.07b9f29b8eae2p+0, -0x1.e72bf2813ce51p-6, -0x1.9000000000000p-54},
    {0x1.06ab59c7912fbp+0, -0x1.a55f548c5c43fp-6, 0x1.8000000000000p-54},
    {0x1.059eea0727586p+0, -0x1.63d6178690bd6p-6, 0x1.8000000000000p-54},
    {0x1.04949cc1664c5p+0, -0x1.228fb1fea2e28p-6, 0x1.e000000000000p-54},
    {0x1.038c6b78247fcp+0, -0x1.c317384c75f06p-7, -0x1.c000000000000p-57},
    {0x1.02864fc7729e9p+0, -0x1.41929f96832f0p-7, -0x1.c000000000000p-55},
    {0x1.0182436517a37p+0, -0x1.8121214586b54p-8, 0x1.4800000000000p-54},
    {0x1.0080402010080p+0, -0x1.0040155d5889ep-9, 0x1.0000000000000p-54},
    {0x1.fe01fe01fe020p-1, 0x1.ff00aa2b10bc0p-9, -0x1.0000000000000p-56},
    {0x1.fa11caa01fa12p-1, 0x1.7dc475f810a77p-7, -0x1.c000000000000p-56},
    {0x1.f6310aca0dbb5p-1, 0x1.3cea44346a575p-6, 0x1.e000000000000p-55},
    {0x1.f25f644230ab5p-1, 0x1.b9fc027af9198p-6, 0x1.0000000000000p-57},
    {0x1.ee9c7f8458e02p-1, 0x1.1b0d98923d980p-5, -0x1.0000000000000p-57},
    {0x1.eae807aba01ebp-1, 0x1.58a5bafc8e4d5p-5, -0x1.0000000000000p-56},
    {0x1.e741aa59750e4p-1, 0x1.95c830ec8e3ebp-5, 0x1.c000000000000p-55},
    {0x1.e3a9179dc1a73p-1, 0x1.d276b8adb0b52p-5, 0x1.0000000000000p-55},
    {0x1.e01e01e01e01ep-1, 0x1.075983598e471p-4, 0x0p+0},
    {0x1.dca01dca01dcap-1, 0x1.253f62f0a1417p-4, 0x0p+0},
    {0x1.d92f2231e7f8ap-1, 0x1.42edcbea646f0p-4, -0x1.0000000000000p-55},
    {0x1.d5cac807572b2p-1, 0x1.60658a93750c4p-4, 0x0p+0},
    {0x1.d272ca3fc5b1ap-1, 0x1.7da766d7b12cdp-4, 0x1.8000000000000p-55},
    {0x1.cf26e5c44bfc6p-1, 0x1.9ab42462033adp-4, 0x1.0000000000000p-56},
    {0x1.cbe6d9601cbe7p-1, 0x1.b78c82bb0eda1p-4, -0x1.0000000000000p-56},
    {0x1.c8b265afb8a42p-1, 0x1.d4313d66cb35dp-4, 0x0p+0},
    {0x1.c5894d10d4986p-1, 0x1.f0a30c01162a6p-4, -0x1.0000000000000p-55},
    {0x1.c26b5392ea01cp-1, 0x1.0671512ca596ep-3, 0x1.0000000000000p-55},
    {0x1.bf583ee868d8bp-1, 0x1.14785846742acp-3, 0x0p+0},
    {0x1.bc4fd65883e7bp-1, 0x1.2266f190a5acbp-3, 0x1.0000000000000p-54},
    {0x1.b951e2b18ff23p-1, 0x1.303d718e47fd3p-3, 0x1.0000000000000p-54},
    {0x1.b65e2e3beee05p-1, 0x1.3dfc2b0ecc62ap-3, 0x0p+0},
    {0x1.b37484ad806cep-1, 0x1.4ba36f39a55e5p-3, 0x0p+0},
    {0x1.b094b31d922a4p-1, 0x1.59338d9982086p-3, -0x1.0000000000000p-55},
    {0x1.adbe87f94905ep-1, 0x1.66acd4272ad51p-3, 0x0p+0},
    {0x1.aaf1d2f87ebfdp-1, 0x1.740f8f54037a5p-3, -0x1.0000000000000p-54},
    {0x1.a82e65130e159p-1, 0x1.815c0a14357ebp-3, -0x1.0000000000000p-54},
    {0x1.a574107688a4ap-1, 0x1.8e928de886d41p-3, 0x0p+0},
    {0x1.a2c2a87c51ca0p-1, 0x1.9bb362e7dfb83p-3, 0x1.0000000000000p-54},
    {0x1.a01a01a01a01ap-1, 0x1.a8becfc882f19p-3, 0x0p+0},
    {0x1.9d79f176b682dp-1, 0x1.b5b519e8fb5a4p-3, 0x1.0000000000000p-54},
    {0x1.9ae24ea5510dap-1, 0x1.c2968558c18c1p-3, 0x1.0000000000000p-55},
    {0x1.9852f0d8ec0ffp-1, 0x1.cf6354e09c5dcp-3, 0x1.0000000000000p-55},
    {0x1.95cbb0be377aep-1, 0x1.dc1bca0abec7dp-3, -0x1.0000000000000p-54},
    {0x1.934c67f9b2ce6p-1, 0x1.e8c0252aa5a60p-3, 0x0p+0},
    {0x1.90d4f120190d5p-1, 0x1.f550a564b7b37p-3, 0x0p+0},
    {0x1.8e6527af1373fp-1, 0x1.00e6c45ad501dp-2, 0x0p+0},
    {0x1.8bfce8062ff3ap-1, 0x1.071b85fcd590dp-2, 0x0p+0},
    {0x1.899c0f601899cp-1, 0x1.0d46b579ab74bp-2, 0x0p+0},
    {0x1.87427bcc092b9p-1, 0x1.136870293a8b0p-2, 0x0p+0},
    {0x1.84f00c2780614p-1, 0x1.1980d2dd4236fp-2, 0x0p+0},
    {0x1.82a4a0182a4a0p-1, 0x1.1f8ff9e48a2f3p-2, 0x0p+0},
    {0x1.8060180601806p-1, 0x1.2596010df763ap-2, 0x0p+0},
    {0x1.7e225515a4f1dp-1, 0x1.2b9303ab89d25p-2, 0x0p+0},
    {0x1.7beb3922e017cp-1, 0x1.31871c9544185p-2, 0x0p+0},
    {0x1.79baa6bb6398bp-1, 0x1.3772662bfd85bp-2, 0x1.0000000000000p-54},
    {0x1.77908119ac60dp-1, 0x1.3d54fa5c1f710p-2, 0x0p+0},
    {0x1.756cac201756dp-1, 0x1.432ef2a04e814p-2, -0x1.0000000000000p-54},
};

/* Polynomial A[] — 5계수, 일반 구간 (algorithm_reference.md §11) */
#define A0 (-0x1.0000000000001p-1)
#define A1  (0x1.555555551305bp-2)
#define A2 (-0x1.fffffffeb459p-3)
#define A3  (0x1.999b324f10111p-3)
#define A4 (-0x1.55575e506c89fp-3)

/* Polynomial B[] — 11계수, 좁은 구간 |r| < 0x1p-4 (algorithm_reference.md §11) */
#define B0  (-0x1p-1)
#define B1   (0x1.5555555555577p-2)
#define B2  (-0x1.ffffffffffdcbp-3)
#define B3   (0x1.999999995dd0cp-3)
#define B4  (-0x1.55555556745a7p-3)
#define B5   (0x1.24924a344de3p-3)
#define B6  (-0x1.fffffa4423d65p-4)
#define B7   (0x1.c7184282ad6cap-4)
#define B8  (-0x1.999eb43b068ffp-4)
#define B9   (0x1.78182f7afd085p-4)
#define B10 (-0x1.5521375d145cdp-4)

/* ln(2) split */
#define LN2HI 0x1.62e42fefa3800p-1
#define LN2LO 0x1.ef35793c76730p-45

/* ln(1+u) for |u| < 0.125 via Horner polynomial on u = x-1.
 * Exact subtraction (Sterbenz: x in [0.875, 1.125] ⊂ [0.5, 2]).
 * Range guarantee: ln_near1 gives <=1 ULP for x in [0.96875, 1.0625).
 */
static inline double _ln_near1(double x)
{
    double u = x - 1.0;
    double p = B0 + u*(B1 + u*(B2 + u*(B3 + u*(B4 + u*(B5 + u*(B6
          + u*(B7 + u*(B8 + u*(B9 + u*B10)))))))));
    return __builtin_fma(u*u, p, u);
}

static double _musl_log(double x)
{
    uint64_t ix, ix2, tmp;
    double z, r, logc, logctail, kd, hi, lo, r2, p;
    int k, i;

    __builtin_memcpy(&ix, &x, sizeof(ix));

    /* log(1.0) = 0 직접 반환 — k=0, r=0 특수 케이스 */
    if (ix == 0x3FF0000000000000ULL)
        return 0.0;

    /* x in [0x3FEF000000000000, 0x3FF1000000000000) = [0.96875, 1.0625)
     * i=79/80 (k=0) 구간: logc + r ≈ 0 catastrophic cancellation 발생
     * 직접 Taylor 경로 사용 (Sterbenz: u=x-1 exact, max ULP=1)
     */
    if ((ix - 0x3FEF000000000000ULL) < 0x0002000000000000ULL)
        return _ln_near1(x);

    tmp = ix - LOG_OFF;
    k   = (int)((int64_t)tmp >> 52);
    ix2 = ix - ((uint64_t)k << 52);
    __builtin_memcpy(&z, &ix2, sizeof(z));

    tmp = ix2 - LOG_OFF;
    i   = (int)(tmp >> (52 - LOG_TABLE_BITS)) & (LOG_N - 1);

    logc     = LOG_T[i].logc;
    logctail = LOG_T[i].logctail;

    r = __builtin_fma(z, LOG_T[i].invc, -1.0);

    kd = (double)k;

    /* musl log.c 원본 hi/lo 분리 — 2Sum 보정 포함
     * t1 = kd*Ln2hi + logc  (kd=0이면 t1 = logc)
     * hi = t1 + r            (cancellation 가능)
     * lo = (t1 - hi) + r     (2Sum: hi+lo = t1+r 정확히 보정)
     * 이후 lo += kd*Ln2lo + logctail + r^2*p(r)
     *
     * 핵심: k==0 시 t1=logc, r≈-logc → hi≈0 으로 cancellation 발생
     *       2Sum(t1,r)로 보정항 lo에 누적 → 정밀도 회복
     */
    {
        double t1, t2;
        t1 = kd * LN2HI + logc;
        t2 = kd * LN2LO + logctail;
        hi = t1 + r;
        lo = t2 + (t1 - hi + r);
    }

    r2 = r * r;

    /* B[] 11계수 Horner 평가 — 전 구간 고정밀도 경로
     * algorithm_reference.md §11: B[] 유효구간 [-0x1p-4, 0x1.09p-4]
     * 128-entry 테이블로 r의 절대값이 항상 이 범위 이내 보장
     */
    p = B0 + r*(B1 + r*(B2 + r*(B3 + r*(B4 + r*(B5 + r*(B6
          + r*(B7 + r*(B8 + r*(B9 + r*B10)))))))));
    lo = __builtin_fma(r2, p, lo);
    return hi + lo;
}
    """
    double _musl_log(double x) noexcept nogil


cdef double _ln_inline(double x) noexcept nogil:
    """
    ln(x) 테이블 기반 구현. ULP <= 4, 평균 <= 1.
    B2: x<0 -> NaN | B3: subnormal -> 2^52 scale
    """
    cdef uint32_t hx, lx
    hx = high_word(x)
    lx = low_word(x)

    # B2: negative
    if hx >> 31:
        return (x - x) / (x - x)
    # zero
    if hx == 0U and lx == 0U:
        return -1.0 / (x * x)
    # inf/NaN
    if hx >= 0x7FF00000U:
        return x
    # B3: subnormal — scale by 2^52, correct k afterwards
    if hx < 0x00100000U:
        return _musl_log(x * 4503599627370496.0) - 52.0 * 6.9314718055994530941723212e-01
    return _musl_log(x)


cpdef double ln(double x) noexcept:
    """ln(x) — table-based, <= 4 ULP."""
    return _ln_inline(x)


cpdef double log(double base, double x) noexcept:
    """log_base(x). Special: base<=0, base==1, x<=0 -> NaN."""
    if x <= 0.0 or base <= 0.0 or base == 1.0:
        return (x - x) / (x - x)
    cdef double ln_x    = _ln_inline(x)
    cdef double ln_base = _ln_inline(base)
    if ln_base == 0.0:
        return (x - x) / (x - x)
    return ln_x / ln_base


# ------------------------------------------------------------------ 복소수 auto-dispatch (방향 A)
cpdef object ln_dispatch(object x):
    """ln: 실수 → cpdef double ln, 복소수 → cmath.log"""
    if type(x) is complex:
        return _cmath.log(x)
    return _ln_inline(<double>x)

cpdef object log_dispatch(object base, object x):
    """log_base(x): 실수 → cpdef double log, 복소수 → cmath.log(x)/cmath.log(base)"""
    if type(base) is complex or type(x) is complex:
        return _cmath.log(complex(x)) / _cmath.log(complex(base))
    return log(<double>base, <double>x)
