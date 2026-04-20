# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# argument_reduction.pyx
#
# musl __rem_pio2.c + __rem_pio2_large.c 포팅
# Cody-Waite 3단계 π/2 환산 + Payne-Hanek 완전 구현
#
# 참조: https://git.musl-libc.org/cgit/musl/tree/src/math/__rem_pio2.c
#       https://git.musl-libc.org/cgit/musl/tree/src/math/__rem_pio2_large.c

from libc.math cimport fabs
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
from ._helpers cimport high_word, low_word, double_to_bits, bits_to_double

# ------------------------------------------------------------------ musl __rem_pio2_large C 구현 인라인
cdef extern from *:
    """
/* ============================================================
   musl __rem_pio2_large.c 완전 인라인 포팅
   원본 저작권: Copyright (C) 1993 by Sun Microsystems
   musl 라이선스: MIT License
   ============================================================ */

static const int ipio2[] = {
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62,
    0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A,
    0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41,
    0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8,
    0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
    0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08,
    0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880,
    0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,
};

static const double PIo2[] = {
    1.57079625129699707031e+00,
    7.54978941586159635335e-08,
    5.39030252995776476554e-15,
    3.28200341580791294836e-22,
    1.27065575308067607349e-29,
    1.22933308981111328932e-36,
    2.73370053816464559624e-44,
    2.16741683877804819444e-51,
};

static const double
TWO24b  = 1.67772160000000000000e+07,
TWON24b = 5.96046447753906250000e-08;

/*
 * __rem_pio2_large(x, y, e0, nx, prec)
 * double x[],y[];
 * int e0, nx, prec;
 *
 * 반환값: n (quadrant), y[0]+y[1] = x - n*pi/2
 */
static int __rem_pio2_large_impl(double *x, double *y, int e0, int nx, int prec)
{
    int jz, jx, jv, jp, jk, carry, n, iq[20], i, j, k, m, q0, ih;
    double z, fw, f[20], fq[20], q[20];

    /* initialize jk */
    jk = 4; /* double precision */
    jp = jk;

    /* determine jx, jv, q0, note that 3>q0 */
    jx = nx - 1;
    jv = (e0 - 3) / 24;
    if (jv < 0) jv = 0;
    q0 = e0 - 24 * (jv + 1);

    /* set up f[0] to f[jx+jk] where f[jx+jk] = ipio2[jv+jk] */
    j = jv - jx;
    m = jx + jk;
    for (i = 0; i <= m; i++, j++)
        f[i] = (j < 0) ? 0.0 : (double)ipio2[j];

    /* compute q[0],q[1],...q[jk] */
    for (i = 0; i <= jk; i++) {
        for (j = 0, fw = 0.0; j <= jx; j++)
            fw += x[j] * f[i - j + jx];
        q[i] = fw;
    }

    jz = jk;
recompute:
    /* distill q[] into iq[] reversingly */
    for (i = 0, j = jz, z = q[jz]; j > 0; i++, j--) {
        fw = (double)(int)(TWON24b * z);
        iq[i] = (int)(z - TWO24b * fw);
        z = q[j - 1] + fw;
    }

    /* compute n */
    z = __builtin_scalbn(z, q0);   /* actual value of z */
    z -= 8.0 * floor(z * 0.125);  /* trim to small */
    n = (int)z;
    z -= (double)n;
    ih = 0;
    if (q0 > 0) {   /* need iq[jz-1] to determine n */
        i = (iq[jz - 1] >> (24 - q0));
        n += i;
        iq[jz - 1] -= i << (24 - q0);
        ih = iq[jz - 1] >> (23 - q0);
    } else if (q0 == 0) {
        ih = iq[jz - 1] >> 23;
    } else if (z >= 0.5) {
        ih = 2;
    }

    if (ih > 0) { /* q > 0.5 */
        n += 1;
        carry = 0;
        for (i = 0; i < jz; i++) { /* compute 1-q */
            j = iq[i];
            if (carry == 0) {
                if (j != 0) {
                    carry = 1;
                    iq[i] = 0x1000000 - j;
                }
            } else {
                iq[i] = 0xffffff - j;
            }
        }
        if (q0 > 0) { /* rare case: chance is 1 in 12 */
            switch (q0) {
            case 1:
                iq[jz - 1] &= 0x7fffff;
                break;
            case 2:
                iq[jz - 1] &= 0x3fffff;
                break;
            }
        }
        if (ih == 2) {
            z = 1.0 - z;
            if (carry != 0)
                z -= __builtin_scalbn(1.0, q0);
        }
    }

    /* check if recomputation is needed */
    if (z == 0.0) {
        j = 0;
        for (i = jz - 1; i >= jk; i--)
            j |= iq[i];
        if (j == 0) { /* need recomputation */
            for (k = 1; iq[jk - k] == 0; k++); /* k = no. of terms needed */
            for (i = jz + 1; i <= jz + k; i++) { /* add q[jz+1] to q[jz+k] */
                f[jx + i] = (jv + i < 66) ? (double)ipio2[jv + i] : 0.0;
                for (j = 0, fw = 0.0; j <= jx; j++)
                    fw += x[j] * f[jx + i - j];
                q[i] = fw;
            }
            jz += k;
            goto recompute;
        }
    }

    /* chop off zero terms */
    if (z == 0.0) {
        jz--;
        q0 -= 24;
        while (iq[jz] == 0) {
            jz--;
            q0 -= 24;
        }
    } else { /* break z into 24-bit if necessary */
        z = __builtin_scalbn(z, -q0);
        if (z >= TWO24b) {
            fw = (double)(int)(TWON24b * z);
            iq[jz] = (int)(z - TWO24b * fw);
            jz++;
            q0 += 24;
            iq[jz] = (int)fw;
        } else {
            iq[jz] = (int)z;
        }
    }

    /* convert integer "bit" chunk to floating-point value */
    fw = __builtin_scalbn(1.0, q0);
    for (i = jz; i >= 0; i--) {
        q[i] = fw * (double)iq[i];
        fw *= TWON24b;
    }

    /* compute PIo2[0,...,jp]*q[jz,...,0] */
    for (i = jz; i >= 0; i--) {
        for (fw = 0.0, k = 0; k <= jp && k <= jz - i; k++)
            fw += PIo2[k] * q[i + k];
        fq[jz - i] = fw;
    }

    /* compress fq[] into y[] */
    switch (prec) {
    case 0:
        fw = 0.0;
        for (i = jz; i >= 0; i--)
            fw += fq[i];
        /* ih != 0: remainder is negative, flip sign (musl __rem_pio2_large.c) */
        y[0] = (ih == 0) ? fw : -fw;
        break;
    case 1:
    case 2:
        fw = 0.0;
        for (i = jz; i >= 0; i--)
            fw += fq[i];
        /* ih != 0: remainder is negative, flip sign */
        y[0] = (ih == 0) ? fw : -fw;
        fw = fq[0] - fw;
        for (i = 1; i <= jz; i++)
            fw += fq[i];
        y[1] = (ih == 0) ? fw : -fw;
        break;
    case 3:
        for (i = jz; i > 0; i--) {
            fw = fq[i - 1] + fq[i];
            fq[i] += fq[i - 1] - fw;
            fq[i - 1] = fw;
        }
        for (i = jz; i > 1; i--) {
            fw = fq[i - 1] + fq[i];
            fq[i] += fq[i - 1] - fw;
            fq[i - 1] = fw;
        }
        for (fw = 0.0, i = jz; i >= 2; i--)
            fw += fq[i];
        /* ih != 0: remainder is negative, flip sign */
        y[0] = (ih == 0) ? fq[0] : -fq[0];
        y[1] = (ih == 0) ? fq[1] : -fq[1];
        y[2] = (ih == 0) ? fw : -fw;
        break;
    }
    return n & 3;
}
    """
    int __rem_pio2_large_impl(double *x, double *y, int e0, int nx, int prec) noexcept nogil


# ------------------------------------------------------------------ 상수
# Cody-Waite π/2 분할 상수 (musl __rem_pio2.c)
cdef double PIO2_1  = 1.57079632673412561417e+00   # 0x3FF921FB54400000
cdef double PIO2_1T = 6.07710050650619224932e-11   # 0x3DD0B4611A626331
cdef double PIO2_2  = 6.07710050630396597660e-11   # 0x3DD0B4611A600000
cdef double PIO2_2T = 2.02226624879595063154e-21   # 0x3BA3198A2E037073
cdef double PIO2_3  = 2.02226624871116645580e-21   # 0x3BA3198A2E000000
cdef double PIO2_3T = 8.47842766036889956997e-32   # 0x397B839A252049C1

cdef double INV_PIO2 = 6.36619772367581382433e-01  # 2/pi


# ------------------------------------------------------------------ rem_pio2
cdef int rem_pio2(double x, double* y) noexcept nogil:
    """
    musl __rem_pio2 포팅: x를 [-π/4, π/4]로 환산하고 quadrant 번호 반환.

    반환값: n (quadrant 0~3, n & 3 로 사용)
    y[0], y[1]: 환산된 값 (y[0] + y[1] ≈ x - n*π/2)

    범위:
    - |x| < 2^20*π/2: Cody-Waite 3단계
    - |x| >= 2^20*π/2: musl __rem_pio2_large 완전 포팅
    """
    cdef uint32_t ix, hx, high_y0
    cdef int n, j, i
    cdef double fn, r, w, t
    cdef double ax

    ax = fabs(x)
    ix = high_word(ax) & 0x7FFFFFFFU
    j = ix >> 20   # 지수 필드

    # |x| <= pi/4: 직접 반환 (caller에서 처리하지만 여기서도 처리)
    if ix <= 0x3FE921FBU:
        y[0] = x
        y[1] = 0.0
        return 0

    # |x| < 2^20 * pi/2 → Cody-Waite 최대 3단계
    if ix < 0x413921FBU:
        # fn = round(|x| * 2/pi), 부호는 x에서 가져옴
        fn = x * INV_PIO2 + 6755399441055744.0
        fn = fn - 6755399441055744.0
        n = <int>fn

        # 1단계: y[0] = x - n*PIO2_1
        r = x - fn * PIO2_1
        w = fn * PIO2_1T           # PIO2_1 하위 보정
        y[0] = r - w
        y[1] = (r - y[0]) - w     # 임시 y[1]

        # 정밀도 체크: 지수 비트가 충분히 작아야 2단계 불필요
        high_y0 = high_word(y[0]) & 0x7FFFFFFFU
        i = j - (<int>((high_y0 >> 20) & 0x7FFU))
        if i > 16:
            # 2단계: y[0] = (y[0] part) - n*PIO2_2
            t = r
            w = fn * PIO2_2
            r = t - w
            w = fn * PIO2_2T - ((t - r) - w)
            y[0] = r - w
            y[1] = (r - y[0]) - w

            high_y0 = high_word(y[0]) & 0x7FFFFFFFU
            i = j - (<int>((high_y0 >> 20) & 0x7FFU))
            if i > 49:
                # 3단계
                t = r
                w = fn * PIO2_3
                r = t - w
                w = fn * PIO2_3T - ((t - r) - w)
                y[0] = r - w
                y[1] = (r - y[0]) - w

        return n

    # |x| >= 2^20*pi/2: Payne-Hanek (__rem_pio2_large 완전 포팅)
    cdef int e0
    cdef int nx_val
    cdef double tx[3]
    cdef double ty[2]

    # e0: 지수 - 1046
    e0 = <int>((ix >> 20) - 1046)

    # x를 24비트 정수 청크로 분해
    cdef double zz = ax * bits_to_double(<uint64_t>(<int64_t>(0x3FF - e0) << 52))  # ax * 2^(-e0)
    tx[0] = <double>(<int>zz)
    zz = (zz - tx[0]) * 16777216.0   # * 2^24
    tx[1] = <double>(<int>zz)
    zz = (zz - tx[1]) * 16777216.0
    tx[2] = <double>(<int>zz)

    nx_val = 3
    if tx[2] == 0.0:
        nx_val = 2
        if tx[1] == 0.0:
            nx_val = 1

    ty[0] = 0.0
    ty[1] = 0.0

    n = __rem_pio2_large_impl(tx, ty, e0, nx_val, 1)

    if x < 0.0:
        y[0] = -ty[0]
        y[1] = -ty[1]
        return -n
    else:
        y[0] = ty[0]
        y[1] = ty[1]
        return n
