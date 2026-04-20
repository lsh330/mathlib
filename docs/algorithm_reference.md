# Elementary Function Algorithm Reference

> 출처: musl-1.2.x (git.musl-libc.org/cgit/musl), fdlibm (FreeBSD msun),
> CORE-MATH (INRIA), Muller "Elementary Functions" 3rd ed. 2016
> 작성 목적: Cython 재구현 시 계수를 그대로 인용할 수 있는 유일한 참조 문서
> 모든 계수는 IEEE 754 double-precision hex literal 또는 bit-exact decimal로 표기

---

## 1. sin(x)

### Argument Reduction

- 전략: `__rem_pio2` 호출로 x를 `[-π/4, π/4]`로 환산 후 quadrant에 따라 분기
- 임계값: `|x| ≤ 0x3fe921fb` (≈ π/4)이면 바로 `__sin(x, 0, 0)` 호출
- `|x| < 0x3e400000` (≈ 2^-27)이면 x를 그대로 반환 (inexact 예외 없음)
- 큰 인수: `ix < 0x413921fb` (≈ 2^20·π/2) → Cody-Waite 3단계, 그 이상 → Payne-Hanek

#### Cody-Waite π/2 분할 상수 (musl `__rem_pio2.c`)

| 상수 | Hex | Decimal |
|---|---|---|
| `pio2_1`  | `0x3FF921FB54400000` | 1.57079632673412561417e+00 |
| `pio2_1t` | `0x3DD0B4611A626331` | 6.07710050650619224932e-11 |
| `pio2_2`  | `0x3DD0B4611A600000` | 6.07710050630396597660e-11 |
| `pio2_2t` | `0x3BA3198A2E037073` | 2.02226624879595063154e-21 |
| `pio2_3`  | `0x3BA3198A2E000000` | 2.02226624871116645580e-21 |
| `pio2_3t` | `0x397B839A252049C1` | 8.47842766036889956997e-32 |

#### quadrant 분기 (n = `__rem_pio2(x, y)` 반환값)

| n & 3 | 결과 |
|---|---|
| 0 | `+__sin(y[0], y[1], 1)` |
| 1 | `+__cos(y[0], y[1])` |
| 2 | `-__sin(y[0], y[1], 1)` |
| 3 | `-__cos(y[0], y[1])` |

### Polynomial Approximation (`__sin.c`)

- 유효 구간: `[-π/4, π/4]`
- 방법: Remez minimax, degree-13 (홀수 항만 존재 → 실질 degree-7 다항식)
- 근사식: `sin(x) ≈ x + x³·(S1 + z·(S2 + z·(S3 + z·S4))) + x³·z²·(S5 + z·S6)` (z = x²)
- Horner 평가: `r = S2 + z*(S3 + z*S4) + z*w*(S5 + z*S6)`, 결과 = `x + v*(S1 + z*r)` (v = x²·x)

| 계수 | IEEE 754 Hex | Bit-exact Decimal |
|---|---|---|
| S1 | `0xBFC5555555555549` | -1.66666666666666324348e-01 |
| S2 | `0x3F8111111110F8A6` | 8.33333333332248946124e-03 |
| S3 | `0xBF2A01A019C161D5` | -1.98412698298579493134e-04 |
| S4 | `0x3EC71DE357B1FE7D` | 2.75573137070700676789e-06 |
| S5 | `0xBE5AE5E68A2B9CEB` | -2.50507602534068634195e-08 |
| S6 | `0x3DE5D93A5ACFD57C` | 1.58969099521155010221e-10 |

출처: musl `src/math/__sin.c`, 원 저작권 Sun Microsystems / FreeBSD msun

### Special Cases

| 입력 | 결과 |
|---|---|
| `sin(NaN)` | NaN |
| `sin(±0)` | ±0 (부호 보존) |
| `sin(±∞)` | NaN (invalid operation 예외) |
| 비정규수 (subnormal) | x 그대로 반환 (|x| < 2^-26) |

### ULP Accuracy

- 최대 1 ULP (musl 기준)
- Correctly rounded: CORE-MATH `binary64/sin` 참조

### Reference

- `musl src/math/sin.c`, `src/math/__sin.c`, `src/math/__rem_pio2.c`
- Muller Ch.5 (삼각함수 argument reduction)

---

## 2. cos(x)

### Argument Reduction

- sin(x)와 동일한 `__rem_pio2` 전략 사용

#### quadrant 분기

| n & 3 | 결과 |
|---|---|
| 0 | `+__cos(y[0], y[1])` |
| 1 | `-__sin(y[0], y[1], 1)` |
| 2 | `-__cos(y[0], y[1])` |
| 3 | `+__sin(y[0], y[1], 1)` |

### Polynomial Approximation (`__cos.c`)

- 유효 구간: `[-π/4, π/4]`
- 방법: Remez minimax, degree-14 (짝수 항만 → 실질 degree-7)
- 근사식: `cos(x) ≈ 1 - x²/2 + x⁴·(C1 + z·(C2 + z·C3)) + (x²)²·(x²)·(C4 + z·(C5 + z·C6))` (z = x², w = z²)
- Horner 평가: `r = z*(C1+z*(C2+z*C3)) + w*w*(C4+z*(C5+z*C6))`
- y 보정 포함 최종식: `w + (((1.0-w)-hz) + (z*r - x*y))` (hz = 0.5·x²)

| 계수 | IEEE 754 Hex | Bit-exact Decimal |
|---|---|---|
| C1 | `0x3FA555555555554C` | 4.16666666666666019037e-02 |
| C2 | `0xBF56C16C16C15177` | -1.38888888888741095749e-03 |
| C3 | `0x3EFA01A019CB1590` | 2.48015872894767294178e-05 |
| C4 | `0xBE927E4F809C52AD` | -2.75573143513906633035e-07 |
| C5 | `0x3E21EE9EBDB4B1C4` | 2.08757232129817482790e-09 |
| C6 | `0xBDA8FAE9BE8838D4` | -1.13596475577881948265e-11 |

출처: musl `src/math/__cos.c`

### Special Cases

| 입력 | 결과 |
|---|---|
| `cos(NaN)` | NaN |
| `cos(±0)` | 1.0 |
| `cos(±∞)` | NaN (invalid operation 예외) |

### ULP Accuracy

- 최대 1 ULP

### Reference

- `musl src/math/cos.c`, `src/math/__cos.c`

---

## 3. tan(x)

### Argument Reduction

- `__rem_pio2` 호출로 `[-π/4, π/4]`로 환산
- n이 짝수이면 `__tan(y[0], y[1], 0)`, 홀수이면 `__tan(y[0], y[1], 1)` (두 번째 인자는 `-1/tan` 전환 여부)
- `|x| < 2^-27`이면 x 그대로 반환

### Polynomial Approximation (`__tan.c`)

- 유효 구간: `[0, 0.67434]` (≈ π/4)
- 방법: Remez minimax, degree-25 (홀수 항)
- 13개 계수 T[0]~T[12]

| 인덱스 | IEEE 754 Hex | Bit-exact Decimal |
|---|---|---|
| T[0] | `0x3FD5555555555563` | 3.33333333333334091986e-01 |
| T[1] | `0x3FC111111110FE7A` | 1.33333333333201242699e-01 |
| T[2] | `0x3FABA1BA1BB341FE` | 5.39682539762260521377e-02 |
| T[3] | `0x3F9664F48406D637` | 2.18694882948595424599e-02 |
| T[4] | `0x3F8226E3E96E8493` | 8.86323982359930005737e-03 |
| T[5] | `0x3F6D6D22C9560328` | 3.59207910759131235356e-03 |
| T[6] | `0x3F57DBC8FEE08315` | 1.45620945432529025516e-03 |
| T[7] | `0x3F4344D8F2F26501` | 5.88041240820264096874e-04 |
| T[8] | `0x3F3026F71A8D1068` | 2.46463134818469906812e-04 |
| T[9] | `0x3F147E88A03792A6` | 7.81794442939557092300e-05 |
| T[10] | `0x3F12B80F32F0A7E9` | 7.14072491382608190305e-05 |
| T[11] | `0xBEF375CBDB605373` | -1.85586374855275456654e-05 |
| T[12] | `0x3EFB2A7074BF7AD4` | 2.59073051863633712884e-05 |

출처: musl `src/math/__tan.c`, 원 저작권 Sun Microsystems / FreeBSD msun

### Special Cases

| 입력 | 결과 |
|---|---|
| `tan(NaN)` | NaN |
| `tan(±0)` | ±0 |
| `tan(±∞)` | NaN |

### ULP Accuracy

- 최대 1 ULP (π/2 근방 제외, 그 근방에서 catastrophic cancellation 불가피)

### Reference

- `musl src/math/tan.c`, `src/math/__tan.c`

---

## 4. asin(x)

### Argument Reduction

- `|x| < 0.5`: 직접 근사 `asin(x) = x + x·R(x²)`
- `0.5 ≤ |x| < 1`: 항등식 `asin(x) = π/2 - 2·asin(√((1-x)/2))` 적용
- `|x| = 1`: `±π/2` 직접 반환
- `|x| > 1` 또는 NaN: NaN 반환 (invalid operation)

#### π/2 분할 상수

| 상수 | IEEE 754 Hex | Bit-exact Decimal |
|---|---|---|
| `pio2_hi` | `0x3FF921FB54442D18` | 1.57079632679489655800e+00 |
| `pio2_lo` | `0x3C91A62633145C07` | 6.12323399573676603587e-17 |

### Polynomial Approximation — Rational R(z) = P(z)/Q(z)

유효 구간: `z = x² ∈ [0, 0.25]`

**분자 P(z) — pS 계수:**

| 계수 | IEEE 754 Hex | Bit-exact Decimal |
|---|---|---|
| pS0 | `0x3FC5555555555555` | 1.66666666666666657415e-01 |
| pS1 | `0xBFD4D61203EB6F7D` | -3.25565818622400915405e-01 |
| pS2 | `0x3FC9C1550E884455` | 2.01212532134862925881e-01 |
| pS3 | `0xBFA48228B5688F3B` | -4.00555345006794114027e-02 |
| pS4 | `0x3F49EFE07501B288` | 7.91534994289814532176e-04 |
| pS5 | `0x3F023DE10DFDF709` | 3.47933107596021167570e-05 |

**분모 Q(z) — qS 계수 (선행 계수 1 생략):**

| 계수 | IEEE 754 Hex | Bit-exact Decimal |
|---|---|---|
| qS1 | `0xC0033A271C8A2D4B` | -2.40339491173441421878e+00 |
| qS2 | `0x40002AE59C598AC8` | 2.02094576023350569471e+00 |
| qS3 | `0xBFE6066C1B8D0159` | -6.88283971605453293030e-01 |
| qS4 | `0x3FB3B8C5B12E9282` | 7.70381505559019352791e-02 |

출처: musl `src/math/asin.c`

### Special Cases

| 입력 | 결과 |
|---|---|
| `asin(NaN)` | NaN |
| `asin(±0)` | ±0 |
| `asin(±1)` | ±π/2 |
| `|x| > 1` | NaN (invalid) |
| `|x| < 2^-26` | x 그대로 (inexact 없이) |

### ULP Accuracy

- 최대 1 ULP (Remez 오차 ≤ 2^-58.75)

### Reference

- `musl src/math/asin.c`
- Muller Ch.7 (역삼각함수)

---

## 5. acos(x)

### Argument Reduction

- `|x| < 0.5`: `acos(x) = π/2 - (x + x·R(x²))`
- `x > 0.5`: `acos(x) = 2·asin(√((1-x)/2))` → `2·(f + w)` (f = sqrt(z) 상위 비트, w = 보정항)
- `x < -0.5`: `acos(x) = π - 2·asin(√((1-|x|)/2))`

π/2 분할 상수는 asin과 동일.

추가 상수:
- `pio2_hi = 1.57079632679489655800e+00` (`0x3FF921FB54442D18`)
- `pio2_lo = 6.12323399573676603587e-17` (`0x3C91A62633145C07`)

### Polynomial Approximation — Rational R(z) = P(z)/Q(z)

asin(x)과 동일한 pS0~pS5, qS1~qS4 계수 공유 (musl 구현에서 같은 값 사용).

### Special Cases

| 입력 | 결과 |
|---|---|
| `acos(NaN)` | NaN |
| `acos(1)` | +0 |
| `acos(-1)` | π |
| `|x| > 1` | NaN (invalid) |
| `|x| < 2^-57` | π/2 (pio2_hi) |

### ULP Accuracy

- 최대 1 ULP

### Reference

- `musl src/math/acos.c`

---

## 6. atan(x)

### Argument Reduction — 5구간 분기

| 범위 | 변환 | 사용 상수 |
|---|---|---|
| `|x| < 7/16` | 직접 다항식 평가 | — |
| `7/16 ≤ |x| < 11/16` | `atan(1/2) + atan((x-0.5)/(1+x/2))` | atanhi[0], atanlo[0] |
| `11/16 ≤ |x| < 19/16` | `atan(1) + atan((x-1)/(1+x))` | atanhi[1], atanlo[1] |
| `19/16 ≤ |x| < 39/16` | `atan(3/2) + atan((x-1.5)/(1+1.5x))` | atanhi[2], atanlo[2] |
| `39/16 ≤ |x|` | `atan(∞) + atan(-1/x)` | atanhi[3], atanlo[3] |

#### atanhi / atanlo 테이블 (musl `src/math/atan.c`)

| i | atanhi Hex | atanlo Hex | 의미 |
|---|---|---|---|
| 0 | `0x3FDDAC670561BB4F` | `0x3C7A2B7F222F65E2` | atan(0.5) |
| 1 | `0x3FE921FB54442D18` | `0x3C81A62633145C07` | atan(1) = π/4 |
| 2 | `0x3FEF730BD281F69B` | `0x3C7007887AF0CBBD` | atan(1.5) |
| 3 | `0x3FF921FB54442D18` | `0x3C91A62633145C07` | atan(∞) = π/2 |

#### aT[] — 11개 다항식 계수

| 인덱스 | IEEE 754 Hex | 부호 |
|---|---|---|
| aT[0] | `0x3FD555555555550D` | +1/3 |
| aT[1] | `0xBFC999999998EBC4` | -1/5 |
| aT[2] | `0x3FC24924920083FF` | +1/7 |
| aT[3] | `0xBFBC71C6FE231671` | -1/9 |
| aT[4] | `0x3FB745CDC54C206E` | +1/11 |
| aT[5] | `0xBFB3B0F2AF749A6D` | -1/13 |
| aT[6] | `0x3FB10D66A0D03D51` | +1/15 |
| aT[7] | `0xBFADDE2D52DEFD9A` | -1/17 |
| aT[8] | `0x3FA97B4B24760DEB` | +1/19 |
| aT[9] | `0xBFA2B4442C6A6C2F` | -1/21 |
| aT[10] | `0x3F90AD3AE322DA11` | +1/23 |

평가식: `s1 = aT[0]+z*(aT[2]+z*(aT[4]+z*(aT[6]+z*(aT[8]+z*aT[10]))))`,
`s2 = z*(aT[1]+z*(aT[3]+z*(aT[5]+z*(aT[7]+z*aT[9]))))`,
`atan(x) ≈ x + x·(s1 + s2)` (z = x²)

### Special Cases

| 입력 | 결과 |
|---|---|
| `atan(NaN)` | NaN |
| `atan(±0)` | ±0 |
| `atan(±∞)` | ±π/2 |
| `|x| < 2^-27` | x 그대로 |

### ULP Accuracy

- 최대 1 ULP

### Reference

- `musl src/math/atan.c`

---

## 7. sinh(x)

### 알고리즘 전략

musl은 직접 다항식이 아닌 **`expm1` 기반 공식** 사용:
`sinh(x) = (exp(x) - 1/exp(x)) / 2 = (expm1(x) + expm1(x)/(expm1(x)+1)) / 2`

### Range Split

| 범위 (w = 상위 32비트) | 공식 |
|---|---|
| `|x| < 2^-26` | `x` 그대로 반환 (underflow 방지) |
| `|x| < log(DBL_MAX)` (w < `0x40862e42`) | `h = 0.5 * sign(x)`, `t = expm1(|x|)`, 두 가지 sub-branch |
| `|x| ≥ log(DBL_MAX)` | `__expo2(|x|, 2h)` |

#### moderate 범위 sub-branch

- `t + t/(t+1)`이 overflow 위험 없을 때: `h * (t + t/(t+1))`
- 그 외: `h * (2t - t·t/(t+1))`

### expm1 내부 다항식 — 유리 근사 Q(z) (z = x²/2)

| 계수 | IEEE 754 Hex | Bit-exact Decimal |
|---|---|---|
| Q1 | `0xBFA11111111110F4` | -3.33333333333331316428e-02 |
| Q2 | `0x3F5A01A019FE5585` | 1.58730158725481460165e-03 |
| Q3 | `0xBF14CE199EAADBB7` | -7.93650757867487942473e-05 |
| Q4 | `0x3ED0CFCA86E65239` | 4.00821782732936239552e-06 |
| Q5 | `0xBE8AFDB76E09C32D` | -2.01099218183624371326e-07 |

출처: musl `src/math/expm1.c`

### 항등식 기반 축소 상수 (expm1)

| 상수 | 값 |
|---|---|
| `ln2_hi` | 6.93147180369123816490e-01 |
| `ln2_lo` | 1.90821492927058770002e-10 |
| `invln2` | 1.44269504088896338700e+00 |

### Special Cases

| 입력 | 결과 |
|---|---|
| `sinh(NaN)` | NaN |
| `sinh(±0)` | ±0 |
| `sinh(±∞)` | ±∞ |

### ULP Accuracy

- expm1의 ULP (≤1) 에 의존; 전체 ≤ 2 ULP

### Reference

- `musl src/math/sinh.c`, `src/math/expm1.c`

---

## 8. cosh(x)

### 알고리즘 전략

**`exp` 및 `expm1` 기반**:

| 범위 (w = 상위 32비트) | 공식 |
|---|---|
| `|x| < 2^-26` | 1.0 반환 (inexact 예외) |
| `|x| < log(2)` (w < `0x3fe62e42`) | `t = expm1(x)` → `1 + t·t/(2·(1+t))` |
| `log(2) ≤ |x| < log(DBL_MAX)` (w < `0x40862e42`) | `t = exp(x)` → `0.5·(t + 1/t)` |
| `|x| ≥ log(DBL_MAX)` | `__expo2(x, 1.0)` |

### Special Cases

| 입력 | 결과 |
|---|---|
| `cosh(NaN)` | NaN |
| `cosh(±0)` | 1.0 |
| `cosh(±∞)` | +∞ |

### ULP Accuracy

- exp, expm1에 의존; ≤ 2 ULP

### Reference

- `musl src/math/cosh.c`

---

## 9. tanh(x)

### 알고리즘 전략

**`expm1` 기반**: `tanh(x) = (exp(2x)-1)/(exp(2x)+1)` 을 수치 안정적으로 계산

### Range Split (w = 상위 32비트 of |x|)

| 범위 | 공식 |
|---|---|
| `|x| > 20` | `1 - 0/x` (±1 근사, overflow 방지) |
| `log(3)/2 < |x| ≤ 20` (w > `0x3fe193ea`) | `t = expm1(2x)` → `1 - 2/(t+2)` |
| `log(5/3)/2 < |x| ≤ log(3)/2` (w > `0x3fd058ae`) | `t = expm1(2x)` → `t/(t+2)` |
| `|x| < log(5/3)/2` | `t = expm1(-2x)` → `-t/(t+2)` |
| subnormal | `x` 그대로 |

### Special Cases

| 입력 | 결과 |
|---|---|
| `tanh(NaN)` | NaN |
| `tanh(±0)` | ±0 |
| `tanh(±∞)` | ±1 |

### ULP Accuracy

- ≤ 2 ULP

### Reference

- `musl src/math/tanh.c`

---

## 10. exp(x)

### Argument Reduction — 2^(k/N) 분해

`exp(x) = 2^(k/N) · exp(r)` 단, `r = x - k·ln2/N`, `|r| ≤ ln2/(2N)`

| 상수 | 값 |
|---|---|
| `InvLn2N` | `N / ln2 = N · 0x1.71547652b82fep0` |
| `NegLn2hiN` | `-ln2/N` (high) = `-0x1.62e42fefa0000p-8` (N=256 기준) |
| `NegLn2loN` | `-ln2/N` (low) = `-0x1.cf79abc9e3b3ap-47` |

- 테이블 `exp_data.tab[N]`: 2^(k/N) 를 미리 계산한 128쌍 (hi, tail)

### Polynomial Approximation — exp(r) ≈ 1 + r + r²·P(r)

`P(r) = C1 + r·C2 + r²·(C3 + r·C4)` 형태 (Horner)

| 계수 | IEEE 754 Hex (hex float literal) | 오차 |
|---|---|---|
| C1 | `0x1.ffffffffffdbdp-2` | — |
| C2 | `0x1.555555555543cp-3` | — |
| C3 | `0x1.55555cf172b91p-5` | — |
| C4 | `0x1.1111167a4d017p-7` | — |

- 절대 오차: `1.555·2^-66`
- ULP 오차: 0.509

출처: musl `src/math/exp_data.c`

### Special Cases

| 입력 | 결과 |
|---|---|
| `exp(NaN)` | NaN |
| `exp(+∞)` | +∞ |
| `exp(-∞)` | +0 |
| `exp(x)` x > 709.78 | +∞ (overflow) |
| `exp(x)` x < -745.13 | +0 (underflow) |

### ULP Accuracy

- ≤ 0.5 ULP (FMA 사용 시), ≤ 1 ULP (스칼라)

### Reference

- `musl src/math/exp.c`, `src/math/exp_data.c`, `src/math/exp_data.h`
- Muller Ch.11 (지수함수)

---

## 11. ln(x) / log(x)

### Argument Reduction — 테이블 기반

`log(x) = k·ln2 + log(c) + log(z/c)` 단, `z/c ≈ 1`

1. `x = 2^k · z` 추출 (비트 조작: `k = (int64_t)tmp >> 52`)
2. 테이블 인덱스 `i = (tmp >> (52 - LOG_TABLE_BITS)) % N`
3. 테이블에서 `invc`, `logc`, `logctail` 조회
4. `r = z·invc - 1` 계산 후 다항식 근사

`OFF = 0x3fe6000000000000` (감산 기준)

#### ln2 분할 상수

| 상수 | IEEE 754 Hex Literal | 값 |
|---|---|---|
| `ln2hi` | `0x1.62e42fefa3800p-1` | 6.93147180369123816490e-01 (상위) |
| `ln2lo` | `0x1.ef35793c76730p-45` | 1.28864392504924193000e-17 (하위) |

### Polynomial Approximation

#### 일반 구간 poly (A[], 5계수, 좁은 범위 `[-0x1.fp-9, 0x1.fp-9]`)

| 계수 | IEEE 754 Hex Literal |
|---|---|
| A[0] | `-0x1.0000000000001p-1` |
| A[1] | `0x1.555555551305bp-2` |
| A[2] | `-0x1.fffffffeb459p-3` |
| A[3] | `0x1.999b324f10111p-3` |
| A[4] | `-0x1.55575e506c89fp-3` |

#### 확장 구간 poly1 (B[], 11계수, `[-0x1p-4, 0x1.09p-4]`)

| 계수 | IEEE 754 Hex Literal |
|---|---|
| B[0] | `-0x1p-1` |
| B[1] | `0x1.5555555555577p-2` |
| B[2] | `-0x1.ffffffffffdcbp-3` |
| B[3] | `0x1.999999995dd0cp-3` |
| B[4] | `-0x1.55555556745a7p-3` |
| B[5] | `0x1.24924a344de3p-3` |
| B[6] | `-0x1.fffffa4423d65p-4` |
| B[7] | `0x1.c7184282ad6cap-4` |
| B[8] | `-0x1.999eb43b068ffp-4` |
| B[9] | `0x1.78182f7afd085p-4` |
| B[10] | `-0x1.5521375d145cdp-4` |

출처: musl `src/math/log_data.c`, `src/math/log.c`

### Special Cases

| 입력 | 결과 |
|---|---|
| `log(NaN)` | NaN |
| `log(+0)` | -∞ (divide-by-zero) |
| `log(-0)` | -∞ |
| `log(+∞)` | +∞ |
| `log(x < 0)` | NaN (invalid) |
| `log(1)` | +0 |
| subnormal | 2^52 곱셈으로 정규화 후 처리 |

### ULP Accuracy

- ≤ 0.507 ULP (1에 가까운 값), 일반적 ≤ 0.5 ULP (FMA 사용)

### Reference

- `musl src/math/log.c`, `src/math/log_data.c`, `src/math/log_data.h`

---

## 12. log2(x) / log10(x)

### 구현 전략

- `log2(x)`: `ln(x) / ln(2)` 혹은 musl에서 별도 `log2_data.c` 사용 (log와 유사 구조, N=128 테이블)
- `log10(x)`: `ln(x) / ln(10)` 혹은 `log10_data.c` 별도 구현
- 계수는 musl `src/math/log2_data.c`, `src/math/log10_data.c` 참조

---

## 13. pow(x, y)

### 핵심 전략

`pow(x, y) = exp(y · log(x))`

3단계 파이프라인:
1. `log_inline(x)` — 확장 정밀도 로그 (pow_data.h의 별도 계수 사용)
2. `y · log(x)` 곱
3. `exp_inline(y·log(x))` — 확장 정밀도 지수

### 정수 지수 특수처리

`checkint(y)` 함수:
- 반환 0: 비정수
- 반환 1: 홀수 정수
- 반환 2: 짝수 정수

음수 밑 처리: y가 홀수 정수이면 `sign_bias = SIGN_BIAS` 설정 후 결과 부호 반전, 그 외 NaN.

### pow 전용 log 계수 (pow_data.c)

7계수 P(z):

| 계수 | IEEE 754 Hex Literal | 스케일 |
|---|---|---|
| P[0] | `-0x1p-1` | ×1 |
| P[1] | `0x1.555555555556p-2` | ×(-2) |
| P[2] | `-0x1.0000000000006p-2` | ×(-2) |
| P[3] | `0x1.999999959554ep-3` | ×4 |
| P[4] | `-0x1.555555529a47ap-3` | ×4 |
| P[5] | `0x1.2495b9b4845e9p-3` | ×(-8) |
| P[6] | `-0x1.0002b8b263fc3p-3` | ×(-8) |

상대 오차: `0x1.11922ap-70` (구간 `[-0x1.6bp-8, 0x1.6bp-8]`)

### Overflow/Underflow 처리

- overflow (k > 0): 지수를 1009 감소 후 계산, 결과에 2^1009 곱
- underflow (k < 0): 지수를 1022 증가 후 계산, 결과에 2^-1022 곱 (subnormal 신호 필요)

### Special Cases (IEEE 754-2019 §9.2.1)

| 입력 | 결과 |
|---|---|
| `pow(x, ±0)` | 1 (모든 x, NaN 포함) |
| `pow(1, y)` | 1 (모든 y, NaN 포함) |
| `pow(NaN, y≠0)` | NaN |
| `pow(x, NaN)` | NaN (x≠1) |
| `pow(-1, ±∞)` | 1 |
| `pow(|x|<1, -∞)` | +∞ |
| `pow(|x|>1, +∞)` | +∞ |
| `pow(+0, y<0)` | +∞ (divide-by-zero) |
| `pow(-0, y<0 홀수정수)` | -∞ |
| `pow(x<0, 비정수)` | NaN (invalid) |

### ULP Accuracy

- 0.54 ULP (pow_data.c 주석)

### Reference

- `musl src/math/pow.c`, `src/math/pow_data.c`, `src/math/pow_data.h`
- Muller Ch.12 (power 함수)

---

## 14. sqrt(x)

### 구현 전략 — Cython에서는 하드웨어 sqrt 직접 호출

Cython에서는 **하드웨어 sqrt 직접 호출 권장**:

```cython
from libc.math cimport sqrt
```

C99 표준 `sqrt()`는 x86에서 `sqrtsd`, ARM에서 `fsqrt` 등 단일 하드웨어 명령으로 컴파일된다. IEEE 754 필수 요구사항상 correctly rounded (≤0.5 ULP) 보장.

이것이 `math.sqrt`의 1.1~1.5x 범위 달성에 가장 확실한 경로.

### 소프트웨어 알고리즘 (하드웨어 미지원 환경용, 참고)

1. 초기 역제곱근 추정: 7비트 테이블 룩업 (1비트 지수 + 6비트 가수)
   - 오차: < `0x1.fdp-9`
2. Goldschmidt 반복 2회 (32비트 산술)
3. 최종 64비트 반복 1회
   - 정확도: `-0x1p-57 < s - sqrt(m) < 0x1.8001p-61`

### Special Cases

| 입력 | 결과 |
|---|---|
| `sqrt(NaN)` | NaN |
| `sqrt(+∞)` | +∞ |
| `sqrt(±0)` | ±0 |
| `sqrt(x < 0)` | NaN (invalid) |
| subnormal | 2^52 곱셈 정규화 후 처리 |

### ULP Accuracy

- ≤ 0.5 ULP (IEEE 754 필수: sqrt는 correctly rounded)

### Reference

- `musl src/math/sqrt.c`
- IEEE 754-2019 §5.3.1 (sqrt는 correctly rounded 의무)

---

## 부록 A: Payne-Hanek 요약

sin/cos/tan에서 `|x| ≥ 2^20·π/2` (ix ≥ `0x413921fb`) 이상의 큰 인수에 사용.

- 2/π의 396자리 16진 테이블 `ipio2[]` 사용
- 입력 크기에 무관한 연산 수로 정밀 나머지 계산
- `__rem_pio2_large(x, y, e0, nx, prec)` 형태 호출

출처: `musl src/math/__rem_pio2_large.c`

---

## 부록 B: ULP 오차 요약표

| 함수 | musl 최대 ULP | CORE-MATH |
|---|---|---|
| sin, cos | 1 | correctly rounded (0.5) |
| tan | 1 | correctly rounded |
| asin, acos | 1 | correctly rounded |
| atan | 1 | correctly rounded |
| sinh, cosh, tanh | ≤ 2 | — |
| exp | ≤ 1 (FMA: 0.5) | correctly rounded |
| log, ln | ≤ 1 (FMA: 0.5) | correctly rounded |
| pow | ≤ 1 (설계: 0.54) | — |
| sqrt | 0.5 (IEEE 필수) | 0.5 |

---

## 부록 C: Cython 구현 시 주의사항

1. **FMA 활용**: `libc.math cimport fma` — 내적 계산에서 1 ULP 개선 가능
2. **sqrt**: `from libc.math cimport sqrt` — 하드웨어 명령 직접 발행
3. **IEEE 754 특수값 처리 순서**: NaN 체크 → ±Inf 체크 → ±0 체크 → 일반 경로
4. **argument reduction 분기**: `|x|` 임계값을 비트 조작으로 빠르게 판별 (`(<unsigned long long>x >> 52) & 0x7FF` 지수 추출)
5. **poly 계수**: 반드시 본 문서의 bit-exact 값 사용, 반올림 오차 누적 방지
6. **pow의 오차 누적**: pow = exp(y·log(x)) 구조상 log와 exp 오차가 합산 → 정수 지수는 반드시 이진 거듭제곱(`_power_integer_real` 패턴)으로 분기

---

## 참고문헌

1. Jean-Michel Muller, "Elementary Functions: Algorithms and Implementation", 3rd ed., Birkhäuser, 2016.
2. Sun Microsystems / FreeBSD msun, fdlibm, `src/math/` — musl 계수의 원 출처.
3. musl libc, git.musl-libc.org/cgit/musl, `src/math/`, 최신 커밋.
4. CORE-MATH, core-math.gitlabpages.inria.fr, INRIA, correctly rounded 구현.
5. IEEE 754-2019, IEEE Standard for Floating-Point Arithmetic.
6. RLIBM, rlibm.github.io, Rutgers, 구간 별 올바른 반올림 라이브러리.
