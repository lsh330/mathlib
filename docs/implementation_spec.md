# Cython 구현 명세 (Implementation Specification)

> 대상: `sim-engineer`가 한 치 망설임 없이 그대로 구현 가능한 수준의 명세  
> 목표: `math` 모듈 대비 1.1~1.5배 이내 속도, musl 수준 ULP 오차 (최대 1 ULP)  
> 작성일: 2026-04-20  
> 선행 문서: `algorithm_reference.md`, `cython_best_practices.md`

---

## 목차

1. [Scope & Goals](#1-scope--goals)
2. [전체 파일 구조](#2-전체-파일-구조)
3. [공용 API 설계](#3-공용-api-설계)
4. [Elementary Primitives 상세 명세](#4-elementary-primitives-상세-명세)
5. [`__rem_pio2` Argument Reduction 설계](#5-__rem_pio2-argument-reduction-설계)
6. [특수 함수 변환 전략](#6-특수-함수-변환-전략)
7. [Differentiation Ridders 재설계](#7-differentiation-ridders-재설계)
8. [빌드 시스템](#8-빌드-시스템)
9. [누락 함수 / TODO 정리](#9-누락-함수--todo-정리)
10. [실수/복소수 분기 방침](#10-실수복소수-분기-방침)
11. [예상 성능 요약표](#11-예상-성능-요약표)

---

## 1. Scope & Goals

### 1.1 범위

| 카테고리 | 함수 | 구현 방식 |
|---|---|---|
| 삼각 | sin, cos, tan, sec, cosec, cotan | musl 계수 + Cython Horner-FMA |
| 역삼각 | arcsin, arccos, arctan, arcsec, arccosec, arccotan | musl rational R(z) |
| 쌍곡 | hypersin, hypercos, hypertan, hypersec, hypercosec, hypercotan | expm1/exp 기반 |
| 지수/로그 | exp, ln, log, power, sqrt | musl 테이블 기반 + `libc sqrt` |
| 특수 | gamma, beta, bessel, legendre, lambert_w, zeta, euler_pi, heaviside | 알고리즘 유지, `.py` → `.pyx` 변환 |
| 산술 | gcd, lcm | `.py` → `.pyx` 변환 |
| 미분 | Differentiation | Ridders 재설계 |

### 1.2 철학

- **자체 구현 우선**: `libc.math`의 `sin`/`cos`/`exp`/`log`는 호출 금지.
- **허용된 수치 헬퍼**: `fma`, `frexp`, `ldexp`, `sqrt`, `isnan`, `isinf`, `fabs`, `copysign`.
- **bit-exact 계수**: `algorithm_reference.md`의 hex literal을 그대로 인용 (재발명 금지).
- **ULP 보장**: musl 참조 — sin/cos/tan/asin/acos/atan <= 1 ULP, sqrt <= 0.5 ULP, exp/log <= 1 ULP, pow <= 1 ULP.
- **실수 모드 기본 최적화**: 1차 스코프에서 **실수 모드만 Cython 고속 구현**. 복소수 모드는 Python 폴백으로 유지 (10절 참조).

### 1.3 성공 기준

1. `math.sin(1.2345)` 대비 `mathlib.sin(1.2345)`가 **<= 1.5x 시간** 이내
2. 1M 랜덤 입력에 대한 `math.sin` vs `mathlib.sin` 평균 오차 **<= 2 ULP**
3. IEEE 754 특수값 (NaN, +/-Inf, +/-0, subnormal) 전체 매칭
4. 기존 Python 테스트 (`tests/`) 95% 이상 통과

---

## 2. 전체 파일 구조

```
mathlib/
├── pyproject.toml                           # Cython 빌드 시스템 선언
├── setup.py                                 # Cython Extension 빌드
├── src/math_library/
│   ├── __init__.py                          # 공용 API 재노출
│   │
│   ├── _core/                               # [NEW] Cython elementary primitives
│   │   ├── __init__.py                      # (빈 파일)
│   │   ├── _helpers.pxd                     # cdef inline 유틸 (poly, FMA, bit 조작)
│   │   ├── _helpers.pyx                     # 구현 보조 (없으면 생략)
│   │   ├── _constants.pxd                   # cdef 상수 선언
│   │   ├── _constants.pyx                   # pi, e, epsilon Python 재노출
│   │   ├── argument_reduction.pxd           # rem_pio2 선언
│   │   ├── argument_reduction.pyx           # Cody-Waite 구현
│   │   ├── exponential.pxd                  # exp, _exp_inline, _expm1_inline 선언
│   │   ├── exponential.pyx                  # exp 구현
│   │   ├── logarithmic.pxd                  # ln, log, _ln_inline 선언
│   │   ├── logarithmic.pyx                  # ln 구현
│   │   ├── power_sqrt.pxd                   # power, sqrt 선언
│   │   ├── power_sqrt.pyx                   # power = exp*log, sqrt = libc.math.sqrt
│   │   ├── trigonometric.pxd                # sin/cos/tan/sec/cosec/cotan 선언
│   │   ├── trigonometric.pyx                # 삼각함수 구현
│   │   ├── inverse_trig.pxd                 # arcsin/arccos/arctan/... 선언
│   │   ├── inverse_trig.pyx                 # 역삼각함수 구현
│   │   ├── hyperbolic.pxd                   # hypersin/... 선언
│   │   └── hyperbolic.pyx                   # 쌍곡함수 구현
│   │
│   ├── constant/                            # [WRAP] 기존 Python 유지 + _core 재노출
│   │   ├── __init__.py
│   │   ├── e.py                             # 얇은 래퍼
│   │   ├── pi.py                            # 얇은 래퍼
│   │   └── epsilon.py                       # 얇은 래퍼
│   │
│   ├── trigonometric_function/              # [WRAP] 기존 API 호환 래퍼
│   │   ├── __init__.py
│   │   ├── sin.py                           # validate -> _core.trigonometric.sin
│   │   └── ... (cos, tan, sec, cosec, cotan)
│   │
│   ├── inverse_trigonometric_function/      # [WRAP] 래퍼
│   ├── hyperbolic_function/                 # [WRAP] 래퍼
│   ├── exponential_function/                # [WRAP] 래퍼
│   │   ├── power.py                         # 기존 시그니처 유지, 내부 _core 호출
│   │   └── (신규) exp.py, sqrt.py
│   ├── logarithmic_function/                # [WRAP] 래퍼
│   │   └── log.py                           # 기존 시그니처 유지, 내부 _core.ln 호출
│   │
│   ├── gamma_function/
│   │   ├── __init__.py
│   │   └── gamma.pyx                        # [CONVERT] .py -> .pyx
│   ├── beta_function/
│   │   └── beta.pyx
│   ├── bessel_function/
│   │   └── bessel.pyx
│   ├── legendre_function/
│   │   └── legendre.pyx
│   ├── lambert_w_function/
│   │   └── lambert_w.pyx
│   ├── zeta_function/
│   │   └── zeta.pyx
│   ├── euler_pi_function/
│   │   └── euler_pi.pyx
│   ├── heaviside_step_function/
│   │   └── heaviside.pyx
│   ├── gcd/
│   │   └── gcd.pyx
│   ├── lcm/
│   │   └── lcm.pyx
│   │
│   └── differentiation/
│       └── differentiation.pyx              # [REWRITE] Ridders 적용
│
├── docs/
│   ├── algorithm_reference.md               # (선행)
│   ├── cython_best_practices.md             # (선행)
│   └── implementation_spec.md               # (본 문서)
└── tests/                                   # 기존 테스트 유지 (리팩터 후 검증)
```

### 2.1 `cimport` 의존성 그래프

```
_helpers.pxd
     |
     +--> _constants.pxd
     |
     +--> argument_reduction.pxd
     |         |
     |         +--> trigonometric.pxd
     |
     +--> exponential.pxd ----+
     |                         +-> power_sqrt.pxd
     +--> logarithmic.pxd ----+
     |
     +--> trigonometric.pxd ---> inverse_trig.pxd
     |                            |                        +-> hyperbolic.pxd (sinh는 expm1 의존)
     |
     +--> gamma.pyx, beta.pyx, ...
          (특수 함수는 _core.* 전체 의존)
```

**원칙**: 모듈 간 호출은 반드시 `cimport`로 C 수준에서 수행. Python 경계(`import`) 절대 금지 (핫패스).

---

## 3. 공용 API 설계

### 3.1 기존 API (유지)

기존 테스트 (`tests/`) 호환을 위해 deep-path import는 유지:

```python
from math_library.trigonometric_function.sin import sin  # 기존
from math_library.exponential_function.power import power  # 기존
```

이들은 래퍼 모듈로서 **validate -> `_core.*` 호출 -> `_normalize`** 구조를 유지한다.

### 3.2 신규 top-level API

`src/math_library/__init__.py`를 새로 작성:

```python
# 상수
from ._core._constants import pi, e, epsilon

# elementary primitives (고속 C 경로)
from ._core.trigonometric import sin, cos, tan, sec, cosec, cotan
from ._core.inverse_trig import arcsin, arccos, arctan, arcsec, arccosec, arccotan
from ._core.hyperbolic import hypersin, hypercos, hypertan, hypersec, hypercosec, hypercotan
from ._core.exponential import exp
from ._core.logarithmic import ln, log
from ._core.power_sqrt import power, sqrt

# 특수 함수
from .gamma_function.gamma import gamma
from .beta_function.beta import beta
from .bessel_function.bessel import bessel
from .legendre_function.legendre import legendre
from .lambert_w_function.lambert_w import lambert_w
from .zeta_function.zeta import zeta
from .euler_pi_function.euler_pi import euler_pi
from .heaviside_step_function.heaviside import heaviside
from .gcd.gcd import gcd
from .lcm.lcm import lcm

# 미분
from .differentiation.differentiation import Differentiation

__all__ = [
    "pi", "e", "epsilon",
    "sin", "cos", "tan", "sec", "cosec", "cotan",
    "arcsin", "arccos", "arctan", "arcsec", "arccosec", "arccotan",
    "hypersin", "hypercos", "hypertan", "hypersec", "hypercosec", "hypercotan",
    "exp", "ln", "log", "power", "sqrt",
    "gamma", "beta", "bessel", "legendre", "lambert_w", "zeta",
    "euler_pi", "heaviside", "gcd", "lcm",
    "Differentiation",
]
```

사용 예:
```python
from math_library import sin, cos, exp, ln, Differentiation
```

### 3.3 API 시그니처 통일

- elementary primitives (실수 핫패스): `cpdef double sin(double x) noexcept`
- 기존 래퍼 함수: `def sin(x, unit="rad", tol=None, max_terms=100, number_system="real")` (호환성 유지)

**이중 API 방침**:
- top-level `from math_library import sin` -> **Cython 고속 경로** (스칼라 double 전용, 라디안, 실수)
- deep-path `from math_library.trigonometric_function.sin import sin` -> **기존 호환 경로** (unit/tol/number_system 지원)

---

## 4. Elementary Primitives 상세 명세

이 섹션은 17개 elementary primitive를 모두 다룬다. 각 primitive는:

1. 파일/의존성
2. 내부 `cdef` 함수 계층
3. `cpdef` 공개 API 의사코드
4. 특수 케이스 표
5. 계수 (algorithm_reference.md 각 섹션 인용)
6. 예상 성능
7. ULP 보장

순서로 기술된다.

---

### 4.0 공통 헬퍼 (`_helpers.pxd`)

```cython
# _helpers.pxd
# cython: language_level=3

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.math cimport fma

cdef extern from *:
    """
    typedef union {
        double d;
        unsigned long long u;
    } DoubleUnion;
    """
    ctypedef struct DoubleUnion:
        double d
        unsigned long long u

cdef inline uint64_t double_to_bits(double x) noexcept nogil:
    cdef DoubleUnion u
    u.d = x
    return u.u

cdef inline double bits_to_double(uint64_t bits) noexcept nogil:
    cdef DoubleUnion u
    u.u = bits
    return u.d

cdef inline uint32_t high_word(double x) noexcept nogil:
    # 상위 32비트 (IEEE 754 double의 부호+지수+가수 상위 20비트)
    return <uint32_t>(double_to_bits(x) >> 32)

cdef inline uint32_t low_word(double x) noexcept nogil:
    return <uint32_t>(double_to_bits(x) & 0xFFFFFFFFULL)

# Horner + FMA 다항식 평가 (필요 차수까지)
cdef inline double poly2(double x, double c0, double c1, double c2) noexcept nogil:
    return fma(fma(c2, x, c1), x, c0)

cdef inline double poly3(double x, double c0, double c1, double c2, double c3) noexcept nogil:
    return fma(fma(fma(c3, x, c2), x, c1), x, c0)

cdef inline double poly4(double x, double c0, double c1, double c2, double c3, double c4) noexcept nogil:
    return fma(fma(fma(fma(c4, x, c3), x, c2), x, c1), x, c0)

cdef inline double poly5(double x, double c0, double c1, double c2, double c3, double c4, double c5) noexcept nogil:
    return fma(fma(fma(fma(fma(c5, x, c4), x, c3), x, c2), x, c1), x, c0)

cdef inline double poly6(double x, double c0, double c1, double c2, double c3,
                          double c4, double c5, double c6) noexcept nogil:
    return fma(fma(fma(fma(fma(fma(c6, x, c5), x, c4), x, c3), x, c2), x, c1), x, c0)
```

**주의**: `cdef inline` 선언+구현을 `.pxd`에 두어야 `cimport` 후 C 수준 인라인 호출 가능.

---

### 4.1 sin(x)

#### 파일: `src/math_library/_core/trigonometric.pyx`
#### 의존성: `_helpers.pxd`, `argument_reduction.pxd`

#### 계수 선언 (파일 상단)

```cython
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False

from ._helpers cimport double_to_bits, high_word, low_word
from .argument_reduction cimport rem_pio2
from libc.math cimport fma
from libc.stdint cimport uint32_t

# musl __sin.c 계수 (algorithm_reference.md 섹션 1)
cdef double S1 = -1.66666666666666324348e-01  # 0xBFC5555555555549
cdef double S2 =  8.33333333332248946124e-03  # 0x3F8111111110F8A6
cdef double S3 = -1.98412698298579493134e-04  # 0xBF2A01A019C161D5
cdef double S4 =  2.75573137070700676789e-06  # 0x3EC71DE357B1FE7D
cdef double S5 = -2.50507602534068634195e-08  # 0xBE5AE5E68A2B9CEB
cdef double S6 =  1.58969099521155010221e-10  # 0x3DE5D93A5ACFD57C
```

#### 내부 `_sin_kernel`

```cython
cdef inline double _sin_kernel(double x, double y) noexcept nogil:
    # 입력: x in [-pi/4, pi/4], y는 rem_pio2의 low-word 보정
    cdef double z = x * x
    cdef double w = z * z
    cdef double r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6)
    cdef double v = z * x
    if y == 0.0:
        return x + v * (S1 + z * r)
    return x - ((z * (0.5 * y - v * r) - y) - v * S1)
```

#### 공개 API `sin`

```cython
cpdef double sin(double x) noexcept:
    cdef double y[2]
    cdef int n
    cdef int q
    cdef uint32_t ix = high_word(x) & 0x7FFFFFFFU

    # |x| < pi/4 -> 직접 kernel
    if ix < 0x3FE921FBU:
        # |x| < 2^-26 -> x 그대로 (inexact 없이)
        if ix < 0x3E500000U:
            return x
        return _sin_kernel(x, 0.0)

    # NaN, +/-Inf -> NaN (x - x 트릭)
    if ix >= 0x7FF00000U:
        return x - x

    # 일반 argument reduction
    n = rem_pio2(x, y)
    q = n & 3
    if q == 0:
        return  _sin_kernel(y[0], y[1])
    elif q == 1:
        return  _cos_kernel(y[0], y[1])
    elif q == 2:
        return -_sin_kernel(y[0], y[1])
    else:  # q == 3
        return -_cos_kernel(y[0], y[1])
```

#### 특수 케이스

| 입력 | 결과 | 구현 근거 |
|---|---|---|
| NaN | NaN | `x - x` (NaN 전파) |
| +/-Inf | NaN | `ix >= 0x7FF00000` -> `x - x` |
| +/-0 | +/-0 | `ix < 0x3E500000` -> x 그대로 |
| subnormal | x 그대로 | 위와 동일 |

#### 예상 성능

- `math.sin`: ~55ns/call (libm sin 래퍼)
- 목표 `mathlib.sin`: 60~80ns/call (1.1~1.45x)
- 내부: `_sin_kernel` ~15ns, `high_word` ~1ns, `rem_pio2` ~10ns (작은 x 스킵 시 0ns)

#### ULP 보장

- `|x| < pi/4`: <= 1 ULP (musl `__sin` 기준)
- `|x| < 2^20*pi/2`: <= 1 ULP (Cody-Waite rem_pio2)
- `|x| >= 2^20*pi/2`: <= 수 ULP (섹션 5 참조, Payne-Hanek 미구현 트레이드오프)

---

### 4.2 cos(x)

#### 파일: `src/math_library/_core/trigonometric.pyx`

#### 계수 (musl `__cos.c`, algorithm_reference.md 섹션 2)

```cython
cdef double C1 =  4.16666666666666019037e-02  # 0x3FA555555555554C
cdef double C2 = -1.38888888888741095749e-03  # 0xBF56C16C16C15177
cdef double C3 =  2.48015872894767294178e-05  # 0x3EFA01A019CB1590
cdef double C4 = -2.75573143513906633035e-07  # 0xBE927E4F809C52AD
cdef double C5 =  2.08757232129817482790e-09  # 0x3E21EE9EBDB4B1C4
cdef double C6 = -1.13596475577881948265e-11  # 0xBDA8FAE9BE8838D4
```

#### 내부 `_cos_kernel`

```cython
cdef inline double _cos_kernel(double x, double y) noexcept nogil:
    cdef double z = x * x
    cdef double w = z * z
    cdef double r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6))
    cdef double hz = 0.5 * z
    cdef double w1 = 1.0 - hz
    return w1 + (((1.0 - w1) - hz) + (z * r - x * y))
```

#### 공개 API `cos`

```cython
cpdef double cos(double x) noexcept:
    cdef double y[2]
    cdef int n, q
    cdef uint32_t ix = high_word(x) & 0x7FFFFFFFU

    if ix < 0x3FE921FBU:
        if ix < 0x3E46A09EU:  # |x| < 2^-27
            return 1.0
        return _cos_kernel(x, 0.0)

    if ix >= 0x7FF00000U:
        return x - x

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
```

#### 특수 케이스

| 입력 | 결과 |
|---|---|
| NaN | NaN |
| +/-Inf | NaN |
| +/-0 | 1.0 |

#### 성능/ULP

- 목표: 60~80ns/call (sin과 동일)
- ULP: <= 1

---

### 4.3 tan(x)

#### 파일: `src/math_library/_core/trigonometric.pyx`

#### 계수 (musl `__tan.c`, algorithm_reference.md 섹션 3)

```cython
cdef double[13] T = [
     3.33333333333334091986e-01,  # T[0]
     1.33333333333201242699e-01,  # T[1]
     5.39682539762260521377e-02,  # T[2]
     2.18694882948595424599e-02,  # T[3]
     8.86323982359930005737e-03,  # T[4]
     3.59207910759131235356e-03,  # T[5]
     1.45620945432529025516e-03,  # T[6]
     5.88041240820264096874e-04,  # T[7]
     2.46463134818469906812e-04,  # T[8]
     7.81794442939557092300e-05,  # T[9]
     7.14072491382608190305e-05,  # T[10]
    -1.85586374855275456654e-05,  # T[11]
     2.59073051863633712884e-05,  # T[12]
]

# 큰 인수 경로 사용 상수 (pi/4)
cdef double PIO4_HI = 7.85398163397448278999e-01
cdef double PIO4_LO = 3.06161699786838301793e-17
```

#### 내부 `_tan_kernel(double x, double y, int iy)`

`iy = 0`이면 `tan(x+y)`, `iy = 1`이면 `-1/tan(x+y)` 반환.  
전체 코드는 musl `src/math/__tan.c` 원본을 **bit-exact 복제**할 것. 핵심 흐름은 다음과 같다.

```cython
cdef inline double _tan_kernel(double x, double y, int iy) noexcept nogil:
    cdef uint32_t hx = high_word(x)
    cdef uint32_t ix = hx & 0x7FFFFFFFU
    cdef double z, r, v, w, s
    cdef int sign

    # 작은 x 경로: |x| < 2^-28
    if ix < 0x3E300000U:
        if iy == 1:
            return 1.0 / x
        return x

    # 큰 x (|x| >= 0.6744) 경로: pi/4-x 재매핑
    if ix >= 0x3FE59428U:
        sign = 1 if (hx >> 31) == 0 else -1
        if sign < 0:
            x = -x
            y = -y
        z = PIO4_HI - x
        w = PIO4_LO - y
        x = z + w
        y = 0.0

    z = x * x
    w = z * z
    # musl Horner (홀수/짝수 분리)
    r = T[1] + w*(T[3] + w*(T[5] + w*(T[7] + w*(T[9] + w*T[11]))))
    v = z*(T[2] + w*(T[4] + w*(T[6] + w*(T[8] + w*(T[10] + w*T[12])))))
    s = z * x
    r = y + z*(s*(r + v) + y)
    r = r + T[0]*s
    w = x + r

    if ix >= 0x3FE59428U:
        # 부호 보정 (musl 원본 참조)
        return sign * (1.0 - 2.0*((hx >> 30) & 2)) * (...)  # 실제 구현 필요
    if iy == 1:
        # -1/tan(x+y) 수치 안정 계산
        return -1.0 / w
    return w
```

**중요**: 위 의사코드는 **구조만 제시**. sim-engineer는 musl `src/math/__tan.c` 43~76행 로직을 정확히 그대로 이식할 것.

#### 공개 API `tan`

```cython
cpdef double tan(double x) noexcept:
    cdef double y[2]
    cdef int n
    cdef uint32_t ix = high_word(x) & 0x7FFFFFFFU

    if ix < 0x3FE921FBU:
        if ix < 0x3E400000U:  # |x| < 2^-27
            return x
        return _tan_kernel(x, 0.0, 0)  # iy=0

    if ix >= 0x7FF00000U:
        return x - x

    n = rem_pio2(x, y)
    return _tan_kernel(y[0], y[1], n & 1)
```

#### 특수 케이스

| 입력 | 결과 |
|---|---|
| NaN | NaN |
| +/-Inf | NaN |
| +/-0 | +/-0 |
| pi/2 근방 | 대규모 값 (catastrophic cancellation 불가피) |

#### 성능/ULP

- 목표: 80~110ns/call
- ULP: <= 1 (pi/2 근방 제외)

---

### 4.4 arcsin(x) (asin)

#### 파일: `src/math_library/_core/inverse_trig.pyx`
#### 의존성: `_helpers.pxd`, `power_sqrt.pxd` (sqrt)

#### 계수 (musl `asin.c`, algorithm_reference.md 섹션 4)

```cython
cdef double ASIN_PIO2_HI = 1.57079632679489655800e+00  # 0x3FF921FB54442D18
cdef double ASIN_PIO2_LO = 6.12323399573676603587e-17  # 0x3C91A62633145C07

cdef double pS0 =  1.66666666666666657415e-01
cdef double pS1 = -3.25565818622400915405e-01
cdef double pS2 =  2.01212532134862925881e-01
cdef double pS3 = -4.00555345006794114027e-02
cdef double pS4 =  7.91534994289814532176e-04
cdef double pS5 =  3.47933107596021167570e-05

cdef double qS1 = -2.40339491173441421878e+00
cdef double qS2 =  2.02094576023350569471e+00
cdef double qS3 = -6.88283971605453293030e-01
cdef double qS4 =  7.70381505559019352791e-02
```

#### 내부 `_asin_R(double z)` (z = x^2)

```cython
cdef inline double _asin_R(double z) noexcept nogil:
    # P(z) = pS0 + z*(pS1 + z*(pS2 + z*(pS3 + z*(pS4 + z*pS5))))
    cdef double p = z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5))))
    p += pS0
    # Q(z) = 1 + z*(qS1 + z*(qS2 + z*(qS3 + z*qS4)))
    cdef double q = 1.0 + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)))
    return p / q
```

#### 공개 API `arcsin`

```cython
from .power_sqrt cimport sqrt as _sqrt_c
from ._helpers cimport double_to_bits, bits_to_double, high_word, low_word
from libc.stdint cimport uint32_t, uint64_t

cpdef double arcsin(double x) noexcept:
    cdef uint32_t hx = high_word(x)
    cdef uint32_t ix = hx & 0x7FFFFFFFU
    cdef double z, r, s, t, w, p, fx
    cdef uint64_t sbits
    cdef double df, c, result

    # |x| >= 1
    if ix >= 0x3FF00000U:
        if ix == 0x3FF00000U and low_word(x) == 0U:
            # |x| == 1: +/-pi/2
            if (hx >> 31):
                return -(ASIN_PIO2_HI + ASIN_PIO2_LO)
            return ASIN_PIO2_HI + ASIN_PIO2_LO
        return (x - x) / (x - x)  # NaN (|x| > 1 또는 NaN)

    # |x| < 0.5
    if ix < 0x3FE00000U:
        if ix < 0x3E500000U:  # |x| < 2^-26
            return x
        z = x * x
        r = _asin_R(z)
        return x + x * r

    # 0.5 <= |x| < 1: 항등식 asin(x) = pi/2 - 2*asin(sqrt((1-|x|)/2))
    if (hx >> 31):
        fx = -x
    else:
        fx = x
    w = 1.0 - fx
    t = w * 0.5
    p = t * _asin_R(t)
    s = _sqrt_c(t)
    # 고정밀 보정: s의 상위 32비트만 취한 df 사용
    sbits = double_to_bits(s) & 0xFFFFFFFF00000000ULL
    df = bits_to_double(sbits)
    c = (t - df * df) / (s + df)
    result = 2.0 * s * p + 2.0 * c
    result = ASIN_PIO2_HI - (2.0 * (df + result) - ASIN_PIO2_LO)
    # sim-engineer 주의: musl src/math/asin.c 47-70행의 원본 보정식을 그대로 쓰되
    # 위 식은 그 구조 요약임. 실구현은 musl 정확 복제.

    if (hx >> 31):
        return -result
    return result
```

**중요**: sim-engineer는 musl `src/math/asin.c` 원본의 큰-x 경로 보정 로직을 bit-exact 복제할 것.

#### 특수 케이스

| 입력 | 결과 |
|---|---|
| NaN | NaN |
| +/-1 | +/-pi/2 |
| \|x\| > 1 | NaN |
| +/-0 | +/-0 |
| \|x\| < 2^-26 | x 그대로 |

#### 성능/ULP

- 목표: 70~95ns/call
- ULP: <= 1

---

### 4.5 arccos(x) (acos)

#### 파일: `src/math_library/_core/inverse_trig.pyx`

**계수 공유**: asin과 동일한 `pS`, `qS`, `ASIN_PIO2_HI/LO` 계수 사용.

#### 공개 API `arccos`

```cython
cpdef double arccos(double x) noexcept:
    cdef uint32_t hx = high_word(x)
    cdef uint32_t ix = hx & 0x7FFFFFFFU
    cdef double z, s, w, r, p, c, df
    cdef uint64_t sbits

    # |x| >= 1
    if ix >= 0x3FF00000U:
        if ix == 0x3FF00000U and low_word(x) == 0U:
            if (hx >> 31):
                # acos(-1) = pi
                return 2.0 * ASIN_PIO2_HI + 2.0 * ASIN_PIO2_LO
            return 0.0  # acos(1) = 0
        return (x - x) / (x - x)  # NaN

    # |x| < 0.5
    if ix < 0x3FE00000U:
        if ix <= 0x3C600000U:  # |x| < 2^-57
            return ASIN_PIO2_HI + ASIN_PIO2_LO
        z = x * x
        p = z * _asin_R(z)
        # acos(x) = pi/2 - (x + x*R)
        return ASIN_PIO2_HI - (x - (ASIN_PIO2_LO - x * p))

    # x <= -0.5
    if (hx >> 31):
        z = (1.0 + x) * 0.5
        p = z * _asin_R(z)
        s = _sqrt_c(z)
        w = p * s - ASIN_PIO2_LO
        return 2.0 * (ASIN_PIO2_HI - (s + w))

    # x >= 0.5
    z = (1.0 - x) * 0.5
    s = _sqrt_c(z)
    sbits = double_to_bits(s) & 0xFFFFFFFF00000000ULL
    df = bits_to_double(sbits)
    c = (z - df * df) / (s + df)
    p = z * _asin_R(z)
    w = p * s + c
    return 2.0 * (df + w)
```

#### 특수 케이스

| 입력 | 결과 |
|---|---|
| NaN | NaN |
| 1 | +0 |
| -1 | pi |
| \|x\| > 1 | NaN |

#### 성능/ULP

- 목표: 70~95ns/call
- ULP: <= 1

---

### 4.6 arctan(x) (atan)

#### 파일: `src/math_library/_core/inverse_trig.pyx`

#### 계수 (musl `atan.c`, algorithm_reference.md 섹션 6)

```cython
cdef double[4] atanhi = [
     4.63647609000806093515e-01,  # atan(0.5)
     7.85398163397448278999e-01,  # atan(1) = pi/4
     9.82793723247329054082e-01,  # atan(1.5)
     1.57079632679489655800e+00,  # atan(inf) = pi/2
]
cdef double[4] atanlo = [
     2.26987774529616870924e-17,
     3.06161699786838301793e-17,
     1.39033110312309984516e-17,
     6.12323399573676603587e-17,
]
cdef double[11] aT = [
     3.33333333333329318027e-01,  # +1/3
    -1.99999999998764832476e-01,  # -1/5
     1.42857142725034663711e-01,  # +1/7
    -1.11111104054623557880e-01,  # -1/9
     9.09088713343650656196e-02,  # +1/11
    -7.69187620504482999495e-02,  # -1/13
     6.66107313738753120669e-02,  # +1/15
    -5.83357013379057348645e-02,  # -1/17
     4.97687799461593236017e-02,  # +1/19
    -3.65315727442169155270e-02,  # -1/21
     1.62858201153657823623e-02,  # +1/23
]
```

#### 공개 API `arctan`

**중요**: Cython은 **블록 중간 `cdef` 선언을 허용하지 않는다**. 모든 지역 변수를 함수 상단에서 선언할 것.

```cython
cpdef double arctan(double x) noexcept:
    cdef uint32_t hx = high_word(x)
    cdef uint32_t ix = hx & 0x7FFFFFFFU
    cdef double z, w, s1, s2, absx
    cdef int id

    # |x| >= 2^66 -> atan(inf)
    if ix >= 0x44100000U:
        if ix > 0x7FF00000U or (ix == 0x7FF00000U and low_word(x) != 0):
            return x + x  # NaN 전파
        if (hx >> 31):
            return -(ASIN_PIO2_HI + ASIN_PIO2_LO)
        return ASIN_PIO2_HI + ASIN_PIO2_LO

    # 5구간 분기
    if ix < 0x3FDC0000U:
        # |x| < 7/16
        if ix < 0x3E400000U:  # |x| < 2^-27
            return x
        id = -1
        # x는 원본 유지
    else:
        if (hx >> 31):
            absx = -x
        else:
            absx = x

        if ix < 0x3FF30000U:
            if ix < 0x3FE60000U:
                # 7/16 <= |x| < 11/16: atan(0.5) + atan((2x-1)/(2+x))
                id = 0
                absx = (2.0 * absx - 1.0) / (2.0 + absx)
            else:
                # 11/16 <= |x| < 19/16: atan(1) + atan((x-1)/(x+1))
                id = 1
                absx = (absx - 1.0) / (absx + 1.0)
        else:
            if ix < 0x40038000U:
                # 19/16 <= |x| < 39/16
                id = 2
                absx = (absx - 1.5) / (1.0 + 1.5 * absx)
            else:
                # 39/16 <= |x|
                id = 3
                absx = -1.0 / absx

        x = absx  # 이후 다항식 평가에서 |x| 사용

    # 다항식 평가: z = x^2, s1은 짝수 차수, s2는 홀수 차수
    z = x * x
    w = z * z
    s1 = z * (aT[0] + w * (aT[2] + w * (aT[4] + w * (aT[6] + w * (aT[8] + w * aT[10])))))
    s2 = w * (aT[1] + w * (aT[3] + w * (aT[5] + w * (aT[7] + w * aT[9]))))

    if id < 0:
        return x - x * (s1 + s2)

    z = atanhi[id] - ((x * (s1 + s2) - atanlo[id]) - x)
    if (hx >> 31):
        return -z
    return z
```

#### 특수 케이스

| 입력 | 결과 |
|---|---|
| NaN | NaN |
| +/-Inf | +/-pi/2 |
| +/-0 | +/-0 |
| \|x\| < 2^-27 | x 그대로 |

#### 성능/ULP

- 목표: 60~85ns/call
- ULP: <= 1

---

### 4.7 hypersin(x) (sinh)

#### 파일: `src/math_library/_core/hyperbolic.pyx`
#### 의존성: `exponential.pxd` (`_expm1_inline`, `_exp_inline`)

#### 전략 (algorithm_reference.md 섹션 7)

```
sinh(x) = (expm1(x) + expm1(x) / (expm1(x) + 1)) / 2
```

#### 공개 API

```cython
from ._helpers cimport high_word
from .exponential cimport _expm1_inline, _exp_inline
from libc.stdint cimport uint32_t

cpdef double hypersin(double x) noexcept:
    cdef uint32_t hx = high_word(x)
    cdef uint32_t ix = hx & 0x7FFFFFFFU
    cdef double h, absx, t

    # NaN, +/-Inf
    if ix >= 0x7FF00000U:
        return x + x

    # |x| < 2^-26 -> x 그대로
    if ix < 0x3E500000U:
        return x

    # 부호
    if (hx >> 31):
        h = -0.5
        absx = -x
    else:
        h = 0.5
        absx = x

    # |x| < log(DBL_MAX) ~ 709.78
    if ix < 0x40862E42U:
        t = _expm1_inline(absx)
        if ix < 0x3FF00000U:  # |x| < 1
            return h * (2.0 * t - t * t / (t + 1.0))
        return h * (t + t / (t + 1.0))

    # |x| >= log(DBL_MAX): overflow
    # TODO: __expo2 구현 (musl 참조)
    # 1차 스코프: 단순 오버플로우
    return 2.0 * h * 1.7976931348623157e+308  # +/-Inf
```

#### 특수 케이스

| 입력 | 결과 |
|---|---|
| NaN | NaN |
| +/-0 | +/-0 |
| +/-Inf | +/-Inf |
| \|x\| > 710 | +/-Inf (overflow) |

#### 성능/ULP

- 목표: 100~150ns/call (exp 의존)
- ULP: <= 2

---

### 4.8 hypercos(x) (cosh)

#### 파일: `src/math_library/_core/hyperbolic.pyx`

#### 전략 (algorithm_reference.md 섹션 8)

```
|x| < log(2)      -> cosh = 1 + t^2 / (2*(1+t)), t = expm1(|x|)
|x| < log(DBLMAX) -> cosh = 0.5 * (e + 1/e), e = exp(|x|)
|x| >= ...         -> __expo2 (TODO) or overflow
```

#### 공개 API

```cython
cpdef double hypercos(double x) noexcept:
    cdef uint32_t hx = high_word(x)
    cdef uint32_t ix = hx & 0x7FFFFFFFU
    cdef double t, absx

    if ix >= 0x7FF00000U:
        return x * x  # NaN 전파 or Inf->Inf

    if (hx >> 31):
        absx = -x
    else:
        absx = x

    # |x| < log(2) ~ 0.693
    if ix < 0x3FE62E42U:
        if ix < 0x3C800000U:  # |x| < 2^-55
            return 1.0
        t = _expm1_inline(absx)
        return 1.0 + t * t / (2.0 * (1.0 + t))

    # log(2) <= |x| < log(DBLMAX)
    if ix < 0x40862E42U:
        t = _exp_inline(absx)
        return 0.5 * t + 0.5 / t

    # overflow
    return 1.7976931348623157e+308 * 2.0
```

#### 특수 케이스

| 입력 | 결과 |
|---|---|
| NaN | NaN |
| +/-0 | 1.0 |
| +/-Inf | +Inf |

#### 성능/ULP

- 목표: 100~150ns/call
- ULP: <= 2

---

### 4.9 hypertan(x) (tanh)

#### 파일: `src/math_library/_core/hyperbolic.pyx`

#### 전략 (algorithm_reference.md 섹션 9)

```
|x| > 20                -> sign(x) * 1
log(3)/2 < |x| <= 20    -> sign(x) * (1 - 2/(e^{2|x|} + 1))
log(5/3)/2 < |x| <= log(3)/2 -> t/(t+2), t = expm1(2|x|)
|x| <= log(5/3)/2       -> -t/(t+2), t = expm1(-2|x|)
|x| < 2^-55             -> x(1 + x)
```

#### 공개 API

```cython
cpdef double hypertan(double x) noexcept:
    cdef uint32_t hx = high_word(x)
    cdef uint32_t ix = hx & 0x7FFFFFFFU
    cdef double t, z, absx, sign

    # NaN or +/-Inf
    if ix >= 0x7FF00000U:
        if ix > 0x7FF00000U:
            return x + x  # NaN
        if (hx >> 31):
            return -1.0
        return 1.0

    # |x| < 2^-28
    if ix < 0x3E300000U:
        return x * (1.0 + x)

    if (hx >> 31):
        absx = -x
        sign = -1.0
    else:
        absx = x
        sign = 1.0

    # |x| > log(3)/2 ~ 0.549
    if ix > 0x3FE193EAU:
        if ix > 0x40340000U:  # |x| > 20
            return sign
        t = _expm1_inline(2.0 * absx)
        z = 1.0 - 2.0 / (t + 2.0)
    elif ix > 0x3FD058AEU:  # |x| > log(5/3)/2 ~ 0.255
        t = _expm1_inline(2.0 * absx)
        z = t / (t + 2.0)
    else:
        t = _expm1_inline(-2.0 * absx)
        z = -t / (t + 2.0)

    return sign * z
```

#### 특수 케이스

| 입력 | 결과 |
|---|---|
| NaN | NaN |
| +/-0 | +/-0 |
| +/-Inf | +/-1 |

#### 성능/ULP

- 목표: 100~165ns/call
- ULP: <= 2

---

### 4.10 exp(x) — 신규

#### 파일: `src/math_library/_core/exponential.pyx`
#### 의존성: `_helpers.pxd`, `libc.math.ldexp`

#### 전략 — 1차 간략 방식

algorithm_reference.md 섹션 10의 **테이블 기반 방식**은 최종 목표이나, 1차 구현에서는 **Cody-Waite + 5항 Remez 다항식**으로 시작 (구현 복잡도/성능 트레이드오프).

```
exp(x) = 2^k * exp(r)
k = round(x / ln2)
r = x - k*ln2_hi - k*ln2_lo  (Cody-Waite 2-step)
exp(r) ~ 1 + r + r^2 * P(r^2)
```

#### 계수 (musl `exp.c` 간략형)

```cython
# Cody-Waite ln2 분할
cdef double LN2_HI = 6.93147180369123816490e-01  # 0x3FE62E42FEE00000
cdef double LN2_LO = 1.90821492927058770002e-10  # 0x3DEA39EF35793C76
cdef double INV_LN2 = 1.44269504088896338700e+00  # 0x3FF71547652B82FE

# 5항 Remez (musl 간략 exp)
cdef double P1 =  1.66666666666666019037e-01  # ~1/6
cdef double P2 = -2.77777777770155933842e-03
cdef double P3 =  6.61375632143793436117e-05
cdef double P4 = -1.65339022054652515390e-06
cdef double P5 =  4.13813679705723846039e-08
```

#### 내부 `_exp_inline(double x)`

```cython
from libc.math cimport ldexp as _ldexp

cdef inline double _exp_inline(double x) noexcept nogil:
    cdef uint32_t hx = high_word(x)
    cdef uint32_t ix = hx & 0x7FFFFFFFU
    cdef double hi, lo, t, c, y, xr
    cdef int k

    # |x| < 2^-28 -> 1 + x
    if ix < 0x3E300000U:
        return 1.0 + x

    # overflow / underflow 경계
    if ix >= 0x40862E42U:  # |x| >= 709.78
        if (hx >> 31):
            return 0.0  # underflow
        return 1.7976931348623157e+308 * 2.0  # overflow

    # Cody-Waite argument reduction
    if (hx >> 31) == 0:
        k = <int>(INV_LN2 * x + 0.5)
    else:
        k = <int>(INV_LN2 * x - 0.5)
    hi = x - k * LN2_HI
    lo = k * LN2_LO
    xr = hi - lo

    # 5항 Remez
    t = xr * xr
    c = xr - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))))
    y = 1.0 + (xr * c / (2.0 - c) - lo + hi)
    return _ldexp(y, k)
```

#### 공개 API `exp`

```cython
cpdef double exp(double x) noexcept:
    cdef uint32_t hx = high_word(x)
    cdef uint32_t ix = hx & 0x7FFFFFFFU
    if ix >= 0x7FF00000U:
        if ix > 0x7FF00000U or low_word(x) != 0:
            return x + x  # NaN
        if (hx >> 31):
            return 0.0  # -Inf
        return x  # +Inf
    return _exp_inline(x)
```

#### 내부 `_expm1_inline(double x)`

sinh/cosh/tanh가 의존. 정확한 `expm1`은 musl 137줄. 1차 구현은 간략화:

```cython
cdef inline double _expm1_inline(double x) noexcept nogil:
    cdef uint32_t hx = high_word(x)
    cdef uint32_t ix = hx & 0x7FFFFFFFU

    # |x| < 2^-54 -> x
    if ix < 0x3C900000U:
        return x

    # |x| > 56*log(2) ~ 38.8 -> exp(x) - 1
    if ix >= 0x4043687AU:
        if (hx >> 31):
            return -1.0
        return _exp_inline(x) - 1.0

    # 그 외: exp(x) - 1 직접 계산
    # 주의: x가 0 근방일 때 cancellation 발생하므로 musl의 전용 expm1 구현이 이상적
    # 1차는 간략화 수용 (ULP <= 2)
    return _exp_inline(x) - 1.0
```

**주의**: 이 간략 `expm1`은 `|x|` 작을 때 정밀도 저하. 필요 시 musl `src/math/expm1.c` 정밀 복제 업그레이드.

#### 특수 케이스

| 입력 | 결과 |
|---|---|
| NaN | NaN |
| +Inf | +Inf |
| -Inf | +0 |
| x > 709.78 | +Inf |
| x < -745.13 | +0 |

#### 성능/ULP

- 목표: 75~105ns/call
- ULP: <= 1 (간략), <= 0.5 (테이블 업그레이드 시)

---

### 4.11 ln(x) — 신규

#### 파일: `src/math_library/_core/logarithmic.pyx`
#### 의존성: `_helpers.pxd`, `libc.math.frexp`

#### 전략 — 1차 간략 방식

algorithm_reference.md 섹션 11의 **테이블 기반 방식**은 최종 목표. 1차 구현은 `frexp`로 지수 추출 + 유리 근사.

```
ln(x) = k*ln2 + ln(m)  where x = m * 2^k, m in [1/sqrt(2), sqrt(2)]
ln(m) = 2*atanh((m-1)/(m+1))
      ~ 2*s*P(s^2) with s = (m-1)/(m+1)
```

#### 계수 (musl `log.c` 간략형)

```cython
# 1차 간략 계수 (7항). 정확한 bit-exact 계수는 musl src/math/__log.c 참조
cdef double LOG_P0 = 6.666666666666735130e-01
cdef double LOG_P1 = 3.999999999940941908e-01
cdef double LOG_P2 = 2.857142874366239149e-01
cdef double LOG_P3 = 2.222219843214978396e-01
cdef double LOG_P4 = 1.818357216161805012e-01
cdef double LOG_P5 = 1.531383769920937332e-01
cdef double LOG_P6 = 1.479819860511658591e-01

cdef double LN2_HI_LOG = 6.93147180369123816490e-01
cdef double LN2_LO_LOG = 1.90821492927058770002e-10
```

#### 내부 `_ln_inline(double x)`

```cython
from libc.math cimport frexp as _frexp

cdef inline double _ln_inline(double x) noexcept nogil:
    cdef int k
    cdef double f, s, z, R, w, hfsq, dk

    # x == 0, x < 0, +Inf, NaN은 공개 API에서 처리

    # x = m * 2^k with m in [0.5, 1)
    f = _frexp(x, &k)
    if f < 0.7071067811865476:  # 1/sqrt(2)
        k -= 1
        f *= 2.0
    f -= 1.0  # f in [-0.293, 0.414]

    dk = <double>k
    s = f / (2.0 + f)
    z = s * s
    w = z * z

    # 7항 Horner 분리 (짝수/홀수)
    R = w * (LOG_P1 + w * (LOG_P3 + w * LOG_P5)) \
      + z * (LOG_P0 + w * (LOG_P2 + w * (LOG_P4 + w * LOG_P6)))

    hfsq = 0.5 * f * f
    return dk * LN2_HI_LOG - ((hfsq - (s * (hfsq + R) + dk * LN2_LO_LOG)) - f)
```

#### 공개 API `ln`

```cython
cpdef double ln(double x) noexcept:
    cdef uint32_t hx = high_word(x)
    cdef uint32_t ix = hx & 0x7FFFFFFFU

    # x == 0 -> -Inf
    if ix == 0U and low_word(x) == 0:
        return -1.0 / 0.0
    # x < 0 -> NaN
    if (hx >> 31):
        return (x - x) / 0.0
    # x == +Inf -> +Inf; NaN -> NaN
    if ix >= 0x7FF00000U:
        return x + x
    return _ln_inline(x)
```

#### 공개 API `log(base, x)`

```cython
cpdef double log(double base, double x) noexcept:
    # log_base(x) = ln(x) / ln(base)
    if base <= 0.0 or base == 1.0:
        return (x - x) / (x - x)  # NaN
    if x <= 0.0:
        # x == 0 -> -Inf (ln 내부에서 처리)
        # x < 0 -> NaN
        pass
    return ln(x) / ln(base)
```

#### 특수 케이스

| 입력 | 결과 |
|---|---|
| NaN | NaN |
| +Inf | +Inf |
| +0 | -Inf |
| x < 0 | NaN |
| 1 | +0 |

#### 성능/ULP

- 목표: 80~110ns/call
- ULP: <= 1 (간략), <= 0.5 (테이블 업그레이드 시)

---

### 4.12 log(base, x) — 재구현

§4.11에 포함. (ln의 비율로 구현)

---

### 4.13 power(base, exp) (pow)

#### 파일: `src/math_library/_core/power_sqrt.pyx`
#### 의존성: `exponential.pxd` (`_exp_inline`), `logarithmic.pxd` (`_ln_inline`)

#### 전략 (algorithm_reference.md 섹션 13)

```
pow(x, y) = exp(y * ln(x))
```

**정수 지수**는 이진 거듭제곱으로 분기 (오차 누적 방지).

#### 계수/헬퍼

```cython
from .exponential cimport _exp_inline
from .logarithmic cimport _ln_inline
from libc.math cimport floor as _floor, fabs as _fabs
from libc.stdint cimport int64_t

cdef inline int _check_int(double y) noexcept nogil:
    """0: 비정수, 1: 홀수 정수, 2: 짝수 정수"""
    cdef double ay = _fabs(y)
    cdef double f
    cdef int64_t n
    if ay > 9.007199254740992e+15:  # 2^53
        return 2  # 큰 실수는 짝수로 간주
    if ay < 1.0:
        return 0
    f = _floor(ay)
    if f != ay:
        return 0
    n = <int64_t>f
    return 1 if (n & 1) else 2

cdef inline double _pow_int_real(double base, int64_t n) noexcept nogil:
    cdef double result = 1.0
    cdef double b = base
    cdef int negative = 0
    if n < 0:
        negative = 1
        n = -n
    while n > 0:
        if n & 1:
            result *= b
        b *= b
        n >>= 1
    if negative:
        return 1.0 / result
    return result
```

#### 공개 API `power`

```cython
cpdef double power(double base, double exponent) noexcept:
    cdef uint32_t hx, ix, hy, iy
    cdef int int_kind
    cdef int64_t n

    hx = high_word(base)
    ix = hx & 0x7FFFFFFFU
    hy = high_word(exponent)
    iy = hy & 0x7FFFFFFFU

    # y == 0 -> 1 (NaN 포함 모든 x에 대해 IEEE 754-2019)
    if exponent == 0.0:
        return 1.0
    # x == 1 -> 1
    if base == 1.0:
        return 1.0
    # NaN 전파
    if ix > 0x7FF00000U or iy > 0x7FF00000U:
        return base + exponent
    # x == 0
    if base == 0.0:
        if exponent < 0.0:
            return 1.0 / 0.0  # +Inf
        return 0.0

    # 정수 지수 검출
    int_kind = _check_int(exponent)
    if int_kind != 0:
        n = <int64_t>exponent
        return _pow_int_real(base, n)

    # 음수 밑 + 비정수 지수 -> NaN
    if base < 0.0:
        return (base - base) / (base - base)

    # 일반: exp(y * ln(x))
    return _exp_inline(exponent * _ln_inline(base))
```

#### 특수 케이스 (IEEE 754-2019 §9.2.1)

| 입력 | 결과 |
|---|---|
| pow(x, +/-0) | 1 (모든 x, NaN 포함) |
| pow(1, y) | 1 (모든 y, NaN 포함) |
| pow(NaN, y!=0) | NaN |
| pow(x, NaN) | NaN (x!=1) |
| pow(-1, +/-Inf) | 1 |
| pow(+0, y<0) | +Inf |
| pow(-0, y<0 홀수정수) | -Inf |
| pow(x<0, 비정수) | NaN |

**주의**: 위 표의 모든 엣지 케이스 처리는 sim-engineer가 IEEE 754-2019 §9.2.1 직접 참조 구현.

#### 성능/ULP

- 목표 (일반): 80~150ns/call
- 목표 (정수 지수): 20~40ns/call
- ULP: <= 2 (log + exp 오차 합산)

---

### 4.14 sqrt(x) — 신규

#### 파일: `src/math_library/_core/power_sqrt.pyx`

#### 전략

algorithm_reference.md 섹션 14: **`libc.math.sqrt` 직접 호출**. 하드웨어 `sqrtsd` (x86) 또는 `fsqrt` (ARM) 단일 명령. IEEE 754 correctly rounded (<= 0.5 ULP) 필수 보장.

```cython
from libc.math cimport sqrt as _sqrt_c

cpdef double sqrt(double x) noexcept:
    return _sqrt_c(x)
```

**중요**: "자체 구현 철학"의 명시적 예외 (algorithm_reference.md §14 및 cython_best_practices.md 허용).

#### 특수 케이스

| 입력 | 결과 |
|---|---|
| NaN | NaN |
| +Inf | +Inf |
| +/-0 | +/-0 |
| x < 0 | NaN |

#### 성능/ULP

- 목표: 25~30ns/call (math.sqrt와 동등)
- ULP: <= 0.5

---

### 4.15 sec, cosec, cotan (파생)

#### 파일: `src/math_library/_core/trigonometric.pyx` (sin/cos 아래 추가)

```cython
cpdef double sec(double x) noexcept:
    cdef double c = cos(x)
    return 1.0 / c  # 분모 0이면 +/-Inf 자연 반환 (IEEE 754)

cpdef double cosec(double x) noexcept:
    cdef double s = sin(x)
    return 1.0 / s

cpdef double cotan(double x) noexcept:
    cdef double s = sin(x)
    cdef double c = cos(x)
    return c / s
```

**주의**: 기존 Python 래퍼 (`trigonometric_function/sec.py`)는 `cos(x) ~= 0`일 때 `ZeroDivisionError` 발생. Cython 고속 경로는 `+/-Inf` 반환. **에러 체크는 래퍼에서 유지**.

#### 성능/ULP

- 목표: 70~110ns/call (cos + 1 division)
- ULP: <= 2

---

### 4.16 arcsec, arccosec, arccotan (파생)

#### 파일: `src/math_library/_core/inverse_trig.pyx`

```cython
cpdef double arcsec(double x) noexcept:
    # arcsec(x) = arccos(1/x), |x| >= 1
    return arccos(1.0 / x)

cpdef double arccosec(double x) noexcept:
    # arccosec(x) = arcsin(1/x), |x| >= 1
    return arcsin(1.0 / x)

cpdef double arccotan(double x) noexcept:
    # arccotan(x) = pi/2 - arctan(x) (주 branch)
    return (ASIN_PIO2_HI + ASIN_PIO2_LO) - arctan(x)
```

**주의**: 정의역 검증 (|x| >= 1 등)은 **래퍼**에서 수행. 고속 경로는 NaN/Inf 자연 발생 허용.

#### 성능/ULP

- 목표: 80~125ns/call
- ULP: <= 2

---

### 4.17 hypersec, hypercosec, hypercotan (파생)

#### 파일: `src/math_library/_core/hyperbolic.pyx`

```cython
cpdef double hypersec(double x) noexcept:
    return 1.0 / hypercos(x)

cpdef double hypercosec(double x) noexcept:
    return 1.0 / hypersin(x)

cpdef double hypercotan(double x) noexcept:
    return hypercos(x) / hypersin(x)
```

#### 성능/ULP

- 목표: 120~200ns/call (cosh/sinh 의존)
- ULP: <= 3

---

## 5. `__rem_pio2` Argument Reduction 설계

### 5.1 파일: `src/math_library/_core/argument_reduction.pyx`

### 5.2 전략

| 범위 | 방법 |
|---|---|
| `\|x\| < 2^-26` | 반환 불필요 (호출자가 스킵) |
| `2^-26 <= \|x\| < 3*pi/4` | 1단계 reduction: `x - pi/2` |
| `3*pi/4 <= \|x\| < 2^20*pi/2` | **Cody-Waite 3단계** (`pio2_1`, `pio2_2`, `pio2_3`) |
| `\|x\| >= 2^20*pi/2` | **1차 스코프 생략**: `fmod(x, 2*pi)` 폴백 (TODO: Payne-Hanek) |

### 5.3 계수 (algorithm_reference.md 섹션 1)

```cython
cdef double PIO2_1  = 1.57079632673412561417e+00  # 0x3FF921FB54400000
cdef double PIO2_1T = 6.07710050650619224932e-11  # 0x3DD0B4611A626331
cdef double PIO2_2  = 6.07710050630396597660e-11  # 0x3DD0B4611A600000
cdef double PIO2_2T = 2.02226624879595063154e-21  # 0x3BA3198A2E037073
cdef double PIO2_3  = 2.02226624871116645580e-21  # 0x3BA3198A2E000000
cdef double PIO2_3T = 8.47842766036889956997e-32  # 0x397B839A252049C1

cdef double INV_PIO2 = 6.36619772367581382433e-01  # 2/pi
```

### 5.4 `.pxd` 선언

```cython
# argument_reduction.pxd
cdef int rem_pio2(double x, double *y) noexcept nogil
```

### 5.5 `.pyx` 구현 의사코드

```cython
# argument_reduction.pyx
from libc.math cimport fmod as _fmod, fabs as _fabs
from ._helpers cimport double_to_bits, high_word, low_word
from libc.stdint cimport uint32_t, uint64_t

cdef int rem_pio2(double x, double *y) noexcept nogil:
    cdef uint32_t hx = high_word(x)
    cdef uint32_t ix = hx & 0x7FFFFFFFU
    cdef int n, sign
    cdef double z, w, t, r, fn
    cdef uint32_t high_y0

    # |x| <= pi/4: 호출자가 이미 스킵하지만 안전망
    if ix <= 0x3FE921FBU:
        y[0] = x
        y[1] = 0.0
        return 0

    sign = 1 if (hx >> 31) == 0 else -1

    # |x| < 3*pi/4: 1단계 reduction
    if ix < 0x4002D97CU:
        if sign > 0:
            z = x - PIO2_1
            if ix != 0x3FF921FBU:
                y[0] = z - PIO2_1T
                y[1] = (z - y[0]) - PIO2_1T
            else:
                z = z - PIO2_2
                y[0] = z - PIO2_2T
                y[1] = (z - y[0]) - PIO2_2T
            return 1
        else:
            z = x + PIO2_1
            if ix != 0x3FF921FBU:
                y[0] = z + PIO2_1T
                y[1] = (z - y[0]) + PIO2_1T
            else:
                z = z + PIO2_2
                y[0] = z + PIO2_2T
                y[1] = (z - y[0]) + PIO2_2T
            return -1

    # |x| < 2^20 * pi/2: 3단계 Cody-Waite
    if ix < 0x413921FBU:
        t = _fabs(x)
        n = <int>(t * INV_PIO2 + 0.5)
        fn = <double>n
        r = t - fn * PIO2_1
        w = fn * PIO2_1T
        y[0] = r - w
        high_y0 = <uint32_t>(double_to_bits(y[0]) >> 32) & 0x7FFFFFFFU

        # 고정밀 보정: 필요 시 PIO2_2, PIO2_3 추가 단계
        if (ix - high_y0) > 0x13700000U:
            t = r
            w = fn * PIO2_2
            r = t - w
            w = fn * PIO2_2T - ((t - r) - w)
            y[0] = r - w
            high_y0 = <uint32_t>(double_to_bits(y[0]) >> 32) & 0x7FFFFFFFU
            if (ix - high_y0) > 0x15F00000U:
                t = r
                w = fn * PIO2_3
                r = t - w
                w = fn * PIO2_3T - ((t - r) - w)
                y[0] = r - w

        y[1] = (r - y[0]) - w
        if sign < 0:
            y[0] = -y[0]
            y[1] = -y[1]
            return -n
        return n

    # |x| >= 2^20 * pi/2: Payne-Hanek 미구현, fmod 폴백
    # TODO: musl __rem_pio2_large.c (~1500 줄) 이식
    # 1차 스코프: 정확도 저하 수용
    z = _fmod(x, 6.283185307179586)  # 2*pi
    if z > 3.141592653589793:
        z -= 6.283185307179586
    elif z < -3.141592653589793:
        z += 6.283185307179586
    if z > 0:
        n = <int>(z * INV_PIO2 + 0.5)
    else:
        n = <int>(z * INV_PIO2 - 0.5)
    y[0] = z - n * 1.5707963267948966  # pi/2
    y[1] = 0.0
    return n
```

**중요**: 위 Cody-Waite 고정밀 보정 분기 (`(ix - high_y0) > 0x13700000U`)는 musl `src/math/__rem_pio2.c` 원본 그대로 복제할 것. 잘못 구현하면 pi/2 근방에서 오차 폭증.

### 5.6 Payne-Hanek 트레이드오프 (중요 결정)

- **1차 스코프에서 Payne-Hanek 생략**
  - 이유: 구현 복잡도 (~1500 줄) + 실제 기계공학/제어 시뮬레이션에서 `|x| >= 2^20*pi/2 ~ 1.65e6`인 입력은 매우 희귀
  - 영향: `|x| >= 2^20*pi/2`에서 sin/cos/tan 정확도 1~수 ULP -> 수백 ULP까지 저하 가능
- **완화**:
  - 래퍼(`trigonometric_function/sin.py`)에서 큰 인수 입력 시 경고 로그 (선택적)
  - `README.md` 및 docstring에 "`|x| >= 1.65e6`에서 정확도 저하" 명시
- **TODO 등재**: `|x| >= 2^20*pi/2` 에서 Payne-Hanek 필요 (`musl __rem_pio2_large.c` 포팅)

---

## 6. 특수 함수 변환 전략

9개 모듈의 알고리즘은 **유지**, Cython 타입 선언 + elementary primitive 교체만 수행.

### 6.1 공통 변환 템플릿

`.py` -> `.pyx` 변환 시 파일 상단:

```cython
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False

from libc.math cimport fabs as _fabs, isnan as _isnan, floor as _floor
from libc.stdint cimport int64_t, uint32_t
from .._core._helpers cimport high_word, low_word
from .._core.exponential cimport _exp_inline
from .._core.logarithmic cimport _ln_inline
from .._core.trigonometric cimport sin as _sin_c, cos as _cos_c
from .._core.power_sqrt cimport sqrt as _sqrt_c, power as _power_c
```

**원칙**:

1. **`_validate_*` 검증 함수**: 순수 Python으로 유지 (공개 API 진입부에서만 호출, 내부 핫패스에서 금지)
2. **내부 헬퍼**: `cdef inline double _helper(double x) noexcept nogil` 형태로 변경
3. **공개 API**: `cpdef double gamma_real(double x) noexcept` + `def gamma(x, tol=..., number_system="real")` 래퍼 공존
4. **실수 모드 핫패스**: Cython `cpdef`로 직접 계산 (Python 경계 최소화)
5. **복소수 모드**: 기존 Python 경로 유지 (10절 참조)

### 6.2 모듈별 변경점

#### 6.2.1 gamma.pyx

- **알고리즘**: Lanczos (계수 9개 유지) + 반사 공식
- **교체 사항**:
  - `from ..trigonometric_function.sin import sin` -> `from .._core.trigonometric cimport sin as _sin_c`
  - `power(2.0 * pi(), 0.5, ...)` -> `_sqrt_c(6.283185307179586)` (상수 미리 계산)
  - `power(e(), -t, ...)` -> `_exp_inline(-t)`
  - `power(t, z_shifted + 0.5, ...)` -> `_exp_inline((z_shifted + 0.5) * _ln_inline(t))`
- **실수 모드 핵심 함수 신규**:

  ```cython
  cdef double[9] LANCZOS_COEFFS = [
       0.9999999999998099,
       676.5203681218851,
      -1259.1392167224028,
       771.3234287776531,
      -176.6150291621406,
       12.507343278686905,
      -0.13857109526572012,
       0.000009984369578019572,
       0.00000015056327351493116,
  ]
  cdef double LANCZOS_G = 7.0
  cdef double SQRT_2PI = 2.5066282746310002  # sqrt(2*pi)

  cdef double _gamma_lanczos_real(double z) noexcept nogil:
      cdef double acc, t, z_shifted
      cdef int i
      if z < 0.5:
          # 반사 공식
          return 3.141592653589793 / (_sin_c(3.141592653589793 * z) * _gamma_lanczos_real(1.0 - z))
      z_shifted = z - 1.0
      acc = LANCZOS_COEFFS[0]
      for i in range(1, 9):
          acc += LANCZOS_COEFFS[i] / (z_shifted + <double>i)
      t = z_shifted + LANCZOS_G + 0.5
      return SQRT_2PI * _exp_inline((z_shifted + 0.5) * _ln_inline(t)) * _exp_inline(-t) * acc

  cpdef double gamma_real(double x) noexcept:
      if _isnan(x):
          return x
      if x <= 0.0 and _floor(x) == x:
          return (x - x) / (x - x)  # pole
      return _gamma_lanczos_real(x)
  ```

- **호환 래퍼**: 기존 `def gamma(x, tol=None, number_system="real")`는 `number_system == "real"`일 때 `gamma_real` 호출, 복소수 모드는 기존 Python 로직 유지.
- **예상 속도**: 10x (Python sin/power 호출 제거 효과)

#### 6.2.2 beta.pyx

- **알고리즘**: `B(x, y) = Gamma(x)*Gamma(y) / Gamma(x+y)` 유지
- **변경**: `gamma()` 호출을 `gamma_real()` (Cython)로 교체
- **예상 속도**: gamma 3x 개선 -> beta도 동일 비율

#### 6.2.3 bessel.pyx

- 테일러 급수 + 재귀 관계 유지
- 루프 카운터 `cdef int n`으로 타입 명시
- `power` -> `_power_c` 또는 `_exp_inline * _ln_inline`
- `gamma` 호출 -> `gamma_real`

#### 6.2.4 legendre.pyx

- Rodrigues 공식 또는 재귀:
  $$P_{n+1}(x) = \frac{(2n+1) x P_n(x) - n P_{n-1}(x)}{n+1}$$
- 순수 산술 -> `cdef double`/`cdef int` 타입 명시만 적용
- 예상 속도: 10x

#### 6.2.5 lambert_w.pyx

- Halley 또는 Newton 반복
- `exp` -> `_exp_inline`
- `ln` -> `_ln_inline`
- 수렴 판정 `while` 루프를 `cdef int iter` + `while iter < max_iter:`로 타입 명시

#### 6.2.6 zeta.pyx

- Euler-Maclaurin 또는 직접 급수 또는 반사 공식
- `power(n, -s)` -> `_exp_inline(-s * _ln_inline(<double>n))`
- 합산 루프 타입 명시

#### 6.2.7 euler_pi.pyx

- pi/e 관련 상수 계산 또는 L함수 합산
- 순수 산술 루프 타입 명시

#### 6.2.8 heaviside.pyx

- 단순 분기: `H(x) = 0 if x<0 else 1 if x>0 else 0.5`
- `cpdef double heaviside(double x) noexcept`로 선언만

#### 6.2.9 gcd.pyx, lcm.pyx

- 정수 연산, 유클리드 호제법
- `cpdef int64_t gcd(int64_t a, int64_t b) noexcept`
- lcm은 `a // gcd(a, b) * b`

### 6.3 `_validate_*` 유지 방침

- **원칙**: validate 함수는 **공개 API의 `def` 래퍼**에서만 호출
- **내부 `cdef` 함수**: validate 절대 호출 금지 (Python 경계 진입 방지)
- **`cpdef` 고속 경로**: validate는 사용자에게 맡기거나, 최소한의 `_isnan` 검사만 수행

### 6.4 예상 성능 향상

| 모듈 | 현재 (Python) | Cython 목표 | 배율 |
|---|---|---|---|
| gamma (실수) | 50~200μs/call | 5~20μs/call | 10x |
| beta | 100~400μs/call | 10~40μs/call | 10x |
| bessel | 500μs ~ 5ms/call | 50~500μs/call | 10x |
| legendre (n=10) | 50μs | 5μs | 10x |
| lambert_w | 100~500μs | 10~50μs | 10x |
| zeta | 100μs~1ms | 10~100μs | 10x |
| euler_pi | 100μs | 10μs | 10x |
| heaviside | 0.5μs | 0.2μs | 2.5x |
| gcd | 1μs | 0.3μs | 3x |
| lcm | 1.5μs | 0.4μs | 4x |

(실제 값은 벤치마크로 검증 필요)

---

## 7. Differentiation Ridders 재설계

### 7.1 현재 구현의 치명적 결함

- `h`를 `1e-3`에서 반감시키며 수렴 판정
- 중심차분 최적 `h ~ eps^{1/3} ~ 6e-6`을 지나면 roundoff 폭증
- 수렴 전 roundoff가 truncation error를 추월 → 부정확한 결과
- 현재 default `tol=1e-8`이 double precision 한계 (~1e-16)에 비해 느슨하여 실제 오차는 훨씬 큼
- 결과적으로 `sin의 도함수 cos(x)` 계산 시 7자리 정도만 맞고 그 이상은 불규칙

### 7.2 Ridders' 방법

**Richardson extrapolation table**로 `h`와 `h/1.4, h/1.4^2, ...`의 중심차분 결과를 조합하여 고차 오차 항을 상쇄:

$$A_{i,j} = \frac{A_{i,j-1} \cdot 1.4^{2j} - A_{i-1,j-1}}{1.4^{2j} - 1}$$

- `A[0][0] = D(h_0)`: 초기 중심차분
- h 감소: `h_i = h_{i-1} / 1.4` (`1.4` = 7/5, `1.4^2` = 1.96)
- 오차 추정: `err[i][j] = max(|A[i][j] - A[i][j-1]|, |A[i][j] - A[i-1][j-1]|)`
- 최소 err 기록, err 증가 시 중단 (roundoff 진입 신호)

**참고**: W. H. Press et al., "Numerical Recipes in C", §5.7 "Numerical Derivatives".

### 7.3 의사코드 (핵심)

```cython
# differentiation.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport fabs as _fabs, isfinite as _isfinite, INFINITY

cdef int NTAB = 10
cdef double CON = 1.4
cdef double CON2 = 1.96  # CON * CON
cdef double SAFE = 2.0

cdef double _ridders(object func, double x, double h0, double tol):
    """Ridders extrapolation for central difference.
    Returns best approximation of f'(x)."""
    cdef double[10][10] A
    cdef double hh = h0
    cdef double errt, fac, ans, err
    cdef int i, j

    if h0 == 0.0:
        raise ValueError("h0 must be non-zero")

    ans = 0.0
    err = INFINITY

    A[0][0] = (<double>func(x + hh) - <double>func(x - hh)) / (2.0 * hh)

    for i in range(1, NTAB):
        hh = hh / CON
        A[i][0] = (<double>func(x + hh) - <double>func(x - hh)) / (2.0 * hh)
        fac = CON2
        for j in range(1, i + 1):
            A[i][j] = (A[i][j - 1] * fac - A[i - 1][j - 1]) / (fac - 1.0)
            fac = fac * CON2
            errt = _fabs(A[i][j] - A[i][j - 1])
            if _fabs(A[i][j] - A[i - 1][j - 1]) > errt:
                errt = _fabs(A[i][j] - A[i - 1][j - 1])
            if errt <= err:
                err = errt
                ans = A[i][j]
                if err < tol:
                    return ans
        # Roundoff 감지: 가장 바깥 열이 이전보다 SAFE배 이상 벌어지면 중단
        if _fabs(A[i][i] - A[i - 1][i - 1]) >= SAFE * err:
            return ans

    return ans
```

**중요**:
- `func`는 Python callable이므로 `nogil` 불가 → 함수 시그니처에 `nogil` 미포함
- 그러나 `func()` 호출이 비용 dominant이므로 nogil 여부는 전체 성능에 미치는 영향 적음

### 7.4 `_ridders_2nd` (2차 미분용)

```cython
cdef double _ridders_2nd(object func, double x, double h0, double tol):
    cdef double[10][10] A
    cdef double hh = h0
    cdef double f_x, errt, fac, ans, err
    cdef int i, j

    f_x = <double>func(x)
    ans = 0.0
    err = INFINITY

    A[0][0] = (<double>func(x + hh) - 2.0 * f_x + <double>func(x - hh)) / (hh * hh)

    for i in range(1, NTAB):
        hh = hh / CON
        A[i][0] = (<double>func(x + hh) - 2.0 * f_x + <double>func(x - hh)) / (hh * hh)
        fac = CON2
        for j in range(1, i + 1):
            A[i][j] = (A[i][j - 1] * fac - A[i - 1][j - 1]) / (fac - 1.0)
            fac = fac * CON2
            errt = _fabs(A[i][j] - A[i][j - 1])
            if _fabs(A[i][j] - A[i - 1][j - 1]) > errt:
                errt = _fabs(A[i][j] - A[i - 1][j - 1])
            if errt <= err:
                err = errt
                ans = A[i][j]
                if err < tol:
                    return ans
        if _fabs(A[i][i] - A[i - 1][i - 1]) >= SAFE * err:
            return ans

    return ans
```

### 7.5 각 메서드 적용

`Differentiation` 클래스의 메서드 별 적용:

| 메서드 | Ridders 적용 방식 |
|---|---|
| `single_variable(f, x)` | `_ridders(f, x, h0, tol)` 직접 호출 |
| `partial_derivative(f, pt, i)` | `_ridders(lambda t: f(pt[:i]+[t]+pt[i+1:]), pt[i], h0, tol)` |
| `gradient(f, pt)` | 각 축에 `partial_derivative` 호출 |
| `jacobian(fs, pt)` | 각 스칼라 함수에 `gradient` 호출 |
| `hessian(f, pt)` | 대각: `_ridders_2nd`, 비대각: 혼합 스텐실 + Ridders |
| `directional_derivative(f, pt, dir)` | 단위 벡터 방향 1D Ridders |
| `laplacian(f, pt)` | hessian 대각 합 |
| `divergence`, `curl` | 각 성분 `partial_derivative` |
| `nth_derivative(f, x, n)` | n번 재귀 `_ridders` (반복 호출) |

### 7.6 Hessian 비대각 요소

```cython
cdef double _hessian_offdiag(object func, double[:] pt, int i, int j, double h0, double tol):
    """d2f/dxi dxj by 4-point stencil + Ridders"""
    cdef double[10][10] A
    cdef double hh = h0
    # (f(pt + h*ei + h*ej) - f(pt + h*ei - h*ej) - f(pt - h*ei + h*ej) + f(pt - h*ei - h*ej)) / (4h^2)
    # ... (루프 구조는 _ridders와 동일)
```

### 7.7 디폴트 파라미터 조정

```cython
class Differentiation:
    def __init__(self,
                 tol: float = 1e-12,        # 기존 1e-8 -> 1e-12
                 initial_h: float = 0.1,    # 기존 1e-3 -> 0.1
                 max_iter: int = 10,        # NTAB과 일치
                 number_system: str = "real"):
        # 기존 _validate_* 유지
        ...
```

### 7.8 성능 및 정확도 개선 예상

| 함수 | 현재 오차 | Ridders 목표 오차 |
|---|---|---|
| sin의 미분 | ~1e-7 | ~1e-13 |
| 다항식 `x^5` 미분 | ~1e-6 | ~1e-14 |
| exp 의 Hessian | ~1e-4 | ~1e-10 |

**성능**: 함수 평가 수 증가 (~2 * NTAB = 20회) 하지만 정확도 6자리 이상 개선. 핫루프 반복 계산에서는 Python 오버헤드 감소로 상쇄.

### 7.9 복소수 모드 대응

- `Number` 타입을 `double` 또는 `double complex`로 나누기
- **Fused type** 사용 시 Cython 디스패치 오버헤드 (~370ns) -> 성능 목표 달성 불가
- **결정**: 1차 구현은 **실수 모드만 Cython**, 복소수 모드는 Python fallback (기존 Python 로직 유지)

### 7.10 주의사항

- `func`가 numeric 스칼라가 아닌 list 반환 시 (divergence, curl 등): 컴포넌트별로 `_ridders` 반복 호출 (Python 레벨에서 루프)
- 리턴 타입 `Number` (int/float/complex) -> Cython `double` 캐스팅 시 복소수 입력은 `TypeError` 발생 -> 래퍼에서 분기

---

## 8. 빌드 시스템

### 8.1 `pyproject.toml` (갱신안)

```toml
[build-system]
requires = ["setuptools>=61", "wheel", "Cython>=3.0"]
build-backend = "setuptools.build_meta"

[project]
name = "mathlib"
version = "0.2.0"
description = "A high-performance mathematical library (Cython implementation)"
readme = "README.md"
requires-python = ">=3.10"
authors = [
    {name = "Author", email = "author@example.com"},
]

[tool.setuptools]
packages = [
    "math_library",
    "math_library._core",
    "math_library.constant",
    "math_library.trigonometric_function",
    "math_library.inverse_trigonometric_function",
    "math_library.hyperbolic_function",
    "math_library.exponential_function",
    "math_library.logarithmic_function",
    "math_library.gamma_function",
    "math_library.beta_function",
    "math_library.bessel_function",
    "math_library.legendre_function",
    "math_library.lambert_w_function",
    "math_library.zeta_function",
    "math_library.euler_pi_function",
    "math_library.heaviside_step_function",
    "math_library.gcd",
    "math_library.lcm",
    "math_library.differentiation",
]

[tool.setuptools.package-dir]
"" = "src"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

### 8.2 `setup.py` (신규 작성)

```python
"""mathlib - Cython build script (Windows MinGW-w64 UCRT64 + MSVC 지원)"""
import sys
import os
from setuptools import setup, Extension
from Cython.Build import cythonize


def is_msvc():
    compiler = os.environ.get("CC", "")
    if sys.platform == "win32":
        if not compiler or "cl" in compiler.lower():
            return True
    return False


if is_msvc():
    compile_args = ["/O2", "/fp:fast", "/arch:AVX2"]
    link_args = []
    define_macros = []
else:
    # GCC / MinGW-w64
    compile_args = [
        "-O3",
        "-march=native",
        "-mfma",
        "-fno-math-errno",
        "-fno-trapping-math",
    ]
    link_args = []
    define_macros = [("MS_WIN64", None)] if sys.platform == "win32" else []


libraries = ["m"] if sys.platform != "win32" else []


# 빌드할 Cython 모듈 목록
extensions_sources = [
    # _core (elementary primitives)
    ("math_library._core._helpers", "src/math_library/_core/_helpers.pyx"),
    ("math_library._core._constants", "src/math_library/_core/_constants.pyx"),
    ("math_library._core.argument_reduction", "src/math_library/_core/argument_reduction.pyx"),
    ("math_library._core.exponential", "src/math_library/_core/exponential.pyx"),
    ("math_library._core.logarithmic", "src/math_library/_core/logarithmic.pyx"),
    ("math_library._core.power_sqrt", "src/math_library/_core/power_sqrt.pyx"),
    ("math_library._core.trigonometric", "src/math_library/_core/trigonometric.pyx"),
    ("math_library._core.inverse_trig", "src/math_library/_core/inverse_trig.pyx"),
    ("math_library._core.hyperbolic", "src/math_library/_core/hyperbolic.pyx"),
    # 특수 함수
    ("math_library.gamma_function.gamma", "src/math_library/gamma_function/gamma.pyx"),
    ("math_library.beta_function.beta", "src/math_library/beta_function/beta.pyx"),
    ("math_library.bessel_function.bessel", "src/math_library/bessel_function/bessel.pyx"),
    ("math_library.legendre_function.legendre", "src/math_library/legendre_function/legendre.pyx"),
    ("math_library.lambert_w_function.lambert_w", "src/math_library/lambert_w_function/lambert_w.pyx"),
    ("math_library.zeta_function.zeta", "src/math_library/zeta_function/zeta.pyx"),
    ("math_library.euler_pi_function.euler_pi", "src/math_library/euler_pi_function/euler_pi.pyx"),
    ("math_library.heaviside_step_function.heaviside", "src/math_library/heaviside_step_function/heaviside.pyx"),
    ("math_library.gcd.gcd", "src/math_library/gcd/gcd.pyx"),
    ("math_library.lcm.lcm", "src/math_library/lcm/lcm.pyx"),
    ("math_library.differentiation.differentiation", "src/math_library/differentiation/differentiation.pyx"),
]

extensions = [
    Extension(
        name=name,
        sources=[source],
        libraries=libraries,
        define_macros=define_macros,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
    for name, source in extensions_sources
]

setup(
    name="mathlib",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
            "embedsignature": True,
        },
        annotate=False,  # 개발 중에는 True로 HTML 어노테이션 확인
        nthreads=4,
    ),
)
```

### 8.3 개발 워크플로우

```bash
# 최초 설치
pip install -e . --no-build-isolation

# .pyx 수정 후 재컴파일
python setup.py build_ext --inplace

# 테스트 실행
pytest tests/

# 벤치마크 실행
python tests/benchmark.py  # (신규 작성 권장)
```

### 8.4 CI/배포 (추후)

- 최초 지원 OS: Windows (MinGW-w64 UCRT64, MSVC), Linux (GCC)
- Wheel 빌드: `cibuildwheel` 활용
- `pip install mathlib` 시 플랫폼별 wheel 자동 선택

---

## 9. 누락 함수 / TODO 정리

### 9.1 1차 스코프 포함 (이번 구현)

- `sqrt` (신규): `libc.math.sqrt` 직접 호출
- `exp` (신규): musl 간략 방식 (Cody-Waite + 5항 Remez)
- `ln` (신규): musl 간략 방식 (`frexp` + 유리 근사)

### 9.2 2차 스코프 (추후 구현 예정, TODO)

`math` 모듈에 있으나 본 라이브러리 미구현:

| 함수 | 우선순위 | 구현 방식 (추후) |
|---|---|---|
| `cbrt(x)` | 낮음 | Newton 반복 or musl `cbrt.c` |
| `hypot(x, y)` | 중간 | `sqrt(x*x + y*y)` with overflow 방지 |
| `atan2(y, x)` | 중간 | arctan + 사분면 분기 (musl `atan2.c`) |
| `expm1(x)` (공개) | 중간 | musl `expm1.c` 복제 (현재 내부 사용) |
| `log1p(x)` | 중간 | musl `log1p.c` |
| `log2(x)`, `log10(x)` | 중간 | musl `log2_data`/`log10_data` 테이블 |
| `fmod(x, y)`, `fabs(x)` | 낮음 | `libc.math` 직접 노출 |
| `floor`, `ceil`, `trunc` | 낮음 | `libc.math` 직접 |
| `copysign(x, y)` | 낮음 | `libc.math` 직접 |
| `asinh`, `acosh`, `atanh` | 중간 | 로그 항등식 기반 |
| `isnan`, `isinf`, `isfinite` | 낮음 | `libc.math` 직접 |

### 9.3 1차 구현에서의 트레이드오프 수용

| 항목 | 트레이드오프 | 대안 (TODO) |
|---|---|---|
| `\|x\| >= 2^20*pi/2`에서 sin/cos/tan | 정확도 수백 ULP까지 저하 가능 | Payne-Hanek 구현 (musl `__rem_pio2_large.c`) |
| exp 간략 방식 | ~0.5 ULP 추가 오차 | 128-entry 테이블 (musl `exp_data.c`) |
| ln 간략 방식 | ~0.5 ULP 추가 오차 | 128-entry 테이블 (musl `log_data.c`) |
| expm1 간략 방식 | \|x\| 작을 때 cancellation 가능 | musl `expm1.c` 정밀 이식 |
| 쌍곡함수 overflow 경로 | 단순 overflow 반환 | musl `__expo2` 이식 |

### 9.4 TODO 문서화 정책

각 TODO 함수/항목은:
1. `README.md`의 "Roadmap" 섹션에 나열
2. 해당 모듈 docstring에 `# TODO(v0.3): implement cbrt` 형식
3. 사용 시도 시 `AttributeError: ... see roadmap` 메시지

---

## 10. 실수/복소수 분기 방침

### 10.1 결정: **실수 모드만 Cython 고속 구현**

**근거**:

1. 핫패스 99%는 실수 계산 (동역학/제어 시뮬레이션 문맥)
2. 복소수 구현은 musl 계수를 재사용 불가 (실수 전용)
3. Cython fused type 사용 시 ~370ns 디스패치 오버헤드 -> 성능 목표 달성 불가
4. 별도 `cpdef double complex sin_c(double complex z)` 함수로 분리는 유지보수 부담 과다

### 10.2 구현 방침

| 모드 | 경로 | 속도 | 기능 |
|---|---|---|---|
| **실수 (고속)** | `math_library._core.*` Cython | `math` 대비 1.1~1.5x | elementary primitive만 |
| **복소수 (호환)** | 기존 `_complex_log`, `_exp_series` Python 유지 | 현재와 동일 | 전체 기능 (deg/rad, tol 등) |

### 10.3 래퍼에서의 분기 (예시: `trigonometric_function/sin.py`)

```python
from .._core.trigonometric import sin as _sin_c
from ..exponential_function.power import (
    _validate_max_terms, _validate_number, _validate_number_system,
    _validate_real_number, _validate_tol,
)

def sin(x, unit="rad", tol=None, max_terms=100, number_system="real"):
    # 실수 모드 + 기본 옵션 -> 고속 경로
    if number_system == "real" and unit == "rad" and tol is None:
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return _sin_c(float(x))

    # 그 외 (deg, complex, tol 지정) -> 기존 Python 경로
    return _sin_python_fallback(x, unit, tol, max_terms, number_system)
```

### 10.4 복소수 모드 장래 확장 (TODO)

향후 `double complex` 오버로드 필요 시:

```cython
cpdef double complex sin_complex(double complex z) noexcept:
    # sin(z) = sin(x)*cosh(y) + i*cos(x)*sinh(y)
    cdef double x = z.real
    cdef double y = z.imag
    return complex(sin(x) * hypercos(y), cos(x) * hypersin(y))
```

별도 함수명 (`mathlib.sin_complex`)으로 노출, 기본 `sin`과 명확히 구분.

### 10.5 사용자 승인 필요 사항

- [x] 1차 구현에서 복소수 Cython 경로 **생략** (명세 기본 방침)
- [x] 2차 계획: 별도 `*_complex` 함수군 추가 (실수 경로 안정화 후)

(사용자의 최종 승인은 sim-engineer 착수 전 확인 필요)

---

## 11. 예상 성능 요약표

### 11.1 Elementary primitives

| 함수 | `math` 기준 | mathlib 목표 | 비율 | 구현 난이도 |
|---|---|---|---|---|
| sin | 55ns | 60~80ns | 1.1~1.45x | 중간 |
| cos | 55ns | 60~80ns | 1.1~1.45x | 중간 |
| tan | 75ns | 80~110ns | 1.1~1.45x | 어려움 |
| arcsin | 65ns | 70~95ns | 1.1~1.45x | 중간 |
| arccos | 65ns | 70~95ns | 1.1~1.45x | 중간 |
| arctan | 60ns | 60~85ns | 1.0~1.4x | 중간 |
| hypersin (sinh) | 100ns | 110~150ns | 1.1~1.5x | 어려움 |
| hypercos (cosh) | 100ns | 110~150ns | 1.1~1.5x | 어려움 |
| hypertan (tanh) | 110ns | 120~165ns | 1.1~1.5x | 어려움 |
| exp | 70ns | 75~105ns | 1.07~1.5x | 중간 |
| ln | 75ns | 80~110ns | 1.07~1.47x | 중간 |
| log(base, x) | 100ns | 110~150ns | 1.1~1.5x | 쉬움 |
| power (일반) | 120ns | 130~180ns | 1.08~1.5x | 중간 |
| power (정수 지수) | - | 20~40ns | N/A | 쉬움 |
| sqrt | 25ns | 25~30ns | 1.0~1.2x | 매우 쉬움 |
| sec/cosec/cotan | 80ns | 80~110ns | 1.0~1.4x | 매우 쉬움 |
| arcsec/arccosec/arccotan | 90ns | 90~125ns | 1.0~1.4x | 매우 쉬움 |
| hypersec/hypercosec/hypercotan | 150ns | 150~200ns | 1.0~1.3x | 매우 쉬움 |

### 11.2 특수 함수

| 함수 | 현재 (Python) | Cython 목표 | 배율 |
|---|---|---|---|
| gamma | 50~200us | 5~20us | 10x |
| beta | 100~400us | 10~40us | 10x |
| bessel (첫 몇 차수) | 500us ~ 5ms | 50~500us | 10x |
| legendre (n=10) | 50us | 5us | 10x |
| lambert_w | 100~500us | 10~50us | 10x |
| zeta | 100us ~ 1ms | 10~100us | 10x |
| euler_pi | 100us | 10us | 10x |
| heaviside | 0.5us | 0.2us | 2.5x |
| gcd | 1us | 0.3us | 3x |
| lcm | 1.5us | 0.4us | 4x |

### 11.3 Differentiation

| 메서드 | 현재 (Python) | Cython 목표 | 정확도 개선 |
|---|---|---|---|
| single_variable | 10~50us | 5~20us | 1e-7 -> 1e-13 |
| gradient (3D) | 30~150us | 15~60us | 동일 |
| hessian (3D) | 100~500us | 50~250us | 동일 |
| jacobian (3x3) | 100~500us | 50~250us | 동일 |

### 11.4 벤치마크 방법 (권장)

`tests/benchmark.py` 작성 (sim-engineer 과제):

```python
import math
import timeit
import math_library as ml

N = 1_000_000
x = 1.2345

t_math = min(timeit.repeat(lambda: math.sin(x), number=N, repeat=7)) / N * 1e9
t_ml = min(timeit.repeat(lambda: ml.sin(x), number=N, repeat=7)) / N * 1e9
print(f"sin: math={t_math:.1f}ns, ml={t_ml:.1f}ns, ratio={t_ml/t_math:.3f}x")
```

**공정 비교 원칙** (cython_best_practices.md 섹션 7.3):
1. 입력 분포 고정 (동일 난수 시드)
2. 캐시 워밍업 1회 후 측정
3. `timeit.repeat` 7회 반복 min (best-of)
4. 배열 연산은 numpy와 비교 (math는 스칼라 전용)

---

## 부록 A: 구현 순서 권고

sim-engineer 작업 흐름 (의존성 순):

1. **_helpers.pxd** (기반)
2. **_constants.pyx/pxd** (pi, e 등)
3. **power_sqrt.pyx** (sqrt만 먼저, `libc.math` 래핑)
4. **exponential.pyx** (exp, _exp_inline, _expm1_inline)
5. **logarithmic.pyx** (ln, log, _ln_inline)
6. **power_sqrt.pyx** (power 추가 - exp, ln 의존)
7. **argument_reduction.pyx** (rem_pio2)
8. **trigonometric.pyx** (sin, cos, tan, sec/cosec/cotan)
9. **inverse_trig.pyx** (arcsin, arccos, arctan, 파생)
10. **hyperbolic.pyx** (hypersin/..., expm1 의존)
11. **기존 래퍼 조정** (`trigonometric_function/sin.py` 등 -> Cython 호출로 전환)
12. **특수 함수 변환** (gamma -> beta -> bessel -> ... 순)
13. **differentiation.pyx** (Ridders)
14. **`setup.py`, `pyproject.toml` 갱신**
15. **테스트 통과 검증** (`tests/` 전체)
16. **벤치마크 수행** (`tests/benchmark.py` 신규)

---

## 부록 B: 리스크 요소

1. **rem_pio2 Cody-Waite 정밀도**: musl 원본의 고정밀 보정 로직 (`(ix - high_y0) > 0x13700000U` 분기) 누락 시 pi/2 근방에서 심각한 오차 발생. musl `__rem_pio2.c` **bit-exact 복제 필수**.

2. **exp 테이블 미구현**: 1차 간략 방식 (5항 Remez)은 테이블 방식 대비 0.5~1 ULP 저하. 성능 목표 미달 시 musl 128-entry 테이블 업그레이드 필요.

3. **MinGW-w64 FMA 지원**: `-mfma` 플래그로 `fma()`가 하드웨어 `vfmadd231sd` 명령으로 컴파일되는지 **Cython annotate HTML 또는 disassembly 검증 필수**.

4. **Cython 블록 중간 cdef 선언 금지**: 본 명세 의사코드에 블록 중간 `cdef double absx` 등이 있으나 Cython은 함수 상단 선언만 허용. sim-engineer는 이를 모든 `cdef`로 통합할 것.

5. **복소수 Python fallback 성능**: deep-path 호출 시 복소수 분기로 진입 시 기존 Python 속도 유지 → 문제 없음. 단, top-level `mathlib.sin`은 **실수만 지원** 명확히 문서화.

6. **테스트 호환성**: 기존 17개 테스트 중 `test_differentiation.py`의 수렴 판정은 `tol=1e-8` 기준이므로 Ridders 도입 시 **더 엄격한 판정**으로 통과 가능. 기타 테스트는 IEEE 754 표준 결과 기반이므로 Cython 구현도 통과 예상.

7. **정수 지수 검출**: `power(2, 30)`에서 `exponent=30.0`이 `_check_int`에서 정수로 인식되어야 함. `floor(30.0) == 30.0` 확인 필수. Edge case: `exponent = 2**53` 근처.

8. **음수 밑 + 정수 지수 부호**: `pow(-2, 3) = -8`, `pow(-2, 2) = 4`. `_pow_int_real`에서 base 부호 자동 처리 (이진 거듭제곱에서 `result *= b`로 부호 자연 전파).

9. **Windows Python 3.11 + MinGW-w64 UCRT64 환경**: cython_best_practices.md 섹션 8 참조. `-DMS_WIN64` 매크로 필수. `libgcc_s_seh-1.dll` 등 런타임 DLL 경로 관리.

---

## 부록 C: 테스트 전략 (권장)

1. **단위 테스트**: 기존 `tests/` 17개 파일 유지 + Cython 빌드 후 재실행

2. **ULP 테스트** (신규, `tests/test_ulp.py`):

```python
import math, random
import math_library as ml

random.seed(42)
for _ in range(100_000):
    x = random.uniform(-100, 100)
    expected = math.sin(x)
    actual = ml.sin(x)
    diff_ulp = abs(actual - expected) / max(abs(expected) * 2.22e-16, 2.22e-308)
    assert diff_ulp < 2, f"sin({x}): diff={diff_ulp} ULP"
```

3. **특수값 테스트** (신규, `tests/test_special_values.py`):

```python
import math
import math_library as ml

assert math.isnan(ml.sin(float('nan')))
assert math.isnan(ml.sin(float('inf')))
assert ml.sin(0.0) == 0.0
assert ml.sin(-0.0) == -0.0
```

4. **벤치마크 테스트**: `tests/benchmark.py` (11.4 참조)

5. **고정밀 정답 비교** (선택): `mpmath`로 생성한 50자리 정답과 비교하여 ULP 확인.

---

## 참고 출처

- `algorithm_reference.md` (본 프로젝트 `docs/`)
- `cython_best_practices.md` (본 프로젝트 `docs/`)
- musl libc, git.musl-libc.org/cgit/musl, `src/math/`
- W. H. Press et al., "Numerical Recipes in C", 2nd ed., Cambridge Univ. Press, 1992, section 5.7 (Ridders 미분)
- Jean-Michel Muller, "Elementary Functions: Algorithms and Implementation", 3rd ed., Birkhauser, 2016, Ch.5 (삼각), Ch.7 (역삼각), Ch.11 (지수), Ch.12 (power)
- IEEE 754-2019, IEEE Standard for Floating-Point Arithmetic, section 5.3.1 (sqrt correctly rounded), section 9.2.1 (pow 특수값)
- Sun Microsystems / FreeBSD msun, fdlibm, `src/math/` (musl 계수 원출처)
- CORE-MATH, core-math.gitlabpages.inria.fr, INRIA (correctly rounded 구현)
