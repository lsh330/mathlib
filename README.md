# mathlib

Python `math` 모듈의 대체를 목표로 설계한 **Cython 기반 자체 구현 수학 라이브러리**입니다.
단순 libm 래퍼가 아닌, musl libc / fdlibm 계수를 Cython Horner-FMA 형태로 직접 재구현하여
`math` 모듈과 동등하거나 더 빠른 성능을 달성하면서 IEEE 754 완전 준수와 ULP 수준의 정확도를 보장합니다.

17개 elementary primitive 외에 역쌍곡·수치 안정·다인수·이산·집계·특수·IEEE·술어 함수 35개를 추가하여
총 **52개+ 함수**를 제공합니다. `math` 모듈의 모든 실용 기능을 커버하며, `log2` / `log10` / `exp2`
등 일반화 표현으로 대체 가능한 항목은 `log(a, b)` / `power(a, b)` 패턴으로 통일합니다.

---

## 목차

1. [특징](#1-특징)
2. [설치](#2-설치)
3. [5분 빠른 시작](#3-5분-빠른-시작)
4. [API 레퍼런스](#4-api-레퍼런스)
5. [math 모듈 마이그레이션 가이드](#5-math-모듈-마이그레이션-가이드)
6. [Laplace 변환](#6-laplace-변환)
7. [수치해석 (NumericalAnalysis)](#7-수치해석-numericalanalysis)
8. [선형대수 (LinearAlgebra)](#8-선형대수-linearalgebra)
9. [성능 벤치마크](#9-성능-벤치마크-기본-함수)
9. [성능 벤치마크](#9-성능-벤치마크-기본-함수)
10. [정확도 보장](#10-정확도-보장)
11. [구현 철학](#11-구현-철학)
12. [프로젝트 구조](#12-프로젝트-구조)
13. [개발 및 테스트](#13-개발-및-테스트)
14. [현재 한계 및 TODO](#14-현재-한계-및-todo)
15. [라이선스](#15-라이선스)
16. [참고문헌](#16-참고문헌)

---

## 1. 특징

| 항목 | 내용 |
|---|---|
| **대상 Python** | 3.10+ |
| **구현 언어** | Cython 3.0+ (C 확장 모듈로 컴파일) |
| **자체 구현** | `libc.math`의 sin/cos/exp/log 직접 호출 금지 — musl 계수 Cython 재구현 |
| **복소수 self-implementation** | `cmath` 의존성 0 — 모든 복소수 경로를 Euler 공식 기반 자체 Cython으로 구현 |
| **자동 real/complex 감지** | `number_system` 플래그 불필요. 입력 타입과 수학적 결과로 자동 분기 |
| **허용 헬퍼** | `fma`, `frexp`, `ldexp`, `sqrt`, `isnan`, `isinf` 등 IEEE 754 헬퍼만 허용 |
| **ULP 정확도** | sin/cos/arctan/arccos/exp/ln 최대 1 ULP, sqrt 0 ULP (correctly rounded) |
| **함수 커버리지** | 17 primitive + 35 신규 = 총 52개+ 함수. `math` 모듈의 모든 실용 기능 커버 |
| **IEEE 754 특수값** | NaN, ±∞, ±0, subnormal 전 항목 준수 |
| **큰 인수 대응** | Payne-Hanek 구현으로 `sin(1e20)` 등 대규모 인수에서 `math.sin`과 0 ULP 일치 |
| **추가 특수 함수** | gamma, beta, Bessel J, Legendre, Lambert W, Riemann zeta, Euler φ, Heaviside |
| **수치 미분** | Ridders-Richardson extrapolation 기반 Differentiation 클래스 (25+ 메서드) |
| **Laplace 변환 (C++ 백엔드)** | 기호 Laplace 변환·역변환·극점/영점·부분분수·transfer function 합성 (SymPy 의존 없음) |

---

## 2. 설치

### 요구사항

| 항목 | 버전 |
|---|---|
| Python | 3.10 이상 |
| Cython | 3.0 이상 |
| C 컴파일러 | 아래 플랫폼별 안내 참조 |

### Windows — MinGW-w64 UCRT64 (권장, 검증 환경)

```bash
# 1. MSYS2 설치 후 UCRT64 환경에서 gcc 설치
# https://www.msys2.org/ 참조

# 2. MSYS2 UCRT64 셸에서:
pacman -S mingw-w64-ucrt-x86_64-gcc

# 3. 저장소 클론
git clone https://github.com/lsh330/mathlib.git
cd mathlib

# 4. Cython 설치
pip install cython

# 5. MinGW로 빌드 (UCRT64 gcc가 PATH에 있을 때)
python setup.py build_ext --inplace --compiler=mingw32
```

### Windows — MSVC Build Tools

```bash
# Visual Studio Build Tools 설치:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

git clone https://github.com/lsh330/mathlib.git
cd mathlib
pip install cython
python setup.py build_ext --inplace
```

### Linux / macOS

```bash
# GCC 또는 Clang이 설치된 환경에서
git clone https://github.com/lsh330/mathlib.git
cd mathlib
pip install cython
python setup.py build_ext --inplace
```

> **주의**: Linux/macOS는 이론적으로 동작하나, 이번 릴리스에서 검증된 환경은 Windows 11 + MinGW-w64 UCRT64입니다.

### Editable 설치 (개발용)

> **주의**: `pip install -e .` 는 **Windows MinGW 환경에서 지원되지 않습니다** (pip가 내부적으로 MSVC를 호출하여 `Microsoft Visual C++ 14.0 or greater is required` 오류 발생).
>
> MinGW 환경에서는 아래 명령을 사용하십시오:

```bash
pip install cython
python setup.py build_ext --inplace --compiler=mingw32
```

`.pyx` 파일 수정 후 재컴파일이 필요할 때도 위 명령을 다시 실행합니다.

MSVC가 설치된 환경(Visual Studio 2022 이상)에서는 표준 editable 설치도 가능합니다:

```bash
pip install -e . --no-build-isolation
```

---

## 3. 5분 빠른 시작

빌드 완료 후 아래 코드를 실행하면 라이브러리의 주요 기능을 바로 확인할 수 있습니다.

```python
import math_library as m

# --- 기본 함수 ---
print(m.sin(1.2345))      # 0.9439833239445111
print(m.cos(0.0))          # 1.0
print(m.exp(1.0))          # 2.7182818284590455
print(m.ln(m.e()))         # 1.0
print(m.sqrt(2.0))         # 1.4142135623730951

# --- 자동 real→complex 승격 ---
print(m.sqrt(-1))           # 1j
print(m.ln(-1))             # 3.141592653589793j
print(m.arcsin(2))          # (1.5707963267948966-1.3169578969248166j)
print(m.arccos(2))          # 1.3169578969248166j
print(m.arc_hypercos(0.5))  # (복소수, 실수 정의역 밖)
print(m.arc_hypertan(1.5))  # (복소수, 실수 정의역 밖)

# --- 복소수 직접 입력 ---
print(m.sin(1 + 2j))        # (3.165778513216168+1.9596010414216063j)
print(m.ln(1 + 2j))         # (0.8047189562170503+1.1071487177940904j)

# --- 신규 함수 (Phase 2) ---
print(m.arc_hypersin(1.0))       # 0.881373587019543
print(m.cbrt(-8.0))              # -1.9999999999999998
print(m.atan2(1.0, 1.0))         # 0.7853981633974483  (π/4)
print(m.hypot(3, 4))             # 5.0
print(m.factorial(20))           # 2432902008176640000
print(m.erf(1.0))                # 0.8427007929497149
print(m.fsum([1e20, 1, -1e20]))  # 1.0  (Kahan 보상 합산)

# --- 일반화 표현 (math.log2, math.log10, math.exp2 대체) ---
print(m.log(2, 1024))   # 10.0  (= log2(1024))
print(m.log(10, 100))   # 2.0   (= log10(100))
print(m.power(2, 10))   # 1024.0 (= 2^10)

# --- 특수 함수 ---
print(m.gamma(5))           # 23.999999999999996  (≈ 4! = 24)
print(m.zeta(2))            # 1.6449340668481436  (≈ π²/6)
print(m.bessel_j(0, 1.0))  # 0.7651976865579666

# --- 수치 미분 (Ridders-Richardson extrapolation) ---
from math_library import Differentiation
d = Differentiation()
print(d.single_variable(m.sin, 1.0))              # ≈ cos(1) ≈ 0.5403...
print(d.nth_derivative(lambda x: x**4, 1.0, 2))  # ≈ 12.0  (x^4의 2차 미분 @ 1)
print(d.gradient(lambda x, y: x**2 + y**2, [1.0, 2.0]))  # ≈ [2.0, 4.0]
```

성능 벤치마크 재현:

```bash
python bench/perf_compare.py
```

---

## 4. API 레퍼런스

### 4.1 상수

```python
from math_library import pi, e, epsilon

pi()       # 3.141592653589793
e()        # 2.718281828459045
epsilon()  # 2.220446049250313e-16  (머신 엡실론)
```

### 4.2 삼각함수

| 함수 | 시그니처 | 실수 정의역 | 복소수 |
|---|---|---|---|
| `sin` | `sin(x)` | 모든 실수 | 자동 dispatch |
| `cos` | `cos(x)` | 모든 실수 | 자동 dispatch |
| `tan` | `tan(x)` | `x ≠ π/2 + nπ` | 자동 dispatch |
| `sec` | `sec(x)` | `x ≠ π/2 + nπ` | 자동 dispatch |
| `cosec` | `cosec(x)` | `x ≠ nπ` | 자동 dispatch |
| `cotan` | `cotan(x)` | `x ≠ nπ` | 자동 dispatch |

특수값 동작:

| 입력 | `sin` | `cos` |
|---|---|---|
| `NaN` | `NaN` | `NaN` |
| `±0` | `±0` | `1.0` |
| `±∞` | `NaN` | `NaN` |

### 4.3 역삼각함수

| 함수 | 시그니처 | 실수 정의역 | 반환 범위 | 정의역 밖 |
|---|---|---|---|---|
| `arcsin` | `arcsin(x)` | `[-1, 1]` | `[-π/2, π/2]` | 복소수 승격 |
| `arccos` | `arccos(x)` | `[-1, 1]` | `[0, π]` | 복소수 승격 |
| `arctan` | `arctan(x)` | 모든 실수 | `(-π/2, π/2)` | — |
| `arcsec` | `arcsec(x)` | `|x| ≥ 1` | — | — |
| `arccosec` | `arccosec(x)` | `|x| ≥ 1` | — | — |
| `arccotan` | `arccotan(x)` | 모든 실수 | — | — |

### 4.4 쌍곡함수

| 함수 | 시그니처 | 비고 |
|---|---|---|
| `hypersin` | `hypersin(x)` | sinh(x) |
| `hypercos` | `hypercos(x)` | cosh(x) |
| `hypertan` | `hypertan(x)` | tanh(x) |
| `hypersec` | `hypersec(x)` | 1/cosh(x) |
| `hypercosec` | `hypercosec(x)` | 1/sinh(x), x≠0 |
| `hypercotan` | `hypercotan(x)` | cosh(x)/sinh(x), x≠0 |

### 4.5 지수 및 로그

| 함수 | 시그니처 | 특수값 및 비고 |
|---|---|---|
| `exp` | `exp(x)` | `exp(-∞)=0`, `exp(+∞)=+∞`, x>709.78 → `+∞` |
| `expm1` | `expm1(x)` | `exp(x)-1`을 x≈0 근방에서 수치 안정하게 계산 |
| `ln` | `ln(x)` | `ln(0)=-∞`, x<0 → 복소수 승격, `ln(+∞)=+∞` |
| `log` | `log(base, x)` | 임의 밑 로그. `log(2, x)` = log2, `log(10, x)` = log10 |
| `log1p` | `log1p(x)` | `ln(1+x)`을 x≈0 근방에서 수치 안정하게 계산 |
| `sqrt` | `sqrt(x)` | `sqrt(±0)=±0`, x<0 → 복소수 승격, correctly rounded |
| `power` | `power(base, exp)` | `power(x,±0)=1`, `power(-1,±∞)=1`, 음수 밑 분수지수 → 복소수 승격 |
| `cbrt` | `cbrt(x)` | 실수 세제곱근. `cbrt(-8)` ≈ `-2.0` |

### 4.6 역쌍곡함수 (신규)

역쌍곡함수는 mathlib 네이밍(`arc_hyper*`)과 `math` 모듈 호환 alias 양쪽을 모두 제공합니다.

| 함수 | alias | 실수 정의역 | 정의역 밖 |
|---|---|---|---|
| `arc_hypersin` | `asinh` | 모든 실수 | — |
| `arc_hypercos` | `acosh` | `x ≥ 1` | 복소수 승격 |
| `arc_hypertan` | `atanh` | `|x| < 1` | 복소수 승격 |
| `arc_hypersec` | — | `0 < x ≤ 1` | — |
| `arc_hypercosec` | — | `x ≠ 0` | — |
| `arc_hypercotan` | — | `|x| > 1` | — |

```python
import math_library as m
print(m.arc_hypersin(1.0))   # 0.881373587019543
print(m.asinh(1.0))          # 0.881373587019543  (alias)
print(m.arc_hypercos(0.5))   # 복소수 (정의역 밖)
print(m.acosh(2.0))          # 1.3169578969248166
```

### 4.7 수치 안정 함수 (신규)

부동소수점 소거(cancellation) 문제를 회피하기 위한 함수들입니다.

| 함수 | 시그니처 | 언제 사용 |
|---|---|---|
| `expm1` | `expm1(x)` | `exp(x) - 1` 계산 시 x≈0에서 발생하는 소거 오차 방지 |
| `log1p` | `log1p(x)` | `ln(1 + x)` 계산 시 x≈0에서 발생하는 소거 오차 방지 |
| `cbrt` | `cbrt(x)` | 실수 입력에 대한 세제곱근 (`power(x, 1/3)`은 음수 밑에서 복소수 반환) |

```python
import math_library as m
# x = 1e-10 근방에서 정밀도 비교
print(m.expm1(1e-10))   # 1.00000000005e-10  (정확)
print(m.exp(1e-10) - 1) # 같은 값을 직접 계산하면 소거 오차 발생 가능
print(m.log1p(1e-10))   # 9.999999999500001e-11
```

### 4.8 다인수 함수 (신규)

| 함수 | 시그니처 | 설명 |
|---|---|---|
| `atan2` | `atan2(y, x)` | 4사분면 아크탄젠트. 반환 범위 `(-π, π]` |
| `hypot` | `hypot(*args)` | `sqrt(x²+y²+...)` 오버플로우 없이 계산 |
| `dist` | `dist(p, q)` | 두 점(리스트/튜플) 간 유클리드 거리 |

```python
import math_library as m
print(m.atan2(1.0, 1.0))     # 0.7853981633974483  (π/4)
print(m.hypot(3, 4))          # 5.0
print(m.dist([0, 0], [3, 4])) # 5.0
```

### 4.9 이산 함수 (신규)

| 함수 | 시그니처 | 설명 | 제약 |
|---|---|---|---|
| `factorial` | `factorial(n)` | n! | n ≥ 0 정수. 실수/복소수는 `gamma` 사용 |
| `comb` | `comb(n, k)` | 이항계수 C(n, k) | n, k ≥ 0 정수 |
| `perm` | `perm(n, k)` | 순열 P(n, k) | n, k ≥ 0 정수 |
| `isqrt` | `isqrt(n)` | 정수 제곱근 (내림) | n ≥ 0 정수 |

```python
import math_library as m
print(m.factorial(20))  # 2432902008176640000
print(m.comb(10, 3))    # 120
print(m.perm(5, 2))     # 20
print(m.isqrt(17))      # 4
```

> 실수 또는 복소수에 대한 일반화 팩토리얼은 `m.gamma(n+1)`을 사용하십시오.

### 4.10 집계 함수 (신규)

| 함수 | 시그니처 | 설명 |
|---|---|---|
| `fsum` | `fsum(iterable)` | Neumaier-Kahan 보상 합산. 부동소수점 오차 누적 방지 |
| `prod` | `prod(iterable)` | 순차 곱셈 |

```python
import math_library as m
print(m.fsum([1e20, 1, -1e20]))  # 1.0  (일반 sum은 0.0 반환 가능)
print(m.prod([1, 2, 3, 4, 5]))   # 120
```

### 4.11 특수 수학 함수 (신규)

| 함수 | 시그니처 | 설명 |
|---|---|---|
| `erf` | `erf(x)` | 오차 함수 ∫₀ˣ e^(-t²) dt × 2/√π |
| `erfc` | `erfc(x)` | 상보 오차 함수 1 - erf(x) |
| `lgamma` | `lgamma(x)` | ln(|Γ(x)|). x≈0, -1, -2, ... 근방에서 최대 ~256 ULP 허용 |

```python
import math_library as m
print(m.erf(1.0))      # 0.8427007929497149
print(m.erfc(1.0))     # 0.1572992070502851
print(m.lgamma(5.0))   # 3.178053830347944  (≈ ln(4!) = ln(24))
```

### 4.12 IEEE 754 연산 (신규)

| 함수 | 시그니처 | 설명 |
|---|---|---|
| `ceil` | `ceil(x)` | 올림 (float 반환) |
| `floor` | `floor(x)` | 내림 (float 반환) |
| `trunc` | `trunc(x)` | 0 방향 절사 (float 반환) |
| `fmod` | `fmod(x, y)` | 부동소수점 나머지 (부호는 x를 따름) |
| `remainder` | `remainder(x, y)` | IEEE 754 나머지 (가장 가까운 정수 몫 기준) |
| `copysign` | `copysign(x, y)` | x의 크기에 y의 부호 적용 |
| `modf` | `modf(x)` | `(소수 부분, 정수 부분)` 튜플 반환 |
| `nextafter` | `nextafter(x, y)` | x에서 y 방향으로의 다음 표현 가능한 double |
| `ulp` | `ulp(x)` | x 근방의 ULP(Unit in the Last Place) 크기 |

```python
import math_library as m
print(m.ceil(1.2))              # 2.0
print(m.floor(1.9))             # 1.0
print(m.trunc(1.9))             # 1.0
print(m.fmod(10.0, 3.0))        # 1.0
print(m.copysign(1.0, -2.0))    # -1.0
print(m.modf(3.7))              # (0.7000000000000002, 3.0)
print(m.nextafter(1.0, 2.0))    # 1.0000000000000002
print(m.ulp(1.0))               # 2.220446049250313e-16
```

### 4.13 술어 함수 (신규)

| 함수 | 시그니처 | 설명 |
|---|---|---|
| `isnan` | `isnan(x)` | NaN 여부 판정 |
| `isinf` | `isinf(x)` | 무한대(±∞) 여부 판정 |
| `isfinite` | `isfinite(x)` | 유한값 여부 판정 |
| `isclose` | `isclose(a, b, rel_tol, abs_tol)` | 근사 동등성 판정 |

```python
import math_library as m
print(m.isnan(float('nan')))   # True
print(m.isinf(float('inf')))   # True
print(m.isfinite(1.0))         # True
print(m.isclose(1.0, 1.0))     # True
```

### 4.14 기존 특수 함수

| 함수 | 시그니처 | 설명 |
|---|---|---|
| `gamma` | `gamma(n)` | 감마 함수. `gamma(n) = (n-1)!` (양의 정수) |
| `beta` | `beta(a, b)` | 베타 함수 B(a, b) = Γ(a)Γ(b)/Γ(a+b) |
| `bessel_j` | `bessel_j(n, x)` | 제1종 베셀 함수 Jₙ(x) |
| `bessel_j0` | `bessel_j0(x)` | J₀(x) 고속 경로 |
| `bessel_j1` | `bessel_j1(x)` | J₁(x) 고속 경로 |
| `legendre` | `legendre(n, x)` | 르장드르 다항식 Pₙ(x) |
| `lambert_w` | `lambert_w(x)` | Lambert W 함수 (주가지 W₀) |
| `zeta` | `zeta(s)` | 리만 제타 함수 ζ(s) |
| `euler_pi` | `euler_pi(n)` | 오일러 수 (Euler numbers) |
| `euler_phi` | `euler_phi(n)` | 오일러 토션트 함수 φ(n) |
| `heaviside` | `heaviside(x)` | 헤비사이드 계단 함수 |
| `gcd` | `gcd(a, b)` | 최대공약수 |
| `lcm` | `lcm(a, b)` | 최소공배수 |

### 4.15 Differentiation 클래스

Ridders-Richardson extrapolation 기반 수치 미분. 25개 이상의 메서드 제공.

```python
from math_library import Differentiation

d = Differentiation(
    tol=1e-8,       # 수렴 허용 오차
    initial_h=1e-3, # 초기 차분 스텝
    max_iter=20,    # 수렴 반복 상한
)
```

#### 주요 메서드

| 메서드 | 역할 | 예시 |
|---|---|---|
| `single_variable(f, x)` | 일변수 도함수 | `d.single_variable(sin, 1.0)` → `cos(1)` |
| `nth_derivative(f, x, n)` | n차 도함수 | `d.nth_derivative(f, 1.0, 2)` |
| `left_derivative(f, x)` | 좌미분 | — |
| `right_derivative(f, x)` | 우미분 | — |
| `partial_derivative(f, point, idx)` | 편미분 | `d.partial_derivative(f, [1,2], 0)` |
| `mixed_partial(f, point, multi_idx)` | 혼합편미분 | — |
| `gradient(f, point)` | 그래디언트 | `d.gradient(f, [1,2])` → `[∂f/∂x, ∂f/∂y]` |
| `jacobian(funcs, point)` | 야코비안 행렬 | — |
| `hessian(f, point)` | 헤시안 행렬 | — |
| `hessian_vector_product(f, point, v)` | 헤시안-벡터 곱 | — |
| `laplacian(f, point)` | 라플라시안 | — |
| `vector_laplacian(F, point)` | 벡터 라플라시안 | — |
| `directional_derivative(f, point, dir)` | 방향도함수 | — |
| `divergence(F, point)` | 발산 (div) | — |
| `curl(F, point)` | 회전 (curl, 3D) | — |
| `implicit_derivative(rel, x, y)` | 음함수 미분 | `d.implicit_derivative(lambda x,y: x²+y²-25, 3, 4)` → `-0.75` |
| `parametric_derivative(coords, t, n)` | 매개변수 미분 | — |
| `wirtinger_derivatives(f, z)` | 비르팅거 미분 (∂/∂z, ∂/∂z̄) | 복소 해석학 |
| `total_differential(f, point)` | 전미분 계수 | — |
| `total_derivative(outer, inners, point)` | 합성함수 전체 미분 | — |
| `gateaux_derivative(f, point, dir)` | Gateaux 미분 | — |
| `frechet_derivative(f, point)` | Frechet 미분 (선형 근사 행렬) | — |
| `generalized_derivative(f, x)` | 일반화 미분 | — |
| `subgradient(f, x)` | 부분그래디언트 | — |

---

## 5. math 모듈 마이그레이션 가이드

기존 `import math` 코드를 `import math_library as m`으로 전환할 때 참조하십시오.
상세한 철학 및 정확도/성능 특성은 `docs/generalization_guide.md`를 참조하십시오.

### 동일한 이름으로 사용 가능

| `math` | `math_library` | 비고 |
|---|---|---|
| `math.sin(x)` | `m.sin(x)` | 실수/복소수 자동 분기 |
| `math.cos(x)` | `m.cos(x)` | 실수/복소수 자동 분기 |
| `math.tan(x)` | `m.tan(x)` | 실수/복소수 자동 분기 |
| `math.asin(x)` | `m.arcsin(x)` | 정의역 밖 → 복소수 승격 |
| `math.acos(x)` | `m.arccos(x)` | 정의역 밖 → 복소수 승격 |
| `math.atan(x)` | `m.arctan(x)` | |
| `math.atan2(y, x)` | `m.atan2(y, x)` | 동일 |
| `math.sinh(x)` | `m.hypersin(x)` | |
| `math.cosh(x)` | `m.hypercos(x)` | |
| `math.tanh(x)` | `m.hypertan(x)` | |
| `math.asinh(x)` | `m.arc_hypersin(x)` 또는 `m.asinh(x)` | alias 제공 |
| `math.acosh(x)` | `m.arc_hypercos(x)` 또는 `m.acosh(x)` | alias 제공 |
| `math.atanh(x)` | `m.arc_hypertan(x)` 또는 `m.atanh(x)` | alias 제공 |
| `math.exp(x)` | `m.exp(x)` | 실수/복소수 자동 분기 |
| `math.expm1(x)` | `m.expm1(x)` | 동일 |
| `math.log(x)` | `m.ln(x)` | 자연로그. 단일 인수 |
| `math.log(x, base)` | `m.log(base, x)` | 인수 순서 주의: base 먼저 |
| `math.log1p(x)` | `m.log1p(x)` | 동일 |
| `math.sqrt(x)` | `m.sqrt(x)` | x<0 → 복소수 승격 |
| `math.pow(x, y)` | `m.power(x, y)` | |
| `math.cbrt(x)` | `m.cbrt(x)` | 동일 |
| `math.hypot(x, y)` | `m.hypot(x, y)` | 동일 |
| `math.dist(p, q)` | `m.dist(p, q)` | 동일 |
| `math.factorial(n)` | `m.factorial(n)` | 동일 |
| `math.comb(n, k)` | `m.comb(n, k)` | 동일 |
| `math.perm(n, k)` | `m.perm(n, k)` | 동일 |
| `math.isqrt(n)` | `m.isqrt(n)` | 동일 |
| `math.fsum(iter)` | `m.fsum(iter)` | 동일 |
| `math.prod(iter)` | `m.prod(iter)` | 동일 |
| `math.erf(x)` | `m.erf(x)` | 동일 |
| `math.erfc(x)` | `m.erfc(x)` | 동일 |
| `math.lgamma(x)` | `m.lgamma(x)` | 동일 |
| `math.gamma(x)` | `m.gamma(x)` | 동일 |
| `math.ceil(x)` | `m.ceil(x)` | 동일 |
| `math.floor(x)` | `m.floor(x)` | 동일 |
| `math.trunc(x)` | `m.trunc(x)` | 동일 |
| `math.fmod(x, y)` | `m.fmod(x, y)` | 동일 |
| `math.remainder(x, y)` | `m.remainder(x, y)` | 동일 |
| `math.copysign(x, y)` | `m.copysign(x, y)` | 동일 |
| `math.modf(x)` | `m.modf(x)` | 동일 |
| `math.nextafter(x, y)` | `m.nextafter(x, y)` | 동일 |
| `math.ulp(x)` | `m.ulp(x)` | 동일 |
| `math.isnan(x)` | `m.isnan(x)` | 동일 |
| `math.isinf(x)` | `m.isinf(x)` | 동일 |
| `math.isfinite(x)` | `m.isfinite(x)` | 동일 |
| `math.isclose(a, b)` | `m.isclose(a, b)` | 동일 |
| `math.gcd(a, b)` | `m.gcd(a, b)` | 동일 |
| `math.lcm(a, b)` | `m.lcm(a, b)` | 동일 |
| `math.pi` | `m.pi()` | 상수 → 함수 호출 |
| `math.e` | `m.e()` | 상수 → 함수 호출 |

### 일반화 표현으로 대체

아래 항목들은 기존 구현으로 완전히 표현 가능하여 별도 함수를 추가하지 않습니다.

| `math` | `math_library` 표현 | 이유 |
|---|---|---|
| `math.log2(x)` | `m.log(2, x)` | `log(base, x)` 일반화 |
| `math.log10(x)` | `m.log(10, x)` | 동일 |
| `math.exp2(x)` | `m.power(2, x)` | `power(base, exp)` 일반화 |
| `math.tau` | `2 * m.pi()` | 상수 조합 |
| `math.inf` | `float('inf')` | Python 내장 |
| `math.nan` | `float('nan')` | Python 내장 |
| `math.fabs(x)` | `abs(x)` | Python 내장 |
| `math.degrees(x)` | `x * 180 / m.pi()` | 산술 표현 |
| `math.radians(x)` | `x * m.pi() / 180` | 산술 표현 |

### 자동 real→complex 승격

`number_system` 플래그를 지정하지 않아도, 수학적으로 복소수 결과가 나오는 경우
자동으로 복소수를 반환합니다.

| 호출 | 결과 | 설명 |
|---|---|---|
| `m.sqrt(-1)` | `1j` | 음수의 제곱근 |
| `m.ln(-1)` | `3.141592653589793j` | 음수의 자연로그 (πj) |
| `m.power(-1, 0.5)` | `≈ 1j` | 음수 밑 분수 지수 |
| `m.arcsin(2)` | `(1.5708...-1.3170j)` | 실수 정의역 `[-1,1]` 밖 |
| `m.arccos(2)` | `1.3170j` | 실수 정의역 `[-1,1]` 밖 |
| `m.arc_hypercos(0.5)` | 복소수 | 실수 정의역 `x≥1` 밖 |
| `m.arc_hypertan(1.5)` | 복소수 | 실수 정의역 `|x|<1` 밖 |

---

## 6. Laplace 변환

`math_library.laplace` 는 기호 Laplace 변환 전용 서브패키지입니다.
C++ AST(Abstract Syntax Tree) 백엔드 위에 Cython 래퍼를 얹은 구조로, SymPy·GiNaC·Pynac 등 외부 CAS에 의존하지 않고 자체 구현합니다.
상세 사용 방법은 [docs/laplace_guide.md](docs/laplace_guide.md)를 참조하십시오.

### 6.1 설계 철학

| 항목 | 내용 |
|---|---|
| **백엔드** | C++17 AST + pool (hash-consing). Cython은 Python ↔ C++ 경계만 담당 |
| **외부 CAS** | SymPy / GiNaC / Pynac 의존 없음 — 완전 자체 구현 |
| **성능 목표** | Pynac 대비 1.5~3배 느림(허용), SymPy 대비 50~500배 빠름 (실측 달성) |
| **기존 회귀** | mathlib 52개+ primitive 전부 기존 동작 유지 |

### 6.2 빠른 시작

```python
from math_library.laplace import Laplace, t, s, Sin, Exp, const

L = Laplace()
F = L.transform(Exp(-t) * Sin(2*t))
print(F)              # ((s + 1)**2 + 4)**-1*2
print(F.latex())      # \frac{2}{\left(s + 1\right)^{2} + 4}
print(F.evalf(s=2))   # 0.15384615384615385

f_back = L.inverse(F)
print(f_back)         # sin(t*2)*exp(t*-1)
print(L.poles(F))     # [(-1+2j), (-1-2j)]
G = F.feedback()      # Unity feedback 연결
print(F.magnitude(1.0))  # 0.447213595499958
```

### 6.3 지원 함수 목록 (f(t) 형태)

변환 가능한 t-영역 표현식에 사용 가능한 함수:

| 심볼 함수 | 의미 | Laplace 변환 |
|---|---|---|
| `Sin(expr)` | sin | `ω/(s²+ω²)` |
| `Cos(expr)` | cos | `s/(s²+ω²)` |
| `Tan(expr)` | tan | — |
| `Exp(expr)` | exp (지수) | `1/(s-a)` |
| `Sinh(expr)` | sinh (쌍곡사인) | `a/(s²-a²)` |
| `Cosh(expr)` | cosh (쌍곡코사인) | `s/(s²-a²)` |
| `Tanh(expr)` | tanh (쌍곡탄젠트) | — |
| `Ln(expr)` | 자연로그 (변환 한정 사용) | — |
| `Sqrt(expr)` | 제곱근 | — |
| `Arcsin`, `Arccos`, `Arctan` | 역삼각함수 | — |
| `Heaviside(expr)` / `H(expr)` | 헤비사이드 계단 함수 u(t-a) | `e^{-as}/s` |
| `Dirac(expr)` | 디락 델타 δ(t-a) | `e^{-as}` |

- **t-shift 규칙**: `u(t-a)*f(t-a)` 형태는 자동으로 `e^{-as}·L{f(t)}` 로 변환됩니다.
- **Non-rational inverse**: `L⁻¹{e^{-as}·G(s)}` = `u(t-a)·g(t-a)` 를 자동 처리합니다.
- 산술 연산자 `+`, `-`, `*`, `/`, `**` 및 상수·심볼 조합 모두 지원합니다.

```python
from math_library.laplace import Laplace, t, s, Exp, Heaviside, Dirac, const

L = Laplace()

# Heaviside 변환
F = L.transform(Heaviside(t - const(2)))   # e^{-2s} / s
print(F.evalf(s=1.0))   # 0.135335...

# Dirac delta 변환
G = L.transform(Dirac(t - const(1)))       # e^{-s}
print(G.evalf(s=1.0))   # 0.367879...

# t-shift: u(t-1)*e^{-(t-1)}  →  e^{-s}/(s+1)
a = const(1)
H_shift = Heaviside(t - a) * Exp(-(t - a))
F_shift = L.transform(H_shift)
print(F_shift.evalf(s=2.0))   # 0.045111...

# non-rational inverse: e^{-2s}/(s+1)  →  u(t-2)*e^{-(t-2)}
G_inv = L.transform(Exp(-t))    # 1/(s+1)
s_sym = symbol('s')
from math_library.laplace import symbol
F_nr = Exp(-2 * symbol('s')) * G_inv
f_back = L.inverse(F_nr)
print(f_back.evalf(t=2.5))   # 0.606530...  (= e^{-0.5})
```

### 6.4 주요 메서드 요약

#### `Laplace` 클래스

| 메서드 | 설명 |
|---|---|
| `transform(f)` | F(s) = L{f(t)} |
| `inverse(F)` | f(t) = L⁻¹{F(s)}. 유리함수 및 `e^{-as}·G(s)` 형태 지원 |
| `poles(F)` | 분모 근 (복소수 리스트) |
| `zeros(F)` | 분자 근 (복소수 리스트) |
| `final_value(F)` | lim_{s→0} s·F(s), (값, 유효여부) 반환 |
| `initial_value(F)` | lim_{s→∞} s·F(s), (값, 유효여부) 반환 |

#### `PyExpr` 메서드

| 메서드 | 설명 |
|---|---|
| `evalf(**subs)` | 변수 치환 후 수치 평가 |
| `subs(**kwargs)` | 기호 또는 수치 치환, 새 PyExpr 반환 |
| `latex()` | LaTeX 문자열 반환 |
| `diff(var)` | 기호 미분 |
| `diff_numeric(var_name)` | Ridders 수치 미분 (`LazyDerivative`) |
| `series(var_name, about, order)` | Taylor 급수 (`TaylorSeries`) |
| `expand()` | 분배 전개 |
| `cancel(var)` | 유리식 약분 |
| `simplify(var)` | expand + cancel |
| `collect(var)` | 동류항 정리: `3s²+2s²+5s+7 → 5s²+5s+7` |
| `partial_fractions(var)` | 부분분수 분해 |
| `series_connect(G)` | 직렬 연결: self * G |
| `parallel_connect(G)` | 병렬 연결: self + G |
| `feedback(H=None)` | 피드백 연결: self / (1 + self·H), 내부 `simplify()` 자동 적용 |
| `frequency_response(ω)` | H(jω) 계산 |
| `magnitude(ω)` | \|H(jω)\| |
| `phase(ω)` | ∠H(jω) [rad] |
| `lambdify(var_names, backend)` | Python callable 변환 |

### 6.5 성능 요약 (실측값)

| 항목 | 측정값 |
|---|---|
| 상수 노드 생성 (`const(2.5)`) | 101 ns |
| 덧셈 (`a + b`) | 246 ns |
| hash-consing 조회 | 33 ns |
| `L{sin(2t)}` forward 변환 | 0.44~1.0 μs |
| `L{e^(-t)sin(3t)}` forward 변환 | 1.3 μs |
| 캐시 hit 재변환 | 84 ns |
| `partial_fractions` (2차 분모) | 8.2 μs |
| `inverse` (e^{-t}sin(2t)) | 9.5 μs |
| `lambdify` 컴파일 | 34 μs |
| `lambdify` 호출 | 79 ns |
| `diff` (기호 미분) | 11.7 μs |
| `expand` | 3.5 μs |
| `cancel` | 34 μs |

---

## 7. 수치해석 (NumericalAnalysis)

`math_library.NumericalAnalysis` 클래스는 외부 의존성 없이 Cython 자체 구현된 수치해석 알고리즘을 제공합니다.
Simpson 적분법 8종, 근 찾기 2종(Newton-Raphson, Secant), ODE 적분 5종(Euler, RK2, RK4, DOPRI5, RKF45),
Gauss-Legendre 적분 2종, Dormand-Prince RK78, Adams 계열 3종(AB/AM/PECE)으로 총 **21개 메서드**를 제공합니다.

### 7.1 빠른 시작

```python
import math
from math_library import NumericalAnalysis

na = NumericalAnalysis()

# --- 적분 ---
I = na.simpson_13(math.sin, 0, math.pi)                    # ≈ 2.094 (3점 단순)
I = na.composite_simpson_13(math.sin, 0, math.pi, n=100)   # ≈ 2.0000000108 (합성, n=100)
I = na.adaptive_simpson(math.sin, 0, math.pi, tol=1e-12)   # = 2.0 (적응형, 오차 < 1e-11)
I = na.romberg(math.sin, 0, math.pi, depth=6)              # ≈ 2.0 (오차 ≈ 1e-15)

# --- 근 찾기 ---
root = na.newton_raphson(lambda x: x**2 - 2, 1.5)         # = sqrt(2) ≈ 1.41421356...
root = na.secant_method(lambda x: x**3 - x - 2, 1.0, 2.0) # ≈ 1.52137970...

# --- ODE: dy/dt = -y, y(0)=1 ---
y = na.euler(lambda t, y: -y, 0.0, 1.0, 1.0, n=1000)      # ≈ 0.3677 (오차 1.8e-4)
y = na.rk4(lambda t, y: -y,   0.0, 1.0, 1.0, n=100)       # ≈ e^(-1) (오차 3e-11)
y = na.rk45(lambda t, y: -y,  0.0, 1.0, 1.0, tol=1e-10)   # ≈ e^(-1) (오차 1.2e-11)

# --- PyExpr ODE ---
from math_library.laplace import t, symbol
y_sym = symbol('y')
y = na.rk4(-2*t*y_sym, 0.0, 1.0, 1.0, n=200, vars=('t', 'y'))  # y=exp(-t^2)
```

### 7.2 Simpson 적분법 8종

| 메서드 | 설명 | n 조건 | 절대 오차 (sin, 0~π) |
|---|---|---|---|
| `simpson_13(f, a, b)` | 3점 단순 Simpson 1/3 | — | O(h^4), 실제 ≈ 0.094 |
| `simpson_38(f, a, b)` | 4점 단순 Simpson 3/8 | — | O(h^4), 실제 ≈ 0.040 |
| `composite_simpson_13(f, a, b, n)` | 합성 Simpson 1/3 | n 짝수, ≥ 2 | O(h^4), n=100: ≈ 1e-8 |
| `composite_simpson_38(f, a, b, n)` | 합성 Simpson 3/8 | n 3의 배수, ≥ 3 | O(h^4), n=99: ≈ 2.5e-8 |
| `adaptive_simpson(f, a, b, tol)` | 적응형 (명시적 스택) | — | < tol, 실제 ≈ 0 |
| `mixed_simpson(f, a, b, n)` | 혼합 (n 임의, ≥ 2) | n ≥ 2 | O(h^4), 혼합 공식 |
| `simpson_irregular(x, y)` | 불균등 간격 Simpson | ≥ 3점 | 점 간격 의존 |
| `romberg(f, a, b, depth)` | Romberg (Richardson 외삽) | — | depth=6: ≈ 1e-15 |

### 7.3 메서드 상세

#### `simpson_13(f, a, b, *, var='t', return_error=False)`

3점 공식: `I = (h/3) * [f(a) + 4*f(m) + f(b)]`, `h = (b-a)/2`

#### `simpson_38(f, a, b, *, var='t', return_error=False)`

4점 공식: `I = (3h/8) * [f(a) + 3*f(x1) + 3*f(x2) + f(b)]`, `h = (b-a)/3`

#### `composite_simpson_13(f, a, b, n, *, var='t', return_error=False)`

```python
# n=100 (짝수 필수)
I, err = na.composite_simpson_13(math.sin, 0, math.pi, n=100, return_error=True)

# n 홀수이면 ValueError 발생
try:
    na.composite_simpson_13(math.sin, 0, math.pi, n=7)
except ValueError as e:
    print(e)  # "composite_simpson_13 requires even n, got n=7"
```

#### `composite_simpson_38(f, a, b, n, *, var='t', return_error=False)`

```python
# n=99 (3의 배수 필수)
I = na.composite_simpson_38(math.sin, 0, math.pi, n=99)

# n이 3의 배수 아니면 ValueError 발생
try:
    na.composite_simpson_38(math.sin, 0, math.pi, n=10)
except ValueError as e:
    print(e)  # "composite_simpson_38 requires n divisible by 3, got n=10"
```

#### `adaptive_simpson(f, a, b, *, tol=1e-10, max_depth=50, var='t', return_error=False)`

명시적 스택으로 Python 재귀 제한 없이 깊은 구간 분할 처리. Richardson 보정 적용.

```python
I, err = na.adaptive_simpson(math.sin, 0, math.pi, tol=1e-12, return_error=True)
# I ≈ 2.0,  err ≈ 3e-13
```

#### `mixed_simpson(f, a, b, n, *, var='t', return_error=False)`

n이 짝수이면 합성 1/3, 홀수이면 앞 (n-3) 구간에 1/3, 끝 3 구간에 3/8 적용.

```python
I = na.mixed_simpson(math.sin, 0, math.pi, n=7)   # n=7 (홀수) → 1/3(n=4) + 3/8(3구간)
I = na.mixed_simpson(math.sin, 0, math.pi, n=8)   # n=8 (짝수) → 합성 1/3
```

#### `simpson_irregular(x_points, y_points, *, return_error=False)`

불균등 간격 비균등 Simpson 3점 공식 적용:
```
h0 = x1-x0, h1 = x2-x1
I = (h0+h1)/6 * [(2-h1/h0)*f0 + (h0+h1)^2/(h0*h1)*f1 + (2-h0/h1)*f2]
```

```python
x = [0.0, 0.3, 0.7, 1.2, 2.0]   # 불균등 간격
y = [math.sin(xi) for xi in x]
I = na.simpson_irregular(x, y)   # ≈ 1 - cos(2)

# 입력 검증
try:
    na.simpson_irregular([0, 1, 0.5], [0, 1, 0.5])   # 단조 감소
except ValueError as e:
    print(e)  # "x_points must be monotonically increasing"
```

#### `romberg(f, a, b, *, depth=5, var='t', return_error=False)`

Richardson 외삽 테이블:
```
T[i][0] = 사다리꼴 (2^i 구간)
T[i][j] = (4^j * T[i][j-1] - T[i-1][j-1]) / (4^j - 1)
```

```python
I = na.romberg(math.sin, 0, math.pi, depth=6)   # 오차 ≈ 1e-15
I, err = na.romberg(math.sin, 0, math.pi, depth=6, return_error=True)
# I ≈ 2.0, err ≈ 0.0
```

### 7.4 공통 파라미터

| 파라미터 | 타입 | 설명 |
|---|---|---|
| `f` | callable 또는 PyExpr | 피적분 함수. PyExpr 입력 시 `evalf` 자동 적용 |
| `a`, `b` | float | 적분 구간 하한, 상한 (`a < b`) |
| `var` | str | PyExpr 변수명 (기본 `'t'`) |
| `return_error` | bool | `True`이면 `(value, error_estimate)` 튜플 반환 |

### 7.5 예외 처리

| 예외 | 조건 |
|---|---|
| `ValueError("a must be less than b")` | `a >= b` |
| `ValueError("composite_simpson_13 requires even n, got n=...")` | n이 홀수 |
| `ValueError("composite_simpson_38 requires n divisible by 3, got n=...")` | n이 3의 배수 아님 |
| `ValueError("mixed_simpson requires n >= 2, got n=...")` | n < 2 |
| `ValueError("tol must be positive, ...")` | tol <= 0 |
| `ValueError("romberg requires depth >= 1, ...")` | depth < 1 |
| `ValueError("x_points must be monotonically increasing, ...")` | 단조성 위반 |
| `ValueError("x_points[i] is NaN or Inf")` | NaN/Inf 포함 |
| `TypeError("f must be callable or have lambdify method, ...")` | callable 아닌 입력 |
| `RuntimeError("adaptive_simpson exceeded max_depth=...")` | max_depth 초과 |

### 7.6 정확도 및 수치 안정성

**Kahan 보상 누산**: 합성 Simpson 류의 내부 합산에 Kahan 알고리즘을 사용하여
대규모 n(예: 10,000)에서도 부동소수점 누적 오차를 억제합니다.

```python
I_10k = na.composite_simpson_13(math.sin, 0, math.pi, n=10000)
# |I_10k - 2.0| = 0.0  (double 정밀도 한계까지 정확)
```

**Richardson 오차 추정**: `return_error=True` 시 n 대비 n/2 구간으로 두 번 계산하고
Richardson 계수(`15 = 2^4 - 1`)로 나눈 추정값을 반환합니다.

**적응형 Richardson 보정**: `adaptive_simpson`은 각 수렴 구간에서
`S2 + (S2 - S1) / 15` 보정을 적용하여 실제 오차가 tol보다 훨씬 작습니다.

### 7.7 성능

| 메서드 | 실측 (Windows 11, MinGW gcc 15.2.0) |
|---|---|
| `simpson_13` | 0.13 µs/call |
| `composite_simpson_13 n=100` | 2.39 µs/call |
| `composite_simpson_13 n=10000` | 219 µs/call |
| `adaptive_simpson tol=1e-10` | 29 µs/call |
| `romberg depth=6` | 1.97 µs/call |

---

### 7.8 근 찾기 (Newton-Raphson / Secant)

#### `newton_raphson(f, x0, *, fprime=None, var='x', tol=1e-10, max_iter=100, return_info=False)`

Newton-Raphson 방법으로 `f(x) = 0`의 근을 찾습니다.

- `fprime=None` (기본): `Differentiation.single_variable(f, x)` 수치 미분 자동 사용
- `fprime` 명시: callable 또는 PyExpr 모두 허용
- `return_info=True`: `(root, iter_count, |f(root)|)` 튜플 반환

```python
na = NumericalAnalysis()

# 기본 (수치 미분 자동)
root = na.newton_raphson(lambda x: x**2 - 2, 1.5)          # = sqrt(2)

# fprime 명시 (빠른 경로: 0.48 µs/call)
root = na.newton_raphson(lambda x: x**2 - 2, 1.5, fprime=lambda x: 2*x)

# 상세 정보 반환
root, iters, residual = na.newton_raphson(lambda x: x**2 - 2, 1.5, return_info=True)
# root=1.414..., iters=4, residual=4.44e-16

# PyExpr 입력
from math_library.laplace import symbol
x = symbol('x')
root = na.newton_raphson(x**2 - 2, 1.5, var='x')
```

| 예외 | 조건 |
|---|---|
| `ValueError` | `tol <= 0` 또는 `max_iter <= 0` |
| `ZeroDivisionError` | `f'(x) = 0` 발생 시 |
| `RuntimeError` | `max_iter` 내 미수렴: `"newton_raphson did not converge in N iterations, |f(x)|=..."` |

#### `secant_method(f, x0, x1, *, var='x', tol=1e-10, max_iter=100, return_info=False)`

도함수 없이 두 초기점으로 근을 찾는 Secant 방법입니다.

```
x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
```

```python
root = na.secant_method(lambda x: x**3 - x - 2, 1.0, 2.0)  # ≈ 1.52138
```

---

### 7.9 ODE (Runge-Kutta 계열)

모든 ODE 메서드는 `dy/dt = f(t, y)` 형식의 scalar ODE를 대상으로 합니다.

#### 메서드 요약

| 메서드 | 차수 | 특징 | 오차 (dy/dt=-y, t=0→1, n=100) |
|---|---|---|---|
| `euler` | 1 | 고정 step | O(h), ≈ 1.8e-4 (n=1000) |
| `rk2` | 2 | 고정 step, 3가지 변형 | O(h²), ≈ 6.2e-6 (n=100) |
| `rk4` | 4 | 고전 RK4, 고정 step | O(h⁴), ≈ 3.1e-11 (n=100) |
| `rk45` | 5(4) | Dormand-Prince, 적응형 | tol 제어, ≈ 1.2e-11 |
| `rk_fehlberg` | 5(4) | Fehlberg, 적응형 | tol 제어, ≈ 2.4e-11 |

#### `euler(f, t0, y0, t_end, n, *, vars=('t','y'), return_trajectory=False)`

```python
y = na.euler(lambda t, y: -y, 0.0, 1.0, 1.0, n=1000)
traj = na.euler(lambda t, y: -y, 0.0, 1.0, 1.0, n=10, return_trajectory=True)
# [(0.0, 1.0), (0.1, 0.9), ..., (1.0, y_end)]
```

#### `rk2(f, t0, y0, t_end, n, *, method='midpoint', vars=('t','y'), return_trajectory=False)`

method: `'midpoint'` (기본) / `'heun'` / `'ralston'`

```python
y = na.rk2(lambda t, y: -y, 0.0, 1.0, 1.0, n=100, method='heun')
```

#### `rk4(f, t0, y0, t_end, n, *, vars=('t','y'), return_trajectory=False)`

고전 4차 Runge-Kutta. 고정 step에서 정확도와 성능의 균형이 최적입니다.

```python
y = na.rk4(lambda t, y: -y, 0.0, 1.0, 1.0, n=100)
```

#### `rk45(f, t0, y0, t_end, *, tol=1e-8, h_init=None, h_min=1e-12, vars=('t','y'), return_trajectory=False, max_steps=10000)`

Dormand-Prince DOPRI5 적응형. FSAL(First Same As Last) 최적화로 step당 6회 함수 평가.

```python
y = na.rk45(lambda t, y: -y, 0.0, 1.0, 1.0, tol=1e-10)
y = na.rk45(lambda t, y: math.sin(t)*y, 0.0, 1.0, 2*math.pi, tol=1e-10)
```

#### `rk_fehlberg(f, t0, y0, t_end, *, tol=1e-8, h_init=None, h_min=1e-12, vars=('t','y'), return_trajectory=False, max_steps=10000)`

Fehlberg(1969) RKF45. step당 6회 함수 평가.

```python
y = na.rk_fehlberg(lambda t, y: -y, 0.0, 1.0, 1.0, tol=1e-10)
```

#### PyExpr ODE 입력

```python
from math_library.laplace import t, symbol

y_sym = symbol('y')
# dy/dt = -2*t*y → y = exp(-t^2)
y = na.rk4(-2*t*y_sym, 0.0, 1.0, 1.0, n=200, vars=('t', 'y'))
```

#### ODE 예외 처리

| 예외 | 조건 |
|---|---|
| `ValueError` | `n < 1`, `t0 >= t_end`, `method` 오류 |
| `RuntimeError` | `max_steps` 초과, `h underflow` (적응형) |

#### ODE 성능 (Windows 11, MinGW gcc 15.2.0)

| 메서드 | 실측 |
|---|---|
| `newton_raphson` (fprime 명시) | 0.48 µs/call |
| `newton_raphson` (Differentiation 수치 미분) | 18.7 µs/call |
| `secant_method` | 0.69 µs/call |
| `rk4 n=1000` | 117 µs/call |
| `rk45 tol=1e-8` | 2.56 µs/call |
| `rk_fehlberg tol=1e-8` | 2.69 µs/call |
| `rk78 tol=1e-10` | 4.5 µs/call |
| `predictor_corrector PC4 n=100` | 25 µs/call |

### 7.7 Gauss-Legendre 적분

n-point Gauss-Legendre 구적법 (n = 2 ~ 16):

```python
I = na.gauss_legendre(math.sin, 0, math.pi, n=5)          # 단일 구간
I = na.composite_gauss_legendre(math.sin, 0, math.pi, m=10, n=5)  # 합성
```

| 메서드 | 설명 | 제약 |
|---|---|---|
| `gauss_legendre(f, a, b, n=5)` | n-point GL (2 ≤ n ≤ 16) | (2n-1)차 다항식 정확 |
| `composite_gauss_legendre(f, a, b, m, n=5)` | m 구간 합성 GL | m ≥ 1 |

**정확도**: n=5로 9차 다항식 정확 (오차 < 1e-14). 초월함수는 composite 사용 권장.
**성능**: `gauss_legendre n=5` — 178 ns/call.

### 7.9 RK78 (Dormand-Prince 8(7))

Hairer 등 "Solving ODEs I" Table II.5.4 기반 13-stage 적응형 방법:

```python
y_end = na.rk78(f, t0, y0, t_end, tol=1e-12)
```

### 7.10 Adams 계열 (AB / AM / PECE)

다단계 적분법 3종. 부트스트랩은 RK4 사용:

```python
y_end = na.adams_bashforth(f, t0, y0, t_end, n=1000, order=4)   # explicit
y_end = na.adams_moulton(f, t0, y0, t_end, n=1000, order=4)     # implicit
y_end = na.predictor_corrector(f, t0, y0, t_end, n=1000, order=4)  # PECE
```

| 메서드 | 차수 범위 | 특성 |
|---|---|---|
| `adams_bashforth` | 1 ~ 5 | explicit, 조건부 안정 |
| `adams_moulton` | 1 ~ 5 | implicit, fixed-point 수렴 |
| `predictor_corrector` | 2 ~ 5 | PECE, 매 step 1회 교정 |

---

## 8. 선형대수 (LinearAlgebra)

`math_library.LinearAlgebra` 클래스는 `np.linalg.*` 없이 Cython 자체 구현된 행렬 분해 및 텐서 연산 알고리즘 **33종**을 제공합니다.

### 8.1 메서드 목록

#### 분해 (9종)

| 메서드 | 알고리즘 | 반환 | 비고 |
|---|---|---|---|
| `lu(A, *, pivot=True)` | Doolittle + partial pivoting | `(P, L, U)` | PA = LU |
| `ldu(A)` | LU 유도 | `(L, D, U)` | D: 1D 배열 |
| `qr(A, *, mode='reduced', pivot=False)` | Householder reflections | `(Q, R)` 또는 `(Q, R, P)` | reduced/complete/column-pivot |
| `cholesky(A, *, lower=True)` | Banachiewicz | `L` or `U` | SPD 행렬 전용 |
| `svd(A, *, full_matrices=False)` | Golub-Reinsch bidiag + QR shift | `(U, S, Vt)` | thin/full 모드 |
| `gaussian_elimination(A, b, *, form)` | Gauss-Jordan + partial pivoting | RREF 또는 해 `x` | form='rref'/'ref' |
| `det(A)` | LU 기반 | `float` | sign(P) · prod(diag(U)) |
| `rank(A, *, tol=None)` | SVD 기반 | `int` | tol 기본: max(m,n)·ε·σ_max |
| `inverse(A)` | LU 기반 열별 대입 | ndarray | AX=I n 개 선형계 |

#### 소거·직교화·부분공간 (8종)

| 메서드 | 설명 | 반환 |
|---|---|---|
| `logdet(A, *, return_sign=False)` | log\|det(A)\| (오버플로 안전) | float 또는 (sign, logabsdet) |
| `gauss_jordan(A)` | 명시적 RREF | ndarray (m, n) |
| `gram_schmidt(A)` | 수정 Gram-Schmidt 직교화 | `(Q, R)` |
| `null_space(A, *, tol=None)` | 우측 영공간 (SVD 기반) | ndarray (n, dim) |
| `column_space(A, *, tol=None)` | 열공간 (SVD 기반) | ndarray (m, rank) |
| `row_space(A, *, tol=None)` | 행공간 (SVD 기반) | ndarray (n, rank) |
| `left_null_space(A, *, tol=None)` | 좌측 영공간 (SVD 기반) | ndarray (m, m-rank) |
| `pinv(A, *, tol=None)` | Moore-Penrose 의사역행렬 (SVD 기반) | ndarray |

#### 고유값 (5종)

| 메서드 | 알고리즘 | 반환 | 비고 |
|---|---|---|---|
| `eigen(A, *, symmetric=False)` | QR + Wilkinson shift (Hessenberg) | `(vals, vecs)` complex128 | 복소 지원 |
| `diagonalize(A)` | eigen 기반 | `(P, D)` | P^{-1}AP = D |
| `characteristic_polynomial(A)` | Faddeev-LeVerrier | 1D 배열 (n+1 계수) | p(λ) = det(λI−A) |
| `cayley_hamilton(A)` | p_A(A) = 0 검증 | ndarray (잔차) | 0 행렬 반환 |

#### 텐서 (5종)

| 메서드 | 설명 | 반환 |
|---|---|---|
| `tensor_product(A, B)` | Kronecker product A⊗B (Cython 루프) | ndarray (m·p, n·q) |
| `outer_product(a, b)` | 벡터 외적 a⊗b | ndarray (m, n) |
| `inner_product(a, b)` | Hermitian 내적 sum(conj(a)·b) | scalar |
| `tensor_contract(A, B, axes_a, axes_b)` | 일반 텐서 축약 (reshape/matmul 기반) | ndarray |
| `tensor_transpose(T, axes=None)` | 텐서 축 재배열 (copy 반환) | ndarray |

#### 대칭 (4종)

| 메서드 | 설명 | 반환 |
|---|---|---|
| `is_symmetric(A, tol=1e-10)` | A == A^T 검사 (max-norm) | bool |
| `is_skew_symmetric(A, tol=1e-10)` | A + A^T ≈ 0 검사 | bool |
| `symmetrize(A)` | (A + A^H) / 2 (복소 Hermitian 지원) | ndarray |
| `skew_part(A)` | (A - A^T) / 2 | ndarray |

#### 미분 wrap (2종)

| 메서드 | 설명 | 반환 |
|---|---|---|
| `jacobian(f, point)` | Differentiation.jacobian 래핑 — f: R^n → R^m | ndarray (m, n) |
| `hessian(f, point)` | Differentiation.hessian 래핑 — f: R^n → R | ndarray (n, n) |

### 8.2 사용 예시

```python
import numpy as np
from math_library import LinearAlgebra

la = LinearAlgebra()
rng = np.random.default_rng(42)
A = rng.standard_normal((5, 5))

# --- 분해 ---
P, L, U = la.lu(A)           # PA = LU
Q, R = la.qr(A)              # A = QR
U_s, S, Vt = la.svd(A)      # A = U diag(S) Vt
M = A @ A.T + 5*np.eye(5)
L_c = la.cholesky(M)         # M = L L^T
b = rng.standard_normal(5)
x = la.gaussian_elimination(A, b)
d = la.det(A)
A_inv = la.inverse(A)
r = la.rank(A)

# --- 고유값 ---
vals, vecs = la.eigen(A, symmetric=False)   # 복소 지원
P_d, D = la.diagonalize(A)                  # A = P D P^{-1}
coeffs = la.characteristic_polynomial(A)    # len(n+1)

# --- 텐서 ---
K = la.tensor_product(A, np.eye(3))         # Kronecker product
O = la.outer_product(np.array([1,2,3.0]), np.array([4,5.0]))
ip = la.inner_product(np.array([1+2j, 3+4j]), np.array([5+6j, 7+8j]))
C = la.tensor_contract(A, A, axes_a=1, axes_b=0)   # A @ A

# --- 대칭 ---
print(la.is_symmetric(M))           # True
S_sym = la.symmetrize(A)            # (A + A^T)/2
K_skew = la.skew_part(A)            # (A - A^T)/2

# --- 미분 wrap ---
def f(x, y): return [x**2 + y, x*y]
J = la.jacobian(f, [1.0, 2.0])     # shape (2, 2)

def g(x, y): return x**2 + 3*x*y + y**3
H = la.hessian(g, [1.0, 2.0])      # shape (2, 2), symmetric
```

### 8.3 정확도 (재구성 상대오차 기준)

| 알고리즘 | n=5 | n=50 | n=200 |
|---|---|---|---|
| LU (PA = LU) | ~1e-16 | ~4e-16 | ~1e-15 |
| QR (A = QR) | ~3e-16 | ~8e-16 | ~1e-15 |
| Cholesky (A = LL^T) | ~1e-16 | ~3e-16 | ~4e-16 |
| SVD (A = USVt) | ~1e-15 | ~4e-15 | ~6e-15 |
| inverse (AA^{-1} = I) | ~6e-17 | ~4e-16 | ~6e-16 |
| Kronecker product | ~1e-15 | — | — |
| tensor_contract (matmul) | ~1e-12 | — | — |
| jacobian (수치 미분) | ~1e-6 | — | — |
| hessian (수치 미분) | ~1e-4 | — | — |

### 8.4 성능 (Windows 11, gcc 15.2.0 O3, Intel i7)

| 알고리즘 | n=100 | n=500 | n=1000 |
|---|---|---|---|
| LU | 0.1 ms | 6 ms | 63 ms |
| Cholesky | 0.5 ms | 6 ms | 59 ms |
| SVD | 0.5 ms (50x50) | 16 ms (200x200) | — |
| Kronecker 100x100 | ~400 ms | — | — |

---

## 9. 성능 벤치마크 (기본 함수)

### 측정 환경

| 항목 | 내용 |
|---|---|
| Python | 3.11.9 |
| OS | Windows 11 Pro 10.0.26200 |
| 컴파일러 | MinGW-w64 UCRT64 gcc 15.2.0 |
| 컴파일 플래그 | `-O3 -march=native -mfma -fno-math-errno -fno-trapping-math` |
| 측정 방법 | 워밍업 5,000회 + 반복 1,000,000회, `time.perf_counter_ns()` |
| 테스트 입력 | `x = 1.2345` (실수 단일 스칼라) |

### 17개 primitive 결과

| 함수 | `math` (ns) | `mathlib` (ns) | 비율 | 판정 |
|---|---|---|---|---|
| sin | 27.5 | 25.7 | **0.93x** | math보다 빠름 |
| cos | 26.2 | 24.4 | **0.93x** | math보다 빠름 |
| tan | 34.5 | 27.9 | **0.81x** | math보다 빠름 |
| arcsin | 26.7 | 23.8 | **0.89x** | math보다 빠름 |
| arccos | 25.9 | 23.9 | **0.92x** | math보다 빠름 |
| arctan | 25.9 | 23.9 | **0.92x** | math보다 빠름 |
| hypersin | 29.3 | 36.2 | 1.23x | 허용 범위 이내 |
| hypercos | 29.7 | 33.9 | 1.14x | 허용 범위 이내 |
| hypertan | 33.1 | 37.0 | 1.12x | 허용 범위 이내 |
| exp | 25.9 | 27.7 | 1.07x | 허용 범위 이내 |
| ln | 46.0 | 23.2 | **0.50x** | math보다 2배 빠름 |
| sqrt | 24.6 | 23.9 | **0.97x** | math보다 빠름 |

목표 기준(1.5x 이내): 전 함수 달성. sin/cos/tan/arc*/ln/sqrt는 `math` 모듈보다 빠릅니다.

### 신규 함수 대표 샘플

신규 35개 함수는 `math` 직접 대응 함수가 없거나 성능 기준이 별도로 정의되지 않아,
참고용 절대 수치만 제공합니다.

| 함수 | `mathlib` (ns) | 비고 |
|---|---|---|
| `arc_hypersin` | ~40 | asinh 구현 |
| `erf` | ~35 | 오차 함수 |
| `factorial(20)` | ~50 | 정수 팩토리얼 |
| `isnan` | ~20 | 술어, 매우 빠름 |
| `fsum (3 elements)` | ~200 | Kahan 합산 루프 포함 |

### 벤치마크 재현

```bash
python bench/perf_compare.py
```

---

## 9. 정확도 보장

### ULP 오차 측정 결과

100k 무작위 입력에 대한 측정:

| 함수 | 평균 ULP | 최대 ULP | 판정 |
|---|---|---|---|
| sin | 0.015 | 1 | PASS |
| cos | 0.020 | 1 | PASS |
| tan | 0.189 | 2 | PASS |
| arcsin | 0.184 | 2 | PASS |
| arccos | 0.009 | 1 | PASS |
| arctan | 0.000 | 1 | PASS |
| exp | 0.097 | 1 | PASS |
| ln | 0.250 | 1 | PASS |
| sqrt | 0.000 | 0 | PASS (correctly rounded) |

### Payne-Hanek 큰 인수 대응

`|x| ≥ 2²⁰·π/2 ≈ 1.647×10⁶` 이상의 인수에서 Payne-Hanek 알고리즘을 사용합니다.
2/π의 396자리 16진 테이블(`ipio2[]`)을 기반으로 하며, 입력 크기와 무관하게 정밀 나머지 계산을 수행합니다.

### IEEE 754 특수값 완전 준수

```
sin(NaN) = NaN        sin(±∞) = NaN      sin(±0) = ±0
ln(0)    = -∞         ln(-x)  → πj (복소수 승격)
exp(-∞)  = +0         exp(+∞) = +∞
sqrt(+0) = +0         sqrt(-0) = -0      sqrt(-1) = 1j
exp(-745) → 5e-324 (subnormal 정상 처리)
```

---

## 10. 구현 철학

**자체 구현 원칙**: `libc.math`의 `sin`, `cos`, `exp`, `log` 등 elementary 함수를 직접 호출하지 않습니다.
musl libc와 fdlibm의 bit-exact 계수를 그대로 인용하여 Cython + Horner-FMA 형태로 재구현합니다.

**복소수 self-implementation**: `cmath` 모듈에 대한 의존성을 완전히 제거하였습니다.
모든 복소수 경로는 Euler 공식(`e^(iy) = cos y + i·sin y`)을 기반으로 자체 Cython으로 구현합니다.

**일반화 우선**: 기존 구현으로 표현 가능한 함수는 별도 추가하지 않습니다.
`log2(x)` = `log(2, x)`, `exp2(x)` = `power(2, x)` 등이 그 예입니다.
이는 API 표면을 최소화하고, 각 함수의 의미를 명확히 유지하기 위한 원칙입니다.

- **Argument Reduction**: sin/cos/tan에 Cody-Waite 3단계 + Payne-Hanek 구현
- **다항식 근사**: musl 출처 Remez minimax 계수, Horner 전개 + `libc.math.fma` 활용
- **테이블 기반 ln/exp**: 128-entry 룩업 테이블로 musl `log.c`, `exp.c` 재구현
- **pow**: `exp(y·log(x))` 구조, 정수 지수는 이진 거듭제곱 분기로 오차 억제
- **sqrt**: IEEE 754 필수 요건(correctly rounded)을 만족하는 `libc.math.sqrt` 하드웨어 호출 허용

계수 출처 및 알고리즘 상세는 `docs/algorithm_reference.md`를 참조하십시오.

---

## 11. 프로젝트 구조

```
mathlib/
├── pyproject.toml                      # 빌드 시스템 선언 (Cython>=3.0)
├── setup.py                            # Cython Extension 빌드 (MinGW/MSVC 자동 감지)
├── src/math_library/
│   ├── __init__.py                     # 공용 API 재노출 (Cython 고속 경로 우선, Python 폴백)
│   ├── _core/                          # Cython elementary primitives
│   │   ├── _helpers.pxd                # cdef inline 유틸 (poly, bit 조작)
│   │   ├── _constants.pyx/.pxd         # pi, e, epsilon
│   │   ├── argument_reduction.pyx/.pxd # Cody-Waite + Payne-Hanek
│   │   ├── exponential.pyx/.pxd        # exp, expm1
│   │   ├── logarithmic.pyx/.pxd        # ln, log, log1p (128-entry 테이블)
│   │   ├── power_sqrt.pyx/.pxd         # power, sqrt, cbrt
│   │   ├── trigonometric.pyx/.pxd      # sin, cos, tan, sec, cosec, cotan
│   │   ├── inverse_trig.pyx/.pxd       # arcsin, arccos, arctan, arcsec, ...
│   │   ├── hyperbolic.pyx/.pxd         # hypersin, hypercos, hypertan, ...
│   │   ├── inverse_hyperbolic.pyx/.pxd # arc_hypersin, arc_hypercos, ... (신규)
│   │   ├── multi_arg.pyx/.pxd          # atan2, hypot, dist (신규)
│   │   ├── discrete.pyx/.pxd           # factorial, comb, perm, isqrt (신규)
│   │   ├── aggregate.pyx/.pxd          # fsum, prod (신규)
│   │   ├── special_functions.pyx/.pxd  # erf, erfc, lgamma (신규)
│   │   ├── ieee_ops.pyx/.pxd           # ceil, floor, trunc, fmod, ... (신규)
│   │   └── predicates.pyx/.pxd         # isnan, isinf, isfinite, isclose (신규)
│   ├── gamma_function/gamma.pyx
│   ├── beta_function/beta.pyx
│   ├── bessel_function/bessel.pyx
│   ├── legendre_function/legendre.pyx
│   ├── lambert_w_function/lambert_w.pyx
│   ├── zeta_function/zeta.pyx
│   ├── euler_pi_function/euler_pi.pyx
│   ├── heaviside_step_function/heaviside.pyx
│   ├── gcd/gcd.pyx
│   ├── lcm/lcm.pyx
│   ├── differentiation/differentiation.pyx
│   ├── numerical_analysis/             # 수치해석 서브패키지
│   │   ├── __init__.py                 # NumericalAnalysis 노출
│   │   ├── numerical_analysis.pyx      # 메인 클래스 (Simpson 8종)
│   │   ├── numerical_analysis.pxd      # cdef 선언
│   │   └── _kahan.pxd                  # Kahan 누산 인라인 유틸
│   └── laplace/                        # Laplace 변환 서브패키지 (C++ 백엔드)
│       ├── __init__.py                 # 공개 API (Laplace, PyExpr, t, s, Sin, ...)
│       ├── laplace_ast.pyx/.pxd        # PyExpr cdef class + 심볼·상수·함수 팩토리
│       ├── laplace.pyx                 # Laplace 클래스 (transform/inverse/poles/zeros/...)
│       ├── lambdify.py                 # AST → Python callable 변환
│       ├── series.py                   # TaylorSeries 클래스
│       ├── numeric_diff.py             # LazyDerivative (Ridders 수치 미분 연동)
│       └── cpp/                        # C++17 AST 코어
│           ├── pool.hpp/.cpp           # 표현식 풀 (hash-consing)
│           ├── expr.hpp/.cpp           # AST 노드 정의
│           ├── laplace.hpp/.cpp        # forward 변환 규칙
│           ├── inverse.hpp/.cpp        # 역변환 (partial fractions → t-domain)
│           ├── partial.hpp/.cpp        # 부분분수 분해
│           ├── polynomial.hpp/.cpp     # 다항식 연산 (Durand-Kerner 포함)
│           ├── simplify.hpp/.cpp       # expand / cancel
│           ├── rules.hpp/.cpp          # 변환 규칙 테이블
│           ├── hash.hpp                # 64-bit Zobrist 해시
│           ├── subst.hpp               # 기호 치환
│           └── capi.hpp                # Cython ↔ C++ extern "C" 인터페이스
├── bench/
│   ├── perf_compare.py                 # math vs mathlib 속도 비교 벤치마크
│   └── numerical_analysis_tests.py     # NumericalAnalysis Simpson 8종 검증
├── tests/                              # 카테고리별 테스트 스크립트 (18개)
└── docs/
    ├── algorithm_reference.md          # 알고리즘 상세 및 bit-exact 계수
    ├── generalization_guide.md         # math 모듈 마이그레이션 상세 가이드
    ├── cython_best_practices.md        # 빌드 및 Cython 모범 사례
    ├── implementation_spec.md          # 구현 명세
    ├── performance_benchmark.md        # 벤치마크 원본 결과
    ├── verification_report.md          # 검증 보고서
    └── laplace_guide.md                # Laplace 모듈 상세 사용자 가이드
```

패키지 임포트 이름과 dist 이름 모두 `math_library`입니다.

```python
import math_library as m   # 올바른 임포트
```

---

## 12. 개발 및 테스트

### 빌드

```bash
# Cython 생성 C 파일 재컴파일 (--inplace: src/ 안에 .pyd 생성)
python setup.py build_ext --inplace

# MinGW 명시
python setup.py build_ext --inplace --compiler=mingw32
```

### 테스트 실행

```bash
# 카테고리별 스크립트 실행 (PASS/FAIL 터미널 출력)
python tests/test_trigonomectric_function.py
python tests/test_differentiation.py
python tests/test_comprehensive_types.py

# pytest
pytest tests/
```

### Cython 어노테이션으로 성능 병목 확인

```bash
cython -a src/math_library/_core/trigonometric.pyx
# 생성된 trigonometric.html을 브라우저로 열어 노란색 라인(CPython 상호작용) 확인
```

---

## 13. 현재 한계 및 TODO

### 해결된 항목 (Phase 1/2 완료)

- 복소수 dispatch 일부 미지원 → 완료 (17개 primitive 전체)
- `cmath` 의존성 → 완전 제거 (Euler 공식 기반 자체 구현)
- exp 언더플로우 버그 → 해결 (`exp(-745)` = 5e-324 정상 처리)
- Payne-Hanek 부호 플립 → 해결
- `pow(-1, ±∞)` NaN → 해결 (1.0 반환)
- ln x≈1 영역 정확도 → 해결
- 기존 테스트 pytest discovery → 해결 (18/18 PASS)

### 유지 중인 한계

- **Linux/macOS CI 미검증**: 현재 릴리스는 Windows 11 + MinGW-w64 UCRT64 환경에서만 검증됨. Linux/macOS는 이론적으로 동작하나 CI 파이프라인 미구성.
- **lgamma 특정 구간 ULP**: x≈0, -1, -2, ... (극점 근방) 에서 최대 ~256 ULP. 특수 함수 전용 ULP 기준 별도 정의 필요.
- **zeta(2) 오차**: 구현값 `1.6449340668481436`, 정답 `π²/6 = 1.6449340668482264` (절대오차 8.28e-14, ~370 ULP). 특수 함수 ULP 스펙 미명시로 허용 범위 내로 처리.

### Laplace 모듈 한계

- **Piecewise 함수 미지원**: 구간별 정의 함수의 변환 미구현.
- **`cancel` 비전개 분자/분모**: `(1/(s+1) + 1)**-1 * (s+1)**-1` 형태의 중첩 피드백 표현을 자동 단순화하지 않음 — 수동 `simplify` 필요.

---

## 14. 라이선스

MIT License — `LICENSE` 파일 참조.

계수 출처인 musl libc, fdlibm은 각각 MIT-style 및 BSD 라이선스입니다.

---

## 15. 참고문헌

1. **musl libc** — git.musl-libc.org/cgit/musl — sin, cos, exp, log 등 bit-exact 계수 출처
2. **fdlibm / FreeBSD msun** — 역삼각·쌍곡 함수 계수의 원 출처 (Sun Microsystems)
3. **CORE-MATH (INRIA)** — core-math.gitlabpages.inria.fr — correctly rounded 구현 참조
4. **Jean-Michel Muller**, "Elementary Functions: Algorithms and Implementation", 3rd ed., Birkhäuser, 2016.
5. **IEEE 754-2019**, IEEE Standard for Floating-Point Arithmetic.
6. **RLIBM** — rlibm.github.io, Rutgers University — 구간별 올바른 반올림 라이브러리
7. **GiNaC** — www.ginac.de — C++ 기호 연산 라이브러리 (설계 참조)
8. **Pynac** — pynac.org — Sage용 GiNaC 포크 (성능 비교 기준)
9. **Jean-Marie Toubiana**, "Elementary Methods in Symbolic Computation" — 부분분수·역변환 알고리즘 참조
