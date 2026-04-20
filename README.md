# mathlib

Python `math` 모듈의 대체를 목표로 설계한 **Cython 기반 자체 구현 수학 라이브러리**입니다.
단순 libm 래퍼가 아닌, musl libc / fdlibm 계수를 Cython Horner-FMA 형태로 직접 재구현하여
`math` 모듈과 동등하거나 더 빠른 성능을 달성하면서 IEEE 754 완전 준수와 ULP 수준의 정확도를 보장합니다.
삼각·역삼각·쌍곡·지수/로그 함수 17개 primitive 외에, `math`에 없는 특수 함수(gamma, beta,
Bessel, Legendre, Lambert W, zeta, Euler φ 등)와 다변수 자동 미분 클래스(`Differentiation`)를 포함합니다.

---

## 목차

1. [특징](#1-특징)
2. [설치](#2-설치)
3. [5분 빠른 시작](#3-5분-빠른-시작)
4. [API 레퍼런스](#4-api-레퍼런스)
5. [성능 벤치마크](#5-성능-벤치마크)
6. [정확도 보장](#6-정확도-보장)
7. [구현 철학](#7-구현-철학)
8. [프로젝트 구조](#8-프로젝트-구조)
9. [개발 및 테스트](#9-개발-및-테스트)
10. [현재 한계 및 TODO](#10-현재-한계-및-todo)
11. [라이선스](#11-라이선스)
12. [참고문헌](#12-참고문헌)

---

## 1. 특징

| 항목 | 내용 |
|---|---|
| **대상 Python** | 3.10+ |
| **구현 언어** | Cython 3.0+ (C 확장 모듈로 컴파일) |
| **자체 구현** | `libc.math`의 sin/cos/exp/log 직접 호출 금지 — musl 계수 Cython 재구현 |
| **허용 헬퍼** | `fma`, `frexp`, `ldexp`, `sqrt`, `isnan`, `isinf` 등 IEEE 754 헬퍼만 허용 |
| **ULP 정확도** | sin/cos/arctan/arccos/exp/ln 최대 1 ULP, sqrt 0 ULP (correctly rounded) |
| **복소수 dispatch** | 모든 17개 primitive가 `complex` 입력 자동 감지 후 복소 경로 분기 |
| **IEEE 754 특수값** | NaN, ±∞, ±0, subnormal 전 항목 준수 |
| **큰 인수 대응** | Payne-Hanek 구현으로 `sin(1e20)` 등 대규모 인수에서 `math.sin`과 0 ULP 일치 |
| **추가 특수 함수** | gamma, beta, Bessel J, Legendre, Lambert W, Riemann zeta, Euler φ, Heaviside |
| **수치 미분** | Ridders-Richardson extrapolation 기반 Differentiation 클래스 (25+ 메서드) |

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

# --- 복소수 자동 dispatch ---
print(m.sin(1 + 2j))       # (3.165778513216168+1.959601041421606j)
print(m.ln(1 + 2j))        # (0.8047189562170503+1.1071487177940904j)
print(m.sqrt(-1 + 0j))     # 1j

# --- 특수 함수 ---
print(m.gamma(5))           # 23.999999999999996  (≈ 4! = 24)
print(m.zeta(2))            # 1.6449340668481436  (≈ π²/6)
print(m.bessel_j(0, 1.0))  # 0.7651976865579666

# --- IEEE 754 특수값 ---
import math
print(m.ln(0.0))            # -inf
print(m.ln(-1.0))           # nan
print(m.sqrt(-1.0))         # nan  (실수 입력에서는 NaN)

# --- 수치 미분 (Ridders-Richardson extrapolation) ---
from math_library import Differentiation
d = Differentiation()
print(d.single_variable(m.sin, 1.0))              # ≈ cos(1) = 0.540302...  (수치미분 오차 포함)
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

특수값 동작 (sin 기준, cos/tan 유사):

| 입력 | `sin` | `cos` |
|---|---|---|
| `NaN` | `NaN` | `NaN` |
| `±0` | `±0` | `1.0` |
| `±∞` | `NaN` | `NaN` |

### 4.3 역삼각함수

| 함수 | 시그니처 | 실수 정의역 | 반환 범위 |
|---|---|---|---|
| `arcsin` | `arcsin(x)` | `[-1, 1]` | `[-π/2, π/2]` |
| `arccos` | `arccos(x)` | `[-1, 1]` | `[0, π]` |
| `arctan` | `arctan(x)` | 모든 실수 | `(-π/2, π/2)` |
| `arcsec` | `arcsec(x)` | `|x| ≥ 1` | — |
| `arccosec` | `arccosec(x)` | `|x| ≥ 1` | — |
| `arccotan` | `arccotan(x)` | 모든 실수 | — |

### 4.4 쌍곡함수

| 함수 | 시그니처 | 비고 |
|---|---|---|
| `hypersin` | `hypersin(x)` | sinh(x) |
| `hypercos` | `hypercos(x)` | cosh(x) |
| `hypertan` | `hypertan(x)` | tanh(x) |
| `hypersec` | `hypersec(x)` | 1/cosh(x) |
| `hypercosec` | `hypercosec(x)` | 1/sinh(x), x≠0 예외 |
| `hypercotan` | `hypercotan(x)` | cosh(x)/sinh(x), x≠0 예외 |

### 4.5 지수 및 로그

| 함수 | 시그니처 | 특수값 |
|---|---|---|
| `exp` | `exp(x)` | `exp(-∞)=0`, `exp(+∞)=+∞`, x>709.78 → `+∞`, x<-745.13 → subnormal |
| `ln` | `ln(x)` | `ln(0)=-∞`, `ln(x<0)=NaN`, `ln(+∞)=+∞` |
| `log` | `log(base, x)` | 임의 밑 로그 |
| `sqrt` | `sqrt(x)` | `sqrt(±0)=±0`, `sqrt(-x)=NaN` (실수), `sqrt(+∞)=+∞` |
| `power` | `power(base, exponent)` | `pow(x,±0)=1`, `pow(-1,±∞)=1` |

### 4.6 특수 함수

| 함수 | 시그니처 | 설명 |
|---|---|---|
| `gamma` | `gamma(n)` | 감마 함수, `gamma(n) = (n-1)!` (양의 정수) |
| `beta` | `beta(a, b)` | 베타 함수 |
| `bessel_j` | `bessel_j(n, x)` | 제1종 베셀 함수 J_n(x) |
| `bessel_j0` | `bessel_j0(x)` | J_0(x) 고속 경로 |
| `bessel_j1` | `bessel_j1(x)` | J_1(x) 고속 경로 |
| `legendre` | `legendre(n, x)` | 르장드르 다항식 P_n(x) |
| `lambert_w` | `lambert_w(x)` | Lambert W 함수 (주가지 W_0) |
| `zeta` | `zeta(s)` | 리만 제타 함수 ζ(s) |
| `euler_pi` | `euler_pi(n)` | 오일러 수 (Euler numbers) |
| `euler_phi` | `euler_phi(n)` | 오일러 토션트 함수 φ(n) |
| `heaviside` | `heaviside(x)` | 헤비사이드 계단 함수 |
| `gcd` | `gcd(a, b)` | 최대공약수 |
| `lcm` | `lcm(a, b)` | 최소공배수 |

### 4.7 Differentiation 클래스

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

## 5. 성능 벤치마크

### 측정 환경

| 항목 | 내용 |
|---|---|
| Python | 3.11.9 |
| OS | Windows 11 Pro 10.0.26200 |
| 컴파일러 | MinGW-w64 UCRT64 gcc 15.2.0 |
| 컴파일 플래그 | `-O3 -march=native -mfma -fno-math-errno -fno-trapping-math` |
| 측정 방법 | 워밍업 5,000회 + 반복 1,000,000회, `time.perf_counter_ns()` |
| 테스트 입력 | `x = 1.2345` (실수 단일 스칼라) |

### 결과

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

### 벤치마크 재현

```bash
python bench/perf_compare.py
```

---

## 6. 정확도 보장

### ULP 오차 측정 결과

1M 무작위 입력(double 범위)에 대한 측정:

| 함수 | 평균 ULP | 최대 ULP | 판정 |
|---|---|---|---|
| sin | 0.017 | 1 | PASS |
| cos | 0.017 | 1 | PASS |
| tan | 0.186 | 2 | PASS |
| arcsin | 0.184 | 2 | PASS |
| arccos | 0.009 | 1 | PASS |
| arctan | 0.000 | 1 | PASS |
| exp | 0.096 | 1 | PASS |
| ln | 0.247 | 1 | PASS |
| sqrt | 0.000 | 0 | PASS (correctly rounded) |

### Payne-Hanek 큰 인수 대응

`|x| ≥ 2²⁰·π/2 ≈ 1.647×10⁶` 이상의 인수에서 Payne-Hanek 알고리즘을 사용합니다.
2/π의 396자리 16진 테이블(`ipio2[]`)을 기반으로 하며, 입력 크기와 무관하게 정밀 나머지 계산을 수행합니다.

### IEEE 754 특수값 완전 준수

```
sin(NaN) = NaN        sin(±∞) = NaN      sin(±0) = ±0
ln(0)    = -∞         ln(-x)  = NaN      ln(+∞)  = +∞
exp(-∞)  = +0         exp(+∞) = +∞
sqrt(+0) = +0         sqrt(-0) = -0      sqrt(-1) = NaN
exp(-745) → 5e-324 (subnormal 정상 처리)
```

---

## 7. 구현 철학

**자체 구현 원칙**: `libc.math`의 `sin`, `cos`, `exp`, `log` 등 elementary 함수를 직접 호출하지 않습니다.
musl libc와 fdlibm의 bit-exact 계수를 그대로 인용하여 Cython + Horner-FMA 형태로 재구현합니다.

- **Argument Reduction**: sin/cos/tan에 Cody-Waite 3단계 + Payne-Hanek 구현
- **다항식 근사**: musl 출처 Remez minimax 계수, Horner 전개 + `libc.math.fma` 활용
- **테이블 기반 ln/exp**: 128-entry 룩업 테이블로 musl `log.c`, `exp.c` 재구현
- **pow**: `exp(y·log(x))` 구조, 정수 지수는 이진 거듭제곱 분기로 오차 억제
- **sqrt**: IEEE 754 필수 요건(correctly rounded)을 만족하는 `libc.math.sqrt` 하드웨어 호출 허용

계수 출처 및 알고리즘 상세는 `docs/algorithm_reference.md`를 참조하십시오.

---

## 8. 프로젝트 구조

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
│   │   ├── logarithmic.pyx/.pxd        # ln, log (128-entry 테이블)
│   │   ├── power_sqrt.pyx/.pxd         # power, sqrt
│   │   ├── trigonometric.pyx/.pxd      # sin, cos, tan, sec, cosec, cotan
│   │   ├── inverse_trig.pyx/.pxd       # arcsin, arccos, arctan, arcsec, ...
│   │   └── hyperbolic.pyx/.pxd         # hypersin, hypercos, hypertan, ...
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
│   └── differentiation/differentiation.pyx
├── bench/
│   └── perf_compare.py                 # math vs mathlib 속도 비교 벤치마크
├── tests/                              # 카테고리별 테스트 스크립트 (18개)
└── docs/
    ├── algorithm_reference.md          # 알고리즘 상세 및 bit-exact 계수
    ├── cython_best_practices.md        # 빌드 및 Cython 모범 사례
    ├── implementation_spec.md          # 구현 명세
    ├── performance_benchmark.md        # 벤치마크 원본 결과
    └── verification_report.md          # 검증 보고서
```

패키지 임포트 이름과 dist 이름 모두 `math_library`입니다.

```python
import math_library as m   # 올바른 임포트
```

---

## 9. 개발 및 테스트

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

# pytest (현재 설정 기준)
pytest tests/
```

### Cython 어노테이션으로 성능 병목 확인

```bash
cython -a src/math_library/_core/trigonometric.pyx
# 생성된 trigonometric.html을 브라우저로 열어 노란색 라인(CPython 상호작용) 확인
```

---

## 10. 현재 한계 및 TODO

### 알려진 이슈

이 목록은 내부 검증 보고서(`docs/verification_report.md`) 기준입니다.

- **zeta(2) 정밀도**: 구현값 `1.6449340668481436`, 정답 `π²/6 = 1.6449340668482264` (절대오차 8.28e-14, ~370 ULP). 특수 함수 전용 개선 예정.
- **복소수 dispatch**: `cmath` 위임 경로는 내부적으로 Python cmath 모듈을 사용 (libc.math 직접 호출 없음).

> 이전 버전에서 보고된 Payne-Hanek 부호 처리, exp 언더플로우, pow(-1, ±∞), 21개 primitive 복소수 dispatch 이슈는 모두 수정 완료되었습니다.

### 추가 예정 함수 (2차 계획)

현재 구현되지 않은 함수들로, 향후 버전에서 추가할 예정입니다.

```
cbrt, expm1, log1p, log2, log10, atan2, hypot
asinh, acosh, atanh
fmod, fabs, floor, ceil, trunc, copysign
isnan, isinf, isfinite
```

### 장기 계획

- NumPy 배열/벡터 연산 지원 (`numpy.sin`과 경쟁 가능한 배열 경로)
- Linux / macOS CI 검증

---

## 11. 라이선스

MIT License — `LICENSE` 파일 참조.

계수 출처인 musl libc, fdlibm은 각각 MIT-style 및 BSD 라이선스입니다.

---

## 12. 참고문헌

1. **musl libc** — git.musl-libc.org/cgit/musl — sin, cos, exp, log 등 bit-exact 계수 출처
2. **fdlibm / FreeBSD msun** — 역삼각·쌍곡 함수 계수의 원 출처 (Sun Microsystems)
3. **CORE-MATH (INRIA)** — core-math.gitlabpages.inria.fr — correctly rounded 구현 참조
4. **Jean-Michel Muller**, "Elementary Functions: Algorithms and Implementation", 3rd ed., Birkhäuser, 2016.
5. **IEEE 754-2019**, IEEE Standard for Floating-Point Arithmetic.
6. **RLIBM** — rlibm.github.io, Rutgers University — 구간별 올바른 반올림 라이브러리
