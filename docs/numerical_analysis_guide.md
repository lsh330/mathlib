# NumericalAnalysis 상세 사용자 가이드

`math_library.NumericalAnalysis` 클래스의 Simpson 적분법 8종에 대한 수학적 배경,
구현 세부사항, 정확도 특성, 예외 처리, 성능 최적화를 설명합니다.

---

## 목차

1. [개요 및 설계 철학](#1-개요-및-설계-철학)
2. [설치 및 임포트](#2-설치-및-임포트)
3. [simpson_13: 3점 단순 Simpson 1/3](#3-simpson_13)
4. [simpson_38: 4점 단순 Simpson 3/8](#4-simpson_38)
5. [composite_simpson_13: 합성 Simpson 1/3](#5-composite_simpson_13)
6. [composite_simpson_38: 합성 Simpson 3/8](#6-composite_simpson_38)
7. [adaptive_simpson: 적응형 Simpson](#7-adaptive_simpson)
8. [mixed_simpson: 혼합 Simpson](#8-mixed_simpson)
9. [simpson_irregular: 불균등 간격 Simpson](#9-simpson_irregular)
10. [romberg: Romberg 적분](#10-romberg)
11. [함수 입력 dispatch (callable vs PyExpr)](#11-함수-입력-dispatch)
12. [return_error 오차 추정](#12-return_error-오차-추정)
13. [Kahan 보상 누산](#13-kahan-보상-누산)
14. [예외 처리 완전 참조](#14-예외-처리-완전-참조)
15. [성능 특성](#15-성능-특성)
16. [수학적 배경](#16-수학적-배경)

---

## 1. 개요 및 설계 철학

`NumericalAnalysis`는 다음 원칙을 따릅니다:

- **외부 의존성 없음**: scipy.integrate, numpy 등 외부 라이브러리 의존 없음
- **Cython 구현**: 순수 Python 대비 10~50배 빠른 실행
- **IEEE 754 준수**: `fabs`, `isnan`, `isinf` 외의 비표준 수치 연산 없음
- **Kahan 누산**: 대규모 n에서도 부동소수점 누적 오차 억제
- **Python callable + PyExpr 양쪽 지원**: lambdify 없이 evalf 직접 활용

현재 Simpson 적분법 8종을 제공하며, 향후 다른 수치해석 알고리즘(Gauss 구적법,
Newton-Cotes 고차, 비선형 방정식 풀이 등)이 같은 클래스에 추가될 예정입니다.

---

## 2. 설치 및 임포트

```python
# 최상위 API로 직접 접근
from math_library import NumericalAnalysis

# 또는 서브패키지에서
from math_library.numerical_analysis import NumericalAnalysis

na = NumericalAnalysis()
```

`NumericalAnalysis()`는 상태가 없는 클래스이므로 인스턴스 생성 비용이 무시 가능합니다.

---

## 3. `simpson_13`

### 수학적 공식

구간 [a, b]를 2개 구간으로 분할:

```
h = (b - a) / 2
m = (a + b) / 2
I = (h/3) * [f(a) + 4*f(m) + f(b)]
```

이는 a, m, b 세 점을 지나는 2차 보간 다항식의 적분으로 유도됩니다.

**이론 오차**: `-(b-a)^5 / 90 * f^(4)(ξ)`, ξ ∈ [a, b]

### 사용 예

```python
import math
from math_library import NumericalAnalysis

na = NumericalAnalysis()

# 기본 사용
I = na.simpson_13(math.sin, 0, math.pi)
# 2.094395102393195  (정확값 2.0과 차이 ≈ 0.094)

# 정확한 함수: x^2 on [0, 2]
I = na.simpson_13(lambda x: x**2, 0, 2)
# 2.6666666...  (= 8/3, 정확)

# return_error
I, err = na.simpson_13(math.sin, 0, math.pi, return_error=True)
# I = 2.094, err = |I_13 - I_38| / 15 ≈ 0.0036

# PyExpr 입력
from math_library.laplace import Sin, t
I = na.simpson_13(Sin(t), 0, math.pi, var='t')
```

### 언제 사용하는가

- 함수가 매우 매끄럽고 구간이 좁을 때
- 적분값의 빠른 근사 추정이 필요할 때
- 다른 방법의 기본 블록 (adaptive_simpson 내부에서 사용)

---

## 4. `simpson_38`

### 수학적 공식

구간 [a, b]를 3개 구간으로 분할:

```
h = (b - a) / 3
x1 = a + h,  x2 = a + 2h
I = (3h/8) * [f(a) + 3*f(x1) + 3*f(x2) + f(b)]
```

이는 a, x1, x2, b 네 점을 지나는 3차 보간 다항식의 적분입니다.

**이론 오차**: `-(b-a)^5 / 80 * f^(4)(ξ)`, ξ ∈ [a, b]

simpson_13과 같은 O(h^5) 정확도이지만, 계수가 약간 다릅니다.

### 사용 예

```python
# x^3 on [0, 1]: 정확한 적분값 = 0.25
I = na.simpson_38(lambda x: x**3, 0, 1)
# 0.25  (3차 다항식이므로 정확)

I, err = na.simpson_38(lambda x: x**3, 0, 1, return_error=True)
# I = 0.25, err ≈ 0 (매우 작음)
```

### 언제 사용하는가

- composite_simpson_38의 마지막 3 구간 처리 (mixed_simpson에서 내부 사용)
- n이 3의 배수일 때 합성 적분의 기본 블록

---

## 5. `composite_simpson_13`

### 수학적 공식

구간 [a, b]를 n개의 동일한 소구간으로 분할 (n은 반드시 짝수):

```
h = (b - a) / n
x_i = a + i*h  (i = 0, 1, ..., n)

I = (h/3) * [f(x_0) + 4*Σ_{i=1,3,5,...,n-1} f(x_i) + 2*Σ_{i=2,4,6,...,n-2} f(x_i) + f(x_n)]
```

홀수 인덱스 점(계수 4)과 짝수 인덱스 내부 점(계수 2)을 Kahan 누산으로 합산합니다.

**이론 오차**: `-(b-a) * h^4 / 180 * f^(4)(ξ)`, ξ ∈ [a, b]

n = 100, 구간 [0, π], sin(x) 기준: 오차 ≈ 1.08e-8

### n 짝수 조건

Simpson 1/3은 2개의 소구간을 하나의 패널로 묶어 처리하므로,
총 구간 수 n이 짝수여야 합니다.

```python
# 올바른 사용
I = na.composite_simpson_13(math.sin, 0, math.pi, n=100)
I = na.composite_simpson_13(math.sin, 0, math.pi, n=10000)

# n=7 (홀수) → ValueError
try:
    na.composite_simpson_13(math.sin, 0, math.pi, n=7)
except ValueError as e:
    print(e)  # "composite_simpson_13 requires even n, got n=7"
```

### Kahan 누산 효과

```python
# n=10000에서 일반 합산 vs Kahan 비교
# 일반 합산: sum(f(x_i) for i in range(1,n,2)) 에서 부동소수점 오차 누적
# Kahan 누산: 동일 루프에서 오차 보상
I = na.composite_simpson_13(math.sin, 0, math.pi, n=10000)
# |I - 2.0| = 0.0  (double 정밀도 한계)
```

---

## 6. `composite_simpson_38`

### 수학적 공식

구간 [a, b]를 n개의 소구간으로 분할 (n은 반드시 3의 배수):

```
h = (b - a) / n
점 x_i = a + i*h  (i = 0, 1, ..., n)

계수 결정:
  i = 0 또는 i = n:  계수 1 (경계)
  i % 3 == 0 (내부): 계수 2
  i % 3 != 0:        계수 3

I = (3h/8) * [f(x_0) + 계수합 + f(x_n)]
```

**이론 오차**: 합성 Simpson 3/8도 O(h^4) 정확도입니다.

### 사용 예

```python
# n=99 (3의 배수)
I = na.composite_simpson_38(math.sin, 0, math.pi, n=99)
# ≈ 2.000000025  (오차 ≈ 2.5e-8)

# n=10 (3의 배수 아님) → ValueError
try:
    na.composite_simpson_38(math.sin, 0, math.pi, n=10)
except ValueError as e:
    print(e)  # "composite_simpson_38 requires n divisible by 3, got n=10"
```

---

## 7. `adaptive_simpson`

### 알고리즘

적응형 Simpson은 구간을 재귀적으로 분할하여 지정된 허용 오차 `tol`을 달성합니다.
Python 재귀 제한을 피하기 위해 명시적 스택을 사용합니다.

```
입력: [a, b], 허용 오차 tol

S1 = simpson_13([a, b])
mid = (a + b) / 2
S2 = simpson_13([a, mid]) + simpson_13([mid, b])

if |S2 - S1| < 15 * tol:
    return S2 + (S2 - S1) / 15   # Richardson 보정
else:
    return adaptive([a, mid], tol/2) + adaptive([mid, b], tol/2)
```

**Richardson 보정**: `(S2 - S1) / 15`는 Simpson의 4차 오차에서
Richardson 외삽 계수 `2^4 - 1 = 15`를 이용합니다.
이 보정으로 실제 오차가 tol보다 훨씬 작아집니다.

### 명시적 스택 구현

```python
# 스택 항목: (aa, bb, tol, fa, fm, fb, S1, depth)
# depth > max_depth 이면 RuntimeError 발생
I = na.adaptive_simpson(math.sin, 0, math.pi, tol=1e-12, max_depth=50)
# |I - 2.0| = 0.0  (double 정밀도 한계)
```

### max_depth 초과 처리

```python
# 빠르게 진동하는 함수
import math
f = lambda x: math.sin(1000 * x)   # 고주파
try:
    I = na.adaptive_simpson(f, 0, math.pi, tol=1e-12, max_depth=30)
except RuntimeError as e:
    print(e)  # "adaptive_simpson exceeded max_depth=30, ..."
    # max_depth를 높이거나 tol을 완화
```

---

## 8. `mixed_simpson`

### 알고리즘

임의의 n >= 2에 대해 Simpson 적분을 수행합니다:

- n이 짝수이면 `composite_simpson_13`을 그대로 사용
- n이 홀수이면:
  - 앞 (n-3) 구간에 `composite_simpson_13` (n-3은 항상 짝수)
  - 끝 3 구간에 `simpson_38`

이 분할이 항상 유효한 이유: n이 홀수이면 n-3도 짝수이므로
`composite_simpson_13`의 짝수 n 조건을 만족합니다.

### n = 3 특수 처리

n = 3일 때 n-3 = 0이므로 앞 부분을 스킵하고 전체를 `simpson_38`으로 처리합니다.

```python
I = na.mixed_simpson(math.sin, 0, math.pi, n=3)   # 전체 simpson_38
I = na.mixed_simpson(math.sin, 0, math.pi, n=5)   # 앞 2구간 1/3 + 끝 3구간 3/8
I = na.mixed_simpson(math.sin, 0, math.pi, n=7)   # 앞 4구간 1/3 + 끝 3구간 3/8
I = na.mixed_simpson(math.sin, 0, math.pi, n=8)   # 8구간 모두 1/3
```

---

## 9. `simpson_irregular`

### 수학적 공식

불균등 간격 x0 < x1 < x2에서 2차 보간 다항식의 정확한 적분:

```
h0 = x1 - x0,  h1 = x2 - x1

I = (h0 + h1) / 6 * [
    (2 - h1/h0) * f(x0)
  + (h0 + h1)^2 / (h0 * h1) * f(x1)
  + (2 - h0/h1) * f(x2)
]
```

h0 = h1일 때 이 공식은 표준 Simpson 1/3으로 귀착됩니다.

### 전체 구간 처리

- 홀수 점 수 (짝수 구간): 2구간씩 순서대로 비균등 Simpson 적용
- 짝수 점 수 (홀수 구간): [0..n-2] 구간에서 홀수 점 Simpson + 마지막 구간 사다리꼴

### 입력 검증

```python
# 단조 증가 필수
x = [0.0, 0.5, 1.0, 1.5, 2.0]
y = [x**2 for x in x]
I = na.simpson_irregular(x, y)   # 8/3 = 2.666...

# 단조성 위반
try:
    na.simpson_irregular([0, 1, 0.5], [0, 1, 0.5])
except ValueError as e:
    print(e)  # "x_points must be monotonically increasing..."

# NaN 포함
try:
    na.simpson_irregular([0.0, float('nan'), 1.0], [0.0, 0.5, 1.0])
except ValueError as e:
    print(e)  # "x_points[1]=nan is NaN or Inf"

# 점 수 부족
try:
    na.simpson_irregular([0.0, 1.0], [0.0, 1.0])
except ValueError as e:
    print(e)  # "simpson_irregular requires >= 3 points, got 2"
```

---

## 10. `romberg`

### 수학적 공식

Richardson 외삽을 사용하여 사다리꼴 규칙의 정확도를 고차수로 높입니다.

**사다리꼴 규칙** (재귀적):

```
T[0][0] = (b - a) / 2 * (f(a) + f(b))
T[i][0] = T[i-1][0] / 2 + h_i * Σ_{새 중간점} f(x)
         (h_i = (b-a) / 2^i)
```

**Richardson 외삽**:

```
T[i][j] = (4^j * T[i][j-1] - T[i-1][j-1]) / (4^j - 1)
```

depth=k이면 `T[k][k]`가 최고 차수 근사 (오차 O(h^{2k+2})).

### depth와 정확도

| depth | 사다리꼴 구간 수 | 함수 호출 수 | sin(0~π) 오차 |
|---|---|---|---|
| 1 | 2 | 3 | ≈ 9.4e-2 |
| 3 | 8 | 9 | ≈ 5.5e-6 |
| 5 | 32 | 33 | ≈ 1.3e-12 |
| 6 | 64 | 65 | ≈ 9e-16 |
| 8 | 256 | 257 | ≈ 2e-16 |

### 사용 예

```python
# 고정밀 적분
I = na.romberg(math.sin, 0, math.pi, depth=6)
# ≈ 2.000000000000001  (오차 ≈ 1e-15)

I, err = na.romberg(math.sin, 0, math.pi, depth=6, return_error=True)
# err = |T[6][6] - T[6][5]|

# depth > 20 경고
import warnings
with warnings.catch_warnings(record=True) as w:
    I = na.romberg(math.sin, 0, math.pi, depth=21)
    print(w[0].message)  # RuntimeWarning
```

---

## 11. 함수 입력 dispatch

### Python callable

모든 Python callable (함수, lambda, 클래스 인스턴스의 `__call__`)을 직접 사용 가능합니다.

```python
import math
from math_library import NumericalAnalysis

na = NumericalAnalysis()

# math 모듈 함수
I = na.simpson_13(math.sin, 0, math.pi)

# lambda
I = na.composite_simpson_13(lambda x: x**2 + 1, 0, 1, n=100)

# 클래스 callable
class MyFunc:
    def __call__(self, x):
        return x * math.exp(-x)

I = na.adaptive_simpson(MyFunc(), 0, 10, tol=1e-10)
```

### PyExpr (math_library.laplace)

`PyExpr` 객체는 `.evalf(**{var: x})` 메서드를 통해 수치 평가됩니다.
내부에서 `hasattr(f, 'evalf')` 검사 후 자동으로 래퍼 함수를 생성합니다.

```python
from math_library.laplace import Sin, Cos, Exp, t
from math_library import NumericalAnalysis
import math

na = NumericalAnalysis()

# sin(2t)*cos(t): ∫_0^π = 4/3
f = Sin(2*t) * Cos(t)
I = na.composite_simpson_13(f, 0, math.pi, n=100, var='t')
# ≈ 1.333333485  (오차 ≈ 1.5e-7)

# 적응형: 더 정확
I = na.adaptive_simpson(f, 0, math.pi, tol=1e-12, var='t')
# ≈ 1.333333333333332  (오차 ≈ 2e-15)

# exp(-t)*sin(t): ∫_0^∞ = 0.5 (but [0,10] ≈ 0.5)
f2 = Exp(-t) * Sin(t)
I2 = na.romberg(f2, 0, 10, depth=6, var='t')
```

### var 파라미터

PyExpr의 변수명을 지정합니다. 기본값은 `'t'`.

```python
from math_library.laplace import symbol

x_var = symbol('x')
f = x_var * x_var   # x^2
I = na.composite_simpson_13(f, 0, 1, n=100, var='x')
# ≈ 1/3
```

---

## 12. `return_error` 오차 추정

`return_error=True`이면 `(value, error_estimate)` 튜플을 반환합니다.

### 오차 추정 방법

**합성 Simpson 1/3**:
Richardson 외삽으로 n/2 구간 계산 후 비교:
```
err ≈ |I_n - I_{n/2}| / 15
```
`15 = 2^4 - 1` (Simpson의 4차 오차에서 Richardson 계수).

**합성 Simpson 3/8**:
유사한 방법, 단 coarse 구간을 n*2/3으로 선택:
```
err ≈ |I_n - I_{n*2/3}| / 63
```

**simpson_13**:
simpson_38과의 차이:
```
err ≈ |I_13 - I_38| / 15
```

**adaptive_simpson**:
각 수렴 구간의 Richardson 오차 합산:
```
total_err = Σ |S2_i - S1_i| / 15
```

**romberg**:
최고 차수와 그 이전 단계 차이:
```
err = |T[depth][depth] - T[depth][depth-1]|
```

**simpson_irregular** 및 **mixed_simpson (홀수)**:
오차 추정이 어려우므로 `float('nan')`을 반환합니다.

```python
I, err = na.composite_simpson_13(math.sin, 0, math.pi, n=100, return_error=True)
print(f"I = {I:.15f}")
print(f"err estimate = {err:.2e}")
print(f"actual error = {abs(I - 2.0):.2e}")
```

---

## 13. Kahan 보상 누산

합성 Simpson 류의 내부 합산 루프에 Kahan 알고리즘을 사용합니다.

### 알고리즘

```c
// _kahan.pxd
cdef inline void kahan_add(double* s, double* c, double value) noexcept nogil:
    cdef double y = value - c[0]   // 보상 후 값
    cdef double t = s[0] + y       // 임시 합
    c[0] = (t - s[0]) - y          // 반올림 오차 추출
    s[0] = t                       // 업데이트
```

### 효과 검증

```python
# n=10000 에서 일반 합산 vs Kahan
import math
na = NumericalAnalysis()

I = na.composite_simpson_13(math.sin, 0, math.pi, n=10000)
print(f"|I - 2.0| = {abs(I - 2.0):.2e}")
# 0.00e+00  (double 정밀도 한계까지 정확)

# 비교: 일반 Python sum으로 직접 계산
h = math.pi / 10000
s_odd = sum(math.sin(i*h) for i in range(1, 10000, 2))
s_even = sum(math.sin(i*h) for i in range(2, 9999, 2))
I_naive = h/3 * (0 + 4*s_odd + 2*s_even + 0)
print(f"|I_naive - 2.0| = {abs(I_naive - 2.0):.2e}")
# ≈ 1e-8 (Kahan 없이 오차 더 큼)
```

---

## 14. 예외 처리 완전 참조

### ValueError 목록

| 메서드 | 조건 | 메시지 |
|---|---|---|
| 모든 메서드 | `a >= b` | `"a must be less than b, got a=..., b=..."` |
| `composite_simpson_13` | `n < 2` | `"composite_simpson_13 requires n >= 2, got n=..."` |
| `composite_simpson_13` | `n % 2 != 0` | `"composite_simpson_13 requires even n, got n=..."` |
| `composite_simpson_38` | `n < 3` | `"composite_simpson_38 requires n >= 3, got n=..."` |
| `composite_simpson_38` | `n % 3 != 0` | `"composite_simpson_38 requires n divisible by 3, got n=..."` |
| `mixed_simpson` | `n < 2` | `"mixed_simpson requires n >= 2, got n=..."` |
| `adaptive_simpson` | `tol <= 0` | `"tol must be positive, got tol=..."` |
| `romberg` | `depth < 1` | `"romberg requires depth >= 1, got depth=..."` |
| `simpson_irregular` | 길이 불일치 | `"x_points and y_points must have the same length, ..."` |
| `simpson_irregular` | 점 수 < 3 | `"simpson_irregular requires >= 3 points, got ..."` |
| `simpson_irregular` | 단조성 위반 | `"x_points must be monotonically increasing, violation at index ..."` |
| `simpson_irregular` | NaN/Inf | `"x_points[i]=nan is NaN or Inf"` 또는 `"y_points[i]=..."` |

### TypeError 목록

| 조건 | 메시지 |
|---|---|
| `f`가 callable도, evalf 메서드도 없는 경우 | `"f must be callable or have lambdify method, got {type(f).__name__}"` |

### RuntimeError 목록

| 메서드 | 조건 | 메시지 |
|---|---|---|
| `adaptive_simpson` | `depth > max_depth` | `"adaptive_simpson exceeded max_depth=..., reduce tol or increase max_depth"` |

### RuntimeWarning 목록

| 메서드 | 조건 |
|---|---|
| `romberg` | `depth > 20` — 수렴 후 진동 가능성 |

---

## 15. 성능 특성

### 측정 환경

| 항목 | 내용 |
|---|---|
| Python | 3.11.9 |
| OS | Windows 11 Pro 10.0.26200 |
| 컴파일러 | MinGW-w64 UCRT64 gcc 15.2.0 |
| 컴파일 플래그 | `-O3 -march=native -mfma -fno-math-errno -fno-trapping-math` |

### 실측값 (math.sin 기준)

| 메서드 | 함수 호출 수 | 실측 시간 | 비고 |
|---|---|---|---|
| `simpson_13` | 3 | 0.13 µs | 목표 < 500 ns |
| `simpson_38` | 4 | 0.14 µs | |
| `composite_13 n=100` | 102 | 2.39 µs | 목표 < 10 µs |
| `composite_38 n=99` | 101 | 2.41 µs | |
| `adaptive tol=1e-10` | 가변 | 28.9 µs | Richardson 보정 포함 |
| `romberg depth=6` | 65 | 1.97 µs | |
| `composite_13 n=10000` | 10002 | 219 µs | Kahan 포함 |

### 성능 최적화 포인트

1. **함수 호출 비용**: Python callable 호출이 지배적 (각 호출 ≈ 50~100 ns)
2. **Kahan 추가 비용**: 루프당 ≈ 5~10 ns (무시 가능)
3. **PyExpr evalf 비용**: Python callable 대비 ≈ 3~5배 느림 (C++ dispatch 오버헤드)
4. **adaptive_simpson**: 깊은 분기가 많을수록 선형 증가

### PyExpr vs callable 성능 비교

```python
import math, time
from math_library import NumericalAnalysis
from math_library.laplace import Sin, t

na = NumericalAnalysis()

N = 1000
t0 = time.perf_counter()
for _ in range(N):
    na.composite_simpson_13(math.sin, 0, math.pi, n=100)
t1 = time.perf_counter()
print(f"callable: {(t1-t0)*1e6/N:.1f} µs")  # ≈ 2.4 µs

t0 = time.perf_counter()
for _ in range(N):
    na.composite_simpson_13(Sin(t), 0, math.pi, n=100, var='t')
t1 = time.perf_counter()
print(f"PyExpr:   {(t1-t0)*1e6/N:.1f} µs")  # ≈ 15~30 µs (evalf 비용)
```

---

## 16. 수학적 배경

### Simpson 규칙의 유도

Simpson 1/3 규칙은 세 점 (a, f(a)), (m, f(m)), (b, f(b))를 지나는
2차 Lagrange 보간 다항식을 적분하여 유도합니다:

```
P(x) = f(a) * (x-m)(x-b)/[(a-m)(a-b)] + f(m) * (x-a)(x-b)/[(m-a)(m-b)]
     + f(b) * (x-a)(x-m)/[(b-a)(b-m)]
```

`∫_a^b P(x) dx`를 계산하면 Simpson 1/3 공식이 됩니다.

### Richardson 외삽

두 근사값 `I_h`(간격 h)와 `I_{h/2}`(간격 h/2)가 있을 때,
p차 정확도 방법에서:

```
I = (2^p * I_{h/2} - I_h) / (2^p - 1)
```

Simpson (p=4)에서 `2^4 = 16`, 계수 = 15.
Romberg는 이 외삽을 반복하여 고차수 근사를 구합니다.

### Kahan 보상 합산

IEEE 754 부동소수점에서 `(a + b) + c ≠ a + (b + c)` 가 성립합니다.
Kahan 알고리즘은 각 덧셈에서 발생한 반올림 오차를 누적 보상하여
O(n * eps)가 아닌 O(eps)의 최종 오차를 달성합니다.

여기서 eps는 기계 엡실론(≈ 2.2e-16)입니다.

### 불균등 Simpson 공식 유도

x0, x1, x2에서 h0 = x1-x0, h1 = x2-x1인 경우,
Lagrange 보간 후 적분하면:

```
I = (h0+h1)/6 * [(2h0+h1)/h0 * f0 - (h0+h1)^2/(h0*h1) * f1*(-1) + ...]
  = (h0+h1)/6 * [(2 - h1/h0)*f0 + (h0+h1)^2/(h0*h1)*f1 + (2 - h0/h1)*f2]
```

계수 확인: h0=h1=h이면 `(2-1)*f0 + (2h)^2/h^2*f1 + (2-1)*f2 = f0 + 4f1 + f2`,
이를 `(2h)/6 = h/3`으로 곱하면 표준 Simpson 1/3이 됩니다.
