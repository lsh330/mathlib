# math 모듈 마이그레이션 상세 가이드

> 대상: 기존 Python `math` 모듈 사용자
> 목적: `import math` → `import math_library as m` 전환 시 모든 함수의 정확한 대응 관계와 이유를 설명

---

## 목차

1. [일반화 원칙](#1-일반화-원칙)
2. [이름이 다른 함수 상세](#2-이름이-다른-함수-상세)
3. [일반화 표현 상세](#3-일반화-표현-상세)
4. [자동 real→complex 승격 상세](#4-자동-realcomplex-승격-상세)
5. [역쌍곡함수 마이그레이션](#5-역쌍곡함수-마이그레이션)
6. [상수 마이그레이션](#6-상수-마이그레이션)
7. [완전 마이그레이션 예시](#7-완전-마이그레이션-예시)

---

## 1. 일반화 원칙

mathlib는 **기존 구현된 것들로 일반화해서 표현 가능하면 일반화** 원칙을 따릅니다.

이 원칙의 근거는 다음과 같습니다.

1. **API 표면 최소화**: 동일한 계산을 여러 이름으로 제공하면 함수의 의미가 흐려지고
   사용자가 어떤 것을 써야 하는지 판단하기 어려워집니다.

2. **일관성**: `log(2, x)` 형태는 밑이 무엇이든 동일한 방식으로 표현됩니다.
   `log2`, `log10`은 이 일반형의 특수 케이스일 뿐입니다.

3. **정확도 동등성**: `m.log(2, x)`와 `math.log2(x)`는 동일한 수학적 결과를 산출합니다.
   일반화 표현을 쓴다고 해서 정밀도가 낮아지지 않습니다.

---

## 2. 이름이 다른 함수 상세

### 2.1 자연로그: `math.log(x)` → `m.ln(x)`

```python
import math
import math_library as m

math.log(2.718281828459045)   # 1.0
m.ln(2.718281828459045)       # 1.0
```

`math.log`는 단일 인수일 때 자연로그, 두 번째 인수 지정 시 임의 밑 로그로 동작합니다.
mathlib는 이 두 역할을 분리하여 자연로그는 `ln`, 임의 밑 로그는 `log(base, x)`로 구분합니다.

> **인수 순서 주의**: `math.log(x, base)` vs `m.log(base, x)`

```python
math.log(1000, 10)    # 2.9999999999999996
m.log(10, 1000)       # 2.9999999999999996  (base가 첫 번째)
```

### 2.2 역삼각함수: `math.asin/acos/atan` → `m.arcsin/arccos/arctan`

```python
import math
import math_library as m

math.asin(0.5)   # 0.5235987755982988
m.arcsin(0.5)    # 0.5235987755982988
```

mathlib는 `arc*` 접두어를 사용합니다. 기하학적 의미를 명확히 표현하기 위한 네이밍입니다.

### 2.3 쌍곡함수: `math.sinh/cosh/tanh` → `m.hypersin/hypercos/hypertan`

```python
import math
import math_library as m

math.sinh(1.0)   # 1.1752011936438014
m.hypersin(1.0)  # 1.1752011936438014
```

`hyper` 접두어는 쌍곡(hyperbolic)의 약어입니다.

### 2.4 거듭제곱: `math.pow(x, y)` → `m.power(x, y)`

```python
import math
import math_library as m

math.pow(2, 10)    # 1024.0
m.power(2, 10)     # 1024.0
```

`power`는 음수 밑 + 분수 지수에서 복소수를 자동 반환합니다.
`math.pow`는 이 경우 `ValueError`를 발생시킵니다.

---

## 3. 일반화 표현 상세

### 3.1 log2, log10 → log(base, x)

```python
import math_library as m

# math.log2(x) 대체
m.log(2, 1024)      # 10.0
m.log(2, 0.125)     # -3.0

# math.log10(x) 대체
m.log(10, 100)      # 2.0
m.log(10, 0.001)    # -3.0
m.log(10, 1e6)      # 6.0
```

**정확도**: `m.log(2, x)`는 `math.log2(x)`와 동일한 정밀도를 제공합니다.
두 구현 모두 최대 1 ULP 오차 이내입니다.

### 3.2 exp2 → power(2, x)

```python
import math_library as m

# math.exp2(x) 대체
m.power(2, 10)      # 1024.0
m.power(2, -3)      # 0.125
m.power(2, 0.5)     # 1.4142135623730951  (sqrt(2))
```

### 3.3 tau → 2 * pi()

```python
import math_library as m

tau = 2 * m.pi()    # 6.283185307179586
```

`math.tau`는 `2π`의 단축 표현일 뿐입니다. 산술로 직접 표현하면 동일한 값이 나옵니다.

### 3.4 fabs → abs()

```python
import math_library as m

abs(-3.14)          # 3.14  (Python 내장)
```

`math.fabs`는 Python 내장 `abs()`와 부동소수점 인수에서 동일하게 동작합니다.

### 3.5 degrees, radians → 산술 표현

```python
import math_library as m

# math.degrees(x) 대체
def degrees(x): return x * 180 / m.pi()

# math.radians(x) 대체
def radians(x): return x * m.pi() / 180

print(degrees(m.pi()))   # 180.0
print(radians(180))      # 3.141592653589793
```

단위 변환은 `π`와의 산술이므로 별도 함수화가 불필요합니다.

### 3.6 inf, nan → float 내장

```python
# math.inf 대체
pos_inf = float('inf')
neg_inf = float('-inf')

# math.nan 대체
nan_val = float('nan')
```

Python 내장 `float` 생성자로 동일하게 표현됩니다.

---

## 4. 자동 real→complex 승격 상세

mathlib는 `number_system` 플래그 없이 입력값과 수학적 결과를 기반으로
자동으로 실수/복소수 경로를 결정합니다.

### 4.1 sqrt(-x)

```python
import math_library as m
import math

math.sqrt(-1)     # ValueError: math domain error
m.sqrt(-1)        # 1j  (자동 복소수 승격)
m.sqrt(-4)        # 2j
m.sqrt(-2)        # 1.4142135623730951j
```

### 4.2 ln(음수)

```python
import math_library as m
import math

math.log(-1)      # ValueError: math domain error
m.ln(-1)          # 3.141592653589793j  (= πj)
m.ln(-math.e)     # (1+3.141592653589793j)
```

자연로그의 복소수 확장: `ln(-x) = ln(x) + πi` (x > 0)

### 4.3 arcsin, arccos (정의역 밖)

```python
import math_library as m
import math

math.asin(2)      # ValueError
m.arcsin(2)       # (1.5707963267948966-1.3169578969248166j)
m.arccos(2)       # 1.3169578969248166j
m.arcsin(-2)      # (-1.5707963267948966+1.3169578969248166j)
```

### 4.4 arc_hypercos, arc_hypertan (정의역 밖)

```python
import math_library as m

# arc_hypercos: 실수 정의역은 x >= 1
# x = 0.5 (정의역 밖) → 복소수
m.arc_hypercos(0.5)   # (-5.551115123125783e-17+1.0471975511965976j)
m.arc_hypercos(1.0)   # 0.0  (경계)
m.arc_hypercos(2.0)   # 1.3169578969248166  (정상)

# arc_hypertan: 실수 정의역은 |x| < 1
# x = 1.5 (정의역 밖) → 복소수
m.arc_hypertan(1.5)   # (0.8047189562170503+1.5707963267948966j)
m.arc_hypertan(0.5)   # 0.5493061443340548  (정상)
```

### 4.5 power(음수, 분수)

```python
import math_library as m
import math

math.pow(-1, 0.5)     # ValueError
m.power(-1, 0.5)      # (6.123233995736766e-17+1j)  ≈ 1j
m.power(-8, 1/3)      # 복소수  (cbrt(-8) = -2.0 와 다름)
```

> 음수의 실수 세제곱근이 필요하다면 `cbrt(-8)` 을 사용하십시오. `cbrt`는 실수 결과 `-2.0`을 반환합니다.

---

## 5. 역쌍곡함수 마이그레이션

`math.asinh/acosh/atanh`는 두 가지 방식으로 마이그레이션할 수 있습니다.

| `math` | mathlib 네이밍 | math 호환 alias |
|---|---|---|
| `math.asinh(x)` | `m.arc_hypersin(x)` | `m.asinh(x)` |
| `math.acosh(x)` | `m.arc_hypercos(x)` | `m.acosh(x)` |
| `math.atanh(x)` | `m.arc_hypertan(x)` | `m.atanh(x)` |

두 방식은 완전히 동일한 구현을 호출합니다. `m.asinh`는 `m.arc_hypersin`의 alias입니다.

```python
import math_library as m

m.asinh(1.0)           # 0.881373587019543
m.arc_hypersin(1.0)    # 0.881373587019543  (동일)

m.acosh(2.0)           # 1.3169578969248166
m.arc_hypercos(2.0)    # 1.3169578969248166  (동일)

m.atanh(0.5)           # 0.5493061443340548
m.arc_hypertan(0.5)    # 0.5493061443340548  (동일)
```

정의역 밖에서의 차이점: `math.acosh(0.5)`는 `ValueError`, `m.acosh(0.5)`는 복소수 반환.

---

## 6. 상수 마이그레이션

mathlib의 상수는 값이 아닌 **함수 호출** 형태입니다.

```python
import math
import math_library as m

# 상수 접근 방식
math.pi          # 3.141592653589793  (속성)
m.pi()           # 3.141592653589793  (함수 호출)

math.e           # 2.718281828459045
m.e()            # 2.718281828459045

# math에 없는 상수
m.epsilon()      # 2.220446049250313e-16  (머신 엡실론)
```

변수에 저장해두면 호출 오버헤드를 제거할 수 있습니다.

```python
import math_library as m

PI = m.pi()
E = m.e()

# 이후 PI, E를 상수처럼 사용
circumference = 2 * PI * r
```

---

## 7. 완전 마이그레이션 예시

기존 `math` 기반 코드를 mathlib로 전환하는 실제 예시입니다.

### Before (math 사용)

```python
import math

def cartesian_to_polar(x, y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return r, math.degrees(theta)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def safe_log(x):
    if x <= 0:
        return float('-inf')
    return math.log(x)

def normalize_angle(rad):
    return math.fmod(rad, 2 * math.pi)
```

### After (math_library 사용)

```python
import math_library as m

_PI = m.pi()  # 상수는 한 번만 호출

def cartesian_to_polar(x, y):
    r = m.hypot(x, y)                    # hypot 직접 사용
    theta = m.atan2(y, x)
    return r, theta * 180 / _PI          # degrees 산술 표현

def sigmoid(x):
    return 1 / (1 + m.exp(-x))

def safe_log(x):
    # 음수 입력 시 자동 복소수 반환 — 필요에 따라 분기
    if x == 0:
        return float('-inf')
    return m.ln(x)

def normalize_angle(rad):
    return m.fmod(rad, 2 * _PI)
```

마이그레이션 포인트 요약:
- `math.sqrt(x**2 + y**2)` → `m.hypot(x, y)` (오버플로우 없음)
- `math.log(x)` → `m.ln(x)` (이름 변경)
- `math.degrees(x)` → `x * 180 / m.pi()` (산술 표현)
- `math.pi` → `m.pi()` (함수 호출)
- `math.fmod` → `m.fmod` (이름 동일)
