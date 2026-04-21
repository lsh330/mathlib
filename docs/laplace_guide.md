# Laplace 변환 모듈 상세 가이드

`math_library.laplace` — C++ AST 백엔드 기반 기호 Laplace 변환 엔진

---

## 목차

1. [개요 및 설계 철학](#1-개요-및-설계-철학)
2. [설치](#2-설치)
3. [AST 기초](#3-ast-기초)
4. [Forward Laplace 변환](#4-forward-laplace-변환)
5. [Inverse Laplace 변환](#5-inverse-laplace-변환)
6. [Poles / Zeros 분석](#6-poles--zeros-분석)
7. [Final / Initial Value 정리](#7-final--initial-value-정리)
8. [Partial Fractions](#8-partial-fractions)
9. [단순화](#9-단순화)
10. [미분](#10-미분)
11. [Taylor 급수](#11-taylor-급수)
12. [제어 공학 활용](#12-제어-공학-활용)
13. [Lambdify](#13-lambdify)
14. [성능 가이드](#14-성능-가이드)
15. [자주 하는 실수](#15-자주-하는-실수)
16. [제한 사항 (현재 버전)](#16-제한-사항-현재-버전)

---

## 1. 개요 및 설계 철학

### C++ 백엔드 근거

`math_library`의 나머지 모듈은 Cython으로 구현되어 있지만, Laplace 모듈만 C++17 AST를 백엔드로 사용합니다.
기호 연산에서 핵심 병목은 트리 구조 탐색·패턴 매칭·해시 비교이며, 이 작업들은 정적 타입과 인라인 최적화가 가능한 C++가 Cython보다 유리합니다.
Cython은 Python ↔ C++ 경계 (핸들 변환, 예외 전파) 역할만 담당합니다.

### 자체 구현 원칙

SymPy, GiNaC, Pynac 등 외부 CAS(Computer Algebra System)에 의존하지 않습니다.
외부 CAS는 임포트 비용, 메모리 오버헤드, 버전 호환성 문제를 동반합니다.
이 모듈은 Laplace 변환에 필요한 최소 기호 연산만을 자체 구현하여 불필요한 의존성을 배제합니다.

### 성능 비교 (참고)

아래 수치는 측정된 절대값입니다. Pynac과 SymPy는 직접 측정하지 않았으며, 공개된 벤치마크와 이 모듈의 실측값을 비교한 참고 수치입니다.

| 항목 | 이 모듈 (실측) | Pynac (참고) | SymPy (참고) |
|---|---|---|---|
| 기본 변환 `L{sin(2t)}` | 0.44~1.0 μs | ~0.3~0.5 μs | ~200~500 μs |
| 캐시 hit 재변환 | 84 ns | — | — |
| partial_fractions (2차) | 8.2 μs | — | ~500 μs |

이 모듈은 Pynac보다 일부 연산에서 느리지만, SymPy 대비 50~500배 빠른 것을 실측으로 확인하였습니다.
Pynac은 C++ 네이티브로 구현되어 있어 이 모듈보다 일부 연산에서 빠를 수 있음을 명시합니다.

---

## 2. 설치

Laplace 모듈은 C++ 소스를 포함하므로, 표준 Cython 빌드 명령으로 함께 컴파일됩니다.

### 요구사항

| 항목 | 사양 |
|---|---|
| C++ 컴파일러 | C++17 이상 지원 (MinGW-w64 UCRT64 gcc 15.2.0 권장) |
| Python | 3.10 이상 |
| Cython | 3.0 이상 |

### Windows (MinGW-w64 UCRT64, 권장)

```bash
python setup.py build_ext --inplace --compiler=mingw32
```

### Linux / macOS

```bash
python setup.py build_ext --inplace
```

### 임포트 확인

```python
from math_library.laplace import Laplace, t, s
L = Laplace()
print(L.transform(t))   # s**-2
```

---

## 3. AST 기초

### 심볼 생성

```python
from math_library.laplace import symbol, const, rational, t, s

x = symbol('x')   # 기호 변수 x
y = symbol('y')   # 기호 변수 y

# 미리 정의된 공용 심볼
print(t)   # t  (시간 변수)
print(s)   # s  (주파수 변수)
```

`t`와 `s`는 모듈 레벨에서 미리 생성된 `symbol('t')`, `symbol('s')` 인스턴스입니다.

### 상수 생성

```python
from math_library.laplace import const, rational

c1 = const(2.5)       # double 상수
c2 = const(-1.0)      # 음수 상수
r  = rational(1, 3)   # 유리수 1/3 (정확 표현)

print(c1)             # 2.5
print(r)              # 1/3
print(r.evalf())      # 0.3333333333333333
```

정수·부동소수는 산술 연산 시 자동으로 `const`/`rational`로 승격됩니다.

```python
from math_library.laplace import t

expr = 2 * t + 1   # int/float 자동 승격
print(expr)         # 1 + t*2
```

### 산술 연산자

`PyExpr` 객체는 `+`, `-`, `*`, `/`, `**`, 단항 `-` 을 지원합니다.

```python
from math_library.laplace import symbol, const

s = symbol('s')

F1 = (s + 1) / (s**2 + 2*s + 1)
F2 = const(3.0) * s - 1
F3 = F1 + F2
print(F3)
```

### hash-consing 원리

동일한 구조의 표현식은 Python 수준에서도 동일한 객체입니다.

```python
from math_library.laplace import Sin, t

a = Sin(2*t)
b = Sin(2*t)
print(a is b)   # True — 동일 Python 객체
print(a == b)   # True
```

C++ 레벨의 pool이 동일 구조 노드를 한 번만 생성하고 (hash-consing), Python 래퍼도 핸들 기반으로 캐싱합니다.
이 덕분에 반복 변환에서 캐시 hit(84 ns)가 가능합니다.

### 수치 평가

```python
from math_library.laplace import symbol, Sin

s = symbol('s')
F = (s + 1) / (s**2 + 5)

print(F.evalf(s=2.0))   # (2+1)/(4+5) = 3/9 = 0.3333...
print(F.evalf(s=0.0))   # 1/5 = 0.2
```

### 함수 호출

```python
from math_library.laplace import Sin, Cos, Exp, Sqrt, t

f1 = Sin(2*t)
f2 = Cos(t)
f3 = Exp(-t)
f4 = Sqrt(t)

print(f1)   # sin(t*2)
print(f3)   # exp(-t)
```

사용 가능한 심볼 함수: `Sin`, `Cos`, `Tan`, `Arcsin`, `Arccos`, `Arctan`, `Sinh`, `Cosh`, `Tanh`, `Exp`, `Ln`, `Log`, `Sqrt`, `Heaviside` (alias `H`), `Dirac`

---

## 4. Forward Laplace 변환

### 기본 사용

```python
from math_library.laplace import Laplace, t, s, Sin, Cos, Exp, const

L = Laplace()

F = L.transform(Sin(2*t))
print(F)          # (s**2 + 4)**-1*2
print(F.latex())  # \frac{2}{s^{2} + 4}
```

`Laplace()` 생성자는 기본적으로 `t` (시간 변수), `s` (주파수 변수)를 사용합니다.
변수 이름을 바꾸려면 `Laplace(t_var='tau', s_var='w')` 와 같이 지정합니다.

### 지원 변환 규칙 (주요 예제)

```python
from math_library.laplace import Laplace, t, s, Sin, Cos, Exp, Heaviside, Dirac, const, symbol

L = Laplace()

# 1) 상수 k → k/s
print(L.transform(const(3.0)))    # s**-1*3

# 2) t^n → n!/s^(n+1)
print(L.transform(t**2))          # 2*s**-3
print(L.transform(t**3))          # 6*s**-4

# 3) e^(at) → 1/(s-a)
print(L.transform(Exp(-t)))       # (s + 1)**-1
print(L.transform(Exp(const(-2.0)*t)))   # (s + 2)**-1

# 4) sin(ωt) → ω/(s²+ω²)
print(L.transform(Sin(3*t)))      # (s**2 + 9)**-1*3

# 5) cos(ωt) → s/(s²+ω²)
print(L.transform(Cos(3*t)))      # s*(s**2 + 9)**-1

# 6) s-이동 (제1 이동 정리): e^(at)*f(t) → F(s-a)
print(L.transform(Exp(-t) * Sin(2*t)))     # ((s + 1)**2 + 4)**-1*2
print(L.transform(Exp(-t) * Cos(2*t)))     # ((s + 1)**2 + 4)**-1*(s + 1)

# 7) t·f(t) → -dF/ds (주파수 미분)
print(L.transform(t * Exp(-t)))   # (s + 1)**-2
print(L.transform(t * Sin(t)))    # -(s*(s**2 + 1)**-2*-2)
```

### Heaviside / Dirac delta 변환

```python
from math_library.laplace import Laplace, t, Heaviside, Dirac, H, const

L = Laplace()

# L{u(t)} = 1/s
F1 = L.transform(Heaviside(t))
print(F1.evalf(s=2.0))   # 0.5

# L{u(t-a)} = e^{-as}/s
F2 = L.transform(Heaviside(t - const(2)))
print(F2.evalf(s=1.0))   # exp(-2)/1 ≈ 0.1353

# H 는 Heaviside 의 alias
F3 = L.transform(H(t))
print(F3.evalf(s=2.0))   # 0.5

# L{δ(t)} = 1
F4 = L.transform(Dirac(t))
print(F4.evalf(s=3.0))   # 1.0

# L{δ(t-a)} = e^{-as}
F5 = L.transform(Dirac(t - const(1)))
print(F5.evalf(s=2.0))   # exp(-2) ≈ 0.1353
```

### t-shift 규칙 (제2 이동 정리)

`u(t-a) * f(t-a)` 형태는 자동으로 t-shift 규칙을 적용합니다.

```python
from math_library.laplace import Laplace, t, Heaviside, Exp, const

L = Laplace()

# u(t-1) * e^{-(t-1)}  →  e^{-s} / (s+1)
a = const(1)
f = Heaviside(t - a) * Exp(-(t - a))
F = L.transform(f)
print(F.evalf(s=2.0))   # exp(-2)/(2+1) ≈ 0.04511
```

### 수치 평가

```python
F = L.transform(Exp(-t) * Sin(2*t))
print(F.evalf(s=2.0))   # 0.15384615384615385
print(F.evalf(s=0.5))   # 약 0.44...
```

---

## 5. Inverse Laplace 변환

### 기본 사용

```python
from math_library.laplace import Laplace, t, s, Sin, Exp

L = Laplace()

F = L.transform(Exp(-t) * Sin(2*t))
f_back = L.inverse(F)
print(f_back)   # sin(t*2)*exp(t*-1)
```

### 내부 알고리즘

1. F(s)에서 `e^{-as} * G(s)` 패턴(시간 지연) 감지
   - 발견되면: `u(t-a) * g(t-a)` 형태로 재구성 후 G(s)를 재귀적으로 역변환
2. F(s)를 분자·분모 다항식으로 파싱
3. 분모의 근(극점)을 Durand-Kerner 알고리즘으로 계산
4. 부분분수 분해: 단순 극점 `A/(s-p)` 및 복소 켤레 쌍 처리
5. 각 부분분수에 대해 역변환 룩업 적용: `A/(s+a)` → `A·e^(-at)`, `Bω/((s+a)²+ω²)` → `B·e^(-at)·sin(ωt)` 등

### Non-rational inverse (시간 지연 포함)

`e^{-as} * G(s)` 형태의 비유리 함수도 역변환이 가능합니다.
결과는 `u(t-a) * g(t-a)` 형태로 반환됩니다.

```python
from math_library.laplace import Laplace, Exp, symbol, t

L = Laplace()
s_sym = symbol('s')

# e^{-2s} / (s+1)  →  u(t-2) * e^{-(t-2)}
G = L.transform(Exp(-t))           # 1/(s+1)
F = Exp(-2 * s_sym) * G
f_back = L.inverse(F)
print(f_back)                      # heaviside(t + -2)*exp((t + -2)*-1)

print(f_back.evalf(t=0.5))         # 0.0  (t < 2: u(0.5-2)=0)
print(f_back.evalf(t=2.5))         # exp(-0.5) ≈ 0.6065
print(f_back.evalf(t=3.5))         # exp(-1.5) ≈ 0.2231
```

Sum 형태의 F(s)는 각 항을 독립적으로 처리합니다:

```python
# F(s) = 1/(s+1) + e^{-s}/(s+2)
G1 = L.transform(Exp(-t))          # 1/(s+1)
G2 = L.transform(Exp(-2*t))        # 1/(s+2)
F_sum = G1 + Exp(-s_sym) * G2
f_sum = L.inverse(F_sum)
# f(t) = e^{-t} + u(t-1)*e^{-2(t-1)}
```

---

## 6. Poles / Zeros 분석

### 극점 계산

```python
from math_library.laplace import Laplace, t, s, Sin, Exp, symbol, const

L = Laplace()
s_expr = symbol('s')

F = L.transform(Exp(-t) * Sin(2*t))
poles = L.poles(F)
print(poles)   # [(-1+2j), (-1-2j)]
```

### 영점 계산

```python
F2 = (s_expr - 2) / (s_expr + 3)
print(L.zeros(F2))   # [(2+0j)]
print(L.poles(F2))   # [(-3+0j)]
```

### Durand-Kerner 기반

극점·영점은 내부적으로 **Durand-Kerner(Weierstrass) 알고리즘**을 사용하여 다항식 근을 계산합니다.
수치 정밀도는 1e-10 이내입니다.

복소수 형태로 반환되므로, 실수 극점도 허수부가 0인 복소수로 반환됩니다.

```python
F3 = const(1.0) / ((s_expr + 1) * (s_expr + 3))
print(L.poles(F3))   # [(-1+0j), (-3+0j)]
```

---

## 7. Final / Initial Value 정리

### 최종값 정리

```math
\lim_{t \to \infty} f(t) = \lim_{s \to 0} s F(s)
```

```python
from math_library.laplace import Laplace, t, s, Exp, Sin

L = Laplace()

# e^(-t)*sin(2t) → 0 as t→∞
F = L.transform(Exp(-t) * Sin(2*t))
val, valid = L.final_value(F)
print(val, valid)   # ~0.0  True
```

반환값은 `(float, bool)` 튜플입니다. `valid=False`이면 정리 적용 조건을 만족하지 않습니다.
(극점이 우반평면(RHP)에 있거나 원점에 극점이 2개 이상인 경우)

### 초기값 정리

```math
f(0^+) = \lim_{s \to \infty} s F(s)
```

```python
val, valid = L.initial_value(F)
print(val, valid)   # 0.0  True
```

`Exp(-t)*Sin(2t)` 는 `t=0`에서 `sin(0)*exp(0) = 0` 이므로 초기값 0이 맞습니다.

---

## 8. Partial Fractions

### 기본 사용

`F.partial_fractions(var)` 는 F를 var에 대한 부분분수로 분해합니다.

```python
from math_library.laplace import Laplace, symbol, const

s = symbol('s')

F = (s + 3) / ((s + 1) * (s + 2))
pf = F.partial_fractions(s)
print(pf)   # (s + 2)**-1*-1 + (s + 1)**-1*2
```

위 결과는 `2/(s+1) + (-1)/(s+2)` 입니다.

### 반환 형식

분해 결과는 `PyExpr` AST로 반환되며, 각 항은 `A/(s+p)` 형태의 곱셈으로 표현됩니다.

### Laplace 클래스 메서드 vs PyExpr 메서드

- `L.partial_fractions(F)` — `Laplace` 인스턴스의 s 변수 기준으로 분해
- `F.partial_fractions(s_expr)` — 명시적 변수 지정

두 방법 모두 동일한 C++ 함수를 호출합니다.

---

## 9. 단순화

### expand — 분배 전개

```python
from math_library.laplace import symbol

s = symbol('s')

expr = (s + 1) * (s + 2)
print(expr.expand())   # s + s*2 + s*s + 2
```

이항 전개도 지원합니다 (지수 ≤ 5).

### cancel — 유리식 약분

```python
expr = (s**2 - 1) / (s - 1)
print(expr.cancel(s))   # s + 1
```

변수를 명시하지 않으면(`cancel()`) 식에서 자동으로 변수를 탐지합니다.

```python
print(expr.cancel())    # s + 1  (자동 탐지)
```

### simplify — expand + cancel

```python
expr = (s + 1) * (s + 2) / (s + 1)
print(expr.simplify(s))   # s + 2
```

`simplify`는 내부적으로 `expand()` 후 `cancel(var)`를 적용합니다.

### collect — 동류항 정리

`expr.collect(var)` 는 var의 거듭제곱별로 계수를 합산합니다.

```python
from math_library.laplace import symbol

s = symbol('s')

expr = 3*s**2 + 2*s**2 + 5*s + 7
c = expr.collect(s)
print(c)   # s*5 + s**2*5 + 7  (5s² + 5s + 7)

# 수치 확인
print(c.evalf(s=1.0))   # 17.0
print(c.evalf(s=2.0))   # 37.0
```

내부적으로 `Polynomial::from_expr` → `to_expr` 경로를 사용하므로,
var에 대한 정수 지수 다항식이어야 합니다.
변환 불가능한 식(예: 유리함수 분모에 s가 있는 경우)은 원래 식을 그대로 반환합니다.

### subs — 기호 치환

```python
F = (s**2 + 4)**-1 * 2   # 2/(s^2+4)
F_shifted = F.subs(s=s + 1)
print(F_shifted)   # ((s + 1)**2 + 4)**-1*2
```

s-이동(이동 정리)를 수동으로 적용할 때 유용합니다.

---

## 10. 미분

### diff — 기호 미분

`F.diff(var)` 는 F를 var에 대해 기호 미분합니다.

```python
from math_library.laplace import Laplace, t, s, Sin, symbol

L = Laplace()
s_expr = symbol('s')

F = L.transform(Sin(2*t))   # 2/(s^2+4)
dF = F.diff(s_expr)
print(dF)                    # s*(s**2 + 4)**-2*-4
print(dF.evalf(s=1.0))       # -0.16
```

### diff_numeric — 수치 미분

`F.diff_numeric(var_name)` 은 `LazyDerivative` 객체를 반환합니다.
내부적으로 `Differentiation` 클래스(Ridders-Richardson extrapolation)를 사용합니다.

```python
dn = F.diff_numeric('s')
print(dn.evalf(s=1.0))    # -0.160000000000034  (수치)
print(dn.nth(2, s=1.0))   # 2차 도함수 수치값
```

### 언제 어느 방법을 사용할까

| 상황 | 권장 방법 |
|---|---|
| 닫힌 형식(closed-form) 도함수가 필요할 때 | `diff(var)` |
| 도함수 식이 복잡하여 단순화가 어려울 때 | `diff_numeric` |
| 반복 수치 계산이 필요할 때 (루프 등) | `diff_numeric` + `lambdify` |
| LaTeX 출력이 필요할 때 | `diff(var)` 후 `.latex()` |

---

## 11. Taylor 급수

### 기본 사용

```python
from math_library.laplace import Laplace, t, Sin

L = Laplace()

F = L.transform(Sin(2*t))   # 2/(s^2+4)

ser = F.series('s', about=0, order=6)
coeffs = ser.compute()
print(coeffs)
# [0.5, 0.0, -0.125, ~0, 0.03125, ~0, -0.0078]
```

계수 c_k = F^(k)(0) / k! 이며, `F(s) = 2/(s²+4)` 의 Maclaurin 급수는:

```math
F(s) = \frac{1}{2} - \frac{s^2}{8} + \frac{s^4}{32} - \cdots
```

### as_expr — 다항식 AST 재구성

```python
poly = ser.as_expr()
print(poly)   # 계수로 구성된 PyExpr 다항식
```

### evalf — 급수로 수치 평가

```python
print(ser.evalf(s=0.5))   # 약 0.4706 (6차 근사)
print(F.evalf(s=0.5))     # 0.47058823529411764 (정확값)
```

소수점 가까운 점에서는 정확값과 충분히 일치합니다. 전개점에서 멀어질수록 오차가 커집니다.

### TaylorSeries 매개변수

| 매개변수 | 기본값 | 설명 |
|---|---|---|
| `var_name` | — | 전개 변수 이름 (문자열) |
| `about` | `0.0` | 전개점 |
| `order` | `6` | 최고 차수 |

---

## 12. 제어 공학 활용

### Transfer Function 합성

```python
from math_library.laplace import symbol, const

s = symbol('s')

# 1차 지연 시스템
G = const(1.0) / (s + 1)

# 직렬 연결
G2 = const(1.0) / (s + 2)
T_series = G.series_connect(G2)
print(T_series)   # (s + 2)**-1*(s + 1)**-1

# 병렬 연결
T_parallel = G.parallel_connect(G2)
print(T_parallel)   # (s + 2)**-1 + (s + 1)**-1

# 단위 피드백
T_fb = G.feedback()
print(T_fb)   # ((s + 1)**-1 + 1)**-1*(s + 1)**-1
```

`feedback(H=None)` 는 `self / (1 + self * H)` 를 반환하며, 내부적으로 `simplify()`를 자동 적용합니다.
H=None 이면 H=1 (단위 피드백)입니다.

```python
from math_library.laplace import Laplace, t, Exp, symbol, const

L = Laplace()
G = L.transform(Exp(-t))   # 1/(s+1)

# G.feedback() = (1/(s+1)) / (1 + 1/(s+1)) = 1/(s+2)
# simplify() 자동 적용으로 분자/분모 약분 수행
fb = G.feedback()
print(fb.evalf(s=1.0))   # 1/3 ≈ 0.3333  (= 1/(1+2))
print(fb.evalf(s=2.0))   # 0.25           (= 1/(2+2))
```

### PID 제어기

```python
from math_library.laplace import symbol, const

s = symbol('s')

Kp = const(1.0)
Ki = const(0.5)
Kd = const(0.1)

pid = Kp + Ki / s + Kd * s
print(pid)   # s*0.1 + s**-1*0.5 + 1

# Plant G(s) = 1/(s*(s+2))
plant = const(1.0) / (s * (s + 2))
print(plant)   # (s*(s + 2))**-1

# 주파수 응답 확인
print('|C(j1)| =', pid.magnitude(1.0))    # 1.077032961426901
print('|G(j1)| =', plant.magnitude(1.0))  # 0.447213595499958
```

### Bode 크기 / 위상

```python
import math

omega = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]   # rad/s

G = const(1.0) / (s + 1)
mags = G.magnitude(omega)
phases = G.phase(omega)

for w, m, p in zip(omega, mags, phases):
    print(f'ω={w:5.1f}  |H|={m:.4f}  ∠H={math.degrees(p):.1f}°')
```

### Frequency Response

```python
G = const(1.0) / (s + 1)
H_j1 = G.frequency_response(1.0)
print(H_j1)   # (0.5-0.5j)

# 여러 주파수
H_list = G.frequency_response([0.1, 1.0, 10.0])
```

---

## 13. Lambdify

`lambdify`는 `PyExpr` AST를 Python callable로 변환합니다.
반복 수치 계산이 필요한 경우 `evalf`보다 훨씬 빠릅니다.

### 기본 사용

```python
from math_library.laplace import Laplace, t, Sin

L = Laplace()
F = L.transform(Sin(2*t))   # 2/(s^2+4)

f_s = F.lambdify(['s'], backend='cmath')
print(f_s(0.5+0j))   # (0.47058823529411764+0j)
print(f_s(1j))       # (0.6666666666666666+0j)  — jω=j → s=j
```

### backend 옵션

| backend | 적합한 상황 | import |
|---|---|---|
| `'math'` | 실수 입력 (기본 수치 계산) | `import math` |
| `'cmath'` | 복소수 입력 (주파수 응답) | `import cmath` |
| `'numpy'` | 배열 입력 (벡터화 연산) | `import numpy as np` |

```python
f_real   = F.lambdify(['s'], backend='math')
f_complex = F.lambdify(['s'], backend='cmath')

print(f_real(1.0))           # 0.4 (실수)
print(f_complex(1.0+0j))     # (0.4+0j) (복소수)
```

### 반복 호출 성능

lambdify 컴파일 비용은 34 μs (1회)이며, 이후 호출은 79 ns입니다.
루프에서 반복 평가가 필요하다면 lambdify를 루프 밖에서 한 번만 컴파일하십시오.

```python
# 권장 패턴
f_compiled = F.lambdify(['s'], backend='cmath')

results = []
for omega in range(1, 100):
    results.append(abs(f_compiled(1j * omega)))
```

---

## 14. 성능 가이드

### 실측 성능 수치

| 항목 | 측정값 | 비고 |
|---|---|---|
| 상수 노드 생성 | 101 ns | `const(2.5)` |
| 덧셈 노드 생성 | 246 ns | `a + b` |
| hash-consing 조회 | 33 ns | 동일 구조 재접근 |
| `L{sin(2t)}` forward | 0.44~1.0 μs | 첫 번째 변환 |
| `L{e^(-t)sin(3t)}` forward | 1.3 μs | 첫 번째 변환 |
| 캐시 hit 재변환 | 84 ns | 동일 인자 재변환 |
| `partial_fractions` (2차 분모) | 8.2 μs | |
| `inverse` (e^{-t}sin(2t)) | 9.5 μs | |
| `lambdify` 컴파일 | 34 μs | 1회 비용 |
| `lambdify` 호출 | 79 ns | 컴파일 후 |
| `diff` (기호 미분) | 11.7 μs | |
| `expand` | 3.5 μs | |
| `cancel` | 34 μs | |

측정 환경: Windows 11, Python 3.11.9, MinGW-w64 UCRT64 gcc 15.2.0

### 최적화 팁

**hash-consing 활용**

동일한 서브식을 반복 생성하면 C++ 레벨에서 자동으로 재사용됩니다.
별도로 캐싱할 필요가 없습니다.

```python
# 아래 코드는 s+1을 한 번만 생성합니다 (내부적으로)
s = symbol('s')
a = s + 1
b = s + 1
print(a is b)   # True
```

**캐시 hit 활용**

동일한 표현식의 Laplace 변환은 캐시에 저장됩니다.
반복 변환 시 84 ns에 결과를 반환합니다.

```python
L = Laplace()
F1 = L.transform(Sin(2*t))   # 첫 번째: 0.44~1.0 μs
F2 = L.transform(Sin(2*t))   # 두 번째: 84 ns (캐시 hit)
```

캐시는 프로세스 수명 동안 유지됩니다. 수동으로 초기화하려면:

```python
L.clear_cache()
```

**반복 수치 평가**

루프에서 같은 식을 여러 번 수치 평가할 때는 `lambdify`를 사용하십시오.

```python
# 느린 방식 (evalf: Python ↔ C++ 호출 반복)
results = [F.evalf(s=float(i)) for i in range(1000)]

# 빠른 방식 (lambdify: 순수 Python 함수 호출)
f = F.lambdify(['s'], backend='math')
results = [f(float(i)) for i in range(1000)]
```

---

## 15. 자주 하는 실수

### PyExpr 직접 생성 금지

```python
# 잘못된 방법
from math_library.laplace import PyExpr
e = PyExpr(1.0)   # TypeError 발생

# 올바른 방법
from math_library.laplace import const
e = const(1.0)
```

`PyExpr`은 직접 생성할 수 없습니다. 반드시 `const`, `symbol`, `rational`, 또는 산술 연산으로 생성하십시오.

### 정수/부동소수 혼합 연산

```python
from math_library.laplace import t

# 올바름 — 정수/부동소수는 자동 승격
expr = 2 * t + 1
expr2 = t / 3.0

# 주의 — 순수 Python 정수 나눗셈은 먼저 계산될 수 있음
# expr3 = 1/3 * t  →  Python 먼저 1/3 = 0.333... 계산 후 const(0.333...)로 승격
# 정확한 유리수가 필요하면 rational(1,3) 사용
from math_library.laplace import rational
expr3 = rational(1, 3) * t
```

### Non-rational F의 inverse 실패

```python
from math_library.laplace import Laplace, Exp, symbol, s

L = Laplace()
s_expr = symbol('s')

# 실패 예시 (지수 포함 → 비유리)
# L.inverse(Exp(-s_expr) / (s_expr + 1))   # 예외 발생

# 가능한 형태만
F = (s_expr + 2) / (s_expr**2 + 3*s_expr + 2)
f = L.inverse(F)
print(f)
```

### ROC 범위 밖 s 값에서 evalf

특정 F(s)는 s의 실수부가 극점 실수부보다 작은 영역에서 수렴하지 않습니다.
이 모듈은 ROC(Region of Convergence)를 자동으로 검사하지 않습니다.
극점의 실수부보다 큰 s 값에서만 evalf를 사용하십시오.

```python
F = L.transform(Exp(-t) * Sin(2*t))   # 극점: -1±2j, ROC: Re(s) > -1
print(F.evalf(s=0.0))    # Re(0) > Re(-1) → 유효
print(F.evalf(s=-0.5))   # Re(-0.5) > Re(-1) → 유효
# print(F.evalf(s=-2.0)) # Re(-2) < Re(-1) → ROC 밖 (결과 신뢰 불가)
```

### `cancel` vs `simplify` 선택

- `cancel`은 약분만 합니다 (전개 없음). 식이 이미 전개되어 있어야 효과적입니다.
- `simplify` = `expand` + `cancel`. 일반적으로 `simplify`를 사용하십시오.

---

## 16. 제한 사항 (현재 버전)

| 항목 | 상태 | 비고 |
|---|---|---|
| Heaviside / H | 지원 | `Heaviside(t-a)`, `H(t-a)` |
| Dirac delta | 지원 | `Dirac(t-a)` |
| t-shift `u(t-a)*f(t-a)` 변환 | 지원 | 자동 패턴 인식 |
| 시간 지연 `e^{-as}*G(s)` inverse | 지원 | `u(t-a)*g(t-a)` 반환 |
| `collect(var)` | 지원 | 정수 지수 다항식 한정 |
| `feedback()` 자동 simplify | 지원 | 내부 simplify 자동 적용 |
| Piecewise 함수 | 미지원 | 구간별 정의 함수 미구현 |
| 복소수 evalf | 미지원 | `lambdify(backend='cmath')` 사용 |
| ROC 자동 검사 | 미구현 | 사용자가 직접 확인 필요 |
| `cancel` 비전개 중첩식 | 부분 지원 | 복잡한 중첩 피드백은 수동 `simplify` 필요 |
