# mathlib

`mathlib`는 외부 수학 라이브러리(`math`, `cmath` 등)를 호출하지 않고, 수학 함수와 미분 연산을 직접 구현하는 **파이썬 수학 라이브러리**입니다.

## 1. 프로젝트 개요

- 라이브러리명: `mathlib`
- 패키지 경로: `src/math_library`
- 핵심 컨셉: 급수/수치해석 기반으로 직접 구현한 수학 함수 제공
- 기본 수 체계: `real`
- 복소수 연산 사용 방식: `number_system="complex"`를 명시해서 호출

## 2. 설치 및 실행

## 요구사항

- Python 3.10+

## 로컬 설치

```bash
pip install -e .
```

## 테스트 실행 (터미널 출력형 스크립트)

```bash
python tests/test_comprehensive_types.py
python tests/test_differentiation.py
```

## 3. 기본 사용법

```python
from math_library.constant import pi, e, epsilon
from math_library.exponential_function import power
from math_library.logarithmic_function import log
from math_library.trigonometric_function import sin
from math_library.differentiation import Differentiation

print(pi())
print(power(2, 10))
print(log(2, 8))
print(sin(pi() / 2))

diff = Differentiation()
print(diff.single_variable(lambda x: x*x, 3.0))
```

## 4. 수 체계(`number_system`) 규칙

- `number_system="real"` (기본값)
- 입력은 `int`, `float`만 허용
- 결과는 실수(`float`) 형태로 반환
- 실수 영역에서 정의되지 않으면 `ValueError` 또는 `ZeroDivisionError`

- `number_system="complex"`
- 복소수 입력 허용
- 복소 결과가 필요하면 `complex`로 반환
- 허수부가 수치 오차 범위(`tol`)로 매우 작으면 실수로 정규화될 수 있음

## 5. 상수 API

| 함수 | 호출 형식 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `epsilon` | `epsilon()` | 없음 | `float` | 머신 엡실론 |
| `pi` | `pi()` | 없음 | `float` | 원주율 |
| `e` | `e()` | 없음 | `float` | 자연상수 |

## 6. 기본 함수 API

### 6.1 지수/로그

| 함수 | 호출 형식 | 주요 입력 | 출력 | 비고 |
|---|---|---|---|---|
| `power` | `power(base, exponent, tol=None, max_terms=100, number_system='real')` | `base`, `exponent` | `float | complex` | `sqrt(x)`는 `power(x, 0.5)`로 사용 |
| `log` | `log(base, x, tol=None, max_terms=100, number_system='real')` | `base`, `x` | `float | complex` | 실수 모드에서 정의역 검사 |

### 6.2 삼각함수

| 함수 | 호출 형식 | 주요 입력 | 출력 | 비고 |
|---|---|---|---|---|
| `sin` | `sin(x, unit='rad', tol=None, max_terms=100, number_system='real')` | `x`, `unit` | `float | complex` | `unit`: `rad` 또는 `deg` |
| `cos` | `cos(x, unit='rad', tol=None, max_terms=100, number_system='real')` | `x`, `unit` | `float | complex` |  |
| `tan` | `tan(x, unit='rad', tol=None, max_terms=100, number_system='real')` | `x`, `unit` | `float | complex` |  |
| `cosec` | `cosec(x, unit='rad', tol=None, max_terms=100, number_system='real')` | `x`, `unit` | `float | complex` | 0 분모 지점 예외 가능 |
| `sec` | `sec(x, unit='rad', tol=None, max_terms=100, number_system='real')` | `x`, `unit` | `float | complex` | 0 분모 지점 예외 가능 |
| `cotan` | `cotan(x, unit='rad', tol=None, max_terms=100, number_system='real')` | `x`, `unit` | `float | complex` | 0 분모 지점 예외 가능 |

### 6.3 역삼각함수

| 함수 | 호출 형식 | 주요 입력 | 출력 | 비고 |
|---|---|---|---|---|
| `arcsin` | `arcsin(x, tol=None, max_terms=100, number_system='real')` | `x` | `float | complex` | 실수 모드에서 `x in [-1, 1]` |
| `arccos` | `arccos(x, tol=None, max_terms=100, number_system='real')` | `x` | `float | complex` | 실수 모드에서 `x in [-1, 1]` |
| `arctan` | `arctan(x, tol=None, max_terms=100, number_system='real')` | `x` | `float | complex` |  |
| `arcsec` | `arcsec(x, tol=None, max_terms=100, number_system='real')` | `x` | `float | complex` | 실수 모드에서 `|x| >= 1`, `x=0` 예외 |
| `arccosec` | `arccosec(x, tol=None, max_terms=100, number_system='real')` | `x` | `float | complex` | 실수 모드에서 `|x| >= 1`, `x=0` 예외 |
| `arccotan` | `arccotan(x, tol=None, max_terms=100, number_system='real')` | `x` | `float | complex` |  |

### 6.4 쌍곡함수

| 함수 | 호출 형식 | 주요 입력 | 출력 | 비고 |
|---|---|---|---|---|
| `hypersin` | `hypersin(x, tol=None, max_terms=100, number_system='real')` | `x` | `float | complex` |  |
| `hypercos` | `hypercos(x, tol=None, max_terms=100, number_system='real')` | `x` | `float | complex` |  |
| `hypertan` | `hypertan(x, tol=None, max_terms=100, number_system='real')` | `x` | `float | complex` |  |
| `hypersec` | `hypersec(x, tol=None, max_terms=100, number_system='real')` | `x` | `float | complex` | 0 분모 지점 예외 가능 |
| `hypercosec` | `hypercosec(x, tol=None, max_terms=100, number_system='real')` | `x` | `float | complex` | 0 분모 지점 예외 가능 |
| `hypercotan` | `hypercotan(x, tol=None, max_terms=100, number_system='real')` | `x` | `float | complex` | 0 분모 지점 예외 가능 |

## 7. Differentiation 클래스

`Differentiation`은 미분의 정의(차분몫 + `h -> 0` 수렴)를 사용해 미분 연산을 수행합니다.

## 생성자

```python
from math_library.differentiation import Differentiation

diff = Differentiation(
    tol=1e-8,
    initial_h=1e-3,
    max_iter=20,
    number_system="real",  # "real" | "complex"
)
```

- `tol`: 수렴 판정/정규화 허용 오차
- `initial_h`: 초기 차분 스텝
- `max_iter`: 수렴 반복 횟수 상한
- `number_system`: 기본 수 체계

## 메서드 목록 (역할 / 입력 / 출력)

| 메서드 | 역할 | 주요 입력 | 출력 |
|---|---|---|---|
| `difference_quotient` | 1차 차분몫 계산 | `function`, `x`, `h`, `method` | `Number` |
| `single_variable` | 일변수 도함수 | `function`, `x` | `Number` |
| `nth_derivative` | n차 도함수 | `function`, `x`, `order` | `Number` |
| `left_derivative` | 좌미분 | `function`, `x` | `Number` |
| `right_derivative` | 우미분 | `function`, `x` | `Number` |
| `partial_derivative` | 편미분 | `function`, `point`, `variable_index` | `Number` |
| `mixed_partial` | 혼합편미분(다중지수) | `function`, `point`, `order_multi_index` | `Number` |
| `gradient` | 그래디언트 | `function`, `point` | `List[Number]` |
| `directional_derivative` | 방향도함수 | `function`, `point`, `direction` | `Number` |
| `higher_order_directional_derivative` | 고계 방향도함수 | `function`, `point`, `direction`, `order` | `Number` |
| `jacobian` | 야코비안 행렬 | `functions`, `point` | `List[List[Number]]` |
| `hessian` | 헤시안 행렬 | `function`, `point` | `List[List[Number]]` |
| `hessian_vector_product` | 헤시안-벡터 곱 | `function`, `point`, `vector` | `List[Number]` |
| `laplacian` | 라플라시안 | `function`, `point` | `Number` |
| `vector_laplacian` | 벡터 라플라시안 | `vector_field`, `point` | `List[Number]` |
| `total_differential` | 전미분 계수 반환 | `function`, `point` | `dict` |
| `total_derivative` | 합성함수 전체 미분 | `outer_function`, `inner_functions`, `point` | `dict` |
| `parametric_derivative` | 매개변수 미분(`dy/dx`, 고계 포함) | `coordinate_functions`, `parameter`, `order` | `Number` |
| `implicit_derivative` | 음함수 미분(2변수) | `relation`, `x`, `y` | `Number` |
| `implicit_partial` | 음함수 편미분(다변수) | `relation`, `point`, `dependent_index`, `independent_index` | `Number` |
| `divergence` | 발산 | `vector_field`, `point` | `Number` |
| `curl` | 회전(3D) | `vector_field`, `point` | `List[Number]` |
| `wirtinger_derivatives` | 비르팅거 미분(`∂/∂z`, `∂/∂z̄`) | `function`, `z` | `dict` |
| `generalized_derivative` | 일반화 미분(미분가능성/클라크 구간) | `function`, `x` | `dict` |
| `subgradient` | 부분그래디언트(또는 구간) | `function`, `x` | `List[Number]` 또는 `Tuple[Number, Number]` |
| `gateaux_derivative` | Gateaux 미분(방향) | `function`, `point`, `direction` | `Number` 또는 `List[Number]` |
| `frechet_derivative` | Frechet 미분(선형 근사 행렬) | `function`, `point` | `dict` |

## Differentiation 예시

### 일변수 미분

```python
diff = Differentiation()
value = diff.single_variable(lambda x: x**3, 2.0)
print(value)  # 약 12
```

### 그래디언트

```python
diff = Differentiation()
value = diff.gradient(lambda x, y: x*x + y*y, [1.0, 2.0])
print(value)  # 약 [2, 4]
```

### 음함수 미분

```python
diff = Differentiation()
relation = lambda x, y: x*x + y*y - 25
print(diff.implicit_derivative(relation, 3.0, 4.0))  # 약 -0.75
```

### 비르팅거 미분

```python
diff = Differentiation(number_system="complex")
info = diff.wirtinger_derivatives(lambda z: z*z, 1 + 2j)
print(info["df_dz"])            # 약 2+4j
print(info["df_dz_conjugate"])  # 약 0
```

## 8. 에러 처리 원칙

- 숫자 타입이 아닌 입력(`str`, `list`, `dict`, `None`, `bool`)은 `TypeError`
- 수학적으로 정의되지 않는 입력은 `ValueError` 또는 `ZeroDivisionError`
- 미분 수렴 실패 시 `RuntimeError`

## 9. 패키지 구조

```text
src/math_library/
  constant/
  exponential_function/
  logarithmic_function/
  trigonometric_function/
  inverse_trigonometric_function/
  hyperbolic_function/
  differentiation/
```

## 10. 참고

- `tests/`에는 카테고리별 테스트 스크립트가 있으며 실행 시 터미널에 PASS/FAIL이 출력됩니다.
- 복소수 연산이 필요한 경우 반드시 `number_system="complex"`를 명시해 주세요.
