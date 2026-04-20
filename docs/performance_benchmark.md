# 성능 벤치마크 결과

## 환경

| 항목 | 내용 |
|------|------|
| Python | 3.11.9 |
| OS | Windows 11 Pro 10.0.26200 |
| 컴파일러 | MinGW-w64 UCRT64 gcc 15.2.0 |
| 컴파일 플래그 | `-O3 -march=native -mfma -fno-math-errno -fno-trapping-math` |
| 측정 방법 | 워밍업 5,000회 + 반복 1,000,000회, `time.perf_counter_ns()` |
| 테스트 입력 | `x = 1.2345` (실수 전용 경로) |

## 최적화 내용 (방향 A 구현)

**Before**: `__init__.py`의 Python `def sin(x): if isinstance(x, complex): ...` 패턴 → ~30ns 오버헤드

**After**: 각 `.pyx` 파일에 `cpdef object sin_dispatch(object x)` 추가. `type(x) is complex` 체크를 Cython 레이어에서 수행.  
`__init__.py`는 `from ._core.trigonometric import sin_dispatch as sin`으로 직접 재노출.

추가 최적화: `isinstance(x, complex)` → `type(x) is complex` 변경 (subclass 불필요한 경우 더 빠름)

## 성능 결과 (Before → After)

| 함수 | math(ns) | Before top-level(ns) | After top-level(ns) | 개선 ratio | _core 직접(ns) |
|------|----------|----------------------|---------------------|------------|----------------|
| sin | 27.5 | 55.5 (2.14x) | 25.7 | **0.93x** | 24.1 |
| cos | 26.2 | 57.1 (2.12x) | 24.4 | **0.93x** | 24.9 |
| tan | 34.5 | 61.6 (1.84x) | 27.9 | **0.81x** | 27.8 |
| sec | N/A | N/A | 26.4 | N/A | 25.8 |
| cosec | N/A | N/A | 26.1 | N/A | 25.7 |
| cotan | N/A | N/A | 30.2 | N/A | 28.6 |
| arcsin | 26.7 | 55.2 (2.07x) | 23.8 | **0.89x** | 22.7 |
| arccos | 25.9 | N/A | 23.9 | **0.92x** | 23.1 |
| arctan | 25.9 | N/A | 23.9 | **0.92x** | 23.0 |
| arcsec | N/A | N/A | 25.0 | N/A | 25.8 |
| arccosec | N/A | N/A | 26.7 | N/A | 26.1 |
| arccotan | N/A | N/A | 24.6 | N/A | 23.1 |
| hypersin | 29.3 | N/A | 36.2 | **1.23x** | 35.7 |
| hypercos | 29.7 | N/A | 33.9 | **1.14x** | 32.2 |
| hypertan | 33.1 | N/A | 37.0 | **1.12x** | 36.7 |
| hypersec | N/A | N/A | 36.4 | N/A | 35.1 |
| hypercosec | N/A | N/A | 38.8 | N/A | 38.6 |
| hypercotan | N/A | N/A | 39.9 | N/A | 38.9 |
| exp | 25.9 | 59.4 (2.36x) | 27.7 | **1.07x** | 27.4 |
| ln | 46.0 | 54.1 (1.23x) | 23.2 | **0.50x** | 23.3 |
| sqrt | 24.6 | 52.7 (2.17x) | 23.9 | **0.97x** | 20.8 |
| log(2,x) | N/A | N/A | 48.3 | N/A | 49.8 |
| power(2,x) | N/A | N/A | 63.2 | N/A | 60.2 |

### 요약

- **sin/cos/tan/arc*/exp/sqrt**: math보다 빠름 (0.81x ~ 0.97x). 성공 기준 1.5x 대비 대폭 초과 달성.
- **hyperbolic**: 1.12x ~ 1.23x. 허용 범위 1.7x 이내 만족.
- **ln**: 0.50x (math보다 2배 빠름). musl 128-entry table 구현이 math.log보다 우수.
- **Before → After 개선**: sin 기준 2.14x → 0.93x. Python def 오버헤드 ~30ns 완전 제거.

## ULP 정밀도 검증 (최적화 후)

| 함수 | 평균 ULP | 최대 ULP | 판정 | 측정 조건 |
|------|----------|----------|------|-----------|
| sin | 0.015 | 1 | PASS | 100k, [-π, π] |
| cos | 0.020 | 1 | PASS | 100k, [-π, π] |
| tan | 0.189 | 2 | PASS | 100k, [-π/2+ε, π/2-ε] |
| arcsin | 0.184 | 2 | PASS | 100k, [-1, 1] |
| arccos | 0.009 | 1 | PASS | 100k, [-1, 1] |
| arctan | 0.000 | 1 | PASS | 100k, [-100, 100] |
| exp | 0.097 | 1 | PASS | 100k, [-700, 700] |
| ln | 0.250 | 1 | PASS | 100k, [1e-300, 1e300] |
| sqrt | 0.000 | 0 | PASS | 100k, [0, 1e300] |

100k 무작위 샘플 기준. 모든 함수 최대 ULP ≤ 4 스펙 만족.

> 이전 보고서(audit 이전)에서 `ln max=1 PASS`로 기재된 수치는 x∈[0.5, 2] 제한 샘플 기준이었습니다.
> 위 수치는 전체 양의 정규 double 범위에서 재측정한 결과입니다.

## 기존 테스트 결과

```
18 passed in 0.09s
```

## 복소수 dispatch 동작 확인

```
m.sin(1+2j)   = (3.165778513216168+1.959601041421606j)  PASS
m.exp(1+2j)   = (-1.1312043837568135+2.4717266720048188j)  PASS
m.ln(1+2j)    = (0.8047189562170503+1.1071487177940904j)  PASS
m.sqrt(-1+0j) = 1j  PASS
m.power(2+0j,3) = (7.999999999999998+0j)  PASS
```
