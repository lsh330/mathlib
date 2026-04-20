# Verification Report — Cython mathlib

> 검증일: 2026-04-20
> 검증자: critical-reviewer (verification-engineer 모드)
> 대상: `C:/Users/User/mathlib` Cython 재구현 baseline
> 최종 판정: **NO-GO (배포 불가)** — Blocker 5건, Major 7건

---

## 요약

- **전체 검증 항목**: 10
- **PASS**: 3 (테스트 프레임워크 실행, 기본 IEEE 상수·Differentiation 1차, 빌드 산출물 gitignore)
- **FAIL (Blocker)**: 5
- **WARNING (Major/Minor)**: 7 Major + 7 Minor

---

## Blocker 이슈 (배포 전 반드시 해결)

### B1. [CRITICAL] argument_reduction.pyx — Payne-Hanek `ih != 0` 부호 플립 누락

**파일**: `src/math_library/_core/argument_reduction.pyx` 210~221행 (prec=1/2 분기)

**무엇이 틀렸나**: `__rem_pio2_large_impl`의 prec=1/2 분기에서 musl 원본

```c
y[0] = ih==0 ? fw : -fw;
y[1] = ih==0 ? fw : -fw;
```

이 생략돼 있음. `ih`가 1 또는 2일 때 `y[0]`, `y[1]`을 음수화해야 하는데 항상 양수로 남아 있음.

**임팩트**:
- `|x| ≥ 2²⁰·π/2 ≈ 1.647e6`에서 sin/cos/tan의 약 25~46% 값이 부호 반전
- 예: `sin(1e10)` → +0.4875 (정답 -0.4875)
- 500 랜덤 샘플 중 rem<0 케이스 230건 중 106건 부호 플립

**수정 방향**:
```c
y[0] = ih==0 ? fw : -fw;
fw = fq[0] - fw;
for (i=1; i<=jz; i++) fw += fq[i];
y[1] = ih==0 ? fw : -fw;
```
case 3 분기도 동일하게 `ih != 0` 처리 필요.

---

### B2. [CRITICAL] logarithmic.pyx — 음수 입력 필터 누락 (부호 없는 비교 오류)

**파일**: `src/math_library/_core/logarithmic.pyx` 65~66행

**무엇이 틀렸나**: `if hx >= 0x7FF00000U`가 unsigned 비교라 음수(sign bit = 1)인 모든 finite 값이 조건을 만족하여 `return x`로 빠져나감.

**임팩트**: `ln(-1.0) = -1.0`, `ln(-10.0) = -10.0` 같은 터무니없는 결과. 모든 음수 유한 입력에서 `ln(x) = x` 반환.

**수정 방향**: 이 체크 이전에 `if hx >> 31: return (x-x)/(x-x)` 삽입 (음수 → NaN)하거나, 비교 시 `hx & 0x7FFFFFFF`를 사용.

---

### B3. [CRITICAL] logarithmic.pyx — subnormal 입력에서 `lx` 재로드 누락

**파일**: `src/math_library/_core/logarithmic.pyx` 60~63행

**무엇이 틀렸나**: subnormal 스케일링 경로(`k -= 54; x *= 2^54; hx = high_word(x)`)에서 `hx`만 갱신하고 `lx`는 스케일 이전 값을 재사용.

**임팩트**: `ln(2.225e-308)` → -708.396451818985 (정답 -708.3964517265479), **813086 ULP** 오차.

**수정 방향**: 63행 뒤에 `lx = low_word(x)` 한 줄 추가.

---

### B4. [CRITICAL] exponential.pyx — 언더플로우 하한 임계값 설계 결함

**파일**: `src/math_library/_core/exponential.pyx` 63~78행

**무엇이 틀렸나**: `EXP_UNDERFLOW = -745.13...`까지만 언더플로우로 처리. -709.78 < x < -745.13 영역에서 정규 경로의 2^k 스케일링이 k < -1023일 때 IEEE 표현 범위를 벗어남.

**임팩트**: `exp(-745.0) = -9.12e+292` (정답 5e-324). |x|∈(709.78, 745.13] 범위에서 완전히 잘못된 결과 (subnormal 결과를 생성해야 함).

**수정 방향**: 네거티브 영역 스케일링 시 k ≤ -1022이면 `ldexp(y, k-54)*2^-54` 또는 musl 원본의 subnormal 분기 포팅.

---

### B5. [CRITICAL] __init__.py — Top-level 복소수 자동 dispatch 누락

**파일**: `src/math_library/__init__.py` 13~19행

**무엇이 틀렸나**: `from math_library import sin; sin(1+2j)` → `TypeError: must be real number, not complex`. `__init__.py`가 Cython primitive(실수 전용 `cpdef double`)를 직접 import.

**임팩트**: **사용자 승인된 핵심 요구사항 위반**. 17개 elementary 함수 모두 복소수 직접 호출 불가. `exp`, `ln`, `sqrt`에는 Python wrapper도 없어 우회 경로조차 없음.

**수정 방향**: `__init__.py`에서 Python wrapper를 import하거나, Cython primitive를 감싸는 얕은 dispatcher 작성(`isinstance(x, complex)` 분기).

---

## Major 이슈 (강하게 권고)

### M1. ln Chebyshev 끝자리 손실 10~43 ULP
- `ln(3.0)`: 10 ULP, `ln(1.5)`: 38 ULP, `ln(0.75)`: 38 ULP, `ln(0.7512...)`: 43 ULP
- 원인: 표준 fdlibm 최종 식과 달리 괄호 구조가 바뀌어 `f`와 `hfsq` 사이 취소 오차 미보상. `k=0` 분기도 별도 처리되지 않음.
- 목표 `평균 ≤1 ULP, 최대 ≤4 ULP` 대비 실측 **최대 43 ULP** — 스펙 위반.

### M2. sinh/cosh — x ≈ 710에서 308 ULP
- `sinh(710)` 308 ULP, 상대오차 ~6e-14
- 원인: `expo2(x) = exp(x - ln2)` 근사가 DBL_MAX 지수 접근 시 누적 오차. musl은 exp(x/2)²로 분해.

### M3. tan(π/2) 부호 반전
- 구현 -1.63e16, math +1.63e16. 경계값이지만 대칭성 기준 부호 상이.

### M4. power(-1.0, ±inf) — IEEE 754-2019 위반
- IEEE: `pow(-1, ±∞) = 1`. 구현은 NaN.
- `power_sqrt.pyx` 112~113행 `if base < 0.0: return NaN`가 inf 지수 체크 전에 걸림.

### M5. power(2.0, 0.5) 1 ULP 오차
- sqrt(2) = 1.4142135623730951 vs power(2, 0.5) = 1.414213562373095
- `exponent == 0.5` 시 `sqrt(base)` 고속 경로 추가 권고.

### M6. zeta(2) 2.4e-8 상대오차
- 해석해 `π²/6 = 1.6449340668482264` vs 구현 `1.6449340268562156`. 3.7e-8 절대 오차.

### M7. Differentiation — 고차 미분 정확도 급락
- `d4(x^4)/dx⁴ @ 1`: 23.91 (정답 24, 0.4% 오차)
- 원인: `nth_derivative`가 1차 Ridders를 재귀 호출하는 구조. 직접 고차 유한차분 스텐실 (central 4차, 6차) 사용 권장.

---

## Minor 이슈

- **m1**: `log(base, x)` 특수값 (예: `log(2, inf)`) 명시 처리 권고
- **m2**: pytest discovery 0개 — `def test_*` 래퍼 부재 (CI 통합 시 필요)
- **m3**: pyproject.toml dist name `mathlib` vs import name `math_library` 불일치 (혼란 유발)
- **m4**: 레포 루트의 `tmp_bench.py`, `tmp_expm1_sim.py`, `tmp_ulp_check.py` 디버그 산출물 잔존
- **m5**: `setup.py` MinGW 감지 로직 (`"mingw" in arg.lower()`) 취약
- **m6**: `__rem_pio2_large_impl` 리턴값 `n & 3` vs musl `n & 7` (prec=3 확장 시 버그)
- **m7**: `argument_reduction.pyx` L339: `tx[2] = <double>(<int>zz)` int 캐스팅으로 하위 비트 손실

---

## 최종 판정 및 다음 단계

**현재 상태: 배포 불가 (NO-GO)**

Blocker B1–B5 중 **B1과 B2는 요구사항·IEEE 준수 양쪽에서 수용 불가**하며 전체 라이브러리 사용성을 파괴함. B3–B4는 희귀 입력 영역이지만 수치적 대참사 유발. B5는 명시적 사용자 승인 요구사항 위반.

**권고 우선순위**:
1. **sim-engineer 재투입** — Blocker 전부 수정
2. 수정 후 재검증: ULP 벤치 전면 재실행, Payne-Hanek 대규모 스윕, IEEE 754 특수값 자동 비교
3. Major(M1–M7)도 가능하면 같이 해결. 특히 M1(ln ULP)은 기본 스펙 위반
4. **perf-optimizer 단계로 넘어가지 말 것** — 정확성 깨진 상태에서 속도 최적화는 위험
5. 장기: Minor(m2 pytest, m3 dist name)도 문서화/CI 정비 시 같이

---

## 관련 파일 경로

- `src/math_library/_core/argument_reduction.pyx` (B1, m6, m7)
- `src/math_library/_core/logarithmic.pyx` (B2, B3, M1)
- `src/math_library/_core/exponential.pyx` (B4)
- `src/math_library/_core/power_sqrt.pyx` (M4, M5)
- `src/math_library/_core/hyperbolic.pyx` (M2)
- `src/math_library/_core/trigonometric.pyx` (M3)
- `src/math_library/__init__.py` (B5)
- `src/math_library/zeta_function/zeta.pyx` (M6)
- `src/math_library/differentiation/differentiation.pyx` (M7)
- `tests/` 18개 파일 (m2)
- `pyproject.toml` (m3)
- `tmp_bench.py`, `tmp_expm1_sim.py`, `tmp_ulp_check.py` (m4)
