# Final Audit Report — mathlib Release Audit (역사적 기록)

> ⚠️ **이 감사 보고서는 Phase 1 이전(2026-04-20) 시점의 스냅샷입니다.**
> 당시 발견된 Critical 5건은 모두 해결되었으며, 현재 상태는 `phase_3_completion.md`를 참조하십시오.
>
> 감사일: 2026-04-20
> 감사자: critical-reviewer (최종 감사 모드)
> 당시 판정: **NO-GO** (Critical 5건 — 전부 이후 Phase에서 해결됨)

---

## 최종 판정: NO-GO

이 상태로 `github.com/lsh330/mathlib`에 push하면 사용자가 README를 복붙하는 첫 1분 안에 3가지 불일치를 목격하고, ULP 정밀도 주장 중 **ln 항목은 실측과 257배 벗어난 허위 주장**. perf-optimizer/doc-writer 단계에서 M1(ln 정확도)이 완전히 수정되지 않은 상태로 "수정 완료" 보고된 상태.

---

## Critical 이슈 (배포 전 반드시 해결)

### C1. [CRITICAL] ln 정확도 — 스펙 위반, M1 미완전 수정

- **실측**: 평균 0.26 ULP, **최대 257 ULP** (sim-engineer 1차 수정 후에도 여전히 위반)
  - `ln(0.9996078892174366)`: **1023 ULP**
  - `ln(0.9958772610468909)`: 83 ULP
  - `ln(1.0)`: `1.5687050183149642e-17` (정답 정확히 0.0)
  - `ln(2.0)`: 1 ULP (`0.6931471805599454` vs 정답 `0.6931471805599453`)
- **원인**: `_musl_log()`의 정수 `k=0` 특수 처리 누락, `hi/lo` 2중 보정 구조가 musl 원본과 다름
- **위치**: `src/math_library/_core/logarithmic.pyx` 172~206행
- **임팩트**: README §6 "ULP 최대 1 ULP PASS"는 **허위**
- **수정 방향**: musl 원본대로 `k==0 && r==0 → return 0.0` 특수 처리, `hi = (kd*Ln2hi + logc) + r` 단일 가산으로 환원

### C2. [CRITICAL] README 빠른 시작 예시 3개 주석값 불일치

- **라인 120**: `print(m.exp(1.0))  # 2.718281828459045` → 실제 `2.7182818284590455`
- **라인 122**: `print(m.ln(m.e()))  # 1.0` → 실제 `0.9999999999999999`
- **라인 131**: `print(m.gamma(5))   # 24.0 (4! = 24)` → 실제 `23.999999999999996`
- **임팩트**: README "5분 빠른 시작" 8줄 중 3줄이 문서와 불일치. 문서 신뢰도 근본적 훼손
- **수정 방향**: 실제 출력값으로 주석 갱신, 또는 `math.isclose()` 기반 검증으로 변경

### C3. [CRITICAL] README "알려진 이슈"가 이미 수정된 버그를 "작업 중"으로 표기

README.md 456~464행의 5개 불릿:
- "Payne-Hanek 부호 처리 … 수정 작업 중" → **B1 이미 수정 완료** (sin(1e10) 0 ULP 일치)
- "ln 정확도 … 스펙 초과하는 경우 수정 중" → **반대**, 실제론 수정되지 않았으나 README는 PASS 주장
- "exp 언더플로우 … 개선 중" → **B4 이미 수정 완료** (exp(-745) = 5e-324)
- "pow(-1, ±∞) … NaN 반환" → **실제로는 1.0 반환하도록 수정 완료**
- "복소수 dispatch … 일부 미지원" → **21개 primitive 전부 복소수 dispatch 동작 확인**

**수정 방향**: B1/B4/M4/M5 항목 제거. ln 스펙 위반은 명시적으로 "⚠️ ln 일부 구간 최대 1000 ULP 초과" 또는 구현 수정.

### C4. [CRITICAL] pyproject.toml name vs setup.py name 불일치 (Minor m3 미수정)

- `pyproject.toml` 라인 6: `name = "math_library"`
- `setup.py` 라인 176: `name="mathlib"`
- 실제: `pip install -e .` 시 `math-library-0.2.0` 사용 (pyproject.toml 우선), setup.py 이름 무시
- README 라인 413: "dist 이름 `mathlib`와 구분" — **거짓**
- **수정 방향**: 한쪽으로 통일. 권장: `math_library` 단일화

### C5. [CRITICAL] `pip install -e .` MinGW 환경 실패

- README 라인 100~107의 editable 설치 명령이 **Windows MinGW 환경에서 무조건 실패**
  1. `wheel` 패키지 누락 시 `error: invalid command 'bdist_wheel'`
  2. `wheel` 설치 후에도 `Microsoft Visual C++ 14.0 or greater is required` — pip가 MSVC 호출
- **수정 방향**: README에 "pip install -e . 는 MinGW에서 미지원, `python setup.py build_ext --inplace --compiler=mingw32`만 지원" 명시 또는 경로 전환 문서화

---

## Major 이슈 (강권, 배포 전 해결 권고)

### M-audit-1. tan(π/2) 부호 반전 — M3 미수정

- 구현 `-1.633e16`, `math.tan(π/2) = +1.633e16`
- verification_report M3이 그대로 남아 있음
- 위치: `src/math_library/_core/trigonometric.pyx:223` tan 로직 또는 argument_reduction `n & 3` 매핑
- 영향: π/2 근방 부호 반전, 기하학/제어 오작동 위험

### M-audit-2. zeta(2) 오차 잔존 — M6 부분 수정

- 구현 `1.6449340668481436`, 정답 `π²/6 = 1.6449340668482264`
- 절대오차 8.28e-14 (sim-engineer 이전 대비 600배 개선됐으나 ~370 ULP)
- 특수 함수 ULP 스펙 명시적 기준 없으므로 허용 가능하나 README 정직 표기 필요

### M-audit-3. performance_benchmark.md의 ULP 표 허위 PASS

- 라인 62~70: `ln: avg=0.247, max=1 PASS` — **실측 max=257**
- doc-writer가 재측정 없이 이전 벤치 복사

### M-audit-4. README §7 "구현 철학"과 복소수 경로 실구현 모순

- README: "libc.math의 sin/cos/exp/log 직접 호출하지 않음"
- 실제 `src/math_library/_core/trigonometric.pyx:275`: `return _cmath.sin(x)` (복소수 경로는 Python cmath 위임, 내부적으로 libm 호출)
- "복소수 경로는 Python cmath 위임" 명시 필요

### M-audit-5. `help(m.sin)` 내부 이름 `sin_dispatch` 노출

- 사용자 `help(m.sin)` 시 `sin_dispatch(x)` 표시 — 내부 구현이 API에 새는 중
- `__name__` 덮어쓰기 or docstring 정비

### M-audit-6. `log_table.txt` vs `log_table_generated.txt` 중복

- `src/math_library/_core/`에 두 파일 공존, 내용 다름 (7705 vs 8441 byte)
- 어느 것이 빌드 소스인지 불분명
- 중복 제거 필요

### M-audit-7. 기존 테스트 `tol=1e-6` — ULP 회귀 미탐지

- `tests/test_*.py` 18개 모두 `tol=1e-6 / 1e-8` 허용오차
- C1(ln 1023 ULP) 같은 회귀를 탐지 불가
- ULP-기반 검증 테스트 신규 필요

### M-audit-8. `build/` 디렉토리 24개 .pyd 중복

- `.gitignore`에는 포함됨 (push 안 됨)
- 로컬 정리 관점에서 커밋 전 삭제 권장

---

## Minor 이슈

- **m-audit-1**: README 라인 413 dist name 문구 → C4 해결 후 정정
- **m-audit-2**: README "장기 계획 — pytest 자동 발견 래퍼 정비" → 이미 완료됨, 제거
- **m-audit-3**: README Differentiation 예시 주석 "≈ 0.5403023058681398" (math.cos(1) 값) vs 실제 수치미분 출력 `0.5403023058683184` — "≈ cos(1) = 0.540..." 형태 권장
- **m-audit-4**: `src/math_library/_core/__init__.py` untracked 상태
- **m-audit-5**: `setup.py annotate=True` → 매 빌드 HTML 100MB 생성, 배포용 `False` 권장
- **m-audit-6**: `power(0.0, -1.0)` ZeroDivisionError/OverflowError IEEE 754-2019 정책 명시
- **m-audit-7**: `noexcept nogil` 선언 함수 중 복소수 경로의 `cmath` 호출이 GIL 재획득

---

## 테스트 결과

| 범주 | PASS | FAIL | 비고 |
|---|---|---|---|
| pytest 수집 | 18/18 | 0 | test_main() 래퍼 동작 |
| pytest 실행 | 18/18 | 0 | 0.90s |
| README 빠른 시작 복붙 | 5/8 | **3** | exp, ln, gamma 주석 |
| IEEE 754 특수값 | 13/14 | 1 | `ln(1.0) = 1.57e-17` ≠ 0 |
| Payne-Hanek 큰 인수 | 4/4 | 0 | 0 ULP 일치 |
| 21 primitive 복소수 dispatch | 21/21 | 0 | cmath 일치 |
| Clean 빌드 (MinGW) | ✔ | — | 성공 |
| `pip install -e .` | ✘ | 1 | **MSVC 요구 실패** |
| ULP 실측 sin | avg 0.015, max 1 | | 스펙 내 |
| ULP 실측 **ln** | avg 0.264, **max 257** | | **스펙 위반** |
| ULP 실측 exp | avg 0.095, max 1 | | 스펙 내 |

---

## 성능 재측정

| 함수 | math (ns) | mathlib (ns) | ratio | 판정 |
|---|---|---|---|---|
| sin | 26.4 | 24.3 | 0.92x | OK |
| cos | 26.4 | 24.8 | 0.94x | OK |
| tan | 33.9 | 28.6 | 0.85x | OK |
| ln | 44.5 | 23.6 | 0.53x | OK |
| sqrt | 24.0 | 21.8 | 0.91x | OK |
| exp | 25.1 | 27.8 | 1.11x | OK |
| hypersin | 29.6 | 36.1 | 1.22x | OK |

성능 수치는 README와 ±10% 이내 일치 — **이 부분만 신뢰 가능**.

---

## 배포 승인 조건

GO 전환을 위한 최소 조건:

1. **C1** ln 구현 수정 또는 "ln 최대 1 ULP" 주장 철회
2. **C2** 빠른 시작 3개 주석값 교정
3. **C3** 알려진 이슈 5개 false-positive 정정
4. **C4** pyproject.toml/setup.py name 통일
5. **C5** README Editable 설치 섹션 수정 또는 제거

Major 항목은 릴리스 노트 정직 기록 시 허용 가능 (M-audit-3 벤치 문서 정정은 필수).

---

## 관련 파일 경로

- `README.md` (C2, C3, M-audit-4, Minor들)
- `pyproject.toml` (C4)
- `setup.py` (C4, m-audit-5)
- `src/math_library/_core/logarithmic.pyx` (C1)
- `src/math_library/_core/trigonometric.pyx` (M-audit-1, M-audit-5)
- `src/math_library/zeta_function/zeta.pyx` (M-audit-2)
- `src/math_library/_core/log_table*.txt` (M-audit-6)
- `docs/performance_benchmark.md` (M-audit-3)
- `tests/test_*.py` (M-audit-7)
- `build/` (M-audit-8)
