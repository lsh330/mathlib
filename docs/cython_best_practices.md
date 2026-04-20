# Cython 고성능 수학 라이브러리 모범 사례

> 플랫폼: Windows 10/11, Python 3.11.9, MinGW-w64 UCRT64 gcc 15.2.0 (주 환경), MSVC 향후 지원
> 목표: CPython `math` 모듈 (libm 래퍼) 대비 1.1~1.5배 이내 속도
> 작성일: 2026-04-20

---

## 1. 디렉티브 조합

### 1.1 권장 파일 헤더

```cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
```

`language_level=3`은 Cython 3.0부터 기본값이 되었으나 명시적으로 선언하는 것이 권장된다.
파일 헤더 디렉티브는 커맨드라인 플래그로 재정의(override)할 수 있다.

### 1.2 각 디렉티브의 의미와 리스크

| 디렉티브 | 효과 | 리스크 | 수학 라이브러리 적합성 |
|---|---|---|---|
| `boundscheck=False` | 배열 인덱스 범위 검사 제거 | 범위 초과 시 segfault 또는 데이터 오염 | 스칼라 연산에는 영향 없음 |
| `wraparound=False` | 음수 인덱스 지원 제거 | `arr[-1]` 같은 음수 인덱스가 미정의 동작 | 스칼라 함수엔 무관 |
| `cdivision=True` | C 방식 정수 나눗셈 (Python 차이 미보정) | ZeroDivisionError 미발생, 음수 나눗셈 결과 상이 | 부동소수점 연산 위주라면 오버헤드 제거 효과 |
| `initializedcheck=False` | 메모리뷰 초기화 여부 검사 제거 | 미초기화 메모리 접근 위험 | 타입드 배열 연산에서 유효 |
| `nonecheck=False` | `None` 값 검사 제거 | None 전달 시 segfault | 내부 `cdef` 함수에서만 적용 권장 |

**주의**: scikit-learn, scipy 등 주요 프로젝트의 Cython 가이드라인은 "모든 함수에 무조건 모든 디렉티브를 붙이는 것은 cargo-cult 프로그래밍"이라고 경고한다. 각 디렉티브가 실제로 적용되는 코드 경로가 있는 함수에만 붙일 것.

### 1.3 함수 단위 적용 (권장)

```cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _inner_loop(double[:] arr, Py_ssize_t n) noexcept nogil:
    ...
```

파일 전체 디렉티브보다 함수 단위 데코레이터가 의도를 명확히 한다.

### 1.4 `@cython.inline` vs `cdef inline`

- `cdef inline`: Cython 수준 인라인 힌트. C 컴파일러에게 인라인 추천을 전달한다. `.pxd` 헤더 파일에 선언하기 적합.
- `@cython.inline`: `def` 함수에 사용. 해당 함수를 호출하는 위치에 코드를 삽입하도록 요청.
- 고성능 수치 함수에서는 `cdef inline`이 표준 패턴이다.

---

## 2. 타입 선언 패턴

### 2.1 기본 스칼라 타입

```cython
from libc.stdint cimport int32_t, int64_t, uint64_t

cdef double x
cdef long long n
cdef int64_t bits
cdef uint64_t raw
```

- `double` = C `double` (64비트 IEEE 754)
- `long long` = 플랫폼 독립적 64비트 정수 (Windows에서 보장)
- `int64_t` / `uint64_t`: bit-level 조작에 사용. `from libc.stdint cimport`로 가져옴.

### 2.2 `Py_ssize_t` vs `int` vs `int64_t`

| 타입 | 용도 | 비고 |
|---|---|---|
| `Py_ssize_t` | 배열 길이, 메모리뷰 인덱스 | Python 컨테이너 크기 타입과 호환 (64비트 플랫폼에서 64비트) |
| `int` | 단순 카운터, 소규모 정수 | 32비트 고정 |
| `int64_t` | IEEE 754 비트 조작 | 부호 있는 64비트 보장 |
| `uint64_t` | 비트 마스크, 비트 추출 | 부호 없는 64비트 보장 |

루프 카운터와 메모리뷰 인덱스에는 `Py_ssize_t`를 사용하는 것이 Cython 내부와 정합성이 맞는다.

### 2.3 함수 시그니처 타입 명시

```cython
# 외부 공개 함수 (Python에서 호출 가능)
cpdef double sin(double x) noexcept:
    return _sin_impl(x)

# 내부 구현 (C 레벨만)
cdef inline double _sin_impl(double x) noexcept nogil:
    ...
```

- `def sin(double x) -> double:` 형태도 가능하나, Cython에서는 `cpdef double sin(double x)`가 더 명확하다.
- 리턴 타입 명시: `double` 반환 타입을 선언하면 Cython이 Python `float` 박싱 코드를 최소화한다.
- `noexcept`: Cython 3.0+에서 예외 검사 코드 완전 제거. 수학 함수처럼 예외를 발생시키지 않는 함수에 필수.

### 2.4 `cpdef` vs `cdef` vs `def` 선택 기준

```
Python에서 직접 호출 필요? → cpdef 또는 def
내부 구현 전용? → cdef
최고 성능 내부 루틴? → cdef inline noexcept nogil
```

성능 수치 (참고):
- `cdef` 함수 간 호출: ~3ns (C 함수 호출 수준)
- Python → `cpdef` 경계 호출: ~30ns (boxing/unboxing 포함)
- Python → `def` 경계 호출: ~62ns
- `cpdef` fused type 디스패치: ~370ns 이상 (fused type 회피 권장)

외부 공개 API는 `cpdef`로 선언하되, 내부 계산 루틴은 전부 `cdef inline noexcept nogil`로 구성하는 것이 최적이다.

---

## 3. C 수준 호출 패턴

### 3.1 libc.math 함수 cimport

```cython
from libc.math cimport (
    fma, fmaf,
    frexp, frexpf,
    ldexp, ldexpf,
    copysign, copysignf,
    isnan, isinf, isfinite, isnormal,
    signbit, fpclassify,
    sqrt, cbrt, exp, log, log2, log10,
    sin, cos, tan, asin, acos, atan, atan2,
    sinh, cosh, tanh, asinh, acosh, atanh,
    ceil, floor, round, trunc, fabs, fmod,
    INFINITY, NAN, HUGE_VAL,
    M_PI, M_E, M_SQRT2, M_LN2,
    FP_NAN, FP_INFINITE, FP_ZERO, FP_NORMAL, FP_SUBNORMAL,
)
from libc.stdint cimport int64_t, uint64_t, int32_t, uint32_t
```

`libc.math`은 C99 수학 함수를 모두 제공한다. `double`, `float`, `long double` 세 정밀도 버전이 각각 기본 이름, `f` 접미사, `l` 접미사로 구분된다.

Linux에서는 `libraries=["m"]`을 Extension에 추가해야 하지만, Windows(MinGW/MSVC)에서는 math 라이브러리가 자동 링킹되는 경우가 많다.

**중요**: 본 라이브러리는 자체 구현 철학이므로 `sin`, `cos`, `exp`, `log` 같은 **elementary 함수는 libc.math에서 가져오지 않고 자체 구현**한다. 단, `fma`, `frexp`, `ldexp`, `sqrt`, `isnan`, `isinf` 같은 **수치 헬퍼**는 libc.math 사용을 허용한다.

### 3.2 IEEE 754 비트 패턴 접근 — C union 트릭

```cython
# .pxd 파일에 선언
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

# .pyx 파일에서 사용
cdef inline uint64_t double_to_bits(double x) noexcept nogil:
    cdef DoubleUnion u
    u.d = x
    return u.u

cdef inline double bits_to_double(uint64_t bits) noexcept nogil:
    cdef DoubleUnion u
    u.u = bits
    return u.d
```

C99 수준에서는 union 트릭이 GCC/MSVC 모두에서 정의된 동작이다.

**주의**: `cdef extern from *:`의 `"""..."""` 블록은 Cython이 생성하는 C 파일에 인라인으로 삽입된다.

---

## 4. FMA와 Horner 스킴

### 4.1 Horner's Method + FMA 패턴

차수 n 다항식 p(x) = c₀ + c₁x + c₂x² + c₃x³을 Horner 형식으로:

p(x) = c₀ + x(c₁ + x(c₂ + x·c₃))

이를 FMA로 구현하면:

```cython
from libc.math cimport fma

cdef inline double poly3(double x,
                          double c0, double c1,
                          double c2, double c3) noexcept nogil:
    # Horner + FMA: 3 FMA 명령으로 완결
    return fma(fma(fma(c3, x, c2), x, c1), x, c0)

cdef inline double poly5(double x,
                          double c0, double c1, double c2,
                          double c3, double c4, double c5) noexcept nogil:
    return fma(fma(fma(fma(fma(c5, x, c4), x, c3), x, c2), x, c1), x, c0)
```

### 4.2 FMA의 장점

1. **정확도**: 중간 반올림이 없어 표준 multiply-add보다 0.5 ulp 오차 보장
2. **속도**: 단일 명령 (vfmadd231sd 등) — 곱셈+덧셈을 한 명령으로 처리
3. **파이프라인**: 의존성 체인이 줄어 Out-of-Order 실행 효율 증가

### 4.3 컴파일러 FMA 활성화

GCC/MinGW: `-mfma` (또는 `-march=native`) 플래그 필요.
MSVC: `/arch:AVX2` 플래그 필요.

C99 `fma()` 함수는 소프트웨어 폴백이 있을 수 있다. 실제 하드웨어 FMA 명령 생성은 컴파일러 플래그 의존적이다.

**`__builtin_fma` vs C99 `fma()`**:
- `__builtin_fma`는 GCC 전용, MSVC 불호환. 이식성 관점에서 C99 `fma()`를 권장.
- `-mfma` 플래그 설정 시 GCC는 `fma()` 호출을 자동으로 하드웨어 FMA 명령으로 변환한다.

---

## 5. 컴파일 플래그 권장안

### 5.1 GCC / MinGW-w64

```python
extra_compile_args_gcc = [
    "-O3",
    "-march=native",       # 현재 CPU 최적화 (배포용은 -march=x86-64-v3 등 고려)
    "-mfma",               # FMA 명령 활성화
    "-fno-math-errno",     # errno 설정 코드 제거 (안전)
    "-fno-trapping-math",  # 부동소수점 트랩 없다고 가정 (안전)
    # 아래는 주의 필요 → 별도 섹션 5.3 참고
    # "-ffast-math",       # IEEE754 위반 가능 — 기본 비활성화
    # "-ffinite-math-only",# NaN/Inf 검사 제거 — 위험
]
```

### 5.2 MSVC

```python
extra_compile_args_msvc = [
    "/O2",
    "/fp:fast",     # MSVC fast-math (ffast-math 유사하나 덜 공격적)
    "/arch:AVX2",   # FMA 및 AVX2 명령 활성화
    # "/fp:strict"  # 완전 IEEE754 준수가 필요하면 대신 사용
]
```

### 5.3 `-ffast-math`의 위험성 분석 (중요)

`-ffast-math`는 다음 8개 하위 플래그를 일괄 활성화한다:

| 하위 플래그 | 동작 | 수학 라이브러리 위험도 |
|---|---|---|
| `-fno-math-errno` | errno 설정 코드 제거 | 낮음 (errno 미사용 시 안전) |
| `-fno-trapping-math` | 부동소수점 트랩 없다고 가정 | 낮음 |
| `-ffinite-math-only` | NaN/Inf 없다고 가정, `isnan()` 결과 항상 false | **높음** — NaN/Inf 반환 함수에서 파괴적 |
| `-fassociative-math` | 덧셈 순서 재배열 허용 | **높음** — Kahan summation 등 보정 산술 파괴 |
| `-fno-signed-zeros` | -0.0과 +0.0 구분 없음 | 중간 |
| `-freciprocal-math` | `x/y` → `x*(1/y)` 변환 | 중간 — 정밀도 손실 |
| `-funsafe-math-optimizations` | sqrt(x)*sqrt(x) → x 등 변환 | **높음** — 결과 변경 가능 |
| `-fcx-limited-range` | 복소수 산술 단순화 | 중간 |

**특히 위험한 사례**:
- `isnan(x)` 호출이 컴파일러에 의해 항상 `false`를 반환하는 코드로 교체됨
- 공유 라이브러리에 링킹 시 MXCSR 레지스터의 DAZ/FTZ 플래그를 전역 설정하여, 해당 프로세스의 **모든** 부동소수점 연산에서 비정규수(subnormal/denormal)가 0으로 플러시됨. 이는 **비관련 코드에도 영향**을 미치는 비지역 효과(non-local effect)임.
- Sterbenz Lemma, 2-sum 알고리즘 등의 오차 분석 정리가 성립하지 않게 됨

**권장 안전 구성**:
```
-fno-math-errno -fno-trapping-math
```
이 두 플래그만 사용하면 IEEE 754 준수를 유지하면서 errno 체크와 트랩 코드만 제거하므로 수학 라이브러리에 적합하다.

`-ffast-math`는 사용하지 않거나, 사용 시에도 `-fno-finite-math-only` 및 `-fno-associative-math`를 명시적으로 재활성화하여 일부 위험을 제한해야 한다.

---

## 6. setup.py / pyproject.toml 템플릿

### 6.1 setup.py (Windows MinGW + MSVC 자동 감지)

```python
import sys
import os
from setuptools import setup, Extension
from Cython.Build import cythonize

def is_msvc():
    """MSVC 컴파일러 사용 여부 감지"""
    compiler = os.environ.get("CC", "")
    if sys.platform == "win32":
        # 환경변수 미설정 시 MSVC 기본 가정
        if not compiler or "cl" in compiler.lower():
            return True
    return False

# 컴파일러별 플래그
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
    # MinGW 64비트 빌드 필수 매크로
    define_macros = [("MS_WIN64", None)] if sys.platform == "win32" else []

# MinGW에서 math 라이브러리 명시적 링킹 (Linux에서도 필요)
libraries = []
if sys.platform != "win32":
    libraries = ["m"]

extensions = cythonize(
    [
        Extension(
            name="math_library.cython.trigonometric",
            sources=["src/math_library/cython/trigonometric.pyx"],
            libraries=libraries,
            define_macros=define_macros,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
        # 추가 모듈을 여기에 나열
    ],
    compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "initializedcheck": False,
        "nonecheck": False,
        "embedsignature": True,
    },
    annotate=True,      # 개발 중 HTML 어노테이션 생성 (배포 시 False)
    nthreads=4,
)

setup(
    name="mathlib",
    ext_modules=extensions,
)
```

### 6.2 pyproject.toml (build-system 선언)

```toml
[build-system]
requires = ["setuptools>=61", "Cython>=3.0"]
build-backend = "setuptools.build_meta"

[project]
name = "mathlib"
version = "0.1.0"
requires-python = ">=3.10"
```

`Cython>=3.0`을 `build-system.requires`에 명시하면 `pip install`이 빌드 전 Cython을 자동 설치한다.

### 6.3 editable install과 재컴파일

```bash
# 최초 설치
pip install -e . --no-build-isolation

# .pyx 파일 수정 후 재컴파일
pip install -e . --no-build-isolation

# 또는 직접 빌드
python setup.py build_ext --inplace
```

`pip install -e .`는 `.pyx` 파일 변경을 자동 감지하지 않는다. 매 수정 후 위 명령을 반복해야 한다. 빠른 개발 반복을 위해서는 `python setup.py build_ext --inplace`가 더 빠르다.

### 6.4 `.pyx` vs `.pxd` 파일 역할

| 파일 확장자 | 역할 | C 대응 |
|---|---|---|
| `.pyx` | 구현 파일 — 실제 코드 포함 | `.c` |
| `.pxd` | 선언 파일 — `cdef` 타입·함수 선언만 포함 | `.h` |

`.pxd`에는 `cdef inline` 함수를 선언하고, 다른 `.pyx` 모듈에서 `cimport`로 가져올 수 있다. scipy의 `cython_special` 모듈이 이 패턴을 사용한다.

```cython
# helpers.pxd
from libc.math cimport fma

cdef inline double poly3(double x, double c0, double c1,
                          double c2, double c3) noexcept nogil:
    return fma(fma(fma(c3, x, c2), x, c1), x, c0)
```

```cython
# trigonometric.pyx
from .helpers cimport poly3
```

---

## 7. 벤치마크 템플릿

### 7.1 공정한 비교 원칙

Python `math.sin`은 이미 C 라이브러리(libm) 래퍼이다. Cython `cpdef double sin(double x)`의 경쟁 상대는 libm 자체이며, Python 경계 오버헤드(~30-60ns)가 양쪽 모두에 존재하므로 이를 감안해야 한다.

```python
import math
import timeit

# 측정 대상 함수
import math_library.cython.trigonometric as cy_trig

N = 1_000_000
x = 1.2345

def bench_stdlib():
    return math.sin(x)

def bench_cython():
    return cy_trig.sin(x)

t_stdlib = timeit.repeat(bench_stdlib, repeat=7, number=N)
t_cython = timeit.repeat(bench_cython, repeat=7, number=N)

best_stdlib = min(t_stdlib) / N * 1e9  # ns/call
best_cython = min(t_cython) / N * 1e9  # ns/call

print(f"math.sin:    {best_stdlib:.1f} ns/call")
print(f"cy_trig.sin: {best_cython:.1f} ns/call")
print(f"ratio:       {best_cython / best_stdlib:.3f}x")
```

### 7.2 Python 호출 오버헤드 제거 (내부 C 속도 측정)

Python 경계 오버헤드를 제외하고 순수 수학 연산만 측정하려면 Cython 내부에서 루프를 돌린다.

```cython
# benchmark_inner.pyx
def bench_cdef_inner(int N):
    cdef double x = 1.2345, s = 0.0
    cdef int i
    for i in range(N):
        s += _sin_impl(x)
    return s  # 컴파일러 최적화 방지를 위한 누적 반환
```

### 7.3 공정 비교 주의사항

1. **입력 분포 고정**: 같은 `x` 값 또는 동일한 난수 시드로 생성된 배열 사용
2. **캐시 효과**: 첫 번째 실행은 워밍업으로 제외 (`timeit`의 `repeat` 첫 번째 값 제외)
3. **CPU 주파수 변동**: `perf_counter` 기반 단일 측정보다 `timeit.repeat`의 최솟값(best-of-N)이 더 신뢰성 있음
4. **배열 연산 vs 스칼라**: Python `math.sin`은 스칼라 전용. 배열 처리 비교는 `numpy.sin`과 해야 공정함

### 7.4 Cython 어노테이션으로 병목 확인

```bash
cython -a src/math_library/cython/trigonometric.pyx
# trigonometric.html 생성
# 브라우저로 열어 노란색 라인(CPython 상호작용) 확인
# 완전 흰색 = C 수준 실행 = 최적
```

---

## 8. MinGW + Python 3.11 조합 알려진 주의사항

### 8.1 `-DMS_WIN64` 매크로

MinGW로 64비트 Python 확장을 빌드할 때 `_WIN64` 및 `MS_WIN64` 매크로가 자동 정의되지 않을 수 있다. `setup.py`에서 명시적으로 추가:

```python
define_macros=[("MS_WIN64", None)]  # MinGW 64비트 빌드 시
```

이 프로젝트에서는 이미 검증된 사항이다 (`hello.pyx` 빌드 성공).

### 8.2 -lm 링킹

Windows(MinGW/MSVC)에서는 대부분 자동 링킹되지만, Python 자체가 math 심볼을 포함하고 있어 별도 `-lm` 없이도 동작할 수 있다. 단, Linux 배포를 고려한다면 `libraries=["m"]` 조건부 추가가 안전하다.

### 8.3 Python 3.8+ DLL 검색 경로

Python 3.8부터 DLL 검색 경로가 변경되었다. MinGW 런타임 DLL(`libgcc_s_seh-1.dll`, `libwinpthread-1.dll` 등)이 PATH에 있어도 로드되지 않을 수 있다. 해결:
- `-static-libgcc -static-libstdc++` 플래그로 정적 링킹
- 또는 `os.add_dll_directory()` 호출

### 8.4 MinGW vs MSVC 런타임 비호환

MinGW는 MSVCRT 대신 자체 C 런타임을 사용한다. numpy, scipy 등 MSVC로 빌드된 패키지와 C++ 예외나 `FILE*` 포인터를 경계 너머로 전달하면 충돌 가능성이 있다. **순수 C (C++ 미사용) Cython 모듈**이라면 대부분 문제없다.

### 8.5 Cython 3.0+ noexcept 필수화

Cython 3.0부터 `cdef` 함수에 예외 명시가 없으면 경고가 발생한다. `noexcept`나 `except -1`을 명시적으로 선언할 것. 수학 함수에는 `noexcept`가 적합하다.

---

## 9. 권장 파일 구조

```
src/math_library/
├── __init__.py             # 공용 API 재노출
└── cython/
    ├── __init__.py
    ├── _helpers.pxd        # cdef inline 유틸리티 (poly, FMA, bit manip 등)
    ├── _constants.pyx      # pi, e, epsilon (한 번만 계산, 캐싱)
    ├── _constants.pxd      # 상수 cdef inline 선언
    ├── trigonometric.pyx   # sin, cos, tan 등
    ├── trigonometric.pxd   # 외부 cimport용 선언
    ├── inverse_trig.pyx    # asin, acos, atan
    ├── inverse_trig.pxd
    ├── hyperbolic.pyx      # sinh, cosh, tanh
    ├── hyperbolic.pxd
    ├── exponential.pyx     # exp, pow, sqrt
    ├── exponential.pxd
    ├── logarithmic.pyx     # log, ln
    ├── logarithmic.pxd
    └── special.pyx         # gamma, beta, Bessel, ... (기존 알고리즘 유지)
```

`.pxd` 파일에 선언을 두면 다른 Cython 모듈이 C 수준으로 직접 호출할 수 있어 모듈 간 오버헤드가 없다.
