"""
setup.py — mathlib Cython 빌드 시스템
MinGW-w64 UCRT64 (gcc 15.2.0) 및 MSVC 자동 감지
"""
import sys
import os

# ------------------------------------------------------------------ VS 2026 Preview 보조
# VS 2026 Preview의 vcvarsall.bat은 내부에서 vswhere.exe를 PATH 기반으로 호출한다.
# 시스템 PATH에 Installer 폴더가 없으면 env dump가 실패하고 setuptools는
# "Unable to find a compatible Visual Studio installation" 에러를 낸다.
# 따라서 Windows에서 해당 경로가 누락된 경우 PATH 앞에 삽입한다(프로세스 한정).
if sys.platform == "win32":
    _vs_installer = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer"
    if os.path.isfile(os.path.join(_vs_installer, "vswhere.exe")):
        _path = os.environ.get("PATH", "")
        if _vs_installer.lower() not in _path.lower():
            os.environ["PATH"] = _vs_installer + os.pathsep + _path

import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

# ------------------------------------------------------------------ 컴파일러 감지
def is_mingw():
    """MinGW/GCC 컴파일러 사용 여부 감지 (m5: 보강된 감지 로직)

    우선순위: 명시적 --compiler 인자 > CC 환경변수 > PATH 스니핑.
    --compiler=msvc가 명시되면 PATH에 mingw가 있어도 MSVC로 간주.
    """
    # 1) 명시적 --compiler 인자를 최우선 처리
    for arg in sys.argv:
        a = arg.lower()
        if a in ("--compiler=msvc", "--compiler=clang", "--compiler=bcpp",
                "--compiler=unix", "--compiler=cygwin"):
            return False
        if a in ("--compiler=mingw32", "--compiler=mingw64"):
            return True
        if "mingw32" in a or "mingw64" in a:
            return True
    # 2) CC 환경변수 감지
    cc = os.environ.get("CC", "").lower()
    if cc in ("cl", "cl.exe", "msvc"):
        return False
    if "gcc" in cc or "mingw" in cc:
        return True
    # 3) MSYS2/UCRT64 환경 감지 (fallback)
    path_env = os.environ.get("PATH", "")
    if "msys64" in path_env.lower() or "mingw" in path_env.lower():
        return True
    return False

_USE_MINGW = is_mingw()

if _USE_MINGW:
    # MinGW-w64 GCC 15.2.0 최적화 플래그 (C++17로 통일: Cython 3.x의 C 모드가
    # MSVC와 호환되지 않는 GCC 확장 __complex__를 생성하므로 C++17로 일관)
    compile_args = [
        "-std=c++17",
        "-O3",
        "-march=native",
        "-mfma",
        "-fno-math-errno",
        "-fno-trapping-math",
        # IEEE 754 준수: -ffast-math 금지
    ]
    # MinGW 런타임 DLL 의존성 제거: 정적 링킹
    link_args = [
        "-static-libgcc",
        "-static-libstdc++",
        "-Wl,-Bstatic,--whole-archive",
        "-lgcc",
        "-Wl,--no-whole-archive,-Bdynamic",
    ] if sys.platform == "win32" else []
    define_macros = [("MS_WIN64", None)] if sys.platform == "win32" else []
else:
    # MSVC — C++17 모드 통일
    compile_args = ["/std:c++17", "/O2", "/fp:fast", "/arch:AVX2", "/EHsc"]
    link_args = []
    # CYTHON_CCOMPLEX=0 : Cython이 C99 _Complex / GCC __real__ 확장 대신
    #                     자체 struct 기반 complex를 사용하도록 강제 (MSVC 호환)
    define_macros = [("CYTHON_CCOMPLEX", "0")]

# Linux: math 라이브러리 명시 링킹
libraries = ["m"] if sys.platform != "win32" else []

# Cython 컴파일러 지시자
COMPILER_DIRECTIVES = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
    "nonecheck": False,
    "embedsignature": True,
}

def make_ext(name, sources, **kwargs):
    # language="c++"로 통일 — Cython 3.x의 C 모드 __complex__ (GCC 확장) 회피
    kwargs.setdefault("language", "c++")
    return Extension(
        name=name,
        sources=sources,
        libraries=libraries,
        define_macros=define_macros,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        **kwargs,
    )


# ------------------------------------------------------------------ Extension 목록
extensions = [
    # Tier 1: 의존성 없는 모듈
    make_ext(
        "math_library._core._constants",
        ["src/math_library/_core/_constants.pyx"],
    ),
    # Tier 2: argument reduction
    make_ext(
        "math_library._core.argument_reduction",
        ["src/math_library/_core/argument_reduction.pyx"],
    ),
    # Tier 2: exp/expm1
    make_ext(
        "math_library._core.exponential",
        ["src/math_library/_core/exponential.pyx"],
    ),
    # Tier 2: ln/log
    make_ext(
        "math_library._core.logarithmic",
        ["src/math_library/_core/logarithmic.pyx"],
    ),
    # Tier 2: sqrt/power
    make_ext(
        "math_library._core.power_sqrt",
        ["src/math_library/_core/power_sqrt.pyx"],
    ),
    # Tier 3: 삼각함수
    make_ext(
        "math_library._core.trigonometric",
        ["src/math_library/_core/trigonometric.pyx"],
    ),
    # Tier 3: 역삼각함수
    make_ext(
        "math_library._core.inverse_trig",
        ["src/math_library/_core/inverse_trig.pyx"],
    ),
    # Tier 3: 쌍곡함수
    make_ext(
        "math_library._core.hyperbolic",
        ["src/math_library/_core/hyperbolic.pyx"],
    ),
    # Tier 4: 특수 함수
    make_ext(
        "math_library.gamma_function.gamma",
        ["src/math_library/gamma_function/gamma.pyx"],
    ),
    make_ext(
        "math_library.beta_function.beta",
        ["src/math_library/beta_function/beta.pyx"],
    ),
    make_ext(
        "math_library.bessel_function.bessel",
        ["src/math_library/bessel_function/bessel.pyx"],
    ),
    make_ext(
        "math_library.legendre_function.legendre",
        ["src/math_library/legendre_function/legendre.pyx"],
    ),
    make_ext(
        "math_library.lambert_w_function.lambert_w",
        ["src/math_library/lambert_w_function/lambert_w.pyx"],
    ),
    make_ext(
        "math_library.zeta_function.zeta",
        ["src/math_library/zeta_function/zeta.pyx"],
    ),
    make_ext(
        "math_library.euler_pi_function.euler_pi",
        ["src/math_library/euler_pi_function/euler_pi.pyx"],
    ),
    make_ext(
        "math_library.heaviside_step_function.heaviside",
        ["src/math_library/heaviside_step_function/heaviside.pyx"],
    ),
    make_ext(
        "math_library.gcd.gcd",
        ["src/math_library/gcd/gcd.pyx"],
    ),
    make_ext(
        "math_library.lcm.lcm",
        ["src/math_library/lcm/lcm.pyx"],
    ),
    # Tier 5: Differentiation (Ridders)
    make_ext(
        "math_library.differentiation.differentiation",
        ["src/math_library/differentiation/differentiation.pyx"],
    ),
    # Phase 2: 신규 모듈
    make_ext(
        "math_library._core.inverse_hyperbolic",
        ["src/math_library/_core/inverse_hyperbolic.pyx"],
    ),
    make_ext(
        "math_library._core.multi_arg",
        ["src/math_library/_core/multi_arg.pyx"],
    ),
    make_ext(
        "math_library._core.discrete",
        ["src/math_library/_core/discrete.pyx"],
    ),
    make_ext(
        "math_library._core.aggregate",
        ["src/math_library/_core/aggregate.pyx"],
    ),
    make_ext(
        "math_library._core.special_functions",
        ["src/math_library/_core/special_functions.pyx"],
    ),
    make_ext(
        "math_library._core.ieee_ops",
        ["src/math_library/_core/ieee_ops.pyx"],
    ),
    make_ext(
        "math_library._core.predicates",
        ["src/math_library/_core/predicates.pyx"],
    ),
    # Tier 6: NumericalAnalysis (Simpson 적분법 8종)
    make_ext(
        "math_library.numerical_analysis.numerical_analysis",
        ["src/math_library/numerical_analysis/numerical_analysis.pyx"],
    ),
    # Tier 7: LinearAlgebra (행렬 분해 9종)
    make_ext(
        "math_library.linear_algebra.linear_algebra",
        ["src/math_library/linear_algebra/linear_algebra.pyx"],
        include_dirs=[np.get_include()],
    ),
]

# ------------------------------------------------------------------ Laplace Extension (C++17, Phase A)
# 기존 C Extension들과 별도로 cythonize하여 language="c++" 충돌 방지
_laplace_compile_args = (
    ["-std=c++17", "-O3", "-march=native", "-fno-strict-aliasing"]
    if _USE_MINGW
    else ["/std:c++17", "/O2"]
)
_laplace_link_args = (
    [
        "-static-libgcc",
        "-static-libstdc++",
        # libwinpthread 정적 링킹 (DLL 의존성 제거)
        "-Wl,-Bstatic",
        "-lpthread",
        "-Wl,-Bdynamic",
    ]
    if _USE_MINGW and sys.platform == "win32"
    else []
)
_laplace_macros = [("MS_WIN64", None)] if sys.platform == "win32" else []
# SIZEOF_VOID_P가 정의되지 않으면 Cython이 생성하는 코드에서 division-by-zero 발생
# Python 3.11 (MSVC 빌드) + MinGW 컴파일 조합에서 누락되는 경우 명시 보완
_laplace_macros += [("SIZEOF_VOID_P", "8")]

laplace_ext = Extension(
    name="math_library.laplace.laplace_ast",
    sources=[
        "src/math_library/laplace/laplace_ast.pyx",
        "src/math_library/laplace/cpp/expr.cpp",
        "src/math_library/laplace/cpp/pool.cpp",
        "src/math_library/laplace/cpp/rules.cpp",
        "src/math_library/laplace/cpp/laplace.cpp",
        "src/math_library/laplace/cpp/polynomial.cpp",
        "src/math_library/laplace/cpp/partial.cpp",
        "src/math_library/laplace/cpp/inverse.cpp",
        "src/math_library/laplace/cpp/simplify.cpp",
    ],
    language="c++",
    include_dirs=["src/math_library/laplace/cpp"],
    extra_compile_args=_laplace_compile_args,
    extra_link_args=_laplace_link_args,
    define_macros=_laplace_macros,
)

laplace_transform_ext = Extension(
    name="math_library.laplace.laplace",
    sources=[
        "src/math_library/laplace/laplace.pyx",
        "src/math_library/laplace/cpp/expr.cpp",
        "src/math_library/laplace/cpp/pool.cpp",
        "src/math_library/laplace/cpp/rules.cpp",
        "src/math_library/laplace/cpp/laplace.cpp",
        "src/math_library/laplace/cpp/polynomial.cpp",
        "src/math_library/laplace/cpp/partial.cpp",
        "src/math_library/laplace/cpp/inverse.cpp",
        "src/math_library/laplace/cpp/simplify.cpp",
    ],
    language="c++",
    include_dirs=["src/math_library/laplace/cpp"],
    extra_compile_args=_laplace_compile_args,
    extra_link_args=_laplace_link_args,
    define_macros=_laplace_macros,
)

laplace_extensions = [laplace_ext, laplace_transform_ext]


setup(
    name="math_library",
    ext_modules=(
        cythonize(
            extensions,
            compiler_directives=COMPILER_DIRECTIVES,
            annotate=False,
        ) +
        cythonize(
            laplace_extensions,
            compiler_directives=COMPILER_DIRECTIVES,
            annotate=False,
        )
    ),
    package_dir={"": "src"},
    packages=[
        "math_library",
        "math_library._core",
        "math_library.constant",
        "math_library.trigonometric_function",
        "math_library.inverse_trigonometric_function",
        "math_library.hyperbolic_function",
        "math_library.exponential_function",
        "math_library.logarithmic_function",
        "math_library.gamma_function",
        "math_library.beta_function",
        "math_library.bessel_function",
        "math_library.legendre_function",
        "math_library.lambert_w_function",
        "math_library.zeta_function",
        "math_library.euler_pi_function",
        "math_library.heaviside_step_function",
        "math_library.gcd",
        "math_library.lcm",
        "math_library.differentiation",
        "math_library.laplace",
        "math_library.numerical_analysis",
        "math_library.linear_algebra",
    ],
)
