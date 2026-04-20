"""
setup.py — mathlib Cython 빌드 시스템
MinGW-w64 UCRT64 (gcc 15.2.0) 및 MSVC 자동 감지
"""
import sys
import os
from setuptools import setup, Extension
from Cython.Build import cythonize

# ------------------------------------------------------------------ 컴파일러 감지
def is_mingw():
    """MinGW/GCC 컴파일러 사용 여부 감지 (m5: 보강된 감지 로직)"""
    cc = os.environ.get("CC", "").lower()
    # --compiler=mingw32 또는 --compiler mingw32 인자 감지
    for arg in sys.argv:
        if "mingw32" in arg.lower() or "mingw64" in arg.lower():
            return True
        if arg.lower() in ("--compiler=mingw32", "--compiler=mingw64"):
            return True
    # CC 환경변수 감지
    if "gcc" in cc or "mingw" in cc:
        return True
    # MSYS2/UCRT64 환경 감지
    path_env = os.environ.get("PATH", "")
    if "msys64" in path_env.lower() or "mingw" in path_env.lower():
        return True
    return False

_USE_MINGW = is_mingw()

if _USE_MINGW:
    # MinGW-w64 GCC 15.2.0 최적화 플래그
    compile_args = [
        "-O3",
        "-march=native",
        "-mfma",
        "-fno-math-errno",
        "-fno-trapping-math",
        # IEEE 754 준수: -ffast-math 금지
    ]
    # MinGW 런타임 DLL 의존성 제거: 정적 링킹
    # -Wl,-Bstatic: 이후 라이브러리를 정적으로 링킹
    # -lgcc -lgcc_s: gcc 런타임을 명시적으로 정적으로 포함
    link_args = [
        "-static-libgcc",
        "-static-libstdc++",
        "-Wl,-Bstatic,--whole-archive",
        "-lgcc",
        "-Wl,--no-whole-archive,-Bdynamic",
    ] if sys.platform == "win32" else []
    define_macros = [("MS_WIN64", None)] if sys.platform == "win32" else []
else:
    # MSVC
    compile_args = ["/O2", "/fp:fast", "/arch:AVX2"]
    link_args = []
    define_macros = []

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
]


setup(
    name="math_library",
    ext_modules=cythonize(
        extensions,
        compiler_directives=COMPILER_DIRECTIVES,
        annotate=False,     # 배포용: HTML 어노테이션 비활성화 (개발 시 True로 변경)
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
    ],
)
