# math_library/__init__.py
# 공용 API 재노출 — Cython _core 고속 경로 우선, 폴백은 Python 구현

import cmath as _cmath

# ------------------------------------------------------------------ 상수
try:
    from ._core._constants import pi, e, epsilon
except ImportError:
    from .constant.pi import pi
    from .constant.e import e
    from .constant.epsilon import epsilon

# ------------------------------------------------------------------ elementary primitives (Cython 고속 경로 — 방향 A dispatch)
try:
    # 방향 A: cpdef object *_dispatch 함수를 최상위 API로 직접 노출
    # isinstance 체크가 Cython 레이어에서 수행되어 Python def 트램펄린 ~30ns 제거
    from ._core.trigonometric import (
        sin_dispatch as sin,
        cos_dispatch as cos,
        tan_dispatch as tan,
        sec_dispatch as sec,
        cosec_dispatch as cosec,
        cotan_dispatch as cotan,
    )
    from ._core.inverse_trig import (
        arcsin_dispatch as arcsin,
        arccos_dispatch as arccos,
        arctan_dispatch as arctan,
        arcsec_dispatch as arcsec,
        arccosec_dispatch as arccosec,
        arccotan_dispatch as arccotan,
    )
    from ._core.hyperbolic import (
        hypersin_dispatch as hypersin,
        hypercos_dispatch as hypercos,
        hypertan_dispatch as hypertan,
        hypersec_dispatch as hypersec,
        hypercosec_dispatch as hypercosec,
        hypercotan_dispatch as hypercotan,
    )
    from ._core.exponential import exp_dispatch as exp
    from ._core.logarithmic import ln_dispatch as ln, log_dispatch as log
    from ._core.power_sqrt import sqrt_dispatch as sqrt, power_dispatch as power

    # help() 시 내부 dispatch 이름 대신 공개 API 이름 표시
    for _name in ('sin', 'cos', 'tan', 'sec', 'cosec', 'cotan',
                  'arcsin', 'arccos', 'arctan', 'arcsec', 'arccosec', 'arccotan',
                  'hypersin', 'hypercos', 'hypertan', 'hypersec', 'hypercosec', 'hypercotan',
                  'exp', 'ln', 'log', 'sqrt', 'power'):
        _fn = vars().get(_name)
        if _fn is not None:
            try:
                _fn.__name__ = _name
                _fn.__qualname__ = _name
            except (AttributeError, TypeError):
                pass
    del _name, _fn

except ImportError:
    # Cython 빌드 전 폴백 (테스트 등에서 사용)
    from .trigonometric_function.sin import sin
    from .trigonometric_function.cos import cos
    from .trigonometric_function.tan import tan
    from .trigonometric_function.sec import sec
    from .trigonometric_function.cosec import cosec
    from .trigonometric_function.cotan import cotan
    from .inverse_trigonometric_function.arcsin import arcsin
    from .inverse_trigonometric_function.arccos import arccos
    from .inverse_trigonometric_function.arctan import arctan
    from .inverse_trigonometric_function.arcsec import arcsec
    from .inverse_trigonometric_function.arccosec import arccosec
    from .inverse_trigonometric_function.arccotan import arccotan
    from .hyperbolic_function.hypersin import hypersin
    from .hyperbolic_function.hypercos import hypercos
    from .hyperbolic_function.hypertan import hypertan
    from .hyperbolic_function.hypersec import hypersec
    from .hyperbolic_function.hypercosec import hypercosec
    from .hyperbolic_function.hypercotan import hypercotan
    from .exponential_function.power import power
    from .logarithmic_function.log import log
    # ln, exp, sqrt: 폴백에서 구현 없음 — 기존 코드와 호환
    ln = log  # 임시
    exp = _cmath.exp
    sqrt = _cmath.sqrt

# ------------------------------------------------------------------ 특수 함수
from .gamma_function.gamma import gamma
from .beta_function.beta import beta
from .bessel_function.bessel import bessel_j, bessel_j0, bessel_j1

try:
    from .legendre_function.legendre import legendre_polynomial as legendre
except ImportError:
    from .legendre_function.legendre import legendre_polynomial as legendre

from .lambert_w_function.lambert_w import lambert_w
from .zeta_function.zeta import zeta
from .euler_pi_function.euler_pi import euler_pi, euler_phi
from .heaviside_step_function.heaviside import heaviside
from .gcd.gcd import gcd
from .lcm.lcm import lcm

# ------------------------------------------------------------------ 미분
from .differentiation import Differentiation

__all__ = [
    # 상수
    "pi", "e", "epsilon",
    # 삼각
    "sin", "cos", "tan", "sec", "cosec", "cotan",
    # 역삼각
    "arcsin", "arccos", "arctan", "arcsec", "arccosec", "arccotan",
    # 쌍곡
    "hypersin", "hypercos", "hypertan", "hypersec", "hypercosec", "hypercotan",
    # 지수/로그
    "exp", "ln", "log", "power", "sqrt",
    # 특수
    "gamma", "beta",
    "bessel_j", "bessel_j0", "bessel_j1",
    "legendre",
    "lambert_w", "zeta",
    "euler_pi", "euler_phi",
    "heaviside",
    "gcd", "lcm",
    # 미분
    "Differentiation",
]
