# math_library/laplace/__init__.py
# Phase B/C: AST + Laplace 변환 엔진 공개 API

try:
    from .laplace_ast import (
        PyExpr,
        symbol, const, rational,
        t, s, ZERO, ONE,
        Sin, Cos, Tan,
        Arcsin, Arccos, Arctan,
        Sinh, Cosh, Tanh,
        Exp, Ln, Log, Sqrt,
        Heaviside, Dirac, H,
        ExprPoolStats,
    )

    # ExprPoolStats를 ExprPool 이름으로도 접근 가능 (벤치마크 스크립트 호환)
    ExprPool = ExprPoolStats

except ImportError as _e:
    import warnings
    warnings.warn(
        f"math_library.laplace C++ 백엔드 로드 실패: {_e}\n"
        "빌드 후 재시도: python setup.py build_ext --inplace --compiler=mingw32",
        ImportWarning,
        stacklevel=2,
    )

try:
    from .laplace import Laplace

except ImportError as _e:
    import warnings
    warnings.warn(
        f"math_library.laplace 변환 엔진 로드 실패: {_e}\n"
        "빌드 후 재시도: python setup.py build_ext --inplace --compiler=mingw32",
        ImportWarning,
        stacklevel=2,
    )
    Laplace = None

# Phase C: lambdify (순수 Python, 별도 임포트 없이 PyExpr.lambdify()로 접근)
from .lambdify import lambdify

__all__ = [
    'PyExpr',
    'symbol', 'const', 'rational',
    't', 's', 'ZERO', 'ONE',
    'Sin', 'Cos', 'Tan',
    'Arcsin', 'Arccos', 'Arctan',
    'Sinh', 'Cosh', 'Tanh',
    'Exp', 'Ln', 'Log', 'Sqrt',
    'Heaviside', 'Dirac', 'H',
    'ExprPoolStats', 'ExprPool',
    'Laplace',
    'lambdify',
]
