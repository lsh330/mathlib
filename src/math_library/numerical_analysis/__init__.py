# math_library/numerical_analysis/__init__.py
# NumericalAnalysis 클래스 공개 API

try:
    from .numerical_analysis import NumericalAnalysis
except ImportError as _e:
    import warnings
    warnings.warn(
        f"math_library.numerical_analysis Cython 모듈 로드 실패: {_e}\n"
        "빌드 후 재시도: python setup.py build_ext --inplace --compiler=mingw32",
        ImportWarning,
        stacklevel=2,
    )
    raise

__all__ = ["NumericalAnalysis"]
