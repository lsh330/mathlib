# math_library/linear_algebra/__init__.py
# LinearAlgebra 클래스 공개 API

try:
    from .linear_algebra import LinearAlgebra
except ImportError as _e:
    import warnings
    warnings.warn(
        f"math_library.linear_algebra Cython 모듈 로드 실패: {_e}\n"
        "빌드 후 재시도: python setup.py build_ext --inplace --compiler=mingw32",
        ImportWarning,
        stacklevel=2,
    )
    raise

__all__ = ["LinearAlgebra"]
