try:
    from .differentiation import Differentiation
except ImportError:
    # Cython 빌드 전 폴백: 기존 Python 구현
    from .Differentiation import Differentiation

__all__ = ["Differentiation"]
