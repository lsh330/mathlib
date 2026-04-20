try:
    from .gamma import gamma
except ImportError:
    from .gamma import gamma  # Python fallback (same file name)

__all__ = ["gamma"]
