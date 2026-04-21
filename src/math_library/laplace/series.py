# series.py — Phase D: Taylor/Maclaurin 급수 전개
# TaylorSeries: LazyDerivative + 계수 캐싱 + AST 재구성

from __future__ import annotations
from typing import List, Optional
import math


def _factorial(n: int) -> float:
    """양정수 n 의 팩토리얼 (float)."""
    if n < 0:
        raise ValueError(f"factorial: n={n} < 0")
    r = 1
    for i in range(2, n + 1):
        r *= i
    return float(r)


class TaylorSeries:
    """
    PyExpr 에 대한 Taylor/Maclaurin 급수.

    compute()  → 계수 리스트 [c_0, c_1, ..., c_order]
    as_expr()  → PyExpr (다항식 AST)
    evalf(**subs) → float (급수로 수치 평가)

    Parameters
    ----------
    expr     : PyExpr   — 전개할 식
    var_name : str      — 전개 변수 이름 ('s', 't', ...)
    about    : float    — 전개점 (기본 0.0)
    order    : int      — 최고 차수 (기본 6)
    """

    def __init__(self, expr, var_name: str, about: float = 0.0,
                 order: int = 6) -> None:
        self._expr = expr
        self._var = str(var_name)
        self._about = float(about)
        self._order = int(order)
        self._coeffs: Optional[List[float]] = None

    def compute(self) -> List[float]:
        """
        Taylor 계수 c_k = f^(k)(about) / k! 계산.

        Returns
        -------
        list of float, 길이 = order + 1
        """
        if self._coeffs is not None:
            return self._coeffs

        from .numeric_diff import LazyDerivative

        lazy = LazyDerivative(self._expr, self._var)
        coeffs: List[float] = []
        subs = {self._var: self._about}

        for k in range(self._order + 1):
            if k == 0:
                c = float(self._expr.evalf(**subs))
            else:
                c = lazy.nth(k, **subs) / _factorial(k)
            coeffs.append(c)

        self._coeffs = coeffs
        return coeffs

    def as_expr(self):
        """
        계수로부터 PyExpr 다항식 재구성.

        Returns
        -------
        PyExpr  Σ c_k * (var - about)^k
        """
        from math_library.laplace import symbol, const

        var_expr  = symbol(self._var)
        about_cst = const(self._about)
        shift = var_expr - about_cst  # (var - about)

        coeffs = self.compute()
        result = None
        for k, c in enumerate(coeffs):
            if abs(c) < 1e-18:
                continue  # 0 항 스킵
            c_expr = const(c)
            if k == 0:
                term = c_expr
            elif k == 1:
                term = c_expr * shift
            else:
                term = c_expr * (shift ** k)
            result = term if result is None else (result + term)

        if result is None:
            from math_library.laplace import const as _c
            return _c(0.0)
        return result

    def evalf(self, **subs) -> float:
        """
        급수로 수치 평가: Σ c_k * (x - about)^k

        Parameters
        ----------
        **subs : {var_name: float}

        Returns
        -------
        float
        """
        x = float(subs[self._var])
        dx = x - self._about
        return sum(c * (dx ** k) for k, c in enumerate(self.compute()))

    def __repr__(self) -> str:
        return (f"TaylorSeries(var='{self._var}', about={self._about}, "
                f"order={self._order})")
