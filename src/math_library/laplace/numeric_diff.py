# numeric_diff.py — Phase D: Differentiation 엔진 연동 수치 도함수
# LazyDerivative: lambdify + Ridders 수치 미분 조합

from __future__ import annotations
from typing import Callable


class LazyDerivative:
    """
    PyExpr + Differentiation(Ridders) 연동 수치 도함수 객체.

    evalf(**subs)      : 1계 도함수 수치값
    nth(n, **subs)     : n계 도함수 수치값
    __call__(**subs)   : evalf와 동일

    Parameters
    ----------
    expr     : PyExpr   — 미분할 식
    var_name : str      — 미분 변수 이름 ('s', 't', ...)
    """

    def __init__(self, expr, var_name: str) -> None:
        self._expr = expr
        self._var_name = str(var_name)
        # lambdify 캐시: 변수에 대한 Python callable
        # backend='math' — 실수 입력에 적합, 속도 최적
        self._f: Callable = expr.lambdify([self._var_name], backend='math')
        # Differentiation 엔진 (지연 초기화)
        self._diff_engine = None

    def _get_engine(self):
        if self._diff_engine is None:
            from math_library.differentiation import Differentiation
            self._diff_engine = Differentiation(tol=1e-8, initial_h=1e-3)
        return self._diff_engine

    def evalf(self, **subs) -> float:
        """
        1계 도함수 수치값.

        Parameters
        ----------
        **subs : {var_name: float}

        Returns
        -------
        float
        """
        x = float(subs[self._var_name])
        engine = self._get_engine()
        return engine.single_variable(self._f, x)

    def nth(self, n: int, **subs) -> float:
        """
        n계 도함수 수치값.

        Parameters
        ----------
        n      : int   — 미분 차수 (1 이상)
        **subs : {var_name: float}

        Returns
        -------
        float
        """
        x = float(subs[self._var_name])
        engine = self._get_engine()
        return engine.nth_derivative(self._f, x, order=int(n))

    def __call__(self, **subs) -> float:
        """evalf 동일."""
        return self.evalf(**subs)
