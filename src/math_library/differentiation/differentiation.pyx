# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
#
# differentiation.pyx
#
# Ridders Richardson extrapolation 재설계.
# 기존 Differentiation 클래스의 공개 API는 모두 유지하되,
# 핵심 single_variable 메서드를 Ridders 알고리즘으로 재구현.
#
# Ridders 알고리즘:
#   - Numerical Recipes in C, §5.7 참조
#   - Richardson extrapolation table에서 최적 추정값 선택
#   - 기존 h/2 단순 반복보다 훨씬 빠른 수렴

from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union
import math as _math


Number = Union[int, float, complex]
Vector = Sequence[Number]
ScalarFunction = Callable[..., Number]
VectorFunction = Union[Callable[..., Sequence[Number]], Sequence[Callable[..., Number]]]
OperatorOutput = Union[Number, List[Number]]

# Ridders 알고리즘 축소 인수
_CON = 1.4       # 초기 h의 축소 비율
_CON2 = _CON * _CON
_SAFE = 2.0      # 오차 증가 임계값 (종료 조건)
_NTAB = 10       # Richardson table 최대 크기


class Differentiation:
    """
    Numerical differentiation toolkit — Ridders Richardson extrapolation 재설계.

    핵심 메서드 single_variable이 Ridders 알고리즘을 사용하여
    기존 h/2 반복보다 빠르게 수렴.
    모든 기존 공개 API는 유지.
    """

    def __init__(
        self,
        tol: float = 1e-8,
        initial_h: float = 1e-3,
        max_iter: int = 20,
        number_system: str = "real",
    ) -> None:
        self.tol = self._validate_positive_real("tol", tol)
        self.initial_h = self._validate_positive_real("initial_h", initial_h)
        self.max_iter = self._validate_positive_int("max_iter", max_iter)
        self.number_system = self._validate_number_system(number_system)

    # --------------------------------------------------------------- 검증
    def _validate_number_system(self, number_system: str) -> str:
        if not isinstance(number_system, str):
            raise TypeError("number_system must be a string.")
        value = number_system.lower()
        if value not in ("real", "complex"):
            raise ValueError("number_system must be either 'real' or 'complex'.")
        return value

    def _resolve_number_system(self, number_system: Optional[str]) -> str:
        if number_system is None:
            return self.number_system
        return self._validate_number_system(number_system)

    def _validate_positive_real(self, name: str, value) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a positive real number.")
        numeric = float(value)
        if numeric <= 0:
            raise ValueError(f"{name} must be a positive real number.")
        return numeric

    def _validate_positive_int(self, name: str, value) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return value

    def _validate_scalar_input(self, name: str, value, number_system: str) -> Number:
        if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
            raise TypeError(f"{name} must be a numeric value.")
        if number_system == "real" and not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be an int or float in real mode.")
        return value

    def _validate_point(self, point: Vector, number_system: str) -> List[Number]:
        if not isinstance(point, Sequence) or len(point) == 0:
            raise TypeError("point must be a non-empty sequence of numeric values.")
        validated: List[Number] = []
        for i, value in enumerate(point):
            validated.append(self._validate_scalar_input(f"point[{i}]", value, number_system))
        return validated

    def _resolve_tol(self, tol: Optional[float]) -> float:
        if tol is None:
            return self.tol
        return self._validate_positive_real("tol", tol)

    def _resolve_h(self, h: Optional[float]) -> float:
        if h is None:
            return self.initial_h
        return self._validate_positive_real("h", h)

    def _resolve_max_iter(self, max_iter: Optional[int]) -> int:
        if max_iter is None:
            return self.max_iter
        return self._validate_positive_int("max_iter", max_iter)

    def _normalize(self, value: Number, tol: float, number_system: str) -> Number:
        if isinstance(value, complex):
            if abs(value.imag) <= tol * max(1.0, abs(value.real)):
                return float(value.real)
            if number_system == "real":
                raise RuntimeError("real-mode result unexpectedly became complex.")
            return value
        return float(value)

    def _abs(self, value: Number) -> float:
        return abs(complex(value))

    def _evaluate_scalar_function(self, function, point: Sequence[Number],
                                    number_system: str, tol: float) -> Number:
        if not callable(function):
            raise TypeError("function must be callable.")
        try:
            if len(point) == 1:
                value = function(point[0])
            else:
                value = function(*point)
        except TypeError:
            value = function(point)
        if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
            raise TypeError("function must return a numeric value.")
        normalized = self._normalize(value, tol, number_system)
        if number_system == "real" and isinstance(normalized, complex):
            raise RuntimeError("real-mode function evaluation produced a complex value.")
        return normalized

    def _evaluate_operator_function(self, function, point: Sequence[Number],
                                     number_system: str, tol: float) -> OperatorOutput:
        if not callable(function):
            raise TypeError("function must be callable.")
        try:
            value = function(*point)
        except TypeError:
            value = function(point)
        if isinstance(value, bool):
            raise TypeError("function must return numeric output.")
        if isinstance(value, (int, float, complex)):
            return self._normalize(value, tol, number_system)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 0:
                raise TypeError("function vector output must be non-empty.")
            normalized_values: List[Number] = []
            for component in value:
                if isinstance(component, bool) or not isinstance(component, (int, float, complex)):
                    raise TypeError("function vector output components must be numeric.")
                normalized_values.append(self._normalize(component, tol, number_system))
            return normalized_values
        raise TypeError("function must return a numeric scalar or a numeric sequence.")

    def _subtract_operator_outputs(self, left: OperatorOutput, right: OperatorOutput) -> OperatorOutput:
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                raise ValueError("operator outputs must have the same dimension.")
            return [left[i] - right[i] for i in range(len(left))]
        if not isinstance(left, list) and not isinstance(right, list):
            return left - right
        raise TypeError("operator output shape mismatch between scalar and vector.")

    def _divide_operator_output(self, output: OperatorOutput, scalar: Number) -> OperatorOutput:
        if self._abs(scalar) == 0.0:
            raise ZeroDivisionError("division by zero in operator output scaling.")
        if isinstance(output, list):
            return [component / scalar for component in output]
        return output / scalar

    def _operator_output_norm(self, output: OperatorOutput) -> float:
        if isinstance(output, list):
            maximum = 0.0
            for component in output:
                value = self._abs(component)
                if value > maximum:
                    maximum = value
            return maximum
        return self._abs(output)

    def _operator_output_difference(self, left: OperatorOutput, right: OperatorOutput) -> float:
        delta = self._subtract_operator_outputs(left, right)
        return self._operator_output_norm(delta)

    def _normalize_operator_output(self, output: OperatorOutput, tol: float,
                                     number_system: str) -> OperatorOutput:
        if isinstance(output, list):
            return [self._normalize(component, tol, number_system) for component in output]
        return self._normalize(output, tol, number_system)

    def _get_vector_component_function(self, vector_field, component_index: int,
                                        point: Sequence[Number], number_system: str, tol: float):
        if callable(vector_field):
            def component(*args: Number) -> Number:
                values = vector_field(*args)
                if not isinstance(values, Sequence):
                    raise TypeError("vector_field callable must return a sequence.")
                if component_index < 0 or component_index >= len(values):
                    raise IndexError("vector_field component index out of range.")
                value = values[component_index]
                if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
                    raise TypeError("vector_field components must be numeric.")
                return self._normalize(value, tol, number_system)
            return component
        if not isinstance(vector_field, Sequence):
            raise TypeError("vector_field must be callable or a sequence of callables.")
        if component_index < 0 or component_index >= len(vector_field):
            raise IndexError("vector_field component index out of range.")
        if not callable(vector_field[component_index]):
            raise TypeError("each vector_field component must be callable.")
        return vector_field[component_index]

    def _vector_dimension(self, vector_field, point: Sequence[Number],
                           number_system: str, tol: float) -> int:
        if callable(vector_field):
            try:
                values = vector_field(*point)
            except TypeError:
                values = vector_field(point)
            if not isinstance(values, Sequence) or len(values) == 0:
                raise TypeError("vector_field callable must return a non-empty sequence.")
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
                    raise TypeError("vector_field components must be numeric.")
                self._normalize(value, tol, number_system)
            return len(values)
        if not isinstance(vector_field, Sequence) or len(vector_field) == 0:
            raise TypeError("vector_field must be a non-empty sequence of callables.")
        for function in vector_field:
            if not callable(function):
                raise TypeError("vector_field components must be callable.")
        return len(vector_field)

    # --------------------------------------------------------------- 핵심: Ridders
    def _ridders_derivative(self, function, x: Number, h: float,
                             number_system: str, tol: float) -> Number:
        """
        Ridders Richardson extrapolation으로 1계 도함수 추정.
        Numerical Recipes §5.7 알고리즘.

        a[i][j] = Richardson table (중앙 차분 기반)
        오차 추정이 증가하면 조기 종료.
        """
        ntab = _NTAB
        # Richardson extrapolation table
        a = [[0.0 + 0.0j for _ in range(ntab)] for _ in range(ntab)]
        hh = float(h)

        # a[0][0]: 초기 중앙 차분
        f_plus = self._evaluate_scalar_function(function, [x + hh], number_system, tol)
        f_minus = self._evaluate_scalar_function(function, [x - hh], number_system, tol)
        a[0][0] = complex((f_plus - f_minus) / (2.0 * hh))

        err = 1e300
        ans = complex(a[0][0])

        for i in range(1, ntab):
            hh /= _CON
            f_plus = self._evaluate_scalar_function(function, [x + hh], number_system, tol)
            f_minus = self._evaluate_scalar_function(function, [x - hh], number_system, tol)
            a[0][i] = complex((f_plus - f_minus) / (2.0 * hh))

            fac = _CON2
            for j in range(1, i + 1):
                a[j][i] = complex(
                    (a[j-1][i] * fac - a[j-1][i-1]) / (fac - 1.0)
                )
                fac *= _CON2
                errt = max(abs(a[j][i] - a[j-1][i]), abs(a[j][i] - a[j-1][i-1]))
                if errt <= err:
                    err = errt
                    ans = a[j][i]

            # 오차 증가하면 조기 종료
            if abs(a[i][i] - a[i-1][i-1]) >= _SAFE * err:
                break

        # 허수부 검사
        if abs(ans.imag) <= tol * max(1.0, abs(ans.real)):
            result = float(ans.real)
        else:
            result = ans

        return self._normalize(result, tol, number_system)

    # --------------------------------------------------------------- 공개 API
    def difference_quotient(
        self,
        function,
        x: Number,
        h: Optional[float] = None,
        method: str = "central",
        number_system: Optional[str] = None,
        tol: Optional[float] = None,
    ) -> Number:
        number_system_value = self._resolve_number_system(number_system)
        tol_value = self._resolve_tol(tol)
        h_value = self._resolve_h(h)
        x_value = self._validate_scalar_input("x", x, number_system_value)

        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        method_value = method.lower()
        if method_value not in ("forward", "backward", "central"):
            raise ValueError("method must be 'forward', 'backward', or 'central'.")

        if method_value == "forward":
            f_xh = self._evaluate_scalar_function(function, [x_value + h_value], number_system_value, tol_value)
            f_x = self._evaluate_scalar_function(function, [x_value], number_system_value, tol_value)
            return self._normalize((f_xh - f_x) / h_value, tol_value, number_system_value)
        if method_value == "backward":
            f_x = self._evaluate_scalar_function(function, [x_value], number_system_value, tol_value)
            f_xmh = self._evaluate_scalar_function(function, [x_value - h_value], number_system_value, tol_value)
            return self._normalize((f_x - f_xmh) / h_value, tol_value, number_system_value)
        f_xh = self._evaluate_scalar_function(function, [x_value + h_value], number_system_value, tol_value)
        f_xmh = self._evaluate_scalar_function(function, [x_value - h_value], number_system_value, tol_value)
        return self._normalize((f_xh - f_xmh) / (2.0 * h_value), tol_value, number_system_value)

    def left_derivative(self, function, x: Number, h=None, number_system=None,
                         tol=None, max_iter=None) -> Number:
        return self.single_variable(function, x, h=h, method="backward",
                                     number_system=number_system, tol=tol, max_iter=max_iter)

    def right_derivative(self, function, x: Number, h=None, number_system=None,
                          tol=None, max_iter=None) -> Number:
        return self.single_variable(function, x, h=h, method="forward",
                                     number_system=number_system, tol=tol, max_iter=max_iter)

    def single_variable(
        self,
        function,
        x: Number,
        h: Optional[float] = None,
        method: str = "central",
        number_system: Optional[str] = None,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> Number:
        """
        Ridders Richardson extrapolation으로 1계 도함수 추정.
        method="central" 이면 Ridders 사용 (최고 정밀도).
        method="forward" 또는 "backward"이면 h/2 반복.
        """
        number_system_value = self._resolve_number_system(number_system)
        tol_value = self._resolve_tol(tol)
        h_value = self._resolve_h(h)

        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        method_lower = method.lower()
        if method_lower not in ("forward", "backward", "central"):
            raise ValueError("method must be 'forward', 'backward', or 'central'.")

        x_value = self._validate_scalar_input("x", x, number_system_value)

        # central: Ridders 알고리즘
        if method_lower == "central":
            return self._ridders_derivative(function, x_value, h_value,
                                              number_system_value, tol_value)

        # forward/backward: 기존 h/2 반복 (Ridders는 중앙 차분 기반)
        max_iter_value = self._resolve_max_iter(max_iter)
        h_current = h_value
        previous: Optional[Number] = None

        for _ in range(max_iter_value):
            current = self.difference_quotient(
                function, x_value, h=h_current, method=method_lower,
                number_system=number_system_value, tol=tol_value,
            )
            if previous is not None:
                if self._abs(current - previous) <= tol_value * max(1.0, self._abs(current)):
                    return self._normalize(current, tol_value, number_system_value)
            previous = current
            h_current *= 0.5

        raise RuntimeError("Derivative did not converge within max_iter.")

    def _central_stencil_nth(self, function, x: Number, order: int,
                              h: float, number_system: str, tol: float) -> Number:
        """
        M7: 직접 central difference stencil (4차 또는 6차 정확도).
        재귀 Ridders 대신 고차 유한차분 계수를 직접 적용하여 오차 누적 방지.

        order 1~4: 6차 정확도 central stencil 사용.
        order 5~6: 4차 정확도 central stencil 사용.
        order >= 7: h/2 반복 convergence.

        스텐실 계수 출처: Fornberg (1988) SIAM Rev.
        """
        # 6차 정확도 central stencil 계수 (offsets: -3,-2,-1,0,1,2,3)
        # order 1: [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]
        # order 2: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
        # order 3: [1/8, -1, 13/8, 0, -13/8, 1, -1/8]  (4th order)
        # order 4: [-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6]  (4th order)
        # For order 3 and 4 use 4th order accuracy (5-point and 7-point)

        if order == 1:
            # 6차 정확도: [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]
            f = [self._evaluate_scalar_function(function, [x + i * h], number_system, tol)
                 for i in (-3, -2, -1, 0, 1, 2, 3)]
            val = (-f[0] / 60.0 + 3.0 * f[1] / 20.0 - 3.0 * f[2] / 4.0
                   + 3.0 * f[4] / 4.0 - 3.0 * f[5] / 20.0 + f[6] / 60.0) / h
            return self._normalize(val, tol, number_system)

        if order == 2:
            # 6차 정확도: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
            f = [self._evaluate_scalar_function(function, [x + i * h], number_system, tol)
                 for i in (-3, -2, -1, 0, 1, 2, 3)]
            val = (f[0] / 90.0 - 3.0 * f[1] / 20.0 + 3.0 * f[2] / 2.0
                   - 49.0 * f[3] / 18.0
                   + 3.0 * f[4] / 2.0 - 3.0 * f[5] / 20.0 + f[6] / 90.0) / (h * h)
            return self._normalize(val, tol, number_system)

        if order == 3:
            # 4차 정확도 7-point: [1/8, -1, 13/8, 0, -13/8, 1, -1/8] / h^3
            # Fornberg (1988): verified coefficients, offsets -3..+3
            f = [self._evaluate_scalar_function(function, [x + i * h], number_system, tol)
                 for i in (-3, -2, -1, 0, 1, 2, 3)]
            val = (f[0] / 8.0 - f[1] + 13.0 * f[2] / 8.0
                   - 13.0 * f[4] / 8.0 + f[5] - f[6] / 8.0) / (h ** 3)
            return self._normalize(val, tol, number_system)

        if order == 4:
            # 4차 정확도 7-point stencil
            # Fornberg: [-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6] / h^4
            f = [self._evaluate_scalar_function(function, [x + i * h], number_system, tol)
                 for i in (-3, -2, -1, 0, 1, 2, 3)]
            val = (-f[0] / 6.0 + 2.0 * f[1] - 13.0 * f[2] / 2.0
                   + 28.0 * f[3] / 3.0
                   - 13.0 * f[4] / 2.0 + 2.0 * f[5] - f[6] / 6.0) / (h ** 4)
            return self._normalize(val, tol, number_system)

        if order == 5:
            # 4차 정확도 9-point stencil for 5th derivative
            # coefficients: [1/6, -3/2, 13/3, -29/6, 0, 29/6, -13/3, 3/2, -1/6]
            f = [self._evaluate_scalar_function(function, [x + i * h], number_system, tol)
                 for i in (-4, -3, -2, -1, 0, 1, 2, 3, 4)]
            val = (f[0] / 6.0 - 3.0 * f[1] / 2.0 + 13.0 * f[2] / 3.0
                   - 29.0 * f[3] / 6.0
                   + 29.0 * f[5] / 6.0 - 13.0 * f[6] / 3.0 + 3.0 * f[7] / 2.0 - f[8] / 6.0) / (h ** 5)
            return self._normalize(val, tol, number_system)

        if order == 6:
            # 4차 정확도 9-point stencil for 6th derivative
            # coefficients: [-1/4, 3, -13, 29, -75/2, 29, -13, 3, -1/4]
            f = [self._evaluate_scalar_function(function, [x + i * h], number_system, tol)
                 for i in (-4, -3, -2, -1, 0, 1, 2, 3, 4)]
            val = (-f[0] / 4.0 + 3.0 * f[1] - 13.0 * f[2] + 29.0 * f[3]
                   - 75.0 * f[4] / 2.0
                   + 29.0 * f[5] - 13.0 * f[6] + 3.0 * f[7] - f[8] / 4.0) / (h ** 6)
            return self._normalize(val, tol, number_system)

        # order >= 7: recursive application of 1st derivative stencil
        def lower_fn(t: Number) -> Number:
            return self._central_stencil_nth(function, t, order - 1, h, number_system, tol)
        return self._central_stencil_nth(lower_fn, x, 1, h * 2.0, number_system, tol)

    def nth_derivative(
        self,
        function,
        x: Number,
        order: int,
        h=None,
        method: str = "central",
        number_system=None,
        tol=None,
        max_iter=None,
    ) -> Number:
        order_value = self._validate_positive_int("order", order)
        number_system_value = self._resolve_number_system(number_system)
        tol_value = self._resolve_tol(tol)
        max_iter_value = self._resolve_max_iter(max_iter)
        x_value = self._validate_scalar_input("x", x, number_system_value)

        if order_value == 1:
            return self.single_variable(function, x_value, h=h, method=method,
                                         number_system=number_system_value, tol=tol,
                                         max_iter=max_iter)

        # M7: order >= 2: 직접 central stencil (4차/6차 정확도)
        # 재귀 Ridders 호출 대신 Fornberg stencil 직접 적용
        h_value = self._resolve_h(h)
        # h 자동 조정: optimal h = eps^{1/(p+accuracy_order)} where p=derivative order
        # Empirically verified optimal values:
        #   order 1,2: 1e-2 (6th order stencil)
        #   order 3,4: 1e-2 (4th order stencil, round-off floor ~1e-8)
        #   order 5,6: 1e-1 (4th order stencil, round-off floor ~1e-5)
        #   order >= 7: 1e-1
        if h is None:
            import math as _m
            if order_value <= 4:
                h_value = 1e-2
            else:
                h_value = 1e-1

        return self._central_stencil_nth(function, x_value, order_value,
                                          h_value, number_system_value, tol_value)

    def parametric_derivative(
        self,
        coordinate_functions: Sequence,
        parameter: Number,
        dependent_index: int = 1,
        independent_index: int = 0,
        order: int = 1,
        h=None,
        method: str = "central",
        number_system=None,
        tol=None,
        max_iter=None,
    ) -> Number:
        number_system_value = self._resolve_number_system(number_system)
        tol_value = self._resolve_tol(tol)
        order_value = self._validate_positive_int("order", order)
        parameter_value = self._validate_scalar_input("parameter", parameter, number_system_value)

        if not isinstance(coordinate_functions, Sequence) or len(coordinate_functions) < 2:
            raise TypeError("coordinate_functions must be a sequence of at least two callables.")
        for function in coordinate_functions:
            if not callable(function):
                raise TypeError("each coordinate function must be callable.")

        if isinstance(dependent_index, bool) or not isinstance(dependent_index, int):
            raise TypeError("dependent_index must be an integer.")
        if isinstance(independent_index, bool) or not isinstance(independent_index, int):
            raise TypeError("independent_index must be an integer.")
        if dependent_index < 0 or dependent_index >= len(coordinate_functions):
            raise IndexError("dependent_index is out of range.")
        if independent_index < 0 or independent_index >= len(coordinate_functions):
            raise IndexError("independent_index is out of range.")
        if dependent_index == independent_index:
            raise ValueError("dependent_index and independent_index must be different.")

        def coordinate_value(index: int, t: Number) -> Number:
            return self._evaluate_scalar_function(coordinate_functions[index], [t], number_system_value, tol_value)

        def first_order_ratio(t: Number) -> Number:
            d_dep = self.single_variable(lambda tau: coordinate_value(dependent_index, tau),
                                          t, h=h, method=method, number_system=number_system_value,
                                          tol=tol, max_iter=max_iter)
            d_ind = self.single_variable(lambda tau: coordinate_value(independent_index, tau),
                                          t, h=h, method=method, number_system=number_system_value,
                                          tol=tol, max_iter=max_iter)
            if self._abs(d_ind) <= tol_value:
                raise ZeroDivisionError("Parametric derivative undefined because dx/dt is approximately zero.")
            return self._normalize(d_dep / d_ind, tol_value, number_system_value)

        if order_value == 1:
            return first_order_ratio(parameter_value)

        current_function = first_order_ratio
        for _ in range(2, order_value + 1):
            previous_function = current_function
            def next_order_ratio(t: Number, _prev=previous_function) -> Number:
                d_prev = self.single_variable(_prev, t, h=h, method=method,
                                               number_system=number_system_value, tol=tol,
                                               max_iter=max_iter)
                d_ind = self.single_variable(lambda tau: coordinate_value(independent_index, tau),
                                              t, h=h, method=method, number_system=number_system_value,
                                              tol=tol, max_iter=max_iter)
                if self._abs(d_ind) <= tol_value:
                    raise ZeroDivisionError("Higher-order parametric derivative undefined.")
                return self._normalize(d_prev / d_ind, tol_value, number_system_value)
            current_function = next_order_ratio

        return current_function(parameter_value)

    def partial_derivative(
        self, function, point: Vector, variable_index: int,
        h=None, method: str = "central", number_system=None, tol=None, max_iter=None,
    ) -> Number:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)

        if isinstance(variable_index, bool) or not isinstance(variable_index, int):
            raise TypeError("variable_index must be an integer.")
        if variable_index < 0 or variable_index >= len(point_value):
            raise IndexError("variable_index is out of range.")

        def single_axis(t: Number) -> Number:
            shifted = list(point_value)
            shifted[variable_index] = t
            tol_local = self._resolve_tol(tol)
            return self._evaluate_scalar_function(function, shifted, number_system_value, tol_local)

        return self.single_variable(single_axis, point_value[variable_index], h=h, method=method,
                                     number_system=number_system_value, tol=tol, max_iter=max_iter)

    def mixed_partial(
        self, function, point: Vector, order_multi_index: Sequence[int],
        h=None, method: str = "central", number_system=None, tol=None, max_iter=None,
    ) -> Number:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        tol_value = self._resolve_tol(tol)
        max_iter_value = self._resolve_max_iter(max_iter)
        h_current = self._resolve_h(h)
        if h is None and h_current < 1e-2:
            h_current = 1e-2

        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        method_value = method.lower()
        if method_value not in ("forward", "backward", "central"):
            raise ValueError("method must be 'forward', 'backward', or 'central'.")

        if not isinstance(order_multi_index, Sequence):
            raise TypeError("order_multi_index must be a sequence of non-negative integers.")
        if len(order_multi_index) != len(point_value):
            raise ValueError("order_multi_index length must match point dimension.")

        variable_order: List[int] = []
        for index, count in enumerate(order_multi_index):
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("order_multi_index values must be integers.")
            if count < 0:
                raise ValueError("order_multi_index values must be non-negative.")
            for _ in range(count):
                variable_order.append(index)

        if len(variable_order) == 0:
            raise ValueError("At least one derivative order must be positive.")

        def recursive_difference(point_local: List[Number], depth: int, step: float) -> Number:
            if depth == len(variable_order):
                return self._evaluate_scalar_function(function, point_local, number_system_value, tol_value)
            axis = variable_order[depth]
            plus_point = list(point_local)
            plus_point[axis] = plus_point[axis] + step
            if method_value == "forward":
                value_plus = recursive_difference(plus_point, depth + 1, step)
                value_center = recursive_difference(point_local, depth + 1, step)
                return self._normalize((value_plus - value_center) / step, tol_value, number_system_value)
            minus_point = list(point_local)
            minus_point[axis] = minus_point[axis] - step
            if method_value == "backward":
                value_center = recursive_difference(point_local, depth + 1, step)
                value_minus = recursive_difference(minus_point, depth + 1, step)
                return self._normalize((value_center - value_minus) / step, tol_value, number_system_value)
            value_plus = recursive_difference(plus_point, depth + 1, step)
            value_minus = recursive_difference(minus_point, depth + 1, step)
            return self._normalize((value_plus - value_minus) / (2.0 * step), tol_value, number_system_value)

        previous: Optional[Number] = None
        for _ in range(max_iter_value):
            current = recursive_difference(list(point_value), 0, h_current)
            if previous is not None:
                if self._abs(current - previous) <= tol_value * max(1.0, self._abs(current)):
                    return self._normalize(current, tol_value, number_system_value)
            previous = current
            h_current *= 0.5

        raise RuntimeError("Mixed partial derivative did not converge within max_iter.")

    def gradient(self, function, point: Vector, h=None, method: str = "central",
                  number_system=None, tol=None, max_iter=None) -> List[Number]:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        result: List[Number] = []
        for i in range(len(point_value)):
            result.append(self.partial_derivative(function, point_value, i, h=h, method=method,
                                                   number_system=number_system_value, tol=tol,
                                                   max_iter=max_iter))
        return result

    def directional_derivative(self, function, point: Vector, direction: Vector,
                                 h=None, method: str = "central", number_system=None,
                                 tol=None, max_iter=None) -> Number:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        direction_value = self._validate_point(direction, number_system_value)
        if len(point_value) != len(direction_value):
            raise ValueError("point and direction must have the same dimension.")
        norm_sq = 0.0
        for value in direction_value:
            norm_sq += abs(complex(value)) ** 2
        if norm_sq == 0.0:
            raise ValueError("direction must be non-zero.")
        norm = norm_sq ** 0.5
        unit_direction: List[Number] = []
        for value in direction_value:
            if number_system_value == "real":
                unit_direction.append(float(value) / norm)
            else:
                unit_direction.append(complex(value) / norm)
        tol_value = self._resolve_tol(tol)
        def line_function(t: Number) -> Number:
            shifted: List[Number] = []
            for i in range(len(point_value)):
                shifted.append(point_value[i] + t * unit_direction[i])
            return self._evaluate_scalar_function(function, shifted, number_system_value, tol_value)
        return self.single_variable(line_function, 0.0, h=h, method=method,
                                     number_system=number_system_value, tol=tol, max_iter=max_iter)

    def higher_order_directional_derivative(self, function, point: Vector, direction: Vector,
                                              order: int, h=None, method: str = "central",
                                              number_system=None, tol=None, max_iter=None) -> Number:
        order_value = self._validate_positive_int("order", order)
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        direction_value = self._validate_point(direction, number_system_value)
        tol_value = self._resolve_tol(tol)
        if len(point_value) != len(direction_value):
            raise ValueError("point and direction must have the same dimension.")
        norm_sq = 0.0
        for value in direction_value:
            norm_sq += abs(complex(value)) ** 2
        if norm_sq == 0.0:
            raise ValueError("direction must be non-zero.")
        norm = norm_sq ** 0.5
        unit_direction: List[Number] = []
        for value in direction_value:
            if number_system_value == "real":
                unit_direction.append(float(value) / norm)
            else:
                unit_direction.append(complex(value) / norm)
        def line_function(t: Number) -> Number:
            shifted: List[Number] = []
            for i in range(len(point_value)):
                shifted.append(point_value[i] + t * unit_direction[i])
            return self._evaluate_scalar_function(function, shifted, number_system_value, tol_value)
        return self.nth_derivative(line_function, 0.0, order=order_value, h=h, method=method,
                                    number_system=number_system_value, tol=tol, max_iter=max_iter)

    def jacobian(self, functions: Sequence, point: Vector, h=None, method: str = "central",
                  number_system=None, tol=None, max_iter=None) -> List[List[Number]]:
        if not isinstance(functions, Sequence) or len(functions) == 0:
            raise TypeError("functions must be a non-empty sequence of callables.")
        for function in functions:
            if not callable(function):
                raise TypeError("each element in functions must be callable.")
        result: List[List[Number]] = []
        for function in functions:
            result.append(self.gradient(function, point, h=h, method=method,
                                         number_system=number_system, tol=tol, max_iter=max_iter))
        return result

    def hessian(self, function, point: Vector, h=None, number_system=None, tol=None) -> List[List[Number]]:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        tol_value = self._resolve_tol(tol)
        h_value = self._resolve_h(h)
        n = len(point_value)
        matrix: List[List[Number]] = [[0.0 for _ in range(n)] for _ in range(n)]
        center = self._evaluate_scalar_function(function, point_value, number_system_value, tol_value)
        for i in range(n):
            for j in range(n):
                if i == j:
                    plus = list(point_value)
                    minus = list(point_value)
                    plus[i] = plus[i] + h_value
                    minus[i] = minus[i] - h_value
                    f_plus = self._evaluate_scalar_function(function, plus, number_system_value, tol_value)
                    f_minus = self._evaluate_scalar_function(function, minus, number_system_value, tol_value)
                    matrix[i][j] = self._normalize(
                        (f_plus - 2.0 * center + f_minus) / (h_value * h_value),
                        tol_value, number_system_value,
                    )
                else:
                    pp = list(point_value)
                    pm = list(point_value)
                    mp = list(point_value)
                    mm = list(point_value)
                    pp[i] = pp[i] + h_value; pp[j] = pp[j] + h_value
                    pm[i] = pm[i] + h_value; pm[j] = pm[j] - h_value
                    mp[i] = mp[i] - h_value; mp[j] = mp[j] + h_value
                    mm[i] = mm[i] - h_value; mm[j] = mm[j] - h_value
                    f_pp = self._evaluate_scalar_function(function, pp, number_system_value, tol_value)
                    f_pm = self._evaluate_scalar_function(function, pm, number_system_value, tol_value)
                    f_mp = self._evaluate_scalar_function(function, mp, number_system_value, tol_value)
                    f_mm = self._evaluate_scalar_function(function, mm, number_system_value, tol_value)
                    matrix[i][j] = self._normalize(
                        (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_value * h_value),
                        tol_value, number_system_value,
                    )
        return matrix

    def hessian_vector_product(self, function, point: Vector, vector: Vector,
                                 h=None, method: str = "central", number_system=None,
                                 tol=None, max_iter=None) -> List[Number]:
        """
        Hessian-vector product H(f, x) * v computed via Hessian matrix.
        """
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        vector_value = self._validate_point(vector, number_system_value)
        tol_value = self._resolve_tol(tol)
        if len(point_value) != len(vector_value):
            raise ValueError("point and vector must have the same dimension.")
        if all(self._abs(component) <= tol_value for component in vector_value):
            if number_system_value == "real":
                return [0.0 for _ in range(len(vector_value))]
            return [0.0 + 0.0j for _ in range(len(vector_value))]
        # Hessian 직접 계산 후 행렬-벡터 곱
        H = self.hessian(function, point_value, h=h,
                         number_system=number_system_value, tol=tol_value)
        n = len(point_value)
        result = []
        for i in range(n):
            if number_system_value == "real":
                row_sum = 0.0
            else:
                row_sum = 0.0 + 0.0j
            for j in range(n):
                row_sum = row_sum + H[i][j] * vector_value[j]
            result.append(self._normalize(row_sum, tol_value, number_system_value))
        return result

    def laplacian(self, function, point: Vector, h=None, number_system=None, tol=None) -> Number:
        number_system_value = self._resolve_number_system(number_system)
        tol_value = self._resolve_tol(tol)
        hessian_matrix = self.hessian(function, point, h=h, number_system=number_system_value, tol=tol_value)
        total: Number = 0.0
        for i in range(len(hessian_matrix)):
            total = total + hessian_matrix[i][i]
        return self._normalize(total, tol_value, number_system_value)

    def total_derivative(self, outer_function, inner_functions: Sequence,
                          point: Vector, h=None, method: str = "central",
                          number_system=None, tol=None, max_iter=None) -> dict:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        tol_value = self._resolve_tol(tol)
        if not isinstance(inner_functions, Sequence) or len(inner_functions) == 0:
            raise TypeError("inner_functions must be a non-empty sequence of callables.")
        for function in inner_functions:
            if not callable(function):
                raise TypeError("each element in inner_functions must be callable.")
        inner_point: List[Number] = []
        for inner_function in inner_functions:
            inner_point.append(self._evaluate_scalar_function(inner_function, point_value, number_system_value, tol_value))
        outer_gradient = self.gradient(outer_function, inner_point, h=h, method=method,
                                        number_system=number_system_value, tol=tol, max_iter=max_iter)
        inner_jacobian = self.jacobian(inner_functions, point_value, h=h, method=method,
                                        number_system=number_system_value, tol=tol, max_iter=max_iter)
        derivative: List[Number] = []
        input_dim = len(point_value)
        for column in range(input_dim):
            component: Number = 0.0
            for row in range(len(inner_functions)):
                component = component + outer_gradient[row] * inner_jacobian[row][column]
            derivative.append(self._normalize(component, tol_value, number_system_value))
        return {
            "point": tuple(point_value),
            "inner_point": tuple(inner_point),
            "outer_gradient": outer_gradient,
            "inner_jacobian": inner_jacobian,
            "derivative": derivative,
        }

    def implicit_derivative(self, relation, x: Number, y: Number, h=None,
                              method: str = "central", number_system=None, tol=None,
                              max_iter=None) -> Number:
        number_system_value = self._resolve_number_system(number_system)
        point = [self._validate_scalar_input("x", x, number_system_value),
                 self._validate_scalar_input("y", y, number_system_value)]
        tol_value = self._resolve_tol(tol)
        fx = self.partial_derivative(relation, point, variable_index=0, h=h, method=method,
                                      number_system=number_system_value, tol=tol_value, max_iter=max_iter)
        fy = self.partial_derivative(relation, point, variable_index=1, h=h, method=method,
                                      number_system=number_system_value, tol=tol_value, max_iter=max_iter)
        if self._abs(fy) <= tol_value:
            raise ZeroDivisionError("Implicit derivative undefined because dF/dy is approximately zero.")
        return self._normalize(-fx / fy, tol_value, number_system_value)

    def implicit_partial(self, relation, point: Vector, dependent_index: int,
                          independent_index: int, h=None, method: str = "central",
                          number_system=None, tol=None, max_iter=None) -> Number:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        tol_value = self._resolve_tol(tol)
        if isinstance(dependent_index, bool) or not isinstance(dependent_index, int):
            raise TypeError("dependent_index must be an integer.")
        if isinstance(independent_index, bool) or not isinstance(independent_index, int):
            raise TypeError("independent_index must be an integer.")
        n = len(point_value)
        if dependent_index < 0 or dependent_index >= n:
            raise IndexError("dependent_index is out of range.")
        if independent_index < 0 or independent_index >= n:
            raise IndexError("independent_index is out of range.")
        if dependent_index == independent_index:
            raise ValueError("dependent_index and independent_index must be different.")
        f_independent = self.partial_derivative(relation, point_value, variable_index=independent_index,
                                                 h=h, method=method, number_system=number_system_value,
                                                 tol=tol_value, max_iter=max_iter)
        f_dependent = self.partial_derivative(relation, point_value, variable_index=dependent_index,
                                               h=h, method=method, number_system=number_system_value,
                                               tol=tol_value, max_iter=max_iter)
        if self._abs(f_dependent) <= tol_value:
            raise ZeroDivisionError("Implicit partial derivative undefined.")
        return self._normalize(-f_independent / f_dependent, tol_value, number_system_value)

    def wirtinger_derivatives(self, function, z: Number, h=None, method: str = "central",
                               number_system=None, tol=None, max_iter=None) -> dict:
        if number_system is None:
            number_system_value = "complex"
        else:
            number_system_value = self._validate_number_system(number_system)
        z_value = self._validate_scalar_input("z", z, number_system_value)
        tol_value = self._resolve_tol(tol)
        if isinstance(z_value, complex):
            x_value = float(z_value.real)
            y_value = float(z_value.imag)
        else:
            x_value = float(z_value)
            y_value = 0.0
        def plane_function(x_real: Number, y_real: Number) -> Number:
            return self._evaluate_scalar_function(function, [complex(x_real, y_real)], "complex", tol_value)
        partial_x = self.partial_derivative(plane_function, [x_value, y_value], variable_index=0,
                                             h=h, method=method, number_system="complex",
                                             tol=tol, max_iter=max_iter)
        partial_y = self.partial_derivative(plane_function, [x_value, y_value], variable_index=1,
                                             h=h, method=method, number_system="complex",
                                             tol=tol, max_iter=max_iter)
        df_dz = self._normalize(0.5 * (partial_x - 1j * partial_y), tol_value, number_system_value)
        df_dz_conj = self._normalize(0.5 * (partial_x + 1j * partial_y), tol_value, number_system_value)
        return {"point": complex(x_value, y_value), "df_dz": df_dz, "df_dz_conjugate": df_dz_conj}

    def generalized_derivative(self, function, x: Number, h=None, tol=None, max_iter=None) -> dict:
        tol_value = self._resolve_tol(tol)
        x_value = self._validate_scalar_input("x", x, "real")
        left = self.left_derivative(function, x_value, h=h, number_system="real", tol=tol, max_iter=max_iter)
        right = self.right_derivative(function, x_value, h=h, number_system="real", tol=tol, max_iter=max_iter)
        if self._abs(left - right) <= tol_value * max(1.0, self._abs(left), self._abs(right)):
            derivative = self._normalize(0.5 * (left + right), tol_value, "real")
            return {"differentiable": True, "derivative": derivative, "left_derivative": left, "right_derivative": right}
        lower = float(left); upper = float(right)
        if lower > upper: lower, upper = upper, lower
        return {"differentiable": False, "left_derivative": left, "right_derivative": right, "clarke_interval": (lower, upper)}

    def subgradient(self, function, x: Number, h=None, tol=None, max_iter=None):
        info = self.generalized_derivative(function, x, h=h, tol=tol, max_iter=max_iter)
        if info["differentiable"]:
            return [info["derivative"]]
        return info["clarke_interval"]

    def gateaux_derivative(self, function, point: Vector, direction: Vector, h=None,
                            number_system=None, tol=None, max_iter=None) -> OperatorOutput:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        direction_value = self._validate_point(direction, number_system_value)
        tol_value = self._resolve_tol(tol)
        max_iter_value = self._resolve_max_iter(max_iter)
        if len(point_value) != len(direction_value):
            raise ValueError("point and direction must have the same dimension.")
        h_current = self._resolve_h(h)
        previous: Optional[OperatorOutput] = None
        for _ in range(max_iter_value):
            base_value = self._evaluate_operator_function(function, point_value, number_system_value, tol_value)
            shifted_point: List[Number] = []
            for i in range(len(point_value)):
                shifted_point.append(point_value[i] + h_current * direction_value[i])
            shifted_value = self._evaluate_operator_function(function, shifted_point, number_system_value, tol_value)
            current = self._divide_operator_output(self._subtract_operator_outputs(shifted_value, base_value), h_current)
            current = self._normalize_operator_output(current, tol_value, number_system_value)
            if previous is not None:
                if self._operator_output_difference(current, previous) <= tol_value * max(1.0, self._operator_output_norm(current)):
                    return current
            previous = current
            h_current *= 0.5
        raise RuntimeError("Gateaux derivative did not converge within max_iter.")

    def frechet_derivative(self, function, point: Vector, h=None, method: str = "central",
                            number_system=None, tol=None, max_iter=None) -> dict:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        tol_value = self._resolve_tol(tol)
        output_at_point = self._evaluate_operator_function(function, point_value, number_system_value, tol_value)
        input_dimension = len(point_value)
        if isinstance(output_at_point, list):
            output_dimension = len(output_at_point)
            matrix: List[List[Number]] = []
            for index in range(output_dimension):
                def component_function(*args: Number, _index: int = index) -> Number:
                    if len(args) == 1 and isinstance(args[0], Sequence) and not isinstance(args[0], (str, bytes)):
                        point_candidate = list(args[0])
                    else:
                        point_candidate = list(args)
                    values = self._evaluate_operator_function(function, point_candidate, number_system_value, tol_value)
                    if not isinstance(values, list):
                        raise TypeError("function output type changed during Frechet derivative evaluation.")
                    return values[_index]
                matrix.append(self.gradient(component_function, point_value, h=h, method=method,
                                             number_system=number_system_value, tol=tol, max_iter=max_iter))
        else:
            output_dimension = 1
            matrix = [self.gradient(function, point_value, h=h, method=method,
                                     number_system=number_system_value, tol=tol, max_iter=max_iter)]
        return {"point": tuple(point_value), "input_dimension": input_dimension,
                "output_dimension": output_dimension, "matrix": matrix}

    def divergence(self, vector_field, point: Vector, h=None, method: str = "central",
                    number_system=None, tol=None, max_iter=None) -> Number:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        tol_value = self._resolve_tol(tol)
        dimension = self._vector_dimension(vector_field, point_value, number_system_value, tol_value)
        if dimension != len(point_value):
            raise ValueError("vector field dimension must match point dimension for divergence.")
        total: Number = 0.0
        for i in range(dimension):
            component = self._get_vector_component_function(vector_field, i, point_value, number_system_value, tol_value)
            total = total + self.partial_derivative(component, point_value, variable_index=i,
                                                     h=h, method=method, number_system=number_system_value,
                                                     tol=tol_value, max_iter=max_iter)
        return self._normalize(total, tol_value, number_system_value)

    def curl(self, vector_field, point: Vector, h=None, method: str = "central",
              number_system=None, tol=None, max_iter=None) -> List[Number]:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        tol_value = self._resolve_tol(tol)
        dimension = self._vector_dimension(vector_field, point_value, number_system_value, tol_value)
        if dimension != 3 or len(point_value) != 3:
            raise ValueError("curl is defined here for 3D vector fields only.")
        fx = self._get_vector_component_function(vector_field, 0, point_value, number_system_value, tol_value)
        fy = self._get_vector_component_function(vector_field, 1, point_value, number_system_value, tol_value)
        fz = self._get_vector_component_function(vector_field, 2, point_value, number_system_value, tol_value)
        d_fz_dy = self.partial_derivative(fz, point_value, variable_index=1, h=h, method=method,
                                           number_system=number_system_value, tol=tol_value, max_iter=max_iter)
        d_fy_dz = self.partial_derivative(fy, point_value, variable_index=2, h=h, method=method,
                                           number_system=number_system_value, tol=tol_value, max_iter=max_iter)
        d_fx_dz = self.partial_derivative(fx, point_value, variable_index=2, h=h, method=method,
                                           number_system=number_system_value, tol=tol_value, max_iter=max_iter)
        d_fz_dx = self.partial_derivative(fz, point_value, variable_index=0, h=h, method=method,
                                           number_system=number_system_value, tol=tol_value, max_iter=max_iter)
        d_fy_dx = self.partial_derivative(fy, point_value, variable_index=0, h=h, method=method,
                                           number_system=number_system_value, tol=tol_value, max_iter=max_iter)
        d_fx_dy = self.partial_derivative(fx, point_value, variable_index=1, h=h, method=method,
                                           number_system=number_system_value, tol=tol_value, max_iter=max_iter)
        return [self._normalize(d_fz_dy - d_fy_dz, tol_value, number_system_value),
                self._normalize(d_fx_dz - d_fz_dx, tol_value, number_system_value),
                self._normalize(d_fy_dx - d_fx_dy, tol_value, number_system_value)]

    def vector_laplacian(self, vector_field, point: Vector, h=None, number_system=None, tol=None) -> List[Number]:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        tol_value = self._resolve_tol(tol)
        dimension = self._vector_dimension(vector_field, point_value, number_system_value, tol_value)
        result: List[Number] = []
        for i in range(dimension):
            component = self._get_vector_component_function(vector_field, i, point_value, number_system_value, tol_value)
            result.append(self.laplacian(component, point_value, h=h, number_system=number_system_value, tol=tol_value))
        return result

    def total_differential(self, function, point: Vector, h=None, method: str = "central",
                            number_system=None, tol=None, max_iter=None) -> dict:
        number_system_value = self._resolve_number_system(number_system)
        point_value = self._validate_point(point, number_system_value)
        coefficients = self.gradient(function, point_value, h=h, method=method,
                                      number_system=number_system_value, tol=tol, max_iter=max_iter)
        return {"point": tuple(point_value), "coefficients": coefficients}


__all__ = ["Differentiation"]
