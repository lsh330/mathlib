# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# numerical_analysis.pyx
#
# NumericalAnalysis 클래스 — Simpson 적분법 8종 구현
#
# 구현 목록:
#   1. simpson_13           — 3점 단순 Simpson 1/3
#   2. simpson_38           — 4점 단순 Simpson 3/8
#   3. composite_simpson_13 — n 구간 합성 Simpson 1/3 (n 짝수)
#   4. composite_simpson_38 — n 구간 합성 Simpson 3/8 (n 3의 배수)
#   5. adaptive_simpson     — 적응형 재귀 Simpson (명시적 스택)
#   6. mixed_simpson        — 혼합 Simpson (임의 n >= 2)
#   7. simpson_irregular    — 불균등 간격 Simpson
#   8. romberg              — Romberg 적분 (Richardson 확장)

from libc.math cimport fabs, isnan, isinf
include "_kahan.pxd"


# ================================================================== 헬퍼 함수

cdef object _resolve_callable(object f, str var):
    """
    f가 Python callable이면 그대로,
    PyExpr(evalf 메서드 보유)이면 evalf 기반 래퍼 반환.

    PyExpr.lambdify()는 복잡한 곱셈 표현식에서 파싱 버그가 있으므로
    evalf(**{var: x}) 방식으로 직접 수치 평가.
    """
    if hasattr(f, 'evalf'):
        # PyExpr: evalf 래퍼 (lambda로 var 이름 캡처)
        _var = var
        _f = f
        return lambda x, _f=_f, _v=_var: _f.evalf(**{_v: x})
    if callable(f):
        return f
    raise TypeError(
        f"f must be callable or have lambdify method, got {type(f).__name__}"
    )


cdef double _call_f(object f, double x) except? -1.7976931348623157e+308:
    """Python callable 단일 호출."""
    return <double>f(x)


cdef double _simple_13(object f, double a, double b) except? -1.7976931348623157e+308:
    """
    3점 Simpson 1/3 내부 계산.
    I = (h/3) * [f(a) + 4*f(m) + f(b)], h = (b-a)/2
    """
    cdef double h = (b - a) * 0.5
    cdef double m = (a + b) * 0.5
    return (h / 3.0) * (_call_f(f, a) + 4.0 * _call_f(f, m) + _call_f(f, b))


cdef double _simple_38(object f, double a, double b) except? -1.7976931348623157e+308:
    """
    4점 Simpson 3/8 내부 계산.
    I = (3h/8) * [f(a) + 3*f(x1) + 3*f(x2) + f(b)], h = (b-a)/3
    """
    cdef double h = (b - a) / 3.0
    return (3.0 * h / 8.0) * (
        _call_f(f, a) +
        3.0 * _call_f(f, a + h) +
        3.0 * _call_f(f, a + 2.0 * h) +
        _call_f(f, b)
    )


cdef double _composite_13_raw(object f, double a, double b, int n) except? -1.7976931348623157e+308:
    """
    합성 Simpson 1/3 내부 계산 (n은 짝수, 검증 없음).
    Kahan 보상 누산 사용.
    """
    cdef double h = (b - a) / n
    cdef double s_odd_val = 0.0, s_odd_c = 0.0
    cdef double s_even_val = 0.0, s_even_c = 0.0
    cdef int i

    for i in range(1, n, 2):
        kahan_add(&s_odd_val, &s_odd_c, _call_f(f, a + i * h))
    for i in range(2, n - 1, 2):
        kahan_add(&s_even_val, &s_even_c, _call_f(f, a + i * h))

    return (h / 3.0) * (_call_f(f, a) + 4.0 * s_odd_val + 2.0 * s_even_val + _call_f(f, b))


cdef double _trapezoidal_recursive(object f, double a, double b,
                                    int level, double prev_T) except? -1.7976931348623157e+308:
    """
    Romberg 용 재귀적 사다리꼴 계산.
    level=0: T[0][0] = (b-a)/2 * (f(a) + f(b))
    level=i: T[i][0] = T[i-1][0]/2 + h_i * sum(f at new midpoints)
    """
    cdef int n_intervals = 1 << level  # 2^level
    cdef double h = (b - a) / n_intervals
    cdef double s_val = 0.0, s_c = 0.0
    cdef int i

    if level == 0:
        return (b - a) * 0.5 * (_call_f(f, a) + _call_f(f, b))

    # level >= 1: 홀수 인덱스(새 점)만 합산
    for i in range(1, n_intervals, 2):
        kahan_add(&s_val, &s_c, _call_f(f, a + i * h))

    return prev_T * 0.5 + h * s_val


# ================================================================== NumericalAnalysis 클래스

cdef class NumericalAnalysis:
    """
    수치 적분 알고리즘 모음.

    Simpson 적분법 8종을 제공하며, 향후 다른 수치해석 메서드 추가 예정.

    함수 입력:
        - Python callable: lambda, math.sin 등
        - PyExpr: math_library.laplace의 기호 표현식 (lambdify 자동 적용)

    Examples
    --------
    >>> import math
    >>> from math_library.numerical_analysis import NumericalAnalysis
    >>> na = NumericalAnalysis()
    >>> na.simpson_13(math.sin, 0, math.pi)   # ≈ 2.0
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------ 1. simpson_13

    def simpson_13(self, f, double a, double b, *,
                   str var='t', bint return_error=False):
        """
        3점 단순 Simpson 1/3 규칙.

        I = (h/3) * [f(a) + 4*f(m) + f(b)],  h = (b-a)/2,  m = (a+b)/2

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        var          : str    PyExpr용 변수명 (기본 't')
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  a >= b
        TypeError   f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        cdef object _f = _resolve_callable(f, var)
        cdef double h = (b - a) * 0.5
        cdef double m = (a + b) * 0.5
        cdef double fa = _call_f(_f, a)
        cdef double fm = _call_f(_f, m)
        cdef double fb = _call_f(_f, b)
        cdef double I = (h / 3.0) * (fa + 4.0 * fm + fb)
        cdef double I_38, err

        if return_error:
            I_38 = _simple_38(_f, a, b)
            err = fabs(I - I_38) / 15.0
            return (I, err)
        return I

    # ------------------------------------------------------------------ 2. simpson_38

    def simpson_38(self, f, double a, double b, *,
                   str var='t', bint return_error=False):
        """
        4점 단순 Simpson 3/8 규칙.

        I = (3h/8) * [f(a) + 3*f(x1) + 3*f(x2) + f(b)],  h = (b-a)/3

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  a >= b
        TypeError   f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        cdef object _f = _resolve_callable(f, var)
        cdef double h = (b - a) / 3.0
        cdef double x1 = a + h
        cdef double x2 = a + 2.0 * h
        cdef double fa = _call_f(_f, a)
        cdef double f1 = _call_f(_f, x1)
        cdef double f2 = _call_f(_f, x2)
        cdef double fb = _call_f(_f, b)
        cdef double I = (3.0 * h / 8.0) * (fa + 3.0 * f1 + 3.0 * f2 + fb)
        cdef double I_13, err

        if return_error:
            I_13 = _simple_13(_f, a, b)
            err = fabs(I - I_13) / 15.0
            return (I, err)
        return I

    # ------------------------------------------------------------------ 3. composite_simpson_13

    def composite_simpson_13(self, f, double a, double b, int n, *,
                              str var='t', bint return_error=False):
        """
        합성 Simpson 1/3 규칙.

        I = (h/3) * [f(a) + 4*Σ_odd + 2*Σ_even + f(b)]
        h = (b-a)/n

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        n            : int    구간 수 (짝수, >= 2)
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  a >= b, n < 2, n % 2 != 0
        TypeError   f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        if n < 2:
            raise ValueError(f"composite_simpson_13 requires n >= 2, got n={n}")
        if n % 2 != 0:
            raise ValueError(
                f"composite_simpson_13 requires even n, got n={n}"
            )

        cdef object _f = _resolve_callable(f, var)
        cdef double I = _composite_13_raw(_f, a, b, n)
        cdef int n_half
        cdef double I_half, err

        if return_error:
            n_half = n // 2
            if n_half >= 2 and n_half % 2 == 0:
                I_half = _composite_13_raw(_f, a, b, n_half)
                err = fabs(I - I_half) / 15.0
            else:
                err = float('nan')
            return (I, err)
        return I

    # ------------------------------------------------------------------ 4. composite_simpson_38

    def composite_simpson_38(self, f, double a, double b, int n, *,
                              str var='t', bint return_error=False):
        """
        합성 Simpson 3/8 규칙.

        I = (3h/8) * [f(a) + 계수합 + f(b)]
        h = (b-a)/n
        계수: i%3 != 0 이면 3, i%3 == 0 (i≠0, i≠n) 이면 2

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        n            : int    구간 수 (3의 배수, >= 3)
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  a >= b, n < 3, n % 3 != 0
        TypeError   f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        if n < 3:
            raise ValueError(f"composite_simpson_38 requires n >= 3, got n={n}")
        if n % 3 != 0:
            raise ValueError(
                f"composite_simpson_38 requires n divisible by 3, got n={n}"
            )

        cdef object _f = _resolve_callable(f, var)
        cdef double h = (b - a) / n
        cdef double s3_val = 0.0, s3_c = 0.0
        cdef double s2_val = 0.0, s2_c = 0.0
        cdef int i

        for i in range(1, n):
            if i % 3 == 0:
                kahan_add(&s2_val, &s2_c, _call_f(_f, a + i * h))
            else:
                kahan_add(&s3_val, &s3_c, _call_f(_f, a + i * h))

        cdef double I = (3.0 * h / 8.0) * (
            _call_f(_f, a) + 3.0 * s3_val + 2.0 * s2_val + _call_f(_f, b)
        )
        cdef int n_coarse
        cdef double I_coarse, err

        if return_error:
            n_coarse = (n * 2) // 3
            if n_coarse >= 3 and n_coarse % 3 == 0:
                I_coarse = self.composite_simpson_38(f, a, b, n_coarse, var=var)
                err = fabs(I - I_coarse) / 63.0
            else:
                err = float('nan')
            return (I, err)
        return I

    # ------------------------------------------------------------------ 5. adaptive_simpson

    def adaptive_simpson(self, f, double a, double b, *,
                          double tol=1e-10, int max_depth=50,
                          str var='t', bint return_error=False):
        """
        적응형 Simpson 적분 (명시적 스택 기반).

        각 구간에서 Simpson 1/3을 계산하고 오차가 허용값을 초과하면
        구간을 절반으로 나누어 처리.

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        tol          : float  허용 오차 (> 0)
        max_depth    : int    최대 분할 깊이
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError   a >= b 또는 tol <= 0
        TypeError    f가 callable이 아닌 경우
        RuntimeError max_depth 초과
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got tol={tol}")

        cdef object _f = _resolve_callable(f, var)

        # 스택 항목: (aa, bb, tolerance, fa, fm, fb, S1, depth)
        cdef double fa_root = _call_f(_f, a)
        cdef double fm_root = _call_f(_f, (a + b) * 0.5)
        cdef double fb_root = _call_f(_f, b)
        cdef double S_root = ((b - a) * 0.5 / 3.0) * (fa_root + 4.0 * fm_root + fb_root)

        cdef list stack = [(a, b, tol, fa_root, fm_root, fb_root, S_root, 0)]
        cdef double total = 0.0
        cdef double total_err = 0.0
        cdef double aa, bb, tt, fa_i, fm_i, fb_i, S1
        cdef double mid_d, fa_l, fm_l, fb_l, fa_r, fm_r, fb_r
        cdef double S_l, S_r, S2, delta
        cdef int depth

        while stack:
            aa, bb, tt, fa_i, fm_i, fb_i, S1, depth = stack.pop()

            if depth > max_depth:
                raise RuntimeError(
                    f"adaptive_simpson exceeded max_depth={max_depth}, "
                    "reduce tol or increase max_depth"
                )

            mid_d = (aa + bb) * 0.5

            # 왼쪽 구간 [aa, mid]
            fa_l = fa_i
            fm_l = _call_f(_f, (aa + mid_d) * 0.5)
            fb_l = fm_i
            S_l = ((mid_d - aa) * 0.5 / 3.0) * (fa_l + 4.0 * fm_l + fb_l)

            # 오른쪽 구간 [mid, bb]
            fa_r = fm_i
            fm_r = _call_f(_f, (mid_d + bb) * 0.5)
            fb_r = fb_i
            S_r = ((bb - mid_d) * 0.5 / 3.0) * (fa_r + 4.0 * fm_r + fb_r)

            S2 = S_l + S_r
            delta = fabs(S2 - S1)

            if delta < 15.0 * tt or depth == max_depth:
                total += S2 + (S2 - S1) / 15.0
                total_err += delta / 15.0
            else:
                stack.append((aa, mid_d, tt * 0.5, fa_l, fm_l, fb_l, S_l, depth + 1))
                stack.append((mid_d, bb, tt * 0.5, fa_r, fm_r, fb_r, S_r, depth + 1))

        if return_error:
            return (total, total_err)
        return total

    # ------------------------------------------------------------------ 6. mixed_simpson

    def mixed_simpson(self, f, double a, double b, int n, *,
                      str var='t', bint return_error=False):
        """
        혼합 Simpson 적분.

        n이 짝수이면 합성 1/3, 홀수이면 앞 (n-3) 구간은 1/3, 끝 3 구간은 3/8.

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        n            : int    구간 수 (>= 2)
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  n < 2 또는 a >= b
        TypeError   f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        if n < 2:
            raise ValueError(f"mixed_simpson requires n >= 2, got n={n}")

        cdef object _f = _resolve_callable(f, var)
        cdef double I, I_half, err
        cdef double h, split
        cdef double I1, I2, I_total
        cdef int n_half

        if n % 2 == 0:
            I = _composite_13_raw(_f, a, b, n)
            if return_error:
                if n >= 4:
                    n_half = n // 2
                    if n_half % 2 != 0:
                        n_half += 1
                    if n_half >= 2:
                        I_half = _composite_13_raw(_f, a, b, n_half)
                        err = fabs(I - I_half) / 15.0
                    else:
                        err = float('nan')
                else:
                    err = float('nan')
                return (I, err)
            return I
        else:
            # 홀수: 첫 (n-3) 구간 1/3 + 마지막 3 구간 3/8
            # n이 홀수이고 n >= 3 이면 n-3은 항상 짝수(홀수-홀수=짝수)
            h = (b - a) / n
            split = a + (n - 3) * h

            if n == 3:
                I1 = 0.0
                I2 = _simple_38(_f, a, b)
            else:
                # n-3 >= 2 이고 n-3은 짝수 (n 홀수 → n-3 짝수)
                I1 = _composite_13_raw(_f, a, split, n - 3)
                I2 = _simple_38(_f, split, b)

            I_total = I1 + I2
            if return_error:
                return (I_total, float('nan'))
            return I_total

    # ------------------------------------------------------------------ 7. simpson_irregular

    def simpson_irregular(self, x_points, y_points, *,
                           bint return_error=False):
        """
        불균등 간격 Simpson 적분.

        3점 비균등 Simpson 공식:
          h0 = x1-x0, h1 = x2-x1
          I = (h0+h1)/6 * [(2-h1/h0)*f0 + (h0+h1)^2/(h0*h1)*f1 + (2-h0/h1)*f2]

        점의 수가 홀수(짝수 구간)이면 전체를 2구간씩 Simpson 적용.
        점의 수가 짝수(홀수 구간)이면 마지막 구간은 사다리꼴로 처리 후 합산.

        Parameters
        ----------
        x_points     : list of float  단조 증가 x 좌표
        y_points     : list of float  대응 함수값
        return_error : bool           True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  점 수 불일치, 점 수 < 3, 단조성 위반, NaN/Inf 포함
        """
        cdef int n_pts = len(x_points)
        if n_pts != len(y_points):
            raise ValueError(
                f"x_points and y_points must have the same length, "
                f"got {n_pts} and {len(y_points)}"
            )
        if n_pts < 3:
            raise ValueError(
                f"simpson_irregular requires >= 3 points, got {n_pts}"
            )

        # 유효성 검사
        cdef list xp = [float(x) for x in x_points]
        cdef list yp = [float(y) for y in y_points]
        cdef double xi, yi
        cdef int i

        for i in range(n_pts):
            xi = xp[i]
            yi = yp[i]
            if isnan(xi) or isinf(xi):
                raise ValueError(f"x_points[{i}]={xi} is NaN or Inf")
            if isnan(yi) or isinf(yi):
                raise ValueError(f"y_points[{i}]={yi} is NaN or Inf")

        for i in range(1, n_pts):
            if xp[i] <= xp[i-1]:
                raise ValueError(
                    f"x_points must be monotonically increasing, "
                    f"violation at index {i}: x[{i-1}]={xp[i-1]}, x[{i}]={xp[i]}"
                )

        # 적분 계산
        cdef double total_s = 0.0, total_c = 0.0
        cdef double x0, x1, x2, f0, f1, f2
        cdef double h0, h1, coeff0, coeff1, coeff2, contrib

        if n_pts % 2 == 1:
            # 홀수 점 수: 2구간씩 순서대로 Simpson 적용
            i = 0
            while i + 2 < n_pts:
                x0 = xp[i];   f0 = yp[i]
                x1 = xp[i+1]; f1 = yp[i+1]
                x2 = xp[i+2]; f2 = yp[i+2]
                h0 = x1 - x0
                h1 = x2 - x1
                coeff0 = 2.0 - h1 / h0
                coeff1 = (h0 + h1) * (h0 + h1) / (h0 * h1)
                coeff2 = 2.0 - h0 / h1
                contrib = (h0 + h1) / 6.0 * (coeff0 * f0 + coeff1 * f1 + coeff2 * f2)
                kahan_add(&total_s, &total_c, contrib)
                i += 2
        else:
            # 짝수 점 수: 끝 1구간을 사다리꼴로 처리, 나머지는 2구간씩 Simpson
            # [0..n-2] → n-2+1 = n-1개 점 → 홀수 점 수로 Simpson 가능 (if n-1 홀수)
            # 짝수 점: n-1 = 홀수 → [0..n-2] 구간에서 홀수 개 점 Simpson 적용
            # + 마지막 구간 [n-2, n-1] 사다리꼴
            i = 0
            while i + 2 < n_pts - 1:
                x0 = xp[i];   f0 = yp[i]
                x1 = xp[i+1]; f1 = yp[i+1]
                x2 = xp[i+2]; f2 = yp[i+2]
                h0 = x1 - x0
                h1 = x2 - x1
                coeff0 = 2.0 - h1 / h0
                coeff1 = (h0 + h1) * (h0 + h1) / (h0 * h1)
                coeff2 = 2.0 - h0 / h1
                contrib = (h0 + h1) / 6.0 * (coeff0 * f0 + coeff1 * f1 + coeff2 * f2)
                kahan_add(&total_s, &total_c, contrib)
                i += 2

            # 마지막 구간 [n-2, n-1] 사다리꼴
            x0 = xp[n_pts - 2]; f0 = yp[n_pts - 2]
            x1 = xp[n_pts - 1]; f1 = yp[n_pts - 1]
            kahan_add(&total_s, &total_c, (x1 - x0) * 0.5 * (f0 + f1))

        if return_error:
            return (total_s, float('nan'))
        return total_s

    # ------------------------------------------------------------------ 8. romberg

    def romberg(self, f, double a, double b, *,
                int depth=5, str var='t', bint return_error=False):
        """
        Romberg 적분 (Richardson 외삽 확장).

        T[i][0] = 사다리꼴 규칙 (2^i 구간)
        T[i][j] = (4^j * T[i][j-1] - T[i-1][j-1]) / (4^j - 1)

        T[depth][depth]이 최고 차수 근사.

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        depth        : int    Richardson 테이블 깊이 (>= 1, 권장 <= 20)
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError   a >= b 또는 depth < 1
        TypeError    f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        if depth < 1:
            raise ValueError(f"romberg requires depth >= 1, got depth={depth}")
        if depth > 20:
            import warnings
            warnings.warn(
                f"romberg: depth={depth} > 20 may cause numerical instability",
                RuntimeWarning,
                stacklevel=2,
            )

        cdef object _f = _resolve_callable(f, var)

        # T 테이블: (depth+1) x (depth+1) — Python list of lists
        cdef int sz = depth + 1
        cdef list T = [[0.0] * sz for _ in range(sz)]
        cdef double prev_trap = 0.0
        cdef double fac, val
        cdef int i, j

        # T[i][0]: 사다리꼴 규칙
        for i in range(sz):
            prev_trap = _trapezoidal_recursive(_f, a, b, i, prev_trap)
            T[i][0] = prev_trap

        # Richardson 외삽
        for j in range(1, sz):
            fac = <double>(1 << (2 * j))  # 4^j
            for i in range(j, sz):
                val = (fac * <double>T[i][j-1] - <double>T[i-1][j-1]) / (fac - 1.0)
                T[i][j] = val

        cdef double result = <double>T[depth][depth]
        cdef double err

        if return_error:
            err = fabs(result - <double>T[depth][depth - 1]) if depth >= 1 else 0.0
            return (result, err)
        return result
