# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# laplace.pyx — Phase B/C: Laplace 변환 엔진 Python 래퍼

# PyExpr cdef class cimport
from .laplace_ast cimport PyExpr

# ------------------------------------------------------------------ C++ CAPI Phase B/C
cdef extern from "cpp/capi.hpp" namespace "ml_laplace":
    size_t laplace_forward_transform(size_t, size_t, size_t) except +
    void   laplace_clear_cache() except +
    # Phase C
    size_t laplace_inverse_transform(size_t, size_t, size_t) except +
    int    laplace_compute_poles(size_t, size_t, double*, double*, int) except +
    int    laplace_compute_zeros(size_t, size_t, double*, double*, int) except +
    double laplace_final_value(size_t, size_t, int*) except +
    double laplace_initial_value(size_t, size_t, int*) except +
    size_t laplace_partial_fractions(size_t, size_t) except +

# ------------------------------------------------------------------ pool_make_const (수치 치환용)
from libc.stdint cimport int64_t, uint64_t
cdef extern from "cpp/capi.hpp" namespace "ml_laplace":
    size_t pool_make_const(double) except +
    size_t expr_symbolic_substitute(size_t, const char*, size_t) except +


# ================================================================== Laplace 클래스

cdef class Laplace:
    """
    기호 라플라스 변환 엔진.

    사용 예::

        L = Laplace()
        F = L.transform(Sin(2*t))   # → 2/(s^2+4)
        f = L.inverse(F)             # → sin(2t) (Phase C)
    """

    cdef size_t _t_handle
    cdef size_t _s_handle
    cdef object _t_obj
    cdef object _s_obj

    def __init__(self, t_var='t', s_var='s'):
        """
        t_var: 시간 변수 이름 (기본 't')
        s_var: 주파수 변수 이름 (기본 's')
        """
        from .laplace_ast import symbol as _symbol
        self._t_obj = _symbol(t_var)
        self._s_obj = _symbol(s_var)
        self._t_handle = (<PyExpr>self._t_obj)._handle
        self._s_handle = (<PyExpr>self._s_obj)._handle

    def transform(self, f):
        """
        F(s) = L{f(t)} 반환.

        Parameters
        ----------
        f : PyExpr
            t-영역 표현식

        Returns
        -------
        PyExpr
            s-영역 변환 결과

        Raises
        ------
        ValueError
            변환 불가 표현식
        """
        cdef PyExpr fe
        if not isinstance(f, PyExpr):
            raise TypeError(f"transform: expected PyExpr, got {type(f).__name__}")
        fe = <PyExpr>f
        cdef size_t result_h = laplace_forward_transform(
            fe._handle, self._t_handle, self._s_handle)
        return PyExpr._wrap(result_h)

    def inverse(self, F):
        """
        f(t) = L^{-1}{F(s)} 반환. 유리함수만 지원 (Phase C).

        Parameters
        ----------
        F : PyExpr
            s-영역 유리함수 표현식

        Returns
        -------
        PyExpr
            t-영역 역변환 결과
        """
        cdef PyExpr Fe
        if not isinstance(F, PyExpr):
            raise TypeError(f"inverse: expected PyExpr, got {type(F).__name__}")
        Fe = <PyExpr>F
        cdef size_t result_h = laplace_inverse_transform(
            Fe._handle, self._s_handle, self._t_handle)
        return PyExpr._wrap(result_h)

    def poles(self, F):
        """
        F(s)의 극점 (복소수 리스트) 반환.

        Parameters
        ----------
        F : PyExpr

        Returns
        -------
        list of complex
        """
        cdef PyExpr Fe
        if not isinstance(F, PyExpr):
            raise TypeError(f"poles: expected PyExpr, got {type(F).__name__}")
        Fe = <PyExpr>F
        cdef double[64] re_buf
        cdef double[64] im_buf
        cdef int n = laplace_compute_poles(Fe._handle, self._s_handle,
                                            re_buf, im_buf, 64)
        return [complex(re_buf[i], im_buf[i]) for i in range(n)]

    def zeros(self, F):
        """
        F(s)의 영점 (복소수 리스트) 반환.

        Parameters
        ----------
        F : PyExpr

        Returns
        -------
        list of complex
        """
        cdef PyExpr Fe
        if not isinstance(F, PyExpr):
            raise TypeError(f"zeros: expected PyExpr, got {type(F).__name__}")
        Fe = <PyExpr>F
        cdef double[64] re_buf
        cdef double[64] im_buf
        cdef int n = laplace_compute_zeros(Fe._handle, self._s_handle,
                                            re_buf, im_buf, 64)
        return [complex(re_buf[i], im_buf[i]) for i in range(n)]

    def final_value(self, F):
        """
        최종값 정리: lim_{s->0} s*F(s).
        반환값: (value, valid) — valid=False이면 정리 적용 불가
        """
        cdef PyExpr Fe
        if not isinstance(F, PyExpr):
            raise TypeError(f"final_value: expected PyExpr, got {type(F).__name__}")
        Fe = <PyExpr>F
        cdef int valid_i = 0
        cdef double v = laplace_final_value(Fe._handle, self._s_handle, &valid_i)
        return (v, bool(valid_i))

    def initial_value(self, F):
        """
        초기값 정리: lim_{s->inf} s*F(s).
        반환값: (value, valid)
        """
        cdef PyExpr Fe
        if not isinstance(F, PyExpr):
            raise TypeError(f"initial_value: expected PyExpr, got {type(F).__name__}")
        Fe = <PyExpr>F
        cdef int valid_i = 0
        cdef double v = laplace_initial_value(Fe._handle, self._s_handle, &valid_i)
        return (v, bool(valid_i))

    def partial_fractions(self, F):
        """
        F(s)의 partial fraction 분해 결과 AST 반환.

        Parameters
        ----------
        F : PyExpr

        Returns
        -------
        PyExpr
            부분 분수 합으로 표현된 AST
        """
        cdef PyExpr Fe
        if not isinstance(F, PyExpr):
            raise TypeError(f"partial_fractions: expected PyExpr, got {type(F).__name__}")
        Fe = <PyExpr>F
        cdef size_t result_h = laplace_partial_fractions(Fe._handle, self._s_handle)
        return PyExpr._wrap(result_h)

    def clear_cache(self):
        """변환 캐시 초기화 (thread_local)"""
        laplace_clear_cache()

    @property
    def t(self):
        """시간 변수 심볼"""
        return self._t_obj

    @property
    def s(self):
        """주파수 변수 심볼"""
        return self._s_obj
