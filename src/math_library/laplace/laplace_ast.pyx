# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# laplace_ast.pyx — C++ AST 코어의 Python 경계 래퍼
# ExprPtr(const Expr*)를 size_t 핸들로 관리 (Cython const-ptr typedef 호환성 회피)

from libc.stdint cimport int64_t, uint64_t, uint8_t

# ------------------------------------------------------------------ C++ CAPI 선언 (size_t 핸들 기반)
cdef extern from "cpp/capi.hpp" namespace "ml_laplace":
    # SubstMap 생명주기 (핸들: size_t)
    size_t smap_new() except +
    void   smap_delete(size_t h)
    void   smap_set(size_t h, const char* key, double val) except +
    void   smap_clear(size_t h)

    # Phase B: LaTeX + 기호 치환
    void   expr_to_latex_buf(size_t h, char* buf, size_t buflen) except +
    size_t expr_symbolic_substitute(size_t expr_h, const char* var_name, size_t new_expr_h) except +

    # 팩토리 (모두 size_t 핸들 반환)
    size_t pool_make_const(double) except +
    size_t pool_make_rational(int64_t, int64_t) except +
    size_t pool_make_var(const char*) except +
    size_t pool_make_func(uint8_t, size_t) except +
    size_t pool_make_neg(size_t) except +

    # Phase E: Heaviside / Dirac
    size_t pool_make_heaviside(size_t) except +
    size_t pool_make_dirac(size_t) except +

    size_t pool_add(size_t, size_t) except +
    size_t pool_sub(size_t, size_t) except +
    size_t pool_mul(size_t, size_t) except +
    size_t pool_div(size_t, size_t) except +
    size_t pool_pow(size_t, size_t) except +
    size_t pool_var(const char*) except +
    size_t pool_zero() nogil
    size_t pool_one() nogil
    size_t pool_total_nodes() nogil
    size_t pool_intern_hits() nogil
    void   pool_reset() except +

    # Expr 메서드
    double expr_evalf(size_t h, size_t m_h) except +
    void   expr_to_string_buf(size_t h, char* buf, size_t buflen) except +
    size_t expr_substitute(size_t h, size_t m_h) except +
    int    expr_node_type(size_t h) nogil
    uint64_t expr_hash(size_t h) nogil


# ================================================================== FuncId 매핑 상수
DEF FUNCID_SIN    = 0
DEF FUNCID_COS    = 1
DEF FUNCID_TAN    = 2
DEF FUNCID_ARCSIN = 3
DEF FUNCID_ARCCOS = 4
DEF FUNCID_ARCTAN = 5
DEF FUNCID_SINH   = 6
DEF FUNCID_COSH   = 7
DEF FUNCID_TANH   = 8
DEF FUNCID_EXP    = 9
DEF FUNCID_LN     = 10
DEF FUNCID_LOG    = 11
DEF FUNCID_SQRT   = 12


# ================================================================== PyExpr 인스턴스 캐시 (Python is-identity 보장)
# C++ hash-consing으로 handle이 같으면 Python 레벨에서도 동일 인스턴스 반환
# WeakValueDictionary 대신 일반 dict 사용 (오버헤드 절감)
# Pool은 프로세스 수명 동안 유지되므로 메모리 누수 없음
import weakref as _weakref
_pyexpr_cache = {}  # handle(size_t int) -> PyExpr


# ================================================================== 내부 강제 변환

cdef size_t _coerce_handle(object x) except 0:
    """object → size_t 핸들 변환"""
    cdef PyExpr e
    if isinstance(x, PyExpr):
        e = <PyExpr>x
        return e._handle
    if isinstance(x, int):
        return pool_make_rational(<int64_t>x, <int64_t>1)
    if isinstance(x, float):
        return pool_make_const(<double>x)
    raise TypeError(f"Cannot coerce {type(x).__name__} to PyExpr")


# ================================================================== PyExpr cdef class

cdef class PyExpr:
    """
    C++ Expr 노드의 Python 래퍼.
    pool-managed size_t 핸들을 보유 (소유권 없음).
    hash-consing + Python 인스턴스 캐시: 동일 구조 표현식 → 동일 Python 객체 (is True).

    주의: PyExpr(x) 직접 호출은 지원하지 않음. const(1), symbol('x') 팩토리를 사용할 것.
    """

    def __cinit__(self):
        # _handle을 0으로 초기화 (직접 __init__ 호출 시 segfault 방지)
        self._handle = 0
        self._fr_func = None  # frequency response lambdify 캐시

    def __init__(self, *args, **kwargs):
        # 직접 PyExpr(...) 생성 금지 — _wrap 팩토리 사용
        if args or kwargs:
            raise TypeError(
                "PyExpr cannot be constructed directly. "
                "Use const(value), symbol('name'), or arithmetic operators."
            )

    @staticmethod
    cdef PyExpr _wrap(size_t h):
        # Python 인스턴스 캐시 조회: 동일 C++ 포인터 → 동일 Python 객체
        # 일반 dict get()으로 branch 최소화
        cdef PyExpr cached = _pyexpr_cache.get(h)
        if cached is not None:
            return cached
        cdef PyExpr e = PyExpr.__new__(PyExpr)
        e._handle = h
        _pyexpr_cache[h] = e
        return e

    # ------------------------------------------------------------------ 산술 연산자
    # PyExpr+PyExpr fast path: isinstance 체크 없이 바로 handle 접근
    def __add__(PyExpr self, other):
        cdef size_t a = self._handle
        cdef size_t b
        if type(other) is PyExpr:
            b = (<PyExpr>other)._handle
        else:
            b = _coerce_handle(other)
        return PyExpr._wrap(pool_add(a, b))

    def __radd__(PyExpr self, other):
        cdef size_t b = self._handle
        cdef size_t a = _coerce_handle(other)
        return PyExpr._wrap(pool_add(a, b))

    def __sub__(PyExpr self, other):
        cdef size_t a = self._handle
        cdef size_t b
        if type(other) is PyExpr:
            b = (<PyExpr>other)._handle
        else:
            b = _coerce_handle(other)
        return PyExpr._wrap(pool_sub(a, b))

    def __rsub__(PyExpr self, other):
        cdef size_t b = self._handle
        cdef size_t a = _coerce_handle(other)
        return PyExpr._wrap(pool_sub(a, b))

    def __mul__(PyExpr self, other):
        cdef size_t a = self._handle
        cdef size_t b
        if type(other) is PyExpr:
            b = (<PyExpr>other)._handle
        else:
            b = _coerce_handle(other)
        return PyExpr._wrap(pool_mul(a, b))

    def __rmul__(PyExpr self, other):
        cdef size_t b = self._handle
        cdef size_t a = _coerce_handle(other)
        return PyExpr._wrap(pool_mul(a, b))

    def __truediv__(PyExpr self, other):
        cdef size_t a = self._handle
        cdef size_t b
        if type(other) is PyExpr:
            b = (<PyExpr>other)._handle
        else:
            b = _coerce_handle(other)
        return PyExpr._wrap(pool_div(a, b))

    def __rtruediv__(PyExpr self, other):
        cdef size_t b = self._handle
        cdef size_t a = _coerce_handle(other)
        return PyExpr._wrap(pool_div(a, b))

    def __pow__(PyExpr self, other, modulo):
        cdef size_t a = self._handle
        cdef size_t b
        if type(other) is PyExpr:
            b = (<PyExpr>other)._handle
        else:
            b = _coerce_handle(other)
        return PyExpr._wrap(pool_pow(a, b))

    def __neg__(PyExpr self):
        cdef size_t a = self._handle
        return PyExpr._wrap(pool_make_neg(a))

    def __pos__(PyExpr self):
        return self

    # ------------------------------------------------------------------ 비교 / 해시
    def __hash__(PyExpr self):
        return <Py_hash_t>(expr_hash(self._handle))

    def __eq__(PyExpr self, other):
        if not isinstance(other, PyExpr):
            return NotImplemented
        cdef PyExpr o = <PyExpr>other
        return self._handle == o._handle

    def __ne__(PyExpr self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return eq
        return not eq

    # ------------------------------------------------------------------ 출력
    def __repr__(PyExpr self):
        cdef char buf[4096]
        expr_to_string_buf(self._handle, buf, 4096)
        return buf.decode('utf-8')

    def __str__(PyExpr self):
        cdef char buf[4096]
        expr_to_string_buf(self._handle, buf, 4096)
        return buf.decode('utf-8')

    # ------------------------------------------------------------------ 수치 평가
    def evalf(PyExpr self, **subs):
        """
        변수를 double로 치환하여 수치 평가.
        예: expr.evalf(t=1.5, s=0.0)
        """
        cdef size_t m = smap_new()
        cdef bytes key_b
        cdef size_t h = self._handle
        try:
            for k, v in subs.items():
                key_b = k.encode('utf-8') if isinstance(k, str) else k
                smap_set(m, key_b, <double>v)
            return expr_evalf(h, m)
        finally:
            smap_delete(m)

    # ------------------------------------------------------------------ 치환 (수치)
    def substitute(PyExpr self, dict subs):
        """변수를 double로 치환하여 새 AST를 반환."""
        cdef size_t m = smap_new()
        cdef bytes key_b
        cdef size_t h = self._handle
        cdef size_t r
        try:
            for k, v in subs.items():
                if isinstance(v, (int, float)):
                    key_b = k.encode('utf-8') if isinstance(k, str) else k
                    smap_set(m, key_b, <double>v)
            r = expr_substitute(h, m)
            return PyExpr._wrap(r)
        finally:
            smap_delete(m)

    # ------------------------------------------------------------------ 기호 치환 (Phase B)
    def subs(PyExpr self, **kwargs):
        """
        기호 또는 수치 치환.

        수치 치환: F.subs(s=1.0) → Const
        기호 치환: F.subs(s=s+1) → F(s+1)

        Parameters
        ----------
        **kwargs : 변수명 → PyExpr 또는 float/int/complex
        """
        cdef size_t h = self._handle
        cdef bytes key_b
        cdef PyExpr val_expr
        for k, v in kwargs.items():
            key_b = k.encode('utf-8') if isinstance(k, str) else k
            if isinstance(v, PyExpr):
                val_expr = <PyExpr>v
                h = expr_symbolic_substitute(h, key_b, val_expr._handle)
            elif isinstance(v, (int, float)):
                num_expr = PyExpr._wrap(pool_make_const(<double>v))
                val_expr = <PyExpr>num_expr
                h = expr_symbolic_substitute(h, key_b, val_expr._handle)
            else:
                raise TypeError(f"subs: unsupported value type {type(v).__name__}")
        return PyExpr._wrap(h)

    # ------------------------------------------------------------------ LaTeX 출력 (Phase B)
    def latex(PyExpr self):
        """LaTeX 형식 문자열 반환"""
        cdef char buf[8192]
        expr_to_latex_buf(self._handle, buf, 8192)
        return buf.decode('utf-8')

    # ------------------------------------------------------------------ Lambdify (Phase C)
    def lambdify(PyExpr self, var_names, backend='cmath'):
        """
        AST를 Python callable로 변환.

        Parameters
        ----------
        var_names : list of str
            변수 이름 리스트 (예: ['s'], ['t'])
        backend   : str
            'math' | 'cmath' | 'numpy'

        Returns
        -------
        callable

        Examples
        --------
        >>> F = L.transform(Sin(2*t))
        >>> f_s = F.lambdify(['s'], backend='cmath')
        >>> f_s(1.0)  # 2/(1+4) = 0.4
        """
        from .lambdify import lambdify as _lambdify
        return _lambdify(self, list(var_names), backend)

    # ------------------------------------------------------------------ Phase D: 심볼릭 미분
    def diff(PyExpr self, PyExpr var):
        """
        자신을 var 에 대해 심볼릭 미분하여 새 PyExpr 반환.

        Parameters
        ----------
        var : PyExpr
            미분 기준 변수 (symbol('s') 등)

        Returns
        -------
        PyExpr

        Examples
        --------
        >>> F = L.transform(Sin(2*t))   # 2/(s^2+4)
        >>> dF = F.diff(s)              # d/ds [2/(s^2+4)]
        """
        cdef extern from "cpp/capi.hpp" namespace "ml_laplace":
            size_t laplace_differentiate(size_t, size_t) except +
        cdef size_t result_h = laplace_differentiate(self._handle, var._handle)
        return PyExpr._wrap(result_h)

    # ------------------------------------------------------------------ Phase D: 수치 미분 (Differentiation 연동)
    def diff_numeric(PyExpr self, var_name):
        """
        Differentiation (Ridders) 엔진과 연동한 수치 도함수 객체 반환.

        Parameters
        ----------
        var_name : str
            미분 기준 변수 이름 ('s' 등)

        Returns
        -------
        LazyDerivative
            .evalf(**subs), .nth(n, **subs) 메서드 제공
        """
        from .numeric_diff import LazyDerivative
        return LazyDerivative(self, str(var_name))

    # ------------------------------------------------------------------ Phase D: Taylor 급수
    def series(PyExpr self, var_name, about=0.0, order=6):
        """
        Taylor/Maclaurin 급수 객체 반환.

        Parameters
        ----------
        var_name : str
        about    : float  전개점 (기본 0)
        order    : int    최고 차수 (기본 6)

        Returns
        -------
        TaylorSeries
            .compute() → 계수 리스트
            .as_expr()  → PyExpr 다항식
            .evalf(**subs) → float
        """
        from .series import TaylorSeries
        return TaylorSeries(self, str(var_name), float(about), int(order))

    # ------------------------------------------------------------------ Phase D: expand
    def expand(PyExpr self):
        """
        분배 법칙 전개: (a+b)*(c+d) → ac+ad+bc+bd.
        Pow(Sum, n) 이항 전개 (n <= 5).

        Returns
        -------
        PyExpr
        """
        cdef extern from "cpp/capi.hpp" namespace "ml_laplace":
            size_t laplace_expand(size_t) except +
        return PyExpr._wrap(laplace_expand(self._handle))

    # ------------------------------------------------------------------ Phase D: cancel
    def cancel(PyExpr self, var=None):
        """
        유리식 약분: (s^2-1)/(s-1) → s+1.

        Parameters
        ----------
        var : PyExpr, optional
            기준 변수. None 이면 식에서 자동 탐지.

        Returns
        -------
        PyExpr
        """
        cdef extern from "cpp/capi.hpp" namespace "ml_laplace":
            size_t laplace_cancel(size_t, size_t) except +
        cdef size_t var_h = 0
        if var is not None:
            var_h = (<PyExpr>var)._handle
        return PyExpr._wrap(laplace_cancel(self._handle, var_h))

    # ------------------------------------------------------------------ Phase E: collect
    def collect(PyExpr self, var=None):
        """
        var 에 대한 다항식 차수별 계수 합산.
        3*s^2 + 2*s^2 + 5*s + 7  →  5*s^2 + 5*s + 7

        Parameters
        ----------
        var : PyExpr, optional
            기준 변수. None 이면 식에서 자동 탐지.

        Returns
        -------
        PyExpr
        """
        cdef extern from "cpp/capi.hpp" namespace "ml_laplace":
            size_t laplace_collect(size_t, size_t) except +
        cdef size_t var_h = 0
        if var is not None:
            var_h = (<PyExpr>var)._handle
        return PyExpr._wrap(laplace_collect(self._handle, var_h))

    # ------------------------------------------------------------------ Phase D: simplify (expand + cancel)
    def simplify(PyExpr self, var=None):
        """
        expand 후 cancel: 완전 단순화.

        Returns
        -------
        PyExpr
        """
        expanded = self.expand()
        return expanded.cancel(var)

    # ------------------------------------------------------------------ Phase D: Transfer function 편의 메서드
    def series_connect(PyExpr self, PyExpr G):
        """직렬 연결: self * G"""
        return self * G

    def parallel_connect(PyExpr self, PyExpr G):
        """병렬 연결: self + G"""
        return self + G

    def feedback(PyExpr self, H=None):
        """
        피드백 연결: self / (1 + self * H), 자동 simplify 적용.
        H=None 이면 단위 피드백 (H=1).

        G = 1/(s+1) → G.feedback() = 1/(s+2)

        Parameters
        ----------
        H : PyExpr, optional

        Returns
        -------
        PyExpr
        """
        from .laplace_ast import const as _const
        if H is None:
            H = _const(1.0)
        cdef PyExpr one_ = <PyExpr>(_const(1.0))
        cdef PyExpr H_ = <PyExpr>H
        raw = self / (one_ + self * H_)
        # 자동 단순화: expand → cancel
        try:
            return raw.simplify()
        except Exception:
            return raw

    # ------------------------------------------------------------------ Phase D: Frequency response
    def frequency_response(PyExpr self, omega):
        """
        H(jω) 계산.

        Parameters
        ----------
        omega : float 또는 iterable of float

        Returns
        -------
        complex 또는 list of complex
        """
        if not hasattr(self, '_fr_func') or self._fr_func is None:
            self._fr_func = self.lambdify(['s'], backend='cmath')
        import builtins
        if isinstance(omega, (int, float)):
            return self._fr_func(complex(0.0, omega))
        return [self._fr_func(complex(0.0, float(w))) for w in omega]

    def magnitude(PyExpr self, omega):
        """
        |H(jω)| 크기.

        Parameters
        ----------
        omega : float 또는 iterable

        Returns
        -------
        float 또는 list of float
        """
        r = self.frequency_response(omega)
        if isinstance(r, list):
            return [abs(x) for x in r]
        return abs(r)

    def phase(PyExpr self, omega):
        """
        ∠H(jω) 위상 [rad].

        Parameters
        ----------
        omega : float 또는 iterable

        Returns
        -------
        float 또는 list of float
        """
        import cmath
        r = self.frequency_response(omega)
        if isinstance(r, list):
            return [cmath.phase(x) for x in r]
        return cmath.phase(r)

    # ------------------------------------------------------------------ Partial fractions (Phase C)
    def partial_fractions(PyExpr self, var):
        """
        자신을 var에 대한 partial fraction 분해. 분해된 AST 반환.

        Parameters
        ----------
        var : PyExpr
            분해 기준 변수 (주로 s)

        Returns
        -------
        PyExpr
        """
        cdef extern from "cpp/capi.hpp" namespace "ml_laplace":
            size_t laplace_partial_fractions(size_t, size_t) except +
        cdef PyExpr v = <PyExpr>var
        cdef size_t result_h = laplace_partial_fractions(self._handle, v._handle)
        return PyExpr._wrap(result_h)

    # ------------------------------------------------------------------ 속성
    @property
    def node_type(PyExpr self):
        """NodeType 정수값 반환 (0=CONST,1=RATIONAL,2=VAR,3=SUM,4=MUL,5=POW,6=FUNC,7=NEG)"""
        return expr_node_type(self._handle)

    @property
    def cpp_hash(PyExpr self):
        """C++ 레벨 64비트 해시값"""
        return expr_hash(self._handle)


# ================================================================== 모듈 API

def symbol(name):
    """기호 변수 생성. symbol('x') → PyExpr for variable x"""
    cdef bytes nb = name.encode('utf-8') if isinstance(name, str) else name
    return PyExpr._wrap(pool_var(nb))

def const(value):
    """double 상수 생성"""
    return PyExpr._wrap(pool_make_const(<double>value))

def rational(num, den=1):
    """유리수 상수 생성"""
    return PyExpr._wrap(pool_make_rational(<int64_t>num, <int64_t>den))


# ================================================================== 공용 심볼

t = symbol('t')
s = symbol('s')

ZERO = PyExpr._wrap(pool_zero())
ONE  = PyExpr._wrap(pool_one())


# ================================================================== 수학 함수

def Sin(PyExpr arg):
    """sin(arg) — 심볼릭 사인"""
    return PyExpr._wrap(pool_make_func(FUNCID_SIN, arg._handle))

def Cos(PyExpr arg):
    """cos(arg) — 심볼릭 코사인"""
    return PyExpr._wrap(pool_make_func(FUNCID_COS, arg._handle))

def Tan(PyExpr arg):
    """tan(arg) — 심볼릭 탄젠트"""
    return PyExpr._wrap(pool_make_func(FUNCID_TAN, arg._handle))

def Arcsin(PyExpr arg):
    """arcsin(arg)"""
    return PyExpr._wrap(pool_make_func(FUNCID_ARCSIN, arg._handle))

def Arccos(PyExpr arg):
    """arccos(arg)"""
    return PyExpr._wrap(pool_make_func(FUNCID_ARCCOS, arg._handle))

def Arctan(PyExpr arg):
    """arctan(arg)"""
    return PyExpr._wrap(pool_make_func(FUNCID_ARCTAN, arg._handle))

def Sinh(PyExpr arg):
    """sinh(arg) — 심볼릭 쌍곡사인"""
    return PyExpr._wrap(pool_make_func(FUNCID_SINH, arg._handle))

def Cosh(PyExpr arg):
    """cosh(arg) — 심볼릭 쌍곡코사인"""
    return PyExpr._wrap(pool_make_func(FUNCID_COSH, arg._handle))

def Tanh(PyExpr arg):
    """tanh(arg) — 심볼릭 쌍곡탄젠트"""
    return PyExpr._wrap(pool_make_func(FUNCID_TANH, arg._handle))

def Exp(PyExpr arg):
    """exp(arg) — 심볼릭 지수함수"""
    return PyExpr._wrap(pool_make_func(FUNCID_EXP, arg._handle))

def Ln(PyExpr arg):
    """ln(arg) — 심볼릭 자연로그"""
    return PyExpr._wrap(pool_make_func(FUNCID_LN, arg._handle))

def Log(PyExpr arg):
    """log10(arg) — 심볼릭 상용로그"""
    return PyExpr._wrap(pool_make_func(FUNCID_LOG, arg._handle))

def Sqrt(PyExpr arg):
    """sqrt(arg) — 심볼릭 제곱근"""
    return PyExpr._wrap(pool_make_func(FUNCID_SQRT, arg._handle))

def Heaviside(PyExpr arg):
    """heaviside(arg) — Heaviside 계단 함수 u(arg).
    Laplace 변환: L{u(t-a)} = e^{-as}/s
    수치 평가: arg<0 → 0.0, arg=0 → 0.5, arg>0 → 1.0
    """
    return PyExpr._wrap(pool_make_heaviside(arg._handle))

def Dirac(PyExpr arg):
    """dirac(arg) — Dirac delta 함수 δ(arg).
    Laplace 변환: L{δ(t-a)} = e^{-as}
    수치 평가는 0.0 반환 (분포이므로 pointwise 평가 미지원).
    """
    return PyExpr._wrap(pool_make_dirac(arg._handle))

# alias
H = Heaviside


# ================================================================== Pool 통계

class ExprPoolStats:
    """메모리 풀 통계 접근 (싱글턴 래퍼)"""

    @staticmethod
    def total_nodes():
        """풀에 등록된 총 AST 노드 수"""
        return pool_total_nodes()

    @staticmethod
    def intern_hits():
        """hash-consing 적중 횟수"""
        return pool_intern_hits()

    @staticmethod
    def reset():
        """풀 전체 초기화 (주의: 기존 PyExpr 객체 dangling)"""
        pool_reset()
        _pyexpr_cache.clear()


# ================================================================== __all__

__all__ = [
    'PyExpr',
    'symbol', 'const', 'rational',
    't', 's', 'ZERO', 'ONE',
    'Sin', 'Cos', 'Tan',
    'Arcsin', 'Arccos', 'Arctan',
    'Sinh', 'Cosh', 'Tanh',
    'Exp', 'Ln', 'Log', 'Sqrt',
    'Heaviside', 'Dirac', 'H',
    'ExprPoolStats',
]
