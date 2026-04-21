# cython: language_level=3
# laplace_ast.pxd — 외부 모듈에서 PyExpr을 cdef 레벨로 사용할 때 참조
# (현재 Phase A에서는 laplace_ast.pyx 자체가 선언을 내포 — 외부 전용)

from libc.stdint cimport int64_t, uint64_t
from libcpp.string cimport string

cdef extern from "cpp/subst.hpp" namespace "ml_laplace":
    cdef cppclass SubstMap "ml_laplace::SubstMap":
        SubstMap() except +
        void clear()
        size_t size()
        double& operator[](const string& key) except +

cdef extern from "cpp/expr.hpp" namespace "ml_laplace":
    cdef enum NodeType "ml_laplace::NodeType":
        CONST    "ml_laplace::NodeType::CONST"
        RATIONAL "ml_laplace::NodeType::RATIONAL"
        VAR      "ml_laplace::NodeType::VAR"
        SUM      "ml_laplace::NodeType::SUM"
        MUL      "ml_laplace::NodeType::MUL"
        POW      "ml_laplace::NodeType::POW"
        FUNC     "ml_laplace::NodeType::FUNC"
        NEG      "ml_laplace::NodeType::NEG"

    cdef enum FuncId "ml_laplace::FuncId":
        SIN    "ml_laplace::FuncId::SIN"
        COS    "ml_laplace::FuncId::COS"
        TAN    "ml_laplace::FuncId::TAN"
        ARCSIN "ml_laplace::FuncId::ARCSIN"
        ARCCOS "ml_laplace::FuncId::ARCCOS"
        ARCTAN "ml_laplace::FuncId::ARCTAN"
        SINH_  "ml_laplace::FuncId::SINH"
        COSH_  "ml_laplace::FuncId::COSH"
        TANH_  "ml_laplace::FuncId::TANH"
        EXP    "ml_laplace::FuncId::EXP"
        LN     "ml_laplace::FuncId::LN"
        LOG    "ml_laplace::FuncId::LOG"
        SQRT_  "ml_laplace::FuncId::SQRT"

    cdef cppclass Expr "ml_laplace::Expr":
        NodeType type()
        uint64_t hash() nogil
        double evalf(const SubstMap&) except +
        string to_string() except +
        const Expr* substitute(const SubstMap&) except +

    ctypedef const Expr* ExprPtr

cdef extern from "cpp/pool.hpp" namespace "ml_laplace":
    cdef cppclass ExprPool "ml_laplace::ExprPool":
        @staticmethod
        ExprPool& instance() except +
        ExprPtr make_const(double) except +
        ExprPtr make_rational(long long, long long) except +
        ExprPtr make_var(const string&) except +
        ExprPtr make_func(FuncId, ExprPtr) except +
        ExprPtr make_neg(ExprPtr) except +
        ExprPtr add(ExprPtr, ExprPtr) except +
        ExprPtr sub(ExprPtr, ExprPtr) except +
        ExprPtr mul(ExprPtr, ExprPtr) except +
        ExprPtr div(ExprPtr, ExprPtr) except +
        ExprPtr pow(ExprPtr, ExprPtr) except +
        ExprPtr zero() nogil
        ExprPtr one() nogil
        ExprPtr var(const string&) except +
        size_t total_nodes() nogil
        size_t intern_hits() nogil
        void reset() except +

# Phase B: capi.hpp Phase B 함수 선언
cdef extern from "cpp/capi.hpp" namespace "ml_laplace":
    size_t laplace_forward_transform(size_t, size_t, size_t) except +
    void   expr_to_latex_buf(size_t, char*, size_t) except +
    size_t expr_symbolic_substitute(size_t, const char*, size_t) except +
    void   laplace_clear_cache() except +

# Phase B에서 이 .pxd를 cimport할 때 PyExpr을 사용하려면
# laplace_ast.pyx의 PyExpr cdef class를 cimport하면 됨.
