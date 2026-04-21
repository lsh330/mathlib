# cython: language_level=3
# laplace_ast.pxd — PyExpr cdef class 선언 (외부 cimport용)

cdef class PyExpr:
    cdef size_t _handle
    cdef object __weakref__
    cdef object _fr_func   # frequency response lambdify 캐시

    @staticmethod
    cdef PyExpr _wrap(size_t h)
