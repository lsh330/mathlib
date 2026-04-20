# cython: language_level=3
# exponential.pxd — exp, expm1 선언

cdef double _exp_inline(double x) noexcept nogil
cdef double _expm1_inline(double x) noexcept nogil
cpdef double exp(double x) noexcept

# 복소수 auto-dispatch (방향 A)
cpdef object exp_dispatch(object x)
