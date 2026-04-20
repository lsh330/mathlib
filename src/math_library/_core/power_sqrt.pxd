# cython: language_level=3
# power_sqrt.pxd — power, sqrt 선언

from libc.math cimport sqrt as _libc_sqrt

cdef double _sqrt_c(double x) noexcept nogil
cpdef double sqrt(double x) noexcept
cpdef double power(double base, double exponent) noexcept

# 복소수 auto-dispatch (방향 A)
cpdef object sqrt_dispatch(object x)
cpdef object power_dispatch(object base, object exponent)
