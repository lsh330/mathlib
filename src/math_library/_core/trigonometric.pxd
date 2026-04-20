# cython: language_level=3
# trigonometric.pxd — sin/cos/tan/sec/cosec/cotan 선언

cdef double _sin_kernel(double x, double y) noexcept nogil
cdef double _cos_kernel(double x, double y) noexcept nogil
cdef double _tan_kernel(double x, double y, int iy) noexcept nogil

cpdef double sin(double x) noexcept
cpdef double cos(double x) noexcept
cpdef double tan(double x) noexcept
cpdef double sec(double x) noexcept
cpdef double cosec(double x) noexcept
cpdef double cotan(double x) noexcept

# 복소수 auto-dispatch (방향 A)
cpdef object sin_dispatch(object x)
cpdef object cos_dispatch(object x)
cpdef object tan_dispatch(object x)
cpdef object sec_dispatch(object x)
cpdef object cosec_dispatch(object x)
cpdef object cotan_dispatch(object x)
