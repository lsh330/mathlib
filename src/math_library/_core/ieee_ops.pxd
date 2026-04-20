# cython: language_level=3
# ieee_ops.pxd — IEEE 754 비트 조작 primitive 선언

cpdef double ceil(double x) noexcept
cpdef double floor(double x) noexcept
cpdef double trunc(double x) noexcept
cpdef double fmod(double x, double y) noexcept
cpdef double copysign(double x, double y) noexcept
cpdef double remainder(double x, double y) noexcept
cpdef object modf(double x)
cpdef double nextafter(double x, double y) noexcept
cpdef double ulp(double x) noexcept
