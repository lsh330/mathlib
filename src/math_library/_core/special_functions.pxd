# cython: language_level=3
# special_functions.pxd — erf, erfc, lgamma 선언

cpdef double erf(double x) noexcept
cpdef double erfc(double x) noexcept
cpdef double lgamma(double x) noexcept
