# cython: language_level=3
# multi_arg.pxd — atan2, hypot, dist 선언

cpdef double atan2(double y, double x) noexcept
cpdef object hypot(object a, object b)
cpdef double dist(object p, object q)
