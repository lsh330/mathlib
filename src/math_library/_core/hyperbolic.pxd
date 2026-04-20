# cython: language_level=3
# hyperbolic.pxd — 쌍곡함수 선언

cpdef double hypersin(double x) noexcept
cpdef double hypercos(double x) noexcept
cpdef double hypertan(double x) noexcept
cpdef double hypersec(double x) noexcept
cpdef double hypercosec(double x) noexcept
cpdef double hypercotan(double x) noexcept

# 복소수 auto-dispatch (방향 A)
cpdef object hypersin_dispatch(object x)
cpdef object hypercos_dispatch(object x)
cpdef object hypertan_dispatch(object x)
cpdef object hypersec_dispatch(object x)
cpdef object hypercosec_dispatch(object x)
cpdef object hypercotan_dispatch(object x)
