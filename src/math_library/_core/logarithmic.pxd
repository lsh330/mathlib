# cython: language_level=3
# logarithmic.pxd — ln, log 선언

cdef double _ln_inline(double x) noexcept nogil
cpdef double ln(double x) noexcept
cpdef double log(double base, double x) noexcept

# 복소수 auto-dispatch (방향 A)
cpdef object ln_dispatch(object x)
cpdef object log_dispatch(object base, object x)
