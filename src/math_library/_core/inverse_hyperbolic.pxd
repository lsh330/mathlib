# cython: language_level=3
# inverse_hyperbolic.pxd — arc_hypersin/cos/tan/sec/cosec/cotan 선언

cdef double _arc_hypersin_real(double x) noexcept nogil
cdef double _arc_hypercos_real(double x) noexcept nogil
cdef double _arc_hypertan_real(double x) noexcept nogil
cdef double complex _arc_hypersin_complex(double complex z) noexcept nogil
cdef double complex _arc_hypercos_complex(double complex z) noexcept nogil
cdef double complex _arc_hypertan_complex(double complex z) noexcept nogil

cpdef object arc_hypersin(object x)
cpdef object arc_hypercos(object x)
cpdef object arc_hypertan(object x)
cpdef object arc_hypersec(object x)
cpdef object arc_hypercosec(object x)
cpdef object arc_hypercotan(object x)
