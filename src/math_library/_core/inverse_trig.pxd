# cython: language_level=3
# inverse_trig.pxd — arcsin/arccos/arctan/arcsec/arccosec/arccotan 선언

cpdef double arcsin(double x) noexcept
cpdef double arccos(double x) noexcept
cpdef double arctan(double x) noexcept
cpdef double arcsec(double x) noexcept
cpdef double arccosec(double x) noexcept
cpdef double arccotan(double x) noexcept

# 복소수 auto-dispatch (방향 A)
cpdef object arcsin_dispatch(object x)
cpdef object arccos_dispatch(object x)
cpdef object arctan_dispatch(object x)
cpdef object arcsec_dispatch(object x)
cpdef object arccosec_dispatch(object x)
cpdef object arccotan_dispatch(object x)
