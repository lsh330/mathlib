# cython: language_level=3
# predicates.pxd — isnan, isinf, isfinite, isclose 선언

cpdef bint isnan(double x) noexcept
cpdef bint isinf(double x) noexcept
cpdef bint isfinite(double x) noexcept
cpdef bint isclose(double a, double b, double rel_tol=*, double abs_tol=*)
