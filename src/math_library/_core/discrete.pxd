# cython: language_level=3
# discrete.pxd — factorial, comb, perm, isqrt 선언

cpdef object factorial(object n)
cpdef object comb(object n, object k)
cpdef object perm(object n, object k=*)
cpdef object isqrt(object n)
