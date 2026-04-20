# cython: language_level=3
# aggregate.pxd — fsum, prod 선언

cpdef double fsum(object iterable)
cpdef object prod(object iterable, object start=*)
