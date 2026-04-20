# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# aggregate.pyx
#
# fsum (Neumaier 보정합), prod

from libc.math cimport fabs


# ------------------------------------------------------------------ fsum (Neumaier algorithm)

cpdef double fsum(object iterable):
    """
    fsum(iterable): Neumaier 보정합.
    Kahan 개선판: 누적 보정 항을 올바르게 처리.
    fsum([1e20, 1, -1e20]) = 1.0 (정확).
    """
    cdef double s = 0.0   # 누적합
    cdef double c = 0.0   # 보정항
    cdef double x, t

    for item in iterable:
        x = <double>item
        t = s + x
        if fabs(s) >= fabs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t

    return s + c


# ------------------------------------------------------------------ prod

cpdef object prod(object iterable, object start=1):
    """
    prod(iterable, start=1): 모든 요소의 곱.
    start: 초기 값 (기본 1).
    정수 요소로만 구성되면 Python int로 정확히 계산.
    """
    cdef object result = start
    for item in iterable:
        result = result * item
    return result
