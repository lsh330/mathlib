# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# predicates.pyx
#
# isnan, isinf, isfinite, isclose — IEEE 754 술어 함수

from libc.math cimport INFINITY
from ._helpers cimport double_to_bits
from libc.stdint cimport uint64_t


cpdef bint isnan(double x) noexcept:
    """isnan(x): x가 NaN이면 True."""
    return x != x


cpdef bint isinf(double x) noexcept:
    """isinf(x): x가 ±inf이면 True."""
    cdef uint64_t bits = double_to_bits(x)
    return (bits & 0x7FFFFFFFFFFFFFFFULL) == 0x7FF0000000000000ULL


cpdef bint isfinite(double x) noexcept:
    """isfinite(x): x가 NaN도 inf도 아니면 True."""
    cdef uint64_t bits = double_to_bits(x)
    return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL


cpdef bint isclose(double a, double b, double rel_tol=1e-9, double abs_tol=0.0):
    """
    isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    |a-b| <= max(rel_tol * max(|a|, |b|), abs_tol)
    """
    cdef double diff, maxab
    if a == b:
        return True
    # 양쪽 모두 inf면 True (위에서 처리), 한쪽만 inf면 False
    if isinf(a) or isinf(b):
        return False
    diff = a - b
    if diff < 0.0:
        diff = -diff
    maxab = a if a >= 0.0 else -a
    cdef double ab = b if b >= 0.0 else -b
    if ab > maxab:
        maxab = ab
    cdef double tol = rel_tol * maxab
    if abs_tol > tol:
        tol = abs_tol
    return diff <= tol
