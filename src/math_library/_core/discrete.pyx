# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# discrete.pyx
#
# factorial, comb, perm, isqrt — 정수 이산 수학 함수

from libc.math cimport sqrt as _lsqrt


# ------------------------------------------------------------------ factorial

cpdef object factorial(object n):
    """
    factorial(n) = n!  for non-negative integer n.
    큰 n에도 Python int arbitrary precision으로 정확히 계산.
    실수/복소수 n: gamma(n+1) 사용을 권장 (docstring 안내).
    """
    if not isinstance(n, (int,)):
        raise TypeError(
            f"factorial() requires an integer argument, got {type(n).__name__}. "
            "For non-integer n, use math_library.gamma(n+1)."
        )
    cdef object ni = int(n)
    if ni < 0:
        raise ValueError("factorial() of negative value is undefined")
    cdef object result = 1
    cdef object i
    for i in range(2, ni + 1):
        result = result * i
    return result


# ------------------------------------------------------------------ comb

cpdef object comb(object n, object k):
    """
    comb(n, k) = n! / (k! * (n-k)!)  for non-negative integers.
    중간 overflow 방지: 분자/분모를 GCD로 약분하며 누적.
    """
    if not isinstance(n, (int,)) or not isinstance(k, (int,)):
        raise TypeError("comb() requires integer arguments")
    cdef object ni = int(n)
    cdef object ki = int(k)
    if ni < 0 or ki < 0:
        raise ValueError("comb() requires non-negative integers")
    if ki > ni:
        return 0
    if ki == 0 or ki == ni:
        return 1
    # 대칭성 이용: k = min(k, n-k)
    if ki > ni - ki:
        ki = ni - ki
    # 점진 계산: comb(n,k) = product_{i=0}^{k-1} (n-i)/(i+1)
    cdef object result = 1
    cdef object i
    for i in range(ki):
        result = result * (ni - i) // (i + 1)
    return result


# ------------------------------------------------------------------ perm

cpdef object perm(object n, object k=None):
    """
    perm(n, k=None) = n! / (n-k)!  — k개 순열.
    k=None이면 n! (전순열).
    """
    if not isinstance(n, (int,)):
        raise TypeError("perm() requires integer n")
    cdef object ni = int(n)
    if ni < 0:
        raise ValueError("perm() requires non-negative integer n")
    if k is None:
        return factorial(ni)
    if not isinstance(k, (int,)):
        raise TypeError("perm() requires integer k")
    cdef object ki = int(k)
    if ki < 0:
        raise ValueError("perm() requires non-negative integer k")
    if ki > ni:
        return 0
    # product: n * (n-1) * ... * (n-k+1)
    cdef object result = 1
    cdef object i
    for i in range(ki):
        result = result * (ni - i)
    return result


# ------------------------------------------------------------------ isqrt

cpdef object isqrt(object n):
    """
    isqrt(n) = floor(sqrt(n)) for non-negative integer n.
    Newton's method on integers.
    """
    if not isinstance(n, (int,)):
        raise TypeError("isqrt() requires an integer argument")
    cdef object ni = int(n)
    if ni < 0:
        raise ValueError("isqrt() of negative value is undefined")
    if ni == 0:
        return 0
    # 초기 추정: Python float sqrt 사용 (1e15 이하에서 정확)
    cdef object x = int(_lsqrt(<double>ni))
    # Newton 반복으로 수렴 보장 (arbitrary precision)
    while True:
        x1 = (x + ni // x) // 2
        if x1 >= x:
            return x
        x = x1
