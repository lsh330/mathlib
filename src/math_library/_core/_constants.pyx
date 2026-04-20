# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# _constants.pyx — pi, e, epsilon (bit-exact IEEE 754 상수)

# 이 상수들은 musl/fdlibm에서 사용하는 bit-exact 값을 직접 사용

cpdef double pi() noexcept:
    """원주율 π (bit-exact: 3.14159265358979323846...)"""
    return 3.141592653589793238462643383279502884197e+00

cpdef double e() noexcept:
    """자연상수 e (bit-exact: 2.71828182845904523536...)"""
    return 2.718281828459045235360287471352662497757e+00

cpdef double epsilon() noexcept:
    """machine epsilon (double precision: 2^-52 ≈ 2.220446049250313e-16)"""
    return 2.220446049250313080847263336181640625e-16
