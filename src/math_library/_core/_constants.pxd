# cython: language_level=3
# _constants.pxd — 수학 상수 선언

# M_PI, M_E 값을 컴파일 타임 상수로 사용하기 위해 외부 C 선언 없이
# Python cpdef 함수로 재노출하는 방식 사용

cpdef double pi() noexcept
cpdef double e() noexcept
cpdef double epsilon() noexcept
