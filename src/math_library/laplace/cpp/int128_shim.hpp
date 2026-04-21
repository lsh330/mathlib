// int128_shim.hpp — 크로스-컴파일러 signed 128-bit 정수 alias
//
// GCC/Clang은 `__int128` 네이티브 키워드를 제공하지만 MSVC는 지원하지 않는다.
// 또한 `__`로 시작하는 식별자는 C++ 표준에서 implementation-reserved이므로
// 사용자 코드에서 `using __int128 = ...`를 직접 선언할 수 없다.
//
// 본 헤더는 `ml_laplace::ml_int128` 이라는 공용 alias를 제공한다:
//   - MSVC           : 수동 정의된 signed 128-bit 구조체
//   - GCC / Clang    : ::__int128 (네이티브)
//
// 제공 연산 (pool.cpp 요구사항):
//   - int64_t → 128 암묵 변환 (sign-extend)
//   - 128 → int64_t 명시 변환 (호출부에서 범위 체크 전제)
//   - +, -(unary), -(binary), *, *=
//   - ==, !=, <, <=, >, >=
//   - constexpr 생성자/비교 (constexpr MAX64/MIN64 상수 지원)
//
// 원리: 2's complement {uint64 lo, int64 hi}. 128×128 하위 128비트 곱셈은
//       네이티브 __int128과 비트-동일 wraparound 결과를 낸다.

#pragma once

#include <cstdint>

namespace ml_laplace {

#ifdef _MSC_VER

struct Int128 {
    std::uint64_t lo;
    std::int64_t  hi;

    constexpr Int128() noexcept : lo(0), hi(0) {}
    constexpr Int128(int v) noexcept
        : lo(static_cast<std::uint64_t>(static_cast<std::int64_t>(v))),
          hi(v < 0 ? -1 : 0) {}
    constexpr Int128(long v) noexcept
        : lo(static_cast<std::uint64_t>(static_cast<std::int64_t>(v))),
          hi(v < 0 ? -1 : 0) {}
    constexpr Int128(long long v) noexcept
        : lo(static_cast<std::uint64_t>(v)),
          hi(v < 0 ? -1 : 0) {}
    constexpr Int128(unsigned v) noexcept
        : lo(static_cast<std::uint64_t>(v)), hi(0) {}
    constexpr Int128(unsigned long v) noexcept
        : lo(static_cast<std::uint64_t>(v)), hi(0) {}
    constexpr Int128(unsigned long long v) noexcept
        : lo(static_cast<std::uint64_t>(v)), hi(0) {}
    constexpr Int128(std::uint64_t l, std::int64_t h) noexcept
        : lo(l), hi(h) {}

    constexpr explicit operator std::int64_t() const noexcept {
        return static_cast<std::int64_t>(lo);
    }
};

constexpr bool operator==(const Int128& a, const Int128& b) noexcept {
    return a.lo == b.lo && a.hi == b.hi;
}
constexpr bool operator!=(const Int128& a, const Int128& b) noexcept {
    return !(a == b);
}
constexpr bool operator<(const Int128& a, const Int128& b) noexcept {
    return (a.hi != b.hi) ? (a.hi < b.hi) : (a.lo < b.lo);
}
constexpr bool operator>(const Int128& a, const Int128& b) noexcept  { return b < a; }
constexpr bool operator<=(const Int128& a, const Int128& b) noexcept { return !(b < a); }
constexpr bool operator>=(const Int128& a, const Int128& b) noexcept { return !(a < b); }

constexpr Int128 operator-(const Int128& a) noexcept {
    std::uint64_t nlo = ~a.lo + 1ULL;
    std::int64_t  nhi = static_cast<std::int64_t>(
                            ~static_cast<std::uint64_t>(a.hi))
                        + (nlo == 0 ? 1 : 0);
    return Int128(nlo, nhi);
}

constexpr Int128 operator+(const Int128& a, const Int128& b) noexcept {
    std::uint64_t rlo = a.lo + b.lo;
    std::int64_t carry = (rlo < a.lo) ? 1 : 0;
    std::int64_t rhi = a.hi + b.hi + carry;
    return Int128(rlo, rhi);
}
constexpr Int128 operator-(const Int128& a, const Int128& b) noexcept {
    return a + (-b);
}

// 128×128 → 하위 128
constexpr Int128 operator*(const Int128& a, const Int128& b) noexcept {
    const std::uint64_t a_lo_lo = a.lo & 0xFFFFFFFFULL;
    const std::uint64_t a_lo_hi = a.lo >> 32;
    const std::uint64_t b_lo_lo = b.lo & 0xFFFFFFFFULL;
    const std::uint64_t b_lo_hi = b.lo >> 32;

    const std::uint64_t t_ll = a_lo_lo * b_lo_lo;
    const std::uint64_t t_lh = a_lo_lo * b_lo_hi;
    const std::uint64_t t_hl = a_lo_hi * b_lo_lo;
    const std::uint64_t t_hh = a_lo_hi * b_lo_hi;

    const std::uint64_t mid   = (t_ll >> 32) + (t_lh & 0xFFFFFFFFULL) + (t_hl & 0xFFFFFFFFULL);
    const std::uint64_t r_lo  = (t_ll & 0xFFFFFFFFULL) | (mid << 32);
    const std::uint64_t ll_hi = t_hh + (t_lh >> 32) + (t_hl >> 32) + (mid >> 32);

    const std::uint64_t ub_hi = static_cast<std::uint64_t>(b.hi);
    const std::uint64_t ua_hi = static_cast<std::uint64_t>(a.hi);
    const std::uint64_t r_hi_u = ll_hi + a.lo * ub_hi + ua_hi * b.lo;
    return Int128(r_lo, static_cast<std::int64_t>(r_hi_u));
}

constexpr Int128& operator*=(Int128& a, const Int128& b) noexcept {
    a = a * b;
    return a;
}

using ml_int128 = Int128;

#else  // GCC / Clang: 네이티브 __int128

using ml_int128 = __int128;

#endif  // _MSC_VER

} // namespace ml_laplace
