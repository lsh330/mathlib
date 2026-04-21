#pragma once
#include <cstdint>
#include <vector>

// hash.hpp — FNV-1a 64비트 해시 유틸리티
namespace ml_laplace {

constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

// 바이트 시퀀스에 대한 FNV-1a 64비트 해시 (seed 지원)
inline uint64_t fnv1a_64(const uint8_t* data, size_t len,
                          uint64_t seed = FNV_OFFSET) noexcept {
    uint64_t h = seed;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(data[i]);
        h *= FNV_PRIME;
    }
    return h;
}

// 임의 POD 값에 대한 FNV-1a
template <typename T>
inline uint64_t fnv1a_val(T val, uint64_t seed = FNV_OFFSET) noexcept {
    return fnv1a_64(reinterpret_cast<const uint8_t*>(&val), sizeof(T), seed);
}

// 자식 노드 해시 목록 + NodeType 번호를 조합하여 복합 해시 생성
// type_byte: NodeType을 uint8_t로 캐스팅한 값
inline uint64_t combine_hashes(uint8_t type_byte,
                                const std::vector<uint64_t>& child_hashes) noexcept {
    // 루트: type_byte를 시드로 사용
    uint64_t h = fnv1a_val(type_byte);
    for (uint64_t ch : child_hashes) {
        h = fnv1a_64(reinterpret_cast<const uint8_t*>(&ch), 8, h);
    }
    return h;
}

// 가변 인자 버전 (child_hashes 벡터 없이 직접 조합)
inline uint64_t mix(uint64_t a, uint64_t b) noexcept {
    return fnv1a_64(reinterpret_cast<const uint8_t*>(&b), 8, a);
}

} // namespace ml_laplace
