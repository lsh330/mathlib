#pragma once
#include <unordered_map>
#include <string>

// subst.hpp — 변수 이름 → double 매핑 (수치 평가 시 사용)
namespace ml_laplace {

using SubstMap = std::unordered_map<std::string, double>;

} // namespace ml_laplace
