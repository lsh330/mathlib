#pragma once
// inverse.hpp — 역 Laplace 변환 (Phase C)
// F(s) (유리함수) → f(t)

#include "expr.hpp"
#include "pool.hpp"
#include "polynomial.hpp"
#include "partial.hpp"
#include <vector>
#include <complex>

namespace ml_laplace {

// ================================================================== inverse_transform
// F(s) → f(t), 유리함수만 지원 (Phase C)
// 실패 시 std::runtime_error
ExprPtr inverse_transform(ExprPtr F, ExprPtr s_var, ExprPtr t_var);

// ================================================================== compute_poles / compute_zeros
std::vector<std::complex<double>> compute_poles(ExprPtr F, ExprPtr s_var);
std::vector<std::complex<double>> compute_zeros(ExprPtr F, ExprPtr s_var);

// ================================================================== final_value / initial_value
// final_value: lim_{s->0} s*F(s)  (valid=true if all poles in open LHP)
double final_value(ExprPtr F, ExprPtr s_var, bool& valid);

// initial_value: lim_{s->inf} s*F(s)
double initial_value(ExprPtr F, ExprPtr s_var, bool& valid);

} // namespace ml_laplace
