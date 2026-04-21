#pragma once
// partial.hpp — Partial Fraction Decomposition (Phase C)
// F(s) = P(s)/Q(s) → sum of c_k / (s - p_k)^m_k

#include "polynomial.hpp"
#include "expr.hpp"
#include "pool.hpp"
#include <vector>
#include <complex>
#include <string>

namespace ml_laplace {

// ============================================================ PartialFractionTerm

struct PartialFractionTerm {
    std::complex<double> pole;      // 극점 값
    int multiplicity;               // 중복도 m
    // coefficients[i] = c_{m-i} (높은 중복도부터)
    // c_m, c_{m-1}, ..., c_1 대응
    // 즉 coefficients[0] = 1/(s-p)^m 의 계수, coefficients[m-1] = 1/(s-p) 의 계수
    std::vector<std::complex<double>> coefficients;
};

// ============================================================ partial_fractions
// F(s) = P(s)/Q(s) → partial fraction 분해
// deg(P) < deg(Q) 가정 (proper fraction). 아니면 다항식 부분을 먼저 분리.

std::vector<PartialFractionTerm> partial_fractions(
    const RationalFunction& F,
    double tol = 1e-9
);

// ============================================================ partial_fractions_to_expr
// partial fraction 결과를 실수 유리함수 AST 합으로 변환
// 복소 켤레 쌍을 실수화 (σ ± jω → 2차 분모)

ExprPtr partial_fractions_to_expr(
    const std::vector<PartialFractionTerm>& terms,
    ExprPtr s_var,
    double tol = 1e-9
);

} // namespace ml_laplace
