#pragma once
// simplify.hpp — Phase D: AST 단순화 (expand + cancel)
// expand  : Mul(Sum,...) 분배 법칙 전개, Pow(Sum, n) 이항 전개 (n <= 5)
// cancel  : 유리식 약분 (RationalFunction::simplify() 재활용)

#include "expr.hpp"
#include "pool.hpp"
#include "polynomial.hpp"

namespace ml_laplace {

// Mul(a+b, c+d) → ac + ad + bc + bd 형태로 재귀 전개
// Pow(Sum, n) 이항 전개 (n <= 5 한정, n 이 정수 양수)
ExprPtr expand(ExprPtr e);

// 유리식 약분: num/den 에서 GCD 소거
// e 가 유리식이 아닌 경우 그대로 반환
// var_h: 기준 변수 핸들 (s 등); 0이면 식에서 자동 탐지
ExprPtr cancel(ExprPtr e, ExprPtr var);

// 자동 변수 탐지 버전 (VAR 노드를 DFS로 찾아 사용)
ExprPtr cancel_auto(ExprPtr e);

} // namespace ml_laplace
