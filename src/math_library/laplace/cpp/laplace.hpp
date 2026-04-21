#pragma once
// laplace.hpp — Laplace 변환 엔진 최상위 공개 인터페이스
// Phase B: forward_transform 진입점 + LaTeX 출력 + 기호 치환
#include "expr.hpp"
#include "pool.hpp"
#include "rules.hpp"

namespace ml_laplace {

// ------------------------------------------------------------------ LaTeX 출력
// expr AST를 LaTeX 문자열로 변환
// - Const/Rational: 숫자 리터럴
// - Var: 변수 이름
// - Pow(base, n): base^{n}
// - Sum: a + b + ...
// - Mul with Pow(x,-1): \frac{num}{den} 형태 감지
// - Func: \sin(...), \cos(...), \exp(...) 등
std::string to_latex(ExprPtr expr);

// ------------------------------------------------------------------ 기호 치환 (capi 래퍼)
// old_var_name 에 해당하는 변수를 new_expr로 치환
// Python 측에서 size_t 핸들로 사용하기 위한 래퍼
inline ExprPtr symbolic_substitute(ExprPtr expr,
                                    const std::string& old_var_name,
                                    ExprPtr new_expr) {
    ExprPtr old_var = ExprPool::instance().make_var(old_var_name);
    return substitute_var(expr, old_var, new_expr);
}

} // namespace ml_laplace
