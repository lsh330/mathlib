#pragma once
// rules.hpp — Laplace 변환 규칙: 패턴 매처 + 디스패처
// Phase B: forward_transform 진입점과 내부 헬퍼 선언
#include "expr.hpp"
#include "pool.hpp"
#include <unordered_map>

namespace ml_laplace {

// ------------------------------------------------------------------ 최상위 변환 진입점
// expr: t-영역 식,  t_var/s_var: Pool에 등록된 공용 심볼
// 변환 불가 시 nullptr 대신 std::runtime_error 예외 발생 (Phase B 정책)
ExprPtr forward_transform(ExprPtr expr, ExprPtr t_var, ExprPtr s_var);

// ------------------------------------------------------------------ 변환 캐시 (thread_local)
extern thread_local std::unordered_map<ExprPtr, ExprPtr> _transform_cache;

// ================================================================== 내부 패턴 헬퍼 (rules.cpp private, 헤더에서 선언)

// e = a*t 형태인지 확인하고 계수 coef 반환
// e == t_var 이면 coef = 1.0
bool is_linear_in(ExprPtr e, ExprPtr t_var, double& coef) noexcept;

// e = t^n (n ≥ 0 정수) 형태 확인, n 반환
bool is_polynomial_power(ExprPtr e, ExprPtr t_var, int& n) noexcept;

// mul_expr 에서 exp(a*t) 인수를 분리
// mul_expr 자체가 exp(a*t) 이면 rest = nullptr (= 1)
// 성공 시 a 와 나머지 곱 rest 반환 (rest: exp 제외 나머지 곱, nullptr=1)
bool split_exp_factor(ExprPtr expr, ExprPtr t_var,
                      double& a, ExprPtr& rest) noexcept;

// ================================================================== 프리미티브 변환
// 반환: F(s) ExprPtr, 변환 실패 시 nullptr
ExprPtr try_transform_primitive(ExprPtr expr, ExprPtr t_var, ExprPtr s_var);

// ================================================================== 속성 변환
ExprPtr try_linearity(ExprPtr sum_expr,  ExprPtr t_var, ExprPtr s_var);
ExprPtr try_s_shift  (ExprPtr expr,      ExprPtr t_var, ExprPtr s_var);
ExprPtr try_freq_diff(ExprPtr mul_expr,  ExprPtr t_var, ExprPtr s_var);

// ================================================================== s-도메인 미분 (frequency differentiation 용)
// F(s) 를 s에 대해 한 번 미분: F'(s)
// Laplace 출력은 유리함수이므로 제한적 chain/quotient/power/sum/product rule
ExprPtr diff_wrt_s(ExprPtr F, ExprPtr s_var);

// ================================================================== 일반 심볼릭 미분
// expr 을 var 에 대해 한 번 미분 (d/dvar [expr])
// 지원 규칙: Const/Rational→0, Var, Neg, Sum, Mul(product), Pow(constant exp),
//            Func(SIN/COS/TAN/ARCTAN/SINH/COSH/TANH/EXP/LN/SQRT)
// 변수 var 외 다른 변수는 상수 취급 (→0)
ExprPtr differentiate(ExprPtr expr, ExprPtr var);

// ================================================================== 기호 치환 (s → expr)
// expr 내 s_var 을 new_s 로 치환하여 새 ExprPtr 반환
ExprPtr substitute_var(ExprPtr expr, ExprPtr old_var, ExprPtr new_expr);

} // namespace ml_laplace
