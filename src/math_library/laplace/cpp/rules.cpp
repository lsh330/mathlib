// rules.cpp — Laplace 변환 규칙 구현
// Phase B: 6 primitives + 6 properties (linearity, s-shift, t-shift, t-scaling,
//          frequency differentiation, time differentiation)
// Phase B 스코프: Linearity, s-shift, frequency diff 완전 구현
//                 t-scaling, time-diff는 기본 구조만 (확장 예비)
#include "rules.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cassert>

namespace ml_laplace {

// ------------------------------------------------------------------ 캐시
thread_local std::unordered_map<ExprPtr, ExprPtr> _transform_cache;

// ------------------------------------------------------------------ 전방 선언 (내부)
static ExprPtr _transform_impl(ExprPtr expr, ExprPtr t_var, ExprPtr s_var);

// ================================================================== 팩토리 단축 (코드 가독성)

static inline ExprPool& P() { return ExprPool::instance(); }
static inline ExprPtr cR(int64_t n, int64_t d=1) { return P().make_rational(n, d); }
static inline ExprPtr cD(double v)                { return P().make_const(v); }

// ================================================================== numeric 추출 헬퍼

static bool get_numeric(ExprPtr e, double& v) noexcept {
    if (e->type() == NodeType::CONST) {
        v = static_cast<const Const*>(e)->value(); return true;
    }
    if (e->type() == NodeType::RATIONAL) {
        v = static_cast<const Rational*>(e)->as_double(); return true;
    }
    return false;
}

// ================================================================== is_linear_in
// e = coef * t_var  (coef는 수치 상수)
// e == t_var → coef = 1
// e == Mul([num, t_var]) 또는 Mul([t_var, num]) → coef = num
// 단, 다항 패턴 t^n (n > 1)은 false 반환 (is_polynomial_power 로 처리)

bool is_linear_in(ExprPtr e, ExprPtr t_var, double& coef) noexcept {
    if (e == t_var) {
        coef = 1.0; return true;
    }
    // Neg(t_var) → coef = -1
    if (e->type() == NodeType::NEG) {
        const Neg* neg = static_cast<const Neg*>(e);
        if (neg->operand() == t_var) {
            coef = -1.0; return true;
        }
    }
    if (e->type() != NodeType::MUL) return false;
    const Mul* m = static_cast<const Mul*>(e);
    const auto& ops = m->operands();
    if (ops.size() != 2) return false;
    double v = 0.0;
    // [num, t_var]
    if (get_numeric(ops[0], v) && ops[1] == t_var) {
        coef = v; return true;
    }
    // [t_var, num]
    if (ops[0] == t_var && get_numeric(ops[1], v)) {
        coef = v; return true;
    }
    return false;
}

// ================================================================== is_polynomial_power
// e = t^n (n 비음정수) 확인
// e == t_var → n = 1
// Pow(t_var, n_const) → n 반환

bool is_polynomial_power(ExprPtr e, ExprPtr t_var, int& n) noexcept {
    if (e == t_var) { n = 1; return true; }
    if (e->type() != NodeType::POW) return false;
    const Pow* pw = static_cast<const Pow*>(e);
    if (pw->base() != t_var) return false;
    double v = 0.0;
    if (!get_numeric(pw->exp(), v)) return false;
    if (v < 0.0 || v != std::floor(v)) return false;
    if (v > 30.0) return false;  // 합리적 상한
    n = static_cast<int>(v);
    return true;
}

// ================================================================== factorial helper

static int64_t factorial(int n) {
    if (n < 0) throw std::domain_error("factorial: n < 0");
    int64_t r = 1;
    for (int i = 2; i <= n; ++i) r *= i;
    return r;
}

// ================================================================== split_exp_factor
// expr 에서 Func(EXP, a*t) 인수를 찾아 a 와 나머지(rest) 분리
// rest == nullptr 은 1을 의미 (다른 인수 없음)
// Mul 아닌 단일 exp(a*t) 도 처리

bool split_exp_factor(ExprPtr expr, ExprPtr t_var,
                      double& a, ExprPtr& rest) noexcept {
    // 단일 exp(a*t)
    if (expr->type() == NodeType::FUNC) {
        const Func* f = static_cast<const Func*>(expr);
        if (f->id() == FuncId::EXP) {
            double coef = 0.0;
            if (is_linear_in(f->arg(), t_var, coef)) {
                a = coef;
                rest = nullptr;
                return true;
            }
        }
        return false;
    }
    // Mul(...) 에서 exp 인수 분리
    if (expr->type() != NodeType::MUL) return false;
    const Mul* m = static_cast<const Mul*>(expr);
    const auto& ops = m->operands();

    int exp_idx = -1;
    double coef = 0.0;
    for (int i = 0; i < (int)ops.size(); ++i) {
        if (ops[i]->type() == NodeType::FUNC) {
            const Func* f = static_cast<const Func*>(ops[i]);
            if (f->id() == FuncId::EXP && is_linear_in(f->arg(), t_var, coef)) {
                exp_idx = i;
                break;
            }
        }
    }
    if (exp_idx < 0) return false;

    a = coef;
    // 나머지 인수 수집
    std::vector<ExprPtr> rest_ops;
    rest_ops.reserve(ops.size() - 1);
    for (int i = 0; i < (int)ops.size(); ++i) {
        if (i != exp_idx) rest_ops.push_back(ops[i]);
    }
    if (rest_ops.empty()) {
        rest = nullptr;
    } else if (rest_ops.size() == 1) {
        rest = rest_ops[0];
    } else {
        rest = P().make_mul(std::move(rest_ops));
    }
    return true;
}

// ================================================================== substitute_var
// expr 내 old_var 를 new_expr 로 재귀 치환 (기호 치환)

ExprPtr substitute_var(ExprPtr expr, ExprPtr old_var, ExprPtr new_expr) {
    if (expr == old_var) return new_expr;

    switch (expr->type()) {
    case NodeType::CONST:
    case NodeType::RATIONAL:
        return expr;
    case NodeType::VAR:
        return expr;  // old_var != expr 이면 그대로
    case NodeType::NEG: {
        const Neg* neg = static_cast<const Neg*>(expr);
        ExprPtr nop = substitute_var(neg->operand(), old_var, new_expr);
        return (nop == neg->operand()) ? expr : P().make_neg(nop);
    }
    case NodeType::SUM: {
        const Sum* s = static_cast<const Sum*>(expr);
        bool changed = false;
        std::vector<ExprPtr> new_ops;
        new_ops.reserve(s->operands().size());
        for (ExprPtr op : s->operands()) {
            ExprPtr q = substitute_var(op, old_var, new_expr);
            new_ops.push_back(q);
            if (q != op) changed = true;
        }
        return changed ? P().make_sum(std::move(new_ops)) : expr;
    }
    case NodeType::MUL: {
        const Mul* m = static_cast<const Mul*>(expr);
        bool changed = false;
        std::vector<ExprPtr> new_ops;
        new_ops.reserve(m->operands().size());
        for (ExprPtr op : m->operands()) {
            ExprPtr q = substitute_var(op, old_var, new_expr);
            new_ops.push_back(q);
            if (q != op) changed = true;
        }
        return changed ? P().make_mul(std::move(new_ops)) : expr;
    }
    case NodeType::POW: {
        const Pow* pw = static_cast<const Pow*>(expr);
        ExprPtr nb = substitute_var(pw->base(), old_var, new_expr);
        ExprPtr ne = substitute_var(pw->exp(),  old_var, new_expr);
        if (nb == pw->base() && ne == pw->exp()) return expr;
        return P().make_pow(nb, ne);
    }
    case NodeType::FUNC: {
        const Func* f = static_cast<const Func*>(expr);
        ExprPtr na = substitute_var(f->arg(), old_var, new_expr);
        return (na == f->arg()) ? expr : P().make_func(f->id(), na);
    }
    default:
        return expr;
    }
}

// ================================================================== differentiate
// expr 을 임의 var 에 대해 심볼릭 미분 (일반화된 diff_wrt_s)
// diff_wrt_s 는 이 함수를 내부 호출로 위임 (하위 호환 유지)

ExprPtr differentiate(ExprPtr expr, ExprPtr var) {
    switch (expr->type()) {
    // d/dvar [const] = 0
    case NodeType::CONST:
    case NodeType::RATIONAL:
        return P().zero();

    // d/dvar [var] = 1,  d/dvar [other] = 0
    case NodeType::VAR:
        return (expr == var) ? P().one() : P().zero();

    // d/dvar [-f] = -(df/dvar)
    case NodeType::NEG: {
        const Neg* neg = static_cast<const Neg*>(expr);
        ExprPtr d = differentiate(neg->operand(), var);
        // 0 최적화
        double dv = 0.0;
        if (get_numeric(d, dv) && dv == 0.0) return P().zero();
        return P().make_neg(d);
    }

    // d/dvar [f + g + ...] = f' + g' + ...
    case NodeType::SUM: {
        const Sum* s = static_cast<const Sum*>(expr);
        std::vector<ExprPtr> diffs;
        diffs.reserve(s->operands().size());
        for (ExprPtr op : s->operands()) {
            ExprPtr d = differentiate(op, var);
            double dv = 0.0;
            if (get_numeric(d, dv) && dv == 0.0) continue;  // 0항 스킵
            diffs.push_back(d);
        }
        if (diffs.empty()) return P().zero();
        if (diffs.size() == 1) return diffs[0];
        return P().make_sum(std::move(diffs));
    }

    // d/dvar [f*g*...] — 일반 product rule
    case NodeType::MUL: {
        const Mul* m = static_cast<const Mul*>(expr);
        const auto& ops = m->operands();
        std::vector<ExprPtr> terms;
        terms.reserve(ops.size());
        for (size_t i = 0; i < ops.size(); ++i) {
            ExprPtr di = differentiate(ops[i], var);
            double dv = 0.0;
            if (get_numeric(di, dv) && dv == 0.0) continue;
            std::vector<ExprPtr> term_ops;
            term_ops.reserve(ops.size());
            for (size_t j = 0; j < ops.size(); ++j) {
                term_ops.push_back((j == i) ? di : ops[j]);
            }
            terms.push_back(P().make_mul(std::move(term_ops)));
        }
        if (terms.empty()) return P().zero();
        if (terms.size() == 1) return terms[0];
        return P().make_sum(std::move(terms));
    }

    // d/dvar [base^exp]
    // exp 가 수치 상수: power rule → exp * base^(exp-1) * base'
    // 그 외: 지원하지 않음
    case NodeType::POW: {
        const Pow* pw = static_cast<const Pow*>(expr);
        double ve = 0.0;
        if (get_numeric(pw->exp(), ve)) {
            ExprPtr db = differentiate(pw->base(), var);
            double dv = 0.0;
            if (get_numeric(db, dv) && dv == 0.0) return P().zero();
            ExprPtr new_exp  = P().make_const(ve - 1.0);
            ExprPtr coef     = P().make_const(ve);
            ExprPtr base_pow = P().make_pow(pw->base(), new_exp);
            std::vector<ExprPtr> facs = {coef, base_pow, db};
            return P().make_mul(std::move(facs));
        }
        throw std::runtime_error("differentiate: non-constant exponent in Pow");
    }

    // d/dvar [func(arg)] — chain rule: func'(arg) * arg'
    case NodeType::FUNC: {
        const Func* f = static_cast<const Func*>(expr);
        ExprPtr da = differentiate(f->arg(), var);
        double dv = 0.0;
        if (get_numeric(da, dv) && dv == 0.0) return P().zero();

        ExprPtr deriv_outer;
        switch (f->id()) {
        case FuncId::EXP:
            deriv_outer = expr;  // exp'(u) = exp(u)
            break;
        case FuncId::LN:
            // ln'(u) = 1/u
            deriv_outer = P().make_pow(f->arg(), P().make_rational(-1, 1));
            break;
        case FuncId::SIN:
            deriv_outer = P().make_func(FuncId::COS, f->arg());
            break;
        case FuncId::COS:
            deriv_outer = P().make_neg(P().make_func(FuncId::SIN, f->arg()));
            break;
        case FuncId::TAN: {
            // tan'(u) = 1/cos²(u) = sec²(u)
            ExprPtr cos_u = P().make_func(FuncId::COS, f->arg());
            deriv_outer = P().make_pow(cos_u, P().make_rational(-2, 1));
            break;
        }
        case FuncId::ARCTAN: {
            // arctan'(u) = 1/(1+u²)
            ExprPtr u2  = P().make_pow(f->arg(), P().make_rational(2, 1));
            ExprPtr den = P().add(P().one(), u2);
            deriv_outer = P().make_pow(den, P().make_rational(-1, 1));
            break;
        }
        case FuncId::ARCSIN: {
            // arcsin'(u) = 1/sqrt(1-u²)
            ExprPtr u2   = P().make_pow(f->arg(), P().make_rational(2, 1));
            ExprPtr one_m_u2 = P().sub(P().one(), u2);
            ExprPtr sq   = P().make_func(FuncId::SQRT, one_m_u2);
            deriv_outer  = P().make_pow(sq, P().make_rational(-1, 1));
            break;
        }
        case FuncId::ARCCOS: {
            // arccos'(u) = -1/sqrt(1-u²)
            ExprPtr u2   = P().make_pow(f->arg(), P().make_rational(2, 1));
            ExprPtr one_m_u2 = P().sub(P().one(), u2);
            ExprPtr sq   = P().make_func(FuncId::SQRT, one_m_u2);
            ExprPtr inv  = P().make_pow(sq, P().make_rational(-1, 1));
            deriv_outer  = P().make_neg(inv);
            break;
        }
        case FuncId::SINH:
            deriv_outer = P().make_func(FuncId::COSH, f->arg());
            break;
        case FuncId::COSH:
            deriv_outer = P().make_func(FuncId::SINH, f->arg());
            break;
        case FuncId::TANH: {
            // tanh'(u) = 1/cosh²(u)
            ExprPtr cosh_u = P().make_func(FuncId::COSH, f->arg());
            deriv_outer = P().make_pow(cosh_u, P().make_rational(-2, 1));
            break;
        }
        case FuncId::SQRT: {
            // sqrt'(u) = 1/(2*sqrt(u))
            ExprPtr sq  = P().make_func(FuncId::SQRT, f->arg());
            ExprPtr two = P().make_rational(2, 1);
            deriv_outer = P().make_pow(P().mul(two, sq), P().make_rational(-1, 1));
            break;
        }
        default:
            throw std::runtime_error("differentiate: unsupported Func type");
        }

        // chain rule: deriv_outer * da
        if (get_numeric(da, dv) && dv == 1.0) return deriv_outer;
        return P().mul(deriv_outer, da);
    }

    default:
        throw std::runtime_error("differentiate: unsupported node type");
    }
}

// ================================================================== diff_wrt_s
// F(s) 를 s에 대해 심볼릭 미분 — differentiate() 위임
ExprPtr diff_wrt_s(ExprPtr F, ExprPtr s_var) {
    return differentiate(F, s_var);
}

// ================================================================== try_transform_primitive
// 순수 primitive 패턴 탐지 + 변환
// 반환 nullptr: 패턴 불일치 (상위에서 다른 규칙 시도)

ExprPtr try_transform_primitive(ExprPtr expr, ExprPtr t_var, ExprPtr s_var) {
    // --- L{c} = c/s  (c는 수치 상수)
    {
        double v = 0.0;
        if (get_numeric(expr, v)) {
            // c * (1/s) = c * s^(-1)
            ExprPtr s_inv = P().make_pow(s_var, cR(-1));
            return P().mul(expr, s_inv);
        }
    }

    // --- L{t^n} = n! / s^(n+1)
    {
        int n = 0;
        if (is_polynomial_power(expr, t_var, n)) {
            int64_t nfact = factorial(n);
            ExprPtr num = cR(nfact);
            ExprPtr den = P().make_pow(s_var, cR(n + 1));
            return P().div(num, den);
        }
    }

    // --- Func 패턴: sin, cos, sinh, cosh, exp
    if (expr->type() == NodeType::FUNC) {
        const Func* f = static_cast<const Func*>(expr);
        double omega = 0.0;

        switch (f->id()) {
        // L{exp(at)} = 1/(s-a)
        case FuncId::EXP: {
            if (is_linear_in(f->arg(), t_var, omega)) {
                // 1 / (s - a)
                ExprPtr a_node = P().make_const(omega);
                ExprPtr denom  = P().sub(s_var, a_node);
                return P().div(P().one(), denom);
            }
            break;
        }
        // L{sin(ωt)} = ω / (s²+ω²)
        case FuncId::SIN: {
            if (is_linear_in(f->arg(), t_var, omega)) {
                ExprPtr w   = P().make_const(omega);
                ExprPtr w2  = P().make_const(omega * omega);
                ExprPtr s2  = P().make_pow(s_var, cR(2));
                ExprPtr den = P().add(s2, w2);
                return P().div(w, den);
            }
            break;
        }
        // L{cos(ωt)} = s / (s²+ω²)
        case FuncId::COS: {
            if (is_linear_in(f->arg(), t_var, omega)) {
                ExprPtr w2  = P().make_const(omega * omega);
                ExprPtr s2  = P().make_pow(s_var, cR(2));
                ExprPtr den = P().add(s2, w2);
                return P().div(s_var, den);
            }
            break;
        }
        // L{sinh(at)} = a / (s²-a²)
        case FuncId::SINH: {
            if (is_linear_in(f->arg(), t_var, omega)) {
                ExprPtr a   = P().make_const(omega);
                ExprPtr a2  = P().make_const(omega * omega);
                ExprPtr s2  = P().make_pow(s_var, cR(2));
                ExprPtr den = P().sub(s2, a2);
                return P().div(a, den);
            }
            break;
        }
        // L{cosh(at)} = s / (s²-a²)
        case FuncId::COSH: {
            if (is_linear_in(f->arg(), t_var, omega)) {
                ExprPtr a2  = P().make_const(omega * omega);
                ExprPtr s2  = P().make_pow(s_var, cR(2));
                ExprPtr den = P().sub(s2, a2);
                return P().div(s_var, den);
            }
            break;
        }
        default:
            break;
        }
    }

    return nullptr;  // 매칭 실패
}

// ================================================================== try_linearity
// Sum의 각 항을 재귀 변환

ExprPtr try_linearity(ExprPtr sum_expr, ExprPtr t_var, ExprPtr s_var) {
    if (sum_expr->type() != NodeType::SUM) return nullptr;
    const Sum* s = static_cast<const Sum*>(sum_expr);
    std::vector<ExprPtr> transformed;
    transformed.reserve(s->operands().size());
    for (ExprPtr op : s->operands()) {
        ExprPtr Fop = _transform_impl(op, t_var, s_var);
        if (!Fop) return nullptr;  // 하나라도 실패하면 전체 실패
        transformed.push_back(Fop);
    }
    return P().make_sum(std::move(transformed));
}

// ================================================================== try_s_shift
// e^(at) * f(t) → F(s-a) 패턴 감지
// Mul([exp(at), f(t)]) 또는 상수 계수 포함 Mul

ExprPtr try_s_shift(ExprPtr expr, ExprPtr t_var, ExprPtr s_var) {
    double a = 0.0;
    ExprPtr rest = nullptr;
    if (!split_exp_factor(expr, t_var, a, rest)) return nullptr;

    // rest == nullptr 이면 f(t) = 1 → L{e^(at)} = 1/(s-a) (primitive가 처리)
    // 단, split_exp_factor는 Mul 내에서만 rest != nullptr 를 보장
    // 단독 exp(a*t)는 try_transform_primitive가 처리하므로 여기서는 rest != nullptr

    // rest 가 없는 경우는 단독 EXP → primitive에서 처리됨
    if (!rest) return nullptr;

    // F_rest = L{rest} (s 영역)
    ExprPtr F_rest = _transform_impl(rest, t_var, s_var);
    if (!F_rest) return nullptr;

    // s-shift: s → s - a
    ExprPtr a_node = P().make_const(a);
    ExprPtr s_shifted = P().sub(s_var, a_node);  // s - a

    // F_rest(s) 에서 s_var → s - a 로 치환
    ExprPtr result = substitute_var(F_rest, s_var, s_shifted);
    return result;
}

// ================================================================== try_freq_diff
// t^n * f(t) → (-1)^n * d^n/ds^n [L{f(t)}]
// n = 1: t * f(t) → -F'(s)
// n >= 2: 재귀 적용

ExprPtr try_freq_diff(ExprPtr mul_expr, ExprPtr t_var, ExprPtr s_var) {
    if (mul_expr->type() != NodeType::MUL) return nullptr;
    const Mul* m = static_cast<const Mul*>(mul_expr);
    const auto& ops = m->operands();

    // t^n 인수 탐색 (t_var 단독 또는 Pow(t_var, n))
    int t_idx = -1;
    int t_power = 0;
    for (int i = 0; i < (int)ops.size(); ++i) {
        if (ops[i] == t_var) {
            t_idx = i;
            t_power = 1;
            break;
        }
        int n_pow = 0;
        if (is_polynomial_power(ops[i], t_var, n_pow) && n_pow >= 1) {
            t_idx = i;
            t_power = n_pow;
            break;
        }
    }
    if (t_idx < 0) return nullptr;
    if (t_power == 0) return nullptr;

    // t^n 제외 나머지 곱
    std::vector<ExprPtr> rest_ops;
    rest_ops.reserve(ops.size() - 1);
    for (int i = 0; i < (int)ops.size(); ++i) {
        if (i != t_idx) rest_ops.push_back(ops[i]);
    }
    ExprPtr f_t;
    if (rest_ops.empty()) {
        // t^n 단독 → primitive 처리 대상
        return nullptr;
    } else if (rest_ops.size() == 1) {
        f_t = rest_ops[0];
    } else {
        f_t = P().make_mul(std::move(rest_ops));
    }

    // L{f(t)} 계산
    ExprPtr F_s = _transform_impl(f_t, t_var, s_var);
    if (!F_s) return nullptr;

    // (-1)^n * d^n F / ds^n
    ExprPtr result = F_s;
    for (int i = 0; i < t_power; ++i) {
        result = diff_wrt_s(result, s_var);
        result = P().make_neg(result);
    }
    return result;
}

// ================================================================== _transform_impl (내부 디스패처)

static ExprPtr _transform_impl(ExprPtr expr, ExprPtr t_var, ExprPtr s_var) {
    // 1) 캐시 조회
    auto it = _transform_cache.find(expr);
    if (it != _transform_cache.end()) return it->second;

    ExprPtr result = nullptr;

    // 2) Constant / Rational → c/s
    {
        double v = 0.0;
        if (get_numeric(expr, v)) {
            result = try_transform_primitive(expr, t_var, s_var);
            if (result) {
                _transform_cache[expr] = result;
                return result;
            }
        }
    }

    // 3) Var == t_var → 1/s²
    if (expr == t_var) {
        result = try_transform_primitive(expr, t_var, s_var);
        if (result) {
            _transform_cache[expr] = result;
            return result;
        }
    }

    // 4) Pow(t, n) 정수 n → n!/s^(n+1)
    if (expr->type() == NodeType::POW) {
        int n = 0;
        if (is_polynomial_power(expr, t_var, n)) {
            result = try_transform_primitive(expr, t_var, s_var);
            if (result) {
                _transform_cache[expr] = result;
                return result;
            }
        }
    }

    // 5) Func(EXP/SIN/COS/SINH/COSH, linear_in_t) → primitive
    if (expr->type() == NodeType::FUNC) {
        result = try_transform_primitive(expr, t_var, s_var);
        if (result) {
            _transform_cache[expr] = result;
            return result;
        }
    }

    // 6) Sum → Linearity
    if (expr->type() == NodeType::SUM) {
        result = try_linearity(expr, t_var, s_var);
        if (result) {
            _transform_cache[expr] = result;
            return result;
        }
    }

    // 7) Mul: 상수 × f(t) → 상수를 분리해 재귀
    if (expr->type() == NodeType::MUL) {
        const Mul* m = static_cast<const Mul*>(expr);
        const auto& ops = m->operands();

        // 상수 인수 분리
        double const_factor = 1.0;
        bool has_const = false;
        std::vector<ExprPtr> sym_ops;
        sym_ops.reserve(ops.size());
        for (ExprPtr op : ops) {
            double v = 0.0;
            if (get_numeric(op, v)) {
                const_factor *= v;
                has_const = true;
            } else {
                sym_ops.push_back(op);
            }
        }

        if (has_const && !sym_ops.empty()) {
            // 재구성
            ExprPtr sym_part;
            if (sym_ops.size() == 1) {
                sym_part = sym_ops[0];
            } else {
                sym_part = P().make_mul(std::move(sym_ops));
            }
            ExprPtr F_sym = _transform_impl(sym_part, t_var, s_var);
            if (F_sym) {
                ExprPtr cf = P().make_const(const_factor);
                result = P().mul(cf, F_sym);
                _transform_cache[expr] = result;
                return result;
            }
        }

        // 8) Mul (exp(a*t) * f(t)) → s-shift
        result = try_s_shift(expr, t_var, s_var);
        if (result) {
            _transform_cache[expr] = result;
            return result;
        }

        // 9) Mul (t * f(t)) → frequency differentiation
        result = try_freq_diff(expr, t_var, s_var);
        if (result) {
            _transform_cache[expr] = result;
            return result;
        }
    }

    // 10) Neg: -f(t) → -L{f(t)}
    if (expr->type() == NodeType::NEG) {
        const Neg* neg = static_cast<const Neg*>(expr);
        ExprPtr F_inner = _transform_impl(neg->operand(), t_var, s_var);
        if (F_inner) {
            result = P().make_neg(F_inner);
            _transform_cache[expr] = result;
            return result;
        }
    }

    // 변환 실패
    return nullptr;
}

// ================================================================== forward_transform (공개 진입점)

ExprPtr forward_transform(ExprPtr expr, ExprPtr t_var, ExprPtr s_var) {
    ExprPtr result = _transform_impl(expr, t_var, s_var);
    if (!result) {
        throw std::runtime_error(
            "forward_transform: cannot transform expression: " + expr->to_string());
    }
    return result;
}

} // namespace ml_laplace
