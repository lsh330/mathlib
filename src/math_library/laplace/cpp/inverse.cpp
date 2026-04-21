// inverse.cpp — 역 Laplace 변환 구현 (Phase C + Phase E)

#include "inverse.hpp"
#include "rules.hpp"  // substitute_var
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <sstream>

namespace ml_laplace {

// ================================================================== 헬퍼

static inline ExprPool& P() { return ExprPool::instance(); }

static bool get_numeric_v(ExprPtr e, double& v) noexcept {
    if (!e) return false;
    if (e->type() == NodeType::CONST) {
        v = static_cast<const Const*>(e)->value();
        return true;
    }
    if (e->type() == NodeType::RATIONAL) {
        v = static_cast<const Rational*>(e)->as_double();
        return true;
    }
    return false;
}

// 계승 (정수)
static double factorial_d(int n) {
    if (n < 0) return 1.0;
    double r = 1.0;
    for (int i = 2; i <= n; ++i) r *= i;
    return r;
}

// ================================================================== 역변환: 단항 c/(s-a)^n → c * t^(n-1) * e^(at) / (n-1)!
// n=1: c * e^(at)
// n=2: c * t * e^(at)
// n>=1: c * t^(n-1) * e^(at) / (n-1)!

static ExprPtr inv_simple_pole(double c, double a, int n, ExprPtr t_var) {
    if (std::abs(c) < 1e-14) return P().zero();

    // e^(at)
    ExprPtr a_node = P().make_const(a);
    ExprPtr at_expr = P().mul(a_node, t_var);
    ExprPtr exp_at = P().make_func(FuncId::EXP, at_expr);

    if (n == 1) {
        // c * e^(at)
        if (std::abs(c - 1.0) < 1e-14) return exp_at;
        return P().mul(P().make_const(c), exp_at);
    }

    // n >= 2: c / (n-1)! * t^(n-1) * e^(at)
    double coeff = c / factorial_d(n - 1);
    ExprPtr coeff_node = P().make_const(coeff);
    ExprPtr t_pow;
    if (n - 1 == 1) {
        t_pow = t_var;
    } else {
        t_pow = P().make_pow(t_var, P().make_const(static_cast<double>(n - 1)));
    }
    return P().mul(coeff_node, P().mul(t_pow, exp_at));
}

// ================================================================== 역변환: 복소 켤레 쌍
// 2Re(c)*(s-σ) - 2Im(c)*ω  /  ((s-σ)^2 + ω^2)  → 시간 영역
//
// 단순 (단중) 경우:
//   c/(s-p) + c*/(s-p*) where p = σ + jω
//   = e^(σt) * [2Re(c)*cos(ωt) - 2Im(c)*sin(ωt)] * u(t)
//   = e^(σt) * A*cos(ωt + φ) 형태
//
// 중복 경우: t^(k-1) e^(σt) cos/sin 조합

static ExprPtr inv_conjugate_pair(
    double sigma, double omega,
    double c_re, double c_im,  // c = c_re + j*c_im (s-p 분모)
    int power,                 // (s-p)의 차수 (= multiplicity - k_index)
    ExprPtr t_var
) {
    if (std::abs(c_re) < 1e-14 && std::abs(c_im) < 1e-14) return P().zero();

    // e^(σt)
    ExprPtr sigma_e = P().make_const(sigma);
    ExprPtr exp_sig = P().make_func(FuncId::EXP, P().mul(sigma_e, t_var));

    // ω*t
    ExprPtr omega_e = P().make_const(omega);
    ExprPtr wt = P().mul(omega_e, t_var);

    // t^(n-1) / (n-1)!
    double tn_coeff = 1.0 / factorial_d(power - 1);
    ExprPtr t_pow;
    if (power == 1) {
        t_pow = P().one();
    } else if (power == 2) {
        t_pow = P().mul(P().make_const(tn_coeff), t_var);
    } else {
        int n = power - 1;
        t_pow = P().mul(P().make_const(tn_coeff),
                        P().make_pow(t_var, P().make_const((double)n)));
    }

    // 2Re(c) * cos(ωt) - 2Im(c) * sin(ωt)
    std::vector<ExprPtr> parts;

    if (std::abs(c_re) > 1e-14) {
        ExprPtr cos_wt = P().make_func(FuncId::COS, wt);
        parts.push_back(P().mul(P().make_const(2.0 * c_re), cos_wt));
    }
    if (std::abs(c_im) > 1e-14) {
        ExprPtr sin_wt = P().make_func(FuncId::SIN, wt);
        // -2*c_im * sin(wt)
        parts.push_back(P().mul(P().make_const(-2.0 * c_im), sin_wt));
    }

    if (parts.empty()) return P().zero();

    ExprPtr trig_part;
    if (parts.size() == 1) {
        trig_part = parts[0];
    } else {
        trig_part = P().make_sum(std::move(parts));
    }

    // exp_sig * t_pow * trig_part
    ExprPtr result = P().mul(exp_sig, P().mul(t_pow, trig_part));
    return result;
}

// ================================================================== Phase E: extract_exp_shift
// F = e^(-a*s) * G(s) 패턴 탐지
// 성공 시: shift = a (a > 0), rest = G(s) 반환
// 단독 e^(-a*s) 이면 rest = nullptr(=1)

static bool extract_exp_shift_in_s(ExprPtr expr, ExprPtr s_var,
                                    double& shift, ExprPtr& rest) noexcept {
    // 단독 exp(-a*s)
    if (expr->type() == NodeType::FUNC) {
        const Func* f = static_cast<const Func*>(expr);
        if (f->id() != FuncId::EXP) return false;
        // arg = -a*s 또는 s*(-a) 형태
        ExprPtr arg = f->arg();
        double coef = 0.0;
        if (!is_linear_in(arg, s_var, coef)) return false;
        if (coef >= 0.0) return false;  // e^(+a*s)는 t-shift 아님
        shift = -coef;  // a = -coef (양수)
        rest = nullptr;
        return true;
    }
    // Mul 내 exp(-a*s) 분리
    if (expr->type() != NodeType::MUL) return false;
    const Mul* m = static_cast<const Mul*>(expr);
    const auto& ops = m->operands();

    int exp_idx = -1;
    double coef = 0.0;
    for (int i = 0; i < (int)ops.size(); ++i) {
        if (ops[i]->type() == NodeType::FUNC) {
            const Func* f = static_cast<const Func*>(ops[i]);
            if (f->id() == FuncId::EXP) {
                double c = 0.0;
                if (is_linear_in(f->arg(), s_var, c) && c < 0.0) {
                    coef = c;
                    exp_idx = i;
                    break;
                }
            }
        }
    }
    if (exp_idx < 0) return false;

    shift = -coef;
    // 나머지 수집
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

// 전방 선언 (재귀 호출용)
static ExprPtr inverse_with_shift(ExprPtr F, ExprPtr s_var, ExprPtr t_var);

// ================================================================== inverse_rational
// 유리함수 전용 역변환 (Phase C 기존 로직)

static ExprPtr inverse_rational(ExprPtr F, ExprPtr s_var, ExprPtr t_var);

// ================================================================== Phase E: inverse_with_shift
// e^(-as)*G1 + e^(-bs)*G2 + ... 형태를 항별로 처리

static ExprPtr inverse_with_shift(ExprPtr F, ExprPtr s_var, ExprPtr t_var) {
    // Sum → 각 항 재귀
    if (F->type() == NodeType::SUM) {
        const Sum* s = static_cast<const Sum*>(F);
        std::vector<ExprPtr> terms;
        terms.reserve(s->operands().size());
        for (ExprPtr op : s->operands()) {
            terms.push_back(inverse_with_shift(op, s_var, t_var));
        }
        if (terms.empty()) return P().zero();
        if (terms.size() == 1) return terms[0];
        return P().make_sum(std::move(terms));
    }

    double shift = 0.0;
    ExprPtr rest = nullptr;

    if (extract_exp_shift_in_s(F, s_var, shift, rest)) {
        // G(s) = rest (nullptr → 1)
        ExprPtr G;
        if (!rest) {
            // e^(-as) 단독: L^{-1}{e^(-as)} = δ(t-a) — 분포이므로 AST 표현
            // Heaviside/Dirac 기호 AST 생성
            ExprPtr a_node = P().make_const(shift);
            ExprPtr arg_t  = P().sub(t_var, a_node);
            return P().make_func(FuncId::DIRAC, arg_t);
        }
        G = rest;

        // g(t) = L^{-1}{G(s)} — 재귀 (유리함수)
        ExprPtr g_t = inverse_rational(G, s_var, t_var);

        // g(t-a): t → t-a 치환
        ExprPtr a_node = P().make_const(shift);
        ExprPtr t_minus_a = P().sub(t_var, a_node);
        ExprPtr g_shifted = substitute_var(g_t, t_var, t_minus_a);

        // u(t-a) * g(t-a)
        ExprPtr hside = P().make_func(FuncId::HEAVISIDE, t_minus_a);
        return P().mul(hside, g_shifted);
    }

    // exp shift 없음 → 유리함수 경로
    return inverse_rational(F, s_var, t_var);
}

// ================================================================== inverse_transform

static ExprPtr inverse_rational(ExprPtr F, ExprPtr s_var, ExprPtr t_var) {
    // AST → RationalFunction 변환
    RationalFunction rf = [&]() -> RationalFunction {
        try {
            return RationalFunction::from_expr(F, s_var);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("inverse_rational: cannot parse as rational function: ") + e.what());
        }
    }();
    rf = rf.simplify();

    std::vector<PartialFractionTerm> terms;
    try {
        terms = partial_fractions(rf);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("inverse_rational: partial fraction failed: ") + e.what());
    }

    if (terms.empty()) {
        throw std::runtime_error(
            "inverse_rational: F(s) is a polynomial (no poles) — not supported");
    }

    double tol = 1e-9;
    std::vector<bool> used(terms.size(), false);
    std::vector<ExprPtr> f_terms;

    for (size_t i = 0; i < terms.size(); ++i) {
        if (used[i]) continue;
        const auto& ti = terms[i];
        double sigma = ti.pole.real();
        double omega = ti.pole.imag();
        bool is_real = std::abs(omega) < tol;

        if (is_real) {
            for (int k = 0; k < ti.multiplicity; ++k) {
                std::complex<double> ck = ti.coefficients[k];
                double c_re = ck.real();
                int power = ti.multiplicity - k;
                ExprPtr term = inv_simple_pole(c_re, sigma, power, t_var);
                if (term != P().zero()) f_terms.push_back(term);
            }
        } else {
            int j_conj = -1;
            for (size_t j = i + 1; j < terms.size(); ++j) {
                if (used[j]) continue;
                if (terms[j].multiplicity == ti.multiplicity &&
                    std::abs(terms[j].pole - std::conj(ti.pole)) < tol) {
                    j_conj = static_cast<int>(j);
                    break;
                }
            }
            if (j_conj >= 0) {
                used[j_conj] = true;
                for (int k = 0; k < ti.multiplicity; ++k) {
                    std::complex<double> ck = ti.coefficients[k];
                    int power = ti.multiplicity - k;
                    ExprPtr term = inv_conjugate_pair(
                        sigma, omega, ck.real(), ck.imag(), power, t_var);
                    if (term != P().zero()) f_terms.push_back(term);
                }
            } else {
                for (int k = 0; k < ti.multiplicity; ++k) {
                    std::complex<double> ck = ti.coefficients[k];
                    int power = ti.multiplicity - k;
                    ExprPtr term = inv_simple_pole(ck.real(), sigma, power, t_var);
                    if (term != P().zero()) f_terms.push_back(term);
                }
            }
        }
    }

    if (f_terms.empty()) return P().zero();
    if (f_terms.size() == 1) return f_terms[0];
    return P().make_sum(std::move(f_terms));
}

ExprPtr inverse_transform(ExprPtr F, ExprPtr s_var, ExprPtr t_var) {
    // Phase E: non-rational (exp shift) 먼저 시도
    return inverse_with_shift(F, s_var, t_var);
}

// ================================================================== compute_poles / compute_zeros

std::vector<std::complex<double>> compute_poles(ExprPtr F, ExprPtr s_var) {
    RationalFunction rf = RationalFunction::from_expr(F, s_var);
    rf = rf.simplify();
    return rf.poles();
}

std::vector<std::complex<double>> compute_zeros(ExprPtr F, ExprPtr s_var) {
    RationalFunction rf = RationalFunction::from_expr(F, s_var);
    rf = rf.simplify();
    return rf.zeros();
}

// ================================================================== final_value / initial_value

double final_value(ExprPtr F, ExprPtr s_var, bool& valid) {
    // lim_{s->0} s * F(s)
    // 유효성: 모든 극점이 열린 좌반 평면에 있어야 함 (Re(p) < 0)
    try {
        RationalFunction rf = RationalFunction::from_expr(F, s_var);
        rf = rf.simplify();

        auto poles = rf.poles();
        valid = true;
        for (auto& p : poles) {
            if (p.real() >= -1e-9) {
                // s=0 극점은 허용 (최종값 정리 적용 가능)
                // 실제 조건: s*F(s) 의 모든 극점이 Re(p) < 0 이어야 함
                // s*F(s) 의 극점 = F(s) 의 극점에서 s=0 제외
                if (std::abs(p) > 1e-9 && p.real() >= 0.0) {
                    valid = false;
                    break;
                }
            }
        }

        // lim_{s->0} s * F(s) = s * P(s) / Q(s) at s=0
        // = P(0) / [Q(s)/s at s=0] if s=0 is simple pole of F
        // 단순 접근: s -> 0 에서 s*F(s) 수치 계산
        double eps = 1e-6;
        std::complex<double> sF = std::complex<double>(eps, 0.0) * rf.eval(eps);
        return sF.real();

    } catch (...) {
        valid = false;
        return 0.0;
    }
}

double initial_value(ExprPtr F, ExprPtr s_var, bool& valid) {
    // lim_{s->inf} s * F(s)
    // proper fraction 가정: lim = leading_coeff(P) / leading_coeff(Q) if deg(P) = deg(Q)-1
    // or leading_coeff(P) * s^(1+deg(Q)-deg(P)-1) → 0 if deg(P) < deg(Q)-1
    try {
        RationalFunction rf = RationalFunction::from_expr(F, s_var);
        rf = rf.simplify();
        valid = true;

        int deg_n = rf.num().degree();
        int deg_d = rf.den().degree();

        // s * P(s) / Q(s) : degree of numerator s*P = deg_n + 1
        // If deg_n + 1 < deg_d: limit = 0
        // If deg_n + 1 == deg_d: limit = leading_coeff(P) / leading_coeff(Q)
        // If deg_n + 1 > deg_d: limit = infinity

        if (deg_n + 1 < deg_d) {
            return 0.0;
        } else if (deg_n + 1 == deg_d) {
            return rf.num().leading_coeff() / rf.den().leading_coeff();
        } else {
            // improper: lim = ±inf
            valid = false;
            return std::numeric_limits<double>::infinity();
        }
    } catch (...) {
        valid = false;
        return 0.0;
    }
}

} // namespace ml_laplace
