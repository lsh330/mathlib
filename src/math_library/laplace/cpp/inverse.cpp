// inverse.cpp — 역 Laplace 변환 구현 (Phase C)

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

// ================================================================== inverse_transform

ExprPtr inverse_transform(ExprPtr F, ExprPtr s_var, ExprPtr t_var) {
    // 1) AST → RationalFunction 변환
    RationalFunction rf = [&]() -> RationalFunction {
        try {
            return RationalFunction::from_expr(F, s_var);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("inverse_transform: cannot parse as rational function: ") + e.what());
        }
    }();

    rf = rf.simplify();

    // 2) Partial fractions
    std::vector<PartialFractionTerm> terms;
    try {
        terms = partial_fractions(rf);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("inverse_transform: partial fraction failed: ") + e.what());
    }

    if (terms.empty()) {
        // F(s) 가 다항식 (improper fraction 단독) → t=0 에서 impulse 등 (비인과 시스템)
        // Phase C: 지원 안 함
        throw std::runtime_error(
            "inverse_transform: F(s) is a polynomial (no poles) — not supported in Phase C");
    }

    // 3) 각 항 역변환 후 합산
    // 복소 켤레 쌍 처리: terms에서 켤레 쌍 탐지
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
            // 실수 극점: 각 중복도에 대해 처리
            for (int k = 0; k < ti.multiplicity; ++k) {
                std::complex<double> ck = ti.coefficients[k];
                double c_re = ck.real();
                int power = ti.multiplicity - k;  // 분모 차수

                ExprPtr term = inv_simple_pole(c_re, sigma, power, t_var);
                if (term != P().zero()) {
                    f_terms.push_back(term);
                }
            }
        } else {
            // 복소 극점: 켤레 쌍 탐지
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
                // 켤레 쌍: c/(s-p) + c*/(s-p*)
                for (int k = 0; k < ti.multiplicity; ++k) {
                    std::complex<double> ck = ti.coefficients[k];
                    int power = ti.multiplicity - k;
                    ExprPtr term = inv_conjugate_pair(
                        sigma, omega,
                        ck.real(), ck.imag(),
                        power, t_var
                    );
                    if (term != P().zero()) {
                        f_terms.push_back(term);
                    }
                }
            } else {
                // 켤레 없음: 실수 부분만 (비정상적 경우)
                for (int k = 0; k < ti.multiplicity; ++k) {
                    std::complex<double> ck = ti.coefficients[k];
                    int power = ti.multiplicity - k;
                    ExprPtr term = inv_simple_pole(ck.real(), sigma, power, t_var);
                    if (term != P().zero()) {
                        f_terms.push_back(term);
                    }
                }
            }
        }
    }

    if (f_terms.empty()) return P().zero();
    if (f_terms.size() == 1) return f_terms[0];
    return P().make_sum(std::move(f_terms));
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
