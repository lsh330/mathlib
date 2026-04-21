// partial.cpp — Partial Fraction Decomposition 구현 (Phase C)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "partial.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace ml_laplace {

// ================================================================== 헬퍼

static inline ExprPool& P() { return ExprPool::instance(); }

// 두 복소수가 tol 내에서 같은지
static bool complex_eq(std::complex<double> a, std::complex<double> b, double tol) {
    return std::abs(a - b) < tol;
}

// Cauchy 수치 적분으로 g(s) = (s-p)^m * F(s) 의 k번째 Taylor 계수 계산
// c_k = g^(k)(p) / k!
// 공식: c_k = (1/N) * sum_{j=0}^{N-1} g(p + r*e^{i*theta_j}) * e^{-i*k*theta_j} / r^k
static std::complex<double> numerical_derivative_over_kfact(
    const RationalFunction& F,
    std::complex<double> pole,
    int m,
    int k,
    double r = 0.1  // 적분 경로 반경
) {
    int N = 128;  // 경로 점 수
    double dtheta = 2.0 * M_PI / N;
    std::complex<double> result(0.0, 0.0);

    for (int j = 0; j < N; ++j) {
        double theta = dtheta * j;
        std::complex<double> e_itheta(std::cos(theta), std::sin(theta));
        std::complex<double> z = pole + r * e_itheta;

        // g(z) = (z - p)^m * F(z)
        std::complex<double> zm = std::pow(r * e_itheta, m);
        std::complex<double> Fz = F.eval(z);
        std::complex<double> gz = zm * Fz;

        // e^(-i*k*theta)
        std::complex<double> e_neg(std::cos(-k * theta), std::sin(-k * theta));
        result += gz * e_neg;
    }
    // c_k = result / (N * r^k)
    result /= std::complex<double>(static_cast<double>(N) * std::pow(r, k), 0.0);
    return result;
}

// ================================================================== 중복도 탐지 + 극점 그룹화

struct PoleGroup {
    std::complex<double> pole;
    int multiplicity;
};

static std::vector<PoleGroup> group_poles(
    const std::vector<std::complex<double>>& raw_poles,
    double tol
) {
    std::vector<PoleGroup> groups;

    for (auto& p : raw_poles) {
        bool found = false;
        for (auto& g : groups) {
            if (complex_eq(g.pole, p, tol)) {
                g.multiplicity++;
                // 극점 위치를 평균으로 업데이트
                g.pole = (g.pole * (double)(g.multiplicity - 1) + p) / (double)g.multiplicity;
                found = true;
                break;
            }
        }
        if (!found) {
            groups.push_back({p, 1});
        }
    }
    return groups;
}

// ================================================================== partial_fractions

std::vector<PartialFractionTerm> partial_fractions(
    const RationalFunction& F_in,
    double tol
) {
    // 1) 약분
    RationalFunction F = F_in.simplify(tol);

    // 2) 진분수 확인: deg(P) >= deg(Q) 이면 다항식 부분 분리 필요
    //    Phase C에서는 전형적 제어 전달함수 (proper) 가정
    int deg_n = F.num().degree();
    int deg_d = F.den().degree();

    // 다항식 부분 + 순수 진분수
    Polynomial poly_part = Polynomial::scalar(0.0);
    RationalFunction F_proper = F;

    if (deg_n >= deg_d) {
        auto [quot, rem] = F.num().divmod(F.den());
        poly_part = quot;
        F_proper = RationalFunction(rem, F.den());
    }

    // 3) 분모 근 + 중복도 탐지
    // 전략: squarefree 분모의 근을 구해 단순 근 목록을 얻고,
    //       원본 분모로 중복도를 계산 (수치 미분 기반)
    std::vector<std::complex<double>> raw_poles = F_proper.den().roots();
    if (raw_poles.empty()) {
        // 분모가 상수 → 다항식만 존재 (partial fraction 없음)
        return {};
    }

    // 4) 중복도 그룹화
    // squarefree 분모를 이용하여 단순 극점만 구한 뒤 중복도 계산
    // squarefree(Q) 의 근 = Q의 단순 극점 (각 1회)
    Polynomial Q_sf = F_proper.den().squarefree(tol);
    std::vector<std::complex<double>> simple_poles = Q_sf.roots();

    // 단순 극점 각각에 대해 원본 Q에서 중복도 계산
    // 중복도 m: Q(p) ≈ 0, Q'(p) ≈ 0, ..., Q^(m-1)(p) ≈ 0, Q^(m)(p) ≠ 0
    std::vector<PoleGroup> groups;
    {
        Polynomial Qd = F_proper.den();
        for (auto& sp : simple_poles) {
            int mult = 1;
            // Newton refinement: sp를 Q의 정확한 근으로 보정
            for (int ref = 0; ref < 30; ++ref) {
                std::complex<double> qval = Qd.eval(sp);
                std::complex<double> qdval = Qd.derivative().eval(sp);
                if (std::abs(qdval) < 1e-300) break;
                std::complex<double> dz = qval / qdval;
                sp -= dz;
                if (std::abs(dz) < 1e-14) break;
            }

            // 중복도: 연속 미분 값으로 탐지
            Polynomial Q_curr = Qd;
            for (int k = 1; k <= Qd.degree(); ++k) {
                std::complex<double> val = Q_curr.eval(sp);
                if (std::abs(val) > tol * 1e3) break;
                // k차 미분도 0에 가까우면 중복도 증가
                Q_curr = Q_curr.derivative();
                mult = k + 1;
                // k차 미분이 충분히 크면 중복도 = k
                std::complex<double> kval = Q_curr.eval(sp);
                if (std::abs(kval) > tol * 1e3) {
                    mult = k;
                    break;
                }
            }
            groups.push_back({sp, mult});
        }
    }

    // groups가 비어있으면 폴백: raw_poles로 그룹화
    if (groups.empty()) {
        double group_tol = std::max(tol, 1e-6);
        groups = group_poles(raw_poles, group_tol);
    }

    // 중복도 합이 분모 차수와 일치하도록 보정
    {
        int total_mult = 0;
        for (auto& g : groups) total_mult += g.multiplicity;
        int target = F_proper.den().degree();
        if (total_mult != target) {
            // 단순 폴백: raw_poles 기반 그룹화
            double group_tol = std::max(tol, 1e-5);
            groups = group_poles(raw_poles, group_tol);
        }
    }

    // 5) 각 극점에 대해 Laurent 계수 계산
    std::vector<PartialFractionTerm> result;

    for (auto& g : groups) {
        PartialFractionTerm term;
        term.pole = g.pole;
        term.multiplicity = g.multiplicity;
        term.coefficients.resize(g.multiplicity);

        int m = g.multiplicity;

        if (m == 1) {
            // Cover-up: c = P(p) / Q'(p)
            std::complex<double> Pp = F_proper.num().eval(g.pole);
            Polynomial Qd = F_proper.den().derivative();
            std::complex<double> Qp = Qd.eval(g.pole);
            if (std::abs(Qp) < 1e-300) {
                // fallback: Cauchy integral
                term.coefficients[0] = numerical_derivative_over_kfact(F_proper, g.pole, 1, 0, tol * 10.0);
            } else {
                term.coefficients[0] = Pp / Qp;
            }
        } else {
            // 중복 극점: Cauchy 수치 적분으로 각 계수 계산
            // c_k = g^(k)(p) / k!  where g(s) = (s-p)^m * F(s)
            // term.coefficients[k] → 1/(s-p)^(m-k) 의 계수 (k=0: 가장 높은 중복도)
            // 반경: 극점과 가장 가까운 다른 극점 사이 거리의 1/3 (안전 마진)
            double r = 0.3;  // 기본 반경
            // 다른 극점과의 최소 거리 기반 반경 결정
            for (auto& other : groups) {
                if (&other == &g) continue;
                double d = std::abs(other.pole - g.pole);
                if (d > 0.01 && d / 3.0 < r) {
                    r = d / 3.0;
                }
            }
            r = std::max(r, 0.01);

            for (int k = 0; k < m; ++k) {
                term.coefficients[k] = numerical_derivative_over_kfact(
                    F_proper, g.pole, m, k, r);
            }
        }

        result.push_back(std::move(term));
    }

    return result;
}

// ================================================================== partial_fractions_to_expr
// 복소 켤레 근 결합 + AST 생성

ExprPtr partial_fractions_to_expr(
    const std::vector<PartialFractionTerm>& terms,
    ExprPtr s_var,
    double tol
) {
    std::vector<ExprPtr> summands;
    std::vector<bool> used(terms.size(), false);

    for (size_t i = 0; i < terms.size(); ++i) {
        if (used[i]) continue;

        const auto& ti = terms[i];
        double sigma = ti.pole.real();
        double omega = ti.pole.imag();
        bool is_real = std::abs(omega) < tol;

        if (is_real) {
            // 실수 극점: c_k / (s - p)^k 형태
            for (int k = 0; k < ti.multiplicity; ++k) {
                std::complex<double> ck = ti.coefficients[k];
                double c_re = ck.real();
                if (std::abs(c_re) < 1e-14) continue;

                ExprPtr c_expr = P().make_const(c_re);
                // 분모: (s - sigma)^(m-k)
                int power = ti.multiplicity - k;
                ExprPtr sigma_e = P().make_const(sigma);
                ExprPtr denom_base = P().sub(s_var, sigma_e);  // s - sigma
                ExprPtr denom;
                if (power == 1) {
                    denom = denom_base;
                } else {
                    denom = P().make_pow(denom_base, P().make_const(static_cast<double>(power)));
                }
                summands.push_back(P().div(c_expr, denom));
            }
        } else {
            // 복소 극점: 켤레 쌍 찾기
            int j_conj = -1;
            for (size_t j = i + 1; j < terms.size(); ++j) {
                if (used[j]) continue;
                if (terms[j].multiplicity == ti.multiplicity &&
                    complex_eq(terms[j].pole, std::conj(ti.pole), tol)) {
                    j_conj = static_cast<int>(j);
                    break;
                }
            }

            if (j_conj >= 0) {
                // 켤레 쌍 결합: c/(s-p) + c*/(s-p*)
                // = [c*(s-p*) + c*(s-p)] / [(s-p)(s-p*)]
                // = [2Re(c)(s-σ) - 2Im(c)*ω] / [(s-σ)^2 + ω^2]
                used[j_conj] = true;

                for (int k = 0; k < ti.multiplicity; ++k) {
                    std::complex<double> ck = ti.coefficients[k];
                    double c_re = ck.real();
                    double c_im = ck.imag();

                    // 분모 차수: (m-k)
                    int power = ti.multiplicity - k;

                    // 분모: ((s-σ)^2 + ω^2)^power
                    ExprPtr sigma_e = P().make_const(sigma);
                    ExprPtr omega2_e = P().make_const(omega * omega);
                    ExprPtr s_minus_sigma = P().sub(s_var, sigma_e);
                    ExprPtr s_ms2 = P().make_pow(s_minus_sigma, P().make_const(2.0));
                    ExprPtr quad = P().add(s_ms2, omega2_e);

                    ExprPtr denom;
                    if (power == 1) {
                        denom = quad;
                    } else {
                        denom = P().make_pow(quad, P().make_const(static_cast<double>(power)));
                    }

                    // 분자: 2*Re(c)*(s-σ) - 2*Im(c)*ω
                    // = 2*c_re*(s-σ) + 2*(-c_im)*ω   → 부호 주의
                    // 실제로 c/(s-p) + conj(c)/(s-conj(p)) 실수화:
                    // 분자 = 2*Re(c)*(s-σ) + 2*(Im(c)*ω - Re(c)*0)  → 정확히는:
                    // = 2Re(c)·s - 2Re(c)·σ - 2Im(c)·ω
                    // 이므로 분자 = 2Re(c)·(s-σ) - 2Im(c)·ω

                    ExprPtr two_re = P().make_const(2.0 * c_re);
                    ExprPtr two_im_omega = P().make_const(-2.0 * c_im * omega);

                    ExprPtr num_expr;
                    if (std::abs(c_re) < 1e-14 && std::abs(c_im) < 1e-14) continue;

                    if (std::abs(c_re) < 1e-14) {
                        num_expr = two_im_omega;  // 상수 항만
                    } else if (std::abs(c_im * omega) < 1e-14) {
                        num_expr = P().mul(two_re, s_minus_sigma);
                    } else {
                        num_expr = P().add(
                            P().mul(two_re, s_minus_sigma),
                            two_im_omega
                        );
                    }

                    summands.push_back(P().div(num_expr, denom));
                }
            } else {
                // 켤레 없음: 복소 계수로 그대로 표현 (실수 파트만)
                // 이 경우는 분모가 실수 계수가 아닌 상황 → 오류 가능성
                // 안전하게 실부/허부 분리 시도
                for (int k = 0; k < ti.multiplicity; ++k) {
                    std::complex<double> ck = ti.coefficients[k];
                    int power = ti.multiplicity - k;
                    if (std::abs(ck) < 1e-14) continue;

                    ExprPtr sigma_e = P().make_const(sigma);
                    // 복소 분모 (s - sigma - i*omega)
                    // 대략 실수부만
                    ExprPtr denom_base = P().sub(s_var, sigma_e);
                    ExprPtr denom;
                    if (power == 1) {
                        denom = denom_base;
                    } else {
                        denom = P().make_pow(denom_base, P().make_const(static_cast<double>(power)));
                    }
                    ExprPtr c_expr = P().make_const(ck.real());
                    summands.push_back(P().div(c_expr, denom));
                }
            }
        }
    }

    if (summands.empty()) return P().zero();
    if (summands.size() == 1) return summands[0];
    return P().make_sum(std::move(summands));
}

} // namespace ml_laplace
