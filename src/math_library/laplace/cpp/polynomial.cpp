// polynomial.cpp — Polynomial / RationalFunction 구현 (Phase C)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "polynomial.hpp"
#include <sstream>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>
#include <iostream>

namespace ml_laplace {

// ============================================================ 헬퍼

static inline ExprPool& P() { return ExprPool::instance(); }

// 수치 상수 추출 (Const 또는 Rational → double, 아니면 false)
static bool get_numeric_val(ExprPtr e, double& v) noexcept {
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

// ============================================================ Polynomial 생성자

Polynomial::Polynomial(std::vector<double> c) : coeffs_(std::move(c)) {
    if (coeffs_.empty()) coeffs_ = {0.0};
    normalize_();
}

void Polynomial::normalize_() {
    // trailing zero 제거 (최소한 {0} 보장)
    while (coeffs_.size() > 1 && std::abs(coeffs_.back()) < 1e-15) {
        coeffs_.pop_back();
    }
}

Polynomial Polynomial::scalar(double c) {
    return Polynomial({c});
}

Polynomial Polynomial::identity() {
    return Polynomial({0.0, 1.0});
}

// ============================================================ Polynomial::from_expr
// AST → 다항식 파싱 (var에 대한 다항식이어야 함)

static Polynomial _parse_poly(ExprPtr expr, ExprPtr var);

static Polynomial _parse_poly(ExprPtr expr, ExprPtr var) {
    double v = 0.0;

    // 수치 상수
    if (get_numeric_val(expr, v)) {
        return Polynomial::scalar(v);
    }

    // 변수 자체 → x^1
    if (expr == var) {
        return Polynomial::identity();
    }

    // NEG: -(poly) → 계수 반전
    if (expr->type() == NodeType::NEG) {
        const Neg* neg = static_cast<const Neg*>(expr);
        Polynomial p = _parse_poly(neg->operand(), var);
        return -p;
    }

    // POW(expr, n) → poly^n  (n 양의 정수)
    if (expr->type() == NodeType::POW) {
        const Pow* pw = static_cast<const Pow*>(expr);
        double ne = 0.0;
        if (get_numeric_val(pw->exp(), ne)) {
            // 양의 정수 거듭제곱
            if (ne >= 0.0 && ne == std::floor(ne) && ne <= 50.0) {
                int n = static_cast<int>(ne);
                if (n == 0) return Polynomial::scalar(1.0);
                // base가 변수이면 바로 x^n
                if (pw->base() == var) {
                    std::vector<double> c(n + 1, 0.0);
                    c[n] = 1.0;
                    return Polynomial(c);
                }
                // 일반: base 재귀 파싱 후 n회 곱
                Polynomial base_poly = _parse_poly(pw->base(), var);
                Polynomial result = Polynomial::scalar(1.0);
                for (int i = 0; i < n; ++i) result = result * base_poly;
                return result;
            }
            // 상수^상수 → 수치화
            double base_v = 0.0;
            if (get_numeric_val(pw->base(), base_v)) {
                return Polynomial::scalar(std::pow(base_v, ne));
            }
            // 음의 정수 거듭제곱은 유리함수 — 여기서는 불가
            if (ne < 0.0) {
                throw std::runtime_error(
                    "Polynomial::from_expr: negative exponent in Pow — use RationalFunction");
            }
        }
        throw std::runtime_error(
            "Polynomial::from_expr: cannot parse Pow as polynomial: " + expr->to_string());
    }

    // SUM → 각 항 파싱 후 합산
    if (expr->type() == NodeType::SUM) {
        const Sum* s = static_cast<const Sum*>(expr);
        Polynomial acc = Polynomial::scalar(0.0);
        for (ExprPtr op : s->operands()) {
            acc = acc + _parse_poly(op, var);
        }
        return acc;
    }

    // MUL → 각 항 파싱 후 곱산
    if (expr->type() == NodeType::MUL) {
        const Mul* m = static_cast<const Mul*>(expr);
        Polynomial acc = Polynomial::scalar(1.0);
        for (ExprPtr op : m->operands()) {
            acc = acc * _parse_poly(op, var);
        }
        return acc;
    }

    // VAR (다른 변수) → 상수 취급 불가 (다항식이 아님)
    if (expr->type() == NodeType::VAR) {
        throw std::runtime_error(
            "Polynomial::from_expr: expression contains unknown variable '"
            + static_cast<const Var*>(expr)->name() + "'");
    }

    throw std::runtime_error(
        "Polynomial::from_expr: unsupported node type for polynomial: " + expr->to_string());
}

Polynomial Polynomial::from_expr(ExprPtr expr, ExprPtr var) {
    return _parse_poly(expr, var);
}

// ============================================================ Polynomial::to_expr

ExprPtr Polynomial::to_expr(ExprPtr var) const {
    std::vector<ExprPtr> terms;
    for (int k = 0; k <= degree(); ++k) {
        double c = coeffs_[k];
        if (std::abs(c) < 1e-15) continue;

        ExprPtr coef_node = P().make_const(c);
        if (k == 0) {
            terms.push_back(coef_node);
        } else if (k == 1) {
            if (std::abs(c - 1.0) < 1e-15) {
                terms.push_back(var);
            } else {
                terms.push_back(P().mul(coef_node, var));
            }
        } else {
            ExprPtr pow_node = P().make_pow(var, P().make_const(static_cast<double>(k)));
            if (std::abs(c - 1.0) < 1e-15) {
                terms.push_back(pow_node);
            } else {
                terms.push_back(P().mul(coef_node, pow_node));
            }
        }
    }
    if (terms.empty()) return P().zero();
    if (terms.size() == 1) return terms[0];
    return P().make_sum(std::move(terms));
}

// ============================================================ 기본 조회

int Polynomial::degree() const {
    return static_cast<int>(coeffs_.size()) - 1;
}

double Polynomial::leading_coeff() const {
    return coeffs_.back();
}

double Polynomial::operator[](int k) const {
    if (k < 0 || k >= (int)coeffs_.size()) return 0.0;
    return coeffs_[k];
}

double& Polynomial::operator[](int k) {
    if (k < 0) throw std::out_of_range("Polynomial: negative index");
    if (k >= (int)coeffs_.size()) coeffs_.resize(k + 1, 0.0);
    return coeffs_[k];
}

// ============================================================ 산술 연산

Polynomial Polynomial::operator+(const Polynomial& o) const {
    int n = std::max(degree(), o.degree()) + 1;
    std::vector<double> c(n, 0.0);
    for (int k = 0; k < n; ++k) {
        c[k] = (*this)[k] + o[k];
    }
    return Polynomial(c);
}

Polynomial Polynomial::operator-(const Polynomial& o) const {
    int n = std::max(degree(), o.degree()) + 1;
    std::vector<double> c(n, 0.0);
    for (int k = 0; k < n; ++k) {
        c[k] = (*this)[k] - o[k];
    }
    return Polynomial(c);
}

Polynomial Polynomial::operator*(const Polynomial& o) const {
    int n = degree() + o.degree() + 1;
    std::vector<double> c(n, 0.0);
    for (int i = 0; i <= degree(); ++i) {
        for (int j = 0; j <= o.degree(); ++j) {
            c[i + j] += coeffs_[i] * o[j];
        }
    }
    return Polynomial(c);
}

Polynomial Polynomial::operator*(double sc) const {
    std::vector<double> c(coeffs_);
    for (auto& v : c) v *= sc;
    return Polynomial(c);
}

Polynomial Polynomial::operator-() const {
    std::vector<double> c(coeffs_);
    for (auto& v : c) v = -v;
    return Polynomial(c);
}

// ============================================================ divmod (다항식 나눗셈)

std::pair<Polynomial, Polynomial> Polynomial::divmod(const Polynomial& divisor) const {
    if (divisor.is_zero()) {
        throw std::domain_error("Polynomial::divmod: division by zero polynomial");
    }

    // deg(self) < deg(divisor) → 몫=0, 나머지=self
    if (degree() < divisor.degree()) {
        return {Polynomial::scalar(0.0), *this};
    }

    // 다항식 나눗셈 (학교 알고리즘)
    std::vector<double> rem(coeffs_);
    int deg_d = divisor.degree();
    int deg_n = degree();
    std::vector<double> quot(deg_n - deg_d + 1, 0.0);

    double lc = divisor.leading_coeff();
    for (int i = deg_n - deg_d; i >= 0; --i) {
        double coef = rem[i + deg_d] / lc;
        quot[i] = coef;
        for (int j = 0; j <= deg_d; ++j) {
            rem[i + j] -= coef * divisor[j];
        }
    }

    return {Polynomial(quot), Polynomial(rem)};
}

// ============================================================ GCD (Euclidean)

Polynomial Polynomial::gcd(const Polynomial& a, const Polynomial& b, double tol) {
    Polynomial u = a;
    Polynomial v = b;
    u.trim(tol);
    v.trim(tol);

    // Euclidean: gcd(a, b) = gcd(b, a mod b)
    int safety = 200;
    while (!v.is_zero(tol) && --safety > 0) {
        Polynomial r = u % v;
        r.trim(tol);
        u = v;
        v = r;
    }
    // monic 정규화
    if (!u.is_zero(tol)) {
        double lc = u.leading_coeff();
        u = u * (1.0 / lc);
        u.trim(tol);
    }
    return u;
}

// ============================================================ derivative

Polynomial Polynomial::derivative() const {
    if (degree() == 0) return Polynomial::scalar(0.0);
    std::vector<double> c(degree());
    for (int k = 1; k <= degree(); ++k) {
        c[k - 1] = static_cast<double>(k) * coeffs_[k];
    }
    return Polynomial(c);
}

// ============================================================ 수치 평가 (Horner)

double Polynomial::eval(double x) const {
    double result = 0.0;
    for (int k = degree(); k >= 0; --k) {
        result = result * x + coeffs_[k];
    }
    return result;
}

std::complex<double> Polynomial::eval(std::complex<double> x) const {
    std::complex<double> result(0.0, 0.0);
    for (int k = degree(); k >= 0; --k) {
        result = result * x + std::complex<double>(coeffs_[k], 0.0);
    }
    return result;
}

// ============================================================ roots (Durand-Kerner)

std::vector<std::complex<double>> Polynomial::roots(int max_iter, double tol) const {
    int n = degree();
    if (n <= 0) return {};

    // 선형: ax + b = 0 → x = -b/a
    if (n == 1) {
        return {std::complex<double>(-coeffs_[0] / coeffs_[1], 0.0)};
    }

    // monic 다항식으로 정규화
    double lc = leading_coeff();
    Polynomial mono = (*this) * (1.0 / lc);

    // Durand-Kerner 초기 추정: 단위원 위 등간격
    std::vector<std::complex<double>> z(n);
    double angle_step = 2.0 * M_PI / n;
    // 시작 반경: 계수 크기 기반 (Cauchy bound)
    double radius = 1.0;
    for (int k = 0; k < n; ++k) {
        double a = std::abs(mono[k]);
        if (a > 0.0) {
            double r = std::pow(a, 1.0 / (n - k));
            if (r > radius) radius = r;
        }
    }
    radius = std::max(radius, 1.0);

    for (int k = 0; k < n; ++k) {
        double theta = angle_step * k + 0.01;  // 0.01: 축 정렬 회피
        z[k] = std::complex<double>(radius * std::cos(theta), radius * std::sin(theta));
    }

    // Durand-Kerner 반복
    for (int iter = 0; iter < max_iter; ++iter) {
        double max_update = 0.0;
        std::vector<std::complex<double>> z_new(z);

        for (int i = 0; i < n; ++i) {
            std::complex<double> pz = mono.eval(z[i]);
            std::complex<double> denom(1.0, 0.0);
            for (int j = 0; j < n; ++j) {
                if (j != i) denom *= (z[i] - z[j]);
            }
            if (std::abs(denom) < 1e-300) continue;
            std::complex<double> dz = pz / denom;
            z_new[i] = z[i] - dz;
            max_update = std::max(max_update, std::abs(dz));
        }

        z = z_new;
        if (max_update < tol) break;
    }

    // Newton refinement (안정화)
    Polynomial dmono = mono.derivative();
    for (int i = 0; i < n; ++i) {
        for (int ref = 0; ref < 10; ++ref) {
            std::complex<double> fz = mono.eval(z[i]);
            std::complex<double> dfz = dmono.eval(z[i]);
            if (std::abs(dfz) < 1e-300) break;
            std::complex<double> dz = fz / dfz;
            z[i] -= dz;
            if (std::abs(dz) < tol * 1e-3) break;
        }
    }

    return z;
}

// ============================================================ squarefree

Polynomial Polynomial::squarefree(double tol) const {
    Polynomial dp = derivative();
    if (dp.is_zero(tol)) return *this;
    Polynomial g = Polynomial::gcd(*this, dp, tol);
    if (g.is_zero(tol)) return *this;
    Polynomial sq = divmod(g).first;
    sq.trim(tol);
    return sq;
}

// ============================================================ 유틸

void Polynomial::trim(double tol) {
    while (coeffs_.size() > 1 && std::abs(coeffs_.back()) <= tol) {
        coeffs_.pop_back();
    }
}

bool Polynomial::is_zero(double tol) const {
    for (double c : coeffs_) {
        if (std::abs(c) > tol) return false;
    }
    return true;
}

std::string Polynomial::to_string() const {
    std::ostringstream oss;
    bool first = true;
    for (int k = degree(); k >= 0; --k) {
        if (std::abs(coeffs_[k]) < 1e-15) continue;
        if (!first) oss << " + ";
        oss << coeffs_[k];
        if (k > 0) oss << "*x^" << k;
        first = false;
    }
    if (first) oss << "0";
    return oss.str();
}

// ============================================================ RationalFunction

RationalFunction::RationalFunction(Polynomial num, Polynomial den)
    : num_(std::move(num)), den_(std::move(den)) {
    if (den_.is_zero()) {
        throw std::invalid_argument("RationalFunction: denominator is zero");
    }
}

// from_expr: F = P/Q 형태 AST 파싱
// MUL([P, POW(Q, -1)]) 또는 POW(Q, -1) 또는 P (상수/다항식)
RationalFunction RationalFunction::from_expr(ExprPtr expr, ExprPtr var) {
    // 직접 다항식으로 파싱 시도
    try {
        Polynomial p = Polynomial::from_expr(expr, var);
        return RationalFunction(p, Polynomial::scalar(1.0));
    } catch (...) {}

    // Pow(Q, -1) → 1/Q
    if (expr->type() == NodeType::POW) {
        const Pow* pw = static_cast<const Pow*>(expr);
        double exp_v = 0.0;
        if (get_numeric_val(pw->exp(), exp_v) && std::abs(exp_v + 1.0) < 1e-12) {
            Polynomial q = Polynomial::from_expr(pw->base(), var);
            return RationalFunction(Polynomial::scalar(1.0), q);
        }
        // POW(base, -n) → 1 / base^n 처리
        if (get_numeric_val(pw->exp(), exp_v) && exp_v < 0.0 && exp_v == std::floor(exp_v)) {
            int n = static_cast<int>(-exp_v);
            Polynomial q = Polynomial::from_expr(pw->base(), var);
            // q^n 계산
            Polynomial qn = Polynomial::scalar(1.0);
            for (int i = 0; i < n; ++i) qn = qn * q;
            return RationalFunction(Polynomial::scalar(1.0), qn);
        }
    }

    // MUL → 분자/분모 분리
    if (expr->type() == NodeType::MUL) {
        const Mul* m = static_cast<const Mul*>(expr);
        Polynomial num = Polynomial::scalar(1.0);
        Polynomial den = Polynomial::scalar(1.0);

        for (ExprPtr op : m->operands()) {
            // POW(x, -k) 형태 → 분모
            if (op->type() == NodeType::POW) {
                const Pow* pw = static_cast<const Pow*>(op);
                double ev = 0.0;
                if (get_numeric_val(pw->exp(), ev) && ev < 0.0 && ev == std::floor(ev)) {
                    int k = static_cast<int>(-ev);
                    Polynomial q = Polynomial::from_expr(pw->base(), var);
                    Polynomial qk = Polynomial::scalar(1.0);
                    for (int i = 0; i < k; ++i) qk = qk * q;
                    den = den * qk;
                    continue;
                }
            }
            // 분자로 흡수
            Polynomial p_op = Polynomial::from_expr(op, var);
            num = num * p_op;
        }
        return RationalFunction(num, den);
    }

    // SUM → 부분 분수 합산 (각 항을 RationalFunction으로 변환 후 통분)
    if (expr->type() == NodeType::SUM) {
        const Sum* s = static_cast<const Sum*>(expr);
        RationalFunction acc(Polynomial::scalar(0.0), Polynomial::scalar(1.0));
        for (ExprPtr op : s->operands()) {
            RationalFunction rf = RationalFunction::from_expr(op, var);
            // 통분: acc = acc.num * rf.den + rf.num * acc.den
            //             ----------------------------------------
            //                       acc.den * rf.den
            Polynomial new_num = acc.num() * rf.den() + rf.num() * acc.den();
            Polynomial new_den = acc.den() * rf.den();
            acc = RationalFunction(new_num, new_den);
        }
        return acc;
    }

    // NEG: -(rf) → -num / den
    if (expr->type() == NodeType::NEG) {
        const Neg* neg = static_cast<const Neg*>(expr);
        RationalFunction rf = RationalFunction::from_expr(neg->operand(), var);
        return RationalFunction(-rf.num(), rf.den());
    }

    throw std::runtime_error(
        "RationalFunction::from_expr: cannot parse as rational function: " + expr->to_string());
}

ExprPtr RationalFunction::to_expr(ExprPtr var) const {
    ExprPtr num_expr = num_.to_expr(var);
    ExprPtr den_expr = den_.to_expr(var);
    return P().div(num_expr, den_expr);
}

RationalFunction RationalFunction::simplify(double tol) const {
    Polynomial g = Polynomial::gcd(num_, den_, tol);
    if (g.is_zero(tol) || g.degree() == 0) return *this;

    Polynomial sn = num_.divmod(g).first;
    Polynomial sd = den_.divmod(g).first;
    sn.trim(tol);
    sd.trim(tol);

    // 분모 leading 계수 양수로 정규화
    double lc = sd.leading_coeff();
    if (lc < 0.0) {
        sn = -sn;
        sd = -sd;
    }
    return RationalFunction(sn, sd);
}

std::vector<std::complex<double>> RationalFunction::poles() const {
    return den_.roots();
}

std::vector<std::complex<double>> RationalFunction::zeros() const {
    return num_.roots();
}

std::complex<double> RationalFunction::eval(std::complex<double> s) const {
    return num_.eval(s) / den_.eval(s);
}

} // namespace ml_laplace
