#pragma once
// polynomial.hpp — 1변수 다항식 및 유리함수 (Phase C)
// double 계수 벡터: coeffs_[k] = k차 항의 계수
// Phase C: Partial fractions + Inverse Laplace 지원

#include "expr.hpp"
#include "pool.hpp"
#include <vector>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <cmath>
#include <algorithm>

namespace ml_laplace {

// ============================================================ Polynomial

class Polynomial {
public:
    // 기본 생성자: 0 다항식
    Polynomial() : coeffs_({0.0}) {}

    // 계수 벡터로 직접 구성 (coeffs[k] = k차 계수, 저차항부터)
    explicit Polynomial(std::vector<double> coeffs);

    // 단순 스칼라 상수 다항식
    static Polynomial scalar(double c);

    // 변수 다항식 (x^1 = 0 + 1*x)
    static Polynomial identity();

    // AST → 다항식 파싱 (실패 시 throw std::runtime_error)
    static Polynomial from_expr(ExprPtr expr, ExprPtr var);

    // 다항식 → AST 변환
    ExprPtr to_expr(ExprPtr var) const;

    // ------------------------------------------------------------------ 기본 조회
    int  degree() const;
    double leading_coeff() const;
    double operator[](int k) const;
    double& operator[](int k);
    const std::vector<double>& coeffs() const { return coeffs_; }

    // ------------------------------------------------------------------ 산술 연산
    Polynomial operator+(const Polynomial& o) const;
    Polynomial operator-(const Polynomial& o) const;
    Polynomial operator*(const Polynomial& o) const;
    Polynomial operator*(double scalar) const;
    Polynomial operator-() const;

    // 다항식 나눗셈: {몫, 나머지}
    std::pair<Polynomial, Polynomial> divmod(const Polynomial& divisor) const;
    Polynomial operator%(const Polynomial& o) const { return divmod(o).second; }
    Polynomial operator/(const Polynomial& o) const { return divmod(o).first; }

    // GCD (Euclidean, floating-point tolerance)
    static Polynomial gcd(const Polynomial& a, const Polynomial& b, double tol = 1e-10);

    // 미분
    Polynomial derivative() const;

    // ------------------------------------------------------------------ 수치 평가
    double              eval(double x) const;
    std::complex<double> eval(std::complex<double> x) const;

    // ------------------------------------------------------------------ 근 (Durand-Kerner)
    std::vector<std::complex<double>> roots(int max_iter = 500, double tol = 1e-12) const;

    // ------------------------------------------------------------------ Squarefree (P / gcd(P, P'))
    Polynomial squarefree(double tol = 1e-10) const;

    // ------------------------------------------------------------------ 계수 정규화 (소수점 trailing zero 제거)
    void trim(double tol = 1e-14);

    // 0 다항식 여부
    bool is_zero(double tol = 1e-14) const;

    std::string to_string() const;

private:
    std::vector<double> coeffs_;  // coeffs_[k] = k차 계수

    void normalize_();
};

// ============================================================ RationalFunction

class RationalFunction {
public:
    RationalFunction(Polynomial num, Polynomial den);

    static RationalFunction from_expr(ExprPtr expr, ExprPtr var);
    ExprPtr to_expr(ExprPtr var) const;

    const Polynomial& num() const { return num_; }
    const Polynomial& den() const { return den_; }

    // 약분: GCD로 나누기
    RationalFunction simplify(double tol = 1e-10) const;

    // 극점 (분모의 근)
    std::vector<std::complex<double>> poles() const;

    // 영점 (분자의 근)
    std::vector<std::complex<double>> zeros() const;

    // 수치 평가 (복소)
    std::complex<double> eval(std::complex<double> s) const;

private:
    Polynomial num_;
    Polynomial den_;
};

} // namespace ml_laplace
