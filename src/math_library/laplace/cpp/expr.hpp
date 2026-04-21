#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include "hash.hpp"
#include "subst.hpp"

// expr.hpp — Laplace AST 노드 계층 선언
namespace ml_laplace {

// ------------------------------------------------------------------ 열거형

enum class NodeType : uint8_t {
    CONST    = 0,   // double 리터럴
    RATIONAL = 1,   // int64 num / int64 den  (GCD-정규화)
    VAR      = 2,   // 문자열 이름 (interned)
    SUM      = 3,   // 가변 인자 합 (canonical sorted)
    MUL      = 4,   // 가변 인자 곱 (canonical sorted)
    POW      = 5,   // base ^ exp (2 인자)
    FUNC     = 6,   // 단항 함수
    NEG      = 7,   // 단항 부정
};

enum class FuncId : uint8_t {
    SIN    = 0,
    COS    = 1,
    TAN    = 2,
    ARCSIN = 3,
    ARCCOS = 4,
    ARCTAN = 5,
    SINH   = 6,
    COSH   = 7,
    TANH   = 8,
    EXP    = 9,
    LN     = 10,
    LOG    = 11,
    SQRT   = 12,
};

// ------------------------------------------------------------------ 기본 클래스

class Expr;
using ExprPtr = const Expr*;   // pool-managed, non-owning raw pointer

class Expr {
public:
    Expr(NodeType t, uint64_t hash) noexcept : type_(t), hash_(hash) {}
    virtual ~Expr() noexcept = default;

    NodeType type()  const noexcept { return type_; }
    uint64_t hash()  const noexcept { return hash_; }

    // 구조적 동등성 비교 (intern 삽입 시 충돌 해소용)
    virtual bool structurally_equal(const Expr* other) const noexcept = 0;

    // 수치 평가
    virtual double evalf(const SubstMap& subs) const = 0;

    // 문자열 표현
    virtual std::string to_string() const = 0;

    // 변수 치환 → 새 ExprPtr (hash-consed)
    virtual ExprPtr substitute(const SubstMap& subs) const = 0;

protected:
    NodeType type_;
    uint64_t hash_;
};

// ------------------------------------------------------------------ 구체 클래스 선언

// Const: double 상수
class Const final : public Expr {
public:
    explicit Const(double val);
    double value() const noexcept { return value_; }

    bool structurally_equal(const Expr* o) const noexcept override;
    double evalf(const SubstMap&) const override { return value_; }
    std::string to_string() const override;
    ExprPtr substitute(const SubstMap&) const override;

private:
    double value_;
};

// Rational: 유리수 num/den (GCD-정규화, 부호는 항상 num)
class Rational final : public Expr {
public:
    Rational(int64_t num, int64_t den);
    int64_t num() const noexcept { return num_; }
    int64_t den() const noexcept { return den_; }
    double  as_double() const noexcept {
        return static_cast<double>(num_) / static_cast<double>(den_);
    }

    bool structurally_equal(const Expr* o) const noexcept override;
    double evalf(const SubstMap&) const override { return as_double(); }
    std::string to_string() const override;
    ExprPtr substitute(const SubstMap&) const override;

private:
    int64_t num_, den_;

    static int64_t gcd_abs(int64_t a, int64_t b) noexcept;
};

// Var: 변수 (이름 interned)
class Var final : public Expr {
public:
    explicit Var(const std::string& name);
    const std::string& name() const noexcept { return name_; }

    bool structurally_equal(const Expr* o) const noexcept override;
    double evalf(const SubstMap& subs) const override;
    std::string to_string() const override { return name_; }
    ExprPtr substitute(const SubstMap& subs) const override;

private:
    std::string name_;
};

// Sum: 가변 인자 합 (정규화 후)
class Sum final : public Expr {
public:
    explicit Sum(std::vector<ExprPtr>&& operands);
    const std::vector<ExprPtr>& operands() const noexcept { return operands_; }

    bool structurally_equal(const Expr* o) const noexcept override;
    double evalf(const SubstMap& subs) const override;
    std::string to_string() const override;
    ExprPtr substitute(const SubstMap& subs) const override;

private:
    std::vector<ExprPtr> operands_;
};

// Mul: 가변 인자 곱 (정규화 후)
class Mul final : public Expr {
public:
    explicit Mul(std::vector<ExprPtr>&& operands);
    const std::vector<ExprPtr>& operands() const noexcept { return operands_; }

    bool structurally_equal(const Expr* o) const noexcept override;
    double evalf(const SubstMap& subs) const override;
    std::string to_string() const override;
    ExprPtr substitute(const SubstMap& subs) const override;

private:
    std::vector<ExprPtr> operands_;
};

// Pow: base ^ exp
class Pow final : public Expr {
public:
    Pow(ExprPtr base, ExprPtr exp);
    ExprPtr base() const noexcept { return base_; }
    ExprPtr exp()  const noexcept { return exp_;  }

    bool structurally_equal(const Expr* o) const noexcept override;
    double evalf(const SubstMap& subs) const override;
    std::string to_string() const override;
    ExprPtr substitute(const SubstMap& subs) const override;

private:
    ExprPtr base_;
    ExprPtr exp_;
};

// Func: 단항 함수
class Func final : public Expr {
public:
    Func(FuncId id, ExprPtr arg);
    FuncId  id()  const noexcept { return id_;  }
    ExprPtr arg() const noexcept { return arg_; }

    bool structurally_equal(const Expr* o) const noexcept override;
    double evalf(const SubstMap& subs) const override;
    std::string to_string() const override;
    ExprPtr substitute(const SubstMap& subs) const override;

private:
    FuncId  id_;
    ExprPtr arg_;
};

// Neg: 단항 부정 -x
class Neg final : public Expr {
public:
    explicit Neg(ExprPtr operand);
    ExprPtr operand() const noexcept { return operand_; }

    bool structurally_equal(const Expr* o) const noexcept override;
    double evalf(const SubstMap& subs) const override;
    std::string to_string() const override;
    ExprPtr substitute(const SubstMap& subs) const override;

private:
    ExprPtr operand_;
};

} // namespace ml_laplace
