#include "expr.hpp"
#include "pool.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>   // std::gcd (C++17)

namespace ml_laplace {

// ================================================================== Const

static uint64_t hash_const(double val) noexcept {
    return fnv1a_val(val, fnv1a_val(static_cast<uint8_t>(NodeType::CONST)));
}

Const::Const(double val)
    : Expr(NodeType::CONST, hash_const(val))
    , value_(val) {}

bool Const::structurally_equal(const Expr* o) const noexcept {
    if (o->type() != NodeType::CONST) return false;
    const Const* c = static_cast<const Const*>(o);
    // NaN == NaN는 의도적으로 true (hash-consing 전용)
    if (std::isnan(value_) && std::isnan(c->value_)) return true;
    return value_ == c->value_;
}

std::string Const::to_string() const {
    // 정수 값이면 정수 형태로 출력
    double intpart;
    if (std::modf(value_, &intpart) == 0.0 &&
        intpart >= -1e15 && intpart <= 1e15) {
        std::ostringstream oss;
        oss << static_cast<long long>(intpart);
        return oss.str();
    }
    std::ostringstream oss;
    oss << value_;
    return oss.str();
}

ExprPtr Const::substitute(const SubstMap&) const {
    return this;
}

// ================================================================== Rational

int64_t Rational::gcd_abs(int64_t a, int64_t b) noexcept {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { int64_t t = b; b = a % b; a = t; }
    return a;
}

static uint64_t hash_rational(int64_t num, int64_t den) noexcept {
    uint64_t h = fnv1a_val(static_cast<uint8_t>(NodeType::RATIONAL));
    h = fnv1a_val(num, h);
    h = fnv1a_val(den, h);
    return h;
}

Rational::Rational(int64_t num, int64_t den)
    : Expr(NodeType::RATIONAL, 0)   // hash는 아래에서 재계산
    , num_(num), den_(den)
{
    if (den == 0)
        throw std::domain_error("Rational: denominator cannot be zero");
    // GCD 정규화
    int64_t g = gcd_abs(num, den);
    if (g > 1) { num_ /= g; den_ /= g; }
    // 부호는 항상 num에
    if (den_ < 0) { num_ = -num_; den_ = -den_; }
    // hash_ 재계산 (정규화 후)
    const_cast<uint64_t&>(hash_) = hash_rational(num_, den_);
}

bool Rational::structurally_equal(const Expr* o) const noexcept {
    if (o->type() != NodeType::RATIONAL) return false;
    const Rational* r = static_cast<const Rational*>(o);
    return num_ == r->num_ && den_ == r->den_;
}

std::string Rational::to_string() const {
    if (den_ == 1) {
        return std::to_string(num_);
    }
    return std::to_string(num_) + "/" + std::to_string(den_);
}

ExprPtr Rational::substitute(const SubstMap&) const {
    return this;
}

// ================================================================== Var

static uint64_t hash_var(const std::string& name) noexcept {
    uint64_t h = fnv1a_val(static_cast<uint8_t>(NodeType::VAR));
    h = fnv1a_64(reinterpret_cast<const uint8_t*>(name.data()), name.size(), h);
    return h;
}

Var::Var(const std::string& name)
    : Expr(NodeType::VAR, hash_var(name))
    , name_(name) {}

bool Var::structurally_equal(const Expr* o) const noexcept {
    if (o->type() != NodeType::VAR) return false;
    return name_ == static_cast<const Var*>(o)->name_;
}

double Var::evalf(const SubstMap& subs) const {
    auto it = subs.find(name_);
    if (it == subs.end())
        throw std::runtime_error("Var::evalf: no substitution for '" + name_ + "'");
    return it->second;
}

ExprPtr Var::substitute(const SubstMap& subs) const {
    auto it = subs.find(name_);
    if (it == subs.end()) return this;
    return ExprPool::instance().make_const(it->second);
}

// ================================================================== Sum

static uint64_t hash_sum(const std::vector<ExprPtr>& ops) noexcept {
    std::vector<uint64_t> hs;
    hs.reserve(ops.size());
    for (auto p : ops) hs.push_back(p->hash());
    return combine_hashes(static_cast<uint8_t>(NodeType::SUM), hs);
}

Sum::Sum(std::vector<ExprPtr>&& operands)
    : Expr(NodeType::SUM, hash_sum(operands))
    , operands_(std::move(operands)) {}

bool Sum::structurally_equal(const Expr* o) const noexcept {
    if (o->type() != NodeType::SUM) return false;
    const Sum* s = static_cast<const Sum*>(o);
    if (operands_.size() != s->operands_.size()) return false;
    for (size_t i = 0; i < operands_.size(); ++i)
        if (operands_[i] != s->operands_[i]) return false;
    return true;
}

double Sum::evalf(const SubstMap& subs) const {
    double acc = 0.0;
    for (auto p : operands_) acc += p->evalf(subs);
    return acc;
}

std::string Sum::to_string() const {
    std::string r;
    for (size_t i = 0; i < operands_.size(); ++i) {
        if (i > 0) {
            // Neg나 음수 Const/Rational이면 그냥 붙임 (부호 포함)
            const Expr* op = operands_[i];
            bool is_neg = (op->type() == NodeType::NEG);
            bool is_neg_const = (op->type() == NodeType::CONST &&
                                 static_cast<const Const*>(op)->value() < 0.0);
            bool is_neg_rat   = (op->type() == NodeType::RATIONAL &&
                                 static_cast<const Rational*>(op)->num() < 0);
            if (is_neg || is_neg_const || is_neg_rat)
                r += " + ";
            else
                r += " + ";
        }
        r += operands_[i]->to_string();
    }
    return r;
}

ExprPtr Sum::substitute(const SubstMap& subs) const {
    bool changed = false;
    std::vector<ExprPtr> new_ops;
    new_ops.reserve(operands_.size());
    for (auto p : operands_) {
        ExprPtr q = p->substitute(subs);
        new_ops.push_back(q);
        if (q != p) changed = true;
    }
    if (!changed) return this;
    return ExprPool::instance().make_sum(std::move(new_ops));
}

// ================================================================== Mul

static uint64_t hash_mul(const std::vector<ExprPtr>& ops) noexcept {
    std::vector<uint64_t> hs;
    hs.reserve(ops.size());
    for (auto p : ops) hs.push_back(p->hash());
    return combine_hashes(static_cast<uint8_t>(NodeType::MUL), hs);
}

Mul::Mul(std::vector<ExprPtr>&& operands)
    : Expr(NodeType::MUL, hash_mul(operands))
    , operands_(std::move(operands)) {}

bool Mul::structurally_equal(const Expr* o) const noexcept {
    if (o->type() != NodeType::MUL) return false;
    const Mul* m = static_cast<const Mul*>(o);
    if (operands_.size() != m->operands_.size()) return false;
    for (size_t i = 0; i < operands_.size(); ++i)
        if (operands_[i] != m->operands_[i]) return false;
    return true;
}

double Mul::evalf(const SubstMap& subs) const {
    double acc = 1.0;
    for (auto p : operands_) acc *= p->evalf(subs);
    return acc;
}

// 곱 출력: 단일 항이 Sum이면 괄호 처리
static std::string mul_term(const Expr* p) {
    if (p->type() == NodeType::SUM)
        return "(" + p->to_string() + ")";
    return p->to_string();
}

std::string Mul::to_string() const {
    std::string r;
    for (size_t i = 0; i < operands_.size(); ++i) {
        if (i > 0) r += "*";
        r += mul_term(operands_[i]);
    }
    return r;
}

ExprPtr Mul::substitute(const SubstMap& subs) const {
    bool changed = false;
    std::vector<ExprPtr> new_ops;
    new_ops.reserve(operands_.size());
    for (auto p : operands_) {
        ExprPtr q = p->substitute(subs);
        new_ops.push_back(q);
        if (q != p) changed = true;
    }
    if (!changed) return this;
    return ExprPool::instance().make_mul(std::move(new_ops));
}

// ================================================================== Pow

static uint64_t hash_pow(ExprPtr base, ExprPtr exp) noexcept {
    std::vector<uint64_t> hs = { base->hash(), exp->hash() };
    return combine_hashes(static_cast<uint8_t>(NodeType::POW), hs);
}

Pow::Pow(ExprPtr base, ExprPtr exp)
    : Expr(NodeType::POW, hash_pow(base, exp))
    , base_(base), exp_(exp) {}

bool Pow::structurally_equal(const Expr* o) const noexcept {
    if (o->type() != NodeType::POW) return false;
    const Pow* p = static_cast<const Pow*>(o);
    return base_ == p->base_ && exp_ == p->exp_;
}

double Pow::evalf(const SubstMap& subs) const {
    return std::pow(base_->evalf(subs), exp_->evalf(subs));
}

std::string Pow::to_string() const {
    std::string bs = base_->to_string();
    std::string es = exp_->to_string();
    // base가 복합식이면 괄호
    bool base_needs_paren = (base_->type() == NodeType::SUM ||
                              base_->type() == NodeType::MUL ||
                              base_->type() == NodeType::NEG);
    bool exp_needs_paren  = (exp_->type() == NodeType::SUM ||
                              exp_->type() == NodeType::MUL ||
                              exp_->type() == NodeType::POW);
    if (base_needs_paren) bs = "(" + bs + ")";
    if (exp_needs_paren)  es = "(" + es + ")";
    return bs + "**" + es;
}

ExprPtr Pow::substitute(const SubstMap& subs) const {
    ExprPtr nb = base_->substitute(subs);
    ExprPtr ne = exp_->substitute(subs);
    if (nb == base_ && ne == exp_) return this;
    return ExprPool::instance().make_pow(nb, ne);
}

// ================================================================== Func

static const char* func_name(FuncId id) noexcept {
    switch (id) {
        case FuncId::SIN:    return "sin";
        case FuncId::COS:    return "cos";
        case FuncId::TAN:    return "tan";
        case FuncId::ARCSIN: return "arcsin";
        case FuncId::ARCCOS: return "arccos";
        case FuncId::ARCTAN: return "arctan";
        case FuncId::SINH:   return "sinh";
        case FuncId::COSH:   return "cosh";
        case FuncId::TANH:   return "tanh";
        case FuncId::EXP:    return "exp";
        case FuncId::LN:     return "ln";
        case FuncId::LOG:    return "log";
        case FuncId::SQRT:   return "sqrt";
        default:             return "func";
    }
}

static uint64_t hash_func(FuncId id, ExprPtr arg) noexcept {
    std::vector<uint64_t> hs = {
        static_cast<uint64_t>(static_cast<uint8_t>(id)),
        arg->hash()
    };
    return combine_hashes(static_cast<uint8_t>(NodeType::FUNC), hs);
}

Func::Func(FuncId id, ExprPtr arg)
    : Expr(NodeType::FUNC, hash_func(id, arg))
    , id_(id), arg_(arg) {}

bool Func::structurally_equal(const Expr* o) const noexcept {
    if (o->type() != NodeType::FUNC) return false;
    const Func* f = static_cast<const Func*>(o);
    return id_ == f->id_ && arg_ == f->arg_;
}

double Func::evalf(const SubstMap& subs) const {
    double v = arg_->evalf(subs);
    switch (id_) {
        case FuncId::SIN:    return std::sin(v);
        case FuncId::COS:    return std::cos(v);
        case FuncId::TAN:    return std::tan(v);
        case FuncId::ARCSIN: return std::asin(v);
        case FuncId::ARCCOS: return std::acos(v);
        case FuncId::ARCTAN: return std::atan(v);
        case FuncId::SINH:   return std::sinh(v);
        case FuncId::COSH:   return std::cosh(v);
        case FuncId::TANH:   return std::tanh(v);
        case FuncId::EXP:    return std::exp(v);
        case FuncId::LN:     return std::log(v);
        case FuncId::LOG:    return std::log10(v);
        case FuncId::SQRT:   return std::sqrt(v);
        default:
            throw std::runtime_error("Func::evalf: unknown FuncId");
    }
}

std::string Func::to_string() const {
    return std::string(func_name(id_)) + "(" + arg_->to_string() + ")";
}

ExprPtr Func::substitute(const SubstMap& subs) const {
    ExprPtr na = arg_->substitute(subs);
    if (na == arg_) return this;
    return ExprPool::instance().make_func(id_, na);
}

// ================================================================== Neg

static uint64_t hash_neg(ExprPtr op) noexcept {
    std::vector<uint64_t> hs = { op->hash() };
    return combine_hashes(static_cast<uint8_t>(NodeType::NEG), hs);
}

Neg::Neg(ExprPtr operand)
    : Expr(NodeType::NEG, hash_neg(operand))
    , operand_(operand) {}

bool Neg::structurally_equal(const Expr* o) const noexcept {
    if (o->type() != NodeType::NEG) return false;
    return operand_ == static_cast<const Neg*>(o)->operand_;
}

double Neg::evalf(const SubstMap& subs) const {
    return -operand_->evalf(subs);
}

std::string Neg::to_string() const {
    // 원자 노드면 그냥 -x, 복합이면 -(...)
    bool needs_paren = (operand_->type() == NodeType::SUM ||
                        operand_->type() == NodeType::MUL);
    if (needs_paren)
        return "-(" + operand_->to_string() + ")";
    return "-" + operand_->to_string();
}

ExprPtr Neg::substitute(const SubstMap& subs) const {
    ExprPtr no = operand_->substitute(subs);
    if (no == operand_) return this;
    return ExprPool::instance().make_neg(no);
}

} // namespace ml_laplace
