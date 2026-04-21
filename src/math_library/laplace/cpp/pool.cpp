#include "pool.hpp"
#include "int128_shim.hpp"  // MSVC: ml_int128 → ml_laplace::Int128 alias (GCC/Clang은 no-op)
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace ml_laplace {

// ================================================================== 싱글턴

ExprPool& ExprPool::instance() {
    static ExprPool pool;
    return pool;
}

ExprPool::ExprPool() {
    // 상수 싱글턴 초기화 (intern 통해서 pool에 등록)
    zero_      = intern(std::make_unique<Rational>(0, 1));
    one_       = intern(std::make_unique<Rational>(1, 1));
    minus_one_ = intern(std::make_unique<Rational>(-1, 1));
    half_      = intern(std::make_unique<Rational>(1, 2));
    var_t_     = intern(std::make_unique<Var>("t"));
    var_s_     = intern(std::make_unique<Var>("s"));
}

// ================================================================== intern (핵심 hash-consing)

ExprPtr ExprPool::intern(std::unique_ptr<Expr> expr) {
    uint64_t h = expr->hash();
    auto& bucket = intern_table_[h];

    // 동일 구조 검색 (충돌 체인 순회)
    for (ExprPtr existing : bucket) {
        if (existing->structurally_equal(expr.get())) {
            ++intern_hits_;
            return existing;  // 기존 인스턴스 재사용
        }
    }

    // 신규 등록
    ExprPtr raw = expr.get();
    storage_.push_back(std::move(expr));
    bucket.push_back(raw);
    return raw;
}

// ================================================================== 팩토리 메서드

ExprPtr ExprPool::make_const(double value) {
    return intern(std::make_unique<Const>(value));
}

ExprPtr ExprPool::make_rational(int64_t num, int64_t den) {
    return intern(std::make_unique<Rational>(num, den));
}

ExprPtr ExprPool::make_var(const std::string& name) {
    return intern(std::make_unique<Var>(name));
}

ExprPtr ExprPool::make_sum(std::vector<ExprPtr>&& operands) {
    return normalize_sum(std::move(operands));
}

ExprPtr ExprPool::make_mul(std::vector<ExprPtr>&& operands) {
    return normalize_mul(std::move(operands));
}

ExprPtr ExprPool::make_pow(ExprPtr base, ExprPtr exp) {
    return normalize_pow(base, exp);
}

ExprPtr ExprPool::make_func(FuncId id, ExprPtr arg) {
    return intern(std::make_unique<Func>(id, arg));
}

ExprPtr ExprPool::make_neg(ExprPtr operand) {
    // Neg(Neg(x)) → x
    if (operand->type() == NodeType::NEG)
        return static_cast<const Neg*>(operand)->operand();
    // Neg(Const(c)) → Const(-c)
    if (operand->type() == NodeType::CONST)
        return make_const(-static_cast<const Const*>(operand)->value());
    // Neg(Rational(n,d)) → Rational(-n,d)
    if (operand->type() == NodeType::RATIONAL) {
        const Rational* r = static_cast<const Rational*>(operand);
        return make_rational(-r->num(), r->den());
    }
    return intern(std::make_unique<Neg>(operand));
}

// ================================================================== 연산자 API

ExprPtr ExprPool::add(ExprPtr a, ExprPtr b) {
    // 두 피연산자를 Sum으로 묶어서 normalize
    std::vector<ExprPtr> ops = { a, b };
    return normalize_sum(std::move(ops));
}

ExprPtr ExprPool::sub(ExprPtr a, ExprPtr b) {
    return add(a, make_neg(b));
}

ExprPtr ExprPool::mul(ExprPtr a, ExprPtr b) {
    std::vector<ExprPtr> ops = { a, b };
    return normalize_mul(std::move(ops));
}

ExprPtr ExprPool::div(ExprPtr a, ExprPtr b) {
    // a / b = a * b^(-1)
    ExprPtr inv = make_pow(b, minus_one_);
    return mul(a, inv);
}

ExprPtr ExprPool::pow(ExprPtr base, ExprPtr exp) {
    return normalize_pow(base, exp);
}

ExprPtr ExprPool::neg(ExprPtr a) {
    return make_neg(a);
}

// ================================================================== 통계

void ExprPool::reset() {
    intern_table_.clear();
    storage_.clear();
    intern_hits_ = 0;
    // 상수 재초기화
    zero_      = intern(std::make_unique<Rational>(0, 1));
    one_       = intern(std::make_unique<Rational>(1, 1));
    minus_one_ = intern(std::make_unique<Rational>(-1, 1));
    half_      = intern(std::make_unique<Rational>(1, 2));
    var_t_     = intern(std::make_unique<Var>("t"));
    var_s_     = intern(std::make_unique<Var>("s"));
}

// ================================================================== 내부 헬퍼

bool ExprPool::is_numeric(ExprPtr e, double& v) noexcept {
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

bool ExprPool::both_numeric(ExprPtr a, ExprPtr b,
                             double& va, double& vb) noexcept {
    return is_numeric(a, va) && is_numeric(b, vb);
}

// 두 Rational의 합 계산 (overflow 체크 포함)
ExprPtr ExprPool::add_rationals(const Rational* a, const Rational* b) {
    // a.num/a.den + b.num/b.den = (a.num*b.den + b.num*a.den) / (a.den*b.den)
    // overflow 방지: ml_int128 사용
    ml_int128 num = (ml_int128)a->num() * b->den() + (ml_int128)b->num() * a->den();
    ml_int128 den = (ml_int128)a->den() * b->den();
    // int64 범위 벗어나면 Const(double)로 폴백
    constexpr ml_int128 MAX64 = (ml_int128)std::numeric_limits<int64_t>::max();
    constexpr ml_int128 MIN64 = (ml_int128)std::numeric_limits<int64_t>::min();
    if (num > MAX64 || num < MIN64 || den > MAX64 || den < MIN64) {
        return make_const(a->as_double() + b->as_double());
    }
    return make_rational(static_cast<int64_t>(num), static_cast<int64_t>(den));
}

// ================================================================== normalize_sum

ExprPtr ExprPool::normalize_sum(std::vector<ExprPtr>&& raw) {
    // 1) 중첩 Sum 펼치기
    std::vector<ExprPtr> flat;
    flat.reserve(raw.size() * 2);
    for (ExprPtr p : raw) {
        if (p->type() == NodeType::SUM) {
            const Sum* s = static_cast<const Sum*>(p);
            for (ExprPtr child : s->operands())
                flat.push_back(child);
        } else {
            flat.push_back(p);
        }
    }

    // 2) 상수 접기: 모든 Const/Rational을 분리하여 합산
    //    Rational끼리는 add_rationals, Rational + Const는 Const
    ExprPtr const_acc = nullptr;  // 누적 상수 (nullptr = 아직 없음)

    std::vector<ExprPtr> symbolic;
    symbolic.reserve(flat.size());

    for (ExprPtr p : flat) {
        if (p->type() == NodeType::RATIONAL) {
            if (const_acc == nullptr) {
                const_acc = p;
            } else if (const_acc->type() == NodeType::RATIONAL) {
                const_acc = add_rationals(
                    static_cast<const Rational*>(const_acc),
                    static_cast<const Rational*>(p));
            } else {
                // const_acc가 Const(double)인 경우
                double va = static_cast<const Const*>(const_acc)->value();
                double vb = static_cast<const Rational*>(p)->as_double();
                const_acc = make_const(va + vb);
            }
        } else if (p->type() == NodeType::CONST) {
            double vp = static_cast<const Const*>(p)->value();
            if (const_acc == nullptr) {
                const_acc = p;
            } else {
                double va = 0.0;
                is_numeric(const_acc, va);
                const_acc = make_const(va + vp);
            }
        } else {
            symbolic.push_back(p);
        }
    }

    // 3) 상수 항이 0이면 제외 (zero check: Rational(0,1) 또는 Const(0.0))
    if (const_acc != nullptr) {
        double v = 0.0;
        is_numeric(const_acc, v);
        if (v != 0.0 || const_acc == zero_) {
            // zero_ 자체이면 제외; 0이 아니면 추가
            if (const_acc != zero_)
                symbolic.push_back(const_acc);
        }
        // v == 0.0 이지만 zero_가 아닌 Const(0.0)인 경우도 제외
    }

    // 4) 단일 원소 → 그 원소 반환
    if (symbolic.empty()) return zero_;
    if (symbolic.size() == 1) return symbolic[0];

    // 5) Canonical sort: hash 기준 오름차순 (재현 가능한 순서)
    std::sort(symbolic.begin(), symbolic.end(),
              [](ExprPtr a, ExprPtr b) { return a->hash() < b->hash(); });

    return intern(std::make_unique<Sum>(std::move(symbolic)));
}

// ================================================================== normalize_mul

ExprPtr ExprPool::normalize_mul(std::vector<ExprPtr>&& raw) {
    // 1) 중첩 Mul 펼치기
    std::vector<ExprPtr> flat;
    flat.reserve(raw.size() * 2);
    for (ExprPtr p : raw) {
        if (p->type() == NodeType::MUL) {
            const Mul* m = static_cast<const Mul*>(p);
            for (ExprPtr child : m->operands())
                flat.push_back(child);
        } else {
            flat.push_back(p);
        }
    }

    // 2) 0 흡수 조기 체크
    for (ExprPtr p : flat) {
        if (p == zero_) return zero_;
        double v = 0.0;
        if (is_numeric(p, v) && v == 0.0) return zero_;
    }

    // 3) 상수 접기
    ExprPtr const_acc = nullptr;
    std::vector<ExprPtr> symbolic;
    symbolic.reserve(flat.size());

    for (ExprPtr p : flat) {
        if (p->type() == NodeType::RATIONAL) {
            const Rational* rp = static_cast<const Rational*>(p);
            if (const_acc == nullptr) {
                const_acc = p;
            } else if (const_acc->type() == NodeType::RATIONAL) {
                // 두 Rational 곱: (a/b)*(c/d) = (a*c)/(b*d)
                const Rational* ra = static_cast<const Rational*>(const_acc);
                ml_int128 n = (ml_int128)ra->num() * rp->num();
                ml_int128 d = (ml_int128)ra->den() * rp->den();
                constexpr ml_int128 MAX64 = (ml_int128)std::numeric_limits<int64_t>::max();
                constexpr ml_int128 MIN64 = (ml_int128)std::numeric_limits<int64_t>::min();
                if (n > MAX64 || n < MIN64 || d > MAX64 || d < MIN64)
                    const_acc = make_const(ra->as_double() * rp->as_double());
                else
                    const_acc = make_rational((int64_t)n, (int64_t)d);
            } else {
                double va = 0.0;
                is_numeric(const_acc, va);
                const_acc = make_const(va * rp->as_double());
            }
        } else if (p->type() == NodeType::CONST) {
            double vp = static_cast<const Const*>(p)->value();
            if (const_acc == nullptr) {
                const_acc = p;
            } else {
                double va = 0.0;
                is_numeric(const_acc, va);
                const_acc = make_const(va * vp);
            }
        } else {
            symbolic.push_back(p);
        }
    }

    // 4) 상수 처리
    if (const_acc != nullptr) {
        double v = 0.0;
        is_numeric(const_acc, v);
        if (v == 0.0) return zero_;
        // 1 이면 제거 (Rational(1,1))
        bool is_one = (const_acc == one_) ||
                      (const_acc->type() == NodeType::RATIONAL &&
                       static_cast<const Rational*>(const_acc)->num() == 1 &&
                       static_cast<const Rational*>(const_acc)->den() == 1) ||
                      (const_acc->type() == NodeType::CONST && v == 1.0);
        if (!is_one)
            symbolic.push_back(const_acc);
    }

    // 5) 단일 원소 → 그 원소
    if (symbolic.empty()) return one_;
    if (symbolic.size() == 1) return symbolic[0];

    // 6) Canonical sort
    std::sort(symbolic.begin(), symbolic.end(),
              [](ExprPtr a, ExprPtr b) { return a->hash() < b->hash(); });

    return intern(std::make_unique<Mul>(std::move(symbolic)));
}

// ================================================================== normalize_pow

ExprPtr ExprPool::normalize_pow(ExprPtr base, ExprPtr exp_node) {
    // x^0 → 1
    if (exp_node == zero_) return one_;
    {
        double ve = 0.0;
        if (is_numeric(exp_node, ve) && ve == 0.0) return one_;
    }

    // x^1 → x
    if (exp_node == one_) return base;
    {
        double ve = 0.0;
        if (is_numeric(exp_node, ve) && ve == 1.0) return base;
    }

    // 1^x → 1
    if (base == one_) return one_;
    {
        double vb = 0.0;
        if (is_numeric(base, vb) && vb == 1.0) return one_;
    }

    // 0^0 → NaN (정의 불가)
    {
        double vb = 0.0, ve = 0.0;
        if (is_numeric(base, vb) && is_numeric(exp_node, ve) &&
            vb == 0.0 && ve == 0.0)
            return make_const(std::numeric_limits<double>::quiet_NaN());
    }

    // 0^x (x > 0) → 0, 0^x (x < 0) → inf
    {
        double vb = 0.0, ve = 0.0;
        if (is_numeric(base, vb) && vb == 0.0 && is_numeric(exp_node, ve)) {
            if (ve > 0.0) return zero_;
            return make_const(std::numeric_limits<double>::infinity());
        }
    }

    // (x^a)^b → x^(a*b)  (정수 지수인 경우만 적용 — 안전한 변환)
    if (base->type() == NodeType::POW) {
        const Pow* inner = static_cast<const Pow*>(base);
        double va = 0.0, vb = 0.0;
        if (is_numeric(inner->exp(), va) && is_numeric(exp_node, vb)) {
            // 정수 지수인 경우에만 변환 (분수 지수는 분기점 때문에 스킵)
            bool va_int = (va == std::floor(va));
            bool vb_int = (vb == std::floor(vb));
            if (va_int && vb_int) {
                ExprPtr new_exp = mul(inner->exp(), exp_node);
                return normalize_pow(inner->base(), new_exp);
            }
        }
    }

    // Const^Const → 수치 계산
    {
        double vb = 0.0, ve = 0.0;
        if (is_numeric(base, vb) && is_numeric(exp_node, ve)) {
            // 유리 지수이면 Rational 형태 유지 가능하지만
            // 안전을 위해 정수 지수만 Rational로 계산
            if (is_numeric(exp_node, ve) && ve == std::floor(ve) &&
                base->type() == NodeType::RATIONAL) {
                const Rational* rb = static_cast<const Rational*>(base);
                long long n = static_cast<long long>(ve);
                if (n >= 0 && n <= 20) {
                    // 양의 정수 지수: num^n / den^n
                    ml_int128 rn = 1, rd = 1;
                    for (long long i = 0; i < n; ++i) {
                        rn *= rb->num();
                        rd *= rb->den();
                    }
                    constexpr ml_int128 MAX64 = (ml_int128)std::numeric_limits<int64_t>::max();
                    if (rn <= MAX64 && rd <= MAX64 && rn >= -MAX64)
                        return make_rational((int64_t)rn, (int64_t)rd);
                } else if (n < 0 && n >= -20) {
                    // 음의 정수 지수: den^|n| / num^|n|
                    long long absn = -n;
                    ml_int128 rn = 1, rd = 1;
                    for (long long i = 0; i < absn; ++i) {
                        rn *= rb->den();
                        rd *= rb->num();
                    }
                    constexpr ml_int128 MAX64 = (ml_int128)std::numeric_limits<int64_t>::max();
                    if (rn <= MAX64 && rd <= MAX64 && rd != 0 && rn >= -MAX64)
                        return make_rational((int64_t)rn, (int64_t)rd);
                }
            }
            return make_const(std::pow(vb, ve));
        }
    }

    return intern(std::make_unique<Pow>(base, exp_node));
}

} // namespace ml_laplace
