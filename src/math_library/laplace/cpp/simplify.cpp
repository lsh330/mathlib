// simplify.cpp — Phase D: expand + cancel 구현
// expand  : 분배 법칙 (재귀적 Mul/Sum 전개)
// cancel  : RationalFunction::simplify() 기반 약분

#include "simplify.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace ml_laplace {

// ------------------------------------------------------------------ 내부 헬퍼
static inline ExprPool& P() { return ExprPool::instance(); }

static bool get_num(ExprPtr e, double& v) noexcept {
    if (e->type() == NodeType::CONST) {
        v = static_cast<const Const*>(e)->value(); return true;
    }
    if (e->type() == NodeType::RATIONAL) {
        v = static_cast<const Rational*>(e)->as_double(); return true;
    }
    return false;
}

// ================================================================== find_var
// 식에서 첫 번째 VAR 노드를 DFS로 탐색
static ExprPtr find_var(ExprPtr e) {
    if (!e) return nullptr;
    switch (e->type()) {
    case NodeType::VAR:
        return e;
    case NodeType::NEG:
        return find_var(static_cast<const Neg*>(e)->operand());
    case NodeType::SUM: {
        for (ExprPtr op : static_cast<const Sum*>(e)->operands()) {
            ExprPtr v = find_var(op);
            if (v) return v;
        }
        return nullptr;
    }
    case NodeType::MUL: {
        for (ExprPtr op : static_cast<const Mul*>(e)->operands()) {
            ExprPtr v = find_var(op);
            if (v) return v;
        }
        return nullptr;
    }
    case NodeType::POW: {
        ExprPtr v = find_var(static_cast<const Pow*>(e)->base());
        if (v) return v;
        return find_var(static_cast<const Pow*>(e)->exp());
    }
    case NodeType::FUNC:
        return find_var(static_cast<const Func*>(e)->arg());
    default:
        return nullptr;
    }
}

// ================================================================== expand
// 분배 법칙 전개: Mul(a+b, c+d) → ac+ad+bc+bd
// 재귀 순환 방지: distribute_two 는 단순 make_mul/make_sum 만 사용
//                 expand 재귀는 각 인수(operand) 수준에서만 호출

// Sum 내부 항 리스트 추출 (중첩 Sum 펼치기, 비재귀)
static void collect_sum_terms(ExprPtr e, std::vector<ExprPtr>& out) {
    // iterative DFS
    std::vector<ExprPtr> stack = {e};
    while (!stack.empty()) {
        ExprPtr cur = stack.back();
        stack.pop_back();
        if (cur->type() == NodeType::SUM) {
            const auto& ops = static_cast<const Sum*>(cur)->operands();
            for (int i = (int)ops.size() - 1; i >= 0; --i) {
                stack.push_back(ops[i]);
            }
        } else {
            out.push_back(cur);
        }
    }
}

// 두 항 목록의 Cartesian 곱 → 합 (Sum AST)
// 각 곱은 pool.make_mul (정규화 포함)로 생성
static ExprPtr distribute_terms(const std::vector<ExprPtr>& A,
                                 const std::vector<ExprPtr>& B) {
    std::vector<ExprPtr> result;
    result.reserve(A.size() * B.size());
    for (ExprPtr a : A) {
        for (ExprPtr b : B) {
            result.push_back(P().mul(a, b));
        }
    }
    if (result.empty()) return P().zero();
    if (result.size() == 1) return result[0];
    return P().make_sum(std::move(result));
}

ExprPtr expand(ExprPtr e) {
    if (!e) return P().zero();

    switch (e->type()) {
    case NodeType::CONST:
    case NodeType::RATIONAL:
    case NodeType::VAR:
        return e;

    case NodeType::NEG: {
        ExprPtr inner = expand(static_cast<const Neg*>(e)->operand());
        if (inner == static_cast<const Neg*>(e)->operand()) return e;
        return P().make_neg(inner);
    }

    case NodeType::SUM: {
        const Sum* s = static_cast<const Sum*>(e);
        std::vector<ExprPtr> terms;
        terms.reserve(s->operands().size() * 2);
        for (ExprPtr op : s->operands()) {
            ExprPtr ex = expand(op);
            collect_sum_terms(ex, terms);
        }
        if (terms.empty()) return P().zero();
        if (terms.size() == 1) return terms[0];
        return P().make_sum(std::move(terms));
    }

    case NodeType::POW: {
        const Pow* pw = static_cast<const Pow*>(e);
        double ve = 0.0;
        // 양정수 지수 한정 (n >= 2, n <= 5)
        if (get_num(pw->exp(), ve) && ve == std::floor(ve) && ve >= 2.0 && ve <= 5.0) {
            int n = static_cast<int>(ve);
            ExprPtr base_ex = expand(pw->base());
            // base^n: distribute_terms 로 accumulate (재귀 없음)
            std::vector<ExprPtr> acc;
            collect_sum_terms(base_ex, acc);  // 초기값: base_ex 항 목록

            for (int i = 1; i < n; ++i) {
                std::vector<ExprPtr> base_terms;
                collect_sum_terms(base_ex, base_terms);
                ExprPtr combined = distribute_terms(acc, base_terms);
                // combined 의 각 항 내부(Mul 내)를 더 전개할 필요 없음:
                // base_ex 는 이미 expand된 Sum 이므로 각 항은 단항
                acc.clear();
                collect_sum_terms(combined, acc);
            }
            if (acc.empty()) return P().zero();
            if (acc.size() == 1) return acc[0];
            return P().make_sum(std::move(acc));
        }
        // 그 외 Pow: 자식만 expand
        ExprPtr base_ex = expand(pw->base());
        ExprPtr exp_ex  = expand(pw->exp());
        if (base_ex == pw->base() && exp_ex == pw->exp()) return e;
        return P().make_pow(base_ex, exp_ex);
    }

    case NodeType::MUL: {
        const Mul* m = static_cast<const Mul*>(e);
        // 1단계: 각 인수 expand
        std::vector<ExprPtr> expanded_ops;
        expanded_ops.reserve(m->operands().size());
        for (ExprPtr op : m->operands()) {
            expanded_ops.push_back(expand(op));
        }

        // 2단계: 왼쪽부터 순차 분배 전개 (재귀 없음)
        std::vector<ExprPtr> acc;
        collect_sum_terms(expanded_ops[0], acc);

        for (size_t i = 1; i < expanded_ops.size(); ++i) {
            std::vector<ExprPtr> next;
            collect_sum_terms(expanded_ops[i], next);
            ExprPtr combined = distribute_terms(acc, next);
            acc.clear();
            collect_sum_terms(combined, acc);
        }
        if (acc.empty()) return P().zero();
        if (acc.size() == 1) return acc[0];
        return P().make_sum(std::move(acc));
    }

    case NodeType::FUNC: {
        const Func* f = static_cast<const Func*>(e);
        ExprPtr arg_ex = expand(f->arg());
        if (arg_ex == f->arg()) return e;
        return P().make_func(f->id(), arg_ex);
    }

    default:
        return e;
    }
}

// ================================================================== cancel
// RationalFunction::from_expr + simplify + to_expr

ExprPtr cancel(ExprPtr e, ExprPtr var) {
    if (!var) {
        var = find_var(e);
        if (!var) return e;  // 변수 없으면 상수 → 그대로
    }
    try {
        RationalFunction rf = RationalFunction::from_expr(e, var);
        rf = rf.simplify();
        return rf.to_expr(var);
    } catch (...) {
        // 유리함수로 파싱 실패 → 원식 반환
        return e;
    }
}

ExprPtr cancel_auto(ExprPtr e) {
    ExprPtr var = find_var(e);
    return cancel(e, var);
}

} // namespace ml_laplace
