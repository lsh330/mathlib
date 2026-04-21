// laplace.cpp — Laplace 엔진 최상위 구현: LaTeX 출력
// forward_transform 본체는 rules.cpp 에 위치
#include "laplace.hpp"
#include <sstream>
#include <cmath>
#include <stdexcept>

namespace ml_laplace {

// ================================================================== LaTeX 출력 헬퍼

// 분수 패턴 탐지: Mul([a, Pow(b, -1)]) 형태
// num: 분자 ExprPtr, den: 분모 ExprPtr
// 반환 true 이면 \frac{num}{den} 형태로 출력
static bool detect_fraction(ExprPtr expr, ExprPtr& num_out, ExprPtr& den_out) {
    if (expr->type() != NodeType::MUL) return false;
    const Mul* m = static_cast<const Mul*>(expr);
    const auto& ops = m->operands();

    // 분모 인수 수집 (Pow(x, -1) 패턴)
    std::vector<ExprPtr> num_ops;
    std::vector<ExprPtr> den_ops;

    for (ExprPtr op : ops) {
        if (op->type() == NodeType::POW) {
            const Pow* pw = static_cast<const Pow*>(op);
            double ve = 0.0;
            bool is_neg_exp = false;
            if (pw->exp()->type() == NodeType::RATIONAL) {
                is_neg_exp = static_cast<const Rational*>(pw->exp())->num() < 0;
            } else if (pw->exp()->type() == NodeType::CONST) {
                ve = static_cast<const Const*>(pw->exp())->value();
                is_neg_exp = (ve < 0.0);
            }
            if (is_neg_exp) {
                den_ops.push_back(op->type() == NodeType::POW
                    ? static_cast<const Pow*>(op)->base()
                    : op);
                // 지수가 -1 이외의 음수이면 분모에 base^|exp| 형태
                // 단순화: Pow(base, -n) → den = base^n
                if (op->type() == NodeType::POW) {
                    const Pow* pw2 = static_cast<const Pow*>(op);
                    double ve2 = 0.0;
                    if (pw2->exp()->type() == NodeType::RATIONAL) {
                        const Rational* rexp = static_cast<const Rational*>(pw2->exp());
                        ve2 = rexp->as_double();
                    } else if (pw2->exp()->type() == NodeType::CONST) {
                        ve2 = static_cast<const Const*>(pw2->exp())->value();
                    }
                    if (ve2 == -1.0) {
                        den_ops.back() = pw2->base();
                    } else {
                        // base^|exp|
                        ExprPtr abs_exp = ExprPool::instance().make_const(-ve2);
                        den_ops.back() = ExprPool::instance().make_pow(pw2->base(), abs_exp);
                    }
                }
                continue;
            }
        }
        num_ops.push_back(op);
    }

    if (den_ops.empty()) return false;

    // 분자 구성
    if (num_ops.empty()) {
        num_out = ExprPool::instance().one();
    } else if (num_ops.size() == 1) {
        num_out = num_ops[0];
    } else {
        num_out = ExprPool::instance().make_mul(std::vector<ExprPtr>(num_ops));
    }

    // 분모 구성
    if (den_ops.size() == 1) {
        den_out = den_ops[0];
    } else {
        den_out = ExprPool::instance().make_mul(std::vector<ExprPtr>(den_ops));
    }
    return true;
}

// ------------------------------------------------------------------ to_latex 재귀 구현

std::string to_latex(ExprPtr expr) {
    switch (expr->type()) {
    // 수치 상수
    case NodeType::CONST: {
        double v = static_cast<const Const*>(expr)->value();
        double intpart = 0.0;
        if (std::modf(v, &intpart) == 0.0 &&
            intpart >= -1e15 && intpart <= 1e15) {
            return std::to_string(static_cast<long long>(intpart));
        }
        std::ostringstream oss;
        oss << v;
        return oss.str();
    }
    case NodeType::RATIONAL: {
        const Rational* r = static_cast<const Rational*>(expr);
        if (r->den() == 1) return std::to_string(r->num());
        return "\\frac{" + std::to_string(r->num()) + "}{" +
               std::to_string(r->den()) + "}";
    }
    // 변수
    case NodeType::VAR:
        return static_cast<const Var*>(expr)->name();

    // 부정
    case NodeType::NEG:
        return "-" + to_latex(static_cast<const Neg*>(expr)->operand());

    // 합
    case NodeType::SUM: {
        const Sum* s = static_cast<const Sum*>(expr);
        std::string r;
        for (size_t i = 0; i < s->operands().size(); ++i) {
            std::string term = to_latex(s->operands()[i]);
            if (i == 0) {
                r = term;
            } else {
                // 음수 시작 항이면 공백 없이 붙임 (이미 - 포함)
                if (!term.empty() && term[0] == '-') {
                    r += " " + term;
                } else {
                    r += " + " + term;
                }
            }
        }
        return r;
    }

    // 곱 (분수 패턴 우선 탐지)
    case NodeType::MUL: {
        ExprPtr num_e = nullptr, den_e = nullptr;
        if (detect_fraction(expr, num_e, den_e)) {
            return "\\frac{" + to_latex(num_e) + "}{" + to_latex(den_e) + "}";
        }
        // 일반 곱
        const Mul* m = static_cast<const Mul*>(expr);
        std::string r;
        for (size_t i = 0; i < m->operands().size(); ++i) {
            ExprPtr op = m->operands()[i];
            bool needs_paren = (op->type() == NodeType::SUM ||
                                op->type() == NodeType::NEG);
            std::string ts = needs_paren
                ? "\\left(" + to_latex(op) + "\\right)"
                : to_latex(op);
            if (i > 0) r += " \\cdot ";
            r += ts;
        }
        return r;
    }

    // 거듭제곱
    case NodeType::POW: {
        const Pow* pw = static_cast<const Pow*>(expr);
        std::string bs = to_latex(pw->base());
        std::string es = to_latex(pw->exp());
        bool base_needs_paren = (pw->base()->type() == NodeType::SUM ||
                                  pw->base()->type() == NodeType::MUL ||
                                  pw->base()->type() == NodeType::NEG);
        if (base_needs_paren)
            bs = "\\left(" + bs + "\\right)";
        // 지수가 단일 문자/숫자 이외면 중괄호
        return bs + "^{" + es + "}";
    }

    // 함수
    case NodeType::FUNC: {
        const Func* f = static_cast<const Func*>(expr);
        std::string arg_s = to_latex(f->arg());
        switch (f->id()) {
        case FuncId::SIN:    return "\\sin\\left(" + arg_s + "\\right)";
        case FuncId::COS:    return "\\cos\\left(" + arg_s + "\\right)";
        case FuncId::TAN:    return "\\tan\\left(" + arg_s + "\\right)";
        case FuncId::ARCSIN: return "\\arcsin\\left(" + arg_s + "\\right)";
        case FuncId::ARCCOS: return "\\arccos\\left(" + arg_s + "\\right)";
        case FuncId::ARCTAN: return "\\arctan\\left(" + arg_s + "\\right)";
        case FuncId::SINH:   return "\\sinh\\left(" + arg_s + "\\right)";
        case FuncId::COSH:   return "\\cosh\\left(" + arg_s + "\\right)";
        case FuncId::TANH:   return "\\tanh\\left(" + arg_s + "\\right)";
        case FuncId::EXP:    return "e^{" + arg_s + "}";
        case FuncId::LN:     return "\\ln\\left(" + arg_s + "\\right)";
        case FuncId::LOG:    return "\\log\\left(" + arg_s + "\\right)";
        case FuncId::SQRT:   return "\\sqrt{" + arg_s + "}";
        default:             return "\\mathrm{func}\\left(" + arg_s + "\\right)";
        }
    }

    default:
        throw std::runtime_error("to_latex: unsupported node type");
    }
}

} // namespace ml_laplace
