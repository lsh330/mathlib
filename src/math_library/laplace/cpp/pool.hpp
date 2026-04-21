#pragma once
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <string>
#include "expr.hpp"

// pool.hpp — 메모리 풀 + hash-consing 싱글턴
namespace ml_laplace {

class ExprPool {
public:
    // ------------------------------------------------------------------ 싱글턴
    static ExprPool& instance();

    // ------------------------------------------------------------------ 팩토리 (모두 interned 결과 반환)
    ExprPtr make_const(double value);
    ExprPtr make_rational(int64_t num, int64_t den);
    ExprPtr make_var(const std::string& name);
    ExprPtr make_sum(std::vector<ExprPtr>&& operands);
    ExprPtr make_mul(std::vector<ExprPtr>&& operands);
    ExprPtr make_pow(ExprPtr base, ExprPtr exp);
    ExprPtr make_func(FuncId id, ExprPtr arg);
    ExprPtr make_neg(ExprPtr operand);

    // ------------------------------------------------------------------ 연산자 API (정규화 + hash-cons 내장)
    ExprPtr add(ExprPtr a, ExprPtr b);
    ExprPtr sub(ExprPtr a, ExprPtr b);
    ExprPtr mul(ExprPtr a, ExprPtr b);
    ExprPtr div(ExprPtr a, ExprPtr b);
    ExprPtr pow(ExprPtr base, ExprPtr exp);
    ExprPtr neg(ExprPtr a);

    // ------------------------------------------------------------------ 싱글턴 상수 접근
    ExprPtr zero()      const noexcept { return zero_;      }
    ExprPtr one()       const noexcept { return one_;       }
    ExprPtr minus_one() const noexcept { return minus_one_; }
    ExprPtr half()      const noexcept { return half_;      }
    ExprPtr var_t()     const noexcept { return var_t_;     }
    ExprPtr var_s()     const noexcept { return var_s_;     }

    // 변수 조회 (make_var 래퍼)
    ExprPtr var(const std::string& name) { return make_var(name); }

    // ------------------------------------------------------------------ 통계
    size_t total_nodes()  const noexcept { return storage_.size(); }
    size_t intern_hits()  const noexcept { return intern_hits_;    }

    // 모든 노드 해제 (주의: 이후 기존 ExprPtr는 dangling)
    void reset();

private:
    // private 생성자 (싱글턴)
    ExprPool();
    ExprPool(const ExprPool&) = delete;
    ExprPool& operator=(const ExprPool&) = delete;

    // ------------------------------------------------------------------ 내부 intern 메서드
    // expr을 pool에 등록하고 canonical pointer를 반환
    ExprPtr intern(std::unique_ptr<Expr> expr);

    // ------------------------------------------------------------------ 저장소
    std::vector<std::unique_ptr<Expr>> storage_;

    // Hash-consing 테이블: hash → 동일 hash를 가진 ExprPtr 목록 (충돌 체인)
    std::unordered_map<uint64_t, std::vector<ExprPtr>> intern_table_;

    // ------------------------------------------------------------------ 상수 캐시
    ExprPtr zero_;
    ExprPtr one_;
    ExprPtr minus_one_;
    ExprPtr half_;
    ExprPtr var_t_;
    ExprPtr var_s_;

    // ------------------------------------------------------------------ 통계 카운터
    mutable size_t intern_hits_ = 0;

    // ------------------------------------------------------------------ 내부 정규화 헬퍼

    // 두 ExprPtr가 모두 Const 또는 Rational인지 확인하고 double 쌍을 반환
    // 아닐 경우 false 반환
    static bool both_numeric(ExprPtr a, ExprPtr b,
                              double& va, double& vb) noexcept;

    // Const/Rational 여부 판별 및 double 추출
    static bool is_numeric(ExprPtr e, double& v) noexcept;

    // Rational 전용: 두 Rational을 더해서 Rational 반환 (overflow 시 Const)
    ExprPtr add_rationals(const Rational* a, const Rational* b);

    // Sum 피연산자 목록 정규화:
    //   1) 중첩 Sum 펼치기
    //   2) 상수 접기
    //   3) 0 제거
    //   4) hash 기준 canonical sort
    // 결과가 단일 원소이면 그 원소 반환, 0원소이면 zero_ 반환
    ExprPtr normalize_sum(std::vector<ExprPtr>&& raw);

    // Mul 피연산자 목록 정규화:
    //   1) 중첩 Mul 펼치기
    //   2) 상수 접기
    //   3) 1 제거, 0 흡수
    //   4) canonical sort
    ExprPtr normalize_mul(std::vector<ExprPtr>&& raw);

    // Pow 정규화: x^0→1, x^1→x, 1^x→1, (x^a)^b→x^(a*b)
    ExprPtr normalize_pow(ExprPtr base, ExprPtr exp);
};

} // namespace ml_laplace
