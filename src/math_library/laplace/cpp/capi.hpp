#pragma once
// capi.hpp — Cython 경계용 C-호환 API 래퍼
// ExprPtr(const Expr*)와 SubstMap*를 size_t 핸들로 노출.
// Cython의 const-pointer typedef 및 void* 변환 제한 회피.

#include "pool.hpp"
#include "expr.hpp"
#include "subst.hpp"
#include "rules.hpp"
#include "laplace.hpp"
#include "polynomial.hpp"
#include "partial.hpp"
#include "inverse.hpp"
#include "simplify.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <complex>

namespace ml_laplace {

// ------------------------------------------------------------------ 핸들 변환 (Expr*)
inline size_t expr_to_handle(ExprPtr p) noexcept {
    return reinterpret_cast<size_t>(p);
}
inline ExprPtr expr_from_handle(size_t h) noexcept {
    return reinterpret_cast<ExprPtr>(h);
}

// ------------------------------------------------------------------ 핸들 변환 (SubstMap*)
inline size_t smap_to_handle(SubstMap* m) noexcept {
    return reinterpret_cast<size_t>(m);
}
inline SubstMap* smap_from_handle(size_t h) noexcept {
    return reinterpret_cast<SubstMap*>(h);
}

// ------------------------------------------------------------------ SubstMap 생명주기
inline size_t smap_new() {
    return smap_to_handle(new SubstMap());
}
inline void smap_delete(size_t h) {
    delete smap_from_handle(h);
}
inline void smap_set(size_t h, const char* key, double val) {
    (*smap_from_handle(h))[std::string(key)] = val;
}
inline void smap_clear(size_t h) {
    smap_from_handle(h)->clear();
}

// ------------------------------------------------------------------ Pool 팩토리 (size_t 핸들 반환)
inline size_t pool_make_const(double v) {
    return expr_to_handle(ExprPool::instance().make_const(v));
}
inline size_t pool_make_rational(int64_t n, int64_t d) {
    return expr_to_handle(ExprPool::instance().make_rational(n, d));
}
inline size_t pool_make_var(const char* name) {
    return expr_to_handle(ExprPool::instance().make_var(std::string(name)));
}
inline size_t pool_make_func(uint8_t fid, size_t arg) {
    return expr_to_handle(ExprPool::instance().make_func(
        static_cast<FuncId>(fid), expr_from_handle(arg)));
}
inline size_t pool_make_neg(size_t a) {
    return expr_to_handle(ExprPool::instance().make_neg(expr_from_handle(a)));
}
inline size_t pool_add(size_t a, size_t b) {
    return expr_to_handle(ExprPool::instance().add(
        expr_from_handle(a), expr_from_handle(b)));
}
inline size_t pool_sub(size_t a, size_t b) {
    return expr_to_handle(ExprPool::instance().sub(
        expr_from_handle(a), expr_from_handle(b)));
}
inline size_t pool_mul(size_t a, size_t b) {
    return expr_to_handle(ExprPool::instance().mul(
        expr_from_handle(a), expr_from_handle(b)));
}
inline size_t pool_div(size_t a, size_t b) {
    return expr_to_handle(ExprPool::instance().div(
        expr_from_handle(a), expr_from_handle(b)));
}
inline size_t pool_pow(size_t base, size_t exp_h) {
    return expr_to_handle(ExprPool::instance().pow(
        expr_from_handle(base), expr_from_handle(exp_h)));
}
inline size_t pool_var(const char* name) {
    return expr_to_handle(ExprPool::instance().var(std::string(name)));
}
inline size_t pool_zero() noexcept {
    return expr_to_handle(ExprPool::instance().zero());
}
inline size_t pool_one() noexcept {
    return expr_to_handle(ExprPool::instance().one());
}
inline size_t pool_total_nodes() noexcept {
    return ExprPool::instance().total_nodes();
}
inline size_t pool_intern_hits() noexcept {
    return ExprPool::instance().intern_hits();
}
inline void pool_reset() {
    ExprPool::instance().reset();
}

// ------------------------------------------------------------------ Expr 메서드 (size_t 핸들 기반)
inline double expr_evalf(size_t h, size_t m_h) {
    return expr_from_handle(h)->evalf(*smap_from_handle(m_h));
}
inline void expr_to_string_buf(size_t h, char* buf, size_t buflen) {
    std::string s = expr_from_handle(h)->to_string();
    size_t copy_len = (s.size() < buflen - 1) ? s.size() : (buflen - 1);
    s.copy(buf, copy_len);
    buf[copy_len] = '\0';
}
inline size_t expr_substitute(size_t h, size_t m_h) {
    return expr_to_handle(expr_from_handle(h)->substitute(*smap_from_handle(m_h)));
}
inline int expr_node_type(size_t h) noexcept {
    return static_cast<int>(expr_from_handle(h)->type());
}
inline uint64_t expr_hash(size_t h) noexcept {
    return expr_from_handle(h)->hash();
}

// ------------------------------------------------------------------ Phase B: Laplace 변환
inline size_t laplace_forward_transform(size_t expr_h, size_t t_h, size_t s_h) {
    return expr_to_handle(forward_transform(
        expr_from_handle(expr_h),
        expr_from_handle(t_h),
        expr_from_handle(s_h)));
}

// LaTeX 출력 (버퍼에 복사)
inline void expr_to_latex_buf(size_t h, char* buf, size_t buflen) {
    std::string latex = to_latex(expr_from_handle(h));
    size_t copy_len = (latex.size() < buflen - 1) ? latex.size() : (buflen - 1);
    latex.copy(buf, copy_len);
    buf[copy_len] = '\0';
}

// 기호 치환: var_name 변수를 new_expr로 치환
inline size_t expr_symbolic_substitute(size_t expr_h,
                                        const char* var_name,
                                        size_t new_expr_h) {
    ExprPtr old_var = ExprPool::instance().make_var(std::string(var_name));
    ExprPtr result  = substitute_var(
        expr_from_handle(expr_h), old_var, expr_from_handle(new_expr_h));
    return expr_to_handle(result);
}

// 변환 캐시 초기화
inline void laplace_clear_cache() {
    _transform_cache.clear();
}

// ------------------------------------------------------------------ Phase D: 심볼릭 미분
// expr 을 var 에 대해 한 번 미분
inline size_t laplace_differentiate(size_t expr_h, size_t var_h) {
    return expr_to_handle(differentiate(
        expr_from_handle(expr_h),
        expr_from_handle(var_h)));
}

// ------------------------------------------------------------------ Phase D: expand / cancel
inline size_t laplace_expand(size_t expr_h) {
    return expr_to_handle(expand(expr_from_handle(expr_h)));
}

inline size_t laplace_cancel(size_t expr_h, size_t var_h) {
    ExprPtr var = (var_h != 0) ? expr_from_handle(var_h) : nullptr;
    return expr_to_handle(cancel(expr_from_handle(expr_h), var));
}

// ------------------------------------------------------------------ Phase C: 역 Laplace 변환
inline size_t laplace_inverse_transform(size_t F_h, size_t s_h, size_t t_h) {
    return expr_to_handle(inverse_transform(
        expr_from_handle(F_h),
        expr_from_handle(s_h),
        expr_from_handle(t_h)));
}

// ------------------------------------------------------------------ Phase C: Poles / Zeros
// 결과는 복소수 쌍 (real, imag) 배열로 반환
// out_real/out_imag: 사전 할당된 배열, max_n: 최대 개수
// 반환값: 실제 개수
inline int laplace_compute_poles(size_t F_h, size_t s_h,
                                  double* out_real, double* out_imag, int max_n) {
    auto poles = compute_poles(expr_from_handle(F_h), expr_from_handle(s_h));
    int n = std::min((int)poles.size(), max_n);
    for (int i = 0; i < n; ++i) {
        out_real[i] = poles[i].real();
        out_imag[i] = poles[i].imag();
    }
    return n;
}

inline int laplace_compute_zeros(size_t F_h, size_t s_h,
                                  double* out_real, double* out_imag, int max_n) {
    auto zeros = compute_zeros(expr_from_handle(F_h), expr_from_handle(s_h));
    int n = std::min((int)zeros.size(), max_n);
    for (int i = 0; i < n; ++i) {
        out_real[i] = zeros[i].real();
        out_imag[i] = zeros[i].imag();
    }
    return n;
}

// ------------------------------------------------------------------ Phase C: Final / Initial value
inline double laplace_final_value(size_t F_h, size_t s_h, int* valid_out) {
    bool valid = false;
    double v = final_value(expr_from_handle(F_h), expr_from_handle(s_h), valid);
    if (valid_out) *valid_out = valid ? 1 : 0;
    return v;
}

inline double laplace_initial_value(size_t F_h, size_t s_h, int* valid_out) {
    bool valid = false;
    double v = initial_value(expr_from_handle(F_h), expr_from_handle(s_h), valid);
    if (valid_out) *valid_out = valid ? 1 : 0;
    return v;
}

// ------------------------------------------------------------------ Phase C: Partial fractions AST
inline size_t laplace_partial_fractions(size_t F_h, size_t s_h) {
    RationalFunction rf = RationalFunction::from_expr(
        expr_from_handle(F_h), expr_from_handle(s_h));
    rf = rf.simplify();
    auto terms = partial_fractions(rf);
    return expr_to_handle(partial_fractions_to_expr(terms, expr_from_handle(s_h)));
}

} // namespace ml_laplace
