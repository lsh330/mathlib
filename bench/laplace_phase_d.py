"""bench/laplace_phase_d.py — Phase D 검증 스크립트"""
import sys
import os
import math
import time

# 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from math_library.laplace import Laplace, t, s, Sin, Cos, Exp, const, symbol
import math_library as m

L = Laplace()

# ================================================================== 1. diff_symbolic
print("=" * 60)
print("1. diff_symbolic")

F = L.transform(Sin(2 * t))   # 2/(s^2+4)
print(f"   F(s) = {F}")

t0 = time.perf_counter()
dF = F.diff(s)
t1 = time.perf_counter()
diff_us = (t1 - t0) * 1e6

print(f"   dF/ds = {dF}")
for sv in [1.0, 2.0, 3.0]:
    got = dF.evalf(s=sv)
    exp = -4 * sv / (sv**2 + 4)**2
    assert abs(got - exp) < 1e-10, f"diff fail at s={sv}: got={got}, exp={exp}"
print(f"   PASS  (diff AST: {diff_us:.1f} μs)")

# ================================================================== 2. diff_numeric
print("=" * 60)
print("2. diff_numeric (Ridders)")

t0 = time.perf_counter()
ld = F.diff_numeric('s')
for sv in [1.0, 2.0, 3.0]:
    got = ld.evalf(s=sv)
    exp = -4 * sv / (sv**2 + 4)**2
    assert abs(got - exp) < 1e-8, f"numeric diff fail at s={sv}: got={got}, exp={exp}"
t1 = time.perf_counter()
ndiff_us = (t1 - t0) * 1e6

print(f"   PASS  (3 evals: {ndiff_us:.1f} μs)")

# ================================================================== 3. series (e^s at 0, order 5)
print("=" * 60)
print("3. series (exp(s) Taylor at 0, order=5)")

g = Exp(s)
t0 = time.perf_counter()
ts = g.series('s', about=0.0, order=5)
coeffs = ts.compute()
t1 = time.perf_counter()
series_us = (t1 - t0) * 1e6

expected = [1.0, 1.0, 0.5, 1.0/6, 1.0/24, 1.0/120]
print(f"   Coefficients: {[round(c, 8) for c in coeffs]}")
for k, (a, b) in enumerate(zip(coeffs, expected)):
    assert abs(a - b) < 1e-6, f"Taylor coef k={k}: got={a} vs exp={b}"
print(f"   PASS  ({series_us:.1f} μs for order-5 Taylor)")

# as_expr 동작 확인
poly = ts.as_expr()
for sv in [0.0, 0.5, 1.0]:
    got = ts.evalf(s=sv)
    exp = math.exp(sv)   # 급수가 근사이므로 완전 일치는 기대 안 함
    # order=5에서 s=1이면 ~2.7167 vs e≈2.71828, 오차 허용
    assert abs(got - exp) < 1e-2, f"series evalf fail s={sv}"
print("   as_expr / evalf PASS")

# ================================================================== 4. expand
print("=" * 60)
print("4. expand: (s+1)*(s+2)")

e = (s + const(1)) * (s + const(2))
t0 = time.perf_counter()
ex = e.expand()
t1 = time.perf_counter()
expand_us = (t1 - t0) * 1e6

print(f"   expanded = {ex}")
for sv in [0.0, 1.0, 5.0, -1.0]:
    got = ex.evalf(s=sv)
    exp = sv**2 + 3 * sv + 2
    assert abs(got - exp) < 1e-10, f"expand fail at s={sv}: got={got}, exp={exp}"
print(f"   PASS  ({expand_us:.1f} μs)")

# ================================================================== 5. cancel
print("=" * 60)
print("5. cancel: (s^2-1)/(s-1)")

F2 = (s**2 - const(1)) / (s - const(1))
t0 = time.perf_counter()
c = F2.cancel()
t1 = time.perf_counter()
cancel_us = (t1 - t0) * 1e6

print(f"   cancelled = {c}")
for sv in [2.0, 3.0, 5.0]:
    got = c.evalf(s=sv)
    exp = sv + 1.0
    assert abs(got - exp) < 1e-10, f"cancel fail at s={sv}: got={got}, exp={exp}"
print(f"   PASS  ({cancel_us:.1f} μs)")

# ================================================================== 6. Transfer function convenience
print("=" * 60)
print("6. Transfer function 편의 메서드")

G = L.transform(Exp(-t))    # 1/(s+1)
H = L.transform(Exp(-2*t))  # 1/(s+2)

series_conn = G.series_connect(H)
parallel_conn = G.parallel_connect(H)
feedback_conn = G.feedback()

print(f"   G = {G}")
print(f"   H = {H}")
print(f"   G·H (series)   = {series_conn}")
print(f"   G+H (parallel) = {parallel_conn}")
print(f"   G/(1+G) (fb)   = {feedback_conn}")

# 수치 검증
for sv in [1.0, 2.0]:
    g_v = G.evalf(s=sv)
    h_v = H.evalf(s=sv)
    assert abs(series_conn.evalf(s=sv) - g_v * h_v) < 1e-10, "series_connect fail"
    assert abs(parallel_conn.evalf(s=sv) - (g_v + h_v)) < 1e-10, "parallel_connect fail"
    exp_fb = g_v / (1 + g_v)
    got_fb = feedback_conn.evalf(s=sv)
    assert abs(got_fb - exp_fb) < 1e-8, f"feedback fail at s={sv}: got={got_fb}, exp={exp_fb}"

print("   PASS")

# ================================================================== 7. Frequency response
print("=" * 60)
print("7. Frequency response |H(jω)|")

for w in [0.1, 1.0, 10.0]:
    mag = G.magnitude(w)
    exp = 1.0 / math.sqrt(w**2 + 1)
    assert abs(mag - exp) < 1e-8, f"magnitude fail w={w}: got={mag}, exp={exp}"

# phase 검증
import cmath
for w in [0.5, 1.0, 5.0]:
    ph = G.phase(w)
    exp_ph = cmath.phase(complex(0.0, w) + 1.0) * (-1)  # angle of 1/(jw+1)
    # 1/(jw+1) = 1/((1+jw)) → phase = -arctan(w)
    exp_ph2 = -math.atan(w)
    assert abs(ph - exp_ph2) < 1e-8, f"phase fail w={w}: got={ph}, exp={exp_ph2}"

print("   PASS")

# list 입력 테스트
mags = G.magnitude([0.1, 1.0, 10.0])
assert len(mags) == 3
print("   list input PASS")

# ================================================================== 8. 기존 mathlib 회귀
print("=" * 60)
print("8. mathlib 회귀")

assert abs(m.sin(1.2345) - math.sin(1.2345)) < 1e-14, "sin 회귀 실패"
assert abs(m.exp(0.5) - math.exp(0.5)) < 1e-14, "exp 회귀 실패"
assert abs(m.ln(2.0) - math.log(2.0)) < 1e-14, "ln 회귀 실패"
print("   PASS")

# ================================================================== 성능 요약
print()
print("=" * 60)
print("성능 요약")
print(f"  diff_symbolic    : {diff_us:.1f} μs")
print(f"  diff_numeric x3  : {ndiff_us:.1f} μs")
print(f"  series order=5   : {series_us:.1f} μs")
print(f"  expand           : {expand_us:.1f} μs")
print(f"  cancel           : {cancel_us:.1f} μs")
print()
print("모든 검증 통과.")
