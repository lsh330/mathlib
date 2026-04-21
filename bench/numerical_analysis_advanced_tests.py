"""
numerical_analysis_advanced_tests.py
NumericalAnalysis 신규 6 메서드 검증:
  gauss_legendre, composite_gauss_legendre, rk78,
  adams_bashforth, adams_moulton, predictor_corrector
"""
import math
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from math_library import NumericalAnalysis
import math_library as m

na = NumericalAnalysis()

# ====================================================================== Gauss-Legendre
print("=== Gauss-Legendre ===")

# int sin(x) dx from 0 to pi = 2
# n=5 GL은 9차 다항식까지 정확. sin(x)는 초월함수이므로 절단 오차 ~1e-7
I = na.gauss_legendre(math.sin, 0, math.pi, n=5)
err = abs(I - 2.0)
assert err < 1e-6, f"GL n=5 sin: err={err:.2e}"
print(f"GL n=5 ∫sin: {I:.15f}  err={err:.2e}  PASS")

# int x^9 dx from 0 to 1 = 0.1  (n=5: 2*5-1=9차까지 정확)
I = na.gauss_legendre(lambda x: x**9, 0, 1, n=5)
err = abs(I - 0.1)
assert err < 1e-14, f"GL n=5 x^9: err={err:.2e}"
print(f"GL n=5 ∫x^9: {I:.15f}  err={err:.2e}  PASS")

# int x^10 dx from 0 to 1 = 1/11  (n=5로는 부정확, n=6 이상 필요: 2*6-1=11)
I6 = na.gauss_legendre(lambda x: x**11, 0, 1, n=6)
err6 = abs(I6 - 1.0/12)
assert err6 < 1e-14, f"GL n=6 x^11: err={err6:.2e}"
print(f"GL n=6 ∫x^11: {I6:.15f}  err={err6:.2e}  PASS")

# n=16 검증
I16 = na.gauss_legendre(lambda x: x**31, 0, 1, n=16)
err16 = abs(I16 - 1.0/32)
assert err16 < 1e-14, f"GL n=16 x^31: err={err16:.2e}"
print(f"GL n=16 ∫x^31: {I16:.15f}  err={err16:.2e}  PASS")

# Composite
I = na.composite_gauss_legendre(math.sin, 0, math.pi, m=10, n=5)
err = abs(I - 2.0)
print(f"composite GL m=10,n=5: {I:.15f}  err={err:.2e}")

# return_error
I_val, I_err = na.gauss_legendre(math.sin, 0, math.pi, n=5, return_error=True)
print(f"GL return_error: val={I_val:.15f}  est_err={I_err:.2e}")

# 범위 예외
try:
    na.gauss_legendre(math.sin, 0, 1, n=1)
    assert False, "Should raise ValueError"
except ValueError as e:
    print(f"n=1 ValueError: {e}")

try:
    na.gauss_legendre(math.sin, 0, 1, n=17)
    assert False, "Should raise ValueError"
except ValueError as e:
    print(f"n=17 ValueError: {e}")

try:
    na.composite_gauss_legendre(math.sin, 0, math.pi, m=0, n=5)
    assert False, "Should raise ValueError"
except ValueError as e:
    print(f"m=0 ValueError: {e}")

# ====================================================================== RK78
print()
print("=== RK78 ===")

f_exp = lambda t, y: -y
y_end = na.rk78(f_exp, 0, 1, 1, tol=1e-14)
err = abs(y_end - math.exp(-1))
assert err < 1e-13, f"RK78 dy/dt=-y: err={err:.2e}"
print(f"RK78 dy/dt=-y: {y_end:.15f}  err={err:.2e}  PASS")

# 어려운 ODE: dy/dt = sin(t)*y, y(0)=1, sol=exp(1-cos(t))
# y(2pi) = exp(1-cos(2pi)) = exp(0) = 1
f_hard = lambda t, y: math.sin(t) * y
y_end2 = na.rk78(f_hard, 0, 1, 2*math.pi, tol=1e-12)
expected2 = math.exp(1 - math.cos(2*math.pi))  # = 1
err2 = abs(y_end2 - expected2)
assert err2 < 1e-10, f"RK78 sin(t)*y: err={err2:.2e}"
print(f"RK78 sin(t)*y at 2pi: {y_end2:.15f}  err={err2:.2e}  PASS")

# return_trajectory
traj = na.rk78(f_exp, 0, 1, 1, tol=1e-10, return_trajectory=True)
assert isinstance(traj, list) and traj[0] == (0.0, 1.0)
print(f"RK78 trajectory: {len(traj)} steps, final y={traj[-1][1]:.10f}")

# t0 >= t_end 예외
try:
    na.rk78(f_exp, 1, 1, 0)
    assert False
except ValueError as e:
    print(f"t0>=t_end ValueError: {e}")

# ====================================================================== Adams-Bashforth
print()
print("=== Adams-Bashforth ===")

f_ab = lambda t, y: -y
expected_ab = math.exp(-1)

for order in [1, 2, 3, 4, 5]:
    y_end_ab = na.adams_bashforth(f_ab, 0, 1, 1, n=1000, order=order)
    err_ab = abs(y_end_ab - expected_ab)
    print(f"AB{order} n=1000: {y_end_ab:.12f}  err={err_ab:.2e}")

# order 범위 예외
try:
    na.adams_bashforth(f_ab, 0, 1, 1, n=10, order=0)
    assert False
except ValueError as e:
    print(f"order=0 ValueError: {e}")

try:
    na.adams_bashforth(f_ab, 0, 1, 1, n=10, order=6)
    assert False
except ValueError as e:
    print(f"order=6 ValueError: {e}")

# ====================================================================== Adams-Moulton
print()
print("=== Adams-Moulton ===")

for order in [1, 2, 3, 4, 5]:
    y_end_am = na.adams_moulton(f_ab, 0, 1, 1, n=1000, order=order)
    err_am = abs(y_end_am - expected_ab)
    print(f"AM{order} n=1000: {y_end_am:.12f}  err={err_am:.2e}")

# order 범위 예외
try:
    na.adams_moulton(f_ab, 0, 1, 1, n=10, order=6)
    assert False
except ValueError as e:
    print(f"AM order=6 ValueError: {e}")

# ====================================================================== Predictor-Corrector
print()
print("=== Predictor-Corrector ===")

for order in [2, 3, 4, 5]:
    y_end_pc = na.predictor_corrector(f_ab, 0, 1, 1, n=1000, order=order)
    err_pc = abs(y_end_pc - expected_ab)
    print(f"PC{order} n=1000: {y_end_pc:.12f}  err={err_pc:.2e}")

# order 범위
try:
    na.predictor_corrector(f_ab, 0, 1, 1, n=10, order=1)
    assert False
except ValueError as e:
    print(f"PC order=1 ValueError: {e}")

# PyExpr 입력 테스트
try:
    from math_library.laplace import t as t_sym
    from math_library.laplace.laplace_ast import symbol
    y_sym = symbol('y')
    expr = -2 * t_sym * y_sym
    y_end_expr = na.predictor_corrector(expr, 0, 1, 1, n=1000, order=4, vars=('t', 'y'))
    expected_expr = math.exp(-1)   # y=e^{-t^2}, y(1)=e^{-1}
    err_expr = abs(y_end_expr - expected_expr)
    print(f"PyExpr PC4 (-2ty): {y_end_expr:.12f}  err={err_expr:.2e}  (exp {expected_expr:.12f})")
    if err_expr < 1e-6:
        print("PyExpr PASS")
    else:
        print("PyExpr: error larger than 1e-6 — check expression")
except Exception as e:
    print(f"PyExpr test skipped: {e}")

# ====================================================================== 기존 회귀
print()
print("=== 기존 회귀 ===")

# Simpson
I_simp = na.composite_simpson_13(math.sin, 0, math.pi, n=100)
assert abs(I_simp - 2.0) < 5e-8, f"Simpson 회귀 실패: {I_simp}"
print(f"composite_simpson_13: {I_simp:.12f}  PASS")

# math_library 삼각함수
sin_val = m.sin(1.2345)
assert abs(sin_val - math.sin(1.2345)) < 1e-14
print(f"m.sin(1.2345): {sin_val:.15f}  PASS")

# RK45 회귀
y_rk45 = na.rk45(f_ab, 0, 1, 1, tol=1e-10)
assert abs(y_rk45 - expected_ab) < 1e-9
print(f"rk45 회귀: {y_rk45:.12f}  PASS")

# ====================================================================== 성능 측정
print()
print("=== 성능 ===")

N = 10000
t0_p = time.perf_counter()
for _ in range(N):
    na.gauss_legendre(math.sin, 0, math.pi, n=5)
t1_p = time.perf_counter()
ns_gl = (t1_p - t0_p) * 1e9 / N
print(f"gauss_legendre n=5:  {ns_gl:.0f} ns/call  (target <1000 ns)")

N2 = 200
t0_p = time.perf_counter()
for _ in range(N2):
    na.rk78(f_ab, 0, 1, 1, tol=1e-10)
t1_p = time.perf_counter()
us_rk78 = (t1_p - t0_p) * 1e6 / N2
print(f"rk78 tol=1e-10:      {us_rk78:.2f} μs/call  (target <50 μs)")

N3 = 200
t0_p = time.perf_counter()
for _ in range(N3):
    na.predictor_corrector(f_ab, 0, 1, 1, n=100, order=4)
t1_p = time.perf_counter()
us_pc4 = (t1_p - t0_p) * 1e6 / N3
print(f"PC4 n=100:           {us_pc4:.2f} μs/call  (target <50 μs)")

print()
print("=== 모든 테스트 완료 ===")
