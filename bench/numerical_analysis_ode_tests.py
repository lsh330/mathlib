"""
numerical_analysis_ode_tests.py
Newton-Raphson + Runge-Kutta 7종 검증 테스트
"""
import math
import sys
import time

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None

from math_library import NumericalAnalysis
from math_library.laplace import t, Sin, Cos, Exp

na = NumericalAnalysis()

print("=" * 65)
print("NumericalAnalysis — Newton-Raphson + RK 7종 검증")
print("=" * 65)

# ============================================================= Newton-Raphson
print()
print("[1] newton_raphson")
print("-" * 45)

# 기본: f(x) = x^2 - 2, 근 = sqrt(2)
f_quad = lambda x: x**2 - 2
root = na.newton_raphson(f_quad, 1.5)
err_nr = abs(root - math.sqrt(2))
assert err_nr < 1e-12, f"newton sqrt(2) 실패: {root}"
print(f"  sqrt(2): {root}  오차={err_nr:.2e}  PASS")

# fprime 명시
fp_quad = lambda x: 2 * x
root2 = na.newton_raphson(f_quad, 1.5, fprime=fp_quad)
assert abs(root2 - math.sqrt(2)) < 1e-12
print(f"  fprime 명시: {root2}  PASS")

# return_info
root3, it3, res3 = na.newton_raphson(f_quad, 1.5, return_info=True)
assert abs(root3 - math.sqrt(2)) < 1e-12
print(f"  return_info: root={root3}  iter={it3}  residual={res3:.2e}  PASS")

# cos(x) = 0.5 → x = π/3
f_cos = lambda x: math.cos(x) - 0.5
root_cos = na.newton_raphson(f_cos, 1.0)
assert abs(root_cos - math.pi / 3) < 1e-10
print(f"  cos=0.5: {root_cos}  (pi/3={math.pi/3:.10f})  PASS")

# 수렴 실패 (작은 max_iter)
try:
    na.newton_raphson(f_quad, 100.0, tol=1e-30, max_iter=3)
    assert False, "RuntimeError 기대"
except RuntimeError as e:
    assert "did not converge" in str(e)
    print(f"  수렴 실패 OK: {e}")

# f'(x)=0 예외 (x=0에서 f(x)=x^2, f'(x)=2x=0)
try:
    na.newton_raphson(lambda x: x**2, 0.0)
    assert False, "ZeroDivisionError 기대"
except ZeroDivisionError as e:
    assert "f'(x)=0" in str(e)
    print(f"  f'=0 OK: {e}")

# tol <= 0
try:
    na.newton_raphson(f_quad, 1.5, tol=-1.0)
    assert False
except ValueError as e:
    assert "tol" in str(e)
    print(f"  tol<0 OK: {e}")

# max_iter <= 0
try:
    na.newton_raphson(f_quad, 1.5, max_iter=0)
    assert False
except ValueError as e:
    print(f"  max_iter=0 OK: {e}")

# PyExpr 입력 테스트
from math_library.laplace import symbol
x_sym = symbol('x')
expr_f = x_sym**2 - 2
root_expr = na.newton_raphson(expr_f, 1.5, var='x')
assert abs(root_expr - math.sqrt(2)) < 1e-10
print(f"  PyExpr newton: {root_expr}  PASS")

# ============================================================= secant_method
print()
print("[2] secant_method")
print("-" * 45)

# x^3 - x - 2 = 0 → x ≈ 1.5213797068045678
f_cub = lambda x: x**3 - x - 2
root_sec = na.secant_method(f_cub, 1.0, 2.0)
exact_sec = 1.521379706804568
err_sec = abs(root_sec - exact_sec)
assert err_sec < 1e-10, f"secant 실패: {root_sec}"
print(f"  x^3-x-2=0: {root_sec}  오차={err_sec:.2e}  PASS")

# return_info
root_si, it_si, res_si = na.secant_method(f_cub, 1.0, 2.0, return_info=True)
print(f"  return_info: root={root_si}  iter={it_si}  residual={res_si:.2e}  PASS")

# sin(x)=0 근처 (x=π 근방)
root_sin0 = na.secant_method(math.sin, 3.0, 3.5)
assert abs(root_sin0 - math.pi) < 1e-10
print(f"  sin=0 (pi): {root_sin0}  PASS")

# 수렴 실패
try:
    na.secant_method(f_quad, 0.0, 0.0001, tol=1e-30, max_iter=2)
except RuntimeError as e:
    print(f"  수렴 실패 OK: {str(e)[:60]}")

# ============================================================= Euler
print()
print("[3] euler")
print("-" * 45)

# dy/dt = -y, y(0)=1 → y(t) = e^(-t)
f_exp_decay = lambda t, y: -y
y_euler = na.euler(f_exp_decay, 0.0, 1.0, 1.0, n=1000)
exact_val = math.exp(-1.0)
err_euler = abs(y_euler - exact_val)
assert err_euler < 0.01, f"Euler n=1000 실패: {y_euler}"
print(f"  n=1000: y={y_euler}  exact={exact_val}  오차={err_euler:.4f}  PASS")

# 궤적
traj_e = na.euler(f_exp_decay, 0.0, 1.0, 1.0, n=10, return_trajectory=True)
assert len(traj_e) == 11
assert abs(traj_e[0][0] - 0.0) < 1e-14
assert abs(traj_e[-1][0] - 1.0) < 1e-14
print(f"  trajectory 점 수: {len(traj_e)}  PASS")

# n<1 예외
try:
    na.euler(f_exp_decay, 0.0, 1.0, 1.0, n=0)
    assert False
except ValueError as e:
    print(f"  n=0 OK: {e}")

# t0>=t_end 예외
try:
    na.euler(f_exp_decay, 1.0, 1.0, 0.0, n=10)
    assert False
except ValueError as e:
    print(f"  t0>=t_end OK: {e}")

# ============================================================= RK2
print()
print("[4] rk2")
print("-" * 45)

for m_name in ['midpoint', 'heun', 'ralston']:
    y_rk2 = na.rk2(f_exp_decay, 0.0, 1.0, 1.0, n=100, method=m_name)
    err_rk2 = abs(y_rk2 - exact_val)
    assert err_rk2 < 1e-5, f"rk2 {m_name} 실패: {y_rk2}"
    print(f"  {m_name}: y={y_rk2}  오차={err_rk2:.2e}  PASS")

# method 오류
try:
    na.rk2(f_exp_decay, 0.0, 1.0, 1.0, n=10, method='unknown')
    assert False
except ValueError as e:
    assert "method" in str(e)
    print(f"  invalid method OK: {e}")

# 궤적
traj_rk2 = na.rk2(f_exp_decay, 0.0, 1.0, 1.0, n=5, method='heun', return_trajectory=True)
assert len(traj_rk2) == 6
print(f"  trajectory 점 수: {len(traj_rk2)}  PASS")

# ============================================================= RK4
print()
print("[5] rk4")
print("-" * 45)

y_rk4 = na.rk4(f_exp_decay, 0.0, 1.0, 1.0, n=100)
err_rk4 = abs(y_rk4 - exact_val)
assert err_rk4 < 1e-10, f"rk4 n=100 실패: {y_rk4}"
print(f"  n=100: y={y_rk4}  오차={err_rk4:.2e}  PASS")

# 궤적
traj_rk4 = na.rk4(f_exp_decay, 0.0, 1.0, 1.0, n=10, return_trajectory=True)
assert len(traj_rk4) == 11
assert abs(traj_rk4[0][1] - 1.0) < 1e-14  # y(0)=1
print(f"  trajectory 점 수: {len(traj_rk4)}  PASS")

# 정확도: dy/dt = 3t^2, y(0)=0 → y(t) = t^3
f_poly = lambda t, y: 3.0 * t**2
y_poly = na.rk4(f_poly, 0.0, 0.0, 2.0, n=10)
assert abs(y_poly - 8.0) < 1e-12, f"rk4 polynomial 실패: {y_poly}"
print(f"  polynomial: y(2)={y_poly}  exact=8.0  오차={abs(y_poly-8.0):.2e}  PASS")

# ============================================================= RK45 (DOPRI5)
print()
print("[6] rk45 (Dormand-Prince)")
print("-" * 45)

y_rk45 = na.rk45(f_exp_decay, 0.0, 1.0, 1.0, tol=1e-10)
err_rk45 = abs(y_rk45 - exact_val)
assert err_rk45 < 1e-10, f"rk45 실패: {y_rk45}"
print(f"  dy/dt=-y: y={y_rk45}  오차={err_rk45:.2e}  PASS")

# 어려운 ODE: dy/dt = sin(t)*y, y(0)=1
# 해석해: y = exp(1 - cos(t))
f_hard = lambda t, y: math.sin(t) * y
y_hard = na.rk45(f_hard, 0.0, 1.0, 2 * math.pi, tol=1e-10)
expected_hard = math.exp(1.0 - math.cos(2 * math.pi))  # = exp(0) = 1.0
err_hard = abs(y_hard - expected_hard)
assert err_hard < 1e-8, f"rk45 hard ODE 실패: {y_hard}"
print(f"  sin(t)*y: y={y_hard}  exact=1.0  오차={err_hard:.2e}  PASS")

# 궤적 + step 수 확인
traj_45 = na.rk45(f_exp_decay, 0.0, 1.0, 1.0, tol=1e-8, return_trajectory=True)
assert len(traj_45) >= 2
assert abs(traj_45[0][0] - 0.0) < 1e-14
assert abs(traj_45[-1][0] - 1.0) < 1e-12
print(f"  trajectory steps: {len(traj_45)}  PASS")

# max_steps 초과
try:
    na.rk45(lambda t, y: -1000*y, 0.0, 1.0, 1.0, tol=1e-15, max_steps=10)
except RuntimeError as e:
    print(f"  max_steps 초과 OK: {str(e)[:60]}")

# ============================================================= Fehlberg RKF45
print()
print("[7] rk_fehlberg (Fehlberg RKF45)")
print("-" * 45)

y_rkf = na.rk_fehlberg(f_exp_decay, 0.0, 1.0, 1.0, tol=1e-10)
err_rkf = abs(y_rkf - exact_val)
assert err_rkf < 1e-9, f"rk_fehlberg 실패: {y_rkf}"
print(f"  dy/dt=-y: y={y_rkf}  오차={err_rkf:.2e}  PASS")

# 더 어려운 ODE
y_rkf_hard = na.rk_fehlberg(f_hard, 0.0, 1.0, 2 * math.pi, tol=1e-10)
err_rkf_hard = abs(y_rkf_hard - expected_hard)
assert err_rkf_hard < 1e-8, f"rk_fehlberg hard 실패: {y_rkf_hard}"
print(f"  sin(t)*y: y={y_rkf_hard}  exact=1.0  오차={err_rkf_hard:.2e}  PASS")

traj_rkf = na.rk_fehlberg(f_exp_decay, 0.0, 1.0, 1.0, tol=1e-8, return_trajectory=True)
print(f"  trajectory steps: {len(traj_rkf)}  PASS")

# ============================================================= PyExpr ODE
print()
print("[8] PyExpr ODE (2변수 lambdify)")
print("-" * 45)

y_sym = symbol('y')
# dy/dt = -2*t*y → y = exp(-t^2), y(0)=1
expr_ode = -2 * t * y_sym
y_pyexpr = na.rk4(expr_ode, 0.0, 1.0, 1.0, n=200, vars=('t', 'y'))
expected_pyexpr = math.exp(-1.0)
err_pyexpr = abs(y_pyexpr - expected_pyexpr)
assert err_pyexpr < 1e-8, f"PyExpr ODE 실패: {y_pyexpr}"
print(f"  rk4 PyExpr dy/dt=-2ty: y={y_pyexpr}  exact={expected_pyexpr}  오차={err_pyexpr:.2e}  PASS")

# euler + PyExpr
y_euler_expr = na.euler(expr_ode, 0.0, 1.0, 1.0, n=5000, vars=('t', 'y'))
err_euler_expr = abs(y_euler_expr - expected_pyexpr)
assert err_euler_expr < 0.01
print(f"  euler PyExpr: y={y_euler_expr}  오차={err_euler_expr:.2e}  PASS")

# rk2 + PyExpr
y_rk2_expr = na.rk2(expr_ode, 0.0, 1.0, 1.0, n=100, vars=('t', 'y'))
err_rk2_expr = abs(y_rk2_expr - expected_pyexpr)
assert err_rk2_expr < 1e-5
print(f"  rk2 PyExpr: y={y_rk2_expr}  오차={err_rk2_expr:.2e}  PASS")

# ============================================================= Differentiation 간접 검증
print()
print("[9] Differentiation 간접 검증 (fprime=None 기본)")
print("-" * 45)

import math_library as m

# sin(x) = 0.5 → x = π/6
root_sin = na.newton_raphson(lambda x: m.sin(x) - 0.5, 0.5)
assert abs(root_sin - math.pi / 6) < 1e-10
print(f"  sin(x)=0.5: {root_sin}  (pi/6={math.pi/6:.10f})  PASS")

# exp(x) = 2 → x = ln(2)
root_exp2 = na.newton_raphson(lambda x: m.exp(x) - 2, 0.5)
assert abs(root_exp2 - math.log(2)) < 1e-10
print(f"  exp(x)=2: {root_exp2}  (ln2={math.log(2):.10f})  PASS")

# ln(x) = 0 → x = 1
root_ln = na.newton_raphson(lambda x: m.ln(x), 0.5)
assert abs(root_ln - 1.0) < 1e-10
print(f"  ln(x)=0: {root_ln}  PASS")

# ============================================================= 기존 Simpson 회귀
print()
print("[10] 기존 Simpson 회귀 테스트")
print("-" * 45)

I_13 = na.composite_simpson_13(math.sin, 0, math.pi, n=100)
assert abs(I_13 - 2.0) < 1e-7
print(f"  composite_simpson_13: {I_13}  PASS")

I_adp = na.adaptive_simpson(math.sin, 0, math.pi, tol=1e-12)
assert abs(I_adp - 2.0) < 1e-11
print(f"  adaptive_simpson: {I_adp}  PASS")

I_rom = na.romberg(math.sin, 0, math.pi, depth=6)
assert abs(I_rom - 2.0) < 1e-12
print(f"  romberg depth=6: {I_rom}  PASS")

# mathlib core 회귀
assert abs(m.sin(1.2345) - math.sin(1.2345)) < 1e-14
assert abs(m.cos(0.9876) - math.cos(0.9876)) < 1e-14
assert abs(m.exp(1.0) - math.exp(1.0)) < 1e-14
print(f"  mathlib sin/cos/exp 회귀  PASS")

# ============================================================= 성능
print()
print("[11] 성능 측정")
print("-" * 45)

N_nr = 1000
t0p = time.perf_counter()
for _ in range(N_nr):
    na.newton_raphson(f_quad, 1.5)
t1p = time.perf_counter()
time_nr = (t1p - t0p) * 1e6 / N_nr
print(f"  newton_raphson: {time_nr:.2f} μs/call  (목표 < 5 μs)")

N_rk4 = 100
t0p = time.perf_counter()
for _ in range(N_rk4):
    na.rk4(f_exp_decay, 0.0, 1.0, 1.0, n=1000)
t1p = time.perf_counter()
time_rk4 = (t1p - t0p) * 1e6 / N_rk4
print(f"  rk4 n=1000: {time_rk4:.2f} μs/call  (목표 < 100 μs)")

N_rk45 = 100
t0p = time.perf_counter()
for _ in range(N_rk45):
    na.rk45(f_exp_decay, 0.0, 1.0, 1.0, tol=1e-8)
t1p = time.perf_counter()
time_rk45 = (t1p - t0p) * 1e6 / N_rk45
print(f"  rk45 tol=1e-8: {time_rk45:.2f} μs/call  (목표 < 500 μs)")

N_rkf = 100
t0p = time.perf_counter()
for _ in range(N_rkf):
    na.rk_fehlberg(f_exp_decay, 0.0, 1.0, 1.0, tol=1e-8)
t1p = time.perf_counter()
time_rkf = (t1p - t0p) * 1e6 / N_rkf
print(f"  rk_fehlberg tol=1e-8: {time_rkf:.2f} μs/call")

N_sec = 1000
t0p = time.perf_counter()
for _ in range(N_sec):
    na.secant_method(f_cub, 1.0, 2.0)
t1p = time.perf_counter()
time_sec = (t1p - t0p) * 1e6 / N_sec
print(f"  secant_method: {time_sec:.2f} μs/call")

# ============================================================= 정확도 요약
print()
print("=" * 65)
print("정확도 요약 (dy/dt = -y, y(0)=1, t=0→1, exact=e^(-1))")
print("=" * 65)

exact_v = math.exp(-1.0)
results = [
    ("euler n=1000",        na.euler(f_exp_decay, 0.0, 1.0, 1.0, n=1000)),
    ("rk2-midpoint n=100",  na.rk2(f_exp_decay, 0.0, 1.0, 1.0, n=100, method='midpoint')),
    ("rk2-heun n=100",      na.rk2(f_exp_decay, 0.0, 1.0, 1.0, n=100, method='heun')),
    ("rk2-ralston n=100",   na.rk2(f_exp_decay, 0.0, 1.0, 1.0, n=100, method='ralston')),
    ("rk4 n=100",           na.rk4(f_exp_decay, 0.0, 1.0, 1.0, n=100)),
    ("rk45 tol=1e-10",      na.rk45(f_exp_decay, 0.0, 1.0, 1.0, tol=1e-10)),
    ("rk_fehlberg tol=1e-10", na.rk_fehlberg(f_exp_decay, 0.0, 1.0, 1.0, tol=1e-10)),
]

for name, val in results:
    err = abs(val - exact_v)
    print(f"  {name:<28} y={val:.15f}  오차={err:.2e}")

print()
print("=" * 65)
print("모든 테스트 PASS")
print("=" * 65)
