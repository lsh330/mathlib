"""
numerical_analysis_tests.py — NumericalAnalysis 클래스 8종 Simpson 검증 테스트
"""
import math
import sys
import time

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None

from math_library.numerical_analysis import NumericalAnalysis
from math_library.laplace import Sin, Cos, t

na = NumericalAnalysis()

PI = math.pi
print("=" * 60)
print("NumericalAnalysis — Simpson 적분법 8종 검증")
print("=" * 60)

# -------------------------------------------------------------- 1. simpson_13
I = na.simpson_13(math.sin, 0, PI)
err_13 = abs(I - 2.0)
# sin(0~pi) 단순 Simpson 1/3: I = pi/2/3*(0+4*1+0) = 2pi/3 ≈ 2.094, 오차 ≈ 0.094
assert err_13 < 0.1, f"simpson_13 failed: {I}"
print(f"[1] simpson_13:  I={I:.15f}  오차={err_13:.2e}  PASS")

I, est_err = na.simpson_13(math.sin, 0, PI, return_error=True)
print(f"    return_error: I={I:.15f}, est_err={est_err:.2e}")

# -------------------------------------------------------------- 2. simpson_38
I = na.simpson_38(lambda x: x**3, 0, 1)
err_38 = abs(I - 0.25)
assert err_38 < 1e-14, f"simpson_38 failed: {I}"
print(f"[2] simpson_38:  I={I:.15f}  오차={err_38:.2e}  PASS")

I, est_err = na.simpson_38(lambda x: x**3, 0, 1, return_error=True)
print(f"    return_error: I={I:.15f}, est_err={est_err:.2e}")

# -------------------------------------------------------------- 3. composite_simpson_13
I = na.composite_simpson_13(math.sin, 0, PI, n=100)
err_c13 = abs(I - 2.0)
# sin(0~pi) n=100: O(h^4) 이론 오차 ≈ 1e-8
assert err_c13 < 2e-8, f"composite_simpson_13 failed: {I}"
print(f"[3] composite_13 n=100: I={I:.15f}  오차={err_c13:.2e}  PASS")

I, est_err = na.composite_simpson_13(math.sin, 0, PI, n=100, return_error=True)
print(f"    return_error: I={I:.15f}, est_err={est_err:.2e}")

# 대규모 n — Kahan 정확도
I_large = na.composite_simpson_13(math.sin, 0, PI, n=10000)
err_large = abs(I_large - 2.0)
print(f"    n=10000: I={I_large:.15f}  오차={err_large:.2e}")
assert err_large < 1e-12, f"composite_13 n=10000 failed: {I_large}"
print(f"    n=10000 Kahan 정확도: PASS")

# -------------------------------------------------------------- 4. composite_simpson_38
I = na.composite_simpson_38(math.sin, 0, PI, n=99)
err_c38 = abs(I - 2.0)
# sin(0~pi) n=99: O(h^4) 이론 오차 ≈ 2.5e-8
assert err_c38 < 1e-7, f"composite_simpson_38 failed: {I}"
print(f"[4] composite_38 n=99:  I={I:.15f}  오차={err_c38:.2e}  PASS")

I, est_err = na.composite_simpson_38(math.sin, 0, PI, n=99, return_error=True)
print(f"    return_error: I={I:.15f}, est_err={est_err:.2e}")

# -------------------------------------------------------------- 5. adaptive_simpson
I = na.adaptive_simpson(math.sin, 0, PI, tol=1e-12)
err_adp = abs(I - 2.0)
assert err_adp < 1e-11, f"adaptive_simpson failed: {I}"
print(f"[5] adaptive_simpson:   I={I:.15f}  오차={err_adp:.2e}  PASS")

I, est_err = na.adaptive_simpson(math.sin, 0, PI, tol=1e-12, return_error=True)
print(f"    return_error: I={I:.15f}, est_err={est_err:.2e}")

# -------------------------------------------------------------- 6. mixed_simpson
I_odd = na.mixed_simpson(math.sin, 0, PI, n=7)
err_odd = abs(I_odd - 2.0)
assert err_odd < 0.05, f"mixed_simpson n=7 (odd) failed: {I_odd}"
print(f"[6] mixed_simpson n=7(홀수): I={I_odd:.15f}  오차={err_odd:.2e}  PASS")

I_even = na.mixed_simpson(math.sin, 0, PI, n=8)
err_even = abs(I_even - 2.0)
# sin(0~pi) n=8: O(h^4) 이론 오차 ≈ 9.3e-5, 실제 2.7e-4 (저차수 n)
assert err_even < 1e-3, f"mixed_simpson n=8 (even) failed: {I_even}"
print(f"    mixed_simpson n=8(짝수): I={I_even:.15f}  오차={err_even:.2e}  PASS")

I_n3 = na.mixed_simpson(math.sin, 0, PI, n=3)
err_n3 = abs(I_n3 - 2.0)
print(f"    mixed_simpson n=3(홀수): I={I_n3:.15f}  오차={err_n3:.2e}")

# -------------------------------------------------------------- 7. simpson_irregular
x_pts = [0.0, 0.5, 1.0, 1.5, 2.0]
y_pts = [x**2 for x in x_pts]   # ∫_0^2 x^2 dx = 8/3
I = na.simpson_irregular(x_pts, y_pts)
err_irr = abs(I - 8.0 / 3.0)
assert err_irr < 1e-10, f"simpson_irregular failed: {I}, expected {8.0/3.0}"
print(f"[7] simpson_irregular:  I={I:.15f}  오차={err_irr:.2e}  PASS")

I, est_err = na.simpson_irregular(x_pts, y_pts, return_error=True)
print(f"    return_error: I={I:.15f}, est_err={est_err}")

# 불균등 간격 테스트
x_uneven = [0.0, 0.3, 0.7, 1.2, 2.0]
y_uneven = [math.sin(x) for x in x_uneven]
I_uneven = na.simpson_irregular(x_uneven, y_uneven)
exact_uneven = 1.0 - math.cos(2.0)  # ≈ 1.4161...
err_uneven = abs(I_uneven - exact_uneven)
print(f"    불균등 간격: I={I_uneven:.8f}  exact={exact_uneven:.8f}  오차={err_uneven:.2e}")

# -------------------------------------------------------------- 8. romberg
I = na.romberg(math.sin, 0, PI, depth=6)
err_rom = abs(I - 2.0)
assert err_rom < 1e-12, f"romberg depth=6 failed: {I}"
print(f"[8] romberg depth=6:    I={I:.15f}  오차={err_rom:.2e}  PASS")

I, est_err = na.romberg(math.sin, 0, PI, depth=6, return_error=True)
print(f"    return_error: I={I:.15f}, est_err={est_err:.2e}")

# 다양한 깊이 테스트
for d in [1, 3, 5, 8]:
    I_d = na.romberg(math.sin, 0, PI, depth=d)
    print(f"    depth={d}: I={I_d:.15f}  오차={abs(I_d-2.0):.2e}")

print()
print("=" * 60)
print("PyExpr 입력 테스트")
print("=" * 60)

# PyExpr: ∫_0^π sin(2t)cos(t) dt = 4/3
f_expr = Sin(2*t) * Cos(t)
I_expr = na.composite_simpson_13(f_expr, 0, PI, n=100, var='t')
exact_expr = 4.0 / 3.0
err_expr = abs(I_expr - exact_expr)
# n=100: O(h^4) 오차 ≈ 1.5e-7 (sin(2t)cos(t) f^(4) 계수 영향)
assert err_expr < 1e-6, f"PyExpr composite_13 failed: {I_expr}"
print(f"composite_13(sin(2t)*cos(t)): I={I_expr:.15f}  exact={exact_expr:.15f}  오차={err_expr:.2e}  PASS")

I_adp = na.adaptive_simpson(f_expr, 0, PI, tol=1e-10, var='t')
err_adp_expr = abs(I_adp - exact_expr)
assert err_adp_expr < 1e-9, f"PyExpr adaptive failed: {I_adp}"
print(f"adaptive_simpson(sin(2t)*cos(t)): I={I_adp:.15f}  오차={err_adp_expr:.2e}  PASS")

I_rom = na.romberg(f_expr, 0, PI, depth=5, var='t')
err_rom_expr = abs(I_rom - exact_expr)
print(f"romberg(sin(2t)*cos(t)): I={I_rom:.15f}  오차={err_rom_expr:.2e}")

print()
print("=" * 60)
print("예외 처리 테스트")
print("=" * 60)

# n=7 (홀수) → composite_13 거부
try:
    na.composite_simpson_13(math.sin, 0, PI, n=7)
    assert False, "expected ValueError"
except ValueError as e:
    assert "even" in str(e).lower(), f"메시지에 'even' 없음: {e}"
    print(f"[E1] composite_13 n=7 거부 PASS: {e}")

# n=10 (3의 배수 아님) → composite_38 거부
try:
    na.composite_simpson_38(math.sin, 0, PI, n=10)
    assert False
except ValueError as e:
    assert "3" in str(e), f"메시지에 '3' 없음: {e}"
    print(f"[E2] composite_38 n=10 거부 PASS: {e}")

# 단조성 위반
try:
    na.simpson_irregular([0, 1, 0.5], [0, 1, 0.5])
    assert False
except ValueError as e:
    assert "monoton" in str(e).lower(), f"메시지에 'monoton' 없음: {e}"
    print(f"[E3] 단조성 위반 거부 PASS: {e}")

# str 입력 (callable 아님)
try:
    na.simpson_13("not callable", 0, 1)
    assert False
except TypeError as e:
    print(f"[E4] str 거부 PASS: {e}")

# a >= b
try:
    na.simpson_13(math.sin, PI, 0)
    assert False
except ValueError as e:
    assert "less than" in str(e).lower(), f"메시지 확인 실패: {e}"
    print(f"[E5] a >= b 거부 PASS: {e}")

# n < 2
try:
    na.composite_simpson_13(math.sin, 0, PI, n=1)
    assert False
except ValueError as e:
    print(f"[E6] n=1 거부 PASS: {e}")

# tol <= 0
try:
    na.adaptive_simpson(math.sin, 0, PI, tol=-1e-10)
    assert False
except ValueError as e:
    print(f"[E7] tol<0 거부 PASS: {e}")

# depth < 1
try:
    na.romberg(math.sin, 0, PI, depth=0)
    assert False
except ValueError as e:
    print(f"[E8] depth=0 거부 PASS: {e}")

# NaN 포함 불규칙
try:
    na.simpson_irregular([0.0, float('nan'), 1.0], [0.0, 0.5, 1.0])
    assert False
except ValueError as e:
    print(f"[E9] NaN 거부 PASS: {e}")

# 점 수 불일치
try:
    na.simpson_irregular([0.0, 0.5, 1.0], [0.0, 0.5])
    assert False
except ValueError as e:
    print(f"[E10] 점수 불일치 거부 PASS: {e}")

print()
print("=" * 60)
print("성능 벤치마크")
print("=" * 60)

# simpson_13
N_BENCH = 10_000
t0 = time.perf_counter()
for _ in range(N_BENCH):
    na.simpson_13(math.sin, 0, PI)
t1 = time.perf_counter()
time_13 = (t1 - t0) * 1e6 / N_BENCH
print(f"simpson_13:            {time_13:.2f} µs/call  (목표 < 500 ns)")

# composite_simpson_13 n=100
t0 = time.perf_counter()
for _ in range(1000):
    na.composite_simpson_13(math.sin, 0, PI, n=100)
t1 = time.perf_counter()
time_c13 = (t1 - t0) * 1e6 / 1000
print(f"composite_13 n=100:    {time_c13:.2f} µs/call  (목표 < 10 µs)")

# adaptive_simpson
t0 = time.perf_counter()
for _ in range(1000):
    na.adaptive_simpson(math.sin, 0, PI, tol=1e-10)
t1 = time.perf_counter()
time_adp = (t1 - t0) * 1e6 / 1000
print(f"adaptive_simpson 1e-10: {time_adp:.2f} µs/call")

# romberg depth=6
t0 = time.perf_counter()
for _ in range(1000):
    na.romberg(math.sin, 0, PI, depth=6)
t1 = time.perf_counter()
time_rom = (t1 - t0) * 1e6 / 1000
print(f"romberg depth=6:        {time_rom:.2f} µs/call")

# composite n=10000 Kahan
t0 = time.perf_counter()
for _ in range(100):
    na.composite_simpson_13(math.sin, 0, PI, n=10000)
t1 = time.perf_counter()
time_kahan = (t1 - t0) * 1e6 / 100
print(f"composite_13 n=10000:   {time_kahan:.2f} µs/call")

print()
print("=" * 60)
print("sin 0~π 적분 2.0 기준 상대오차 요약")
print("=" * 60)

methods = [
    ("simpson_13",           na.simpson_13(math.sin, 0, PI)),
    ("simpson_38",           na.simpson_38(math.sin, 0, PI)),
    ("composite_13 n=100",   na.composite_simpson_13(math.sin, 0, PI, n=100)),
    ("composite_38 n=99",    na.composite_simpson_38(math.sin, 0, PI, n=99)),
    ("adaptive tol=1e-12",   na.adaptive_simpson(math.sin, 0, PI, tol=1e-12)),
    ("mixed n=8",            na.mixed_simpson(math.sin, 0, PI, n=8)),
    ("romberg depth=6",      na.romberg(math.sin, 0, PI, depth=6)),
]
for name, val in methods:
    rel_err = abs(val - 2.0) / 2.0
    print(f"  {name:<24} val={val:.15f}  rel_err={rel_err:.2e}")

print()
print("=" * 60)
print("기존 mathlib 회귀 테스트")
print("=" * 60)

import math_library as m
assert abs(m.sin(1.2345) - math.sin(1.2345)) < 1e-14, "sin 회귀 실패"
assert abs(m.cos(0.9876) - math.cos(0.9876)) < 1e-14, "cos 회귀 실패"
assert abs(m.exp(1.0) - math.exp(1.0)) < 1e-14, "exp 회귀 실패"
assert abs(m.ln(math.e) - 1.0) < 1e-14, "ln 회귀 실패"
print("mathlib 회귀: sin, cos, exp, ln PASS")

print()
print("=" * 60)
print("전체 테스트 완료 — 8개 Simpson 변형 ALL PASS")
print("=" * 60)
