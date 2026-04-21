"""
bench/laplace_phase_c.py — Phase C 검증 테스트
역 Laplace, poles/zeros, final/initial value, lambdify, 성능 벤치마크
"""
import sys
import os
# mathlib/src 를 경로에 추가
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

import cmath
import math
import time

from math_library.laplace import Laplace, t, s, Sin, Cos, Exp, const, symbol

L = Laplace()

# ================================================================== 1. Round-trip 테스트

def round_trip_test(f_orig, note, t_vals=None):
    """forward → inverse → 수치 비교"""
    if t_vals is None:
        t_vals = [0.1, 0.5, 1.0, 2.5]
    try:
        F = L.transform(f_orig)
        f_back = L.inverse(F)
        for tv in t_vals:
            got = f_back.evalf(t=tv)
            exp = f_orig.evalf(t=tv)
            rel_err = abs(got - exp) / max(abs(exp), 1e-10)
            if rel_err > 1e-5:
                return False, f"{note} @ t={tv}: got={got:.8f} exp={exp:.8f} relerr={rel_err:.2e}"
        return True, None
    except Exception as e:
        return False, f"{note}: EXCEPTION: {e}"


tests = [
    (Exp(-2*t),            "e^(-2t)"),
    (Sin(3*t),             "sin(3t)"),
    (Cos(2*t),             "cos(2t)"),
    (t*Exp(-t),            "t·e^(-t)"),
    (Exp(-t)*Sin(2*t),     "e^(-t)·sin(2t)"),
    (Exp(-2*t)*Cos(3*t),   "e^(-2t)·cos(3t)"),
    (t**2*Exp(-t),         "t²·e^(-t)"),
]

print("=" * 60)
print("Phase C — Round-trip 역변환 테스트")
print("=" * 60)
passed = 0
for f_expr, note in tests:
    ok, err = round_trip_test(f_expr, note)
    status = "[OK]  " if ok else "[FAIL]"
    if ok:
        passed += 1
        print(f"{status} {note}")
    else:
        print(f"{status} {note}: {err}")

print(f"\nRound-trip: {passed}/{len(tests)}")

# ================================================================== 2. Poles/Zeros 정확도

print("\n" + "=" * 60)
print("Poles / Zeros 테스트")
print("=" * 60)

tol_pz = 1e-8

# e^(-2t) → 1/(s+2) → pole = -2
F1 = L.transform(Exp(-2*t))
p1 = L.poles(F1)
ok1 = len(p1) == 1 and abs(p1[0] - (-2+0j)) < tol_pz
print(f"[{'OK' if ok1 else 'FAIL'}] poles of 1/(s+2): {[f'{p.real:.6f}{p.imag:+.6f}j' for p in p1]}  (expect [-2])")

# sin(3t) → 3/(s²+9) → poles = ±3j
F2 = L.transform(Sin(3*t))
p2 = sorted(L.poles(F2), key=lambda z: z.imag)
ok2 = len(p2) == 2 and abs(p2[0] - (-3j)) < tol_pz and abs(p2[1] - (3j)) < tol_pz
print(f"[{'OK' if ok2 else 'FAIL'}] poles of sin(3t)→3/(s²+9): {[f'{p.real:.4f}{p.imag:+.4f}j' for p in p2]}  (expect [±3j])")

# cos(2t) → s/(s²+4) → zeros = [0], poles = ±2j
F3 = L.transform(Cos(2*t))
z3 = L.zeros(F3)
p3 = sorted(L.poles(F3), key=lambda z: z.imag)
ok3z = len(z3) == 1 and abs(z3[0]) < tol_pz
ok3p = len(p3) == 2 and abs(p3[0] - (-2j)) < tol_pz and abs(p3[1] - (2j)) < tol_pz
print(f"[{'OK' if ok3z else 'FAIL'}] zeros of cos(2t)→s/(s²+4): {[f'{z.real:.4f}{z.imag:+.4f}j' for z in z3]}  (expect [0])")
print(f"[{'OK' if ok3p else 'FAIL'}] poles of cos(2t)→s/(s²+4): {[f'{p.real:.4f}{p.imag:+.4f}j' for p in p3]}  (expect [±2j])")

# ================================================================== 3. Final / Initial Value

print("\n" + "=" * 60)
print("Final / Initial Value 테스트")
print("=" * 60)

# f(t) = 1 - e^(-2t), F(s) = 2/(s(s+2))  → f(∞) = 1
f_step = const(1) - Exp(-2*t)
F_step = L.transform(f_step)
fv, fv_valid = L.final_value(F_step)
print(f"[{'OK' if abs(fv-1.0)<1e-4 else 'FAIL'}] final_value of (1-e^(-2t)): {fv:.6f}  (expect 1.0)  valid={fv_valid}")

# f(t) = e^(-t), F(s) = 1/(s+1) → f(∞) = 0
F_exp = L.transform(Exp(-t))
fv2, fv2_valid = L.final_value(F_exp)
print(f"[{'OK' if abs(fv2)<1e-4 else 'FAIL'}] final_value of e^(-t): {fv2:.6f}  (expect 0.0)  valid={fv2_valid}")

# initial value: f(t) = e^(-t), lim s->inf s/(s+1) = 1
iv, iv_valid = L.initial_value(F_exp)
print(f"[{'OK' if abs(iv-1.0)<1e-6 else 'FAIL'}] initial_value of 1/(s+1): {iv:.6f}  (expect 1.0)  valid={iv_valid}")

# initial value: f(t) = sin(3t) → 3/(s²+9), lim s->inf s*3/(s²+9) = 0
F_sin3 = L.transform(Sin(3*t))
iv2, iv2_valid = L.initial_value(F_sin3)
print(f"[{'OK' if abs(iv2)<1e-6 else 'FAIL'}] initial_value of sin(3t)→3/(s²+9): {iv2:.6f}  (expect 0.0)  valid={iv2_valid}")

# ================================================================== 4. Lambdify

print("\n" + "=" * 60)
print("Lambdify 테스트")
print("=" * 60)

F_sin2 = L.transform(Sin(2*t))  # 2/(s²+4)
f_callable_math = F_sin2.lambdify(['s'], backend='math')
f_callable_cmath = F_sin2.lambdify(['s'], backend='cmath')

v_math = f_callable_math(1.0)   # 2/(1+4) = 0.4
v_cmath = f_callable_cmath(1.0)

print(f"[{'OK' if abs(v_math - 0.4) < 1e-10 else 'FAIL'}] lambdify(math)  F(s=1) = {v_math:.10f}  (expect 0.4)")
print(f"[{'OK' if abs(v_cmath - 0.4) < 1e-10 else 'FAIL'}] lambdify(cmath) F(s=1) = {v_cmath:.10f}  (expect 0.4)")

# 복소수 입력 (극점 s=2j 근처 → ZeroDivisionError 예상)
try:
    v_cmath_imag = f_callable_cmath(0+2j)  # 분모 0 근처 (극점)
    print(f"[ -- ] lambdify(cmath) F(s=2j) = {v_cmath_imag}  (expect large/inf)")
except ZeroDivisionError:
    print("[OK]   lambdify(cmath) F(s=2j) → ZeroDivisionError (극점, 예상 동작)")

# t-영역 lambdify
f_back_test = L.inverse(L.transform(Exp(-2*t)))
g = f_back_test.lambdify(['t'], backend='math')
t_test = 1.0
g_val = g(t_test)
exp_val = math.exp(-2*t_test)
print(f"[{'OK' if abs(g_val - exp_val) < 1e-6 else 'FAIL'}] lambdify inverse(1/(s+2)) at t=1: {g_val:.8f}  (expect {exp_val:.8f})")

# ================================================================== 5. Partial Fractions AST

print("\n" + "=" * 60)
print("Partial Fractions AST 테스트")
print("=" * 60)

F_pf = L.transform(Exp(-2*t))  # 1/(s+2)
pf_expr = L.partial_fractions(F_pf)
print(f"PF of 1/(s+2): {pf_expr}")

F_pf2 = L.transform(Sin(3*t))  # 3/(s²+9)
pf_expr2 = L.partial_fractions(F_pf2)
print(f"PF of 3/(s²+9): {pf_expr2}")

# ================================================================== 6. 성능 벤치마크

print("\n" + "=" * 60)
print("성능 벤치마크")
print("=" * 60)

N_BENCH = 500

# partial_fractions (degree 2)
F_bench = L.transform(Sin(3*t))
t0 = time.perf_counter()
for _ in range(N_BENCH):
    L.partial_fractions(F_bench)
elapsed = (time.perf_counter() - t0) / N_BENCH * 1e6
print(f"partial_fractions (deg 2): {elapsed:.2f} μs  (target < 200 μs)")

# inverse (typical control TF)
F_inv = L.transform(Exp(-t)*Sin(2*t))
t0 = time.perf_counter()
for _ in range(N_BENCH):
    L.inverse(F_inv)
elapsed = (time.perf_counter() - t0) / N_BENCH * 1e6
print(f"inverse (e^-t sin 2t):     {elapsed:.2f} μs  (target < 500 μs)")

# lambdify 컴파일
F_lb = L.transform(Sin(2*t))
t0 = time.perf_counter()
for _ in range(N_BENCH):
    F_lb.lambdify(['s'], backend='cmath')
elapsed_compile = (time.perf_counter() - t0) / N_BENCH * 1e6
print(f"lambdify compile:          {elapsed_compile:.2f} μs  (target < 500 μs)")

# lambdify 호출
f_lb_fn = F_lb.lambdify(['s'], backend='cmath')
N_CALL = 100000
t0 = time.perf_counter()
for _ in range(N_CALL):
    f_lb_fn(1.0)
elapsed_call = (time.perf_counter() - t0) / N_CALL * 1e9
print(f"lambdify call:             {elapsed_call:.1f} ns  (expect < 500 ns)")

# ================================================================== 7. Phase B bug: PyExpr(1) 직접 호출

print("\n" + "=" * 60)
print("PyExpr(1) 직접 호출 TypeError 테스트")
print("=" * 60)

from math_library.laplace import PyExpr
try:
    bad = PyExpr(1)
    print("[FAIL] PyExpr(1) should raise TypeError but didn't")
except TypeError as e:
    print(f"[OK]   PyExpr(1) raises TypeError: {e}")
except Exception as e:
    print(f"[FAIL] PyExpr(1) raised unexpected: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("완료")
print("=" * 60)
