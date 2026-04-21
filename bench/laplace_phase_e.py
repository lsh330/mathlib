"""
laplace_phase_e.py — Phase E 검증 테스트
Heaviside/Dirac 변환, Non-rational inverse, collect, feedback simplify
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math

from math_library.laplace import (
    Laplace, t, s, Sin, Cos, Exp, const, symbol,
    Heaviside, Dirac, H
)

L = Laplace()

# ================================================================== 1. Heaviside 변환
print("=== 1. Heaviside L{u(t)} ===")
F1 = L.transform(Heaviside(t))          # expect: 1/s
val = F1.evalf(s=2)
exp_val = 1.0 / 2.0
print(f"  L{{u(t)}} @ s=2: got={val:.15g}  exp={exp_val:.15g}")
assert abs(val - exp_val) < 1e-10, f"FAIL: got {val} exp {exp_val}"

print("=== 1b. L{u(t-a)} ===")
F2 = L.transform(Heaviside(t - const(2)))  # expect: e^{-2s}/s
for sv in [1.0, 2.0, 3.0]:
    got = F2.evalf(s=sv)
    exp = math.exp(-2*sv) / sv
    print(f"  u(t-2) @ s={sv}: got={got:.15g}  exp={exp:.15g}  err={abs(got-exp):.2e}")
    assert abs(got - exp) < 1e-10, f"FAIL u(t-2) @ s={sv}: got={got} exp={exp}"
print("  L{u(t-a)} PASS")

# ================================================================== 2. Dirac delta 변환
print("=== 2. Dirac L{δ(t)} ===")
F3 = L.transform(Dirac(t))              # expect: 1
got = F3.evalf(s=3.0)
print(f"  L{{δ(t)}} @ s=3: got={got:.15g}  exp=1.0")
assert abs(got - 1.0) < 1e-10, f"FAIL: got {got}"

print("=== 2b. L{δ(t-1)} ===")
F4 = L.transform(Dirac(t - const(1)))   # expect: e^{-s}
for sv in [1.0, 2.0]:
    got = F4.evalf(s=sv)
    exp = math.exp(-sv)
    print(f"  δ(t-1) @ s={sv}: got={got:.15g}  exp={exp:.15g}")
    assert abs(got - exp) < 1e-10, f"FAIL δ(t-1) @ s={sv}"
print("  L{δ(t-a)} PASS")

# ================================================================== 3. t-shift 복합
print("=== 3. t-shift: u(t-1) * exp(-(t-1)) ===")
# f(t) = u(t-1) * e^{-(t-1)}  →  L = e^{-s} / (s+1)
a_shift = const(1)
t_minus_1 = t - a_shift
f_ts = Heaviside(t_minus_1) * Exp(-t_minus_1)
F5 = L.transform(f_ts)
for sv in [2.0, 3.0]:
    got = F5.evalf(s=sv)
    exp = math.exp(-sv) / (sv + 1)
    print(f"  @ s={sv}: got={got:.15g}  exp={exp:.15g}  err={abs(got-exp):.2e}")
    assert abs(got - exp) < 1e-8, f"FAIL t-shift @ s={sv}"
print("  t-shift PASS")

# ================================================================== 4. Non-rational inverse
print("=== 4. Non-rational inverse ===")
# e^{-2s} / (s+1)  →  u(t-2) * e^{-(t-2)}
# G(s) = 1/(s+1) = L{e^{-t}}
G = L.transform(Exp(-t))   # 1/(s+1)

# e^{-2s} * G(s) 구성
s_sym = symbol('s')
exp_2s = Exp(-2 * s_sym)
F_shift = exp_2s * G

f_back = L.inverse(F_shift)
print(f"  inverse result AST: {f_back}")

for tv, expected_val in [(0.5, 0.0), (2.5, math.exp(-(2.5-2))), (3.5, math.exp(-(3.5-2)))]:
    got = f_back.evalf(t=tv)
    print(f"  @ t={tv}: got={got:.15g}  exp={expected_val:.15g}  err={abs(got-expected_val):.2e}")
    assert abs(got - expected_val) < 1e-6, f"FAIL non-rational inv @ t={tv}: got={got} exp={expected_val}"
print("  non-rational inverse PASS")

# ================================================================== 5. collect
print("=== 5. collect ===")
# 3*s^2 + 2*s^2 + 5*s + 7  →  5*s^2 + 5*s + 7
e_expr = 3*s**2 + 2*s**2 + 5*s + 7
c_expr = e_expr.collect(s)
print(f"  before: {e_expr}")
print(f"  after:  {c_expr}")
for sv in [1.0, 2.0, 3.0]:
    got = c_expr.evalf(s=sv)
    exp = 5.0*sv**2 + 5.0*sv + 7.0
    print(f"  @ s={sv}: got={got:.15g}  exp={exp:.15g}")
    assert abs(got - exp) < 1e-10, f"FAIL collect @ s={sv}"
print("  collect PASS")

# ================================================================== 6. feedback 자동 simplify
print("=== 6. feedback simplify ===")
# G = 1/(s+1), feedback → 1/(s+2)
G6 = L.transform(Exp(-t))   # 1/(s+1)
fb = G6.feedback()
print(f"  G.feedback() = {fb}")
for sv in [1.0, 2.0, 5.0]:
    got = fb.evalf(s=sv)
    exp = 1.0 / (sv + 2.0)
    print(f"  @ s={sv}: got={got:.15g}  exp={exp:.15g}  err={abs(got-exp):.2e}")
    assert abs(got - exp) < 1e-10, f"FAIL feedback @ s={sv}: got={got} exp={exp}"
print("  feedback simplify PASS")

# ================================================================== 7. Heaviside alias H
print("=== 7. H alias ===")
F_h = L.transform(H(t))
assert abs(F_h.evalf(s=2.0) - 0.5) < 1e-10
print("  H alias PASS")

# ================================================================== 8. 기존 회귀
print("=== 8. 기존 Laplace 회귀 ===")
import math_library as ml

# sin 회귀
assert abs(ml.sin(1.2345) - math.sin(1.2345)) < 1e-14, "sin regression FAIL"
# forward transform 회귀
F_sin = L.transform(Sin(2*t))
for sv in [1.0, 2.0]:
    got = F_sin.evalf(s=sv)
    exp = 2.0 / (sv**2 + 4.0)
    assert abs(got - exp) < 1e-12, f"L{{sin(2t)}} regression FAIL @ s={sv}"
# inverse 회귀
f_inv = L.inverse(L.transform(Exp(-t)))  # 1/(s+1) → e^{-t}
for tv in [0.5, 1.0, 2.0]:
    got = f_inv.evalf(t=tv)
    exp = math.exp(-tv)
    assert abs(got - exp) < 1e-12, f"inverse regression FAIL @ t={tv}"
print("  mathlib 회귀 PASS")

print("\n========== Phase E 전체 PASS ==========")
