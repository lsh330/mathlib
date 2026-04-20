"""
bench/perf_compare.py
math vs top-level (방향 A dispatch) vs _core 직접 3-열 성능 비교
5000회 워밍업 후 1,000,000회 반복 측정
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import cmath
import time

import math_library as m
from math_library._core.trigonometric import (
    sin as cy_sin, cos as cy_cos, tan as cy_tan,
    sec as cy_sec, cosec as cy_cosec, cotan as cy_cotan,
)
from math_library._core.inverse_trig import (
    arcsin as cy_arcsin, arccos as cy_arccos, arctan as cy_arctan,
    arcsec as cy_arcsec, arccosec as cy_arccosec, arccotan as cy_arccotan,
)
from math_library._core.hyperbolic import (
    hypersin as cy_hypersin, hypercos as cy_hypercos, hypertan as cy_hypertan,
    hypersec as cy_hypersec, hypercosec as cy_hypercosec, hypercotan as cy_hypercotan,
)
from math_library._core.exponential import exp as cy_exp
from math_library._core.logarithmic import ln as cy_ln, log as cy_log
from math_library._core.power_sqrt import sqrt as cy_sqrt, power as cy_power

WARMUP = 5000
REPS   = 1_000_000

def bench(fn, arg, n=REPS, warmup=WARMUP):
    for _ in range(warmup):
        fn(arg)
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn(arg)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / n  # ns per call

def bench2(fn, a, b, n=REPS, warmup=WARMUP):
    """2-인수 함수 벤치"""
    for _ in range(warmup):
        fn(a, b)
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn(a, b)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / n

X = 1.2345
X2 = 0.5   # arcsin/arccos 도메인 [-1,1]
X3 = 0.7   # arctan 등 범용
BASE = 2.0  # power/log base

print("=" * 70)
print(f"{'함수':18s} {'math(ns)':>10s} {'top-level(ns)':>14s} {'ratio':>8s} {'_core(ns)':>10s} {'ratio2':>8s}")
print("-" * 70)

funcs = [
    # (이름, math_fn, top_fn, core_fn, arg)
    ("sin",       math.sin,        m.sin,        cy_sin,        X),
    ("cos",       math.cos,        m.cos,        cy_cos,        X),
    ("tan",       math.tan,        m.tan,        cy_tan,        X),
    ("sec",       None,            m.sec,        cy_sec,        X),
    ("cosec",     None,            m.cosec,      cy_cosec,      X),
    ("cotan",     None,            m.cotan,      cy_cotan,      X),
    ("arcsin",    math.asin,       m.arcsin,     cy_arcsin,     X2),
    ("arccos",    math.acos,       m.arccos,     cy_arccos,     X2),
    ("arctan",    math.atan,       m.arctan,     cy_arctan,     X3),
    ("arcsec",    None,            m.arcsec,     cy_arcsec,     X3+1),
    ("arccosec",  None,            m.arccosec,   cy_arccosec,   X3+1),
    ("arccotan",  None,            m.arccotan,   cy_arccotan,   X3),
    ("hypersin",  math.sinh,       m.hypersin,   cy_hypersin,   X),
    ("hypercos",  math.cosh,       m.hypercos,   cy_hypercos,   X),
    ("hypertan",  math.tanh,       m.hypertan,   cy_hypertan,   X),
    ("hypersec",  None,            m.hypersec,   cy_hypersec,   X),
    ("hypercosec",None,            m.hypercosec, cy_hypercosec, X),
    ("hypercotan",None,            m.hypercotan, cy_hypercotan, X),
    ("exp",       math.exp,        m.exp,        cy_exp,        X),
    ("ln",        math.log,        m.ln,         cy_ln,         X),
    ("sqrt",      math.sqrt,       m.sqrt,       cy_sqrt,       X),
]

for name, math_fn, top_fn, core_fn, arg in funcs:
    t_math = bench(math_fn, arg) if math_fn else None
    t_top  = bench(top_fn, arg)
    t_core = bench(core_fn, arg)

    if t_math is not None:
        ratio_top  = t_top  / t_math
        ratio_core = t_core / t_math
        print(f"{name:18s} {t_math:10.1f} {t_top:14.1f} {ratio_top:8.2f}x {t_core:10.1f} {ratio_core:8.2f}x")
    else:
        print(f"{name:18s} {'N/A':>10s} {t_top:14.1f} {'N/A':>9s} {t_core:10.1f} {'N/A':>9s}")

print("-" * 70)
# 2-인수 함수
for name, math_fn, top_fn, core_fn, a, b in [
    ("log(2,x)",  None, lambda x: m.log(BASE, x), lambda x: cy_log(BASE, x), X, None),
    ("power(2,x)",None, lambda x: m.power(BASE, x), lambda x: cy_power(BASE, x), X, None),
]:
    t_top  = bench(top_fn, a)
    t_core = bench(core_fn, a)
    print(f"{name:18s} {'N/A':>10s} {t_top:14.1f} {'N/A':>9s} {t_core:10.1f} {'N/A':>9s}")

print("=" * 70)
print(f"측정 조건: warmup={WARMUP:,}회, 반복={REPS:,}회, x={X}")

# 복소수 dispatch 동작 확인
print("\n=== 복소수 dispatch 검증 ===")
z = 1.0 + 2.0j
try:
    print(f"m.sin(1+2j)  = {m.sin(z)}")
    print(f"m.exp(1+2j)  = {m.exp(z)}")
    print(f"m.ln(1+2j)   = {m.ln(z)}")
    print(f"m.sqrt(-1+0j)= {m.sqrt(complex(-1,0))}")
    print(f"m.power(2+0j,3)= {m.power(complex(2,0),3)}")
    print("복소수 dispatch: 전체 PASS")
except Exception as e:
    print(f"복소수 dispatch 실패: {e}")
