"""
bench/laplace_tests.py — Phase B Laplace 변환 정확성 + 성능 벤치마크
20+ 표준 Laplace 쌍 수치 검증 (여러 s 값에서 1e-9 이내 일치)
"""
import sys
import time
import math

sys.path.insert(0, 'src')

from math_library.laplace import (
    Laplace, t, s, const, symbol,
    Sin, Cos, Exp, Sinh, Cosh,
    PyExpr
)

L = Laplace()

# ====================================================================
# 정확성 테스트: (f_expr, truth_lambda, note, s_test_values)
# s_test_values: ROC 조건을 만족하는 s 값 목록
# ====================================================================

tests = [
    # ---------------------------------------------------------------- primitives
    (const(1),
     lambda sv: 1/sv,
     'L{1}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (t,
     lambda sv: 1/sv**2,
     'L{t}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (t**2,
     lambda sv: 2/sv**3,
     'L{t^2}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (t**3,
     lambda sv: 6/sv**4,
     'L{t^3}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (t**4,
     lambda sv: 24/sv**5,
     'L{t^4}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (Exp(2*t),
     lambda sv: 1/(sv-2),
     'L{e^(2t)}',
     [3.0, 5.0, 10.0]),            # ROC: s > 2

    (Exp(-3*t),
     lambda sv: 1/(sv+3),
     'L{e^(-3t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (Sin(2*t),
     lambda sv: 2/(sv**2+4),
     'L{sin(2t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (Cos(3*t),
     lambda sv: sv/(sv**2+9),
     'L{cos(3t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (Sinh(2*t),
     lambda sv: 2/(sv**2-4),
     'L{sinh(2t)}',
     [3.0, 5.0, 10.0]),            # ROC: s > 2

    (Cosh(2*t),
     lambda sv: sv/(sv**2-4),
     'L{cosh(2t)}',
     [3.0, 5.0, 10.0]),            # ROC: s > 2

    # ---------------------------------------------------------------- linearity
    (3*t + const(2),
     lambda sv: 3/sv**2 + 2/sv,
     'L{3t+2}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (const(5)*t**2 + const(3)*t + const(1),
     lambda sv: 10/sv**3 + 3/sv**2 + 1/sv,
     'L{5t^2+3t+1}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    # ---------------------------------------------------------------- s-shift
    (Exp(-t)*Sin(2*t),
     lambda sv: 2/((sv+1)**2+4),
     'L{e^(-t)sin(2t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (Exp(-2*t)*Cos(3*t),
     lambda sv: (sv+2)/((sv+2)**2+9),
     'L{e^(-2t)cos(3t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (Exp(-t)*t**2,
     lambda sv: 2/(sv+1)**3,
     'L{t^2 e^(-t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (t * Exp(-t),
     lambda sv: 1/(sv+1)**2,
     'L{t e^(-t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (t**2 * Exp(-2*t),
     lambda sv: 2/(sv+2)**3,
     'L{t^2 e^(-2t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    # ---------------------------------------------------------------- frequency differentiation
    (t * Sin(t),
     lambda sv: 2*sv/(sv**2+1)**2,
     'L{t sin(t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (t * Cos(t),
     lambda sv: (sv**2-1)/(sv**2+1)**2,
     'L{t cos(t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (t * Sin(2*t),
     lambda sv: 4*sv/(sv**2+4)**2,
     'L{t sin(2t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    # ---------------------------------------------------------------- combination
    (Sin(t) + Cos(t),
     lambda sv: 1/(sv**2+1) + sv/(sv**2+1),
     'L{sin(t)+cos(t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (const(2)*Sin(3*t) + const(5)*Cos(4*t),
     lambda sv: 6/(sv**2+9) + 5*sv/(sv**2+16),
     'L{2sin(3t)+5cos(4t)}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),

    (Exp(-t) * (Sin(2*t) + Cos(2*t)),
     lambda sv: 2/((sv+1)**2+4) + (sv+1)/((sv+1)**2+4),
     'L{e^(-t)(sin(2t)+cos(2t))}',
     [0.5, 1.0, 2.0, 5.0, 10.0]),
]

# ====================================================================
# 정확성 검증 루프
# ====================================================================

print("=" * 60)
print("Laplace 변환 정확성 검증 (Phase B)")
print("=" * 60)

passed = 0
failed = 0
errors = []

for f_expr, truth_fn, note, s_vals in tests:
    try:
        F = L.transform(f_expr)
        ok = True
        for sv in s_vals:
            got = F.evalf(s=sv)
            exp = truth_fn(sv)
            if abs(got - exp) > 1e-9:
                errors.append(f'  MISMATCH {note} @ s={sv}: got={got:.10f}, exp={exp:.10f}, diff={abs(got-exp):.2e}')
                ok = False
                break
        if ok:
            print(f'  [PASS] {note:40s}  F(s)= {F}')
            passed += 1
        else:
            print(f'  [FAIL] {note}')
            failed += 1
    except Exception as e:
        print(f'  [ERROR] {note}: {e}')
        failed += 1

if errors:
    print("\n수치 불일치 상세:")
    for e in errors:
        print(e)

print(f"\n결과: {passed} PASS, {failed} FAIL / {len(tests)} total")

# ====================================================================
# subs() + latex() 동작 확인
# ====================================================================

print("\n" + "=" * 60)
print("기호 치환 + LaTeX 출력 검증")
print("=" * 60)

F_sin2t = L.transform(Sin(2*t))
print(f"F = L{{sin(2t)}} = {F_sin2t}")
print(f"F.latex() = {F_sin2t.latex()}")

# 수치 치환
F_at_1 = F_sin2t.subs(s=1.0)
print(f"F.subs(s=1.0) = {F_at_1}")
expected_val = 2/(1.0**2 + 4)
got_val = float(str(F_at_1))
print(f"  expected = {expected_val:.8f}")

# 기호 치환: s → s+1
F_shifted = F_sin2t.subs(s=s+1)
print(f"F.subs(s=s+1) = {F_shifted}")
print(f"  evalf at s=2: {F_shifted.evalf(s=2.0):.8f}, expected: {2/((2+1)**2+4):.8f}")

# ====================================================================
# 성능 벤치마크
# ====================================================================

print("\n" + "=" * 60)
print("성능 벤치마크 (Phase B)")
print("=" * 60)

N1 = 10_000
N2 = 10_000
N3 = 100_000

# 1) 단순 primitive: L{sin(2t)}
L.clear_cache()
expr1 = Sin(2*t)
t0 = time.perf_counter()
for _ in range(N1):
    L.clear_cache()
    L.transform(expr1)
t1 = time.perf_counter()
us1 = (t1 - t0) / N1 * 1e6
print(f"L{{sin(2t)}}      : {us1:.1f} us  (target < 15 us)  {'OK' if us1 < 15 else 'SLOW'}")

# 2) 복합 3단: L{e^(-t) sin(3t)}
L.clear_cache()
expr2 = Exp(-t) * Sin(3*t)
t0 = time.perf_counter()
for _ in range(N2):
    L.clear_cache()
    L.transform(expr2)
t1 = time.perf_counter()
us2 = (t1 - t0) / N2 * 1e6
print(f"L{{e^(-t)sin(3t)}} : {us2:.1f} us  (target < 30 us)  {'OK' if us2 < 30 else 'SLOW'}")

# 3) 캐시 재호출
expr3 = t**3 * Exp(-2*t)
L.transform(expr3)  # 워밍업 (캐시에 등록)
t0 = time.perf_counter()
for _ in range(N3):
    L.transform(expr3)
t1 = time.perf_counter()
ns3 = (t1 - t0) / N3 * 1e9
print(f"캐시 hit           : {ns3:.1f} ns  (target < 500 ns) {'OK' if ns3 < 500 else 'SLOW'}")
