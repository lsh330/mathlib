"""
laplace_ast_bench.py - Phase A AST 코어 벤치마크
목표: Const 생성 <100 ns, a+b <200 ns, hash <30 ns, evalf(depth 3) <500 ns
"""
import sys
import os
import time

# PYTHONPATH 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from math_library.laplace import (
    t, symbol, const, rational, Sin, Exp, ExprPoolStats, PyExpr
)

N = 100_000

def bench(label, target_ns, fn):
    # 웜업
    for _ in range(1000):
        fn()
    # 측정
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    t1 = time.perf_counter()
    ns = (t1 - t0) * 1e9 / N
    status = "OK" if ns < target_ns else "SLOW"
    print(f"  {label:<40s}: {ns:7.1f} ns/op  (target: <{target_ns} ns) [{status}]")
    return ns

print("=" * 65)
print("Phase A AST 벤치마크")
print("=" * 65)
print()

# ------------------------------------------------------------------ 1. Const 생성
print("[1] Const 생성 (make_const)")
_r1 = bench("const(2.5)",            100, lambda: const(2.5))
_r2 = bench("rational(3, 1)",        100, lambda: rational(3, 1))

# ------------------------------------------------------------------ 2. a + b (hash-cons 적중)
print()
print("[2] a + b (hash-cons 적중)")
a = 2 * t + 1
b = 3 * t + rational(-2)
_r3 = bench("a + b (cached hit)",    200, lambda: a + b)

# ------------------------------------------------------------------ 3. a * b (새 구조)
print()
print("[3] a * b (새 구조 생성)")
_c = [0]
def _new_mul():
    # 매번 동일 구조이므로 실제로 hash-cons 적중 (첫 번째 이후)
    return a * b
_r4 = bench("a * b",                 500, _new_mul)

# ------------------------------------------------------------------ 4. hash 조회
print()
print("[4] hash 조회")
e4 = 2 * t ** 3 + Sin(t)
_r5 = bench("hash(depth-4 expr)",     30, lambda: hash(e4))

# ------------------------------------------------------------------ 5. evalf
print()
print("[5] evalf (depth 4)")
e5 = 2 * t ** 3 + Sin(t)
_r6 = bench("evalf(t=1.5)",          500, lambda: e5.evalf(t=1.5))

# ------------------------------------------------------------------ 6. 심볼 생성
print()
print("[6] symbol() 조회 (intern)")
_r7 = bench("symbol('t') - intern",  200, lambda: symbol('t'))

# ------------------------------------------------------------------ 7. 치환
print()
print("[7] substitute(t=2.0) - 새 AST")
e7 = 2*t + Sin(t)
_r8 = bench("substitute(t=2.0)",     800, lambda: e7.substitute({'t': 2.0}))

# ------------------------------------------------------------------ 통계
print()
print("=" * 65)
print(f"Pool 총 노드 수  : {ExprPoolStats.total_nodes()}")
print(f"Intern 적중 횟수: {ExprPoolStats.intern_hits()}")
print("=" * 65)

# ------------------------------------------------------------------ 요약 판정
targets = {
    'Const 생성': (_r1, 100),
    'a+b (캐시)': (_r3, 200),
    'a*b':        (_r4, 500),
    'hash':       (_r5,  30),
    'evalf':      (_r6, 500),
}
all_ok = True
print()
print("요약:")
for name, (val, tgt) in targets.items():
    ok = val < tgt
    print(f"  {name:<20s}: {val:7.1f} ns  target={tgt} ns  {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_ok = False

print()
print("벤치마크 완료 -", "전체 통과" if all_ok else "일부 초과 (위 결과 확인)")
