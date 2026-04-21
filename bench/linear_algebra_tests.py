"""
linear_algebra_tests.py  —  LinearAlgebra 클래스 9 메서드 검증 + 성능 벤치
"""
import sys
import os
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from math_library import LinearAlgebra

la = LinearAlgebra()
rng = np.random.default_rng(42)

def rel_err(A, B):
    return np.linalg.norm(A - B) / max(np.linalg.norm(A), 1e-300)

# ===== 1. LU =====
print("=== LU ===")
for n in [5, 50, 200]:
    A = rng.standard_normal((n, n))
    P, L, U = la.lu(A)
    assert rel_err(P @ A, L @ U) < 1e-12, f"LU n={n} failed"
    assert np.allclose(np.tril(L), L, atol=1e-14), "L not lower"
    assert np.allclose(np.triu(U), U, atol=1e-14), "U not upper"
    assert np.allclose(np.diag(L), 1.0, atol=1e-14), "L diag not 1"
    print(f"  n={n}: rel_err={rel_err(P@A, L@U):.2e}")

# 예외 테스트
try:
    la.lu(np.ones((3, 4)))
except ValueError as e:
    print(f"  직사각 거부: {e}")

# ===== 2. LDU =====
print("=== LDU ===")
A = rng.standard_normal((10, 10))
L, D, U = la.ldu(A)
D_mat = np.diag(D)
P, _, _ = la.lu(A)
assert rel_err(P @ A, L @ D_mat @ U) < 1e-11, "LDU wrong"
print(f"  rel_err: {rel_err(P@A, L@D_mat@U):.2e}")

# ===== 3. QR =====
print("=== QR ===")
for m, n in [(5, 5), (10, 5), (100, 30)]:
    A = rng.standard_normal((m, n))
    Q, R = la.qr(A, mode='reduced')
    assert rel_err(A, Q @ R) < 1e-12, f"QR {m}x{n} failed"
    assert rel_err(Q.T @ Q, np.eye(min(m, n))) < 1e-12, "Q not orthogonal"
    assert np.allclose(np.triu(R), R, atol=1e-14), "R not upper"
    print(f"  {m}x{n}: rel_err={rel_err(A, Q@R):.2e}")

# complete 모드
A = rng.standard_normal((6, 4))
Q_c, R_c = la.qr(A, mode='complete')
assert Q_c.shape == (6, 6), f"Q complete shape {Q_c.shape}"
print(f"  complete 6x4: Q={Q_c.shape}, R={R_c.shape}")

# ===== 4. Cholesky =====
print("=== Cholesky ===")
for n in [5, 50, 200]:
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)
    L = la.cholesky(A)
    assert rel_err(A, L @ L.T) < 1e-12
    assert np.allclose(np.tril(L), L, atol=1e-14)
    print(f"  n={n}: rel_err={rel_err(A, L@L.T):.2e}")

# lower=False
M = rng.standard_normal((5, 5))
A = M @ M.T + 5 * np.eye(5)
U_ch = la.cholesky(A, lower=False)
assert rel_err(A, U_ch.T @ U_ch) < 1e-12
print(f"  lower=False: rel_err={rel_err(A, U_ch.T@U_ch):.2e}")

try:
    la.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
except ValueError as e:
    print(f"  비SPD 거부: {e}")

try:
    la.cholesky(np.array([[1.0, 2.0], [3.0, 4.0]]))
except ValueError as e:
    print(f"  비대칭 거부: {e}")

# ===== 5. SVD =====
print("=== SVD ===")
for m, n in [(5, 5), (10, 5), (50, 30)]:
    A = rng.standard_normal((m, n))
    U, S, Vt = la.svd(A, full_matrices=False)
    reconstructed = U @ np.diag(S) @ Vt
    assert rel_err(A, reconstructed) < 1e-10, f"SVD {m}x{n} failed: {rel_err(A, reconstructed):.2e}"
    assert rel_err(U.T @ U, np.eye(U.shape[1])) < 1e-10, "U not ortho"
    assert rel_err(Vt @ Vt.T, np.eye(Vt.shape[0])) < 1e-10, "V not ortho"
    assert np.all(S[:-1] >= S[1:]), "S not sorted"
    print(f"  {m}x{n}: rel_err={rel_err(A, reconstructed):.2e}")

# Hilbert matrix (ill-conditioned)
n = 8
H = np.array([[1.0/(i+j+1) for j in range(n)] for i in range(n)])
U, S, Vt = la.svd(H, full_matrices=False)
err_H = rel_err(H, U @ np.diag(S) @ Vt)
print(f"  Hilbert 8x8: rel_err={err_H:.2e}")

# 랭크 부족 행렬
A_rd = np.zeros((6, 4))
A_rd[:3, :3] = rng.standard_normal((3, 3))
U, S, Vt = la.svd(A_rd, full_matrices=False)
err_rd = rel_err(A_rd, U @ np.diag(S) @ Vt)
print(f"  rank-deficient 6x4: rel_err={err_rd:.2e}")

# ===== 6. Gaussian elimination =====
print("=== Gaussian ===")
A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)
x = la.gaussian_elimination(A, b)
assert np.allclose(A @ x, b, atol=1e-12)
print(f"  Ax=b: x={x}, resid={np.linalg.norm(A@x - b):.2e}")

# RREF
rref = la.gaussian_elimination(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float))
print(f"  RREF:\n{rref}")

# REF
ref = la.gaussian_elimination(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float), form='ref')
print(f"  REF:\n{ref}")

# ===== 7. det =====
print("=== det ===")
A2 = np.array([[1, 2], [3, 4]], dtype=float)
d = la.det(A2)
assert abs(d - (-2.0)) < 1e-14, f"det = {d}"

for n in [5, 50, 100]:
    A = rng.standard_normal((n, n))
    d_mine = la.det(A)
    d_ref = np.linalg.det(A)
    rel = abs(d_mine - d_ref) / max(abs(d_ref), 1e-300)
    assert rel < 1e-10, f"det n={n} rel_err={rel}"
    print(f"  n={n}: rel_err vs numpy={rel:.2e}")

# 특이 행렬 det = 0
d_sing = la.det(np.array([[1, 2], [2, 4]], dtype=float))
print(f"  특이행렬 det={d_sing:.2e} (expected ~0)")

# ===== 8. rank =====
print("=== rank ===")
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
r = la.rank(A)
assert r == 2, f"rank = {r}"

A_full = rng.standard_normal((10, 10))
assert la.rank(A_full) == 10

A_def = np.zeros((5, 5))
A_def[0] = [1, 0, 0, 0, 0]
A_def[1] = [0, 1, 0, 0, 0]
assert la.rank(A_def) == 2

print(f"  rank-2 3x3: {r}")
print(f"  full-rank 10x10: {la.rank(A_full)}")
print(f"  rank-2 5x5: {la.rank(A_def)}")

# ===== 9. inverse =====
print("=== inverse ===")
for n in [5, 50, 100]:
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    A_inv = la.inverse(A)
    assert rel_err(A @ A_inv, np.eye(n)) < 1e-10, f"inverse n={n}"
    print(f"  n={n}: rel_err={rel_err(A @ A_inv, np.eye(n)):.2e}")

try:
    la.inverse(np.array([[1, 2], [2, 4]], dtype=float))
except ValueError as e:
    print(f"  특이 거부: {e}")

# ===== 성능 =====
import time
print("\n=== 성능 ===")
for n in [100, 500, 1000]:
    A = rng.standard_normal((n, n))
    t0 = time.perf_counter()
    P, L, U = la.lu(A)
    t1 = time.perf_counter()
    t0n = time.perf_counter()
    _ = np.linalg.qr(A)
    t1n = time.perf_counter()
    print(f"  n={n}: LU={(t1-t0)*1000:.1f} ms  (numpy QR 참고: {(t1n-t0n)*1000:.1f} ms)")

for n in [100, 500, 1000]:
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)
    t0 = time.perf_counter()
    L = la.cholesky(A)
    t1 = time.perf_counter()
    err = rel_err(A, L @ L.T)
    print(f"  Cholesky n={n}: {(t1-t0)*1000:.1f} ms, err={err:.2e}")

for n in [50, 200]:
    A = rng.standard_normal((n, n))
    t0 = time.perf_counter()
    U, S, Vt = la.svd(A)
    t1 = time.perf_counter()
    err = rel_err(A, U @ np.diag(S) @ Vt)
    print(f"  SVD n={n}x{n}: {(t1-t0)*1000:.1f} ms, err={err:.2e}")

# ===== 기존 회귀 =====
from math_library import NumericalAnalysis, Differentiation
import math
na = NumericalAnalysis()
assert abs(na.rk4(lambda t, y: -y, 0, 1, 1, 100) - math.exp(-1)) < 1e-10
print("\nNumericalAnalysis 회귀: PASS")

import math_library as m
assert abs(m.sin(1.2345) - math.sin(1.2345)) < 1e-14
print("mathlib core 회귀: PASS")

print("\n모든 테스트 완료.")
