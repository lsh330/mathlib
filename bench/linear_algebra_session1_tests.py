"""
linear_algebra_session1_tests.py  —  Session 1 신규 13 메서드 검증
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from math_library import LinearAlgebra

la = LinearAlgebra()
rng = np.random.default_rng(42)

def rel_err(A, B):
    return np.linalg.norm(A - B) / max(np.linalg.norm(A), 1e-300)

PASS = 0
FAIL = 0

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}  {msg}")
        FAIL += 1

# ============================================================== 1. QR pivot
print("=== 1. QR pivot ===")
A = rng.standard_normal((5, 8))
Q, R, P = la.qr(A, pivot=True)
err_qrp = rel_err(A[:, P], Q @ R)
check("A[:,P] = Q@R", err_qrp < 1e-12, f"err={err_qrp:.2e}")
diag_R = np.abs(np.diag(R))
check("|diag(R)| 내림차순", np.all(diag_R[:-1] >= diag_R[1:] - 1e-14),
      f"diag={diag_R}")
# Q 직교성
k = min(5, 8)
check("Q 직교성", rel_err(Q.T @ Q, np.eye(k)) < 1e-12)
print(f"  err={err_qrp:.2e}, P={P}")

# 역방 호환: pivot=False는 2-tuple 반환
Q2, R2 = la.qr(A[:, :5], pivot=False)
check("pivot=False 2-tuple 유지", rel_err(A[:, :5], Q2 @ R2) < 1e-12)

# ============================================================== 2. logdet
print("=== 2. logdet ===")
A_ld = rng.standard_normal((10, 10)) + 10 * np.eye(10)
sign_mine, ld_mine = la.logdet(A_ld, return_sign=True)
actual = np.linalg.slogdet(A_ld)
check("sign", abs(sign_mine - actual[0]) < 1e-14, f"mine={sign_mine}, ref={actual[0]}")
check("logabsdet", abs(ld_mine - actual[1]) < 1e-10, f"mine={ld_mine:.6f}, ref={actual[1]:.6f}")
print(f"  logdet={ld_mine:.4f} (numpy {actual[1]:.4f})")

# 음수 행렬식
A_neg = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # det = -1
s2, ld2 = la.logdet(A_neg, return_sign=True)
check("sign=-1", abs(s2 - (-1.0)) < 1e-14)
check("log|det|=0", abs(ld2) < 1e-14)

# ============================================================== 3. eigen
print("=== 3. eigen ===")
# 대칭 행렬 (실수 고유값)
B = rng.standard_normal((5, 5))
A_sym = B + B.T
vals_s, vecs_s = la.eigen(A_sym, symmetric=True)
sym_pass = True
for i in range(5):
    lhs = A_sym @ vecs_s[:, i].real
    rhs = vals_s[i].real * vecs_s[:, i].real
    e = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(rhs), 1e-300)
    if e > 1e-8:
        sym_pass = False
        print(f"    sym eigvec {i} err={e:.2e}")
check("symmetric eigen A@v=λv", sym_pass)
print(f"  vals={vals_s.real}")

# 90도 회전 행렬: 고유값 ±i
A_rot = np.array([[0, -1], [1, 0]], dtype=float)
vals_rot, _ = la.eigen(A_rot)
imag_sorted = sorted([v.imag for v in vals_rot])
check("complex eigen (±i)", np.allclose(imag_sorted, [-1.0, 1.0], atol=1e-10),
      f"imag={imag_sorted}")
print(f"  complex vals={vals_rot}")

# 일반 5x5 행렬
A_gen = rng.standard_normal((5, 5))
vals_g, vecs_g = la.eigen(A_gen)
gen_pass = True
for i in range(5):
    Av = A_gen.astype(complex) @ vecs_g[:, i]
    lv = vals_g[i] * vecs_g[:, i]
    e = np.linalg.norm(Av - lv) / max(np.linalg.norm(lv), 1e-300)
    if e > 1e-7:
        gen_pass = False
        print(f"    general eigvec {i} err={e:.2e}")
check("general eigen A@v=λv", gen_pass)

# ============================================================== 4. diagonalize
print("=== 4. diagonalize ===")
A_d = rng.standard_normal((5, 5))
try:
    P_d, D_d = la.diagonalize(A_d)
    err_diag = rel_err(A_d.astype(complex), P_d @ D_d @ np.linalg.inv(P_d))
    check("A = P D P^{-1}", err_diag < 1e-7, f"err={err_diag:.2e}")
    print(f"  diag err={err_diag:.2e}")
except Exception as ex:
    print(f"  diagonalize exception: {ex}")
    FAIL += 1

# ============================================================== 5. characteristic_polynomial
print("=== 5. characteristic_polynomial ===")
A_cp = np.array([[1.0, 2.0], [3.0, 4.0]])
c = la.characteristic_polynomial(A_cp)
# p(λ) = λ^2 - 5λ - 2
check("2x2 coeffs", np.allclose(c, [1, -5, -2], atol=1e-12),
      f"coeffs={c}")
print(f"  coeffs={c}  (expected [1,-5,-2])")

# 3x3 항등 행렬: p(λ) = (λ-1)^3 = λ^3 - 3λ^2 + 3λ - 1
c3 = la.characteristic_polynomial(np.eye(3))
check("I_3 char poly", np.allclose(c3, [1, -3, 3, -1], atol=1e-12),
      f"coeffs={c3}")

# ============================================================== 6. cayley_hamilton
print("=== 6. Cayley-Hamilton ===")
A_ch = rng.standard_normal((5, 5))
res = la.cayley_hamilton(A_ch, return_residual=True)
check("||p(A)|| < 1e-7", res < 1e-7, f"res={res:.2e}")
print(f"  ||p(A)||={res:.2e}")

# 2x2 검증
A_ch2 = np.array([[1.0, 2.0], [3.0, 4.0]])
res2 = la.cayley_hamilton(A_ch2, return_residual=True)
check("2x2 ||p(A)||", res2 < 1e-12, f"res={res2:.2e}")

# ============================================================== 7. pinv
print("=== 7. pinv ===")
# fat matrix (5x8)
A_fat = rng.standard_normal((5, 8))
Ap_fat = la.pinv(A_fat)
e1 = rel_err(A_fat @ Ap_fat @ A_fat, A_fat)
e2 = rel_err(Ap_fat @ A_fat @ Ap_fat, Ap_fat)
check("A A+ A = A (fat)", e1 < 1e-10, f"err={e1:.2e}")
check("A+ A A+ = A+ (fat)", e2 < 1e-10, f"err={e2:.2e}")
print(f"  fat pinv shape={Ap_fat.shape}, e1={e1:.2e}, e2={e2:.2e}")

# tall matrix (8x5)
A_tall = rng.standard_normal((8, 5))
Ap_tall = la.pinv(A_tall)
e3 = rel_err(A_tall @ Ap_tall @ A_tall, A_tall)
e4 = rel_err(Ap_tall @ A_tall @ Ap_tall, Ap_tall)
check("A A+ A = A (tall)", e3 < 1e-10, f"err={e3:.2e}")
check("A+ A A+ = A+ (tall)", e4 < 1e-10, f"err={e4:.2e}")

# 정방 non-singular
A_sq = rng.standard_normal((5, 5))
Ap_sq = la.pinv(A_sq)
e5 = rel_err(A_sq @ Ap_sq, np.eye(5))
check("square pinv = inverse", e5 < 1e-10, f"err={e5:.2e}")
print(f"  square pinv err={e5:.2e}")

# ============================================================== 8. gauss_jordan
print("=== 8. gauss_jordan ===")
A_gj = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
rref = la.gauss_jordan(A_gj)
check("RREF = I (rank 3)", rel_err(rref, np.eye(3)) < 1e-12,
      f"rref=\n{rref}")
print(f"  RREF:\n{rref}")

b_gj = np.array([1.0, 2.0, 3.0])
x_gj = la.gauss_jordan(A_gj, b_gj)
check("Ax=b 해", np.allclose(A_gj @ x_gj, b_gj, atol=1e-12),
      f"resid={np.linalg.norm(A_gj @ x_gj - b_gj):.2e}")
print(f"  x={x_gj}")

# rank-deficient RREF
A_rd_gj = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
rref_rd = la.gauss_jordan(A_rd_gj)
print(f"  rank-deficient RREF:\n{rref_rd}")

# ============================================================== 9. gram_schmidt
print("=== 9. gram_schmidt ===")
V_gs = rng.standard_normal((5, 3))
Q_gs = la.gram_schmidt(V_gs, modified=True)
check("직교정규 (modified)", rel_err(Q_gs.T @ Q_gs, np.eye(3)) < 1e-12,
      f"err={rel_err(Q_gs.T @ Q_gs, np.eye(3)):.2e}")
print(f"  Q^T Q err={rel_err(Q_gs.T @ Q_gs, np.eye(3)):.2e}")

Q_cl = la.gram_schmidt(V_gs, modified=False)
check("직교정규 (classical)", rel_err(Q_cl.T @ Q_cl, np.eye(3)) < 1e-10,
      f"err={rel_err(Q_cl.T @ Q_cl, np.eye(3)):.2e}")

# 선형 종속 입력 테스트
V_dep = np.column_stack([V_gs[:, 0], V_gs[:, 0] * 2.0, V_gs[:, 1]])
Q_dep = la.gram_schmidt(V_dep, modified=True)
check("선형 종속 제거", Q_dep.shape[1] == 2, f"shape={Q_dep.shape}")
print(f"  linearly dependent: output shape={Q_dep.shape}")

# ============================================================== 10~13. 부분공간
print("=== 10~13. 부분공간 ===")
A_sub = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # rank 2

# null_space
N = la.null_space(A_sub)
e_null = np.max(np.abs(A_sub @ N))
check("A @ N ~= 0", e_null < 1e-10, f"max_err={e_null:.2e}")
check("null_space orthnorm", rel_err(N.T @ N, np.eye(N.shape[1])) < 1e-10,
      f"err={rel_err(N.T @ N, np.eye(N.shape[1])):.2e}")
print(f"  null_space shape={N.shape}, A@N max={e_null:.2e}")

# column_space
C = la.column_space(A_sub)
proj_err = rel_err(C @ C.T @ A_sub, A_sub)
check("column_space proj", proj_err < 1e-10, f"err={proj_err:.2e}")
check("column_space 직교정규", rel_err(C.T @ C, np.eye(C.shape[1])) < 1e-10)
print(f"  column_space shape={C.shape}, proj_err={proj_err:.2e}")

# row_space
R_sub = la.row_space(A_sub)
check("row_space shape", R_sub.shape[0] == A_sub.shape[1])
check("row_space 직교정규", rel_err(R_sub.T @ R_sub, np.eye(R_sub.shape[1])) < 1e-10)
print(f"  row_space shape={R_sub.shape}")

# left_null_space
LN = la.left_null_space(A_sub)
e_ln = np.max(np.abs(A_sub.T @ LN))
check("A^T @ LN ~= 0", e_ln < 1e-10, f"max_err={e_ln:.2e}")
print(f"  left_null_space shape={LN.shape}")

# 차원 정리 확인: rank + nullity = n
r_sub = int(np.round(la.rank(A_sub)))
check("rank + nullity = n", r_sub + N.shape[1] == A_sub.shape[1],
      f"rank={r_sub}, nullity={N.shape[1]}, n={A_sub.shape[1]}")
check("rank + left_nullity = m", r_sub + LN.shape[1] == A_sub.shape[0],
      f"rank={r_sub}, left_nullity={LN.shape[1]}, m={A_sub.shape[0]}")

# ============================================================== 기존 회귀
print("\n=== 기존 회귀 ===")
# LU
A_reg = rng.standard_normal((50, 50))
P_r, L_r, U_r = la.lu(A_reg)
check("LU PA=LU", rel_err(P_r @ A_reg, L_r @ U_r) < 1e-12)

# QR
A_qr_r = rng.standard_normal((10, 7))
Q_r2, R_r2 = la.qr(A_qr_r, mode='reduced')
check("QR A=QR", rel_err(A_qr_r, Q_r2 @ R_r2) < 1e-12)

# SVD
A_svd_r = rng.standard_normal((10, 7))
U_r2, S_r2, Vt_r2 = la.svd(A_svd_r, full_matrices=False)
check("SVD A=USV^T", rel_err(A_svd_r, U_r2 @ np.diag(S_r2) @ Vt_r2) < 1e-10)

# NumericalAnalysis
from math_library import NumericalAnalysis
import math
na = NumericalAnalysis()
check("RK4 dy/dt=-y", abs(na.rk4(lambda t, y: -y, 0, 1, 1, 100) - math.exp(-1)) < 1e-10)

# mathlib core
import math_library as m_lib
check("sin(1.2345)", abs(m_lib.sin(1.2345) - math.sin(1.2345)) < 1e-14)

print(f"\n{'='*50}")
print(f"총 결과: PASS={PASS}, FAIL={FAIL}")
if FAIL == 0:
    print("모든 테스트 통과.")
else:
    print(f"주의: {FAIL}개 실패.")
