"""
Session 2 테스트: LinearAlgebra 11 신규 메서드 검증
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import math
from math_library import LinearAlgebra

la = LinearAlgebra()
rng = np.random.default_rng(42)

def rel_err(A, B):
    return np.linalg.norm(np.asarray(A, dtype=complex) - np.asarray(B, dtype=complex)) / max(np.linalg.norm(np.asarray(A, dtype=complex)), 1e-300)

PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        FAIL += 1

# ============================================================ 1. tensor_product
print("=== tensor_product (Kronecker) ===")
A = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[0, 5], [6, 7]], dtype=float)
K = la.tensor_product(A, B)
expected = np.array([
    [0, 5, 0, 10],
    [6, 7, 12, 14],
    [0, 15, 0, 20],
    [18, 21, 24, 28]
], dtype=float)
check("2x2 kron shape", K.shape == (4, 4))
check("2x2 kron values", rel_err(K, expected) < 1e-14)

A2 = rng.standard_normal((3, 2))
B2 = rng.standard_normal((2, 4))
K2 = la.tensor_product(A2, B2)
check("3x2 ⊗ 2x4 shape", K2.shape == (6, 8))
check("3x2 ⊗ 2x4 vs np.kron", rel_err(K2, np.kron(A2, B2)) < 1e-12)

# ============================================================ 2. outer_product
print("=== outer_product ===")
a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6, 7], dtype=float)
O = la.outer_product(a, b)
expected_o = np.array([[4, 5, 6, 7], [8, 10, 12, 14], [12, 15, 18, 21]], dtype=float)
check("outer shape", O.shape == (3, 4))
check("outer values", rel_err(O, expected_o) < 1e-14)
check("outer vs np.outer", rel_err(O, np.outer(a, b)) < 1e-14)

# ============================================================ 3. inner_product
print("=== inner_product ===")
a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)
d = la.inner_product(a, b)
check("real inner 32", abs(d - 32) < 1e-14)

ac = np.array([1+2j, 3+4j])
bc = np.array([5+6j, 7+8j])
d_c = la.inner_product(ac, bc)
exp_c = (1-2j)*(5+6j) + (3-4j)*(7+8j)
check("complex Hermitian inner", abs(d_c - exp_c) < 1e-14, f"got {d_c}, expected {exp_c}")

# ============================================================ 4. tensor_contract
print("=== tensor_contract ===")
A3 = rng.standard_normal((3, 5))
B3 = rng.standard_normal((5, 4))
C = la.tensor_contract(A3, B3, axes_a=1, axes_b=0)
check("2D matmul shape", C.shape == (3, 4))
check("2D matmul values", rel_err(C, A3 @ B3) < 1e-12, f"rel_err={rel_err(C, A3@B3):.2e}")

T1 = rng.standard_normal((2, 3, 4))
T2 = rng.standard_normal((4, 5, 6))
R = la.tensor_contract(T1, T2, axes_a=2, axes_b=0)
ref = np.tensordot(T1, T2, axes=([2], [0]))
check("3D contract shape", R.shape == (2, 3, 5, 6), f"got {R.shape}")
check("3D contract values", rel_err(R, ref) < 1e-12, f"rel_err={rel_err(R, ref):.2e}")

# ============================================================ 5. tensor_transpose
print("=== tensor_transpose ===")
T = rng.standard_normal((2, 3, 4))
Tt = la.tensor_transpose(T)
check("transpose None shape", Tt.shape == (4, 3, 2), f"got {Tt.shape}")
check("transpose None values", rel_err(Tt, np.transpose(T)) < 1e-14)

Tt2 = la.tensor_transpose(T, axes=(1, 2, 0))
check("transpose (1,2,0) shape", Tt2.shape == (3, 4, 2), f"got {Tt2.shape}")
check("transpose (1,2,0) values", rel_err(Tt2, np.transpose(T, (1,2,0))) < 1e-14)

# ============================================================ 6-7. is_symmetric / is_skew_symmetric
print("=== is_symmetric / is_skew_symmetric ===")
S = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)
check("symmetric matrix True", la.is_symmetric(S))
check("symmetric matrix not skew", not la.is_skew_symmetric(S))

Ks = np.array([[0, 2, -3], [-2, 0, 5], [3, -5, 0]], dtype=float)
check("skew True", la.is_skew_symmetric(Ks))
check("skew not symmetric", not la.is_symmetric(Ks))

check("non-square False", not la.is_symmetric(np.array([[1, 2, 3], [4, 5, 6]], dtype=float)))

# ============================================================ 8-9. symmetrize / skew_part
print("=== symmetrize / skew_part ===")
A4 = rng.standard_normal((5, 5))
Ss = la.symmetrize(A4)
Ks2 = la.skew_part(A4)
check("symmetrize is symmetric", la.is_symmetric(Ss))
check("skew_part is skew", la.is_skew_symmetric(Ks2))
check("S+K=A", rel_err(Ss + Ks2, A4) < 1e-14, f"rel_err={rel_err(Ss+Ks2, A4):.2e}")

# 복소 Hermitian
Ac = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
H_mat = la.symmetrize(Ac)
check("Hermitian H=H^H", rel_err(H_mat, np.conj(H_mat).T) < 1e-14)

# ============================================================ 10. jacobian
print("=== jacobian ===")
def f_jac(x, y):
    return [x**2 + y, x*y, y**2]

J = la.jacobian(f_jac, [1.0, 2.0])
# ∂f1/∂x=2x=2, ∂f1/∂y=1
# ∂f2/∂x=y=2, ∂f2/∂y=x=1
# ∂f3/∂x=0,   ∂f3/∂y=2y=4
expected_J = np.array([[2, 1], [2, 1], [0, 4]], dtype=float)
check("jacobian shape", J.shape == (3, 2), f"got {J.shape}")
check("jacobian values", rel_err(J, expected_J) < 1e-5, f"rel_err={rel_err(J, expected_J):.2e}\nJ={J}\nexp={expected_J}")

# ============================================================ 11. hessian
print("=== hessian ===")
def g_hess(x, y):
    return x**2 + 3*x*y + y**3

H_res = la.hessian(g_hess, [1.0, 2.0])
# ∂²g/∂x²=2, ∂²g/∂x∂y=3, ∂²g/∂y²=6y=12
expected_H = np.array([[2, 3], [3, 12]], dtype=float)
check("hessian shape", H_res.shape == (2, 2), f"got {H_res.shape}")
check("hessian values", rel_err(H_res, expected_H) < 1e-4, f"rel_err={rel_err(H_res, expected_H):.2e}\nH={H_res}")
check("hessian symmetric", rel_err(H_res, H_res.T) < 1e-6)

# ============================================================ 성능
print("\n=== 성능 ===")
import time
A_perf = rng.standard_normal((100, 100))
B_perf = rng.standard_normal((100, 100))
t0 = time.perf_counter()
for _ in range(100):
    la.tensor_product(A_perf, B_perf)
t1 = time.perf_counter()
print(f"  Kron 100x100 ⊗ 100x100: {(t1-t0)*10:.2f} ms/op")

# ============================================================ Session 1 회귀
print("\n=== Session 1 회귀 ===")
A_reg = rng.standard_normal((5, 5))
A_reg = A_reg + A_reg.T
vals, vecs = la.eigen(A_reg, symmetric=True)
eigen_ok = all(rel_err(A_reg @ vecs[:, i].real, vals[i].real * vecs[:, i].real) < 1e-8
               for i in range(5))
check("eigen regression", eigen_ok)

pinv_A = rng.standard_normal((4, 6))
P = la.pinv(pinv_A)
check("pinv regression", P.shape == (6, 4))

# ============================================================ 기존 mathlib core 회귀
print("\n=== mathlib core 회귀 ===")
from math_library import NumericalAnalysis
import math_library as m
na = NumericalAnalysis()
rk4_val = na.rk4(lambda t, y: -y, 0, 1, 1, 100)
check("rk4 regression", abs(rk4_val - math.exp(-1)) < 1e-10)
check("sin regression", abs(m.sin(1.2345) - math.sin(1.2345)) < 1e-14)

# ============================================================ 요약
print(f"\n{'='*40}")
print(f"PASS: {PASS}  FAIL: {FAIL}  TOTAL: {PASS+FAIL}")
if FAIL == 0:
    print("ALL PASS")
else:
    print(f"FAILED {FAIL} tests")
    sys.exit(1)
