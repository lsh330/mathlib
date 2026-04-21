# _kernels.pxd — 선형대수 커널 인라인 유틸리티
# Householder reflection, Givens rotation, 전/후방 대입
# cython: language_level=3

from libc.math cimport sqrt, fabs, copysign

# ------------------------------------------------------------------ Householder
cdef inline void householder_vec(
    double* x,      # 입력 벡터 (길이 m), 제자리에서 v로 변환
    int m,
    double* beta,   # 출력: beta = 2 / (v^T v), v[0]=1로 정규화
    double* alpha   # 출력: alpha = H x 의 첫 번째 성분 (= -sign(x[0])*||x||)
) nogil:
    """
    Householder 벡터 계산.
    v = x + sign(x[0])*||x||*e_1,  beta = 2/(v^T v)
    H = I - beta * v * v^T,  Hx = alpha * e_1
    """
    cdef int i
    cdef double sigma = 0.0, norm_x, sign_x0, v0

    # ||x[1:]||^2
    for i in range(1, m):
        sigma += x[i] * x[i]

    norm_x = sqrt(x[0]*x[0] + sigma)

    if norm_x == 0.0:
        beta[0] = 0.0
        alpha[0] = 0.0
        return

    # sign(x[0]): 0이면 +1 처리
    if x[0] >= 0.0:
        sign_x0 = 1.0
    else:
        sign_x0 = -1.0

    alpha[0] = -sign_x0 * norm_x

    # v[0] = x[0] - alpha (= x[0] + sign(x[0])*||x||)
    v0 = x[0] - alpha[0]
    x[0] = v0

    # beta = 2 / (v^T v) = 2 / (v0^2 + sigma)
    beta[0] = 2.0 / (v0 * v0 + sigma)


cdef inline void apply_householder_left(
    double* A,      # (m x n) C-contiguous
    int m, int n,
    double* v,      # Householder 벡터 (길이 m), v[0] 암묵적 1
    double beta
) nogil:
    """
    제자리 A <- H A  (H = I - beta*v*v^T, v[0]=1)
    A[i,j] -= beta * v[i] * (v^T A[:,j])
    """
    cdef int i, j
    cdef double dot

    for j in range(n):
        # dot = v^T A[:,j]  (v[0]=1)
        dot = A[j]  # i=0: v[0]=1
        for i in range(1, m):
            dot += v[i] * A[i*n + j]
        # A[:,j] -= beta * dot * v
        A[j] -= beta * dot          # i=0
        for i in range(1, m):
            A[i*n + j] -= beta * dot * v[i]


cdef inline void apply_householder_right(
    double* A,      # (m x n) C-contiguous
    int m, int n,
    double* v,      # Householder 벡터 (길이 n), v[0] 암묵적 1
    double beta
) nogil:
    """
    제자리 A <- A H  (H = I - beta*v*v^T, v[0]=1)
    A[i,j] -= beta * (A[i,:] v) * v[j]
    """
    cdef int i, j
    cdef double dot

    for i in range(m):
        # dot = A[i,:] v  (v[0]=1)
        dot = A[i*n]  # j=0: v[0]=1
        for j in range(1, n):
            dot += A[i*n + j] * v[j]
        # A[i,:] -= beta * dot * v
        A[i*n] -= beta * dot        # j=0
        for j in range(1, n):
            A[i*n + j] -= beta * dot * v[j]


# ------------------------------------------------------------------ Givens
cdef inline void givens_cs(
    double a, double b,
    double* c, double* s
) nogil:
    """
    Givens 회전 (c, s) 계산: [c s; -s c]^T [a; b] = [r; 0]
    수치 안정: Golub & Van Loan §5.1.8
    """
    cdef double tau, r
    if b == 0.0:
        c[0] = 1.0; s[0] = 0.0
    elif fabs(b) > fabs(a):
        tau = -a / b
        s[0] = 1.0 / sqrt(1.0 + tau*tau)
        c[0] = s[0] * tau
    else:
        tau = -b / a
        c[0] = 1.0 / sqrt(1.0 + tau*tau)
        s[0] = c[0] * tau


# ------------------------------------------------------------------ 전/후방 대입
cdef inline void solve_lower_unit(
    double* L,      # (n x n) C-contiguous 단위 하삼각
    double* b,      # 우변 (길이 n), 제자리에서 y로 변환
    int n
) nogil:
    """전방 대입: L y = b  (L 단위 하삼각, diag=1)"""
    cdef int i, k
    for i in range(n):
        for k in range(i):
            b[i] -= L[i*n + k] * b[k]
        # L[i,i] = 1: 나눗셈 불필요


cdef inline void solve_upper(
    double* U,      # (n x n) C-contiguous 상삼각
    double* b,      # 우변 (길이 n), 제자리에서 x로 변환
    int n
) nogil:
    """후방 대입: U x = b  (U 상삼각)"""
    cdef int i, k
    for i in range(n-1, -1, -1):
        for k in range(i+1, n):
            b[i] -= U[i*n + k] * b[k]
        b[i] /= U[i*n + i]
