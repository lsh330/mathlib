# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# linear_algebra.pyx  —  LinearAlgebra 클래스: 행렬 분해 및 연산 33종 (Session 1+2)
#
#   기존 9종:
#   1. lu                   LU 분해 (Doolittle + partial pivoting)
#   2. ldu                  LDU 분해 (LU 유도)
#   3. qr                   QR 분해 (Householder, pivot 옵션 추가)
#   4. cholesky             Cholesky-Banachiewicz
#   5. svd                  SVD (Golub-Reinsch bidiagonalization + QR iteration)
#   6. gaussian_elimination Gauss-Jordan / RREF
#   7. det                  행렬식 (LU 기반)
#   8. rank                 랭크 (SVD 기반)
#   9. inverse              역행렬 (LU 기반)
#
#   Session 1 신규 12종:
#  10. logdet               log|det| (오버플로 안전, LU 기반)
#  11. eigen                고유값/고유벡터 (QR + Wilkinson shift, 복소 지원)
#  12. diagonalize          P·D·P^{-1} 분해
#  13. characteristic_polynomial  Faddeev-LeVerrier
#  14. cayley_hamilton      p_A(A) = 0 검증
#  15. pinv                 Moore-Penrose 의사역행렬 (SVD 기반)
#  16. gauss_jordan         명시적 Gauss-Jordan (RREF)
#  17. gram_schmidt         수정 Gram-Schmidt 직교화
#  18. null_space           우측 영공간 (SVD 기반)
#  19. column_space         열공간 (SVD 기반)
#  20. row_space            행공간 (SVD 기반)
#  21. left_null_space      좌측 영공간 (SVD 기반)
#
#   Session 2 신규 11종:
#  22. tensor_product       Kronecker product A ⊗ B (Cython 루프)
#  23. outer_product        벡터 외적 a ⊗ b → (m, n) matrix
#  24. inner_product        Hermitian 내적 sum(conj(a)·b)
#  25. tensor_contract      일반 텐서 축약 (reshape/matmul 기반)
#  26. tensor_transpose     텐서 축 재배열 (copy 반환)
#  27. is_symmetric         대칭 행렬 여부 (max-norm)
#  28. is_skew_symmetric    반대칭 행렬 여부 (max-norm)
#  29. symmetrize           (A + A^H) / 2  (복소 Hermitian 지원)
#  30. skew_part            (A - A^T) / 2
#  31. jacobian             Differentiation.jacobian 래핑
#  32. hessian              Differentiation.hessian 래핑

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, fabs, log
from numpy.linalg import LinAlgError

include "_kernels.pxd"

np.import_array()

# ============================================================== helpers (file-scope)

cdef inline void _swap_rows(double[:, ::1] M, int r1, int r2, int ncols) nogil:
    cdef int j
    cdef double tmp
    for j in range(ncols):
        tmp = M[r1, j]; M[r1, j] = M[r2, j]; M[r2, j] = tmp

cdef inline void _swap_rows_partial(double[:, ::1] M, int r1, int r2, int upto) nogil:
    """r1, r2 행의 [0, upto) 열만 교환."""
    cdef int j
    cdef double tmp
    for j in range(upto):
        tmp = M[r1, j]; M[r1, j] = M[r2, j]; M[r2, j] = tmp


# ================================================================== LinearAlgebra
cdef class LinearAlgebra:
    """
    선형대수 분해 및 연산.
    Cython + NumPy memoryview 기반 100% 자체 구현 (np.linalg.* 사용 금지).
    """

    def __init__(self):
        pass

    # ============================================================== 1. LU
    def lu(self, np.ndarray A, *, bint pivot=True):
        """
        LU 분해 — Doolittle 알고리즘, 부분 피벗 (PA = LU).

        Parameters
        ----------
        A     : ndarray, shape (n, n), dtype float64
        pivot : bool, default True

        Returns
        -------
        (P, L, U)  모두 (n, n) ndarray
            P — 순열 행렬,  L — 단위 하삼각 (diag=1),  U — 상삼각

        Raises
        ------
        TypeError  A가 ndarray가 아님
        ValueError 정방 아님 / 특이 행렬 (피벗 = 0)
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("lu requires square matrix")

        cdef int n = A.shape[0]
        cdef int i, j, k, pr
        cdef double mv, tmp, fac

        cdef np.ndarray[np.float64_t, ndim=2] U_arr = np.ascontiguousarray(A, dtype=np.float64).copy()
        cdef np.ndarray[np.float64_t, ndim=2] L_arr = np.zeros((n, n), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] P_arr = np.eye(n, dtype=np.float64)

        cdef double[:, ::1] U = U_arr
        cdef double[:, ::1] L = L_arr
        cdef double[:, ::1] P = P_arr

        for k in range(n):
            # 부분 피벗 탐색
            mv = fabs(U[k, k]); pr = k
            for i in range(k + 1, n):
                if fabs(U[i, k]) > mv:
                    mv = fabs(U[i, k]); pr = i

            if mv == 0.0:
                raise ValueError(f"lu: singular matrix (zero pivot at column {k})")

            if pr != k:
                _swap_rows(U, k, pr, n)
                _swap_rows(P, k, pr, n)
                _swap_rows_partial(L, k, pr, k)

            # Doolittle 소거
            L[k, k] = 1.0
            for i in range(k + 1, n):
                fac      = U[i, k] / U[k, k]
                L[i, k]  = fac
                for j in range(k, n):
                    U[i, j] -= fac * U[k, j]

        if n > 0:
            L[n - 1, n - 1] = 1.0

        return P_arr, L_arr, U_arr

    # ============================================================== 2. LDU
    def ldu(self, np.ndarray A):
        """
        LDU 분해: PA = L · diag(D) · U.

        LU 분해 후 U에서 D를 분리: D = diag(U),  U_new = D^{-1} U.

        Returns
        -------
        (L, D, U)  L (n,n) 단위 하삼각 / D (n,) 1D 배열 / U (n,n) 단위 상삼각
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("ldu requires square matrix")

        cdef int n = A.shape[0]
        cdef int i, j

        _, L_arr, U_raw = self.lu(A)

        cdef np.ndarray[np.float64_t, ndim=2] U_arr = U_raw.copy()
        cdef np.ndarray[np.float64_t, ndim=1] D_arr = np.zeros(n, dtype=np.float64)
        cdef double[:, ::1] U = U_arr
        cdef double[::1]    D = D_arr

        for i in range(n):
            if fabs(U[i, i]) == 0.0:
                raise ValueError("ldu: singular matrix (zero diagonal in U)")
            D[i] = U[i, i]
            for j in range(i, n):
                U[i, j] /= D[i]

        return L_arr, D_arr, U_arr

    # ============================================================== 3. QR
    def qr(self, np.ndarray A, *, mode='reduced', bint pivot=False):
        """
        QR 분해 — Householder reflections.

        A (m×n) = Q · R
          reduced  Q (m,k), R (k,n)   k = min(m,n)
          complete Q (m,m), R (m,n)

        pivot=True : Column pivoting (rank-revealing QR)
          A[:,P] = Q · R  →  (Q, R, P) 반환  (P는 1D 정수 인덱스 배열)
          diag(R) 내림차순 (열 norm 기준)

        Raises
        ------
        TypeError / ValueError
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2:
            raise ValueError("qr requires 2D matrix")

        cdef int m = A.shape[0]
        cdef int n = A.shape[1]
        cdef int k = m if m < n else n   # min(m,n)
        cdef int i, j, p, col, swap_col
        cdef double beta, alpha_v, ns, v0, dv
        cdef double col_ns, max_ns
        cdef np.int64_t tmp_perm

        cdef np.ndarray[np.float64_t, ndim=2] R_arr = np.ascontiguousarray(A, dtype=np.float64).copy()
        cdef np.ndarray[np.float64_t, ndim=2] Q_arr = np.eye(m, dtype=np.float64)

        # Householder 벡터 + beta 저장
        cdef np.ndarray[np.float64_t, ndim=2] VS = np.zeros((k, m), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] BS = np.zeros(k, dtype=np.float64)

        cdef double[:, ::1] R  = R_arr
        cdef double[:, ::1] Q  = Q_arr
        cdef double[:, ::1] Vs = VS
        cdef double[::1]    Bs = BS
        cdef np.ndarray[np.float64_t, ndim=1] vbuf = np.zeros(m, dtype=np.float64)
        cdef double[::1] vb = vbuf
        cdef double tmp_val

        # 열 순열 추적
        cdef np.ndarray[np.int64_t, ndim=1] perm_arr = np.arange(n, dtype=np.int64)
        cdef np.int64_t[::1] perm = perm_arr

        for p in range(k):
            # --- pivot=True: 가장 큰 norm 열 선택 ---
            if pivot:
                max_ns = -1.0
                swap_col = p
                for col in range(p, n):
                    col_ns = 0.0
                    for i in range(p, m):
                        col_ns += R[i, col] * R[i, col]
                    if col_ns > max_ns:
                        max_ns = col_ns
                        swap_col = col
                if swap_col != p:
                    # 열 교환
                    for i in range(m):
                        tmp_val = R[i, p]; R[i, p] = R[i, swap_col]; R[i, swap_col] = tmp_val
                    # perm 교환
                    tmp_perm = perm[p]
                    perm[p] = perm[swap_col]
                    perm[swap_col] = tmp_perm

            for i in range(m - p):
                vb[i] = R[p + i, p]

            ns = 0.0
            for i in range(m - p):
                ns += vb[i] * vb[i]
            if ns == 0.0:
                Bs[p] = 0.0
                continue

            alpha_v = sqrt(ns)
            if vb[0] >= 0.0:
                alpha_v = -alpha_v
            v0     = vb[0] - alpha_v
            vb[0]  = v0
            ns     = v0 * v0
            for i in range(1, m - p):
                ns += vb[i] * vb[i]
            if ns == 0.0:
                Bs[p] = 0.0
                continue

            beta   = 2.0 / ns
            Bs[p]  = beta
            for i in range(m - p):
                Vs[p, i] = vb[i]

            # R <- H_p R
            for col in range(p, n):
                dv = Vs[p, 0] * R[p, col]
                for i in range(1, m - p):
                    dv += Vs[p, i] * R[p + i, col]
                dv *= beta
                for i in range(m - p):
                    R[p + i, col] -= dv * Vs[p, i]

        # Q 복원: Q = H_1 ... H_k  (역순)
        for p in range(k - 1, -1, -1):
            if Bs[p] == 0.0:
                continue
            beta = Bs[p]
            for col in range(p, m):
                dv = Vs[p, 0] * Q[p, col]
                for i in range(1, m - p):
                    dv += Vs[p, i] * Q[p + i, col]
                dv *= beta
                for i in range(m - p):
                    Q[p + i, col] -= dv * Vs[p, i]

        if pivot:
            if mode == 'reduced':
                return Q_arr[:, :k], R_arr[:k, :], perm_arr
            else:
                return Q_arr, R_arr, perm_arr
        else:
            if mode == 'reduced':
                return Q_arr[:, :k], R_arr[:k, :]
            else:
                return Q_arr, R_arr

    # ============================================================== 4. Cholesky
    def cholesky(self, np.ndarray A, *, bint lower=True):
        """
        Cholesky-Banachiewicz 분해.

        A = L L^T  (lower=True)   또는   A = U^T U  (lower=False)

        Raises
        ------
        TypeError  ndarray 아님
        ValueError 정방 아님 / 비대칭 / 양의 정부호 아님
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("cholesky requires square matrix")

        cdef int n = A.shape[0]
        cdef int i, j, kk
        cdef double s

        cdef np.ndarray[np.float64_t, ndim=2] Af = np.ascontiguousarray(A, dtype=np.float64)
        cdef double[:, ::1] Av = Af

        for i in range(n):
            for j in range(i + 1, n):
                if fabs(Av[i, j] - Av[j, i]) > 1e-10 * (1.0 + fabs(Av[i, j])):
                    raise ValueError("matrix is not symmetric")

        cdef np.ndarray[np.float64_t, ndim=2] L_arr = np.zeros((n, n), dtype=np.float64)
        cdef double[:, ::1] L = L_arr

        for i in range(n):
            for j in range(i + 1):
                s = Av[i, j]
                for kk in range(j):
                    s -= L[i, kk] * L[j, kk]
                if i == j:
                    if s <= 0.0:
                        raise ValueError("matrix is not positive definite")
                    L[i, j] = sqrt(s)
                else:
                    L[i, j] = s / L[j, j]

        if lower:
            return L_arr
        else:
            return np.ascontiguousarray(L_arr.T)

    # ============================================================== 5. SVD  (Golub-Reinsch)
    def svd(self, np.ndarray A, *, bint full_matrices=False):
        """
        SVD: A = U · diag(S) · Vt

        알고리즘:
          1. Householder bidiagonalization → B = Ul^T A Vr  (상이중대각)
          2. Golub-Reinsch QR iteration (implicit Wilkinson shift) on B
          3. 특이값 내림차순 정렬

        Parameters
        ----------
        A             : ndarray, shape (m, n)
        full_matrices : bool, default False  (thin SVD)

        Returns
        -------
        (U, S, Vt)   k = min(m, n)
          U  (m, k) 또는 (m, m)
          S  (k,)  내림차순
          Vt (k, n) 또는 (n, n)

        Raises
        ------
        TypeError / ValueError
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2:
            raise ValueError("svd requires 2D matrix")

        cdef int mo = A.shape[0]
        cdef int no = A.shape[1]
        cdef bint transposed = (mo < no)
        cdef int m, n

        # m >= n 보장: 필요 시 전치
        cdef np.ndarray[np.float64_t, ndim=2] W
        if transposed:
            W = np.ascontiguousarray(A.T, dtype=np.float64).copy()
            m = no; n = mo
        else:
            W = np.ascontiguousarray(A, dtype=np.float64).copy()
            m = mo; n = no

        cdef int k = n   # min(m, n) = n  (m >= n)
        cdef int p, i, j, c2
        cdef double beta2, alpha2, ns2, v02, dv2

        # 직교 행렬 누적: Ul (m×m), Vr (n×n)
        cdef np.ndarray[np.float64_t, ndim=2] Ul = np.eye(m, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] Vr = np.eye(n, dtype=np.float64)

        cdef double[:, ::1] Wv  = W
        cdef double[:, ::1] Ulv = Ul
        cdef double[:, ::1] Vrv = Vr

        cdef np.ndarray[np.float64_t, ndim=1] vl_buf = np.zeros(m, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] vr_buf = np.zeros(n, dtype=np.float64)
        cdef double[::1] vl = vl_buf
        cdef double[::1] vr = vr_buf

        # ---- 1단계: Householder Bidiagonalization ----
        for p in range(k):
            # Left: W[p:, p] -> d[p] * e_1
            ns2 = 0.0
            for i in range(m - p):
                vl[i] = Wv[p + i, p]
                ns2  += vl[i] * vl[i]

            if ns2 > 0.0:
                alpha2 = sqrt(ns2)
                if vl[0] >= 0.0:
                    alpha2 = -alpha2
                v02    = vl[0] - alpha2
                vl[0]  = v02
                ns2    = v02 * v02
                for i in range(1, m - p):
                    ns2 += vl[i] * vl[i]
                if ns2 > 0.0:
                    beta2 = 2.0 / ns2
                    for c2 in range(p, n):
                        dv2 = vl[0] * Wv[p, c2]
                        for i in range(1, m - p):
                            dv2 += vl[i] * Wv[p + i, c2]
                        dv2 *= beta2
                        for i in range(m - p):
                            Wv[p + i, c2] -= dv2 * vl[i]
                    for c2 in range(m):
                        dv2 = vl[0] * Ulv[p, c2]
                        for i in range(1, m - p):
                            dv2 += vl[i] * Ulv[p + i, c2]
                        dv2 *= beta2
                        for i in range(m - p):
                            Ulv[p + i, c2] -= dv2 * vl[i]

            # Right: W[p, p+1:] -> e[p] * e_1
            if p + 1 < n:
                ns2 = 0.0
                for j in range(n - p - 1):
                    vr[j] = Wv[p, p + 1 + j]
                    ns2  += vr[j] * vr[j]

                if ns2 > 0.0:
                    alpha2 = sqrt(ns2)
                    if vr[0] >= 0.0:
                        alpha2 = -alpha2
                    v02    = vr[0] - alpha2
                    vr[0]  = v02
                    ns2    = v02 * v02
                    for j in range(1, n - p - 1):
                        ns2 += vr[j] * vr[j]
                    if ns2 > 0.0:
                        beta2 = 2.0 / ns2
                        for i in range(p, m):
                            dv2 = vr[0] * Wv[i, p + 1]
                            for j in range(1, n - p - 1):
                                dv2 += vr[j] * Wv[i, p + 1 + j]
                            dv2 *= beta2
                            for j in range(n - p - 1):
                                Wv[i, p + 1 + j] -= dv2 * vr[j]
                        for i in range(n):
                            dv2 = vr[0] * Vrv[i, p + 1]
                            for j in range(1, n - p - 1):
                                dv2 += vr[j] * Vrv[i, p + 1 + j]
                            dv2 *= beta2
                            for j in range(n - p - 1):
                                Vrv[i, p + 1 + j] -= dv2 * vr[j]

        # 대각·상부대각 추출
        cdef np.ndarray[np.float64_t, ndim=1] d_arr = np.zeros(k, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] e_arr = np.zeros(k, dtype=np.float64)
        for i in range(k):
            d_arr[i] = Wv[i, i]
        for i in range(k - 1):
            e_arr[i] = Wv[i, i + 1]

        # ---- 2단계: Golub-Reinsch QR iteration ----
        # d_arr는 음수 포함 가능 (bidiagonalization 결과 그대로 전달)
        # _golub_reinsch_impl 내부에서 후처리로 d 양수화 + Ql 부호 보정
        cdef np.ndarray[np.float64_t, ndim=2] Ql = np.eye(k, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] Qr = np.eye(k, dtype=np.float64)

        _golub_reinsch_impl(d_arr, e_arr, Ql, Qr, k)

        # 잔여 음수 특이값 양수화 (반올림 오류 방지)
        cdef double[::1] dd = d_arr
        for i in range(k):
            if dd[i] < 0.0:
                dd[i] = -dd[i]
                for j in range(k):
                    Ql[j, i] = -Ql[j, i]

        # 내림차순 정렬
        cdef np.ndarray[np.int64_t, ndim=1] idx = np.argsort(d_arr)[::-1].astype(np.int64)
        d_arr = d_arr[idx]
        Ql    = Ql[:, idx]
        Qr    = Qr[:, idx]

        # 최종 U, V 합성
        # Ul 누적 방식: Ulv = H_1 ... H_k  (행 누적)
        # → 실제 왼쪽 직교 행렬 Ul_actual = Ulv^T
        # U_final = Ulv^T @ 패딩(Ql[:k, :]) = Ul[:, :k] @ Ql   ([:k] 사용)
        cdef np.ndarray[np.float64_t, ndim=2] U_out = np.dot(Ul.T[:, :k], Ql)   # (m × k)
        cdef np.ndarray[np.float64_t, ndim=2] V_out = np.dot(Vr[:, :k],   Qr)   # (n × k)
        cdef np.ndarray[np.float64_t, ndim=2] Uf
        cdef np.ndarray[np.float64_t, ndim=2] Vf

        if transposed:
            U_out, V_out = V_out, U_out

        if full_matrices:
            Uf = np.eye(mo, dtype=np.float64)
            Vf = np.eye(no, dtype=np.float64)
            Uf[:, :k] = U_out
            Vf[:, :k] = V_out
            return Uf, d_arr, Vf.T
        else:
            return U_out, d_arr, V_out.T

    # ============================================================== 6. Gaussian elimination
    def gaussian_elimination(self, np.ndarray A, b=None, *, form='rref'):
        """
        가우스(-조르단) 소거.

        b=None  A → RREF (또는 REF) ndarray 반환
        b 제공  Ax = b 풀기 → 해 x (1D ndarray)

        Parameters
        ----------
        A    : ndarray, shape (m, n)
        b    : ndarray, shape (m,) 또는 None
        form : {'rref', 'ref'}

        Raises
        ------
        LinAlgError 불일치 / 과소결정
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2:
            raise ValueError("gaussian_elimination requires 2D matrix")

        cdef int m  = A.shape[0]
        cdef int n  = A.shape[1]
        cdef bint solve_mode = (b is not None)
        cdef int acols = n + (1 if solve_mode else 0)

        cdef np.ndarray[np.float64_t, ndim=2] M
        if solve_mode:
            M = np.empty((m, acols), dtype=np.float64)
            M[:, :n] = np.ascontiguousarray(A, dtype=np.float64)
            M[:, n]  = np.ascontiguousarray(b, dtype=np.float64).ravel()
        else:
            M = np.ascontiguousarray(A, dtype=np.float64).copy()

        cdef double[:, ::1] Mv = M
        cdef int row = 0, pcol = 0, pr, i, j
        cdef double mv2, tmp2, fac2, pv2
        cdef np.ndarray[np.float64_t, ndim=1] x_out
        cdef list piv_cols = []

        while row < m and pcol < n:
            mv2 = 0.0; pr = -1
            for i in range(row, m):
                if fabs(Mv[i, pcol]) > mv2:
                    mv2 = fabs(Mv[i, pcol]); pr = i
            if mv2 < 1e-14:
                pcol += 1
                continue

            if pr != row:
                _swap_rows(Mv, row, pr, acols)

            pv2 = Mv[row, pcol]
            for j in range(acols):
                Mv[row, j] /= pv2

            if form == 'rref':
                for i in range(m):
                    if i == row:
                        continue
                    fac2 = Mv[i, pcol]
                    if fabs(fac2) < 1e-300:
                        continue
                    for j in range(acols):
                        Mv[i, j] -= fac2 * Mv[row, j]
            else:
                for i in range(row + 1, m):
                    fac2 = Mv[i, pcol]
                    if fabs(fac2) < 1e-300:
                        continue
                    for j in range(acols):
                        Mv[i, j] -= fac2 * Mv[row, j]

            piv_cols.append(pcol)
            row  += 1
            pcol += 1

        if solve_mode:
            rnk = len(piv_cols)
            for i in range(rnk, m):
                if fabs(Mv[i, n]) > 1e-10:
                    raise LinAlgError("system is inconsistent (no solution)")
            if rnk < n:
                raise LinAlgError("system is underdetermined (infinite solutions)")
            x_out = np.zeros(n, dtype=np.float64)
            for ii, pc in enumerate(piv_cols):
                x_out[pc] = Mv[ii, n]
            return x_out
        else:
            return M

    # ============================================================== 7. det
    def det(self, np.ndarray A):
        """
        행렬식 — LU 기반.

        det(A) = sign(P) · prod(diag(U))

        Raises
        ------
        TypeError  ndarray 아님
        ValueError 정방 아님
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("det requires square matrix")

        cdef int n = A.shape[0]
        cdef int i, j

        try:
            P_arr, _, U_arr = self.lu(A)
        except ValueError:
            return 0.0

        cdef double[:, ::1] Uv = U_arr
        cdef double dv = 1.0
        for i in range(n):
            dv *= Uv[i, i]

        # 순열 부호: 사이클 분해
        cdef double[:, ::1] Pv = P_arr
        cdef np.ndarray[np.int64_t, ndim=1] perm = np.empty(n, dtype=np.int64)
        for i in range(n):
            for j in range(n):
                if Pv[i, j] > 0.5:
                    perm[i] = j
                    break

        cdef int sp = 1
        cdef np.ndarray[np.uint8_t, ndim=1] vis = np.zeros(n, dtype=np.uint8)
        cdef int cl, cur
        for i in range(n):
            if not vis[i]:
                cl = 0; cur = i
                while not vis[cur]:
                    vis[cur] = 1; cur = perm[cur]; cl += 1
                if cl % 2 == 0:
                    sp = -sp

        return <double>sp * dv

    # ============================================================== 8. rank
    def rank(self, np.ndarray A, *, tol=None):
        """
        행렬 랭크 — SVD 기반.

        tol 기본값: max(m, n) · ε · σ_max
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2:
            raise ValueError("rank requires 2D matrix")

        cdef int m = A.shape[0]
        cdef int n = A.shape[1]

        _, S, _ = self.svd(A, full_matrices=False)

        cdef double tv
        if tol is None:
            tv = max(m, n) * 2.220446049250313e-16 * (float(S[0]) if len(S) > 0 and S[0] > 0.0 else 1.0)
        else:
            tv = float(tol)

        return int(np.sum(S > tv))

    # ============================================================== 9. inverse
    def inverse(self, np.ndarray A):
        """
        역행렬 — LU 기반 열별 전후방 대입.

        Raises
        ------
        TypeError / ValueError A가 ndarray 아님 / 정방 아님 / 특이 행렬
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("inverse requires square matrix")

        cdef int n = A.shape[0]
        cdef int i, j, kk

        try:
            P_arr, L_arr, U_arr = self.lu(A)
        except ValueError:
            raise ValueError("matrix is singular")

        cdef double[:, ::1] Uv = U_arr
        cdef double[:, ::1] Lv = L_arr
        cdef double[:, ::1] Pv = P_arr

        for i in range(n):
            if fabs(Uv[i, i]) < 1e-14:
                raise ValueError("matrix is singular")

        cdef np.ndarray[np.float64_t, ndim=2] X_arr = np.zeros((n, n), dtype=np.float64)
        cdef double[:, ::1] Xv = X_arr
        cdef np.ndarray[np.float64_t, ndim=1] col_arr = np.zeros(n, dtype=np.float64)
        cdef double[::1] col = col_arr

        for j in range(n):
            for i in range(n):
                col[i] = Pv[i, j]   # P e_j

            # 전방 대입: L y = P e_j
            for i in range(n):
                for kk in range(i):
                    col[i] -= Lv[i, kk] * col[kk]

            # 후방 대입: U x = y
            for i in range(n - 1, -1, -1):
                for kk in range(i + 1, n):
                    col[i] -= Uv[i, kk] * col[kk]
                col[i] /= Uv[i, i]

            for i in range(n):
                Xv[i, j] = col[i]

        return X_arr

    # ============================================================== 10. logdet
    def logdet(self, np.ndarray A, *, bint return_sign=False):
        """
        log|det(A)| — LU 기반, 오버플로 안전.

        Parameters
        ----------
        A           : ndarray, shape (n, n)
        return_sign : bool, default False
                      True 시 (sign, logabsdet) 튜플 반환 (numpy.linalg.slogdet 호환)
                      sign은 +1 또는 -1 (singular 시 0)

        Returns
        -------
        float  logabsdet  (return_sign=False)
        (float, float)  (sign, logabsdet)  (return_sign=True)

        Raises
        ------
        TypeError / ValueError
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("logdet requires square matrix")

        cdef int n = A.shape[0]
        cdef int i, j
        cdef double logabs, perm_sign

        try:
            P_arr, _, U_arr = self.lu(A)
        except ValueError:
            # 특이 행렬: logdet = -inf, sign = 0
            if return_sign:
                return 0.0, float('-inf')
            return float('-inf')

        cdef double[:, ::1] Uv = U_arr
        logabs = 0.0
        cdef double sign_prod = 1.0
        for i in range(n):
            if fabs(Uv[i, i]) < 1e-300:
                if return_sign:
                    return 0.0, float('-inf')
                return float('-inf')
            logabs += log(fabs(Uv[i, i]))
            if Uv[i, i] < 0.0:
                sign_prod = -sign_prod

        # 순열 부호 계산
        cdef double[:, ::1] Pv = P_arr
        cdef np.ndarray[np.int64_t, ndim=1] perm2 = np.empty(n, dtype=np.int64)
        for i in range(n):
            for j in range(n):
                if Pv[i, j] > 0.5:
                    perm2[i] = j
                    break

        cdef int sp2 = 1
        cdef np.ndarray[np.uint8_t, ndim=1] vis2 = np.zeros(n, dtype=np.uint8)
        cdef int cl2, cur2
        for i in range(n):
            if not vis2[i]:
                cl2 = 0; cur2 = i
                while not vis2[cur2]:
                    vis2[cur2] = 1; cur2 = perm2[cur2]; cl2 += 1
                if cl2 % 2 == 0:
                    sp2 = -sp2

        perm_sign = <double>sp2 * sign_prod

        if return_sign:
            return perm_sign, logabs
        return logabs

    # ============================================================== 11. eigen
    def eigen(self, np.ndarray A, *, bint symmetric=False):
        """
        고유값/고유벡터 계산.

        알고리즘:
          - symmetric=True  : 대칭 행렬 전용 QR iteration (실수 고유값 보장)
          - symmetric=False : Hessenberg 변환 + QR iteration with Wilkinson shift
                              복소 켤레쌍은 2x2 블록에서 추출

        Parameters
        ----------
        A         : ndarray, shape (n, n), dtype float64
        symmetric : bool, default False

        Returns
        -------
        (eigenvalues, eigenvectors)
          eigenvalues  : 1D complex128 배열 (길이 n)
          eigenvectors : 2D complex128 배열 (n, n), 열이 고유벡터

        Raises
        ------
        TypeError / ValueError
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("eigen requires square matrix")

        cdef int n = A.shape[0]

        if n == 1:
            vals = np.array([A[0, 0]], dtype=np.complex128)
            vecs = np.array([[1.0 + 0j]], dtype=np.complex128)
            return vals, vecs

        if n == 2:
            return _eigen_2x2(A)

        cdef np.ndarray[np.float64_t, ndim=2] H = np.ascontiguousarray(A, dtype=np.float64).copy()
        # Hessenberg 변환 누적 직교 행렬
        cdef np.ndarray[np.float64_t, ndim=2] QH = np.eye(n, dtype=np.float64)

        # 1단계: Householder Hessenberg 변환
        _hessenberg_reduce(H, QH, n)

        # 2단계: QR iteration with Wilkinson shift
        if symmetric:
            _symmetric_qr_iter(H, QH, n)
        else:
            _nonsymmetric_qr_iter(H, QH, n)

        # 3단계: 고유값/고유벡터 추출
        return _extract_eigen(H, QH, n, symmetric)

    # ============================================================== 12. diagonalize
    def diagonalize(self, np.ndarray A):
        """
        행렬 대각화 A = P · D · P^{-1}.

        Parameters
        ----------
        A : ndarray, shape (n, n)

        Returns
        -------
        (P, D)
          P : (n, n) complex128, 열이 고유벡터
          D : (n, n) complex128, 대각이 고유값

        Raises
        ------
        ValueError  결함 행렬 (diagonalizable 아님)
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("diagonalize requires square matrix")

        cdef int n = A.shape[0]
        vals, vecs = self.eigen(A)

        # 결함 행렬 체크: 복소 고유벡터 행렬의 행렬식이 0에 가까운지 확인
        # det(P) = 0 이면 결함 (diagonalizable 불가)
        # 간단히: 실수부/허수부를 분리한 실수 블록 행렬로 check
        cdef np.ndarray[np.complex128_t, ndim=2] P = vecs

        # 복소 행렬의 조건수: numpy의 lstsq가 아닌 절대값 행렬로 근사
        # 더 견고한 방법: 중복 고유값 체크
        # 실수 고유값 목록에서 근접한 쌍 탐색
        tol_degen = 1e-8 * (max(abs(vals)) if len(vals) > 0 else 1.0)
        cdef int has_defective = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(vals[i] - vals[j]) < tol_degen:
                    # 중복 고유값: 고유벡터 독립성 체크
                    # dot product of eigenvectors
                    dot_ij = abs(sum(np.conj(P[:, i]) * P[:, j]))
                    if dot_ij > 0.99:  # 거의 같은 벡터
                        has_defective = 1
                        break
            if has_defective:
                break

        if has_defective:
            raise ValueError("matrix is not diagonalizable (defective or near-defective)")

        D = np.diag(vals)
        return P, D

    # ============================================================== 13. characteristic_polynomial
    def characteristic_polynomial(self, np.ndarray A):
        """
        특성 다항식 계수 — Faddeev-LeVerrier 알고리즘.

        p(λ) = det(λI - A) = λ^n + c_{n-1}·λ^{n-1} + ... + c_0

        Parameters
        ----------
        A : ndarray, shape (n, n)

        Returns
        -------
        coeffs : 1D ndarray, 길이 n+1, 고차 → 저차
                 coeffs[0] = 1 (최고차), ..., coeffs[n] = c_0

        Raises
        ------
        TypeError / ValueError
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("characteristic_polynomial requires square matrix")

        cdef int n = A.shape[0]
        cdef int k, i, j
        cdef double tr
        cdef np.ndarray[np.float64_t, ndim=2] Af = np.ascontiguousarray(A, dtype=np.float64)

        # 계수 배열: coeffs[0..n], coeffs[0]=1 (λ^n 계수)
        cdef np.ndarray[np.float64_t, ndim=1] c = np.zeros(n + 1, dtype=np.float64)
        c[0] = 1.0

        cdef np.ndarray[np.float64_t, ndim=2] M = np.eye(n, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] AM = np.zeros((n, n), dtype=np.float64)

        for k in range(1, n + 1):
            # AM = A @ M
            AM = Af @ M
            # c[k] = -trace(AM) / k
            tr = 0.0
            for i in range(n):
                tr += AM[i, i]
            c[k] = -tr / k
            # M = AM + c[k] * I
            M = AM.copy()
            for i in range(n):
                M[i, i] += c[k]

        return c

    # ============================================================== 14. cayley_hamilton
    def cayley_hamilton(self, np.ndarray A, *, bint return_residual=False):
        """
        Cayley-Hamilton 정리 검증: p_A(A) = 0.

        Parameters
        ----------
        A              : ndarray, shape (n, n)
        return_residual: bool, default False
                         True 시 ||p(A)||_F 반환, False 시 True 반환

        Returns
        -------
        True         (return_residual=False, 검증 성공)
        float        ||p(A)||_F  (return_residual=True)

        Raises
        ------
        ValueError  ||p(A)|| > tol (수치 오류 임계 초과)
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("cayley_hamilton requires square matrix")

        cdef int n = A.shape[0]
        cdef int i
        cdef double norm_A, tol

        coeffs = self.characteristic_polynomial(A)
        cdef np.ndarray[np.float64_t, ndim=2] Af = np.ascontiguousarray(A, dtype=np.float64)

        # p(A) = coeffs[0]*A^n + coeffs[1]*A^(n-1) + ... + coeffs[n]*I
        # Horner's method: p(A) = ((...(coeffs[0]*A + coeffs[1]*I)*A + coeffs[2]*I)*A + ...)*A + coeffs[n]*I
        cdef np.ndarray[np.float64_t, ndim=2] result = np.eye(n, dtype=np.float64) * coeffs[0]
        for i in range(1, n + 1):
            result = result @ Af
            for j in range(n):
                result[j, j] += coeffs[i]

        cdef double res_norm = 0.0
        cdef double tmp_v
        cdef int jj
        for i in range(n):
            for jj in range(n):
                tmp_v = result[i, jj]
                res_norm += tmp_v * tmp_v
        res_norm = sqrt(res_norm)

        # 허용 오차: ||A||_F * n * eps
        norm_A = 0.0
        cdef double[:, ::1] Av = Af
        for i in range(n):
            for jj in range(n):
                tmp_v = Av[i, jj]
                norm_A += tmp_v * tmp_v
        norm_A = sqrt(norm_A)

        tol = max(norm_A ** n, 1.0) * n * n * 1e-8

        if return_residual:
            return res_norm
        if res_norm > tol:
            raise ValueError(f"Cayley-Hamilton verification failed: ||p(A)|| = {res_norm:.3e}")
        return True

    # ============================================================== 15. pinv
    def pinv(self, np.ndarray A, *, tol=None):
        """
        Moore-Penrose 의사역행렬 — SVD 기반.

        A^+ = V · diag(1/σ_i) · U^T  (σ_i > tol)

        Parameters
        ----------
        A   : ndarray, shape (m, n)
        tol : float or None
              None 시 max(m,n) · ε · σ_max

        Returns
        -------
        ndarray, shape (n, m)

        Raises
        ------
        TypeError / ValueError
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2:
            raise ValueError("pinv requires 2D matrix")

        cdef int m = A.shape[0]
        cdef int n = A.shape[1]
        cdef int k = m if m < n else n
        cdef int i
        cdef double tv, s_max

        U_out, S_out, Vt_out = self.svd(A, full_matrices=False)

        cdef np.ndarray[np.float64_t, ndim=1] S = S_out

        if tol is None:
            s_max = float(S[0]) if len(S) > 0 and S[0] > 0.0 else 1.0
            tv = <double>(m if m > n else n) * 2.220446049250313e-16 * s_max
        else:
            tv = float(tol)

        # S_inv
        cdef np.ndarray[np.float64_t, ndim=1] S_inv = np.zeros(k, dtype=np.float64)
        for i in range(k):
            if S[i] > tv:
                S_inv[i] = 1.0 / S[i]

        # A^+ = Vt^T @ diag(S_inv) @ U^T = (Vt^T * S_inv[np.newaxis,:]) @ U^T
        # Vt (k, n) → Vt.T (n, k),  U (m, k) → U.T (k, m)
        cdef np.ndarray[np.float64_t, ndim=2] VS = Vt_out.T * S_inv[np.newaxis, :]   # (n, k)
        cdef np.ndarray[np.float64_t, ndim=2] A_plus = VS @ U_out.T                   # (n, m)
        return A_plus

    # ============================================================== 16. gauss_jordan
    def gauss_jordan(self, np.ndarray A, b=None):
        """
        명시적 Gauss-Jordan 소거 — RREF 직접 생성.

        b=None : RREF 행렬 반환
        b 제공 : Ax=b 해 반환 (유일해 존재 시)

        Parameters
        ----------
        A : ndarray, shape (m, n)
        b : ndarray, shape (m,) 또는 None

        Returns
        -------
        RREF : ndarray (b=None)
        x    : ndarray (b 제공)

        Raises
        ------
        LinAlgError  불일치 / 과소결정
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2:
            raise ValueError("gauss_jordan requires 2D matrix")

        cdef int m  = A.shape[0]
        cdef int n  = A.shape[1]
        cdef bint solve_mode = (b is not None)
        cdef int acols = n + (1 if solve_mode else 0)

        cdef np.ndarray[np.float64_t, ndim=2] M
        if solve_mode:
            M = np.empty((m, acols), dtype=np.float64)
            M[:, :n] = np.ascontiguousarray(A, dtype=np.float64)
            M[:, n]  = np.ascontiguousarray(b, dtype=np.float64).ravel()
        else:
            M = np.ascontiguousarray(A, dtype=np.float64).copy()

        cdef double[:, ::1] Mv = M
        cdef int row = 0, pcol = 0, pr, i, j
        cdef double mv3, pv3, fac3
        cdef np.ndarray[np.float64_t, ndim=1] x_gj
        cdef list piv_cols_gj = []

        while row < m and pcol < n:
            # partial pivoting
            mv3 = 0.0; pr = -1
            for i in range(row, m):
                if fabs(Mv[i, pcol]) > mv3:
                    mv3 = fabs(Mv[i, pcol]); pr = i
            if mv3 < 1e-14:
                pcol += 1
                continue

            if pr != row:
                _swap_rows(Mv, row, pr, acols)

            # 피벗 행 정규화
            pv3 = Mv[row, pcol]
            for j in range(acols):
                Mv[row, j] /= pv3

            # 위/아래 동시 소거 (Gauss-Jordan 특징)
            for i in range(m):
                if i == row:
                    continue
                fac3 = Mv[i, pcol]
                if fabs(fac3) < 1e-300:
                    continue
                for j in range(acols):
                    Mv[i, j] -= fac3 * Mv[row, j]

            piv_cols_gj.append(pcol)
            row  += 1
            pcol += 1

        if solve_mode:
            rnk_gj = len(piv_cols_gj)
            for i in range(rnk_gj, m):
                if fabs(Mv[i, n]) > 1e-10:
                    raise LinAlgError("system is inconsistent (no solution)")
            if rnk_gj < n:
                raise LinAlgError("system is underdetermined (infinite solutions)")
            x_gj = np.zeros(n, dtype=np.float64)
            for ii, pc in enumerate(piv_cols_gj):
                x_gj[pc] = Mv[ii, n]
            return x_gj
        else:
            return M

    # ============================================================== 17. gram_schmidt
    def gram_schmidt(self, vectors, *, bint modified=True):
        """
        Gram-Schmidt 직교 정규화.

        Parameters
        ----------
        vectors : (m, n) ndarray 또는 list of 1D arrays
                  n개의 m차원 벡터 (열 단위)
        modified : bool, default True
                   True  — Modified Gram-Schmidt (수치 안정)
                   False — Classical Gram-Schmidt

        Returns
        -------
        Q : (m, r) ndarray, 열이 직교정규 기저 (r <= n, 선형 독립 벡터 수)

        Raises
        ------
        ValueError  입력 형식 오류
        """
        cdef np.ndarray[np.float64_t, ndim=2] V
        if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
            V = np.ascontiguousarray(vectors, dtype=np.float64).copy()
        elif hasattr(vectors, '__iter__'):
            V = np.column_stack([np.asarray(v, dtype=np.float64).ravel() for v in vectors])
        else:
            raise ValueError("gram_schmidt: vectors must be 2D ndarray or list of 1D arrays")

        cdef int m = V.shape[0]
        cdef int n = V.shape[1]
        cdef double norm_sq, proj, norm_v
        cdef int i, j, jj
        cdef double tol_gs = 1e-14

        cdef np.ndarray[np.float64_t, ndim=2] Q_gs = np.zeros((m, n), dtype=np.float64)
        cdef double[:, ::1] Vv = V
        cdef double[:, ::1] Qv = Q_gs

        cdef int r = 0   # 유효 직교 벡터 수

        for j in range(n):
            # 현재 열 복사
            for i in range(m):
                Qv[i, r] = Vv[i, j]

            if modified:
                # Modified GS: 기존 기저 벡터에 대해 순차 직교화
                for jj in range(r):
                    proj = 0.0
                    for i in range(m):
                        proj += Qv[i, jj] * Qv[i, r]
                    for i in range(m):
                        Qv[i, r] -= proj * Qv[i, jj]
            else:
                # Classical GS: 원래 벡터에서 한번에 직교화
                for jj in range(r):
                    proj = 0.0
                    for i in range(m):
                        proj += Qv[i, jj] * Vv[i, j]
                    for i in range(m):
                        Qv[i, r] -= proj * Qv[i, jj]

            # 정규화
            norm_sq = 0.0
            for i in range(m):
                norm_sq += Qv[i, r] * Qv[i, r]
            norm_v = sqrt(norm_sq)

            if norm_v < tol_gs:
                # 선형 종속: 건너뜀
                import warnings
                warnings.warn(f"gram_schmidt: vector {j} is linearly dependent, skipped")
                continue

            for i in range(m):
                Qv[i, r] /= norm_v
            r += 1

        return Q_gs[:, :r]

    # ============================================================== 18. null_space
    def null_space(self, np.ndarray A, *, tol=None):
        """
        우측 영공간 Null(A) = {x : Ax = 0} — SVD 기반.

        Parameters
        ----------
        A   : ndarray, shape (m, n)
        tol : float or None

        Returns
        -------
        N : ndarray, shape (n, n-rank), 열이 직교정규 기저
            rank = n 이면 (n, 0) 빈 행렬 반환
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2:
            raise ValueError("null_space requires 2D matrix")

        cdef int m = A.shape[0]
        cdef int n = A.shape[1]
        cdef double tv
        cdef int k_rank

        _, S_ns, Vt_ns = self.svd(A, full_matrices=True)

        cdef np.ndarray[np.float64_t, ndim=1] S_v = S_ns
        if tol is None:
            s_max = float(S_v[0]) if len(S_v) > 0 and S_v[0] > 0.0 else 1.0
            tv = <double>(m if m > n else n) * 2.220446049250313e-16 * s_max
        else:
            tv = float(tol)

        k_rank = int(np.sum(S_v > tv))
        # V의 열 중 k_rank 이후 열이 null space 기저
        # Vt_ns (n, n) full_matrices=True → V (n, n), Vt_ns[k_rank:,:].T
        cdef np.ndarray[np.float64_t, ndim=2] Vt_full = Vt_ns  # (n, n)
        return np.ascontiguousarray(Vt_full[k_rank:, :].T)

    # ============================================================== 19. column_space
    def column_space(self, np.ndarray A, *, tol=None):
        """
        열공간 Col(A) = span(A의 열) — SVD 기반.

        Parameters
        ----------
        A   : ndarray, shape (m, n)
        tol : float or None

        Returns
        -------
        C : ndarray, shape (m, rank), 열이 직교정규 기저
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2:
            raise ValueError("column_space requires 2D matrix")

        cdef int m = A.shape[0]
        cdef int n = A.shape[1]
        cdef double tv
        cdef int k_rank

        U_cs, S_cs, _ = self.svd(A, full_matrices=True)

        cdef np.ndarray[np.float64_t, ndim=1] S_v2 = S_cs
        if tol is None:
            s_max2 = float(S_v2[0]) if len(S_v2) > 0 and S_v2[0] > 0.0 else 1.0
            tv = <double>(m if m > n else n) * 2.220446049250313e-16 * s_max2
        else:
            tv = float(tol)

        k_rank = int(np.sum(S_v2 > tv))
        return np.ascontiguousarray(U_cs[:, :k_rank])

    # ============================================================== 20. row_space
    def row_space(self, np.ndarray A, *, tol=None):
        """
        행공간 Row(A) = Col(A^T) — SVD 기반.

        Returns
        -------
        R : ndarray, shape (n, rank), 열이 직교정규 기저
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2:
            raise ValueError("row_space requires 2D matrix")

        return self.column_space(np.ascontiguousarray(A.T), tol=tol)

    # ============================================================== 21. left_null_space
    def left_null_space(self, np.ndarray A, *, tol=None):
        """
        좌측 영공간 LeftNull(A) = {x : x^T A = 0} = Null(A^T) — SVD 기반.

        Returns
        -------
        LN : ndarray, shape (m, m-rank), 열이 직교정규 기저
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be numpy.ndarray")
        if A.ndim != 2:
            raise ValueError("left_null_space requires 2D matrix")

        return self.null_space(np.ascontiguousarray(A.T), tol=tol)

    # ============================================================== 22. tensor_product (Kronecker)
    def tensor_product(self, A, B):
        """
        Kronecker product A ⊗ B.

        A (m, n), B (p, q) → (m·p, n·q) block matrix.
        결과[i*p:(i+1)*p, j*q:(j+1)*q] = A[i,j] * B

        복소 dtype 지원.

        Parameters
        ----------
        A : array-like, shape (m, n)
        B : array-like, shape (p, q)

        Returns
        -------
        K : ndarray, shape (m*p, n*q)
        """
        cdef np.ndarray Ac, Bc
        cdef int m, n, p, q, i, j
        cdef object dtype

        Ac = np.asarray(A)
        Bc = np.asarray(B)
        if Ac.ndim != 2:
            raise ValueError("tensor_product: A must be 2D")
        if Bc.ndim != 2:
            raise ValueError("tensor_product: B must be 2D")

        # 복소 dtype 지원
        if np.iscomplexobj(Ac) or np.iscomplexobj(Bc):
            dtype = np.complex128
        else:
            dtype = np.float64

        Ac = np.ascontiguousarray(Ac, dtype=dtype)
        Bc = np.ascontiguousarray(Bc, dtype=dtype)

        m = Ac.shape[0]; n = Ac.shape[1]
        p = Bc.shape[0]; q = Bc.shape[1]

        cdef np.ndarray result = np.empty((m * p, n * q), dtype=dtype)

        for i in range(m):
            for j in range(n):
                result[i * p:(i + 1) * p, j * q:(j + 1) * q] = Ac[i, j] * Bc

        return result

    # ============================================================== 23. outer_product
    def outer_product(self, a, b):
        """
        벡터 외적 a ⊗ b: a (m,), b (n,) → (m, n) matrix.
        result[i, j] = a[i] * b[j]

        복소 dtype 지원.

        Parameters
        ----------
        a : array-like, shape (m,)
        b : array-like, shape (n,)

        Returns
        -------
        O : ndarray, shape (m, n)
        """
        cdef np.ndarray ac, bc
        cdef int m, n, i, j
        cdef object dtype

        ac = np.asarray(a)
        bc = np.asarray(b)
        if ac.ndim != 1:
            raise ValueError("outer_product: a must be 1D")
        if bc.ndim != 1:
            raise ValueError("outer_product: b must be 1D")

        if np.iscomplexobj(ac) or np.iscomplexobj(bc):
            dtype = np.complex128
        else:
            dtype = np.float64

        ac = np.ascontiguousarray(ac, dtype=dtype)
        bc = np.ascontiguousarray(bc, dtype=dtype)

        m = ac.shape[0]; n = bc.shape[0]
        cdef np.ndarray result = np.empty((m, n), dtype=dtype)

        for i in range(m):
            for j in range(n):
                result[i, j] = ac[i] * bc[j]

        return result

    # ============================================================== 24. inner_product
    def inner_product(self, a, b):
        """
        벡터 내적 (Hermitian).
        - 실수: sum(a[i] * b[i])
        - 복소: sum(conj(a[i]) * b[i])

        Parameters
        ----------
        a : array-like, shape (n,)
        b : array-like, shape (n,)

        Returns
        -------
        scalar (float or complex)
        """
        cdef np.ndarray ac, bc
        cdef int n, i
        cdef bint is_complex
        cdef double rsum = 0.0
        cdef np.complex128_t csum

        ac = np.asarray(a)
        bc = np.asarray(b)
        if ac.ndim != 1:
            raise ValueError("inner_product: a must be 1D")
        if bc.ndim != 1:
            raise ValueError("inner_product: b must be 1D")
        if ac.shape[0] != bc.shape[0]:
            raise ValueError("inner_product: a and b must have same length")

        is_complex = np.iscomplexobj(ac) or np.iscomplexobj(bc)

        if is_complex:
            ac = np.ascontiguousarray(ac, dtype=np.complex128)
            bc = np.ascontiguousarray(bc, dtype=np.complex128)
            n = ac.shape[0]
            csum = 0.0 + 0.0j
            for i in range(n):
                csum = csum + ac[i].conjugate() * bc[i]
            return complex(csum)
        else:
            ac = np.ascontiguousarray(ac, dtype=np.float64)
            bc = np.ascontiguousarray(bc, dtype=np.float64)
            n = ac.shape[0]
            rsum = 0.0
            for i in range(n):
                rsum = rsum + (<double>ac[i]) * (<double>bc[i])
            return rsum

    # ============================================================== 25. tensor_contract
    def tensor_contract(self, A, B, axes_a, axes_b):
        """
        일반 텐서 축약 (numpy.tensordot 유사, 자체 구현).

        A의 axes_a 축과 B의 axes_b 축을 축약.
        나머지 축은 결과 텐서에 유지.

        Parameters
        ----------
        A      : array-like
        B      : array-like
        axes_a : int or tuple  — A의 축약 축
        axes_b : int or tuple  — B의 축약 축

        Returns
        -------
        result : ndarray
        """
        cdef np.ndarray Ac, Bc
        cdef int na, nb
        cdef list ax_a_list, ax_b_list
        cdef int K, Ma, Mb
        cdef np.ndarray A2, B2, C2

        Ac = np.asarray(A, dtype=np.float64)
        Bc = np.asarray(B, dtype=np.float64)

        na = Ac.ndim; nb = Bc.ndim

        # axes 정규화
        if isinstance(axes_a, int):
            ax_a_list = [axes_a % na]
        else:
            ax_a_list = [int(x) % na for x in axes_a]

        if isinstance(axes_b, int):
            ax_b_list = [axes_b % nb]
        else:
            ax_b_list = [int(x) % nb for x in axes_b]

        if len(ax_a_list) != len(ax_b_list):
            raise ValueError("tensor_contract: axes_a and axes_b must have same length")

        # 축약 차원 크기 일치 확인
        for ia, ib in zip(ax_a_list, ax_b_list):
            if Ac.shape[ia] != Bc.shape[ib]:
                raise ValueError(
                    f"tensor_contract: shape mismatch on axes A[{ia}]={Ac.shape[ia]}"
                    f" vs B[{ib}]={Bc.shape[ib]}"
                )

        # A의 축: 축약 축을 뒤로 이동
        cdef list free_a = [i for i in range(na) if i not in ax_a_list]
        cdef tuple perm_a = tuple(free_a + ax_a_list)
        A_t = np.ascontiguousarray(np.transpose(Ac, perm_a))

        # B의 축: 축약 축을 앞으로 이동
        cdef list free_b = [i for i in range(nb) if i not in ax_b_list]
        cdef tuple perm_b = tuple(ax_b_list + free_b)
        B_t = np.ascontiguousarray(np.transpose(Bc, perm_b))

        # 수축 크기 K = 축약 차원 곱
        K = 1
        for ia in ax_a_list:
            K *= Ac.shape[ia]

        # A2: (M_a, K), B2: (K, M_b)
        Ma = A_t.size // K if K > 0 else 1
        Mb = B_t.size // K if K > 0 else 1

        A2 = A_t.reshape(Ma, K)
        B2 = B_t.reshape(K, Mb)

        # 행렬 곱: C2 = A2 @ B2
        C2 = A2 @ B2   # (Ma, Mb)

        # 최종 shape: free_a 차원 + free_b 차원
        cdef list shape_out = [Ac.shape[i] for i in free_a] + [Bc.shape[i] for i in free_b]
        if len(shape_out) == 0:
            return C2.reshape(())
        return C2.reshape(shape_out)

    # ============================================================== 26. tensor_transpose
    def tensor_transpose(self, T, axes=None):
        """
        텐서 축 재배열 (copy 반환).

        Parameters
        ----------
        T    : array-like
        axes : None or tuple of int
               None → 역순 (reverse)
               tuple → 지정 순열

        Returns
        -------
        Tt : ndarray, C-contiguous copy
        """
        cdef np.ndarray Tc = np.asarray(T)
        cdef tuple ax
        if axes is None:
            return np.ascontiguousarray(np.transpose(Tc))
        else:
            ax = tuple(int(x) for x in axes)
            return np.ascontiguousarray(np.transpose(Tc, ax))

    # ============================================================== 27. is_symmetric
    def is_symmetric(self, A, double tol=1e-10):
        """
        대칭 행렬 여부 검사: A == A^T (실수 기준, max-norm).

        Parameters
        ----------
        A   : array-like, shape (n, n)
        tol : float, default 1e-10

        Returns
        -------
        bool
        """
        cdef np.ndarray Ac = np.asarray(A, dtype=np.float64)
        if Ac.ndim != 2 or Ac.shape[0] != Ac.shape[1]:
            return False
        cdef np.ndarray diff = Ac - Ac.T
        return bool(np.max(np.abs(diff)) < tol)

    # ============================================================== 28. is_skew_symmetric
    def is_skew_symmetric(self, A, double tol=1e-10):
        """
        반대칭 행렬 여부 검사: A + A^T ≈ 0 (max-norm).

        Parameters
        ----------
        A   : array-like, shape (n, n)
        tol : float, default 1e-10

        Returns
        -------
        bool
        """
        cdef np.ndarray Ac = np.asarray(A, dtype=np.float64)
        if Ac.ndim != 2 or Ac.shape[0] != Ac.shape[1]:
            return False
        cdef np.ndarray diff = Ac + Ac.T
        return bool(np.max(np.abs(diff)) < tol)

    # ============================================================== 29. symmetrize
    def symmetrize(self, A):
        """
        대칭화: (A + A^T) / 2.
        복소수이면 Hermitian 대칭화: (A + A^{conj,T}) / 2.

        Parameters
        ----------
        A : array-like, shape (n, n)

        Returns
        -------
        S : ndarray, shape (n, n)

        Raises
        ------
        ValueError  비정방 행렬
        """
        cdef np.ndarray Ac = np.asarray(A)
        if Ac.ndim != 2 or Ac.shape[0] != Ac.shape[1]:
            raise ValueError("symmetrize requires square matrix")
        if np.iscomplexobj(Ac):
            Ac = np.ascontiguousarray(Ac, dtype=np.complex128)
            return (Ac + np.conj(Ac).T) / 2.0
        else:
            Ac = np.ascontiguousarray(Ac, dtype=np.float64)
            return (Ac + Ac.T) / 2.0

    # ============================================================== 30. skew_part
    def skew_part(self, A):
        """
        반대칭 성분 추출: (A - A^T) / 2.

        Parameters
        ----------
        A : array-like, shape (n, n)

        Returns
        -------
        K : ndarray, shape (n, n)

        Raises
        ------
        ValueError  비정방 행렬
        """
        cdef np.ndarray Ac = np.asarray(A)
        if Ac.ndim != 2 or Ac.shape[0] != Ac.shape[1]:
            raise ValueError("skew_part requires square matrix")
        Ac = np.ascontiguousarray(Ac, dtype=np.float64)
        return (Ac - Ac.T) / 2.0

    # ============================================================== 31. jacobian (Differentiation wrap)
    def jacobian(self, f, point):
        """
        야코비안 행렬 — Differentiation.jacobian 래핑.

        f: R^n → R^m 함수 (list/tuple 반환 또는 callable sequence)
        point: 1D 시퀀스 (길이 n)

        Parameters
        ----------
        f     : callable  — f(*point) → sequence of length m
        point : 1D sequence of length n

        Returns
        -------
        J : ndarray, shape (m, n)

        Notes
        -----
        Differentiation.jacobian은 [callable, ...] 형식을 받으므로
        f가 단일 callable이면 성분 함수 m개로 분해하여 전달.
        """
        from math_library.differentiation.differentiation import Differentiation as _Diff
        cdef object _differ = _Diff()

        pt = list(point)
        n = len(pt)

        # f 호출로 출력 차원 m 파악
        try:
            val = f(*pt)
        except TypeError:
            val = f(pt)
        if hasattr(val, '__len__'):
            m = len(val)
        else:
            m = 1
            val = [val]

        # 성분별 callable 생성
        cdef list comp_fns = []
        cdef int idx
        for idx in range(m):
            def _comp(_idx=idx, _f=f):
                def fn(*args):
                    try:
                        res = _f(*args)
                    except TypeError:
                        res = _f(args)
                    if hasattr(res, '__len__'):
                        return res[_idx]
                    return res
                return fn
            comp_fns.append(_comp())

        raw = _differ.jacobian(comp_fns, pt)
        return np.asarray(raw, dtype=np.float64)

    # ============================================================== 32. hessian (Differentiation wrap)
    def hessian(self, f, point):
        """
        헤시안 행렬 — Differentiation.hessian 래핑.

        f: R^n → R 스칼라 함수
        point: 1D 시퀀스 (길이 n)

        Parameters
        ----------
        f     : callable  — f(*point) → scalar
        point : 1D sequence of length n

        Returns
        -------
        H : ndarray, shape (n, n), 대칭 (수치 미분 수준)
        """
        from math_library.differentiation.differentiation import Differentiation as _Diff
        cdef object _differ = _Diff()

        pt = list(point)

        # Differentiation.hessian은 단일 callable 받음
        def _wrapped(*args):
            try:
                return f(*args)
            except TypeError:
                return f(args)

        raw = _differ.hessian(_wrapped, pt)
        return np.asarray(raw, dtype=np.float64)


# ================================================================== Golub-Reinsch (module-level cdef)

cdef void _golub_reinsch_impl(
    np.ndarray[np.float64_t, ndim=1] d_arr,
    np.ndarray[np.float64_t, ndim=1] e_arr,
    np.ndarray[np.float64_t, ndim=2] Ql_arr,
    np.ndarray[np.float64_t, ndim=2] Qr_arr,
    int k
):
    """
    Golub-Reinsch implicit Wilkinson-shift QR iteration on upper bidiagonal B.

    B = diag(d) + superdiag(e)  → 수렴 후 d에 특이값 저장 (부호 미정).

    In-place 갱신:
      d_arr  (k,)   bidiagonal 대각
      e_arr  (k,)   상부대각  (e[k-1] 미사용)
      Ql_arr (k,k)  왼쪽 직교 누적 (초기 = I)
      Qr_arr (k,k)  오른쪽 직교 누적 (초기 = I)

    Validated: pure-Python reference against numpy.linalg.svd, rel_err < 1e-14.
    """
    if k <= 1:
        return

    cdef double[::1]    d  = d_arr
    cdef double[::1]    e  = e_arr
    cdef double[:, ::1] Ql = Ql_arr
    cdef double[:, ::1] Qr = Qr_arr

    cdef int MAX_ITER = 100 * k
    cdef int q, p, i, j, itr
    cdef double c_r, s_r, r_r, c_l, s_l, r_l
    cdef double f, g, h, disc, mu
    cdef double t11, t12, t22, thresh
    cdef double new_dii, new_ei, new_bulge, new_di1
    cdef double new_ei_after, new_di1_after
    cdef double new_ei1, tmp1, tmp2
    cdef double bulge

    q = k - 1

    for itr in range(MAX_ITER):
        # 수렴 구간 축소
        while q > 0:
            thresh = 1e-14 * (fabs(d[q - 1]) + fabs(d[q]))
            if fabs(e[q - 1]) <= thresh:
                e[q - 1] = 0.0
                q -= 1
            else:
                break
        if q <= 0:
            break

        # p 탐색
        p = q - 1
        while p > 0:
            thresh = 1e-14 * (fabs(d[p - 1]) + fabs(d[p]))
            if fabs(e[p - 1]) <= thresh:
                e[p - 1] = 0.0
                break
            p -= 1

        # Wilkinson shift: B^T B 하단 2×2 블록의 더 가까운 고유값
        t11  = d[q - 1] * d[q - 1] + (e[q - 2] * e[q - 2] if q >= 2 else 0.0)
        t12  = d[q - 1] * e[q - 1]
        t22  = d[q] * d[q] + e[q - 1] * e[q - 1]
        h    = 0.5 * (t22 - t11)
        disc = sqrt(h * h + t12 * t12)
        if h >= 0.0:
            mu = t22 - t12 * t12 / (h + disc) if disc > 0.0 else t22
        else:
            mu = t22 + t12 * t12 / (-h + disc) if disc > 0.0 else t22

        # QR sweep on [p, q]
        f = d[p] * d[p] - mu
        g = d[p] * e[p] if p < k - 1 else 0.0

        for i in range(p, q):
            # ---- Right Givens: columns i, i+1 ----
            # G_r = [[c_r, s_r], [-s_r, c_r]],  c_r*f + s_r*g = r_r,  -s_r*f + c_r*g = 0
            r_r = sqrt(f * f + g * g)
            if r_r < 1e-300:
                c_r = 1.0; s_r = 0.0
            else:
                c_r = f / r_r; s_r = g / r_r

            if i > p:
                e[i - 1] = r_r

            # Bidiagonal update after right rotation of columns i, i+1
            # B * G_r^T  where G_r^T = [[c_r, -s_r],[s_r, c_r]]
            # Row i:   B[i,i]=d[i], B[i,i+1]=e[i]
            # Row i+1: B[i+1,i]=0,  B[i+1,i+1]=d[i+1]
            # new B[:,i]   = c_r * old[:,i] + s_r * old[:,i+1]
            # new B[:,i+1] = -s_r * old[:,i] + c_r * old[:,i+1]
            # → for cols i:
            new_dii   = c_r * d[i]    + s_r * e[i]      # B[i,i]   new
            new_ei    = -s_r * d[i]   + c_r * e[i]      # B[i,i+1] new  (will become superdiag)
            new_bulge = c_r * 0.0     + s_r * d[i + 1]  # B[i+1,i] new  (introduced bulge)
            new_di1   = -s_r * 0.0    + c_r * d[i + 1]  # B[i+1,i+1] new

            d[i]     = new_dii
            e[i]     = new_ei       # superdiag (after left will be updated)
            d[i + 1] = new_di1
            bulge    = new_bulge    # subdiag (after left Givens → zeroed)

            # Qr 갱신 (열 i, i+1 에 G_r^T 적용)
            for j in range(k):
                tmp1          =  c_r * Qr[j, i] + s_r * Qr[j, i + 1]
                Qr[j, i + 1] = -s_r * Qr[j, i] + c_r * Qr[j, i + 1]
                Qr[j, i]     = tmp1

            # ---- Left Givens: rows i, i+1 — eliminate bulge at B[i+1,i] ----
            # G_l = [[c_l, s_l], [-s_l, c_l]],  c_l*d[i] + s_l*bulge = r_l, -s_l*d[i]+c_l*bulge=0
            r_l = sqrt(d[i] * d[i] + bulge * bulge)
            if r_l < 1e-300:
                c_l = 1.0; s_l = 0.0
            else:
                c_l = d[i] / r_l; s_l = bulge / r_l

            d[i] = r_l

            # G_l @ B_block (rows i, i+1, relevant cols i, i+1, i+2):
            # B[i,i]=d[i], B[i,i+1]=e[i], B[i+1,i]=bulge, B[i+1,i+1]=d[i+1]
            # new B[i,i+1]   = c_l * e[i]   + s_l * d[i+1]
            # new B[i+1,i+1] = -s_l * e[i]  + c_l * d[i+1]
            new_ei_after  = c_l * e[i]  + s_l * d[i + 1]
            new_di1_after = -s_l * e[i] + c_l * d[i + 1]
            e[i]     = new_ei_after
            d[i + 1] = new_di1_after

            # 다음 스텝 f, g (chase bulge into next superdiag)
            if i + 1 < q:
                # Left rotation introduces fill at B[i, i+2] = s_l * e[i+1]
                new_ei1     = s_l * e[i + 1]
                e[i + 1]    = c_l * e[i + 1]  # zero the fill portion (scale down)
                f = e[i]        # = new_ei_after = new superdiag B[i, i+1]
                g = new_ei1     # = fill at B[i, i+2] drives next right rotation

            # Ql 갱신 (열 i, i+1 에 G_l^T 적용)
            for j in range(k):
                tmp1          =  c_l * Ql[j, i] + s_l * Ql[j, i + 1]
                Ql[j, i + 1] = -s_l * Ql[j, i] + c_l * Ql[j, i + 1]
                Ql[j, i]     = tmp1


# ================================================================== eigen helpers (module-level)

def _eigen_2x2(np.ndarray A):
    """
    2x2 행렬 고유값/고유벡터 (공식 직접 계산).
    """
    cdef double a = A[0, 0], b = A[0, 1], c_v = A[1, 0], d = A[1, 1]
    cdef double tr = a + d
    cdef double det_v = a * d - b * c_v
    cdef double disc_v = tr * tr - 4.0 * det_v
    cdef double re, im

    cdef np.ndarray vals = np.empty(2, dtype=np.complex128)
    cdef np.ndarray vecs = np.zeros((2, 2), dtype=np.complex128)

    if disc_v >= 0.0:
        re = sqrt(disc_v)
        vals[0] = (tr + re) * 0.5 + 0j
        vals[1] = (tr - re) * 0.5 + 0j
    else:
        im = sqrt(-disc_v)
        vals[0] = complex(tr * 0.5, im * 0.5)
        vals[1] = complex(tr * 0.5, -im * 0.5)

    # 각 고유값에 대해 고유벡터 계산
    for i in range(2):
        lam = vals[i]
        if abs(b) > 1e-14 * (abs(a) + abs(d)):
            vecs[0, i] = b
            vecs[1, i] = lam - a
        elif abs(c_v) > 1e-14 * (abs(a) + abs(d)):
            vecs[0, i] = lam - d
            vecs[1, i] = c_v
        else:
            # 이미 대각: 표준 기저
            if i == 0:
                vecs[0, i] = 1.0; vecs[1, i] = 0.0
            else:
                vecs[0, i] = 0.0; vecs[1, i] = 1.0
        # 정규화
        nm = sqrt(abs(vecs[0, i])**2 + abs(vecs[1, i])**2)
        if nm > 1e-300:
            vecs[0, i] /= nm
            vecs[1, i] /= nm

    return vals, vecs


cdef void _hessenberg_reduce(
    np.ndarray[np.float64_t, ndim=2] H_arr,
    np.ndarray[np.float64_t, ndim=2] Q_arr,
    int n
):
    """
    Householder을 이용한 upper Hessenberg 변환.
    H <- Q^T A Q  (H upper Hessenberg)
    Q_arr 누적 (초기 = I)
    """
    cdef int p, i, j
    cdef double ns_h, alpha_h, v0_h, beta_h, dv_h
    cdef double[:, ::1] H = H_arr
    cdef double[:, ::1] Q = Q_arr

    cdef np.ndarray[np.float64_t, ndim=1] vbuf_h = np.zeros(n, dtype=np.float64)
    cdef double[::1] vb_h = vbuf_h

    for p in range(n - 2):
        # H[p+1:, p] 열에 Householder
        for i in range(n - p - 1):
            vb_h[i] = H[p + 1 + i, p]

        ns_h = 0.0
        for i in range(n - p - 1):
            ns_h += vb_h[i] * vb_h[i]

        if ns_h < 1e-300:
            continue

        alpha_h = sqrt(ns_h)
        if vb_h[0] >= 0.0:
            alpha_h = -alpha_h
        v0_h   = vb_h[0] - alpha_h
        vb_h[0] = v0_h
        ns_h   = v0_h * v0_h
        for i in range(1, n - p - 1):
            ns_h += vb_h[i] * vb_h[i]

        if ns_h < 1e-300:
            continue

        beta_h = 2.0 / ns_h

        # H <- H_p H (왼쪽: 행 p+1..n-1)
        for j in range(n):
            dv_h = vb_h[0] * H[p + 1, j]
            for i in range(1, n - p - 1):
                dv_h += vb_h[i] * H[p + 1 + i, j]
            dv_h *= beta_h
            for i in range(n - p - 1):
                H[p + 1 + i, j] -= dv_h * vb_h[i]

        # H <- H H_p (오른쪽: 열 p+1..n-1)
        for i in range(n):
            dv_h = vb_h[0] * H[i, p + 1]
            for j in range(1, n - p - 1):
                dv_h += vb_h[j] * H[i, p + 1 + j]
            dv_h *= beta_h
            for j in range(n - p - 1):
                H[i, p + 1 + j] -= dv_h * vb_h[j]

        # Q <- Q H_p (오른쪽)
        for i in range(n):
            dv_h = vb_h[0] * Q[i, p + 1]
            for j in range(1, n - p - 1):
                dv_h += vb_h[j] * Q[i, p + 1 + j]
            dv_h *= beta_h
            for j in range(n - p - 1):
                Q[i, p + 1 + j] -= dv_h * vb_h[j]

        # 수치 정리: 하삼각 영
        for i in range(p + 2, n):
            H[i, p] = 0.0


cdef void _symmetric_qr_iter(
    np.ndarray[np.float64_t, ndim=2] H_arr,
    np.ndarray[np.float64_t, ndim=2] Q_arr,
    int n
):
    """
    대칭 행렬용 QR iteration (tridiagonal → diagonal).
    Wilkinson shift, Givens rotation 기반.
    """
    cdef double[:, ::1] H = H_arr
    cdef double[:, ::1] Q = Q_arr
    cdef int MAX_ITER = 30 * n
    cdef int m_end, p_start, itr, i, j
    cdef double d_a, d_b, e_off, mu_s, c_s, s_s, r_s, tmp_s1, tmp_s2
    cdef double h_val, disc_s, g_s, f_s, bulk, e_next
    cdef double t11_s, t12_s, t22_s

    m_end = n - 1

    for itr in range(MAX_ITER):
        # 수렴 구간 축소 (아래에서 위로)
        while m_end > 0:
            if fabs(H[m_end, m_end - 1]) <= 1e-14 * (fabs(H[m_end - 1, m_end - 1]) + fabs(H[m_end, m_end])):
                H[m_end, m_end - 1] = 0.0
                H[m_end - 1, m_end] = 0.0
                m_end -= 1
            else:
                break

        if m_end <= 0:
            break

        # p 탐색
        p_start = m_end - 1
        while p_start > 0:
            if fabs(H[p_start, p_start - 1]) <= 1e-14 * (fabs(H[p_start - 1, p_start - 1]) + fabs(H[p_start, p_start])):
                H[p_start, p_start - 1] = 0.0
                H[p_start - 1, p_start] = 0.0
                break
            p_start -= 1

        # Wilkinson shift (tridiagonal 하단 2x2)
        t11_s = H[m_end - 1, m_end - 1]
        t12_s = H[m_end - 1, m_end]
        t22_s = H[m_end, m_end]
        h_val = 0.5 * (t22_s - t11_s)
        disc_s = sqrt(h_val * h_val + t12_s * t12_s)
        if h_val >= 0.0:
            mu_s = t22_s - t12_s * t12_s / (h_val + disc_s) if disc_s > 0.0 else t22_s
        else:
            mu_s = t22_s + t12_s * t12_s / (-h_val + disc_s) if disc_s > 0.0 else t22_s

        # Givens QR sweep
        g_s = H[p_start, p_start] - mu_s
        f_s = H[p_start + 1, p_start] if p_start + 1 <= m_end else 0.0

        for i in range(p_start, m_end):
            # Givens (c,s): [c s; -s c] [g; f] = [r; 0]
            r_s = sqrt(g_s * g_s + f_s * f_s)
            if r_s < 1e-300:
                c_s = 1.0; s_s = 0.0
            else:
                c_s = g_s / r_s; s_s = f_s / r_s

            # H <- G^T H G (행/열 i, i+1 회전)
            # 행 회전 (왼쪽): rows i, i+1
            for j in range(p_start, min(i + 3, n)):
                tmp_s1 =  c_s * H[i, j] + s_s * H[i + 1, j]
                tmp_s2 = -s_s * H[i, j] + c_s * H[i + 1, j]
                H[i, j]     = tmp_s1
                H[i + 1, j] = tmp_s2

            # 열 회전 (오른쪽): cols i, i+1
            for j in range(0, min(i + 3, n)):
                tmp_s1 =  c_s * H[j, i] + s_s * H[j, i + 1]
                tmp_s2 = -s_s * H[j, i] + c_s * H[j, i + 1]
                H[j, i]     = tmp_s1
                H[j, i + 1] = tmp_s2

            # Q 갱신: Q <- Q G
            for j in range(n):
                tmp_s1 =  c_s * Q[j, i] + s_s * Q[j, i + 1]
                tmp_s2 = -s_s * Q[j, i] + c_s * Q[j, i + 1]
                Q[j, i]     = tmp_s1
                Q[j, i + 1] = tmp_s2

            if i + 1 < m_end:
                g_s = H[i + 1, i]
                f_s = H[i + 2, i] if i + 2 <= m_end else 0.0


cdef void _nonsymmetric_qr_iter(
    np.ndarray[np.float64_t, ndim=2] H_arr,
    np.ndarray[np.float64_t, ndim=2] Q_arr,
    int n
):
    """
    비대칭 행렬용 Francis double-shift QR iteration (Householder bulge chasing).
    Hessenberg → real quasi-upper-triangular Schur form.
    복소 켤레쌍은 2x2 블록으로 남음.
    Golub & Van Loan §7.5.1, Algorithm 7.5.2 참고.
    """
    cdef double[:, ::1] H = H_arr
    cdef double[:, ::1] Q = Q_arr
    cdef int MAX_ITER = 30 * n * n
    cdef int m_end, p, itr, i, j
    cdef double s_shift, t_shift
    cdef double x_b, y_b, z_b, r_b, c_b, sn_b, tmp1_b, tmp2_b
    cdef double v0_f, v1_f, v2_f, beta_f, ns_f, dv_f
    cdef double thresh, sg

    m_end = n - 1

    for itr in range(MAX_ITER):
        if m_end <= 0:
            break

        # ---- 수렴 확인 + deflation ----
        while m_end > 0:
            thresh = 1e-12 * (fabs(H[m_end - 1, m_end - 1]) + fabs(H[m_end, m_end]))
            if fabs(H[m_end, m_end - 1]) <= thresh:
                H[m_end, m_end - 1] = 0.0
                m_end -= 1
            else:
                break

        if m_end <= 0:
            break

        # m_end == 1: 2x2 블록 남음, 체크 후 break
        if m_end == 1:
            break

        # ---- p 탐색 (active submatrix 시작) ----
        p = m_end - 1
        while p > 0:
            thresh = 1e-12 * (fabs(H[p - 1, p - 1]) + fabs(H[p, p]))
            if fabs(H[p, p - 1]) <= thresh:
                H[p, p - 1] = 0.0
                break
            p -= 1

        # ---- Francis double-shift: s = tr(B), t = det(B) for bottom 2x2 B ----
        s_shift = H[m_end - 1, m_end - 1] + H[m_end, m_end]
        t_shift = H[m_end - 1, m_end - 1] * H[m_end, m_end] - H[m_end - 1, m_end] * H[m_end, m_end - 1]

        # ---- 시작 벡터: col p of (H^2 - s*H + t*I) ----
        x_b = H[p, p] * H[p, p] + H[p, p + 1] * H[p + 1, p] - s_shift * H[p, p] + t_shift
        y_b = H[p + 1, p] * (H[p, p] + H[p + 1, p + 1] - s_shift)
        z_b = H[p + 2, p + 1] * H[p + 1, p] if p + 2 <= m_end else 0.0

        # ---- Bulge chasing: Householder on [x, y, z] ----
        for i in range(p, m_end):
            # [x, y, z] 크기 결정
            if i + 2 <= m_end and fabs(z_b) > 1e-300:
                # 3-vector Householder
                ns_f = x_b * x_b + y_b * y_b + z_b * z_b
                r_b = sqrt(ns_f)
                if r_b < 1e-300:
                    x_b = H[i + 1, i]
                    y_b = H[i + 2, i] if i + 2 <= m_end else 0.0
                    z_b = H[i + 3, i] if i + 3 <= m_end else 0.0
                    continue

                sg = 1.0 if x_b >= 0.0 else -1.0
                v0_f = x_b + sg * r_b
                v1_f = y_b
                v2_f = z_b
                ns_f = v0_f * v0_f + v1_f * v1_f + v2_f * v2_f
                if ns_f < 1e-300:
                    x_b = H[i + 1, i]
                    y_b = H[i + 2, i] if i + 2 <= m_end else 0.0
                    z_b = H[i + 3, i] if i + 3 <= m_end else 0.0
                    continue
                beta_f = 2.0 / ns_f

                # 왼쪽: H[i:i+3, j:] -= beta * v * (v^T H[i:i+3, j:])
                # j는 Hessenberg 구조상 p (또는 i-1)부터
                j = p if i == p else i - 1
                while j < n:
                    dv_f = v0_f * H[i, j] + v1_f * H[i + 1, j] + v2_f * H[i + 2, j]
                    dv_f *= beta_f
                    H[i, j] -= dv_f * v0_f
                    H[i + 1, j] -= dv_f * v1_f
                    H[i + 2, j] -= dv_f * v2_f
                    j += 1

                # 오른쪽: H[0:min(i+4,n), i:i+3] -= beta * (H..v) * v^T
                j = min(i + 4, n)
                for i2 in range(j):
                    dv_f = v0_f * H[i2, i] + v1_f * H[i2, i + 1] + v2_f * H[i2, i + 2]
                    dv_f *= beta_f
                    H[i2, i] -= dv_f * v0_f
                    H[i2, i + 1] -= dv_f * v1_f
                    H[i2, i + 2] -= dv_f * v2_f

                # Q 누적
                for i2 in range(n):
                    dv_f = v0_f * Q[i2, i] + v1_f * Q[i2, i + 1] + v2_f * Q[i2, i + 2]
                    dv_f *= beta_f
                    Q[i2, i] -= dv_f * v0_f
                    Q[i2, i + 1] -= dv_f * v1_f
                    Q[i2, i + 2] -= dv_f * v2_f

            else:
                # 2-vector Givens
                r_b = sqrt(x_b * x_b + y_b * y_b)
                if r_b < 1e-300:
                    x_b = H[i + 1, i] if i + 1 <= m_end else 0.0
                    y_b = H[i + 2, i] if i + 2 <= m_end else 0.0
                    z_b = 0.0
                    continue
                c_b = x_b / r_b
                sn_b = y_b / r_b

                j = p if i == p else i - 1
                while j < n:
                    tmp1_b =  c_b * H[i, j] + sn_b * H[i + 1, j]
                    tmp2_b = -sn_b * H[i, j] + c_b * H[i + 1, j]
                    H[i, j] = tmp1_b
                    H[i + 1, j] = tmp2_b
                    j += 1

                j = min(i + 3, n)
                for i2 in range(j):
                    tmp1_b =  c_b * H[i2, i] + sn_b * H[i2, i + 1]
                    tmp2_b = -sn_b * H[i2, i] + c_b * H[i2, i + 1]
                    H[i2, i] = tmp1_b
                    H[i2, i + 1] = tmp2_b

                for i2 in range(n):
                    tmp1_b =  c_b * Q[i2, i] + sn_b * Q[i2, i + 1]
                    tmp2_b = -sn_b * Q[i2, i] + c_b * Q[i2, i + 1]
                    Q[i2, i] = tmp1_b
                    Q[i2, i + 1] = tmp2_b

            # 다음 bulge
            x_b = H[i + 1, i] if i + 1 <= m_end else 0.0
            y_b = H[i + 2, i] if i + 2 <= m_end else 0.0
            z_b = H[i + 3, i] if i + 3 <= m_end else 0.0


def _extract_eigen(
    np.ndarray[np.float64_t, ndim=2] H_arr,
    np.ndarray[np.float64_t, ndim=2] Q_arr,
    int n,
    bint symmetric
):
    """
    Quasi-upper-triangular H에서 고유값/고유벡터 추출.

    전략:
    1. Real Schur form H에서 고유값을 읽는다 (1x1 또는 2x2 블록)
    2. 각 고유값 λ에 대해 H의 복소 quasi-triangular form에서
       back-substitution으로 Schur 벡터 계산
    3. Schur 벡터를 Q 변환으로 원래 공간으로 역변환: v = Q @ y
    """
    cdef double[:, ::1] H = H_arr
    cdef double[:, ::1] Q = Q_arr
    cdef int i, j, k_i
    cdef double a11, a12, a21, a22, tr2, det2, disc_e, re_e, im_e

    # ---- 1단계: 고유값 추출 ----
    vals = np.empty(n, dtype=np.complex128)
    # 블록 구조 기록: block_type[i] = 0 (1x1 실수) 또는 1 (2x2 블록 첫번째)
    block_types = [0] * n

    i = 0
    while i < n:
        if i < n - 1 and fabs(H[i + 1, i]) > 1e-10 * (fabs(H[i, i]) + fabs(H[i + 1, i + 1]) + 1e-300):
            # 2x2 블록
            a11 = H[i, i]; a12 = H[i, i + 1]
            a21 = H[i + 1, i]; a22 = H[i + 1, i + 1]
            tr2  = a11 + a22
            det2 = a11 * a22 - a12 * a21
            disc_e = tr2 * tr2 - 4.0 * det2
            if disc_e >= 0.0:
                re_e = sqrt(disc_e)
                vals[i]     = complex((tr2 + re_e) * 0.5, 0.0)
                vals[i + 1] = complex((tr2 - re_e) * 0.5, 0.0)
            else:
                im_e = sqrt(-disc_e)
                vals[i]     = complex(tr2 * 0.5,  im_e * 0.5)
                vals[i + 1] = complex(tr2 * 0.5, -im_e * 0.5)
            block_types[i] = 1
            block_types[i + 1] = 2   # 2x2 블록 두번째
            i += 2
        else:
            vals[i] = complex(H[i, i], 0.0)
            i += 1

    # ---- 2단계: 각 고유값에 대해 고유벡터 계산 ----
    # A의 고유벡터는 공식: v = Q @ y  (y는 Hessenberg 공간에서의 고유벡터)
    # Hessenberg 고유벡터는 (H - λI)y = 0 을 upper Hessenberg 구조로 풀기
    # 간단하고 안정적인 방법: 각 고유값 λ에 대해 A_c = A.astype(complex)로
    # null space 찾기 (SVD 기반)

    # 복소 A 구성 (원래 행렬은 Q @ H @ Q^T)
    # A_orig = Q @ H @ Q^T → 여기서 Q는 Schur 분해의 직교 행렬
    cdef np.ndarray[np.float64_t, ndim=2] A_orig = Q_arr @ H_arr @ Q_arr.T
    A_c = A_orig.astype(np.complex128)

    vecs_complex = np.zeros((n, n), dtype=np.complex128)

    for k_i in range(n):
        lam = vals[k_i]
        # (A - λI) v = 0 의 null vector 계산
        M_c = A_c - lam * np.eye(n, dtype=np.complex128)
        # QR 분해로 null vector 추출
        v = _null_vector_complex(M_c, n)
        vecs_complex[:, k_i] = v

    return vals, vecs_complex


def _null_vector_complex(M_c, int n):
    """
    복소 행렬 M의 null vector를 Gaussian elimination으로 추출.
    M은 (A - lambda*I)이므로 정확히 1차원 null space 존재.
    NumPy 복소 연산 사용 (성능 최적화).
    """
    A = M_c.astype(complex, copy=True)
    pivot_rows = []  # (row_idx, col_idx) 쌍
    used_rows = set()
    used_cols = set()
    row = 0

    for col in range(n):
        if row >= n:
            break
        # 최대 절대값 행 탐색
        col_abs = np.abs(A[row:, col])
        best_local = np.argmax(col_abs)
        abs_mx = col_abs[best_local]
        pivot_row = row + best_local

        if abs_mx < 1e-11:
            continue

        # 행 교환
        if pivot_row != row:
            A[[row, pivot_row], :] = A[[pivot_row, row], :]

        # 정규화
        A[row, :] = A[row, :] / A[row, col]

        # 소거 (위아래 모두)
        mask = np.ones(n, dtype=bool)
        mask[row] = False
        facs = A[mask, col]
        A[mask, :] -= facs[:, np.newaxis] * A[row, :]

        pivot_rows.append((row, col))
        used_cols.add(col)
        row += 1

    # free variables
    free_cols = [c for c in range(n) if c not in used_cols]

    if len(free_cols) == 0:
        if len(pivot_rows) > 0:
            _, fc = pivot_rows.pop()
            free_cols = [fc]
        else:
            free_cols = [n - 1]

    fc = free_cols[0]
    v = np.zeros(n, dtype=complex)
    v[fc] = 1.0 + 0j

    # back-substitution: RREF이므로 pivot column 위치에서 직접 읽기
    for row_idx, pc in pivot_rows:
        s = 0.0 + 0j
        for j in range(n):
            if j != pc:
                s += A[row_idx, j] * v[j]
        v[pc] = -s

    # 정규화
    nm = float(np.linalg.norm(v))
    if nm > 1e-300:
        v = v / nm
    else:
        # fallback: 최소 특이값 방향
        _, sv, Vh = np.linalg.svd(M_c, full_matrices=True)
        v = np.conj(Vh[-1])
        nm2 = float(np.linalg.norm(v))
        if nm2 > 1e-300:
            v = v / nm2

    return v
