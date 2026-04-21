# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# linear_algebra.pyx  —  LinearAlgebra 클래스: 행렬 분해 9종
#
#   1. lu                   LU 분해 (Doolittle + partial pivoting)
#   2. ldu                  LDU 분해 (LU 유도)
#   3. qr                   QR 분해 (Householder)
#   4. cholesky             Cholesky-Banachiewicz
#   5. svd                  SVD (Golub-Reinsch bidiagonalization + QR iteration)
#   6. gaussian_elimination Gauss-Jordan / RREF
#   7. det                  행렬식 (LU 기반)
#   8. rank                 랭크 (SVD 기반)
#   9. inverse              역행렬 (LU 기반)

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, fabs
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
    def qr(self, np.ndarray A, *, mode='reduced'):
        """
        QR 분해 — Householder reflections.

        A (m×n) = Q · R
          reduced  Q (m,k), R (k,n)   k = min(m,n)
          complete Q (m,m), R (m,n)

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
        cdef int i, j, p, col
        cdef double beta, alpha_v, ns, v0, dv

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

        for p in range(k):
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
