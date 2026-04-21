# LinearAlgebra Guide — Session 1+2 (33 Methods)

## 1. 개요

`math_library.LinearAlgebra` 클래스는 `np.linalg.*` 없이 Cython으로 자체 구현한 선형대수
알고리즘 33종을 제공합니다. Session 1(21종)에서 분해·소거·직교화·부분공간·고유값을 구현했고,
Session 2(11종)에서 텐서 연산·대칭 검사·미분 wrap을 추가했습니다.

---

## 2. 텐서 연산 이론

### 2.1 Kronecker Product (tensor_product)

A (m, n), B (p, q)일 때 Kronecker product는

    A ⊗ B = [ a_{00}B  a_{01}B  ...  a_{0,n-1}B ]
             [ a_{10}B  a_{11}B  ...  a_{1,n-1}B ]
             [  ...                               ]
             [ a_{m-1,0}B ...    a_{m-1,n-1}B    ]

결과 shape: (m·p, n·q). 블록 (i*p:(i+1)*p, j*q:(j+1)*q) = a_{ij} * B.

**성질**

- (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD)  (차원 호환 시)
- (A ⊗ B)^T = A^T ⊗ B^T
- vec(AXB) = (B^T ⊗ A) vec(X)  — 제어이론 Lyapunov 방정식에서 활용

**구현**: Cython 이중 루프로 각 블록에 a_{ij} * B 스칼라-배열 곱을 직접 배치.

### 2.2 Outer Product (outer_product)

    (a ⊗ b)_{ij} = a_i * b_j,  a (m,), b (n,) → (m, n)

1차 텐서 두 개의 텐서곱으로, rank-1 행렬을 생성합니다. 복소 dtype을 그대로 지원합니다.

### 2.3 Inner Product (inner_product)

Hermitian 내적(표준 복소 내적):

    <a, b> = sum_i conj(a_i) * b_i

실수 벡터에서는 표준 dot product와 동일. 복소 벡터에서는 켤레 취하여 양의 정부호 보장.

### 2.4 Tensor Contraction (tensor_contract)

일반 텐서 축약은 numpy.tensordot의 축약 방식을 직접 구현:

1. A의 축약 축(axes_a)을 뒤로 transpose
2. B의 축약 축(axes_b)을 앞으로 transpose
3. 각각 2D로 reshape: A2 (M_a, K), B2 (K, M_b)
4. matmul C2 = A2 @ B2
5. 최종 shape [free_a dims] + [free_b dims]로 reshape

K = 축약 차원들의 크기 곱. 예: axes 하나이고 크기 5이면 K=5, 두 개이고 크기 (3,4)이면 K=12.

**검증**: numpy.tensordot과 상대오차 < 1e-12 확인.

### 2.5 Tensor Transpose (tensor_transpose)

numpy.transpose를 래핑하되 항상 C-contiguous 복사본 반환. 알고리즘적 의미는 없고 메모리 레이아웃
정리 목적.

---

## 3. Cayley-Hamilton 정리 스케치

**정리**: n×n 행렬 A의 특성 다항식을 p(λ) = det(λI - A)라 하면,

    p(A) = 0  (영 행렬)

**증명 스케치 (Faddeev-LeVerrier 경유)**

Faddeev-LeVerrier 알고리즘으로 계수 c_k 계산:

    M_0 = I
    c_{n-1} = -tr(A)
    M_k = A * M_{k-1} + c_{n-k} * I
    c_{n-k-1} = -tr(A * M_k) / (k+1)

이때 다항식 p(λ) = lambda^n + c_{n-1} lambda^{n-1} + ... + c_0 가 성립.

행렬 A를 λ 자리에 대입:

    p(A) = A^n + c_{n-1} A^{n-1} + ... + c_0 I

Faddeev-LeVerrier의 전개 과정에서 각 항이 서로 소거되어 결과는 0 행렬.
`cayley_hamilton` 메서드는 이 잔차 행렬을 반환하며, 잘 조건화된 행렬에서는 최대 원소 절대값이
machine epsilon (2.22e-16) 수준입니다.

---

## 4. 4대 부분공간 (Fundamental Theorem of Linear Algebra)

m×n 행렬 A에 대해 4개의 부분공간이 정의됩니다.

### 4.1 열공간 Col(A) — column_space

    Col(A) = { Ax : x in R^n }  ⊆ R^m

차원 = rank(A). SVD에서 왼쪽 특이벡터 U[:, :rank]가 직교 기저.

### 4.2 행공간 Row(A) = Col(A^T) — row_space

    Row(A) = Col(A^T)  ⊆ R^n

차원 = rank(A). SVD에서 오른쪽 특이벡터 V[:, :rank]가 직교 기저.

### 4.3 우측 영공간 Null(A) — null_space

    Null(A) = { x : Ax = 0 }  ⊆ R^n

차원 = n - rank(A). SVD에서 Vh[rank:, :]^T 가 기저.

### 4.4 좌측 영공간 LeftNull(A) = Null(A^T) — left_null_space

    LeftNull(A) = { y : y^T A = 0 } = Null(A^T)  ⊆ R^m

차원 = m - rank(A). A^T의 Null(A^T) = U[:, rank:]가 기저.

### 4.5 직교 분해 관계

    Col(A)    ⊥  LeftNull(A)    (R^m 에서)
    Row(A)    ⊥  Null(A)        (R^n 에서)

모든 x in R^n 은 유일하게 x = x_row + x_null 로 분해됩니다.

---

## 5. 대칭 행렬 분해 이론

### 5.1 대칭 / 반대칭 분해

임의 정방 행렬 A는 다음과 같이 유일하게 분해됩니다:

    A = S + K

여기서 S = (A + A^T) / 2 (대칭), K = (A - A^T) / 2 (반대칭).

검증: S + K = A, S^T = S, K^T = -K. `skew_part` + `symmetrize` = 원 행렬.

### 5.2 복소 Hermitian 대칭화

복소 행렬의 경우 Hermitian 분해:

    H = (A + A^H) / 2,  where A^H = conj(A)^T

H^H = H 성립. 복소 수치 최적화나 반정부호 행렬 보장에 활용합니다.

---

## 6. 미분 Wrap 구현 세부

### 6.1 jacobian

`Differentiation.jacobian(functions: list[callable], point)` 시그니처를 받으므로,
단일 callable f가 R^n → R^m인 경우 성분 함수 m개를 동적으로 생성합니다:

    comp_fn_i(*args) = f(*args)[i]

이후 `Differentiation.jacobian([comp_fn_0, ..., comp_fn_{m-1}], point)` 호출.
결과를 ndarray (m, n)으로 변환.

정밀도: Ridders Richardson extrapolation 사용. 해석적 야코비안과의 상대오차 ~1e-6 수준.

### 6.2 hessian

`Differentiation.hessian(f, point)` 직접 래핑. 내부적으로 2차 중앙차분:

    H[i][i] = (f(x+h*e_i) - 2f(x) + f(x-h*e_i)) / h^2
    H[i][j] = (f(x+h*e_i+h*e_j) - f(x+h*e_i-h*e_j)
               - f(x-h*e_i+h*e_j) + f(x-h*e_i-h*e_j)) / (4h^2)

Schwarz 정리에 의해 H[i][j] = H[j][i] (수치 오차 ~1e-6 수준).

---

## 7. API 빠른 참조

### 텐서

```python
la.tensor_product(A, B)           # Kronecker A⊗B
la.outer_product(a, b)            # rank-1 matrix a b^T
la.inner_product(a, b)            # Hermitian dot sum(conj(a)*b)
la.tensor_contract(A, B, ax_a, ax_b)  # general contraction
la.tensor_transpose(T, axes=None) # permute axes, C-contiguous copy
```

### 대칭

```python
la.is_symmetric(A, tol=1e-10)     # bool
la.is_skew_symmetric(A, tol=1e-10)# bool
la.symmetrize(A)                  # (A + A^H) / 2
la.skew_part(A)                   # (A - A^T) / 2
```

### 미분

```python
la.jacobian(f, point)  # f: R^n → R^m, result (m, n)
la.hessian(f, point)   # f: R^n → R,   result (n, n)
```

---

## 8. 알고리즘 정확도 요약

| 메서드 | 정확도 | 비고 |
|---|---|---|
| tensor_product | ~1e-15 | double 정밀도 스칼라 곱 |
| outer_product | ~1e-15 | — |
| inner_product | ~1e-15 | Hermitian |
| tensor_contract | ~1e-12 | matmul 기반 |
| tensor_transpose | ~1e-14 | 메모리 복사 |
| is_symmetric | tol=1e-10 | max-norm 기준 |
| is_skew_symmetric | tol=1e-10 | max-norm 기준 |
| symmetrize | ~1e-15 | 산술 평균 |
| skew_part | ~1e-15 | 산술 평균 |
| jacobian | ~1e-6 | Ridders 수치 미분 |
| hessian | ~1e-4 | 2차 중앙차분 |
