# _adams_coeffs.pxd
# Adams-Bashforth 및 Adams-Moulton 계수 (order=1..5)
#
# Adams-Bashforth (explicit):
#   y_{n+1} = y_n + h * sum_{j=0}^{k-1} beta_j * f_{n-j}
#
# Adams-Moulton (implicit):
#   y_{n+1} = y_n + h * sum_{j=0}^{k} beta*_j * f_{n+1-j}
#   (beta*_0 * f_{n+1}이 암시적 항)
#
# 계수 출처: Hairer, Nørsett, Wanner, "Solving ODEs I", Table III.1.1
#            및 Butcher, "Numerical Methods for ODEs", Table 4.1.
#
# 배열 레이아웃:
#   _AB_COEFFS[k][j]: k차 Adams-Bashforth, j번째 계수 (j=0..k-1)
#   _AM_COEFFS[k][j]: k차 Adams-Moulton,   j번째 계수 (j=0..k)
#
# 인덱스: k=1..5 (0은 미사용)
# cython: language_level=3

# Adams-Bashforth 계수: _AB_COEFFS[order][j], order=1..5
# order 1: [1]
# order 2: [3/2, -1/2]
# order 3: [23/12, -16/12, 5/12]
# order 4: [55/24, -59/24, 37/24, -9/24]
# order 5: [1901/720, -2774/720, 2616/720, -1274/720, 251/720]
cdef double _AB_COEFFS[6][5]

# Adams-Moulton 계수: _AM_COEFFS[order][j], order=1..5
# order 1 (Backward Euler): [1]
# order 2 (Trapezoidal):    [1/2, 1/2]
# order 3:                  [5/12, 8/12, -1/12]
# order 4:                  [9/24, 19/24, -5/24, 1/24]
# order 5:                  [251/720, 646/720, -264/720, 106/720, -19/720]
cdef double _AM_COEFFS[6][5]
