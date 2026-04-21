# _rk78_tableau.pxd
# Dormand-Prince 8(7) — 13-stage Butcher tableau
#
# 출처: Hairer, E., Nørsett, S.P., Wanner, G.,
#       "Solving Ordinary Differential Equations I: Nonstiff Problems",
#       2nd Ed., Springer, 1993, Table II.5.4 (pp. 178-179).
#       계수는 18 유효숫자 정밀도로 하드코딩.
#
# 변수 명명:
#   _DP_C[i]    : c 노드 (i=0..12), c[0]=0, c[12]=1
#   _DP_A[i][j] : 하삼각 a 계수 (j < i)
#   _DP_B8[i]   : 8차 해 가중치 (b_i)
#   _DP_B7[i]   : 7차 해 가중치 (b*_i, 오차 추정용)
#
# FSAL 성질: c[12]=1, B8 벡터로 계산한 y_new에서 k[12]=f(t+h, y_new)
#             → 다음 step의 k[0]으로 재활용 가능.
# cython: language_level=3

cdef double _DP_C[13]
cdef double _DP_A[13][13]
cdef double _DP_B8[13]
cdef double _DP_B7[13]
