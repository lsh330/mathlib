# _gauss_legendre_tables.pxd
# n=2..16 Gauss-Legendre 노드 및 가중치 테이블
#
# 출처: Abramowitz & Stegun, "Handbook of Mathematical Functions",
#       Table 25.4 (p. 916-919), 1964.
#       보완: Steven Dhavala, "High-Precision Gauss-Legendre Quadrature Nodes and Weights"
#             및 NIST DLMF §3.5(v).
#
# 대칭성 활용: 노드는 [-1, 1]에서 양수 노드만 저장 후 ±로 적용.
# 단, 이 구현에서는 완전한 노드 목록(음수 포함)을 평탄화 배열로 저장.
# 인덱스 체계:
#   _GL_OFFSETS[n]   : n-point 노드/가중치가 시작되는 인덱스 (n=2..16)
#   _GL_NODES_FLAT   : 모든 n의 노드를 순서대로 저장
#   _GL_WEIGHTS_FLAT : 대응 가중치
#
# 전체 노드 수: sum(2..16) = 135
#
# 주의: 배열 선언을 pyx에서 include로 사용하므로
#       실제 배열은 numerical_analysis.pyx에 포함됨.
# cython: language_level=3

# 오프셋 배열 (인덱스 0=미사용, 2..16 사용)
# _GL_OFFSETS[n] = sum(2..n-1) = n*(n-1)/2 - 1  로 계산
# 직접 하드코딩:
#  n=2:  offset=0   (2 nodes)
#  n=3:  offset=2   (3 nodes)
#  n=4:  offset=5   (4 nodes)
#  n=5:  offset=9   (5 nodes)
#  n=6:  offset=14  (6 nodes)
#  n=7:  offset=20  (7 nodes)
#  n=8:  offset=27  (8 nodes)
#  n=9:  offset=35  (9 nodes)
#  n=10: offset=44  (10 nodes)
#  n=11: offset=54  (11 nodes)
#  n=12: offset=65  (12 nodes)
#  n=13: offset=77  (13 nodes)
#  n=14: offset=90  (14 nodes)
#  n=15: offset=104 (15 nodes)
#  n=16: offset=119 (16 nodes)
#  total = 135

cdef int _GL_OFFSETS[17]
cdef double _GL_NODES_FLAT[135]
cdef double _GL_WEIGHTS_FLAT[135]
