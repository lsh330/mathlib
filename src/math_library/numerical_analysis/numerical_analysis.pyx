# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
#
# numerical_analysis.pyx
#
# NumericalAnalysis 클래스 — Simpson 적분법 8종 + Newton-Raphson/Secant + RK 5종
#                          + Gauss-Legendre 2종 + RK78 + Adams 3종 = 21 메서드
#
# 구현 목록:
#   1. simpson_13               — 3점 단순 Simpson 1/3
#   2. simpson_38               — 4점 단순 Simpson 3/8
#   3. composite_simpson_13     — n 구간 합성 Simpson 1/3 (n 짝수)
#   4. composite_simpson_38     — n 구간 합성 Simpson 3/8 (n 3의 배수)
#   5. adaptive_simpson         — 적응형 재귀 Simpson (명시적 스택)
#   6. mixed_simpson            — 혼합 Simpson (임의 n >= 2)
#   7. simpson_irregular        — 불균등 간격 Simpson
#   8. romberg                  — Romberg 적분 (Richardson 확장)
#   9. newton_raphson           — Newton-Raphson 근 찾기 (Differentiation 재활용)
#  10. secant_method            — Secant 근 찾기
#  11. euler                    — Euler 1차 ODE 적분
#  12. rk2                      — 2차 Runge-Kutta (midpoint/heun/ralston)
#  13. rk4                      — 고전 4차 Runge-Kutta
#  14. rk45                     — Dormand-Prince 5(4) 적응형 (DOPRI5/FSAL)
#  15. rk_fehlberg              — Fehlberg RKF45 적응형
#  16. gauss_legendre           — n-point Gauss-Legendre 적분 (n=2..16)
#  17. composite_gauss_legendre — m 구간 합성 Gauss-Legendre
#  18. rk78                     — Dormand-Prince 8(7) 13-stage 적응형
#  19. adams_bashforth          — Explicit multistep (order=1..5)
#  20. adams_moulton            — Implicit multistep (order=1..5)
#  21. predictor_corrector      — PECE Adams AB/AM (order=2..5)

from libc.math cimport fabs, isnan, isinf
include "_kahan.pxd"

# ================================================================== Gauss-Legendre 테이블
# 출처: Abramowitz & Stegun, Table 25.4 (18 유효숫자)
#       보완: NIST DLMF §3.5(v), Dhavala online tables
#
# 평탄화 배열: n=2..16 순서로 노드/가중치 연속 저장
# 오프셋: _GL_OFFSETS[n] → 해당 n의 첫 번째 원소 인덱스
#   n=2:  0  (2개)
#   n=3:  2  (3개)
#   n=4:  5  (4개)
#   n=5:  9  (5개)
#   n=6:  14 (6개)
#   n=7:  20 (7개)
#   n=8:  27 (8개)
#   n=9:  35 (9개)
#   n=10: 44 (10개)
#   n=11: 54 (11개)
#   n=12: 65 (12개)
#   n=13: 77 (13개)
#   n=14: 90 (14개)
#   n=15: 104 (15개)
#   n=16: 119 (16개) → 총 135개

cdef int _GL_OFFSETS[17]
_GL_OFFSETS[0]  = 0
_GL_OFFSETS[1]  = 0
_GL_OFFSETS[2]  = 0
_GL_OFFSETS[3]  = 2
_GL_OFFSETS[4]  = 5
_GL_OFFSETS[5]  = 9
_GL_OFFSETS[6]  = 14
_GL_OFFSETS[7]  = 20
_GL_OFFSETS[8]  = 27
_GL_OFFSETS[9]  = 35
_GL_OFFSETS[10] = 44
_GL_OFFSETS[11] = 54
_GL_OFFSETS[12] = 65
_GL_OFFSETS[13] = 77
_GL_OFFSETS[14] = 90
_GL_OFFSETS[15] = 104
_GL_OFFSETS[16] = 119

cdef double _GL_NODES_FLAT[135]
# n=2: ±0.5773502691896257
_GL_NODES_FLAT[0]  = -0.5773502691896257645454878
_GL_NODES_FLAT[1]  =  0.5773502691896257645454878
# n=3: 0, ±0.7745966692414834
_GL_NODES_FLAT[2]  = -0.7745966692414833770358531
_GL_NODES_FLAT[3]  =  0.0
_GL_NODES_FLAT[4]  =  0.7745966692414833770358531
# n=4: ±0.3399810435848563, ±0.8611363115940526
_GL_NODES_FLAT[5]  = -0.8611363115940525752239465
_GL_NODES_FLAT[6]  = -0.3399810435848562648026658
_GL_NODES_FLAT[7]  =  0.3399810435848562648026658
_GL_NODES_FLAT[8]  =  0.8611363115940525752239465
# n=5: 0, ±0.5384693101056831, ±0.9061798459386640
_GL_NODES_FLAT[9]  = -0.9061798459386639927976269
_GL_NODES_FLAT[10] = -0.5384693101056830910363144
_GL_NODES_FLAT[11] =  0.0
_GL_NODES_FLAT[12] =  0.5384693101056830910363144
_GL_NODES_FLAT[13] =  0.9061798459386639927976269
# n=6: ±0.2386191860831969, ±0.6612093864662645, ±0.9324695142031521
_GL_NODES_FLAT[14] = -0.9324695142031520278123016
_GL_NODES_FLAT[15] = -0.6612093864662645136613996
_GL_NODES_FLAT[16] = -0.2386191860831969086305017
_GL_NODES_FLAT[17] =  0.2386191860831969086305017
_GL_NODES_FLAT[18] =  0.6612093864662645136613996
_GL_NODES_FLAT[19] =  0.9324695142031520278123016
# n=7: 0, ±0.4058451513773972, ±0.7415311855993945, ±0.9491079123427585
_GL_NODES_FLAT[20] = -0.9491079123427585245261897
_GL_NODES_FLAT[21] = -0.7415311855993944398638648
_GL_NODES_FLAT[22] = -0.4058451513773971669066064
_GL_NODES_FLAT[23] =  0.0
_GL_NODES_FLAT[24] =  0.4058451513773971669066064
_GL_NODES_FLAT[25] =  0.7415311855993944398638648
_GL_NODES_FLAT[26] =  0.9491079123427585245261897
# n=8: ±0.1834346424956498, ±0.5255324099163290, ±0.7966664774136267, ±0.9602898564975363
_GL_NODES_FLAT[27] = -0.9602898564975362316835609
_GL_NODES_FLAT[28] = -0.7966664774136267395915539
_GL_NODES_FLAT[29] = -0.5255324099163289858177390
_GL_NODES_FLAT[30] = -0.1834346424956498049394761
_GL_NODES_FLAT[31] =  0.1834346424956498049394761
_GL_NODES_FLAT[32] =  0.5255324099163289858177390
_GL_NODES_FLAT[33] =  0.7966664774136267395915539
_GL_NODES_FLAT[34] =  0.9602898564975362316835609
# n=9: 0, ±0.3242534234038089, ±0.6133714327005904, ±0.8360311073266358, ±0.9681602395076261
_GL_NODES_FLAT[35] = -0.9681602395076260898355762
_GL_NODES_FLAT[36] = -0.8360311073266357942994298
_GL_NODES_FLAT[37] = -0.6133714327005903973087020
_GL_NODES_FLAT[38] = -0.3242534234038089290385380
_GL_NODES_FLAT[39] =  0.0
_GL_NODES_FLAT[40] =  0.3242534234038089290385380
_GL_NODES_FLAT[41] =  0.6133714327005903973087020
_GL_NODES_FLAT[42] =  0.8360311073266357942994298
_GL_NODES_FLAT[43] =  0.9681602395076260898355762
# n=10: ±0.1488743389816312, ±0.4333953941292472, ±0.6794095682990244, ±0.8650633666889845, ±0.9739065285171717
_GL_NODES_FLAT[44] = -0.9739065285171717200779640
_GL_NODES_FLAT[45] = -0.8650633666889845107320967
_GL_NODES_FLAT[46] = -0.6794095682990244062343274
_GL_NODES_FLAT[47] = -0.4333953941292471907992659
_GL_NODES_FLAT[48] = -0.1488743389816312108848260
_GL_NODES_FLAT[49] =  0.1488743389816312108848260
_GL_NODES_FLAT[50] =  0.4333953941292471907992659
_GL_NODES_FLAT[51] =  0.6794095682990244062343274
_GL_NODES_FLAT[52] =  0.8650633666889845107320967
_GL_NODES_FLAT[53] =  0.9739065285171717200779640
# n=11: 0, ±0.2695431559523450, ±0.5190961292068118, ±0.7301520055740494, ±0.8870625997680953, ±0.9782286581460570
_GL_NODES_FLAT[54] = -0.9782286581460569928039380
_GL_NODES_FLAT[55] = -0.8870625997680952990751578
_GL_NODES_FLAT[56] = -0.7301520055740493240934163
_GL_NODES_FLAT[57] = -0.5190961292068118159257257
_GL_NODES_FLAT[58] = -0.2695431559523449723315320
_GL_NODES_FLAT[59] =  0.0
_GL_NODES_FLAT[60] =  0.2695431559523449723315320
_GL_NODES_FLAT[61] =  0.5190961292068118159257257
_GL_NODES_FLAT[62] =  0.7301520055740493240934163
_GL_NODES_FLAT[63] =  0.8870625997680952990751578
_GL_NODES_FLAT[64] =  0.9782286581460569928039380
# n=12: ±0.1252334085114689, ±0.3678314989981802, ±0.5873179542866175, ±0.7699026741943047, ±0.9041172563704749, ±0.9815606342467193
_GL_NODES_FLAT[65] = -0.9815606342467192506905491
_GL_NODES_FLAT[66] = -0.9041172563704748566784659
_GL_NODES_FLAT[67] = -0.7699026741943046870368938
_GL_NODES_FLAT[68] = -0.5873179542866174472967024
_GL_NODES_FLAT[69] = -0.3678314989981801937526915
_GL_NODES_FLAT[70] = -0.1252334085114689154724414
_GL_NODES_FLAT[71] =  0.1252334085114689154724414
_GL_NODES_FLAT[72] =  0.3678314989981801937526915
_GL_NODES_FLAT[73] =  0.5873179542866174472967024
_GL_NODES_FLAT[74] =  0.7699026741943046870368938
_GL_NODES_FLAT[75] =  0.9041172563704748566784659
_GL_NODES_FLAT[76] =  0.9815606342467192506905491
# n=13: 0, ±0.2304583159551348, ±0.4484927510364469, ±0.6423493394403402, ±0.8015780907333099, ±0.9175983992229779, ±0.9841830547185881
_GL_NODES_FLAT[77] = -0.9841830547185880975394739
_GL_NODES_FLAT[78] = -0.9175983992229779652065478
_GL_NODES_FLAT[79] = -0.8015780907333099127942065
_GL_NODES_FLAT[80] = -0.6423493394403402206439846
_GL_NODES_FLAT[81] = -0.4484927510364468528779129
_GL_NODES_FLAT[82] = -0.2304583159551347940655281
_GL_NODES_FLAT[83] =  0.0
_GL_NODES_FLAT[84] =  0.2304583159551347940655281
_GL_NODES_FLAT[85] =  0.4484927510364468528779129
_GL_NODES_FLAT[86] =  0.6423493394403402206439846
_GL_NODES_FLAT[87] =  0.8015780907333099127942065
_GL_NODES_FLAT[88] =  0.9175983992229779652065478
_GL_NODES_FLAT[89] =  0.9841830547185880975394739
# n=14: ±0.1080549487073436, ±0.3191123689278897, ±0.5152486363581540, ±0.6872929048116855, ±0.8272013150697650, ±0.9284348836635735, ±0.9862838086968123
_GL_NODES_FLAT[90]  = -0.9862838086968123388415973
_GL_NODES_FLAT[91]  = -0.9284348836635735173363911
_GL_NODES_FLAT[92]  = -0.8272013150697649931897947
_GL_NODES_FLAT[93]  = -0.6872929048116854701480198
_GL_NODES_FLAT[94]  = -0.5152486363581540919652907
_GL_NODES_FLAT[95]  = -0.3191123689278897604515722
_GL_NODES_FLAT[96]  = -0.1080549487073436620662447
_GL_NODES_FLAT[97]  =  0.1080549487073436620662447
_GL_NODES_FLAT[98]  =  0.3191123689278897604515722
_GL_NODES_FLAT[99]  =  0.5152486363581540919652907
_GL_NODES_FLAT[100] =  0.6872929048116854701480198
_GL_NODES_FLAT[101] =  0.8272013150697649931897947
_GL_NODES_FLAT[102] =  0.9284348836635735173363911
_GL_NODES_FLAT[103] =  0.9862838086968123388415973
# n=15: 0, ±0.2011940939974345, ±0.3941513470775634, ±0.5709721726085388, ±0.7244177313601701, ±0.8482065834104272, ±0.9372733924007059, ±0.9879925180204854
_GL_NODES_FLAT[104] = -0.9879925180204854284895657
_GL_NODES_FLAT[105] = -0.9372733924007059043077589
_GL_NODES_FLAT[106] = -0.8482065834104272162006483
_GL_NODES_FLAT[107] = -0.7244177313601700474161861
_GL_NODES_FLAT[108] = -0.5709721726085388475372267
_GL_NODES_FLAT[109] = -0.3941513470775633498498730
_GL_NODES_FLAT[110] = -0.2011940939974345223006283
_GL_NODES_FLAT[111] =  0.0
_GL_NODES_FLAT[112] =  0.2011940939974345223006283
_GL_NODES_FLAT[113] =  0.3941513470775633498498730
_GL_NODES_FLAT[114] =  0.5709721726085388475372267
_GL_NODES_FLAT[115] =  0.7244177313601700474161861
_GL_NODES_FLAT[116] =  0.8482065834104272162006483
_GL_NODES_FLAT[117] =  0.9372733924007059043077589
_GL_NODES_FLAT[118] =  0.9879925180204854284895657
# n=16 (numpy.polynomial.legendre.leggauss 참조)
_GL_NODES_FLAT[119] = -0.9894009349916499385102
_GL_NODES_FLAT[120] = -0.9445750230732326002681
_GL_NODES_FLAT[121] = -0.8656312023878317551961
_GL_NODES_FLAT[122] = -0.7554044083550029986540
_GL_NODES_FLAT[123] = -0.6178762444026437705702
_GL_NODES_FLAT[124] = -0.4580167776572273696800
_GL_NODES_FLAT[125] = -0.2816035507792589154263
_GL_NODES_FLAT[126] = -0.0950125098376374405129
_GL_NODES_FLAT[127] =  0.0950125098376374405129
_GL_NODES_FLAT[128] =  0.2816035507792589154263
_GL_NODES_FLAT[129] =  0.4580167776572273696800
_GL_NODES_FLAT[130] =  0.6178762444026437705702
_GL_NODES_FLAT[131] =  0.7554044083550029986540
_GL_NODES_FLAT[132] =  0.8656312023878317551961
_GL_NODES_FLAT[133] =  0.9445750230732326002681
_GL_NODES_FLAT[134] =  0.9894009349916499385102

cdef double _GL_WEIGHTS_FLAT[135]
# n=2
_GL_WEIGHTS_FLAT[0]  = 1.0
_GL_WEIGHTS_FLAT[1]  = 1.0
# n=3
_GL_WEIGHTS_FLAT[2]  = 0.5555555555555555555555556   # 5/9
_GL_WEIGHTS_FLAT[3]  = 0.8888888888888888888888889   # 8/9
_GL_WEIGHTS_FLAT[4]  = 0.5555555555555555555555556
# n=4
_GL_WEIGHTS_FLAT[5]  = 0.3478548451374538573730639
_GL_WEIGHTS_FLAT[6]  = 0.6521451548625461426269361
_GL_WEIGHTS_FLAT[7]  = 0.6521451548625461426269361
_GL_WEIGHTS_FLAT[8]  = 0.3478548451374538573730639
# n=5
_GL_WEIGHTS_FLAT[9]  = 0.2369268850561890875142640   # (322-13√70)/900
_GL_WEIGHTS_FLAT[10] = 0.4786286704993664680412915
_GL_WEIGHTS_FLAT[11] = 0.5688888888888888888888889   # 128/225
_GL_WEIGHTS_FLAT[12] = 0.4786286704993664680412915
_GL_WEIGHTS_FLAT[13] = 0.2369268850561890875142640
# n=6
_GL_WEIGHTS_FLAT[14] = 0.1713244923791703450402961
_GL_WEIGHTS_FLAT[15] = 0.3607615730481386075698335
_GL_WEIGHTS_FLAT[16] = 0.4679139345726910473898703
_GL_WEIGHTS_FLAT[17] = 0.4679139345726910473898703
_GL_WEIGHTS_FLAT[18] = 0.3607615730481386075698335
_GL_WEIGHTS_FLAT[19] = 0.1713244923791703450402961
# n=7
_GL_WEIGHTS_FLAT[20] = 0.1294849661688696932706114
_GL_WEIGHTS_FLAT[21] = 0.2797053914892766679014678
_GL_WEIGHTS_FLAT[22] = 0.3818300505051189449503698
_GL_WEIGHTS_FLAT[23] = 0.4179591836734693877551020   # 128/225 adjusted
_GL_WEIGHTS_FLAT[24] = 0.3818300505051189449503698
_GL_WEIGHTS_FLAT[25] = 0.2797053914892766679014678
_GL_WEIGHTS_FLAT[26] = 0.1294849661688696932706114
# n=8
_GL_WEIGHTS_FLAT[27] = 0.1012285362903762591525314
_GL_WEIGHTS_FLAT[28] = 0.2223810344533744705443560
_GL_WEIGHTS_FLAT[29] = 0.3137066458778872873379622
_GL_WEIGHTS_FLAT[30] = 0.3626837833783619829651504
_GL_WEIGHTS_FLAT[31] = 0.3626837833783619829651504
_GL_WEIGHTS_FLAT[32] = 0.3137066458778872873379622
_GL_WEIGHTS_FLAT[33] = 0.2223810344533744705443560
_GL_WEIGHTS_FLAT[34] = 0.1012285362903762591525314
# n=9
_GL_WEIGHTS_FLAT[35] = 0.0812743883615744119718922
_GL_WEIGHTS_FLAT[36] = 0.1806481606948574040584720
_GL_WEIGHTS_FLAT[37] = 0.2606106964029354623187429
_GL_WEIGHTS_FLAT[38] = 0.3123470770400028400686304
_GL_WEIGHTS_FLAT[39] = 0.3302393550012597631645251
_GL_WEIGHTS_FLAT[40] = 0.3123470770400028400686304
_GL_WEIGHTS_FLAT[41] = 0.2606106964029354623187429
_GL_WEIGHTS_FLAT[42] = 0.1806481606948574040584720
_GL_WEIGHTS_FLAT[43] = 0.0812743883615744119718922
# n=10
_GL_WEIGHTS_FLAT[44] = 0.0666713443086881375935688
_GL_WEIGHTS_FLAT[45] = 0.1494513491505805931457763
_GL_WEIGHTS_FLAT[46] = 0.2190863625159820439955349
_GL_WEIGHTS_FLAT[47] = 0.2692667193099963550912269
_GL_WEIGHTS_FLAT[48] = 0.2955242247147528701738930
_GL_WEIGHTS_FLAT[49] = 0.2955242247147528701738930
_GL_WEIGHTS_FLAT[50] = 0.2692667193099963550912269
_GL_WEIGHTS_FLAT[51] = 0.2190863625159820439955349
_GL_WEIGHTS_FLAT[52] = 0.1494513491505805931457763
_GL_WEIGHTS_FLAT[53] = 0.0666713443086881375935688
# n=11
_GL_WEIGHTS_FLAT[54] = 0.0556685671161736664827537
_GL_WEIGHTS_FLAT[55] = 0.1255803694649046246346943
_GL_WEIGHTS_FLAT[56] = 0.1862902109277342514260976
_GL_WEIGHTS_FLAT[57] = 0.2331937645919904799185237
_GL_WEIGHTS_FLAT[58] = 0.2628045445102466621806889
_GL_WEIGHTS_FLAT[59] = 0.2729250867779006307144835
_GL_WEIGHTS_FLAT[60] = 0.2628045445102466621806889
_GL_WEIGHTS_FLAT[61] = 0.2331937645919904799185237
_GL_WEIGHTS_FLAT[62] = 0.1862902109277342514260976
_GL_WEIGHTS_FLAT[63] = 0.1255803694649046246346943
_GL_WEIGHTS_FLAT[64] = 0.0556685671161736664827537
# n=12
_GL_WEIGHTS_FLAT[65] = 0.0471753363865118271946160
_GL_WEIGHTS_FLAT[66] = 0.1069393259953184309602547
_GL_WEIGHTS_FLAT[67] = 0.1600783285433462263346525
_GL_WEIGHTS_FLAT[68] = 0.2031674267230659217490645
_GL_WEIGHTS_FLAT[69] = 0.2334925365383548087608499
_GL_WEIGHTS_FLAT[70] = 0.2491470458134027850005624
_GL_WEIGHTS_FLAT[71] = 0.2491470458134027850005624
_GL_WEIGHTS_FLAT[72] = 0.2334925365383548087608499
_GL_WEIGHTS_FLAT[73] = 0.2031674267230659217490645
_GL_WEIGHTS_FLAT[74] = 0.1600783285433462263346525
_GL_WEIGHTS_FLAT[75] = 0.1069393259953184309602547
_GL_WEIGHTS_FLAT[76] = 0.0471753363865118271946160
# n=13
_GL_WEIGHTS_FLAT[77] = 0.0404840047653158795200216
_GL_WEIGHTS_FLAT[78] = 0.0921214998377284593616342
_GL_WEIGHTS_FLAT[79] = 0.1388735102197872384636018
_GL_WEIGHTS_FLAT[80] = 0.1781459807619457382800467
_GL_WEIGHTS_FLAT[81] = 0.2078160475368885023125232
_GL_WEIGHTS_FLAT[82] = 0.2262831802628972384120902
_GL_WEIGHTS_FLAT[83] = 0.2325515532308739101945895
_GL_WEIGHTS_FLAT[84] = 0.2262831802628972384120902
_GL_WEIGHTS_FLAT[85] = 0.2078160475368885023125232
_GL_WEIGHTS_FLAT[86] = 0.1781459807619457382800467
_GL_WEIGHTS_FLAT[87] = 0.1388735102197872384636018
_GL_WEIGHTS_FLAT[88] = 0.0921214998377284593616342
_GL_WEIGHTS_FLAT[89] = 0.0404840047653158795200216
# n=14
_GL_WEIGHTS_FLAT[90]  = 0.0351194603317518630318329
_GL_WEIGHTS_FLAT[91]  = 0.0801580871597602098056333
_GL_WEIGHTS_FLAT[92]  = 0.1215185706879031846894148
_GL_WEIGHTS_FLAT[93]  = 0.1572031671581935345696019
_GL_WEIGHTS_FLAT[94]  = 0.1855383974779378137417166
_GL_WEIGHTS_FLAT[95]  = 0.2051984637212956039659241
_GL_WEIGHTS_FLAT[96]  = 0.2152638534631577901958764
_GL_WEIGHTS_FLAT[97]  = 0.2152638534631577901958764
_GL_WEIGHTS_FLAT[98]  = 0.2051984637212956039659241
_GL_WEIGHTS_FLAT[99]  = 0.1855383974779378137417166
_GL_WEIGHTS_FLAT[100] = 0.1572031671581935345696019
_GL_WEIGHTS_FLAT[101] = 0.1215185706879031846894148
_GL_WEIGHTS_FLAT[102] = 0.0801580871597602098056333
_GL_WEIGHTS_FLAT[103] = 0.0351194603317518630318329
# n=15
_GL_WEIGHTS_FLAT[104] = 0.0307532419961172683546284
_GL_WEIGHTS_FLAT[105] = 0.0703660474881081247092674
_GL_WEIGHTS_FLAT[106] = 0.1071592204671719350118695
_GL_WEIGHTS_FLAT[107] = 0.1395706779261543144478048
_GL_WEIGHTS_FLAT[108] = 0.1662692058169939335532009
_GL_WEIGHTS_FLAT[109] = 0.1861610000155622110268006
_GL_WEIGHTS_FLAT[110] = 0.1984314853271115764561183
_GL_WEIGHTS_FLAT[111] = 0.2025782419255612728806202
_GL_WEIGHTS_FLAT[112] = 0.1984314853271115764561183
_GL_WEIGHTS_FLAT[113] = 0.1861610000155622110268006
_GL_WEIGHTS_FLAT[114] = 0.1662692058169939335532009
_GL_WEIGHTS_FLAT[115] = 0.1395706779261543144478048
_GL_WEIGHTS_FLAT[116] = 0.1071592204671719350118695
_GL_WEIGHTS_FLAT[117] = 0.0703660474881081247092674
_GL_WEIGHTS_FLAT[118] = 0.0307532419961172683546284
# n=16 (numpy.polynomial.legendre.leggauss 참조)
_GL_WEIGHTS_FLAT[119] = 0.0271524594117541762106
_GL_WEIGHTS_FLAT[120] = 0.0622535239386474564816
_GL_WEIGHTS_FLAT[121] = 0.0951585116824926052770
_GL_WEIGHTS_FLAT[122] = 0.1246289712555340711830
_GL_WEIGHTS_FLAT[123] = 0.1495959888165767082135
_GL_WEIGHTS_FLAT[124] = 0.1691565193950026468883
_GL_WEIGHTS_FLAT[125] = 0.1826034150449236392877
_GL_WEIGHTS_FLAT[126] = 0.1894506104550686409471
_GL_WEIGHTS_FLAT[127] = 0.1894506104550686409471
_GL_WEIGHTS_FLAT[128] = 0.1826034150449236392877
_GL_WEIGHTS_FLAT[129] = 0.1691565193950026468883
_GL_WEIGHTS_FLAT[130] = 0.1495959888165767082135
_GL_WEIGHTS_FLAT[131] = 0.1246289712555340711830
_GL_WEIGHTS_FLAT[132] = 0.0951585116824926052770
_GL_WEIGHTS_FLAT[133] = 0.0622535239386474564816
_GL_WEIGHTS_FLAT[134] = 0.0271524594117541762106

# ================================================================== Dormand-Prince 8(7) 테이블
# 출처: Hairer, Nørsett, Wanner, "Solving ODEs I", 2nd Ed. 1993, Table II.5.4
# 13-stage. B8 = 8차 주 솔루션(bhat), B7 = 7차 오차 추정(b).
# 오차 = B7 - B8 사용.
#
# C 노드
cdef double _DP_C[13]
_DP_C[0]  = 0.0
_DP_C[1]  = 1.0/18.0
_DP_C[2]  = 1.0/12.0
_DP_C[3]  = 1.0/8.0
_DP_C[4]  = 5.0/16.0
_DP_C[5]  = 3.0/8.0
_DP_C[6]  = 59.0/400.0
_DP_C[7]  = 93.0/200.0
_DP_C[8]  = 5490023248.0/9719169821.0
_DP_C[9]  = 13.0/20.0
_DP_C[10] = 1201146811.0/1299019798.0
_DP_C[11] = 1.0
_DP_C[12] = 1.0

# A 계수 (하삼각)
cdef double _DP_A[13][13]
cdef int _dp_i, _dp_j
for _dp_i in range(13):
    for _dp_j in range(13):
        _DP_A[_dp_i][_dp_j] = 0.0

_DP_A[1][0]  =  1.0/18.0
_DP_A[2][0]  =  1.0/48.0
_DP_A[2][1]  =  1.0/16.0
_DP_A[3][0]  =  1.0/32.0
_DP_A[3][2]  =  3.0/32.0
_DP_A[4][0]  =  5.0/16.0
_DP_A[4][2]  = -75.0/64.0
_DP_A[4][3]  =  75.0/64.0
_DP_A[5][0]  =  3.0/80.0
_DP_A[5][3]  =  3.0/16.0
_DP_A[5][4]  =  3.0/20.0
_DP_A[6][0]  =  29443841.0/614563906.0
_DP_A[6][3]  =  77736538.0/692538347.0
_DP_A[6][4]  = -28693883.0/1125000000.0
_DP_A[6][5]  =  23124283.0/1800000000.0
_DP_A[7][0]  =  16016141.0/946692911.0
_DP_A[7][3]  =  61564180.0/158732637.0
_DP_A[7][4]  =  22789713.0/633445777.0
_DP_A[7][5]  =  545815736.0/2771057229.0
_DP_A[7][6]  = -180193667.0/1043307555.0
_DP_A[8][0]  =  39632708.0/573591083.0
_DP_A[8][3]  = -433636366.0/683701615.0
_DP_A[8][4]  = -421739975.0/2616292301.0
_DP_A[8][5]  =  100302831.0/723423059.0
_DP_A[8][6]  =  790204164.0/839813087.0
_DP_A[8][7]  =  800635310.0/3783071287.0
_DP_A[9][0]  =  246121993.0/1340847787.0
_DP_A[9][3]  = -37695042795.0/15268766246.0
_DP_A[9][4]  = -309121744.0/1061227803.0
_DP_A[9][5]  = -12992083.0/490766935.0
_DP_A[9][6]  =  6005943493.0/2108947869.0
_DP_A[9][7]  =  393006217.0/1396673457.0
_DP_A[9][8]  =  123872331.0/1001029789.0
_DP_A[10][0]  = -1028468189.0/846180014.0
_DP_A[10][3]  =  8478235783.0/508512852.0
_DP_A[10][4]  =  1311729495.0/1432422823.0
_DP_A[10][5]  = -10304129995.0/1701304382.0
_DP_A[10][6]  = -48777925059.0/3047939560.0
_DP_A[10][7]  =  15336726248.0/1032824649.0
_DP_A[10][8]  = -45442868181.0/3398467696.0
_DP_A[10][9]  =  3065993473.0/597172653.0
_DP_A[11][0]  =  185892177.0/718116043.0
_DP_A[11][3]  = -3185094517.0/667107341.0
_DP_A[11][4]  = -477755414.0/1098053517.0
_DP_A[11][5]  = -703635378.0/230739211.0
_DP_A[11][6]  =  5731566787.0/1027545527.0
_DP_A[11][7]  =  5232866602.0/850066563.0
_DP_A[11][8]  = -4093664535.0/808688257.0
_DP_A[11][9]  =  3962137247.0/1805957418.0
_DP_A[11][10] =  65686358.0/487910083.0
_DP_A[12][0]  =  403863854.0/491063109.0
_DP_A[12][3]  = -5765607357.0/1731183897.0
_DP_A[12][4]  =  1023671723.0/2166884888.0
_DP_A[12][5]  =  732969965.0/3827304573.0
_DP_A[12][6]  =  10700094551.0/10440212544.0
_DP_A[12][7]  = -3745099610.0/3360369832.0
_DP_A[12][8]  =  3010732700.0/3838820591.0
_DP_A[12][9]  = -16098663.0/1004006975.0
_DP_A[12][10] = -1553839.0/1121054474.0

# B8: 8차 주 솔루션 (bhat, Hairer 표기)
cdef double _DP_B8[13]
_DP_B8[0]  =  13451932.0/455176623.0
_DP_B8[1]  =  0.0
_DP_B8[2]  =  0.0
_DP_B8[3]  =  0.0
_DP_B8[4]  =  0.0
_DP_B8[5]  = -808719846.0/976000145.0
_DP_B8[6]  =  1757004468.0/5645159321.0
_DP_B8[7]  =  656045339.0/265891186.0
_DP_B8[8]  = -3867574721.0/1518517206.0
_DP_B8[9]  =  465885868.0/322736535.0
_DP_B8[10] =  53011238.0/667516719.0
_DP_B8[11] =  2.0/45.0
_DP_B8[12] =  0.0

# B7: 7차 오차 추정 (b, Hairer 표기)
cdef double _DP_B7[13]
_DP_B7[0]  =  14005451.0/335480064.0
_DP_B7[1]  =  0.0
_DP_B7[2]  =  0.0
_DP_B7[3]  =  0.0
_DP_B7[4]  =  0.0
_DP_B7[5]  = -59238493.0/1068277825.0
_DP_B7[6]  =  181606767.0/758867731.0
_DP_B7[7]  =  561292985.0/797845732.0
_DP_B7[8]  = -1041891430.0/1371343529.0
_DP_B7[9]  =  760417239.0/1151165299.0
_DP_B7[10] =  118820643.0/751138087.0
_DP_B7[11] = -528747749.0/2220607170.0
_DP_B7[12] =  1.0/4.0

# ================================================================== Adams 계수
# Adams-Bashforth: _AB_COEFFS[order][j], order=1..5
# 출처: Hairer et al., "Solving ODEs I", Table III.1.1
cdef double _AB_COEFFS[6][5]

_AB_COEFFS[1][0] =  1.0
_AB_COEFFS[1][1] =  0.0
_AB_COEFFS[1][2] =  0.0
_AB_COEFFS[1][3] =  0.0
_AB_COEFFS[1][4] =  0.0

_AB_COEFFS[2][0] =  1.5               # 3/2
_AB_COEFFS[2][1] = -0.5               # -1/2
_AB_COEFFS[2][2] =  0.0
_AB_COEFFS[2][3] =  0.0
_AB_COEFFS[2][4] =  0.0

_AB_COEFFS[3][0] =  1.9166666666666667   # 23/12
_AB_COEFFS[3][1] = -1.3333333333333333   # -16/12
_AB_COEFFS[3][2] =  0.4166666666666667   #  5/12
_AB_COEFFS[3][3] =  0.0
_AB_COEFFS[3][4] =  0.0

_AB_COEFFS[4][0] =  2.2916666666666667   # 55/24
_AB_COEFFS[4][1] = -2.4583333333333333   # -59/24
_AB_COEFFS[4][2] =  1.5416666666666667   # 37/24
_AB_COEFFS[4][3] = -0.375                # -9/24
_AB_COEFFS[4][4] =  0.0

_AB_COEFFS[5][0] =  2.6402777777777778   # 1901/720
_AB_COEFFS[5][1] = -3.8527777777777778   # -2774/720
_AB_COEFFS[5][2] =  3.6333333333333333   # 2616/720
_AB_COEFFS[5][3] = -1.7694444444444444   # -1274/720
_AB_COEFFS[5][4] =  0.3486111111111111   # 251/720

# Adams-Moulton: _AM_COEFFS[order][j], order=1..5
# j=0: implicit (f_{n+1}) 계수
cdef double _AM_COEFFS[6][5]

_AM_COEFFS[1][0] =  1.0
_AM_COEFFS[1][1] =  0.0
_AM_COEFFS[1][2] =  0.0
_AM_COEFFS[1][3] =  0.0
_AM_COEFFS[1][4] =  0.0

_AM_COEFFS[2][0] =  0.5               # 1/2
_AM_COEFFS[2][1] =  0.5               # 1/2
_AM_COEFFS[2][2] =  0.0
_AM_COEFFS[2][3] =  0.0
_AM_COEFFS[2][4] =  0.0

_AM_COEFFS[3][0] =  0.4166666666666667   #  5/12
_AM_COEFFS[3][1] =  0.6666666666666667   #  8/12
_AM_COEFFS[3][2] = -0.0833333333333333   # -1/12
_AM_COEFFS[3][3] =  0.0
_AM_COEFFS[3][4] =  0.0

_AM_COEFFS[4][0] =  0.375                #  9/24
_AM_COEFFS[4][1] =  0.7916666666666667   # 19/24
_AM_COEFFS[4][2] = -0.2083333333333333   # -5/24
_AM_COEFFS[4][3] =  0.0416666666666667   #  1/24
_AM_COEFFS[4][4] =  0.0

_AM_COEFFS[5][0] =  0.3486111111111111   # 251/720
_AM_COEFFS[5][1] =  0.8972222222222222   # 646/720
_AM_COEFFS[5][2] = -0.3666666666666667   # -264/720
_AM_COEFFS[5][3] =  0.1472222222222222   # 106/720
_AM_COEFFS[5][4] = -0.0263888888888889   # -19/720


# ================================================================== 헬퍼 함수

cdef object _resolve_callable(object f, str var):
    """
    f가 Python callable이면 그대로,
    PyExpr(evalf 메서드 보유)이면 evalf 기반 래퍼 반환.

    PyExpr.lambdify()는 복잡한 곱셈 표현식에서 파싱 버그가 있으므로
    evalf(**{var: x}) 방식으로 직접 수치 평가.
    """
    if hasattr(f, 'evalf'):
        # PyExpr: evalf 래퍼 (lambda로 var 이름 캡처)
        _var = var
        _f = f
        return lambda x, _f=_f, _v=_var: _f.evalf(**{_v: x})
    if callable(f):
        return f
    raise TypeError(
        f"f must be callable or have lambdify method, got {type(f).__name__}"
    )


cdef object _resolve_callable_2var(object f, str var1, str var2):
    """
    2변수 함수 f(var1, var2) 해석.
    PyExpr이면 evalf(**{var1: v1, var2: v2}) 래퍼 반환.
    callable이면 그대로 반환.
    """
    if hasattr(f, 'evalf'):
        _f = f
        _v1 = var1
        _v2 = var2
        return lambda v1, v2, _f=_f, _v1=_v1, _v2=_v2: _f.evalf(**{_v1: v1, _v2: v2})
    if callable(f):
        return f
    raise TypeError(
        f"f must be callable or have evalf method, got {type(f).__name__}"
    )


cdef double _call_f(object f, double x) except? -1.7976931348623157e+308:
    """Python callable 단일 호출."""
    return <double>f(x)


cdef double _simple_13(object f, double a, double b) except? -1.7976931348623157e+308:
    """
    3점 Simpson 1/3 내부 계산.
    I = (h/3) * [f(a) + 4*f(m) + f(b)], h = (b-a)/2
    """
    cdef double h = (b - a) * 0.5
    cdef double m = (a + b) * 0.5
    return (h / 3.0) * (_call_f(f, a) + 4.0 * _call_f(f, m) + _call_f(f, b))


cdef double _simple_38(object f, double a, double b) except? -1.7976931348623157e+308:
    """
    4점 Simpson 3/8 내부 계산.
    I = (3h/8) * [f(a) + 3*f(x1) + 3*f(x2) + f(b)], h = (b-a)/3
    """
    cdef double h = (b - a) / 3.0
    return (3.0 * h / 8.0) * (
        _call_f(f, a) +
        3.0 * _call_f(f, a + h) +
        3.0 * _call_f(f, a + 2.0 * h) +
        _call_f(f, b)
    )


cdef double _composite_13_raw(object f, double a, double b, int n) except? -1.7976931348623157e+308:
    """
    합성 Simpson 1/3 내부 계산 (n은 짝수, 검증 없음).
    Kahan 보상 누산 사용.
    """
    cdef double h = (b - a) / n
    cdef double s_odd_val = 0.0, s_odd_c = 0.0
    cdef double s_even_val = 0.0, s_even_c = 0.0
    cdef int i

    for i in range(1, n, 2):
        kahan_add(&s_odd_val, &s_odd_c, _call_f(f, a + i * h))
    for i in range(2, n - 1, 2):
        kahan_add(&s_even_val, &s_even_c, _call_f(f, a + i * h))

    return (h / 3.0) * (_call_f(f, a) + 4.0 * s_odd_val + 2.0 * s_even_val + _call_f(f, b))


cdef inline double _rk4_step(object f, double t, double y, double h) noexcept:
    """단일 RK4 step. Python callable 호출은 GIL 필요 (nogil 불가)."""
    cdef double k1 = f(t, y)
    cdef double k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    cdef double k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    cdef double k4 = f(t + h, y + h * k3)
    return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


cdef double _trapezoidal_recursive(object f, double a, double b,
                                    int level, double prev_T) except? -1.7976931348623157e+308:
    """
    Romberg 용 재귀적 사다리꼴 계산.
    level=0: T[0][0] = (b-a)/2 * (f(a) + f(b))
    level=i: T[i][0] = T[i-1][0]/2 + h_i * sum(f at new midpoints)
    """
    cdef int n_intervals = 1 << level  # 2^level
    cdef double h = (b - a) / n_intervals
    cdef double s_val = 0.0, s_c = 0.0
    cdef int i

    if level == 0:
        return (b - a) * 0.5 * (_call_f(f, a) + _call_f(f, b))

    # level >= 1: 홀수 인덱스(새 점)만 합산
    for i in range(1, n_intervals, 2):
        kahan_add(&s_val, &s_c, _call_f(f, a + i * h))

    return prev_T * 0.5 + h * s_val


# ================================================================== NumericalAnalysis 클래스

cdef class NumericalAnalysis:
    """
    수치 적분 알고리즘 모음.

    Simpson 적분법 8종을 제공하며, 향후 다른 수치해석 메서드 추가 예정.

    함수 입력:
        - Python callable: lambda, math.sin 등
        - PyExpr: math_library.laplace의 기호 표현식 (lambdify 자동 적용)

    Examples
    --------
    >>> import math
    >>> from math_library.numerical_analysis import NumericalAnalysis
    >>> na = NumericalAnalysis()
    >>> na.simpson_13(math.sin, 0, math.pi)   # ≈ 2.0
    """

    def __init__(self):
        from math_library import Differentiation
        self._differ = Differentiation()

    # ------------------------------------------------------------------ 1. simpson_13

    def simpson_13(self, f, double a, double b, *,
                   str var='t', bint return_error=False):
        """
        3점 단순 Simpson 1/3 규칙.

        I = (h/3) * [f(a) + 4*f(m) + f(b)],  h = (b-a)/2,  m = (a+b)/2

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        var          : str    PyExpr용 변수명 (기본 't')
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  a >= b
        TypeError   f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        cdef object _f = _resolve_callable(f, var)
        cdef double h = (b - a) * 0.5
        cdef double m = (a + b) * 0.5
        cdef double fa = _call_f(_f, a)
        cdef double fm = _call_f(_f, m)
        cdef double fb = _call_f(_f, b)
        cdef double I = (h / 3.0) * (fa + 4.0 * fm + fb)
        cdef double I_38, err

        if return_error:
            I_38 = _simple_38(_f, a, b)
            err = fabs(I - I_38) / 15.0
            return (I, err)
        return I

    # ------------------------------------------------------------------ 2. simpson_38

    def simpson_38(self, f, double a, double b, *,
                   str var='t', bint return_error=False):
        """
        4점 단순 Simpson 3/8 규칙.

        I = (3h/8) * [f(a) + 3*f(x1) + 3*f(x2) + f(b)],  h = (b-a)/3

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  a >= b
        TypeError   f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        cdef object _f = _resolve_callable(f, var)
        cdef double h = (b - a) / 3.0
        cdef double x1 = a + h
        cdef double x2 = a + 2.0 * h
        cdef double fa = _call_f(_f, a)
        cdef double f1 = _call_f(_f, x1)
        cdef double f2 = _call_f(_f, x2)
        cdef double fb = _call_f(_f, b)
        cdef double I = (3.0 * h / 8.0) * (fa + 3.0 * f1 + 3.0 * f2 + fb)
        cdef double I_13, err

        if return_error:
            I_13 = _simple_13(_f, a, b)
            err = fabs(I - I_13) / 15.0
            return (I, err)
        return I

    # ------------------------------------------------------------------ 3. composite_simpson_13

    def composite_simpson_13(self, f, double a, double b, int n, *,
                              str var='t', bint return_error=False):
        """
        합성 Simpson 1/3 규칙.

        I = (h/3) * [f(a) + 4*Σ_odd + 2*Σ_even + f(b)]
        h = (b-a)/n

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        n            : int    구간 수 (짝수, >= 2)
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  a >= b, n < 2, n % 2 != 0
        TypeError   f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        if n < 2:
            raise ValueError(f"composite_simpson_13 requires n >= 2, got n={n}")
        if n % 2 != 0:
            raise ValueError(
                f"composite_simpson_13 requires even n, got n={n}"
            )

        cdef object _f = _resolve_callable(f, var)
        cdef double I = _composite_13_raw(_f, a, b, n)
        cdef int n_half
        cdef double I_half, err

        if return_error:
            n_half = n // 2
            if n_half >= 2 and n_half % 2 == 0:
                I_half = _composite_13_raw(_f, a, b, n_half)
                err = fabs(I - I_half) / 15.0
            else:
                err = float('nan')
            return (I, err)
        return I

    # ------------------------------------------------------------------ 4. composite_simpson_38

    def composite_simpson_38(self, f, double a, double b, int n, *,
                              str var='t', bint return_error=False):
        """
        합성 Simpson 3/8 규칙.

        I = (3h/8) * [f(a) + 계수합 + f(b)]
        h = (b-a)/n
        계수: i%3 != 0 이면 3, i%3 == 0 (i≠0, i≠n) 이면 2

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        n            : int    구간 수 (3의 배수, >= 3)
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  a >= b, n < 3, n % 3 != 0
        TypeError   f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        if n < 3:
            raise ValueError(f"composite_simpson_38 requires n >= 3, got n={n}")
        if n % 3 != 0:
            raise ValueError(
                f"composite_simpson_38 requires n divisible by 3, got n={n}"
            )

        cdef object _f = _resolve_callable(f, var)
        cdef double h = (b - a) / n
        cdef double s3_val = 0.0, s3_c = 0.0
        cdef double s2_val = 0.0, s2_c = 0.0
        cdef int i

        for i in range(1, n):
            if i % 3 == 0:
                kahan_add(&s2_val, &s2_c, _call_f(_f, a + i * h))
            else:
                kahan_add(&s3_val, &s3_c, _call_f(_f, a + i * h))

        cdef double I = (3.0 * h / 8.0) * (
            _call_f(_f, a) + 3.0 * s3_val + 2.0 * s2_val + _call_f(_f, b)
        )
        cdef int n_coarse
        cdef double I_coarse, err

        if return_error:
            n_coarse = (n * 2) // 3
            if n_coarse >= 3 and n_coarse % 3 == 0:
                I_coarse = self.composite_simpson_38(f, a, b, n_coarse, var=var)
                err = fabs(I - I_coarse) / 63.0
            else:
                err = float('nan')
            return (I, err)
        return I

    # ------------------------------------------------------------------ 5. adaptive_simpson

    def adaptive_simpson(self, f, double a, double b, *,
                          double tol=1e-10, int max_depth=50,
                          str var='t', bint return_error=False):
        """
        적응형 Simpson 적분 (명시적 스택 기반).

        각 구간에서 Simpson 1/3을 계산하고 오차가 허용값을 초과하면
        구간을 절반으로 나누어 처리.

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        tol          : float  허용 오차 (> 0)
        max_depth    : int    최대 분할 깊이
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError   a >= b 또는 tol <= 0
        TypeError    f가 callable이 아닌 경우
        RuntimeError max_depth 초과
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got tol={tol}")

        cdef object _f = _resolve_callable(f, var)

        # 스택 항목: (aa, bb, tolerance, fa, fm, fb, S1, depth)
        cdef double fa_root = _call_f(_f, a)
        cdef double fm_root = _call_f(_f, (a + b) * 0.5)
        cdef double fb_root = _call_f(_f, b)
        cdef double S_root = ((b - a) * 0.5 / 3.0) * (fa_root + 4.0 * fm_root + fb_root)

        cdef list stack = [(a, b, tol, fa_root, fm_root, fb_root, S_root, 0)]
        cdef double total = 0.0
        cdef double total_err = 0.0
        cdef double aa, bb, tt, fa_i, fm_i, fb_i, S1
        cdef double mid_d, fa_l, fm_l, fb_l, fa_r, fm_r, fb_r
        cdef double S_l, S_r, S2, delta
        cdef int depth

        while stack:
            aa, bb, tt, fa_i, fm_i, fb_i, S1, depth = stack.pop()

            if depth > max_depth:
                raise RuntimeError(
                    f"adaptive_simpson exceeded max_depth={max_depth}, "
                    "reduce tol or increase max_depth"
                )

            mid_d = (aa + bb) * 0.5

            # 왼쪽 구간 [aa, mid]
            fa_l = fa_i
            fm_l = _call_f(_f, (aa + mid_d) * 0.5)
            fb_l = fm_i
            S_l = ((mid_d - aa) * 0.5 / 3.0) * (fa_l + 4.0 * fm_l + fb_l)

            # 오른쪽 구간 [mid, bb]
            fa_r = fm_i
            fm_r = _call_f(_f, (mid_d + bb) * 0.5)
            fb_r = fb_i
            S_r = ((bb - mid_d) * 0.5 / 3.0) * (fa_r + 4.0 * fm_r + fb_r)

            S2 = S_l + S_r
            delta = fabs(S2 - S1)

            if delta < 15.0 * tt or depth == max_depth:
                total += S2 + (S2 - S1) / 15.0
                total_err += delta / 15.0
            else:
                stack.append((aa, mid_d, tt * 0.5, fa_l, fm_l, fb_l, S_l, depth + 1))
                stack.append((mid_d, bb, tt * 0.5, fa_r, fm_r, fb_r, S_r, depth + 1))

        if return_error:
            return (total, total_err)
        return total

    # ------------------------------------------------------------------ 6. mixed_simpson

    def mixed_simpson(self, f, double a, double b, int n, *,
                      str var='t', bint return_error=False):
        """
        혼합 Simpson 적분.

        n이 짝수이면 합성 1/3, 홀수이면 앞 (n-3) 구간은 1/3, 끝 3 구간은 3/8.

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        n            : int    구간 수 (>= 2)
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  n < 2 또는 a >= b
        TypeError   f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        if n < 2:
            raise ValueError(f"mixed_simpson requires n >= 2, got n={n}")

        cdef object _f = _resolve_callable(f, var)
        cdef double I, I_half, err
        cdef double h, split
        cdef double I1, I2, I_total
        cdef int n_half

        if n % 2 == 0:
            I = _composite_13_raw(_f, a, b, n)
            if return_error:
                if n >= 4:
                    n_half = n // 2
                    if n_half % 2 != 0:
                        n_half += 1
                    if n_half >= 2:
                        I_half = _composite_13_raw(_f, a, b, n_half)
                        err = fabs(I - I_half) / 15.0
                    else:
                        err = float('nan')
                else:
                    err = float('nan')
                return (I, err)
            return I
        else:
            # 홀수: 첫 (n-3) 구간 1/3 + 마지막 3 구간 3/8
            # n이 홀수이고 n >= 3 이면 n-3은 항상 짝수(홀수-홀수=짝수)
            h = (b - a) / n
            split = a + (n - 3) * h

            if n == 3:
                I1 = 0.0
                I2 = _simple_38(_f, a, b)
            else:
                # n-3 >= 2 이고 n-3은 짝수 (n 홀수 → n-3 짝수)
                I1 = _composite_13_raw(_f, a, split, n - 3)
                I2 = _simple_38(_f, split, b)

            I_total = I1 + I2
            if return_error:
                return (I_total, float('nan'))
            return I_total

    # ------------------------------------------------------------------ 7. simpson_irregular

    def simpson_irregular(self, x_points, y_points, *,
                           bint return_error=False):
        """
        불균등 간격 Simpson 적분.

        3점 비균등 Simpson 공식:
          h0 = x1-x0, h1 = x2-x1
          I = (h0+h1)/6 * [(2-h1/h0)*f0 + (h0+h1)^2/(h0*h1)*f1 + (2-h0/h1)*f2]

        점의 수가 홀수(짝수 구간)이면 전체를 2구간씩 Simpson 적용.
        점의 수가 짝수(홀수 구간)이면 마지막 구간은 사다리꼴로 처리 후 합산.

        Parameters
        ----------
        x_points     : list of float  단조 증가 x 좌표
        y_points     : list of float  대응 함수값
        return_error : bool           True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  점 수 불일치, 점 수 < 3, 단조성 위반, NaN/Inf 포함
        """
        cdef int n_pts = len(x_points)
        if n_pts != len(y_points):
            raise ValueError(
                f"x_points and y_points must have the same length, "
                f"got {n_pts} and {len(y_points)}"
            )
        if n_pts < 3:
            raise ValueError(
                f"simpson_irregular requires >= 3 points, got {n_pts}"
            )

        # 유효성 검사
        cdef list xp = [float(x) for x in x_points]
        cdef list yp = [float(y) for y in y_points]
        cdef double xi, yi
        cdef int i

        for i in range(n_pts):
            xi = xp[i]
            yi = yp[i]
            if isnan(xi) or isinf(xi):
                raise ValueError(f"x_points[{i}]={xi} is NaN or Inf")
            if isnan(yi) or isinf(yi):
                raise ValueError(f"y_points[{i}]={yi} is NaN or Inf")

        for i in range(1, n_pts):
            if xp[i] <= xp[i-1]:
                raise ValueError(
                    f"x_points must be monotonically increasing, "
                    f"violation at index {i}: x[{i-1}]={xp[i-1]}, x[{i}]={xp[i]}"
                )

        # 적분 계산
        cdef double total_s = 0.0, total_c = 0.0
        cdef double x0, x1, x2, f0, f1, f2
        cdef double h0, h1, coeff0, coeff1, coeff2, contrib

        if n_pts % 2 == 1:
            # 홀수 점 수: 2구간씩 순서대로 Simpson 적용
            i = 0
            while i + 2 < n_pts:
                x0 = xp[i];   f0 = yp[i]
                x1 = xp[i+1]; f1 = yp[i+1]
                x2 = xp[i+2]; f2 = yp[i+2]
                h0 = x1 - x0
                h1 = x2 - x1
                coeff0 = 2.0 - h1 / h0
                coeff1 = (h0 + h1) * (h0 + h1) / (h0 * h1)
                coeff2 = 2.0 - h0 / h1
                contrib = (h0 + h1) / 6.0 * (coeff0 * f0 + coeff1 * f1 + coeff2 * f2)
                kahan_add(&total_s, &total_c, contrib)
                i += 2
        else:
            # 짝수 점 수: 끝 1구간을 사다리꼴로 처리, 나머지는 2구간씩 Simpson
            # [0..n-2] → n-2+1 = n-1개 점 → 홀수 점 수로 Simpson 가능 (if n-1 홀수)
            # 짝수 점: n-1 = 홀수 → [0..n-2] 구간에서 홀수 개 점 Simpson 적용
            # + 마지막 구간 [n-2, n-1] 사다리꼴
            i = 0
            while i + 2 < n_pts - 1:
                x0 = xp[i];   f0 = yp[i]
                x1 = xp[i+1]; f1 = yp[i+1]
                x2 = xp[i+2]; f2 = yp[i+2]
                h0 = x1 - x0
                h1 = x2 - x1
                coeff0 = 2.0 - h1 / h0
                coeff1 = (h0 + h1) * (h0 + h1) / (h0 * h1)
                coeff2 = 2.0 - h0 / h1
                contrib = (h0 + h1) / 6.0 * (coeff0 * f0 + coeff1 * f1 + coeff2 * f2)
                kahan_add(&total_s, &total_c, contrib)
                i += 2

            # 마지막 구간 [n-2, n-1] 사다리꼴
            x0 = xp[n_pts - 2]; f0 = yp[n_pts - 2]
            x1 = xp[n_pts - 1]; f1 = yp[n_pts - 1]
            kahan_add(&total_s, &total_c, (x1 - x0) * 0.5 * (f0 + f1))

        if return_error:
            return (total_s, float('nan'))
        return total_s

    # ------------------------------------------------------------------ 8. romberg

    def romberg(self, f, double a, double b, *,
                int depth=5, str var='t', bint return_error=False):
        """
        Romberg 적분 (Richardson 외삽 확장).

        T[i][0] = 사다리꼴 규칙 (2^i 구간)
        T[i][j] = (4^j * T[i][j-1] - T[i-1][j-1]) / (4^j - 1)

        T[depth][depth]이 최고 차수 근사.

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  (a < b)
        depth        : int    Richardson 테이블 깊이 (>= 1, 권장 <= 20)
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError   a >= b 또는 depth < 1
        TypeError    f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        if depth < 1:
            raise ValueError(f"romberg requires depth >= 1, got depth={depth}")
        if depth > 20:
            import warnings
            warnings.warn(
                f"romberg: depth={depth} > 20 may cause numerical instability",
                RuntimeWarning,
                stacklevel=2,
            )

        cdef object _f = _resolve_callable(f, var)

        # T 테이블: (depth+1) x (depth+1) — Python list of lists
        cdef int sz = depth + 1
        cdef list T = [[0.0] * sz for _ in range(sz)]
        cdef double prev_trap = 0.0
        cdef double fac, val
        cdef int i, j

        # T[i][0]: 사다리꼴 규칙
        for i in range(sz):
            prev_trap = _trapezoidal_recursive(_f, a, b, i, prev_trap)
            T[i][0] = prev_trap

        # Richardson 외삽
        for j in range(1, sz):
            fac = <double>(1 << (2 * j))  # 4^j
            for i in range(j, sz):
                val = (fac * <double>T[i][j-1] - <double>T[i-1][j-1]) / (fac - 1.0)
                T[i][j] = val

        cdef double result = <double>T[depth][depth]
        cdef double err

        if return_error:
            err = fabs(result - <double>T[depth][depth - 1]) if depth >= 1 else 0.0
            return (result, err)
        return result

    # ------------------------------------------------------------------ 9. newton_raphson

    def newton_raphson(self, f, double x0, *, fprime=None, str var='x',
                       double tol=1e-10, int max_iter=100, bint return_info=False):
        """
        Newton-Raphson 근 찾기.

        x_{n+1} = x_n - f(x_n) / f'(x_n)

        반복 종료: |x_new - x_old| < tol  또는  |f(x_new)| < tol

        Parameters
        ----------
        f         : callable 또는 PyExpr
        x0        : float  초기 추정값
        fprime    : callable, PyExpr, 또는 None.
                    None이면 Differentiation.single_variable(f, x)로 수치 미분 사용.
        var       : str    PyExpr 변수명 (기본 'x')
        tol       : float  수렴 허용 오차 (> 0)
        max_iter  : int    최대 반복 횟수 (> 0)
        return_info : bool  True이면 (root, iter_count, residual) 튜플 반환

        Returns
        -------
        float 또는 (float, int, float)

        Raises
        ------
        ValueError       tol <= 0 또는 max_iter <= 0
        TypeError        f가 callable이 아닌 경우
        ZeroDivisionError f'(x) == 0 (발산 위험)
        RuntimeError     max_iter 내 미수렴
        """
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got tol={tol}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got max_iter={max_iter}")

        cdef object _f = _resolve_callable(f, var)
        cdef object _fp

        if fprime is None:
            # Differentiation.single_variable 재활용
            _differ_ref = self._differ
            _fp = lambda x, _d=_differ_ref, _fn=_f: _d.single_variable(_fn, x)
        else:
            _fp = _resolve_callable(fprime, var)

        cdef double x = x0
        cdef double x_new, fx, fpx
        cdef int it

        for it in range(max_iter):
            fx = <double>_f(x)
            fpx = <double>_fp(x)
            if fpx == 0.0:
                raise ZeroDivisionError(
                    f"newton_raphson: f'(x)=0 at x={x}, iteration={it}"
                )
            x_new = x - fx / fpx
            if fabs(x_new - x) < tol or fabs(fx) < tol:
                x = x_new
                if return_info:
                    return (x, it + 1, fabs(<double>_f(x)))
                return x
            x = x_new
        else:
            raise RuntimeError(
                f"newton_raphson did not converge in {max_iter} iterations, "
                f"|f(x)|={fabs(<double>_f(x)):.2e}"
            )

    # ------------------------------------------------------------------ 10. secant_method

    def secant_method(self, f, double x0, double x1, *, str var='x',
                      double tol=1e-10, int max_iter=100, bint return_info=False):
        """
        Secant 근 찾기 (도함수 없음, 유한차분).

        x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

        Parameters
        ----------
        f         : callable 또는 PyExpr
        x0, x1    : float  초기 두 추정값
        var       : str    PyExpr 변수명 (기본 'x')
        tol       : float  수렴 허용 오차 (> 0)
        max_iter  : int    최대 반복 횟수 (> 0)
        return_info : bool True이면 (root, iter_count, residual) 반환

        Returns
        -------
        float 또는 (float, int, float)

        Raises
        ------
        ValueError       tol <= 0 또는 max_iter <= 0
        TypeError        f가 callable이 아닌 경우
        ZeroDivisionError f(x1) - f(x0) == 0
        RuntimeError     max_iter 내 미수렴
        """
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got tol={tol}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got max_iter={max_iter}")

        cdef object _f = _resolve_callable(f, var)
        cdef double xa = x0, xb = x1
        cdef double fa = <double>_f(xa)
        cdef double fb = <double>_f(xb)
        cdef double x_new, df
        cdef int it

        for it in range(max_iter):
            df = fb - fa
            if df == 0.0:
                raise ZeroDivisionError(
                    f"secant_method: f(x1)-f(x0)=0 at x0={xa}, x1={xb}, iteration={it}"
                )
            x_new = xb - fb * (xb - xa) / df
            xa = xb
            fa = fb
            xb = x_new
            fb = <double>_f(xb)
            if fabs(xb - xa) < tol or fabs(fb) < tol:
                if return_info:
                    return (xb, it + 1, fabs(fb))
                return xb
        else:
            raise RuntimeError(
                f"secant_method did not converge in {max_iter} iterations, "
                f"|f(x)|={fabs(fb):.2e}"
            )

    # ------------------------------------------------------------------ 11. euler

    def euler(self, f, double t0, double y0, double t_end, int n, *,
              vars=('t', 'y'), bint return_trajectory=False):
        """
        Euler 1차 ODE 적분.

        h = (t_end - t0) / n
        y_{n+1} = y_n + h * f(t_n, y_n)

        Parameters
        ----------
        f                : callable f(t, y) 또는 PyExpr (2변수)
        t0               : float  초기 시간
        y0               : float  초기 상태
        t_end            : float  종료 시간 (> t0)
        n                : int    스텝 수 (>= 1)
        vars             : tuple  PyExpr 변수명 (t변수명, y변수명). 기본 ('t', 'y')
        return_trajectory: bool   True이면 [(t0,y0), ..., (t_n,y_n)] 반환

        Returns
        -------
        float 또는 list of (float, float)

        Raises
        ------
        ValueError  n < 1 또는 t0 >= t_end
        TypeError   f가 callable이 아닌 경우
        """
        if n < 1:
            raise ValueError(f"euler requires n >= 1, got n={n}")
        if t0 >= t_end:
            raise ValueError(f"t0 must be less than t_end, got t0={t0}, t_end={t_end}")

        cdef object _f = _resolve_callable_2var(f, vars[0], vars[1])
        cdef double h = (t_end - t0) / n
        cdef double t = t0
        cdef double y = y0
        cdef int i
        cdef list traj

        if return_trajectory:
            traj = [(t, y)]
            for i in range(n):
                y = y + h * <double>_f(t, y)
                t = t0 + (i + 1) * h
                traj.append((t, y))
            return traj

        for i in range(n):
            y = y + h * <double>_f(t, y)
            t = t0 + (i + 1) * h
        return y

    # ------------------------------------------------------------------ 12. rk2

    def rk2(self, f, double t0, double y0, double t_end, int n, *,
            str method='midpoint', vars=('t', 'y'), bint return_trajectory=False):
        """
        2차 Runge-Kutta ODE 적분.

        method 선택:
          'midpoint' (기본): k1=f(t,y), k2=f(t+h/2, y+h*k1/2), y_new=y+h*k2
          'heun':            k1=f(t,y), k2=f(t+h, y+h*k1),     y_new=y+h*(k1+k2)/2
          'ralston':         k1=f(t,y), k2=f(t+2h/3, y+2h*k1/3), y_new=y+h*(k1/4+3*k2/4)

        Parameters
        ----------
        f                : callable f(t, y) 또는 PyExpr
        t0               : float  초기 시간
        y0               : float  초기 상태
        t_end            : float  종료 시간
        n                : int    스텝 수 (>= 1)
        method           : str    'midpoint', 'heun', 'ralston'
        vars             : tuple  PyExpr 변수명
        return_trajectory: bool

        Returns
        -------
        float 또는 list of (float, float)

        Raises
        ------
        ValueError  method가 3개 중 하나가 아님, n < 1, t0 >= t_end
        """
        if n < 1:
            raise ValueError(f"rk2 requires n >= 1, got n={n}")
        if t0 >= t_end:
            raise ValueError(f"t0 must be less than t_end, got t0={t0}, t_end={t_end}")
        if method not in ('midpoint', 'heun', 'ralston'):
            raise ValueError(
                f"rk2 method must be 'midpoint', 'heun', or 'ralston', got '{method}'"
            )

        cdef object _f = _resolve_callable_2var(f, vars[0], vars[1])
        cdef double h = (t_end - t0) / n
        cdef double t = t0
        cdef double y = y0
        cdef double k1, k2, y_new
        cdef int i
        cdef list traj

        if return_trajectory:
            traj = [(t, y)]

        for i in range(n):
            k1 = <double>_f(t, y)
            if method == 'midpoint':
                k2 = <double>_f(t + 0.5 * h, y + 0.5 * h * k1)
                y_new = y + h * k2
            elif method == 'heun':
                k2 = <double>_f(t + h, y + h * k1)
                y_new = y + h * (k1 + k2) * 0.5
            else:  # ralston
                k2 = <double>_f(t + (2.0 / 3.0) * h, y + (2.0 / 3.0) * h * k1)
                y_new = y + h * (0.25 * k1 + 0.75 * k2)
            y = y_new
            t = t0 + (i + 1) * h
            if return_trajectory:
                traj.append((t, y))

        if return_trajectory:
            return traj
        return y

    # ------------------------------------------------------------------ 13. rk4

    def rk4(self, f, double t0, double y0, double t_end, int n, *,
            vars=('t', 'y'), bint return_trajectory=False):
        """
        고전 4차 Runge-Kutta ODE 적분.

        k1 = f(t, y)
        k2 = f(t+h/2, y+h*k1/2)
        k3 = f(t+h/2, y+h*k2/2)
        k4 = f(t+h,   y+h*k3)
        y_new = y + h*(k1 + 2*k2 + 2*k3 + k4)/6

        Parameters
        ----------
        f                : callable f(t, y) 또는 PyExpr
        t0               : float  초기 시간
        y0               : float  초기 상태
        t_end            : float  종료 시간
        n                : int    스텝 수 (>= 1)
        vars             : tuple  PyExpr 변수명
        return_trajectory: bool

        Returns
        -------
        float 또는 list of (float, float)

        Raises
        ------
        ValueError  n < 1 또는 t0 >= t_end
        """
        if n < 1:
            raise ValueError(f"rk4 requires n >= 1, got n={n}")
        if t0 >= t_end:
            raise ValueError(f"t0 must be less than t_end, got t0={t0}, t_end={t_end}")

        cdef object _f = _resolve_callable_2var(f, vars[0], vars[1])
        cdef double h = (t_end - t0) / n
        cdef double t = t0
        cdef double y = y0
        cdef int i
        cdef list traj

        if return_trajectory:
            traj = [(t, y)]
            for i in range(n):
                y = _rk4_step(_f, t, y, h)
                t = t0 + (i + 1) * h
                traj.append((t, y))
            return traj

        for i in range(n):
            y = _rk4_step(_f, t, y, h)
            t = t0 + (i + 1) * h
        return y

    # ------------------------------------------------------------------ 14. rk45 (Dormand-Prince DOPRI5)

    def rk45(self, f, double t0, double y0, double t_end, *,
             double tol=1e-8, h_init=None, double h_min=1e-12,
             vars=('t', 'y'), bint return_trajectory=False, int max_steps=10000):
        """
        Dormand-Prince RK45 적응형 ODE 적분 (DOPRI5/FSAL).

        scipy/MATLAB ode45의 기본 알고리즘.
        5차 해와 4차 해의 차이로 오차 추정, step 크기 자동 조정.

        Parameters
        ----------
        f                : callable f(t, y) 또는 PyExpr
        t0               : float  초기 시간
        y0               : float  초기 상태
        t_end            : float  종료 시간
        tol              : float  허용 오차 (기본 1e-8)
        h_init           : float 또는 None. None이면 (t_end-t0)/100
        h_min            : float  최소 step 크기 (기본 1e-12)
        vars             : tuple  PyExpr 변수명
        return_trajectory: bool
        max_steps        : int    최대 스텝 수 (기본 10000)

        Returns
        -------
        float 또는 list of (float, float)

        Raises
        ------
        ValueError  t0 >= t_end
        RuntimeError max_steps 초과, h underflow
        """
        if t0 >= t_end:
            raise ValueError(f"t0 must be less than t_end, got t0={t0}, t_end={t_end}")

        cdef object _f = _resolve_callable_2var(f, vars[0], vars[1])

        # Dormand-Prince Butcher tableau 상수
        # c
        cdef double c2 = 0.2, c3 = 0.3, c4 = 0.8, c5 = 8.0/9.0
        # a (하삼각)
        cdef double a21 = 0.2
        cdef double a31 = 3.0/40.0,    a32 = 9.0/40.0
        cdef double a41 = 44.0/45.0,   a42 = -56.0/15.0,    a43 = 32.0/9.0
        cdef double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0
        cdef double a61 = 9017.0/3168.0,  a62 = -355.0/33.0,    a63 = 46732.0/5247.0
        cdef double a64 = 49.0/176.0,     a65 = -5103.0/18656.0
        # b (5차)
        cdef double b1 = 35.0/384.0,  b3 = 500.0/1113.0, b4 = 125.0/192.0
        cdef double b5 = -2187.0/6784.0, b6 = 11.0/84.0
        # b* (4차, 오차 추정용)
        cdef double e1 = 71.0/57600.0, e3 = -71.0/16695.0, e4 = 71.0/1920.0
        cdef double e5 = -17253.0/339200.0, e6 = 22.0/525.0, e7 = -1.0/40.0

        cdef double h = (t_end - t0) / 100.0 if h_init is None else <double>h_init
        cdef double t = t0
        cdef double y = y0
        cdef double k1, k2, k3, k4, k5, k6, k7
        cdef double y5, err_est
        cdef double h_new
        cdef int step_count = 0
        cdef list traj

        if return_trajectory:
            traj = [(t, y)]

        # FSAL: 첫 k1 계산
        k1 = <double>_f(t, y)

        while t < t_end:
            if step_count >= max_steps:
                raise RuntimeError(
                    f"rk45 exceeded max_steps={max_steps} at t={t:.6g}"
                )
            step_count += 1

            # 마지막 step 축소
            if t + h > t_end:
                h = t_end - t

            # k2 ~ k6
            k2 = <double>_f(t + c2 * h,  y + h * (a21 * k1))
            k3 = <double>_f(t + c3 * h,  y + h * (a31 * k1 + a32 * k2))
            k4 = <double>_f(t + c4 * h,  y + h * (a41 * k1 + a42 * k2 + a43 * k3))
            k5 = <double>_f(t + c5 * h,  y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
            k6 = <double>_f(t + h,        y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))

            # 5차 해
            y5 = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)

            # k7 (FSAL: 다음 step의 k1)
            k7 = <double>_f(t + h, y5)

            # 오차 추정 (5차 - 4차 = b - b*)
            # err = h * sum((b_i - b*_i) * k_i)  (단순화 공식)
            err_est = fabs(h * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7))

            if err_est < tol or h <= h_min:
                # step 수락
                t = t + h
                y = y5
                k1 = k7  # FSAL

                if return_trajectory:
                    traj.append((t, y))

            # step 크기 조정
            if err_est > 0.0:
                h_new = h * 0.9 * (tol / err_est) ** 0.2
                if h_new > 5.0 * h:
                    h_new = 5.0 * h
                if h_new < 0.1 * h:
                    h_new = 0.1 * h
                h = h_new
            # else: err_est == 0이면 h 유지

            if h < h_min and t < t_end:
                raise RuntimeError(
                    f"rk45: step size h={h:.2e} underflow at t={t:.6g}"
                )

        if return_trajectory:
            return traj
        return y

    # ------------------------------------------------------------------ 15. rk_fehlberg (RKF45)

    def rk_fehlberg(self, f, double t0, double y0, double t_end, *,
                    double tol=1e-8, h_init=None, double h_min=1e-12,
                    vars=('t', 'y'), bint return_trajectory=False, int max_steps=10000):
        """
        Fehlberg RKF45 적응형 ODE 적분.

        6-stage 5(4) 방법. rk45(DOPRI5)와 유사하나 다른 Butcher tableau.
        Fehlberg(1969) 원래 계수 사용.

        Parameters
        ----------
        f                : callable f(t, y) 또는 PyExpr
        t0               : float  초기 시간
        y0               : float  초기 상태
        t_end            : float  종료 시간
        tol              : float  허용 오차 (기본 1e-8)
        h_init           : float 또는 None
        h_min            : float  최소 step 크기
        vars             : tuple  PyExpr 변수명
        return_trajectory: bool
        max_steps        : int    최대 스텝 수

        Returns
        -------
        float 또는 list of (float, float)

        Raises
        ------
        ValueError   t0 >= t_end
        RuntimeError max_steps 초과, h underflow
        """
        if t0 >= t_end:
            raise ValueError(f"t0 must be less than t_end, got t0={t0}, t_end={t_end}")

        cdef object _f = _resolve_callable_2var(f, vars[0], vars[1])

        # Fehlberg Butcher tableau
        # c
        cdef double c2 = 0.25, c3 = 3.0/8.0, c4 = 12.0/13.0, c5 = 1.0, c6 = 0.5
        # a
        cdef double a21 = 0.25
        cdef double a31 = 3.0/32.0,      a32 = 9.0/32.0
        cdef double a41 = 1932.0/2197.0, a42 = -7200.0/2197.0, a43 = 7296.0/2197.0
        cdef double a51 = 439.0/216.0,   a52 = -8.0,           a53 = 3680.0/513.0,   a54 = -845.0/4104.0
        cdef double a61 = -8.0/27.0,     a62 = 2.0,            a63 = -3544.0/2565.0, a64 = 1859.0/4104.0, a65 = -11.0/40.0
        # b5 (5차 해)
        cdef double b1 = 16.0/135.0,  b3 = 6656.0/12825.0, b4 = 28561.0/56430.0, b5 = -9.0/50.0, b6 = 2.0/55.0
        # b4 (4차 해)
        cdef double d1 = 25.0/216.0,  d3 = 1408.0/2565.0,  d4 = 2197.0/4104.0,   d5 = -0.2

        cdef double h = (t_end - t0) / 100.0 if h_init is None else <double>h_init
        cdef double t = t0
        cdef double y = y0
        cdef double k1, k2, k3, k4, k5, k6
        cdef double y5, y4, err_est
        cdef double h_new
        cdef int step_count = 0
        cdef list traj

        if return_trajectory:
            traj = [(t, y)]

        while t < t_end:
            if step_count >= max_steps:
                raise RuntimeError(
                    f"rk_fehlberg exceeded max_steps={max_steps} at t={t:.6g}"
                )
            step_count += 1

            # 마지막 step 축소
            if t + h > t_end:
                h = t_end - t

            k1 = <double>_f(t,              y)
            k2 = <double>_f(t + c2 * h,     y + h * (a21 * k1))
            k3 = <double>_f(t + c3 * h,     y + h * (a31 * k1 + a32 * k2))
            k4 = <double>_f(t + c4 * h,     y + h * (a41 * k1 + a42 * k2 + a43 * k3))
            k5 = <double>_f(t + c5 * h,     y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
            k6 = <double>_f(t + c6 * h,     y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))

            # 5차 해
            y5 = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
            # 4차 해 (오차 추정용)
            y4 = y + h * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5)

            err_est = fabs(y5 - y4)

            if err_est < tol or h <= h_min:
                # step 수락 (5차 해 사용)
                t = t + h
                y = y5

                if return_trajectory:
                    traj.append((t, y))

            # step 크기 조정
            if err_est > 0.0:
                h_new = h * 0.9 * (tol / err_est) ** 0.2
                if h_new > 5.0 * h:
                    h_new = 5.0 * h
                if h_new < 0.1 * h:
                    h_new = 0.1 * h
                h = h_new

            if h < h_min and t < t_end:
                raise RuntimeError(
                    f"rk_fehlberg: step size h={h:.2e} underflow at t={t:.6g}"
                )

        if return_trajectory:
            return traj
        return y

    # ------------------------------------------------------------------ 16. gauss_legendre

    def gauss_legendre(self, f, double a, double b, int n=5, *,
                       str var='t', bint return_error=False):
        """
        n-point Gauss-Legendre 구적법.

        ∫_a^b f(x) dx ≈ (b-a)/2 · Σ_{i=0}^{n-1} w_i · f(x_i)

        구간 변환: x = (b-a)/2·u + (a+b)/2, u ∈ [-1, 1]
        (2n-1)차 다항식까지 정확 (n point GL은 2n-1차 다항식에 대해 정확).

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  적분 구간 (a < b)
        n            : int    적분점 수 (2 <= n <= 16, 기본 5)
        var          : str    PyExpr용 변수명 (기본 't')
        return_error : bool   True이면 (value, error_estimate) 반환.

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  a >= b, n < 2, n > 16
        TypeError   f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        if n < 2 or n > 16:
            raise ValueError(
                f"gauss_legendre supports n in [2, 16], got n={n}"
            )

        cdef object _f = _resolve_callable(f, var)
        cdef double half = 0.5 * (b - a)
        cdef double mid  = 0.5 * (a + b)
        cdef double s = 0.0, sc = 0.0
        cdef int i, offset = _GL_OFFSETS[n]

        for i in range(n):
            kahan_add(&s, &sc, _GL_WEIGHTS_FLAT[offset + i] * <double>_f(mid + half * _GL_NODES_FLAT[offset + i]))

        cdef double result = half * s
        cdef double s2 = 0.0, sc2 = 0.0, result2, err
        cdef int offset2, n2

        if return_error:
            n2 = n - 1 if n > 2 else 2
            offset2 = _GL_OFFSETS[n2]
            for i in range(n2):
                kahan_add(&s2, &sc2, _GL_WEIGHTS_FLAT[offset2 + i] * <double>_f(mid + half * _GL_NODES_FLAT[offset2 + i]))
            result2 = half * s2
            err = fabs(result - result2) / 100.0 if n > 2 else 0.0
            return (result, err)
        return result

    # ------------------------------------------------------------------ 17. composite_gauss_legendre

    def composite_gauss_legendre(self, f, double a, double b, int m, int n=5, *,
                                  str var='t', bint return_error=False):
        """
        합성 Gauss-Legendre 적분.

        [a, b]를 m개 구간으로 균등 분할, 각 구간에 n-point GL 적용.

        Parameters
        ----------
        f            : callable 또는 PyExpr
        a, b         : float  적분 구간 (a < b)
        m            : int    구간 분할 수 (>= 1)
        n            : int    각 구간 적분점 수 (2 <= n <= 16, 기본 5)
        var          : str    PyExpr용 변수명
        return_error : bool   True이면 (value, error_estimate) 반환.

        Returns
        -------
        float 또는 (float, float)

        Raises
        ------
        ValueError  a >= b, m < 1, n 범위 위반
        TypeError   f가 callable이 아닌 경우
        """
        if a >= b:
            raise ValueError(f"a must be less than b, got a={a}, b={b}")
        if m < 1:
            raise ValueError(f"composite_gauss_legendre requires m >= 1, got m={m}")
        if n < 2 or n > 16:
            raise ValueError(
                f"gauss_legendre supports n in [2, 16], got n={n}"
            )

        cdef object _f = _resolve_callable(f, var)
        cdef double h = (b - a) / m
        cdef double total = 0.0, tc = 0.0
        cdef double sub_a, sub_b, half_sub, mid_sub, s, sc
        cdef int i, j, offset = _GL_OFFSETS[n]

        for i in range(m):
            sub_a    = a + i * h
            sub_b    = sub_a + h
            half_sub = 0.5 * h
            mid_sub  = 0.5 * (sub_a + sub_b)
            s = 0.0; sc = 0.0
            for j in range(n):
                kahan_add(&s, &sc, _GL_WEIGHTS_FLAT[offset + j] * <double>_f(mid_sub + half_sub * _GL_NODES_FLAT[offset + j]))
            kahan_add(&total, &tc, half_sub * s)

        cdef double total2 = 0.0, tc2 = 0.0, h2, err_cgl
        cdef int m2, i2, j2

        if return_error:
            if m >= 2:
                m2 = m // 2
                h2 = (b - a) / m2
                for i2 in range(m2):
                    sub_a    = a + i2 * h2
                    sub_b    = sub_a + h2
                    half_sub = 0.5 * h2
                    mid_sub  = 0.5 * (sub_a + sub_b)
                    s = 0.0; sc = 0.0
                    for j2 in range(n):
                        kahan_add(&s, &sc, _GL_WEIGHTS_FLAT[offset + j2] * <double>_f(mid_sub + half_sub * _GL_NODES_FLAT[offset + j2]))
                    kahan_add(&total2, &tc2, half_sub * s)
                err_cgl = fabs(total - total2) / 100.0
            else:
                err_cgl = 0.0
            return (total, err_cgl)
        return total

    # ------------------------------------------------------------------ 18. rk78 (Dormand-Prince 8(7))

    def rk78(self, f, double t0, double y0, double t_end, *,
             double tol=1e-12, h_init=None, double h_min=1e-14,
             vars=('t', 'y'), bint return_trajectory=False, int max_steps=10000):
        """
        Dormand-Prince 8(7) 적응형 ODE 적분 (13-stage, FSAL).

        8차 해와 7차 해의 차이로 오차 추정, step 크기 자동 조정.

        출처: Hairer, Nørsett, Wanner, "Solving ODEs I", 2nd Ed. 1993,
              Table II.5.4 (Dormand-Prince, 1980).

        Parameters
        ----------
        f                : callable f(t, y) 또는 PyExpr
        t0               : float  초기 시간
        y0               : float  초기 상태
        t_end            : float  종료 시간 (> t0)
        tol              : float  허용 오차 (기본 1e-12)
        h_init           : float 또는 None. None이면 (t_end-t0)/100
        h_min            : float  최소 step 크기 (기본 1e-14)
        vars             : tuple  PyExpr 변수명
        return_trajectory: bool   True이면 [(t,y), ...] 궤적 반환
        max_steps        : int    최대 step 수 (기본 10000)

        Returns
        -------
        float 또는 list of (float, float)

        Raises
        ------
        ValueError   t0 >= t_end
        RuntimeError max_steps 초과, h underflow
        """
        if t0 >= t_end:
            raise ValueError(f"t0 must be less than t_end, got t0={t0}, t_end={t_end}")

        cdef object _f = _resolve_callable_2var(f, vars[0], vars[1])

        cdef double h = (t_end - t0) / 100.0 if h_init is None else <double>h_init
        cdef double t = t0
        cdef double y = y0
        cdef double k[13]
        cdef double y_sum, y8, y7, err_est, h_new
        cdef int step_count = 0
        cdef int si, sj
        cdef list traj_78

        if return_trajectory:
            traj_78 = [(t, y)]

        # FSAL: k[0] 초기화
        k[0] = <double>_f(t, y)

        while t < t_end:
            if step_count >= max_steps:
                raise RuntimeError(
                    f"rk78 exceeded max_steps={max_steps} at t={t:.6g}"
                )
            step_count += 1

            if t + h > t_end:
                h = t_end - t

            # k[1] ~ k[11] 계산
            for si in range(1, 12):
                y_sum = y
                for sj in range(si):
                    y_sum = y_sum + h * _DP_A[si][sj] * k[sj]
                k[si] = <double>_f(t + _DP_C[si] * h, y_sum)

            # 8차 해
            y8 = y
            for si in range(12):
                y8 = y8 + h * _DP_B8[si] * k[si]

            # k[12] (FSAL 후보)
            k[12] = <double>_f(t + h, y8)

            # 7차 해
            y7 = y
            for si in range(13):
                y7 = y7 + h * _DP_B7[si] * k[si]

            err_est = fabs(y8 - y7)

            if err_est < tol or h <= h_min:
                t = t + h
                y = y8
                k[0] = k[12]  # FSAL
                if return_trajectory:
                    traj_78.append((t, y))

            if err_est > 0.0:
                h_new = h * 0.9 * (tol / err_est) ** 0.125
                if h_new > 5.0 * h:
                    h_new = 5.0 * h
                if h_new < 0.1 * h:
                    h_new = 0.1 * h
                h = h_new

            if h < h_min and t < t_end:
                raise RuntimeError(
                    f"rk78: step size h={h:.2e} underflow at t={t:.6g}"
                )

        if return_trajectory:
            return traj_78
        return y

    # ------------------------------------------------------------------ 19. adams_bashforth

    def adams_bashforth(self, f, double t0, double y0, double t_end, int n, *,
                        int order=4, vars=('t', 'y'), bint return_trajectory=False):
        """
        Adams-Bashforth explicit multistep ODE 적분.

        y_{i+1} = y_i + h · Σ_{j=0}^{k-1} coeffs[j] · f_{i-j}

        부트스트랩: 첫 (order-1) step은 RK4로 계산.

        Parameters
        ----------
        f                : callable f(t, y) 또는 PyExpr
        t0               : float  초기 시간
        y0               : float  초기 상태
        t_end            : float  종료 시간 (> t0)
        n                : int    전체 step 수 (>= order)
        order            : int    방법의 차수 (1 <= order <= 5, 기본 4)
        vars             : tuple  PyExpr 변수명
        return_trajectory: bool

        Returns
        -------
        float 또는 list of (float, float)

        Raises
        ------
        ValueError   order < 1, order > 5, n < order, t0 >= t_end
        """
        if t0 >= t_end:
            raise ValueError(f"t0 must be less than t_end, got t0={t0}, t_end={t_end}")
        if order < 1 or order > 5:
            raise ValueError(f"adams_bashforth supports order in [1, 5], got order={order}")
        if n < order:
            raise ValueError(f"adams_bashforth requires n >= order={order}, got n={n}")

        cdef object _f = _resolve_callable_2var(f, vars[0], vars[1])
        cdef double h = (t_end - t0) / n
        cdef double y_next_ab, dy_ab, t_next_ab
        cdef int i_ab, j_ab
        cdef list y_hist_ab = [y0]
        cdef list f_hist_ab = [<double>_f(t0, y0)]
        cdef list traj_ab

        if return_trajectory:
            traj_ab = [(t0, y0)]

        for i_ab in range(order - 1):
            t_cur = t0 + i_ab * h
            y_cur = y_hist_ab[i_ab]
            y_next_ab = _rk4_step(_f, t_cur, y_cur, h)
            t_next_ab = t_cur + h
            y_hist_ab.append(y_next_ab)
            f_hist_ab.append(<double>_f(t_next_ab, y_next_ab))
            if return_trajectory:
                traj_ab.append((t_next_ab, y_next_ab))

        for i_ab in range(order - 1, n):
            y_cur = y_hist_ab[i_ab]
            dy_ab = 0.0
            for j_ab in range(order):
                dy_ab = dy_ab + _AB_COEFFS[order][j_ab] * f_hist_ab[i_ab - j_ab]
            y_next_ab = y_cur + h * dy_ab
            t_next_ab = t0 + (i_ab + 1) * h
            y_hist_ab.append(y_next_ab)
            f_hist_ab.append(<double>_f(t_next_ab, y_next_ab))
            if return_trajectory:
                traj_ab.append((t_next_ab, y_next_ab))

        if return_trajectory:
            return traj_ab
        return y_hist_ab[n]

    # ------------------------------------------------------------------ 20. adams_moulton

    def adams_moulton(self, f, double t0, double y0, double t_end, int n, *,
                      int order=4, vars=('t', 'y'), bint return_trajectory=False,
                      int max_iter=50, double tol=1e-12):
        """
        Adams-Moulton implicit multistep ODE 적분.

        y_{i+1} = y_i + h · [c_0·f(t_{i+1}, y_{i+1}) + c_1·f_i + ...]
        Predictor: Adams-Bashforth (동일 order)
        Corrector: fixed-point iteration

        Parameters
        ----------
        f                : callable f(t, y) 또는 PyExpr
        t0               : float  초기 시간
        y0               : float  초기 상태
        t_end            : float  종료 시간 (> t0)
        n                : int    전체 step 수 (>= order)
        order            : int    방법의 차수 (1 <= order <= 5, 기본 4)
        vars             : tuple  PyExpr 변수명
        return_trajectory: bool
        max_iter         : int    fixed-point 최대 반복 (기본 50)
        tol              : float  fixed-point 수렴 허용 오차 (기본 1e-12)

        Returns
        -------
        float 또는 list of (float, float)

        Raises
        ------
        ValueError   order < 1, order > 5, n < order, t0 >= t_end
        RuntimeError fixed-point 미수렴
        """
        if t0 >= t_end:
            raise ValueError(f"t0 must be less than t_end, got t0={t0}, t_end={t_end}")
        if order < 1 or order > 5:
            raise ValueError(f"adams_moulton supports order in [1, 5], got order={order}")
        if n < order:
            raise ValueError(f"adams_moulton requires n >= order={order}, got n={n}")

        cdef object _f = _resolve_callable_2var(f, vars[0], vars[1])
        cdef double h = (t_end - t0) / n
        cdef double y_next_am, y_pred_am, y_curr_am, f_new_am, expl_sum_am
        cdef double t_next_am
        cdef int i_am, j_am, it_am
        cdef list y_hist_am = [y0]
        cdef list f_hist_am = [<double>_f(t0, y0)]
        cdef list traj_am

        if return_trajectory:
            traj_am = [(t0, y0)]

        for i_am in range(order - 1):
            t_cur = t0 + i_am * h
            y_cur = y_hist_am[i_am]
            y_next_am = _rk4_step(_f, t_cur, y_cur, h)
            t_next_am = t_cur + h
            y_hist_am.append(y_next_am)
            f_hist_am.append(<double>_f(t_next_am, y_next_am))
            if return_trajectory:
                traj_am.append((t_next_am, y_next_am))

        for i_am in range(order - 1, n):
            t_next_am = t0 + (i_am + 1) * h
            y_cur = y_hist_am[i_am]

            # Predictor (AB)
            y_pred_am = y_cur
            for j_am in range(order):
                y_pred_am = y_pred_am + h * _AB_COEFFS[order][j_am] * f_hist_am[i_am - j_am]

            # explicit 부분: c_1*f_i + c_2*f_{i-1} + ...
            expl_sum_am = 0.0
            for j_am in range(1, order):
                expl_sum_am = expl_sum_am + _AM_COEFFS[order][j_am] * f_hist_am[i_am - (j_am - 1)]

            # Corrector: fixed-point
            y_curr_am = y_pred_am
            for it_am in range(max_iter):
                f_new_am = <double>_f(t_next_am, y_curr_am)
                y_next_am = y_cur + h * (_AM_COEFFS[order][0] * f_new_am + expl_sum_am)
                if fabs(y_next_am - y_curr_am) < tol:
                    y_curr_am = y_next_am
                    break
                y_curr_am = y_next_am
            else:
                raise RuntimeError(
                    f"adams_moulton did not converge at t={t_next_am:.6g} "
                    f"(max_iter={max_iter}, |delta|={fabs(y_next_am - y_curr_am):.2e})"
                )

            y_hist_am.append(y_curr_am)
            f_hist_am.append(<double>_f(t_next_am, y_curr_am))
            if return_trajectory:
                traj_am.append((t_next_am, y_curr_am))

        if return_trajectory:
            return traj_am
        return y_hist_am[n]

    # ------------------------------------------------------------------ 21. predictor_corrector (PECE)

    def predictor_corrector(self, f, double t0, double y0, double t_end, int n, *,
                            int order=4, vars=('t', 'y'), bint return_trajectory=False):
        """
        PECE Adams Predictor-Corrector ODE 적분.

        각 step에서 PECE 1회 수행:
          P: Adams-Bashforth (order k) → y_p
          E: f_p = f(t_{i+1}, y_p)
          C: Adams-Moulton (order k, 1회) → y_c
          E: f_c = f(t_{i+1}, y_c)

        부트스트랩: 첫 (order-1) step은 RK4.

        Parameters
        ----------
        f                : callable f(t, y) 또는 PyExpr
        t0               : float  초기 시간
        y0               : float  초기 상태
        t_end            : float  종료 시간 (> t0)
        n                : int    전체 step 수 (>= order)
        order            : int    방법의 차수 (2 <= order <= 5, 기본 4)
        vars             : tuple  PyExpr 변수명
        return_trajectory: bool

        Returns
        -------
        float 또는 list of (float, float)

        Raises
        ------
        ValueError   order < 2, order > 5, n < order, t0 >= t_end
        """
        if t0 >= t_end:
            raise ValueError(f"t0 must be less than t_end, got t0={t0}, t_end={t_end}")
        if order < 2 or order > 5:
            raise ValueError(f"predictor_corrector supports order in [2, 5], got order={order}")
        if n < order:
            raise ValueError(f"predictor_corrector requires n >= order={order}, got n={n}")

        cdef object _f = _resolve_callable_2var(f, vars[0], vars[1])
        cdef double h = (t_end - t0) / n
        cdef double y_p_pc, f_p_pc, y_c_pc, f_c_pc, expl_sum_pc
        cdef double t_next_pc, t_cur_pc
        cdef int i_pc, j_pc
        cdef list y_hist_pc = [y0]
        cdef list f_hist_pc = [<double>_f(t0, y0)]
        cdef list traj_pc

        if return_trajectory:
            traj_pc = [(t0, y0)]

        for i_pc in range(order - 1):
            t_cur_pc = t0 + i_pc * h
            y_bs = y_hist_pc[i_pc]
            y_bs_next = _rk4_step(_f, t_cur_pc, y_bs, h)
            t_bs_next = t_cur_pc + h
            y_hist_pc.append(y_bs_next)
            f_hist_pc.append(<double>_f(t_bs_next, y_bs_next))
            if return_trajectory:
                traj_pc.append((t_bs_next, y_bs_next))

        for i_pc in range(order - 1, n):
            t_cur_pc = t0 + i_pc * h
            t_next_pc = t_cur_pc + h
            y_cur_pc = y_hist_pc[i_pc]

            # P: Predict (Adams-Bashforth)
            y_p_pc = y_cur_pc
            for j_pc in range(order):
                y_p_pc = y_p_pc + h * _AB_COEFFS[order][j_pc] * f_hist_pc[i_pc - j_pc]

            # E: Evaluate predictor
            f_p_pc = <double>_f(t_next_pc, y_p_pc)

            # C: Correct (Adams-Moulton, 1회)
            expl_sum_pc = 0.0
            for j_pc in range(1, order):
                expl_sum_pc = expl_sum_pc + _AM_COEFFS[order][j_pc] * f_hist_pc[i_pc - (j_pc - 1)]
            y_c_pc = y_cur_pc + h * (_AM_COEFFS[order][0] * f_p_pc + expl_sum_pc)

            # E: Evaluate corrected
            f_c_pc = <double>_f(t_next_pc, y_c_pc)

            y_hist_pc.append(y_c_pc)
            f_hist_pc.append(f_c_pc)
            if return_trajectory:
                traj_pc.append((t_next_pc, y_c_pc))

        if return_trajectory:
            return traj_pc
        return y_hist_pc[n]
