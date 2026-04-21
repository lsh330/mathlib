# _kahan.pxd — Kahan 보상 누산 인라인 유틸리티
# Neumaier 변형: 부동소수점 오차를 보상하여 대규모 합산 정확도 향상
# nogil 선언이지만, Python callable 루프 내에서 호출 시 GIL 블록 외부에서 사용

cdef inline void kahan_add(double* s, double* c, double value) noexcept nogil:
    """
    Kahan 보상 덧셈 (Neumaier 변형).

    Parameters
    ----------
    s     : 누산기 (현재 합)
    c     : 보상 항 (오차 누적)
    value : 더할 값

    Note
    ----
    y = value - c[0]        # 보상 후 값
    t = s[0] + y            # 임시 합
    c[0] = (t - s[0]) - y   # 반올림 오차 추출
    s[0] = t                # 업데이트
    """
    cdef double y = value - c[0]
    cdef double t = s[0] + y
    c[0] = (t - s[0]) - y
    s[0] = t
