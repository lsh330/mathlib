from .epsilon import epsilon


def pi() -> float:
    """
    Compute pi using the BBP formula.
    """

    eps = epsilon()
    target_error = eps / 2.0

    s = 0.0
    inv16k = 1.0
    k = 0

    while True:
        # Current BBP term
        term = inv16k * (
            4.0 / (8 * k + 1)
            - 2.0 / (8 * k + 4)
            - 1.0 / (8 * k + 5)
            - 1.0 / (8 * k + 6)
        )
        s += term

        # Next BBP term
        next_inv16k = inv16k / 16.0
        next_k = k + 1
        next_term = next_inv16k * (
            4.0 / (8 * next_k + 1)
            - 2.0 / (8 * next_k + 4)
            - 1.0 / (8 * next_k + 5)
            - 1.0 / (8 * next_k + 6)
        )

        # Estimate an upper bound for the remaining tail
        tail_bound = abs(next_term) * (16.0 / 15.0)

        if tail_bound < target_error:
            return s

        inv16k = next_inv16k
        k += 1