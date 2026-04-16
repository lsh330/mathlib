from .epsilon import epsilon


def e():
    """
    Compute e using the factorial series expansion.
    """

    eps = epsilon()

    s = 1.0
    term = 1.0
    n = 0

    while True:
        n += 1

        # Update the current term: 1 / n!
        term /= n
        s += term

        # Estimate an upper bound for the remaining tail
        next_term = term / (n + 1)
        tail_bound = next_term * (n + 2) / (n + 1)

        if tail_bound < eps:
            return s