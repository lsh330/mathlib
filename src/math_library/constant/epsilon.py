def epsilon() -> float:
    """
    Return the machine epsilon for Python's floating-point arithmetic.
    """

    eps = 1.0

    # Repeatedly halve eps until adding eps / 2 no longer changes 1.0
    while 1.0 + eps / 2.0 != 1.0:
        eps /= 2.0

    return eps