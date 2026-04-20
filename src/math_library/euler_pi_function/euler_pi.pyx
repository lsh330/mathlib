# cython: language_level=3
# cython: embedsignature=True
#
# euler_pi.pyx — Euler totient + Prime counting (기존 알고리즘 유지)


def _validate_non_negative_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def euler_phi(n) -> int:
    """
    Euler's totient function phi(n).
    """
    n_value = _validate_non_negative_integer("n", n)

    if n_value == 0:
        return 0
    if n_value == 1:
        return 1

    result = n_value
    m = n_value
    p = 2

    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            result -= result // p
        p += 1

    if m > 1:
        result -= result // m

    return result


def euler_pi(n) -> int:
    """
    Prime-counting function pi(n): number of primes <= n.
    """
    n_value = _validate_non_negative_integer("n", n)

    if n_value < 2:
        return 0

    sieve = [True] * (n_value + 1)
    sieve[0] = False
    sieve[1] = False

    p = 2
    while p * p <= n_value:
        if sieve[p]:
            multiple = p * p
            while multiple <= n_value:
                sieve[multiple] = False
                multiple += p
        p += 1

    count = 0
    for is_prime in sieve:
        if is_prime:
            count += 1

    return count
