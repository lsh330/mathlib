from math_library.legendre_function import legendre_polynomial


PASS_COUNT = 0
FAIL_COUNT = 0


def _close(actual, expected, tol=1e-7):
    return abs(complex(actual) - complex(expected)) <= tol


def _pass(message):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {message}")


def _fail(message):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {message}")


def _check_value(name, actual, expected, tol=1e-7):
    if _close(actual, expected, tol):
        _pass(f"{name} -> {actual}")
    else:
        _fail(f"{name} -> actual={actual}, expected={expected}, tol={tol}")


def _check_type(name, value, expected_type):
    if isinstance(value, expected_type):
        _pass(f"{name} type -> {expected_type.__name__}")
    else:
        _fail(f"{name} type -> actual={type(value).__name__}, expected={expected_type.__name__}")


def _check_raises(name, fn, expected_exc):
    try:
        fn()
        _fail(f"{name} -> expected {expected_exc.__name__}, but no exception")
    except expected_exc:
        _pass(f"{name} -> raised {expected_exc.__name__}")
    except Exception as exc:
        _fail(f"{name} -> wrong exception {type(exc).__name__}: {exc}")


def run_legendre_tests():
    print("=" * 80)
    print("LEGENDRE CATEGORY TESTS")
    print("=" * 80)

    _check_value("P0(0.2)", legendre_polynomial(0, 0.2), 1.0)
    _check_value("P1(0.2)", legendre_polynomial(1, 0.2), 0.2)
    _check_value("P2(0.2)", legendre_polynomial(2, 0.2), -0.44)
    _check_value("P3(0.5)", legendre_polynomial(3, 0.5), -0.4375)

    v = legendre_polynomial(2, 1 + 1j, number_system="complex")
    _check_type("P2(1+i, complex)", v, complex)

    _check_raises("P(-1, 0)", lambda: legendre_polynomial(-1, 0), ValueError)
    _check_raises("P(2.2, 0)", lambda: legendre_polynomial(2.2, 0), TypeError)
    _check_raises("P(2, 1+1j) real mode", lambda: legendre_polynomial(2, 1 + 1j), TypeError)
    _check_raises("P(True, 0)", lambda: legendre_polynomial(True, 0), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Legendre tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_legendre_tests()
