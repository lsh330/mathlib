from math_library.bessel_function import bessel_j, bessel_j0, bessel_j1


PASS_COUNT = 0
FAIL_COUNT = 0


def _close(actual, expected, tol=1e-6):
    return abs(complex(actual) - complex(expected)) <= tol


def _pass(message):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {message}")


def _fail(message):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {message}")


def _check_value(name, actual, expected, tol=1e-6):
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


def run_bessel_tests():
    print("=" * 80)
    print("BESSEL CATEGORY TESTS")
    print("=" * 80)

    _check_value("J0(0)", bessel_j0(0), 1.0)
    _check_value("J1(0)", bessel_j1(0), 0.0)
    _check_value("J2(0)", bessel_j(2, 0), 0.0)

    _check_value("J0(1)", bessel_j0(1), 0.7651976866, tol=1e-5)
    _check_value("J1(1)", bessel_j1(1), 0.4400505857, tol=1e-5)

    _check_type("J0(1)", bessel_j0(1), float)

    c = bessel_j(1, 1 + 1j, number_system="complex")
    _check_type("J1(1+i, complex)", c, complex)

    _check_raises("J(-1, 1)", lambda: bessel_j(-1, 1), ValueError)
    _check_raises("J(1.5, 1)", lambda: bessel_j(1.5, 1), TypeError)
    _check_raises("J(1, 1+1j) real mode", lambda: bessel_j(1, 1 + 1j), TypeError)
    _check_raises("J(1, 'x')", lambda: bessel_j(1, "x"), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Bessel tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_bessel_tests()
