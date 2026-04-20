from math_library.constant import pi
from math_library.zeta_function import zeta


PASS_COUNT = 0
FAIL_COUNT = 0


def _close(actual, expected, tol=1e-5):
    return abs(complex(actual) - complex(expected)) <= tol


def _pass(message):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {message}")


def _fail(message):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {message}")


def _check_value(name, actual, expected, tol=1e-5):
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


def run_zeta_tests():
    print("=" * 80)
    print("ZETA CATEGORY TESTS")
    print("=" * 80)

    _check_value("zeta(2)", zeta(2), (pi() * pi()) / 6.0, tol=3e-4)
    _check_value("zeta(0)", zeta(0), -0.5, tol=1e-6)
    _check_value("zeta(-2)", zeta(-2), 0.0, tol=1e-5)
    _check_type("zeta(2)", zeta(2), float)

    c = zeta(2 + 1j, number_system="complex")
    _check_type("zeta(2+i, complex)", c, complex)

    _check_raises("zeta(1)", lambda: zeta(1), ValueError)
    _check_raises("zeta(1+1j) real mode", lambda: zeta(1 + 1j), TypeError)
    _check_raises("zeta('2')", lambda: zeta("2"), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Zeta tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_zeta_tests()


def test_main():
    run_zeta_tests()
