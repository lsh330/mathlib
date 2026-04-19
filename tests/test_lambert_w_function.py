from math_library.constant import e
from math_library.lambert_w_function import lambert_w


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


def run_lambert_w_tests():
    print("=" * 80)
    print("LAMBERT W CATEGORY TESTS")
    print("=" * 80)

    _check_value("W0(0)", lambert_w(0), 0.0)
    _check_value("W0(e)", lambert_w(e()), 1.0, tol=1e-5)
    _check_value("W0(-1/e)", lambert_w(-1.0 / e()), -1.0, tol=1e-5)
    _check_type("W0(e)", lambert_w(e()), float)

    v_branch = lambert_w(-0.1, branch=-1)
    _check_type("W-1(-0.1)", v_branch, float)

    c = lambert_w(1 + 1j, number_system="complex")
    _check_type("W0(1+i, complex)", c, complex)

    _check_raises("W0(-1)", lambda: lambert_w(-1.0), ValueError)
    _check_raises("W3(1) in real mode", lambda: lambert_w(1.0, branch=3), ValueError)
    _check_raises("W-1(0)", lambda: lambert_w(0.0, branch=-1), ValueError)
    _check_raises("W('1')", lambda: lambert_w("1"), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Lambert W tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_lambert_w_tests()
