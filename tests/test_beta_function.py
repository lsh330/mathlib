from math_library.beta_function import beta
from math_library.constant import pi


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


def run_beta_tests():
    print("=" * 80)
    print("BETA CATEGORY TESTS")
    print("=" * 80)

    _check_value("beta(1,1)", beta(1, 1), 1.0)
    _check_value("beta(2,3)", beta(2, 3), 1.0 / 12.0, tol=1e-5)
    _check_value("beta(0.5,0.5)", beta(0.5, 0.5), pi(), tol=1e-5)
    _check_type("beta(2,3)", beta(2, 3), float)

    c = beta(1 + 1j, 2 - 0.5j, number_system="complex")
    _check_type("beta(complex, complex)", c, complex)

    _check_raises("beta(1+1j, 2) in real mode", lambda: beta(1 + 1j, 2), TypeError)
    _check_raises("beta('1', 2)", lambda: beta("1", 2), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Beta tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_beta_tests()
