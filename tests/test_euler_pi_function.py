from math_library.euler_pi_function import euler_phi, euler_pi


PASS_COUNT = 0
FAIL_COUNT = 0


def _pass(message):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {message}")


def _fail(message):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {message}")


def _check_value(name, actual, expected):
    if actual == expected:
        _pass(f"{name} -> {actual}")
    else:
        _fail(f"{name} -> actual={actual}, expected={expected}")


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


def run_euler_pi_tests():
    print("=" * 80)
    print("EULER PI CATEGORY TESTS")
    print("=" * 80)

    _check_value("euler_phi(1)", euler_phi(1), 1)
    _check_value("euler_phi(9)", euler_phi(9), 6)
    _check_value("euler_phi(36)", euler_phi(36), 12)
    _check_value("euler_phi(0)", euler_phi(0), 0)

    _check_value("euler_pi(1)", euler_pi(1), 0)
    _check_value("euler_pi(10)", euler_pi(10), 4)
    _check_value("euler_pi(100)", euler_pi(100), 25)

    _check_type("euler_phi(10)", euler_phi(10), int)
    _check_type("euler_pi(10)", euler_pi(10), int)

    _check_raises("euler_phi(-1)", lambda: euler_phi(-1), ValueError)
    _check_raises("euler_pi(-1)", lambda: euler_pi(-1), ValueError)
    _check_raises("euler_phi(1.2)", lambda: euler_phi(1.2), TypeError)
    _check_raises("euler_pi('10')", lambda: euler_pi("10"), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Euler pi tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_euler_pi_tests()


def test_main():
    run_euler_pi_tests()
