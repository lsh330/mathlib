from math_library.lcm import lcm


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


def run_lcm_tests():
    print("=" * 80)
    print("LCM CATEGORY TESTS")
    print("=" * 80)

    _check_value("lcm(12, 18)", lcm(12, 18), 36)
    _check_value("lcm(-12, 18)", lcm(-12, 18), 36)
    _check_value("lcm(0, 18)", lcm(0, 18), 0)
    _check_value("lcm(0, 0)", lcm(0, 0), 0)
    _check_type("lcm(12, 18)", lcm(12, 18), int)

    _check_raises("lcm(1.2, 3)", lambda: lcm(1.2, 3), TypeError)
    _check_raises("lcm('4', 2)", lambda: lcm("4", 2), TypeError)
    _check_raises("lcm(True, 2)", lambda: lcm(True, 2), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"LCM tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_lcm_tests()


def test_main():
    run_lcm_tests()
