from math_library.heaviside_step_function import heaviside


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


def _check_value(name, actual, expected, tol=1e-8):
    if abs(actual - expected) <= tol:
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


def run_heaviside_tests():
    print("=" * 80)
    print("HEAVISIDE CATEGORY TESTS")
    print("=" * 80)

    _check_value("heaviside(-3)", heaviside(-3), 0.0)
    _check_value("heaviside(0)", heaviside(0), 0.5)
    _check_value("heaviside(2)", heaviside(2), 1.0)
    _check_value("heaviside(0, at_zero=0)", heaviside(0, at_zero=0), 0.0)
    _check_type("heaviside(1)", heaviside(1), float)

    _check_value("heaviside(1+0j, complex)", heaviside(1 + 0j, number_system="complex"), 1.0)
    _check_raises("heaviside(1+1j, complex)", lambda: heaviside(1 + 1j, number_system="complex"), ValueError)

    _check_raises("heaviside('1')", lambda: heaviside("1"), TypeError)
    _check_raises("heaviside(1, at_zero='x')", lambda: heaviside(1, at_zero="x"), TypeError)
    _check_raises("heaviside(True)", lambda: heaviside(True), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Heaviside tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_heaviside_tests()


def test_main():
    run_heaviside_tests()
