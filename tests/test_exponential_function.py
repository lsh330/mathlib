from math_library.exponential_function import power


PASS_COUNT = 0
FAIL_COUNT = 0


def _close(actual, expected, tol=1e-8):
    return abs(complex(actual) - complex(expected)) <= tol


def _pass(message):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {message}")


def _fail(message):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {message}")


def _check_value(name, actual, expected, tol=1e-8):
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


def run_exponential_tests():
    print("=" * 80)
    print("EXPONENTIAL CATEGORY TESTS (power)")
    print("=" * 80)

    print("\n[Default real mode: real inputs -> real outputs]")
    v1 = power(2, 3)
    _check_value("power(2, 3)", v1, 8)
    _check_type("power(2, 3)", v1, float)

    v2 = power(2, -2)
    _check_value("power(2, -2)", v2, 0.25)
    _check_type("power(2, -2)", v2, float)

    v3 = power(4, 0.5)
    _check_value("power(4, 0.5)", v3, 2, tol=1e-6)
    _check_type("power(4, 0.5)", v3, float)

    print("\n[Default real mode: real-domain undefined]")
    _check_raises("power(-4, 0.5)", lambda: power(-4, 0.5), ValueError)
    _check_raises("power(0, 0)", lambda: power(0, 0), ValueError)
    _check_raises("power(0, -1)", lambda: power(0, -1), ValueError)

    print("\n[Default real mode: complex input rejected]")
    _check_raises("power(1+1j, 2)", lambda: power(1 + 1j, 2), TypeError)
    _check_raises("power(2, 1+1j)", lambda: power(2, 1 + 1j), TypeError)

    print("\n[Complex mode: explicit complex operations]")
    c1 = power(-4, 0.5, number_system="complex")
    _check_value("power(-4, 0.5, complex)", c1, 2j, tol=1e-6)
    _check_type("power(-4, 0.5, complex)", c1, complex)

    c2 = power(1 + 1j, 2 - 1j, number_system="complex")
    _check_value("power(1+1j, 2-1j, complex)", c2, 1.490014124359447 + 4.125744470161809j, tol=1e-6)
    _check_type("power(1+1j, 2-1j, complex)", c2, complex)

    c3 = power(1 + 1j, 0.5, number_system="complex")
    _check_value("power(1+1j, 0.5, complex)^2", c3 * c3, 1 + 1j, tol=1e-5)

    print("\n[Type errors]")
    invalid_values = ["2", None, [2], {"x": 2}, True]
    for invalid in invalid_values:
        _check_raises(f"power({invalid!r}, 2)", lambda v=invalid: power(v, 2), TypeError)
        _check_raises(f"power(2, {invalid!r})", lambda v=invalid: power(2, v), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Exponential tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_exponential_tests()
