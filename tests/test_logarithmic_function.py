from math_library.constant.e import e
from math_library.exponential_function import power
from math_library.logarithmic_function import log


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


def run_logarithmic_tests():
    print("=" * 80)
    print("LOGARITHMIC CATEGORY TESTS (log)")
    print("=" * 80)

    print("\n[Default real mode: real inputs -> real outputs]")
    r1 = log(2, 8)
    _check_value("log(2, 8)", r1, 3, tol=1e-6)
    _check_type("log(2, 8)", r1, float)

    r2 = log(10, 0.1)
    _check_value("log(10, 0.1)", r2, -1, tol=1e-6)
    _check_type("log(10, 0.1)", r2, float)

    r3 = log(e(), e())
    _check_value("log(e, e)", r3, 1, tol=1e-6)
    _check_type("log(e, e)", r3, float)

    print("\n[Inverse check with power in real mode]")
    value = power(3, 2.75)
    _check_value("log(3, power(3, 2.75))", log(3, value), 2.75, tol=1e-6)

    print("\n[Default real mode: complex input rejected]")
    _check_raises("log(1j, -1)", lambda: log(1j, -1), TypeError)
    _check_raises("log(2, 1+1j)", lambda: log(2, 1 + 1j), TypeError)

    print("\n[Complex mode: explicit complex operations]")
    c1 = log(1j, -1, number_system="complex")
    _check_value("log(1j, -1, complex)", c1, 2, tol=1e-6)
    _check_type("log(1j, -1, complex)", c1, float)

    c2 = log(2 + 1j, 3 - 2j, number_system="complex")
    _check_value("log(2+1j, 3-2j, complex)", c2, 0.8804277871462316 - 1.2379611965883508j, tol=1e-6)
    _check_type("log(2+1j, 3-2j, complex)", c2, complex)

    print("\n[Undefined cases]")
    _check_raises("log(2, 0)", lambda: log(2, 0), ValueError)
    _check_raises("log(0, 2)", lambda: log(0, 2), ValueError)
    _check_raises("log(1, 2)", lambda: log(1, 2), ValueError)

    print("\n[Type errors]")
    invalid_values = ["2", None, [2], {"x": 2}, True]
    for invalid in invalid_values:
        _check_raises(f"log({invalid!r}, 2)", lambda v=invalid: log(v, 2), TypeError)
        _check_raises(f"log(2, {invalid!r})", lambda v=invalid: log(2, v), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Logarithmic tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_logarithmic_tests()


def test_main():
    run_logarithmic_tests()
