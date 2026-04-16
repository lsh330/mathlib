from math_library.constant.e import e
from math_library.constant.epsilon import epsilon
from math_library.constant.pi import pi


PASS_COUNT = 0
FAIL_COUNT = 0


def _close(actual, expected, tol=1e-10):
    return abs(complex(actual) - complex(expected)) <= tol


def _pass(message):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {message}")


def _fail(message):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {message}")


def _check_value(name, actual, expected, tol=1e-10):
    if _close(actual, expected, tol):
        _pass(f"{name} -> {actual}")
    else:
        _fail(f"{name} -> actual={actual}, expected={expected}, tol={tol}")


def run_constant_tests():
    print("=" * 80)
    print("CONSTANT CATEGORY TESTS")
    print("=" * 80)

    print("\n[epsilon]")
    eps = epsilon()
    if isinstance(eps, float):
        _pass("epsilon() type is float")
    else:
        _fail(f"epsilon() type -> {type(eps)}")

    if eps > 0:
        _pass("epsilon() > 0")
    else:
        _fail(f"epsilon() > 0 check failed -> {eps}")

    if 1.0 + eps != 1.0:
        _pass("1.0 + epsilon() != 1.0")
    else:
        _fail("1.0 + epsilon() != 1.0 check failed")

    if 1.0 + eps / 2.0 == 1.0:
        _pass("1.0 + epsilon()/2 == 1.0")
    else:
        _fail("1.0 + epsilon()/2 == 1.0 check failed")

    print("\n[e]")
    _check_value("e()", e(), 2.718281828459045, tol=1e-9)

    print("\n[pi]")
    _check_value("pi()", pi(), 3.141592653589793, tol=1e-9)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Constant tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_constant_tests()
