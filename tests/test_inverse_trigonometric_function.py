from math_library.constant.pi import pi
from math_library.inverse_trigonometric_function import arccos, arccosec, arccotan, arcsec, arcsin, arctan
from math_library.trigonometric_function import cos, cosec, sec, sin, tan


PI = pi()
PASS_COUNT = 0
FAIL_COUNT = 0


def _close(actual, expected, tol=1e-7):
    return abs(complex(actual) - complex(expected)) <= tol


def _pass(message):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {message}")


def _fail(message):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {message}")


def _check_value(name, actual, expected, tol=1e-7):
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


def run_inverse_trigonometric_tests():
    print("=" * 80)
    print("INVERSE TRIGONOMETRIC CATEGORY TESTS")
    print("=" * 80)

    print("\n[Default real mode: real inputs -> real outputs]")
    for x in [-1, -0.5, 0, 0.5, 1]:
        a = arcsin(x)
        b = arccos(x)
        _check_type(f"arcsin({x})", a, float)
        _check_type(f"arccos({x})", b, float)
        _check_value(f"sin(arcsin({x}))", sin(a), x, tol=1e-5)
        _check_value(f"cos(arccos({x}))", cos(b), x, tol=1e-5)

    for x in [-2, -0.5, 0, 0.5, 2]:
        at = arctan(x)
        _check_type(f"arctan({x})", at, float)
        _check_value(f"tan(arctan({x}))", tan(at), x, tol=1e-5)

    for x in [-3, -1, 1, 3]:
        _check_value(f"arccotan({x}) + arctan({x})", arccotan(x) + arctan(x), PI / 2, tol=1e-5)

    _check_value("sec(arcsec(2))", sec(arcsec(2)), 2, tol=1e-5)
    _check_value("cosec(arccosec(2))", cosec(arccosec(2)), 2, tol=1e-5)

    print("\n[Default real mode: domain restrictions]")
    _check_raises("arcsin(2)", lambda: arcsin(2), ValueError)
    _check_raises("arccos(2)", lambda: arccos(2), ValueError)
    _check_raises("arcsec(0.5)", lambda: arcsec(0.5), ValueError)
    _check_raises("arccosec(0.5)", lambda: arccosec(0.5), ValueError)

    print("\n[Default real mode: complex input rejected]")
    _check_raises("arcsin(1+1j)", lambda: arcsin(1 + 1j), TypeError)
    _check_raises("arctan(1+1j)", lambda: arctan(1 + 1j), TypeError)

    print("\n[Complex mode: explicit complex operations]")
    z = 1 + 1j
    ac = arcsin(z, number_system="complex")
    _check_type("arcsin(1+1j, complex)", ac, complex)
    _check_value("sin(arcsin(1+1j, complex))", sin(ac, number_system="complex"), z, tol=1e-5)

    atc = arctan(z, number_system="complex")
    _check_type("arctan(1+1j, complex)", atc, complex)
    _check_value("tan(arctan(1+1j, complex))", tan(atc, number_system="complex"), z, tol=1e-5)

    _check_value(
        "sec(arcsec(1+1j, complex))",
        sec(arcsec(z, number_system="complex"), number_system="complex"),
        z,
        tol=1e-5,
    )
    _check_value(
        "cosec(arccosec(1+1j, complex))",
        cosec(arccosec(z, number_system="complex"), number_system="complex"),
        z,
        tol=1e-5,
    )

    print("\n[Undefined points -> exceptions]")
    _check_raises("arcsec(0)", lambda: arcsec(0), ZeroDivisionError)
    _check_raises("arccosec(0)", lambda: arccosec(0), ZeroDivisionError)

    print("\n[Type errors]")
    invalid_values = ["1", None, [1], {"x": 1}, True]
    for invalid in invalid_values:
        _check_raises(f"arcsin({invalid!r})", lambda v=invalid: arcsin(v), TypeError)
        _check_raises(f"arccos({invalid!r})", lambda v=invalid: arccos(v), TypeError)
        _check_raises(f"arctan({invalid!r})", lambda v=invalid: arctan(v), TypeError)
        _check_raises(f"arcsec({invalid!r})", lambda v=invalid: arcsec(v), TypeError)
        _check_raises(f"arccosec({invalid!r})", lambda v=invalid: arccosec(v), TypeError)
        _check_raises(f"arccotan({invalid!r})", lambda v=invalid: arccotan(v), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Inverse trigonometric tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_inverse_trigonometric_tests()


def test_main():
    run_inverse_trigonometric_tests()
