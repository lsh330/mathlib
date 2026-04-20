from math_library.constant.pi import pi
from math_library.trigonometric_function import cos, cosec, cotan, sec, sin, tan


PI = pi()
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


def run_trigonometric_tests():
    print("=" * 80)
    print("TRIGONOMETRIC CATEGORY TESTS (sin, cos, tan, cosec, sec, cotan)")
    print("=" * 80)

    print("\n[Default real mode: real inputs -> real outputs]")
    s = sin(PI / 2)
    _check_value("sin(PI/2)", s, 1, tol=1e-6)
    _check_type("sin(PI/2)", s, float)

    c = cos(PI)
    _check_value("cos(PI)", c, -1, tol=1e-6)
    _check_type("cos(PI)", c, float)

    t = tan(PI / 4)
    _check_value("tan(PI/4)", t, 1, tol=1e-6)
    _check_type("tan(PI/4)", t, float)

    _check_value("sin(30 deg)", sin(30, unit="deg"), 0.5, tol=1e-6)
    _check_value("cos(60 deg)", cos(60, unit="deg"), 0.5, tol=1e-6)
    _check_value("tan(45 deg)", tan(45, unit="deg"), 1.0, tol=1e-6)

    _check_value("cosec(PI/2)", cosec(PI / 2), 1.0, tol=1e-6)
    _check_value("sec(0)", sec(0), 1.0, tol=1e-6)
    _check_value("cotan(PI/4)", cotan(PI / 4), 1.0, tol=1e-6)

    print("\n[Default real mode: complex input rejected]")
    _check_raises("sin(1+2j)", lambda: sin(1 + 2j), TypeError)
    _check_raises("cos(1+2j)", lambda: cos(1 + 2j), TypeError)
    _check_raises("tan(1+2j)", lambda: tan(1 + 2j), TypeError)

    print("\n[Complex mode: explicit complex operations]")
    z = 1 + 2j
    sz = sin(z, number_system="complex")
    cz = cos(z, number_system="complex")
    _check_type("sin(1+2j, complex)", sz, complex)
    _check_type("cos(1+2j, complex)", cz, complex)
    _check_value("sin^2 + cos^2 (complex)", complex(sz) ** 2 + complex(cz) ** 2, 1.0 + 0.0j, tol=1e-5)

    tz = tan(z, number_system="complex")
    cotz = cotan(z, number_system="complex")
    _check_value("tan * cotan (complex)", complex(tz) * complex(cotz), 1.0 + 0.0j, tol=1e-5)

    _check_value(
        "sec * cos (complex)",
        complex(sec(z, number_system="complex")) * complex(cos(z, number_system="complex")),
        1.0 + 0.0j,
        tol=1e-5,
    )
    _check_value(
        "cosec * sin (complex)",
        complex(cosec(z, number_system="complex")) * complex(sin(z, number_system="complex")),
        1.0 + 0.0j,
        tol=1e-5,
    )

    print("\n[Undefined points -> exceptions]")
    _check_raises("cosec(0)", lambda: cosec(0), ZeroDivisionError)
    _check_raises("cotan(0)", lambda: cotan(0), ZeroDivisionError)

    print("\n[Type and unit errors]")
    invalid_values = ["1", None, [1], {"x": 1}, True]
    for invalid in invalid_values:
        _check_raises(f"sin({invalid!r})", lambda v=invalid: sin(v), TypeError)
        _check_raises(f"cos({invalid!r})", lambda v=invalid: cos(v), TypeError)
        _check_raises(f"tan({invalid!r})", lambda v=invalid: tan(v), TypeError)

    _check_raises("sin(1, unit='grad')", lambda: sin(1, unit="grad"), ValueError)
    _check_raises("cos(1, unit=90)", lambda: cos(1, unit=90), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Trigonometric tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_trigonometric_tests()


def test_main():
    run_trigonometric_tests()
