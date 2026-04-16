from math_library.hyperbolic_function import hypercos, hypercosec, hypercotan, hypersec, hypersin, hypertan


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


def run_hyperbolic_tests():
    print("=" * 80)
    print("HYPERBOLIC CATEGORY TESTS")
    print("=" * 80)

    print("\n[Default real mode: real inputs -> real outputs]")
    hs = hypersin(1)
    hc = hypercos(1)
    ht = hypertan(1)
    _check_type("hypersin(1)", hs, float)
    _check_type("hypercos(1)", hc, float)
    _check_type("hypertan(1)", ht, float)

    _check_value("cosh^2(1) - sinh^2(1)", hc * hc - hs * hs, 1.0, tol=1e-6)
    _check_value("hypersec(1)", hypersec(1), 1.0 / hc, tol=1e-6)
    _check_value("hypercosec(1)", hypercosec(1), 1.0 / hs, tol=1e-6)
    _check_value("hypercotan(1)", hypercotan(1), hc / hs, tol=1e-6)

    print("\n[Default real mode: complex input rejected]")
    _check_raises("hypersin(1+1j)", lambda: hypersin(1 + 1j), TypeError)
    _check_raises("hypercos(1+1j)", lambda: hypercos(1 + 1j), TypeError)

    print("\n[Complex mode: explicit complex operations]")
    z = 1 + 1j
    hs_c = hypersin(z, number_system="complex")
    hc_c = hypercos(z, number_system="complex")
    ht_c = hypertan(z, number_system="complex")
    _check_type("hypersin(1+1j, complex)", hs_c, complex)
    _check_type("hypercos(1+1j, complex)", hc_c, complex)
    _check_type("hypertan(1+1j, complex)", ht_c, complex)

    _check_value("cosh^2(z) - sinh^2(z)", complex(hc_c) ** 2 - complex(hs_c) ** 2, 1.0 + 0.0j, tol=1e-5)
    _check_value(
        "hypersec(z) * hypercos(z)",
        complex(hypersec(z, number_system="complex")) * complex(hc_c),
        1.0 + 0.0j,
        tol=1e-5,
    )
    _check_value(
        "hypercosec(z) * hypersin(z)",
        complex(hypercosec(z, number_system="complex")) * complex(hs_c),
        1.0 + 0.0j,
        tol=1e-5,
    )

    print("\n[Undefined points -> exceptions]")
    _check_raises("hypercosec(0)", lambda: hypercosec(0), ZeroDivisionError)
    _check_raises("hypercotan(0)", lambda: hypercotan(0), ZeroDivisionError)

    print("\n[Type errors]")
    invalid_values = ["1", None, [1], {"x": 1}, True]
    for invalid in invalid_values:
        _check_raises(f"hypersin({invalid!r})", lambda v=invalid: hypersin(v), TypeError)
        _check_raises(f"hypercos({invalid!r})", lambda v=invalid: hypercos(v), TypeError)
        _check_raises(f"hypertan({invalid!r})", lambda v=invalid: hypertan(v), TypeError)
        _check_raises(f"hypersec({invalid!r})", lambda v=invalid: hypersec(v), TypeError)
        _check_raises(f"hypercosec({invalid!r})", lambda v=invalid: hypercosec(v), TypeError)
        _check_raises(f"hypercotan({invalid!r})", lambda v=invalid: hypercotan(v), TypeError)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Hyperbolic tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_hyperbolic_tests()
