from math_library.constant.pi import pi
from math_library.differentiation import Differentiation
from math_library.trigonometric_function import cos, sin


PASS_COUNT = 0
FAIL_COUNT = 0


def _close(actual, expected, tol=1e-5):
    return abs(complex(actual) - complex(expected)) <= tol


def _pass(message):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {message}")


def _fail(message):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {message}")


def _check_value(name, actual, expected, tol=1e-5):
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


def run_differentiation_tests():
    print("=" * 80)
    print("DIFFERENTIATION CATEGORY TESTS")
    print("=" * 80)

    diff = Differentiation(tol=1e-9, initial_h=1e-3, max_iter=25, number_system="real")

    print("\n[One-variable derivative by definition]")
    d1 = diff.single_variable(lambda x: x * x * x, 2.0)
    _check_value("d/dx x^3 at x=2", d1, 12.0, tol=1e-4)
    _check_type("d/dx x^3 at x=2", d1, float)

    p = pi()
    d2 = diff.single_variable(lambda x: sin(x), p / 3.0)
    _check_value("d/dx sin(x) at x=pi/3", d2, cos(p / 3.0), tol=1e-4)

    print("\n[Higher-order derivative]")
    d3 = diff.nth_derivative(lambda x: x**4, 2.0, order=2)
    _check_value("d^2/dx^2 x^4 at x=2", d3, 48.0, tol=5e-3)

    print("\n[Multivariable: partial, gradient, Jacobian, Hessian]")
    f = lambda x, y: x * x * y + y * y * y
    px = diff.partial_derivative(f, [2.0, 1.0], 0)
    py = diff.partial_derivative(f, [2.0, 1.0], 1)
    _check_value("partial_x f(2,1)", px, 4.0, tol=1e-4)
    _check_value("partial_y f(2,1)", py, 7.0, tol=1e-4)

    grad = diff.gradient(f, [2.0, 1.0])
    _check_value("gradient[0]", grad[0], 4.0, tol=1e-4)
    _check_value("gradient[1]", grad[1], 7.0, tol=1e-4)

    jac = diff.jacobian([lambda x, y: x * y, lambda x, y: x * x + y * y], [2.0, 3.0])
    _check_value("Jacobian[0][0]", jac[0][0], 3.0, tol=1e-4)
    _check_value("Jacobian[0][1]", jac[0][1], 2.0, tol=1e-4)
    _check_value("Jacobian[1][0]", jac[1][0], 4.0, tol=1e-4)
    _check_value("Jacobian[1][1]", jac[1][1], 6.0, tol=1e-4)

    hessian = diff.hessian(lambda x, y: x * x + 3.0 * x * y + y * y, [1.0, 2.0])
    _check_value("Hessian[0][0]", hessian[0][0], 2.0, tol=1e-3)
    _check_value("Hessian[0][1]", hessian[0][1], 3.0, tol=1e-3)
    _check_value("Hessian[1][0]", hessian[1][0], 3.0, tol=1e-3)
    _check_value("Hessian[1][1]", hessian[1][1], 2.0, tol=1e-3)

    print("\n[Directional derivative and total differential]")
    directional = diff.directional_derivative(lambda x, y: x * x + y * y, [1.0, 2.0], [3.0, 4.0])
    _check_value("directional derivative", directional, 4.4, tol=1e-3)

    total_diff = diff.total_differential(f, [2.0, 1.0])
    _check_value("total differential coeff[0]", total_diff["coefficients"][0], 4.0, tol=1e-4)
    _check_value("total differential coeff[1]", total_diff["coefficients"][1], 7.0, tol=1e-4)

    print("\n[Implicit differentiation]")
    relation = lambda x, y: x * x + y * y - 25.0
    dy_dx = diff.implicit_derivative(relation, 3.0, 4.0)
    _check_value("implicit dy/dx for x^2+y^2=25 at (3,4)", dy_dx, -0.75, tol=1e-4)

    z0 = (1.0 - 0.3 * 0.3 - 0.4 * 0.4) ** 0.5
    relation3 = lambda x, y, z: x * x + y * y + z * z - 1.0
    dz_dx = diff.implicit_partial(relation3, [0.3, 0.4, z0], dependent_index=2, independent_index=0)
    _check_value("implicit partial dz/dx", dz_dx, -0.3 / z0, tol=1e-4)

    print("\n[Vector calculus: divergence, curl, laplacian]")
    divergence = diff.divergence([lambda x, y, z: x * x, lambda x, y, z: y * y, lambda x, y, z: z * z], [1.0, 2.0, 3.0])
    _check_value("divergence", divergence, 12.0, tol=1e-3)

    curl = diff.curl([lambda x, y, z: -y, lambda x, y, z: x, lambda x, y, z: 0.0], [1.0, 2.0, 3.0])
    _check_value("curl_x", curl[0], 0.0, tol=1e-3)
    _check_value("curl_y", curl[1], 0.0, tol=1e-3)
    _check_value("curl_z", curl[2], 2.0, tol=1e-3)

    laplacian = diff.laplacian(lambda x, y, z: x * x + y * y + z * z, [1.0, 2.0, 3.0])
    _check_value("laplacian", laplacian, 6.0, tol=1e-3)

    print("\n[Complex mode explicit operations]")
    diff_complex = Differentiation(tol=1e-9, initial_h=1e-3, max_iter=25, number_system="complex")
    dc = diff_complex.single_variable(lambda z: z * z, 1.0 + 2.0j)
    _check_type("complex derivative type", dc, complex)
    _check_value("d/dz z^2 at z=1+2j", dc, 2.0 + 4.0j, tol=1e-4)

    print("\n[Default real mode rejects complex input]")
    _check_raises(
        "real-mode derivative with complex x",
        lambda: diff.single_variable(lambda z: z * z, 1.0 + 2.0j),
        TypeError,
    )

    print("\n[One-sided and generalized derivatives]")
    left = diff.left_derivative(lambda x: x * x, 2.0)
    right = diff.right_derivative(lambda x: x * x, 2.0)
    _check_value("left derivative of x^2 at 2", left, 4.0, tol=1e-4)
    _check_value("right derivative of x^2 at 2", right, 4.0, tol=1e-4)

    gen_abs = diff.generalized_derivative(lambda x: abs(x), 0.0)
    _check_type("generalized derivative differentiable flag", gen_abs["differentiable"], bool)
    _check_value("generalized derivative left(abs,0)", gen_abs["left_derivative"], -1.0, tol=1e-3)
    _check_value("generalized derivative right(abs,0)", gen_abs["right_derivative"], 1.0, tol=1e-3)
    _check_value("generalized derivative interval lower(abs,0)", gen_abs["clarke_interval"][0], -1.0, tol=1e-3)
    _check_value("generalized derivative interval upper(abs,0)", gen_abs["clarke_interval"][1], 1.0, tol=1e-3)

    sub_abs = diff.subgradient(lambda x: abs(x), 0.0)
    _check_value("subgradient interval lower(abs,0)", sub_abs[0], -1.0, tol=1e-3)
    _check_value("subgradient interval upper(abs,0)", sub_abs[1], 1.0, tol=1e-3)

    print("\n[Mixed partial and higher-order directional derivatives]")
    mixed = diff.mixed_partial(lambda x, y: x * x * y * y * y, [2.0, 1.0], [1, 2])
    _check_value("mixed partial d^3/dxdy^2 of x^2y^3 at (2,1)", mixed, 24.0, tol=2e-2)

    high_dir = diff.higher_order_directional_derivative(
        lambda x, y: x * x * x + y * y * y,
        [2.0, 1.0],
        [1.0, 0.0],
        order=2,
    )
    _check_value("2nd directional derivative along x-axis", high_dir, 12.0, tol=5e-2)

    print("\n[Total derivative of composition]")
    total_comp = diff.total_derivative(
        lambda u, v: u * u + v,
        [lambda x, y: x + y, lambda x, y: x * y],
        [2.0, 3.0],
    )
    _check_value("total derivative component[0]", total_comp["derivative"][0], 13.0, tol=1e-3)
    _check_value("total derivative component[1]", total_comp["derivative"][1], 12.0, tol=1e-3)

    print("\n[Parametric derivatives]")
    param1 = diff.parametric_derivative(
        [lambda t: t * t, lambda t: t * t * t],
        2.0,
        dependent_index=1,
        independent_index=0,
        order=1,
    )
    _check_value("parametric dy/dx for x=t^2,y=t^3 at t=2", param1, 3.0, tol=1e-3)

    param2 = diff.parametric_derivative(
        [lambda t: t * t, lambda t: t * t * t],
        2.0,
        dependent_index=1,
        independent_index=0,
        order=2,
    )
    _check_value("parametric d2y/dx2 for x=t^2,y=t^3 at t=2", param2, 0.375, tol=1e-2)

    print("\n[Hessian-vector product]")
    hv = diff.hessian_vector_product(
        lambda x, y: x * x + 3.0 * x * y + y * y,
        [1.0, 2.0],
        [4.0, 5.0],
    )
    _check_value("Hessian-vector product component[0]", hv[0], 23.0, tol=5e-2)
    _check_value("Hessian-vector product component[1]", hv[1], 22.0, tol=5e-2)

    print("\n[Vector Laplacian]")
    vec_lap = diff.vector_laplacian(
        [lambda x, y, z: x * x + y * y, lambda x, y, z: x * y, lambda x, y, z: z * z],
        [1.0, 2.0, 3.0],
    )
    _check_value("vector laplacian component[0]", vec_lap[0], 4.0, tol=2e-2)
    _check_value("vector laplacian component[1]", vec_lap[1], 0.0, tol=2e-2)
    _check_value("vector laplacian component[2]", vec_lap[2], 2.0, tol=2e-2)

    print("\n[Wirtinger derivatives]")
    wirtinger = diff_complex.wirtinger_derivatives(lambda z: z * z, 1.0 + 2.0j)
    _check_value("df/dz for z^2 at z=1+2j", wirtinger["df_dz"], 2.0 + 4.0j, tol=2e-3)
    _check_value("df/dz_conjugate for z^2 at z=1+2j", wirtinger["df_dz_conjugate"], 0.0, tol=2e-3)

    print("\n[Gateaux and Frechet derivatives]")
    gateaux_scalar = diff.gateaux_derivative(lambda x, y: x * x + y * y, [1.0, 2.0], [3.0, 4.0])
    _check_value("Gateaux scalar", gateaux_scalar, 22.0, tol=1e-2)

    gateaux_vector = diff.gateaux_derivative(lambda x, y: [x + y, x * y], [1.0, 2.0], [3.0, 4.0])
    _check_value("Gateaux vector component[0]", gateaux_vector[0], 7.0, tol=1e-2)
    _check_value("Gateaux vector component[1]", gateaux_vector[1], 10.0, tol=1e-2)

    frechet_scalar = diff.frechet_derivative(lambda x, y: x * x + y * y, [1.0, 2.0])
    _check_value("Frechet scalar matrix[0][0]", frechet_scalar["matrix"][0][0], 2.0, tol=1e-3)
    _check_value("Frechet scalar matrix[0][1]", frechet_scalar["matrix"][0][1], 4.0, tol=1e-3)

    frechet_vector = diff.frechet_derivative(lambda x, y: [x + y, x * y], [1.0, 2.0])
    _check_value("Frechet vector matrix[0][0]", frechet_vector["matrix"][0][0], 1.0, tol=1e-3)
    _check_value("Frechet vector matrix[0][1]", frechet_vector["matrix"][0][1], 1.0, tol=1e-3)
    _check_value("Frechet vector matrix[1][0]", frechet_vector["matrix"][1][0], 2.0, tol=1e-3)
    _check_value("Frechet vector matrix[1][1]", frechet_vector["matrix"][1][1], 1.0, tol=1e-3)

    print("\n" + "-" * 80)
    total = PASS_COUNT + FAIL_COUNT
    print(f"SUMMARY: total={total}, pass={PASS_COUNT}, fail={FAIL_COUNT}")
    print("-" * 80)

    if FAIL_COUNT > 0:
        raise AssertionError(f"Differentiation tests failed: {FAIL_COUNT} case(s)")


if __name__ == "__main__":
    run_differentiation_tests()
