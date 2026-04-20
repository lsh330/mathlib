from test_bessel_function import run_bessel_tests
from test_beta_function import run_beta_tests
from test_constant import run_constant_tests
from test_differentiation import run_differentiation_tests
from test_euler_pi_function import run_euler_pi_tests
from test_exponential_function import run_exponential_tests
from test_gamma_function import run_gamma_tests
from test_gcd import run_gcd_tests
from test_heaviside_step_function import run_heaviside_tests
from test_hyperbolic_function import run_hyperbolic_tests
from test_inverse_trigonometric_function import run_inverse_trigonometric_tests
from test_lambert_w_function import run_lambert_w_tests
from test_lcm import run_lcm_tests
from test_legendre_function import run_legendre_tests
from test_logarithmic_function import run_logarithmic_tests
from test_trigonomectric_function import run_trigonometric_tests
from test_zeta_function import run_zeta_tests


def run_all_category_tests():
    print("=" * 80)
    print("ALL CATEGORY TESTS")
    print("=" * 80)

    run_constant_tests()
    run_gcd_tests()
    run_lcm_tests()
    run_euler_pi_tests()
    run_heaviside_tests()

    run_exponential_tests()
    run_logarithmic_tests()
    run_trigonometric_tests()
    run_inverse_trigonometric_tests()
    run_hyperbolic_tests()

    run_gamma_tests()
    run_beta_tests()
    run_lambert_w_tests()
    run_legendre_tests()
    run_bessel_tests()
    run_zeta_tests()

    run_differentiation_tests()

    print("\nALL CATEGORY TESTS PASSED")


if __name__ == "__main__":
    run_all_category_tests()


def test_main():
    run_all_category_tests()
