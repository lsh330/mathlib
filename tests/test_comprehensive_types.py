from test_constant import run_constant_tests
from test_differentiation import run_differentiation_tests
from test_exponential_function import run_exponential_tests
from test_hyperbolic_function import run_hyperbolic_tests
from test_inverse_trigonometric_function import run_inverse_trigonometric_tests
from test_logarithmic_function import run_logarithmic_tests
from test_trigonomectric_function import run_trigonometric_tests


def run_all_category_tests():
    print("=" * 80)
    print("ALL CATEGORY TESTS")
    print("=" * 80)

    run_constant_tests()
    run_exponential_tests()
    run_logarithmic_tests()
    run_trigonometric_tests()
    run_inverse_trigonometric_tests()
    run_hyperbolic_tests()
    run_differentiation_tests()

    print("\nALL CATEGORY TESTS PASSED")


if __name__ == "__main__":
    run_all_category_tests()
