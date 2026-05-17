import math
import unittest

import numpy as np

from nesterov_smoothing import (
    BoxDomain,
    SimplexDomain,
    _simplex_l1_squared_step,
    solve_max_affine,
    solve_paper_matrix_game,
)


class NesterovSmoothingTests(unittest.TestCase):
    def test_abs_value_on_box(self) -> None:
        A = np.array([[1.0], [-1.0]], dtype=np.float64)
        b = np.array([0.0, 0.0], dtype=np.float64)
        domain = BoxDomain(lower=np.array([-1.0]), upper=np.array([1.0]))

        result = solve_max_affine(
            A,
            b,
            domain,
            max_iters=300,
            monotone=True,
        )

        self.assertLess(abs(float(result.best_x[0])), 1.0e-4)
        self.assertLess(result.best_nonsmooth_value, 1.0e-4)
        self.assertIsNotNone(result.primal_dual_gap)
        self.assertIsNotNone(result.theoretical_gap_bound)
        self.assertLessEqual(result.primal_dual_gap, result.theoretical_gap_bound + 1.0e-10)

    def test_simplex_symmetry_problem(self) -> None:
        A = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
        b = np.array([0.0, 0.0], dtype=np.float64)
        domain = SimplexDomain(2)

        result = solve_max_affine(
            A,
            b,
            domain,
            max_iters=300,
            monotone=True,
        )

        np.testing.assert_allclose(result.best_x.sum(), 1.0, atol=1.0e-10)
        self.assertTrue(np.all(result.best_x >= -1.0e-12))
        np.testing.assert_allclose(result.best_x, np.array([0.5, 0.5]), atol=1.0e-4)
        self.assertLess(result.best_nonsmooth_value, 1.0e-4)

    def test_theorem3_accuracy_rule_is_reasonable(self) -> None:
        A = np.array([[2.0, -1.0], [-0.5, 1.5], [0.2, -0.1]], dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        domain = BoxDomain(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
        target = 2.5e-1

        result = solve_max_affine(A, b, domain, desired_accuracy=target, monotone=True)

        self.assertIsNotNone(result.theoretical_gap_bound)
        self.assertLessEqual(result.theoretical_gap_bound, target + 1.0e-12)
        self.assertTrue(math.isfinite(result.mu))
        self.assertGreater(result.iterations, 0)

    def test_section_5_1_simplex_step_matches_extreme_case(self) -> None:
        x_bar = np.array([0.5, 0.5], dtype=np.float64)
        gradient = np.array([0.0, 10.0], dtype=np.float64)

        x_next = _simplex_l1_squared_step(x_bar, gradient, L=1.0)

        np.testing.assert_allclose(x_next, np.array([1.0, 0.0]), atol=1.0e-10)

    def test_paper_matrix_game_solver_reaches_target_gap(self) -> None:
        A = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)

        result = solve_paper_matrix_game(A, epsilon=2.0e-1, check_frequency=5)

        self.assertTrue(result.converged)
        self.assertLessEqual(result.gap, 2.0e-1)
        np.testing.assert_allclose(result.x.sum(), 1.0, atol=1.0e-10)
        np.testing.assert_allclose(result.u.sum(), 1.0, atol=1.0e-10)
        self.assertGreater(result.predicted_iterations, 0)


if __name__ == "__main__":
    unittest.main()
