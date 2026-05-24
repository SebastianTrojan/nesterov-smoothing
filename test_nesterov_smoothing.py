import math
import unittest

import numpy as np

from nesterov_smoothing import (
    EntropySimplexDomain,
    _simplex_l1_squared_step,
    run_paper_matrix_game_grid,
    solve_max_affine,
    solve_paper_matrix_game,
)


class NesterovSmoothingTests(unittest.TestCase):
    def test_theorem3_accuracy_rule_is_reasonable(self) -> None:
        A = np.array([[2.0, -1.0], [-0.5, 1.5], [0.2, -0.1]], dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        domain = EntropySimplexDomain(2)
        target = 2.5e-1

        result = solve_max_affine(
            A,
            b,
            domain,
            desired_accuracy=target,
            monotone=True,
            operator_norm=float(np.max(np.abs(A))),
            dual_diameter=math.log(A.shape[0]),
            dual_sigma=1.0,
        )

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

    def test_paper_matrix_game_grid_smoke(self) -> None:
        cells = run_paper_matrix_game_grid(
            base_seed=0,
            epsilons=(2.0e-1,),
            m_values=(2,),
            n_values=(3,),
            check_frequency=5,
        )

        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertTrue(cell.converged)
        self.assertGreater(cell.iterations, 0)
        self.assertGreater(cell.predicted_iterations, 0)

    def test_solve_max_affine_matches_paper_matrix_game_path(self) -> None:
        A = np.array([[0.5, -1.0, 0.2], [-0.25, 0.75, 1.1]], dtype=np.float64)
        epsilon = 2.0e-1

        paper_result = solve_paper_matrix_game(A, epsilon=epsilon, check_frequency=5)
        generic_result = solve_max_affine(
            A=A,
            b=np.zeros(A.shape[0], dtype=np.float64),
            domain=EntropySimplexDomain(A.shape[1]),
            mu=epsilon / (2.0 * math.log(A.shape[0])),
            max_iters=paper_result.iterations,
            desired_accuracy=epsilon,
            check_frequency=5,
            operator_norm=float(np.max(np.abs(A))),
            dual_diameter=math.log(A.shape[0]),
            dual_sigma=1.0,
        )

        np.testing.assert_allclose(generic_result.x, paper_result.x, atol=1.0e-10)
        np.testing.assert_allclose(generic_result.dual_variable, paper_result.u, atol=1.0e-10)
        self.assertAlmostEqual(generic_result.nonsmooth_value, paper_result.primal_value, places=10)
        self.assertAlmostEqual(generic_result.dual_value, paper_result.dual_value, places=10)
        self.assertAlmostEqual(generic_result.primal_dual_gap, paper_result.gap, places=10)


if __name__ == "__main__":
    unittest.main()
