import unittest

import numpy as np

from nesterov_core import OracleState, run_accelerated_method
from nesterov_smoothing import (
    ContinuationConfig,
    _project_to_l2_ball,
    _project_to_simplex,
    _simplex_entropy_argmin,
    _simplex_l1_squared_step,
    run_continuous_location_grid,
    run_paper_matrix_game_grid,
    run_piecewise_linear_grid,
    solve_continuous_location,
    solve_paper_matrix_game,
    solve_piecewise_linear_max_abs,
)


class NesterovSmoothingTests(unittest.TestCase):
    def test_project_to_simplex(self) -> None:
        x = np.array([0.2, -0.5, 2.0], dtype=np.float64)
        projected = _project_to_simplex(x)

        self.assertAlmostEqual(float(np.sum(projected)), 1.0, places=12)
        self.assertTrue(np.all(projected >= -1.0e-12))

    def test_project_to_l2_ball(self) -> None:
        x = np.array([3.0, 4.0], dtype=np.float64)
        projected = _project_to_l2_ball(x, radius=2.0)

        self.assertAlmostEqual(float(np.linalg.norm(projected)), 2.0, places=12)
        np.testing.assert_allclose(projected / np.linalg.norm(projected), x / np.linalg.norm(x), atol=1.0e-12)

    def test_entropy_argmin_returns_simplex_point(self) -> None:
        point = _simplex_entropy_argmin(np.array([1.0, -2.0, 0.5], dtype=np.float64), scale=3.0)

        self.assertAlmostEqual(float(np.sum(point)), 1.0, places=12)
        self.assertTrue(np.all(point > 0.0))

    def test_section_5_1_simplex_step_matches_extreme_case(self) -> None:
        x_bar = np.array([0.5, 0.5], dtype=np.float64)
        gradient = np.array([0.0, 10.0], dtype=np.float64)

        x_next = _simplex_l1_squared_step(x_bar, gradient, L=1.0)

        np.testing.assert_allclose(x_next, np.array([1.0, 0.0]), atol=1.0e-10)

    def test_monotone_y_keeps_best_candidate(self) -> None:
        def oracle(x: np.ndarray) -> OracleState:
            value = float(np.sum(x * x))
            return OracleState(smoothed_value=value, gradient=np.zeros_like(x), objective_value=value)

        initial_x = np.array([0.0], dtype=np.float64)
        bad_step = lambda x, gradient, L: np.array([10.0], dtype=np.float64)
        aggregate_zero = lambda gradient_sum, L: np.zeros_like(gradient_sum)

        non_monotone = run_accelerated_method(
            initial_x=initial_x,
            lipschitz=1.0,
            max_iterations=1,
            check_frequency=1,
            oracle=oracle,
            local_step=bad_step,
            aggregate_step=aggregate_zero,
            should_stop=lambda current: True,
            monotone_y=False,
        )
        monotone = run_accelerated_method(
            initial_x=initial_x,
            lipschitz=1.0,
            max_iterations=1,
            check_frequency=1,
            oracle=oracle,
            local_step=bad_step,
            aggregate_step=aggregate_zero,
            should_stop=lambda current: True,
            monotone_y=True,
        )

        self.assertGreater(non_monotone.state.objective_value, monotone.state.objective_value)
        np.testing.assert_allclose(monotone.y, initial_x, atol=1.0e-12)

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

    def test_paper_matrix_game_continuation_smoke(self) -> None:
        A = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)

        result = solve_paper_matrix_game(
            A,
            epsilon=2.0e-1,
            check_frequency=5,
            continuation=ContinuationConfig(start_factor=2.0, decay=0.5, max_stages=3),
        )

        self.assertTrue(result.converged)
        self.assertGreaterEqual(len(result.continuation_stages), 1)
        self.assertLessEqual(result.gap, 2.0e-1)

    def test_continuous_location_solver_reaches_target_bound(self) -> None:
        cities = np.array([[-1.0], [1.0]], dtype=np.float64)

        result = solve_continuous_location(cities, epsilon=2.0e-1, radius=2.0, check_frequency=5)

        self.assertTrue(result.converged)
        self.assertLessEqual(result.gap, 2.0e-1)
        self.assertGreater(result.theoretical_gap_bound, 0.0)
        self.assertLessEqual(float(np.linalg.norm(result.x)), 2.0 + 1.0e-12)
        self.assertGreater(result.predicted_iterations, 0)

    def test_continuous_location_grid_smoke(self) -> None:
        cells = run_continuous_location_grid(
            base_seed=0,
            epsilons=(2.0e-1,),
            num_cities_values=(4,),
            dimension_values=(2,),
            radius=1.0,
            check_frequency=5,
        )

        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertTrue(cell.converged)
        self.assertGreater(cell.iterations, 0)
        self.assertGreater(cell.predicted_iterations, 0)

    def test_continuous_location_continuation_smoke(self) -> None:
        cities = np.array([[-1.0], [1.0]], dtype=np.float64)

        result = solve_continuous_location(
            cities,
            epsilon=2.0e-1,
            radius=2.0,
            check_frequency=5,
            continuation=ContinuationConfig(start_factor=2.0, decay=0.5, max_stages=3),
        )

        self.assertTrue(result.converged)
        self.assertGreaterEqual(len(result.continuation_stages), 1)
        self.assertLessEqual(result.gap, 2.0e-1)
        self.assertGreater(result.theoretical_gap_bound, 0.0)

    def test_piecewise_linear_solver_reaches_target_gap(self) -> None:
        A = np.array([[1.0], [-1.0]], dtype=np.float64)
        b = np.array([0.5, 0.5], dtype=np.float64)

        result = solve_piecewise_linear_max_abs(A, b, epsilon=2.0e-1, radius=2.0, check_frequency=5)

        self.assertTrue(result.converged)
        self.assertLessEqual(result.gap, 2.0e-1)
        self.assertLessEqual(float(np.linalg.norm(result.x)), 2.0 + 1.0e-12)
        np.testing.assert_allclose(result.u.sum(), 1.0, atol=1.0e-10)
        self.assertGreater(result.predicted_iterations, 0)

    def test_piecewise_linear_grid_smoke(self) -> None:
        cells = run_piecewise_linear_grid(
            base_seed=0,
            epsilons=(2.0e-1,),
            m_values=(4,),
            n_values=(2,),
            radius=1.0,
            check_frequency=5,
        )

        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertTrue(cell.converged)
        self.assertGreater(cell.iterations, 0)
        self.assertGreater(cell.predicted_iterations, 0)

    def test_piecewise_linear_continuation_smoke(self) -> None:
        A = np.array([[1.0], [-1.0]], dtype=np.float64)
        b = np.array([0.5, 0.5], dtype=np.float64)

        result = solve_piecewise_linear_max_abs(
            A,
            b,
            epsilon=2.0e-1,
            radius=2.0,
            check_frequency=5,
            continuation=ContinuationConfig(start_factor=2.0, decay=0.5, max_stages=3),
        )

        self.assertTrue(result.converged)
        self.assertGreaterEqual(len(result.continuation_stages), 1)
        self.assertLessEqual(result.gap, 2.0e-1)


if __name__ == "__main__":
    unittest.main()
