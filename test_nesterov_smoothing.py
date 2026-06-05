import csv
import unittest
from pathlib import Path

import numpy as np

from experiments.run_epsilon_scaling import _fit_loglog_regression
from experiments.run_mu_comparison import _select_history_points
from nesterov_core import GapTracePoint, OracleState, run_accelerated_method
from nesterov_smoothing import (
    ContinuationConfig,
    _project_to_l2_ball,
    _project_to_simplex,
    _simplex_entropy_argmin,
    _simplex_l1_squared_step,
    run_continuous_location_grid,
    run_paper_matrix_game_grid,
    run_piecewise_linear_grid,
    run_sum_absolute_grid,
    save_mu_gap_history_csv,
    solve_continuous_location,
    solve_paper_matrix_game,
    solve_piecewise_linear_max_abs,
    solve_sum_absolute_values,
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

    def test_fixed_mu_gap_history_tracks_final_matrix_game_gap(self) -> None:
        A = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)

        result = solve_paper_matrix_game(A, epsilon=2.0e-1, check_frequency=5)

        self.assertEqual(len(result.mu_gap_history), 1)
        point = result.mu_gap_history[0]
        self.assertAlmostEqual(point.mu, result.mu)
        self.assertAlmostEqual(point.gap, result.gap)
        self.assertAlmostEqual(point.target_value, 2.0e-1)
        self.assertEqual(point.iterations, result.iterations)
        self.assertTrue(point.final_stage)

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
        self.assertEqual(len(result.mu_gap_history), len(result.continuation_stages))
        self.assertAlmostEqual(result.mu_gap_history[-1].gap, result.gap)
        self.assertLessEqual(result.gap, 2.0e-1)

    def test_mu_gap_history_can_be_saved_to_csv(self) -> None:
        A = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
        result = solve_paper_matrix_game(A, epsilon=2.0e-1, check_frequency=5)

        output_path = Path("results") / "test_mu_gap_history.csv"
        save_mu_gap_history_csv(output_path, result.mu_gap_history)

        with output_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stage_index"], "1")
        self.assertAlmostEqual(float(rows[0]["mu"]), result.mu)
        self.assertAlmostEqual(float(rows[0]["gap"]), result.gap)

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

    def test_sum_absolute_solver_reaches_target_gap(self) -> None:
        A = np.array([[1.0], [-1.0]], dtype=np.float64)
        b = np.array([0.5, 0.5], dtype=np.float64)

        result = solve_sum_absolute_values(A, b, epsilon=2.0e-1, radius=2.0, check_frequency=5)

        self.assertTrue(result.converged)
        self.assertLessEqual(result.gap, 2.0e-1)
        self.assertLessEqual(float(np.linalg.norm(result.x)), 2.0 + 1.0e-12)
        np.testing.assert_allclose(np.max(np.abs(result.u)), 1.0, atol=1.0e-10)
        self.assertGreater(result.predicted_iterations, 0)

    def test_sum_absolute_grid_smoke(self) -> None:
        cells = run_sum_absolute_grid(
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

    def test_sum_absolute_continuation_smoke(self) -> None:
        A = np.array([[1.0], [-1.0]], dtype=np.float64)
        b = np.array([0.5, 0.5], dtype=np.float64)

        result = solve_sum_absolute_values(
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

    def test_loglog_regression_recovers_known_slope(self) -> None:
        x_values = np.log(np.array([10.0, 20.0, 50.0, 100.0], dtype=np.float64))
        y_values = 0.7 + 1.5 * x_values

        regression = _fit_loglog_regression(x_values, y_values)

        self.assertAlmostEqual(regression.p_hat, 1.5, places=10)
        self.assertGreaterEqual(regression.ci_upper_95, regression.ci_lower_95)
        self.assertGreaterEqual(regression.p_value, 0.0)
        self.assertLessEqual(regression.p_value, 1.0)

    def test_mu_comparison_history_selection_keeps_required_points(self) -> None:
        history = [GapTracePoint(iteration=0, stage=1, mu=0.8, objective_value=1.0, dual_value=0.1, gap=0.9)]
        history.extend(
            GapTracePoint(iteration=iteration, stage=1, mu=0.8, objective_value=1.0, dual_value=0.2, gap=0.8)
            for iteration in range(1, 151)
        )
        history.extend(
            GapTracePoint(iteration=iteration, stage=2, mu=0.4, objective_value=0.8, dual_value=0.3, gap=0.5)
            for iteration in range(151, 251)
        )

        selected = _select_history_points(history)
        selected_iterations = {(point.iteration, point.stage) for point in selected}

        self.assertIn((0, 1), selected_iterations)
        self.assertIn((1, 1), selected_iterations)
        self.assertIn((100, 1), selected_iterations)
        self.assertIn((150, 1), selected_iterations)
        self.assertIn((151, 2), selected_iterations)
        self.assertIn((200, 2), selected_iterations)
        self.assertIn((250, 2), selected_iterations)


if __name__ == "__main__":
    unittest.main()
