from __future__ import annotations

import numpy as np

from nesterov_continuous_location import (
    ContinuousLocationExperimentCell,
    ContinuousLocationResult,
    continuous_location_default_grid,
    run_continuous_location_grid,
    solve_continuous_location,
)
from nesterov_core import (
    ContinuationConfig,
    ContinuationStage,
    FloatArray,
    GapTracePoint,
    MuGapPoint,
    save_mu_gap_history_csv,
    project_to_l2_ball,
    project_to_simplex,
    simplex_entropy_argmin,
    simplex_l1_squared_step,
)
from nesterov_matrix_game import (
    MatrixGameExperimentCell,
    PaperMatrixGameResult,
    paper_section6_grid,
    run_paper_matrix_game_grid,
    solve_paper_matrix_game,
)
from nesterov_piecewise_linear import (
    PiecewiseLinearExperimentCell,
    PiecewiseLinearResult,
    piecewise_linear_default_grid,
    run_piecewise_linear_grid,
    solve_piecewise_linear_max_abs,
)
from nesterov_sum_absolute import (
    SumAbsoluteExperimentCell,
    SumAbsoluteResult,
    run_sum_absolute_grid,
    solve_sum_absolute_values,
    sum_absolute_default_grid,
)

_project_to_l2_ball = project_to_l2_ball
_project_to_simplex = project_to_simplex
_simplex_entropy_argmin = simplex_entropy_argmin
_simplex_l1_squared_step = simplex_l1_squared_step

__all__ = [
    "ContinuousLocationExperimentCell",
    "ContinuousLocationResult",
    "ContinuationConfig",
    "ContinuationStage",
    "FloatArray",
    "GapTracePoint",
    "MatrixGameExperimentCell",
    "MuGapPoint",
    "PaperMatrixGameResult",
    "PiecewiseLinearExperimentCell",
    "PiecewiseLinearResult",
    "SumAbsoluteExperimentCell",
    "SumAbsoluteResult",
    "_project_to_l2_ball",
    "_project_to_simplex",
    "_simplex_entropy_argmin",
    "_simplex_l1_squared_step",
    "continuous_location_default_grid",
    "paper_section6_grid",
    "piecewise_linear_default_grid",
    "sum_absolute_default_grid",
    "project_to_l2_ball",
    "project_to_simplex",
    "run_continuous_location_grid",
    "run_paper_matrix_game_grid",
    "run_piecewise_linear_grid",
    "run_sum_absolute_grid",
    "save_mu_gap_history_csv",
    "simplex_entropy_argmin",
    "simplex_l1_squared_step",
    "solve_continuous_location",
    "solve_paper_matrix_game",
    "solve_piecewise_linear_max_abs",
    "solve_sum_absolute_values",
]


def _demo() -> None:
    matrix_A = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    matrix_result = solve_paper_matrix_game(matrix_A, epsilon=1.0e-2, check_frequency=10)
    print("Matrix game demo")
    print("Iterations :", matrix_result.iterations)
    print("Predicted  :", matrix_result.predicted_iterations)
    print("Primal x   :", matrix_result.x)
    print("Dual u     :", matrix_result.u)
    print("Gap        :", matrix_result.gap)

    cities = np.array([[-0.75, 0.0], [0.75, 0.0], [0.0, 0.5]], dtype=np.float64)
    location_result = solve_continuous_location(cities, epsilon=1.0e-1, radius=1.0, check_frequency=10)
    print("\nContinuous location demo")
    print("Iterations :", location_result.iterations)
    print("Predicted  :", location_result.predicted_iterations)
    print("Center x   :", location_result.x)
    print("Value      :", location_result.objective_value)
    print("Bound      :", location_result.theoretical_gap_bound)

    piecewise_A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=np.float64)
    piecewise_b = np.array([0.2, -0.3, 0.1], dtype=np.float64)
    piecewise_result = solve_piecewise_linear_max_abs(
        piecewise_A,
        piecewise_b,
        epsilon=1.0e-1,
        radius=1.0,
        check_frequency=10,
    )
    print("\nPiece-wise linear demo")
    print("Iterations :", piecewise_result.iterations)
    print("Predicted  :", piecewise_result.predicted_iterations)
    print("Center x   :", piecewise_result.x)
    print("Value      :", piecewise_result.objective_value)
    print("Gap        :", piecewise_result.gap)

    sum_abs_result = solve_sum_absolute_values(
        piecewise_A,
        piecewise_b,
        epsilon=1.0e-1,
        radius=1.0,
        check_frequency=10,
    )
    print("\nSum of absolute values demo")
    print("Iterations :", sum_abs_result.iterations)
    print("Predicted  :", sum_abs_result.predicted_iterations)
    print("Center x   :", sum_abs_result.x)
    print("Value      :", sum_abs_result.objective_value)
    print("Gap        :", sum_abs_result.gap)


if __name__ == "__main__":
    _demo()
