from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

import numpy as np

from nesterov_continuous_location import _sample_points_in_l2_ball, solve_continuous_location
from nesterov_matrix_game import solve_paper_matrix_game
from nesterov_piecewise_linear import solve_piecewise_linear_max_abs
from nesterov_sum_absolute import solve_sum_absolute_values

EPSILONS: tuple[float, ...] = (1.0e-1, 5.0e-2, 1.0e-2, 5.0e-3, 1.0e-3, 5.0e-4, 1.0e-4)
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
PROBLEMS: tuple[str, ...] = (
    "matrix_game",
    "continuous_location",
    "piecewise_linear_max_abs",
    "sum_absolute",
)

PROBLEM_LABELS = {
    "matrix_game": "Matrix game",
    "continuous_location": "Continuous location",
    "piecewise_linear_max_abs": "Maximum of absolute values",
    "sum_absolute": "Sum of absolute values",
}


@dataclass
class ScalingRawRow:
    problem: str
    seed: int
    epsilon: float
    inverse_epsilon: float
    log_inverse_epsilon: float
    iterations: int
    predicted_iterations: int
    elapsed_seconds: float
    gap: float
    converged: bool


@dataclass
class ScalingSummaryRow:
    problem: str
    epsilon: float
    inverse_epsilon: float
    log_inverse_epsilon: float
    mean_iterations: float
    std_iterations: float
    mean_runtime: float
    mean_gap: float
    max_gap: float


@dataclass
class ScalingRegressionRow:
    problem: str
    p_hat: float
    standard_error: float
    p_value: float
    ci_lower_95: float
    ci_upper_95: float


def _sample_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _normal_two_sided_p_value(z_value: float) -> float:
    return float(math.erfc(abs(z_value) / math.sqrt(2.0)))


def _fit_loglog_regression(x_values: np.ndarray, y_values: np.ndarray) -> ScalingRegressionRow:
    design = np.column_stack([np.ones_like(x_values), x_values])
    xtx = design.T @ design
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ design.T @ y_values
    residuals = y_values - design @ beta
    degrees_of_freedom = x_values.size - 2
    if degrees_of_freedom <= 0:
        raise ValueError("At least three observations are required for regression.")
    sigma_squared = float((residuals @ residuals) / degrees_of_freedom)
    covariance = sigma_squared * xtx_inv
    standard_error = float(math.sqrt(max(covariance[1, 1], 0.0)))
    if standard_error == 0.0:
        z_value = 0.0 if math.isclose(float(beta[1]), 1.0) else math.inf
    else:
        z_value = float((beta[1] - 1.0) / standard_error)
    p_value = _normal_two_sided_p_value(z_value)
    ci_half_width = 1.96 * standard_error
    return ScalingRegressionRow(
        problem="",
        p_hat=float(beta[1]),
        standard_error=standard_error,
        p_value=p_value,
        ci_lower_95=float(beta[1] - ci_half_width),
        ci_upper_95=float(beta[1] + ci_half_width),
    )


def _problem_instance(problem: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if problem == "matrix_game":
        return {"A": rng.uniform(-1.0, 1.0, size=(300, 1000))}
    if problem == "continuous_location":
        radius = 1.0
        cities = _sample_points_in_l2_ball(rng, 50, 5, 0.8 * radius)
        weights = rng.uniform(0.5, 1.5, size=50)
        return {"cities": cities, "weights": weights, "radius": radius}
    if problem == "piecewise_linear_max_abs":
        radius = 1.0
        return {
            "A": rng.uniform(-1.0, 1.0, size=(300, 200)),
            "b": rng.uniform(-1.0, 1.0, size=300),
            "radius": radius,
        }
    if problem == "sum_absolute":
        radius = 1.0
        return {
            "A": rng.uniform(-1.0, 1.0, size=(300, 200)),
            "b": rng.uniform(-1.0, 1.0, size=300),
            "radius": radius,
        }
    raise ValueError(f"Unknown problem: {problem}")


def _run_single(problem: str, instance: dict[str, Any], epsilon: float) -> ScalingRawRow:
    inverse_epsilon = 1.0 / epsilon
    log_inverse_epsilon = math.log(inverse_epsilon)

    if problem == "matrix_game":
        result = solve_paper_matrix_game(instance["A"], epsilon, check_frequency=1)
        return ScalingRawRow(
            problem=problem,
            seed=-1,
            epsilon=epsilon,
            inverse_epsilon=inverse_epsilon,
            log_inverse_epsilon=log_inverse_epsilon,
            iterations=result.iterations,
            predicted_iterations=result.predicted_iterations,
            elapsed_seconds=result.elapsed_seconds,
            gap=result.gap,
            converged=result.converged,
        )

    if problem == "continuous_location":
        result = solve_continuous_location(
            instance["cities"],
            epsilon,
            weights=instance["weights"],
            radius=instance["radius"],
            check_frequency=1,
        )
        return ScalingRawRow(
            problem=problem,
            seed=-1,
            epsilon=epsilon,
            inverse_epsilon=inverse_epsilon,
            log_inverse_epsilon=log_inverse_epsilon,
            iterations=result.iterations,
            predicted_iterations=result.predicted_iterations,
            elapsed_seconds=result.elapsed_seconds,
            gap=result.gap,
            converged=result.converged,
        )

    if problem == "piecewise_linear_max_abs":
        result = solve_piecewise_linear_max_abs(
            instance["A"],
            instance["b"],
            epsilon,
            radius=instance["radius"],
            check_frequency=1,
        )
        return ScalingRawRow(
            problem=problem,
            seed=-1,
            epsilon=epsilon,
            inverse_epsilon=inverse_epsilon,
            log_inverse_epsilon=log_inverse_epsilon,
            iterations=result.iterations,
            predicted_iterations=result.predicted_iterations,
            elapsed_seconds=result.elapsed_seconds,
            gap=result.gap,
            converged=result.converged,
        )

    if problem == "sum_absolute":
        result = solve_sum_absolute_values(
            instance["A"],
            instance["b"],
            epsilon,
            radius=instance["radius"],
            check_frequency=1,
        )
        return ScalingRawRow(
            problem=problem,
            seed=-1,
            epsilon=epsilon,
            inverse_epsilon=inverse_epsilon,
            log_inverse_epsilon=log_inverse_epsilon,
            iterations=result.iterations,
            predicted_iterations=result.predicted_iterations,
            elapsed_seconds=result.elapsed_seconds,
            gap=result.gap,
            converged=result.converged,
        )

    raise ValueError(f"Unknown problem: {problem}")


def run_scaling_experiment() -> tuple[list[ScalingRawRow], list[ScalingSummaryRow], list[ScalingRegressionRow]]:
    raw_rows: list[ScalingRawRow] = []
    for problem in PROBLEMS:
        for seed in SEEDS:
            instance = _problem_instance(problem, seed)
            for epsilon in EPSILONS:
                row = _run_single(problem, instance, epsilon)
                row.seed = seed
                raw_rows.append(row)

    summary_rows: list[ScalingSummaryRow] = []
    regression_rows: list[ScalingRegressionRow] = []

    for problem in PROBLEMS:
        problem_rows = [row for row in raw_rows if row.problem == problem]
        grouped: dict[float, list[ScalingRawRow]] = {}
        for row in problem_rows:
            grouped.setdefault(row.epsilon, []).append(row)

        for epsilon in sorted(grouped, reverse=True):
            bucket = grouped[epsilon]
            iterations = np.array([row.iterations for row in bucket], dtype=np.float64)
            runtimes = np.array([row.elapsed_seconds for row in bucket], dtype=np.float64)
            gaps = np.array([row.gap for row in bucket], dtype=np.float64)
            inverse_epsilon = bucket[0].inverse_epsilon
            log_inverse_epsilon = bucket[0].log_inverse_epsilon
            summary_rows.append(
                ScalingSummaryRow(
                    problem=problem,
                    epsilon=epsilon,
                    inverse_epsilon=inverse_epsilon,
                    log_inverse_epsilon=log_inverse_epsilon,
                    mean_iterations=float(np.mean(iterations)),
                    std_iterations=_sample_std(iterations),
                    mean_runtime=float(np.mean(runtimes)),
                    mean_gap=float(np.mean(gaps)),
                    max_gap=float(np.max(gaps)),
                )
            )

        x_values = np.array([row.log_inverse_epsilon for row in problem_rows], dtype=np.float64)
        y_values = np.log(np.array([row.iterations for row in problem_rows], dtype=np.float64))
        regression = _fit_loglog_regression(x_values, y_values)
        regression.problem = problem
        regression_rows.append(regression)

    return raw_rows, summary_rows, regression_rows


def _ensure_can_write(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing files without --overwrite: {existing_text}")


def _write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)




def create_plots(output_dir: Path, summary_rows: list[ScalingSummaryRow]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    problem_groups: dict[str, list[ScalingSummaryRow]] = {}
    for row in summary_rows:
        problem_groups.setdefault(row.problem, []).append(row)

    iterations_path = plots_dir / "epsilon_scaling_iterations_vs_inverse_epsilon.png"
    plt.figure(figsize=(9, 6))
    for problem in PROBLEMS:
        rows = sorted(problem_groups[problem], key=lambda row: row.inverse_epsilon)
        x_values = [row.inverse_epsilon for row in rows]
        y_values = [row.mean_iterations for row in rows]
        plt.plot(x_values, y_values, marker="o", label=PROBLEM_LABELS.get(problem, problem))
    plt.xlabel("1 / epsilon")
    plt.ylabel("mean iterations")
    plt.title("Mean iterations vs 1/epsilon")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(iterations_path, dpi=160)
    plt.close()

    loglog_path = plots_dir / "epsilon_scaling_loglog.png"
    plt.figure(figsize=(9, 6))
    for problem in PROBLEMS:
        rows = sorted(problem_groups[problem], key=lambda row: row.inverse_epsilon)
        x_values = [row.log_inverse_epsilon for row in rows]
        y_values = [math.log(row.mean_iterations) for row in rows]
        plt.plot(x_values, y_values, marker="o", label=PROBLEM_LABELS.get(problem, problem))
    plt.xlabel("log(1 / epsilon)")
    plt.ylabel("log(mean iterations)")
    plt.title("Log-log epsilon scaling")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loglog_path, dpi=160)
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed-mu epsilon-scaling experiment across all implemented problems."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory in which CSVs and plots are written.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    raw_path = output_dir / "epsilon_scaling.csv"
    summary_path = output_dir / "epsilon_scaling_summary.csv"
    regression_path = output_dir / "epsilon_scaling_regression.csv"
    plot_dir = output_dir / "plots"
    iterations_plot_path = plot_dir / "epsilon_scaling_iterations_vs_inverse_epsilon.png"
    loglog_plot_path = plot_dir / "epsilon_scaling_loglog.png"

    _ensure_can_write(
        [raw_path, summary_path, regression_path, iterations_plot_path, loglog_plot_path],
        overwrite=args.overwrite,
    )

    raw_rows, summary_rows, regression_rows = run_scaling_experiment()
    _write_csv(raw_path, raw_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(regression_path, regression_rows)
    create_plots(output_dir, summary_rows)


if __name__ == "__main__":
    main()
