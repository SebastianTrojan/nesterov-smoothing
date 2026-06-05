from __future__ import annotations

import argparse
import csv
import math
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from nesterov_continuous_location import _sample_points_in_l2_ball, solve_continuous_location
from nesterov_core import GapTracePoint
from nesterov_matrix_game import solve_paper_matrix_game
from nesterov_piecewise_linear import solve_piecewise_linear_max_abs
from nesterov_smoothing import ContinuationConfig
from nesterov_sum_absolute import solve_sum_absolute_values

PROBLEMS: tuple[str, ...] = (
    "matrix_game",
    "continuous_location",
    "piecewise_linear_max_abs",
    "sum_absolute",
)
DEFAULT_EPSILONS: tuple[float, ...] = (1.0e-2, 1.0e-3, 1.0e-4)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

PROBLEM_LABELS = {
    "matrix_game": "Matrix game",
    "continuous_location": "Continuous location",
    "piecewise_linear_max_abs": "Maximum of absolute values",
    "sum_absolute": "Sum of absolute values",
}

MU_MODE_LABELS = {
    "fixed_mu": r"Fixed $\mu$",
    "continuation_mu": "Continuation",
}

EPSILON_COLORS = {
    1.0e-2: "tab:blue",
    1.0e-3: "tab:orange",
    1.0e-4: "tab:green",
}


@dataclass
class ComparisonRawRow:
    problem: str
    epsilon: float
    seed: int
    mu_mode: str
    iterations: int
    elapsed_seconds: float
    objective_value: float
    dual_value: float
    gap: float
    mu_final: float
    stages: int
    converged: bool
    predicted_iterations: int
    percent_of_predicted: float


@dataclass
class ComparisonSummaryRow:
    problem: str
    epsilon: float
    mu_mode: str
    mean_iterations: float
    std_iterations: float
    mean_gap: float
    max_gap: float
    mean_stages: float
    converged_all: bool
    mean_elapsed_seconds: float


@dataclass
class ComparisonSpeedupRow:
    problem: str
    epsilon: float
    seed: int
    fixed_iterations: int
    continuation_iterations: int
    iteration_speedup: float
    fixed_gap: float
    continuation_gap: float
    fixed_elapsed_seconds: float
    continuation_elapsed_seconds: float
    runtime_speedup: float


@dataclass
class GapHistoryRow:
    problem: str
    epsilon: float
    seed: int
    mu_mode: str
    iteration: int
    stage: int
    mu: float
    gap: float
    objective_value: float
    dual_value: float


def _stable_seed(problem: str, seed: int, epsilon: float) -> int:
    payload = f"{problem}|{seed}|{epsilon:.12e}".encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


def _problem_instance(problem: str, seed: int, epsilon: float) -> dict[str, Any]:
    rng = np.random.default_rng(_stable_seed(problem, seed, epsilon))
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


def _sample_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _iteration_or_runtime_speedup(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.inf
    return numerator / denominator


def _select_history_points(points: list[GapTracePoint]) -> list[GapTracePoint]:
    if not points:
        return []
    ordered = sorted(points, key=lambda point: (point.iteration, point.stage))
    total_iterations = ordered[-1].iteration
    stride = 100 if total_iterations < 50_000 else 1000

    selected_pairs: set[tuple[int, int]] = set()
    selected_pairs.add((ordered[0].iteration, ordered[0].stage))
    selected_pairs.add((ordered[-1].iteration, ordered[-1].stage))

    for point in ordered:
        if point.iteration % stride == 0:
            selected_pairs.add((point.iteration, point.stage))

    by_stage: dict[int, list[GapTracePoint]] = {}
    for point in ordered:
        by_stage.setdefault(point.stage, []).append(point)

    for stage_points in by_stage.values():
        selected_pairs.add((stage_points[0].iteration, stage_points[0].stage))
        selected_pairs.add((stage_points[-1].iteration, stage_points[-1].stage))
        if stage_points[0].iteration == 0 and len(stage_points) > 1:
            selected_pairs.add((stage_points[1].iteration, stage_points[1].stage))

    return [point for point in ordered if (point.iteration, point.stage) in selected_pairs]


def _run_single(
    *,
    problem: str,
    epsilon: float,
    seed: int,
    instance: dict[str, Any],
    continuation: ContinuationConfig | None,
    collect_history: bool,
) -> tuple[ComparisonRawRow, list[GapHistoryRow]]:
    trace_points: list[GapTracePoint] = []
    trace_callback = trace_points.append if collect_history else None
    mu_mode = "continuation_mu" if continuation is not None else "fixed_mu"

    if problem == "matrix_game":
        result = solve_paper_matrix_game(
            instance["A"],
            epsilon,
            check_frequency=1,
            continuation=continuation,
            trace_callback=trace_callback,
        )
        objective_value = result.primal_value
    elif problem == "continuous_location":
        result = solve_continuous_location(
            instance["cities"],
            epsilon,
            weights=instance["weights"],
            radius=instance["radius"],
            check_frequency=1,
            continuation=continuation,
            trace_callback=trace_callback,
        )
        objective_value = result.objective_value
    elif problem == "piecewise_linear_max_abs":
        result = solve_piecewise_linear_max_abs(
            instance["A"],
            instance["b"],
            epsilon,
            radius=instance["radius"],
            check_frequency=1,
            continuation=continuation,
            trace_callback=trace_callback,
        )
        objective_value = result.objective_value
    elif problem == "sum_absolute":
        result = solve_sum_absolute_values(
            instance["A"],
            instance["b"],
            epsilon,
            radius=instance["radius"],
            check_frequency=1,
            continuation=continuation,
            trace_callback=trace_callback,
        )
        objective_value = result.objective_value
    else:
        raise ValueError(f"Unknown problem: {problem}")

    raw_row = ComparisonRawRow(
        problem=problem,
        epsilon=epsilon,
        seed=seed,
        mu_mode=mu_mode,
        iterations=result.iterations,
        elapsed_seconds=result.elapsed_seconds,
        objective_value=objective_value,
        dual_value=result.dual_value,
        gap=result.gap,
        mu_final=result.mu,
        stages=max(1, len(result.continuation_stages)),
        converged=result.converged,
        predicted_iterations=result.predicted_iterations,
        percent_of_predicted=100.0 * result.iterations / result.predicted_iterations,
    )

    if not collect_history:
        return raw_row, []

    history_rows = [
        GapHistoryRow(
            problem=problem,
            epsilon=epsilon,
            seed=seed,
            mu_mode=mu_mode,
            iteration=point.iteration,
            stage=point.stage,
            mu=point.mu,
            gap=point.gap,
            objective_value=point.objective_value,
            dual_value=point.dual_value,
        )
        for point in _select_history_points(trace_points)
    ]
    return raw_row, history_rows


def _build_summary_rows(raw_rows: list[ComparisonRawRow]) -> list[ComparisonSummaryRow]:
    grouped: dict[tuple[str, float, str], list[ComparisonRawRow]] = {}
    for row in raw_rows:
        grouped.setdefault((row.problem, row.epsilon, row.mu_mode), []).append(row)

    summary_rows: list[ComparisonSummaryRow] = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2])):
        problem, epsilon, mu_mode = key
        bucket = grouped[key]
        iterations = np.array([row.iterations for row in bucket], dtype=np.float64)
        gaps = np.array([row.gap for row in bucket], dtype=np.float64)
        stages = np.array([row.stages for row in bucket], dtype=np.float64)
        elapsed = np.array([row.elapsed_seconds for row in bucket], dtype=np.float64)
        summary_rows.append(
            ComparisonSummaryRow(
                problem=problem,
                epsilon=epsilon,
                mu_mode=mu_mode,
                mean_iterations=float(np.mean(iterations)),
                std_iterations=_sample_std(iterations),
                mean_gap=float(np.mean(gaps)),
                max_gap=float(np.max(gaps)),
                mean_stages=float(np.mean(stages)),
                converged_all=all(row.converged for row in bucket),
                mean_elapsed_seconds=float(np.mean(elapsed)),
            )
        )
    return summary_rows


def _build_speedup_rows(raw_rows: list[ComparisonRawRow]) -> list[ComparisonSpeedupRow]:
    fixed_rows = {
        (row.problem, row.epsilon, row.seed): row
        for row in raw_rows
        if row.mu_mode == "fixed_mu"
    }
    continuation_rows = {
        (row.problem, row.epsilon, row.seed): row
        for row in raw_rows
        if row.mu_mode == "continuation_mu"
    }

    speedup_rows: list[ComparisonSpeedupRow] = []
    for key in sorted(fixed_rows):
        if key not in continuation_rows:
            continue
        fixed = fixed_rows[key]
        continuation = continuation_rows[key]
        speedup_rows.append(
            ComparisonSpeedupRow(
                problem=fixed.problem,
                epsilon=fixed.epsilon,
                seed=fixed.seed,
                fixed_iterations=fixed.iterations,
                continuation_iterations=continuation.iterations,
                iteration_speedup=_iteration_or_runtime_speedup(fixed.iterations, continuation.iterations),
                fixed_gap=fixed.gap,
                continuation_gap=continuation.gap,
                fixed_elapsed_seconds=fixed.elapsed_seconds,
                continuation_elapsed_seconds=continuation.elapsed_seconds,
                runtime_speedup=_iteration_or_runtime_speedup(fixed.elapsed_seconds, continuation.elapsed_seconds),
            )
        )
    return speedup_rows


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


def _epsilon_name(epsilon: float) -> str:
    if epsilon == 1.0e-2:
        return r"10^{-2}"
    if epsilon == 1.0e-3:
        return r"10^{-3}"
    if epsilon == 1.0e-4:
        return r"10^{-4}"
    return f"{epsilon:g}"


def _fmt_int(value: float) -> str:
    return f"{int(round(float(value))):,}".replace(",", r"\,")


def _fmt_float(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _latex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def _make_booktabs_table(headers: list[str], rows: list[list[str]], align: str) -> str:
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_tables(output_dir: Path, summary_rows: list[ComparisonSummaryRow], speedup_rows: list[ComparisonSpeedupRow]) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_table_rows = [
        [
            _latex_escape(PROBLEM_LABELS.get(row.problem, row.problem)),
            f"${_epsilon_name(row.epsilon)}$",
            _latex_escape(row.mu_mode),
            _fmt_int(row.mean_iterations),
            _fmt_float(row.std_iterations, 2),
            _fmt_float(row.mean_gap, 6),
            _fmt_float(row.max_gap, 6),
            _fmt_float(row.mean_stages, 2),
        ]
        for row in summary_rows
    ]
    _write_text(
        tables_dir / "mu_comparison_summary.tex",
        _make_booktabs_table(
            headers=[
                "Problem",
                r"$\varepsilon$",
                "$\\mu$ mode",
                "Mean iterations",
                "Std iterations",
                "Mean gap",
                "Max gap",
                "Mean stages",
            ],
            rows=summary_table_rows,
            align="lclrrrrr",
        ),
    )

    grouped_speedup: dict[tuple[str, float], list[ComparisonSpeedupRow]] = {}
    for row in speedup_rows:
        grouped_speedup.setdefault((row.problem, row.epsilon), []).append(row)

    speedup_table_rows: list[list[str]] = []
    for (problem, epsilon) in sorted(grouped_speedup, key=lambda item: (item[0], item[1])):
        bucket = grouped_speedup[(problem, epsilon)]
        values = np.array([row.iteration_speedup for row in bucket], dtype=np.float64)
        speedup_table_rows.append(
            [
                _latex_escape(PROBLEM_LABELS.get(problem, problem)),
                f"${_epsilon_name(epsilon)}$",
                _fmt_float(float(np.mean(values)), 3),
                _fmt_float(float(np.min(values)), 3),
                _fmt_float(float(np.max(values)), 3),
            ]
        )

    _write_text(
        tables_dir / "mu_comparison_speedup.tex",
        _make_booktabs_table(
            headers=[
                "Problem",
                r"$\varepsilon$",
                "Mean iteration speedup",
                "Min iteration speedup",
                "Max iteration speedup",
            ],
            rows=speedup_table_rows,
            align="lcrrr",
        ),
    )


def _plot_gap_histories(output_dir: Path, history_rows: list[GapHistoryRow], epsilons: tuple[float, ...]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for problem in PROBLEMS:
        plt.figure(figsize=(10, 6))
        problem_rows = [row for row in history_rows if row.problem == problem]
        for epsilon in epsilons:
            color = EPSILON_COLORS.get(epsilon)
            if color is None:
                color = None
            for mu_mode, linestyle in (("fixed_mu", "-"), ("continuation_mu", "--")):
                curve_rows = [
                    row
                    for row in problem_rows
                    if row.epsilon == epsilon and row.mu_mode == mu_mode
                ]
                curve_rows.sort(key=lambda row: row.iteration)
                if not curve_rows:
                    continue
                x_values = [row.iteration for row in curve_rows]
                y_values = [max(row.gap, 1.0e-16) for row in curve_rows]
                plt.plot(
                    x_values,
                    y_values,
                    linestyle=linestyle,
                    color=color,
                    linewidth=1.8,
                    label=f"{mu_mode}, ε={epsilon:g}",
                )

        for epsilon in epsilons:
            plt.axhline(epsilon, color=EPSILON_COLORS.get(epsilon, "gray"), linestyle=":", linewidth=0.9, alpha=0.8)

        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel("primal-dual gap")
        plt.title(f"Gap history: {PROBLEM_LABELS.get(problem, problem)}")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(plots_dir / f"mu_gap_history_{problem}.png", dpi=160)
        plt.close()


def run_mu_comparison(
    *,
    seeds: tuple[int, ...],
    epsilons: tuple[float, ...],
    history_seed: int,
    continuation: ContinuationConfig,
    collect_history: bool,
) -> tuple[list[ComparisonRawRow], list[ComparisonSummaryRow], list[ComparisonSpeedupRow], list[GapHistoryRow]]:
    raw_rows: list[ComparisonRawRow] = []
    history_rows: list[GapHistoryRow] = []

    for problem in PROBLEMS:
        for epsilon in epsilons:
            for seed in seeds:
                instance = _problem_instance(problem, seed, epsilon)
                fixed_row, fixed_history = _run_single(
                    problem=problem,
                    epsilon=epsilon,
                    seed=seed,
                    instance=instance,
                    continuation=None,
                    collect_history=collect_history and seed == history_seed,
                )
                continuation_row, continuation_history = _run_single(
                    problem=problem,
                    epsilon=epsilon,
                    seed=seed,
                    instance=instance,
                    continuation=continuation,
                    collect_history=collect_history and seed == history_seed,
                )
                raw_rows.extend([fixed_row, continuation_row])
                history_rows.extend(fixed_history)
                history_rows.extend(continuation_history)

    summary_rows = _build_summary_rows(raw_rows)
    speedup_rows = _build_speedup_rows(raw_rows)
    return raw_rows, summary_rows, speedup_rows, history_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed mu against continuation mu across all implemented problems.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory in which CSVs, plots, and tables are written.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS), help="Seeds used in the main comparison.")
    parser.add_argument("--epsilons", type=float, nargs="*", default=list(DEFAULT_EPSILONS), help="Accuracies to compare.")
    parser.add_argument("--history-seed", type=int, default=0, help="Seed used for detailed gap-history plots.")
    parser.add_argument("--start-factor", type=float, default=8.0, help="Initial continuation mu factor relative to mu_final.")
    parser.add_argument("--decay", type=float, default=0.5, help="Geometric decay applied after each continuation stage.")
    parser.add_argument("--stage-factor", type=float, default=1.0, help="Stage target factor for intermediate continuation stages.")
    parser.add_argument("--skip-history", action="store_true", help="Skip the gap-history CSV generation.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip the gap-history plot generation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    continuation = ContinuationConfig(
        start_factor=args.start_factor,
        decay=args.decay,
        stage_factor=args.stage_factor,
    )
    seeds = tuple(args.seeds)
    epsilons = tuple(args.epsilons)

    raw_path = output_dir / "mu_comparison.csv"
    summary_path = output_dir / "mu_comparison_summary.csv"
    speedup_path = output_dir / "mu_comparison_speedup.csv"
    history_path = output_dir / "mu_gap_history.csv"
    summary_table_path = output_dir / "tables" / "mu_comparison_summary.tex"
    speedup_table_path = output_dir / "tables" / "mu_comparison_speedup.tex"
    plot_paths = [output_dir / "plots" / f"mu_gap_history_{problem}.png" for problem in PROBLEMS]

    output_paths = [raw_path, summary_path, speedup_path, summary_table_path, speedup_table_path]
    if not args.skip_history:
        output_paths.append(history_path)
    if not args.skip_plots and not args.skip_history:
        output_paths.extend(plot_paths)
    _ensure_can_write(output_paths, overwrite=args.overwrite)

    raw_rows, summary_rows, speedup_rows, history_rows = run_mu_comparison(
        seeds=seeds,
        epsilons=epsilons,
        history_seed=args.history_seed,
        continuation=continuation,
        collect_history=not args.skip_history,
    )

    _write_csv(raw_path, raw_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(speedup_path, speedup_rows)
    if not args.skip_history:
        _write_csv(history_path, history_rows)
    _write_tables(output_dir, summary_rows, speedup_rows)
    if not args.skip_plots and not args.skip_history:
        _plot_gap_histories(output_dir, history_rows, epsilons)


if __name__ == "__main__":
    main()
