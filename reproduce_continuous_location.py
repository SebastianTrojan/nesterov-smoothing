from __future__ import annotations

import argparse
import csv
from pathlib import Path

from nesterov_smoothing import ContinuationConfig, continuous_location_default_grid, run_continuous_location_grid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the continuous location example from Nesterov (2005)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for random cities and weights.")
    parser.add_argument(
        "--eps",
        type=float,
        nargs="*",
        default=None,
        help="Accuracies to run. Defaults to the small built-in example grid.",
    )
    parser.add_argument(
        "--num-cities",
        type=int,
        nargs="*",
        default=None,
        help="Override the numbers of cities.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="*",
        default=None,
        help="Override the ambient dimensions.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Override the feasible ball radius.",
    )
    parser.add_argument(
        "--city-radius",
        type=float,
        default=None,
        help="Radius used to sample the cities. Defaults to 0.8 * radius.",
    )
    parser.add_argument(
        "--check-every",
        type=int,
        default=None,
        help="Override the periodic bound check frequency.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override the per-instance iteration budget.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--continuation",
        action="store_true",
        help="Use a multi-stage changing-mu continuation scheme.",
    )
    parser.add_argument("--mu-start-factor", type=float, default=8.0, help="Initial mu factor relative to the final paper value.")
    parser.add_argument("--mu-decay", type=float, default=0.5, help="Geometric decay applied after each continuation stage.")
    parser.add_argument("--stage-factor", type=float, default=1.0, help="Stage target factor for the current smoothing bias.")
    parser.add_argument("--max-stages", type=int, default=None, help="Optional cap on the number of continuation stages.")
    parser.add_argument(
        "--monotone-y",
        action="store_true",
        help="Use the paper's modified monotone acceptance rule for y_k.",
    )
    return parser.parse_args()


def _print_tables(rows: list[dict[str, object]]) -> None:
    by_epsilon: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        by_epsilon.setdefault(float(row["epsilon"]), []).append(row)

    for epsilon in sorted(by_epsilon):
        print(f"\nContinuous location runs for epsilon = {epsilon:g}")
        print("cities  dim    iter     pred     %pred    value       gap         bound       mu-mode       stages  time[s]   status")
        print("-----  -----  -------  -------  -------  ----------  ----------  ----------  ------------  ------  --------  --------")
        for row in sorted(by_epsilon[epsilon], key=lambda item: (int(item["num_cities"]), int(item["dimension"]))):
            status = "ok" if bool(row["converged"]) else "not-met"
            print(
                f"{int(row['num_cities']):5d}  {int(row['dimension']):5d}  "
                f"{int(row['iterations']):7d}  {int(row['predicted_iterations']):7d}  "
                f"{float(row['percent_of_predicted']):6.1f}%  {float(row['objective_value']):10.3e}  "
                f"{float(row['gap']):10.3e}  {float(row['theoretical_gap_bound']):10.3e}  {str(row['mu_mode']):12s}  "
                f"{int(row['stages']):6d}  {float(row['elapsed_seconds']):8.2f}  {status}"
            )


def main() -> None:
    args = _parse_args()
    grid = continuous_location_default_grid()
    epsilons = tuple(grid) if args.eps is None or len(args.eps) == 0 else tuple(args.eps)
    continuation = None
    if args.continuation:
        continuation = ContinuationConfig(
            start_factor=args.mu_start_factor,
            decay=args.mu_decay,
            stage_factor=args.stage_factor,
            max_stages=args.max_stages,
        )

    cells = run_continuous_location_grid(
        base_seed=args.seed,
        epsilons=epsilons,
        num_cities_values=None if args.num_cities is None or len(args.num_cities) == 0 else tuple(args.num_cities),
        dimension_values=None if args.dimensions is None or len(args.dimensions) == 0 else tuple(args.dimensions),
        radius=args.radius,
        city_radius=args.city_radius,
        check_frequency=args.check_every,
        max_iterations=args.max_iterations,
        continuation=continuation,
        monotone_y=args.monotone_y,
    )

    rows = [
        {
            "epsilon": cell.epsilon,
            "num_cities": cell.num_cities,
            "dimension": cell.dimension,
            "radius": cell.radius,
            "iterations": cell.iterations,
            "predicted_iterations": cell.predicted_iterations,
            "percent_of_predicted": cell.percent_of_predicted,
            "objective_value": cell.objective_value,
            "dual_value": cell.dual_value,
            "gap": cell.gap,
            "theoretical_gap_bound": cell.theoretical_gap_bound,
            "mu": cell.mu,
            "lipschitz": cell.lipschitz,
            "operator_norm": cell.operator_norm,
            "weight_sum": cell.weight_sum,
            "converged": cell.converged,
            "elapsed_seconds": cell.elapsed_seconds,
            "mu_mode": cell.mu_mode,
            "stages": cell.stages,
        }
        for cell in cells
    ]

    _print_tables(rows)

    if args.csv is not None and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
