from __future__ import annotations

import argparse
import csv
from pathlib import Path

from nesterov_smoothing import ContinuationConfig, run_sum_absolute_grid, sum_absolute_default_grid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sum-of-absolute-values piece-wise linear example from Nesterov (2005)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for random matrices and offsets.")
    parser.add_argument(
        "--eps",
        type=float,
        nargs="*",
        default=None,
        help="Accuracies to run. Defaults to the built-in example grid.",
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="*",
        default=None,
        help="Override the numbers of absolute-value terms.",
    )
    parser.add_argument(
        "--n-values",
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
        "--check-every",
        type=int,
        default=None,
        help="Override the periodic exact-gap check frequency.",
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
        print(f"\nSum-of-absolute-values runs for epsilon = {epsilon:g}")
        print("m     n      iter     pred     %pred    value       gap         mu-mode       stages  time[s]   status")
        print("----  -----  -------  -------  -------  ----------  ----------  ------------  ------  --------  --------")
        for row in sorted(by_epsilon[epsilon], key=lambda item: (int(item["m"]), int(item["n"]))):
            status = "ok" if bool(row["converged"]) else "not-met"
            print(
                f"{int(row['m']):4d}  {int(row['n']):5d}  "
                f"{int(row['iterations']):7d}  {int(row['predicted_iterations']):7d}  "
                f"{float(row['percent_of_predicted']):6.1f}%  {float(row['objective_value']):10.3e}  "
                f"{float(row['gap']):10.3e}  {str(row['mu_mode']):12s}  "
                f"{int(row['stages']):6d}  {float(row['elapsed_seconds']):8.2f}  {status}"
            )


def main() -> None:
    args = _parse_args()
    grid = sum_absolute_default_grid()
    epsilons = tuple(grid) if args.eps is None or len(args.eps) == 0 else tuple(args.eps)
    continuation = None
    if args.continuation:
        continuation = ContinuationConfig(
            start_factor=args.mu_start_factor,
            decay=args.mu_decay,
            stage_factor=args.stage_factor,
            max_stages=args.max_stages,
        )

    cells = run_sum_absolute_grid(
        base_seed=args.seed,
        epsilons=epsilons,
        m_values=None if args.m_values is None or len(args.m_values) == 0 else tuple(args.m_values),
        n_values=None if args.n_values is None or len(args.n_values) == 0 else tuple(args.n_values),
        radius=args.radius,
        check_frequency=args.check_every,
        max_iterations=args.max_iterations,
        continuation=continuation,
        monotone_y=args.monotone_y,
    )

    rows = [
        {
            "epsilon": cell.epsilon,
            "m": cell.m,
            "n": cell.n,
            "radius": cell.radius,
            "iterations": cell.iterations,
            "predicted_iterations": cell.predicted_iterations,
            "percent_of_predicted": cell.percent_of_predicted,
            "objective_value": cell.objective_value,
            "dual_value": cell.dual_value,
            "gap": cell.gap,
            "mu": cell.mu,
            "lipschitz": cell.lipschitz,
            "operator_norm": cell.operator_norm,
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
