from __future__ import annotations

import argparse
import csv
from pathlib import Path

from nesterov_smoothing import paper_section6_grid, run_paper_matrix_game_grid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the matrix-game experiments from Section 6 of Nesterov (2005)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for random matrices.")
    parser.add_argument(
        "--eps",
        type=float,
        nargs="*",
        default=None,
        help="Accuracies to run. Defaults to the Section 6 grid.",
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="*",
        default=None,
        help="Override the paper's row dimensions.",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="*",
        default=None,
        help="Override the paper's column dimensions.",
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
    return parser.parse_args()


def _print_tables(rows: list[dict[str, object]]) -> None:
    by_epsilon: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        by_epsilon.setdefault(float(row["epsilon"]), []).append(row)

    for epsilon in sorted(by_epsilon):
        print(f"\nSection 6 reproduction for epsilon = {epsilon:g}")
        print("m     n      iter     pred     %pred    gap         time[s]   status")
        print("----  -----  -------  -------  -------  ----------  --------  --------")
        for row in sorted(by_epsilon[epsilon], key=lambda item: (int(item["m"]), int(item["n"]))):
            status = "ok" if bool(row["converged"]) else "not-met"
            print(
                f"{int(row['m']):4d}  {int(row['n']):5d}  "
                f"{int(row['iterations']):7d}  {int(row['predicted_iterations']):7d}  "
                f"{float(row['percent_of_predicted']):6.1f}%  {float(row['gap']):10.3e}  "
                f"{float(row['elapsed_seconds']):8.2f}  {status}"
            )


def main() -> None:
    args = _parse_args()
    grid = paper_section6_grid()
    epsilons = tuple(grid) if args.eps is None or len(args.eps) == 0 else tuple(args.eps)

    cells = run_paper_matrix_game_grid(
        base_seed=args.seed,
        epsilons=epsilons,
        m_values=None if args.m_values is None or len(args.m_values) == 0 else tuple(args.m_values),
        n_values=None if args.n_values is None or len(args.n_values) == 0 else tuple(args.n_values),
        check_frequency=args.check_every,
        max_iterations=args.max_iterations,
    )

    rows = [
        {
            "epsilon": cell.epsilon,
            "m": cell.m,
            "n": cell.n,
            "iterations": cell.iterations,
            "predicted_iterations": cell.predicted_iterations,
            "percent_of_predicted": cell.percent_of_predicted,
            "gap": cell.gap,
            "primal_value": cell.primal_value,
            "dual_value": cell.dual_value,
            "mu": cell.mu,
            "lipschitz": cell.lipschitz,
            "operator_norm": cell.operator_norm,
            "converged": cell.converged,
            "elapsed_seconds": cell.elapsed_seconds,
        }
        for cell in cells
    ]

    _print_tables(rows)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
