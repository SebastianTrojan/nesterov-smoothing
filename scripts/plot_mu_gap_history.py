from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

MU_MODE_LABELS = {
    "fixed_mu": r"Fixed $\mu$",
    "continuation_mu": "Continuation",
}


def _epsilon_suffix(epsilon: float) -> str:
    text = f"{epsilon:.0e}"
    return text.replace("+", "").replace(".", "_")


def _ensure_can_write(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing files without --overwrite: {existing_text}")


def plot_mu_gap_history_for_epsilon(
    *,
    csv_path: Path,
    epsilon: float,
    seed: int | None,
    output_dir: Path,
    overwrite: bool,
) -> list[Path]:
    data = pd.read_csv(csv_path)
    filtered = data[data["epsilon"] == float(epsilon)].copy()
    if seed is not None:
        filtered = filtered[filtered["seed"] == seed].copy()
    if filtered.empty:
        raise ValueError(f"No rows found in {csv_path} for epsilon={epsilon:g}" + (f", seed={seed}" if seed is not None else ""))

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _epsilon_suffix(epsilon)
    output_paths = [output_dir / f"mu_gap_history_{problem}_eps_{suffix}.png" for problem in PROBLEMS]
    _ensure_can_write(output_paths, overwrite=overwrite)

    for problem in PROBLEMS:
        problem_rows = filtered[filtered["problem"] == problem].copy()
        if problem_rows.empty:
            continue
        problem_rows.sort_values(["mu_mode", "iteration", "stage"], inplace=True)

        plt.figure(figsize=(10, 6))
        for mu_mode, linestyle in (("fixed_mu", "-"), ("continuation_mu", "--")):
            curve = problem_rows[problem_rows["mu_mode"] == mu_mode].copy()
            if curve.empty:
                continue
            plt.plot(
                curve["iteration"],
                curve["gap"].clip(lower=1.0e-16),
                linestyle=linestyle,
                linewidth=1.8,
                color="tab:blue",
                label=MU_MODE_LABELS.get(mu_mode, mu_mode),
            )

            if mu_mode == "continuation_mu":
                stage_starts = curve.groupby("stage", sort=True)["iteration"].min().tolist()
                for x_value in stage_starts[1:]:
                    plt.axvline(x_value, color="gray", linestyle=":", linewidth=0.9, alpha=0.6)

        plt.axhline(float(epsilon), color="tab:red", linestyle=":", linewidth=1.0, alpha=0.9, label=f"epsilon={epsilon:g}")
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel("primal-dual gap")
        title = f"Gap history: {PROBLEM_LABELS.get(problem, problem)}, epsilon={epsilon:g}"
        if seed is not None:
            title += f", seed={seed}"
        plt.title(title)
        plt.grid(True, which="both", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"mu_gap_history_{problem}_eps_{suffix}.png", dpi=160)
        plt.close()

    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redraw mu-gap-history plots from an existing CSV for one epsilon.")
    parser.add_argument("--csv", type=Path, default=Path("results") / "mu_gap_history.csv", help="Existing mu_gap_history.csv file.")
    parser.add_argument("--epsilon", type=float, required=True, help="Single epsilon value to plot, e.g. 1e-4.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed to filter to.")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "plots", help="Directory for the output PNG files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing plot files.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = plot_mu_gap_history_for_epsilon(
        csv_path=args.csv,
        epsilon=args.epsilon,
        seed=args.seed,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    for path in paths:
        if path.exists():
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
