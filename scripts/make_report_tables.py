from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fmt_int(value: float) -> str:
    return f"{int(round(float(value))):,}".replace(",", r"\,")


def fmt_float(value: float, digits: int = 4) -> str:
    value = float(value)
    if abs(value) >= 1000:
        return fmt_int(value)
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def fmt_percent(value: float) -> str:
    return f"{float(value):.2f}\\%"


def fmt_percent_short(value: float) -> str:
    return f"{float(value):.0f}\\%"


def fmt_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 10:
        return f"{seconds:.2f}s"
    if seconds < 100:
        return f"{seconds:.1f}s"
    return f"{seconds:.0f}s"


def epsilon_name(epsilon: float) -> str:
    epsilon = float(epsilon)
    if epsilon == 1e-1:
        return r"10^{-1}"
    if epsilon == 5e-2:
        return r"5\cdot 10^{-2}"
    if epsilon == 1e-2:
        return r"10^{-2}"
    if epsilon == 5e-3:
        return r"5\cdot 10^{-3}"
    if epsilon == 1e-3:
        return r"10^{-3}"
    if epsilon == 5e-4:
        return r"5\cdot 10^{-4}"
    if epsilon == 1e-4:
        return r"10^{-4}"
    return f"{epsilon:g}"


def epsilon_filename(epsilon: float) -> str:
    return f"{float(epsilon):g}".replace(".", "_").replace("-", "m")


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def problem_display_name(problem: str) -> str:
    mapping = {
        "matrix_game": "Matrix game",
        "continuous_location": "Continuous location",
        "piecewise_linear_max_abs": "Maximum of absolute values",
        "sum_absolute": "Sum of absolute values",
    }
    return mapping.get(str(problem), str(problem))


def mu_mode_display_name(mu_mode: str) -> str:
    mapping = {
        "continuation_mu": "Continuation",
        "fixed_mu": r"Fixed $\mu$",
    }
    return mapping.get(str(mu_mode), str(mu_mode))


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path}")


def make_booktabs_table(
    headers: list[str],
    rows: list[list[str]],
    align: str | None = None,
) -> str:
    if align is None:
        align = "l" + "r" * (len(headers) - 1)

    lines = []
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def summarize_by_epsilon(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("epsilon", sort=True)
        .agg(
            instances=("iterations", "size"),
            avg_iterations=("iterations", "mean"),
            max_iterations=("iterations", "max"),
            avg_percent_predicted=("percent_of_predicted", "mean"),
            max_gap=("gap", "max"),
            avg_time=("elapsed_seconds", "mean"),
        )
        .reset_index()
    )


def make_summary_table(df: pd.DataFrame) -> str:
    summary = summarize_by_epsilon(df)

    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                f"${epsilon_name(row['epsilon'])}$",
                fmt_int(row["instances"]),
                fmt_int(row["avg_iterations"]),
                fmt_int(row["max_iterations"]),
                fmt_percent(row["avg_percent_predicted"]),
                fmt_float(row["max_gap"], 6),
                fmt_time(row["avg_time"]),
            ]
        )

    return make_booktabs_table(
        headers=[
            r"$\varepsilon$",
            "Instances",
            "Avg. iter.",
            "Max iter.",
            "Avg. \\% pred.",
            "Max gap",
            "Avg. time",
        ],
        rows=rows,
        align="crrrrrr",
    )


def make_pivot_tables(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    prefix: str,
    row_col: str,
    col_col: str,
    row_label: str,
) -> None:
    for epsilon, sub in df.groupby("epsilon", sort=True):
        row_values = sorted(sub[row_col].unique())
        col_values = sorted(sub[col_col].unique())

        cell_map: dict[tuple[int, int], str] = {}

        for _, row in sub.iterrows():
            row_value = int(row[row_col])
            col_value = int(row[col_col])

            # Detailed tables show only iterations and percentage of predicted iterations.
            # Runtime is intentionally omitted here.
            cell = (
                f"{fmt_int(row['iterations'])} / "
                f"{fmt_percent_short(row['percent_of_predicted'])}"
            )

            cell_map[(row_value, col_value)] = cell

        headers = [row_label] + [str(int(c)) for c in col_values]

        rows = []
        for r in row_values:
            rows.append(
                [str(int(r))]
                + [cell_map.get((int(r), int(c)), "--") for c in col_values]
            )

        table = make_booktabs_table(
            headers=headers,
            rows=rows,
            align="l" + "c" * len(col_values),
        )

        out_path = out_dir / f"{prefix}_eps_{epsilon_filename(float(epsilon))}.tex"
        write_text(out_path, table)


def make_standard_summary_tables(results_dir: Path, out_dir: Path) -> None:
    mapping = {
        "reproduction.csv": "reproduction_summary.tex",
        "continuous_location.csv": "continuous_location_summary.tex",
        "piecewise_linear.csv": "piecewise_linear_summary.tex",
        "sum_absolute.csv": "sum_absolute_summary.tex",
    }

    for csv_name, tex_name in mapping.items():
        csv_path = results_dir / csv_name
        if not csv_path.exists():
            print(f"skipping missing {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        write_text(out_dir / tex_name, make_summary_table(df))


def make_all_pivot_tables(results_dir: Path, out_dir: Path) -> None:
    reproduction_path = results_dir / "reproduction.csv"
    if reproduction_path.exists():
        make_pivot_tables(
            pd.read_csv(reproduction_path),
            out_dir,
            prefix="reproduction",
            row_col="m",
            col_col="n",
            row_label=r"$m\backslash n$",
        )

    continuous_path = results_dir / "continuous_location.csv"
    if continuous_path.exists():
        make_pivot_tables(
            pd.read_csv(continuous_path),
            out_dir,
            prefix="continuous_location",
            row_col="num_cities",
            col_col="dimension",
            row_label=r"$p\backslash d$",
        )

    piecewise_path = results_dir / "piecewise_linear.csv"
    if piecewise_path.exists():
        make_pivot_tables(
            pd.read_csv(piecewise_path),
            out_dir,
            prefix="piecewise_linear",
            row_col="m",
            col_col="n",
            row_label=r"$m\backslash n$",
        )

    sum_path = results_dir / "sum_absolute.csv"
    if sum_path.exists():
        make_pivot_tables(
            pd.read_csv(sum_path),
            out_dir,
            prefix="sum_absolute",
            row_col="m",
            col_col="n",
            row_label=r"$m\backslash n$",
        )


def make_overall_summary(results_dir: Path, out_dir: Path) -> None:
    files = [
        ("Matrix game", results_dir / "reproduction.csv"),
        ("Continuous location", results_dir / "continuous_location.csv"),
        ("Maximum of absolute values", results_dir / "piecewise_linear.csv"),
        ("Sum of absolute values", results_dir / "sum_absolute.csv"),
    ]

    rows = []
    for problem_name, csv_path in files:
        if not csv_path.exists():
            print(f"skipping missing {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        rows.append(
            [
                latex_escape(problem_name),
                fmt_int(len(df)),
                fmt_int(df["iterations"].mean()),
                fmt_percent(df["percent_of_predicted"].mean()),
                fmt_float(df["gap"].max(), 6),
                fmt_time(df["elapsed_seconds"].mean()),
            ]
        )

    table = make_booktabs_table(
        headers=[
            "Problem",
            "Instances",
            "Avg. iter.",
            "Avg. \\% pred.",
            "Max gap",
            "Avg. time",
        ],
        rows=rows,
        align="lrrrrr",
    )

    write_text(out_dir / "overall_summary.tex", table)


def make_epsilon_scaling_tables(results_dir: Path, out_dir: Path) -> None:
    summary_path = results_dir / "epsilon_scaling_summary.csv"
    regression_path = results_dir / "epsilon_scaling_regression.csv"

    if summary_path.exists():
        df = pd.read_csv(summary_path)

        runtime_col = None
        for candidate in ["mean_runtime", "mean_elapsed_seconds", "mean_time"]:
            if candidate in df.columns:
                runtime_col = candidate
                break

        rows = []
        for _, row in df.iterrows():
            rows.append(
                [
                    latex_escape(problem_display_name(row["problem"])),
                    f"${epsilon_name(row['epsilon'])}$",
                    fmt_int(row["inverse_epsilon"]),
                    fmt_int(row["mean_iterations"]),
                    fmt_float(row.get("std_iterations", 0.0), 2),
                    fmt_float(row.get("mean_gap", 0.0), 6),
                    fmt_float(row.get("max_gap", 0.0), 6),
                ]
            )

        table = make_booktabs_table(
            headers=[
                "Problem",
                r"$\varepsilon$",
                r"$1/\varepsilon$",
                "Mean iter.",
                "Std iter.",
                "Mean gap",
                "Max gap",
            ],
            rows=rows,
            align="lcrrrrr",
        )

        write_text(out_dir / "epsilon_scaling_summary.tex", table)

    if regression_path.exists():
        df = pd.read_csv(regression_path)

        rows = []
        for _, row in df.iterrows():
            se = row.get("standard_error", row.get("std_error", 0.0))
            t_statistic = (row["p_hat"] - 1.0) / se
            rows.append(
                [
                    latex_escape(problem_display_name(row["problem"])),
                    fmt_float(row["p_hat"], 4),
                    fmt_float(se, 4)
                ]
            )

        table = make_booktabs_table(
            headers=[
                "Problem",
                r"$\hat p$",
                "SE"
            ],
            rows=rows,
            align="lrrrr",
        )

        write_text(out_dir / "epsilon_scaling_regression.tex", table)


def make_mu_comparison_tables(results_dir: Path, out_dir: Path) -> None:
    summary_path = results_dir / "mu_comparison_summary.csv"
    speedup_path = results_dir / "mu_comparison_speedup.csv"

    if summary_path.exists():
        df = pd.read_csv(summary_path)

        rows = []
        for _, row in df.iterrows():
            rows.append(
                [
                    latex_escape(problem_display_name(row["problem"])),
                    f"${epsilon_name(row['epsilon'])}$",
                    mu_mode_display_name(row["mu_mode"]),
                    fmt_int(row["mean_iterations"]),
                    fmt_float(row.get("std_iterations", 0.0), 2),
                ]
            )

        table = make_booktabs_table(
            headers=[
                "Problem",
                r"$\varepsilon$",
                "$\\mu$ mode",
                "Mean iter.",
                "Std iter.",
            ],
            rows=rows,
            align="lclrr",
        )

        write_text(out_dir / "mu_comparison_summary.tex", table)

    if speedup_path.exists():
        df = pd.read_csv(speedup_path)

        agg = (
            df.groupby(["problem", "epsilon"], sort=True)
            .agg(
                mean_iteration_speedup=("iteration_speedup", "mean"),
                min_iteration_speedup=("iteration_speedup", "min"),
                max_iteration_speedup=("iteration_speedup", "max"),
            )
            .reset_index()
        )

        rows = []
        for _, row in agg.iterrows():
            rows.append(
                [
                    latex_escape(problem_display_name(row["problem"])),
                    f"${epsilon_name(row['epsilon'])}$",
                    fmt_float(row["mean_iteration_speedup"], 3),
                    fmt_float(row["min_iteration_speedup"], 3),
                    fmt_float(row["max_iteration_speedup"], 3),
                ]
            )

        table = make_booktabs_table(
            headers=[
                "Problem",
                r"$\varepsilon$",
                "Mean iter. speedup",
                "Min iter. speedup",
                "Max iter. speedup",
            ],
            rows=rows,
            align="lcrrr",
        )

        write_text(out_dir / "mu_comparison_speedup.tex", table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_standard_summary_tables(results_dir, out_dir)
    make_all_pivot_tables(results_dir, out_dir)
    make_overall_summary(results_dir, out_dir)
    make_epsilon_scaling_tables(results_dir, out_dir)
    make_mu_comparison_tables(results_dir, out_dir)


if __name__ == "__main__":
    main()

