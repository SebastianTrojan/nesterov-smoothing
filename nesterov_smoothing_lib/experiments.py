from __future__ import annotations

import numpy as np

from .algorithms import solve_paper_matrix_game
from .results import PaperExperimentCell


def paper_section6_grid() -> dict[float, dict[str, tuple[int, ...]]]:
    return {
        1.0e-2: {"m_values": (100, 300, 1000), "n_values": (100, 300, 1000, 3000, 10000)},
        1.0e-3: {"m_values": (100, 300, 1000), "n_values": (100, 300, 1000, 3000, 10000)},
        1.0e-4: {"m_values": (100, 300, 1000), "n_values": (100, 300, 1000, 3000)},
    }


def run_paper_matrix_game_grid(
    *,
    base_seed: int = 0,
    epsilons: tuple[float, ...] | None = None,
    m_values: tuple[int, ...] | None = None,
    n_values: tuple[int, ...] | None = None,
    check_frequency: int | None = None,
    max_iterations: int | None = None,
) -> list[PaperExperimentCell]:
    section6_grid = paper_section6_grid()
    active_epsilons = tuple(section6_grid) if epsilons is None else epsilons
    cells: list[PaperExperimentCell] = []

    for epsilon in active_epsilons:
        epsilon_m_values = section6_grid[epsilon]["m_values"] if m_values is None else m_values
        epsilon_n_values = section6_grid[epsilon]["n_values"] if n_values is None else n_values

        for m in epsilon_m_values:
            for n in epsilon_n_values:
                seed = hash((base_seed, epsilon, m, n)) & 0xFFFFFFFF
                rng = np.random.default_rng(seed)
                matrix_value = rng.uniform(-1.0, 1.0, size=(m, n))
                result = solve_paper_matrix_game(
                    matrix_value,
                    epsilon,
                    check_frequency=check_frequency,
                    max_iterations=max_iterations,
                )
                cells.append(
                    PaperExperimentCell(
                        epsilon=epsilon,
                        m=m,
                        n=n,
                        iterations=result.iterations,
                        predicted_iterations=result.predicted_iterations,
                        percent_of_predicted=100.0 * result.iterations / result.predicted_iterations,
                        primal_value=result.primal_value,
                        dual_value=result.dual_value,
                        gap=result.gap,
                        mu=result.mu,
                        lipschitz=result.lipschitz,
                        operator_norm=result.operator_norm,
                        converged=result.converged,
                        elapsed_seconds=result.elapsed_seconds,
                    )
                )

    return cells
