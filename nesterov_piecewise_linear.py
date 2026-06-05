from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from nesterov_core import (
    ContinuationConfig,
    ContinuationStage,
    FloatArray,
    GapTracePoint,
    MuGapPoint,
    continuation_mu_gap_history,
    default_check_frequency,
    fixed_mu_gap_history,
    matrix,
    max_affine_entropy_oracle,
    optimization_iterations_for_target,
    predicted_iterations,
    project_to_l2_ball,
    run_accelerated_method,
    vector,
)


@dataclass
class PiecewiseLinearResult:
    x: FloatArray
    u: FloatArray
    objective_value: float
    dual_value: float
    gap: float
    iterations: int
    predicted_iterations: int
    mu: float
    lipschitz: float
    operator_norm: float
    radius: float
    check_frequency: int
    converged: bool
    elapsed_seconds: float
    continuation_stages: tuple[ContinuationStage, ...] = ()
    mu_gap_history: tuple[MuGapPoint, ...] = ()


@dataclass
class PiecewiseLinearExperimentCell:
    epsilon: float
    m: int
    n: int
    radius: float
    iterations: int
    predicted_iterations: int
    percent_of_predicted: float
    objective_value: float
    dual_value: float
    gap: float
    mu: float
    lipschitz: float
    operator_norm: float
    converged: bool
    elapsed_seconds: float
    mu_mode: str
    stages: int


def _piecewise_linear_data(A: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    pieces_A = np.vstack([A, -A])
    offsets = np.concatenate([-b, b]).astype(np.float64, copy=False)
    return pieces_A, offsets


def _piecewise_linear_dual_value(pieces_A: FloatArray, offsets: FloatArray, dual: FloatArray, radius: float) -> float:
    return float(offsets @ dual - radius * np.linalg.norm(pieces_A.T @ dual))


def _rounded_stage_iterations(predicted: int, check_frequency: int) -> int:
    return int(math.ceil(predicted / check_frequency) * check_frequency)


def solve_piecewise_linear_max_abs(
    A: ArrayLike,
    b: ArrayLike,
    epsilon: float,
    *,
    radius: float = 1.0,
    check_frequency: int | None = None,
    max_iterations: int | None = None,
    continuation: ContinuationConfig | None = None,
    monotone_y: bool = False,
    trace_callback: Callable[[GapTracePoint], None] | None = None,
) -> PiecewiseLinearResult:
    matrix_value = matrix(A, name="A")
    offset_vector = vector(b, name="b")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    m, n = matrix_value.shape
    if offset_vector.size != m:
        raise ValueError("b must have one entry per row of A.")
    if m < 1 or n < 1:
        raise ValueError("A must have positive dimensions.")

    pieces_A, offsets = _piecewise_linear_data(matrix_value, offset_vector)
    piece_count = pieces_A.shape[0]

    operator_norm = float(np.max(np.linalg.norm(matrix_value, axis=1)))
    primal_diameter = 0.5 * radius * radius
    dual_diameter = math.log(piece_count)
    mu = epsilon / (2.0 * dual_diameter)
    lipschitz = (operator_norm ** 2) / mu if operator_norm > 0.0 else 0.0
    predicted = predicted_iterations(operator_norm, primal_diameter, dual_diameter, epsilon)

    if check_frequency is None:
        check_frequency = min(default_check_frequency(epsilon), predicted)
    if check_frequency <= 0:
        raise ValueError("check_frequency must be positive.")

    initial_x = np.zeros(n, dtype=np.float64)

    def emit_trace(
        *,
        iteration: int,
        stage: int,
        mu_value: float,
        objective_value: float,
        dual_weights: FloatArray | None,
    ) -> None:
        if trace_callback is None or dual_weights is None:
            return
        dual_value = _piecewise_linear_dual_value(pieces_A, offsets, dual_weights, radius)
        trace_callback(
            GapTracePoint(
                iteration=iteration,
                stage=stage,
                mu=mu_value,
                objective_value=objective_value,
                dual_value=dual_value,
                gap=objective_value - dual_value,
            )
        )

    if continuation is None:
        if max_iterations is None:
            max_iterations = _rounded_stage_iterations(predicted, check_frequency)
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive.")

        initial_state = max_affine_entropy_oracle(pieces_A, offsets, initial_x, mu)
        emit_trace(
            iteration=0,
            stage=1,
            mu_value=mu,
            objective_value=initial_state.objective_value,
            dual_weights=initial_state.auxiliary,
        )

        snapshot = run_accelerated_method(
            initial_x=initial_x,
            lipschitz=lipschitz,
            max_iterations=max_iterations,
            check_frequency=check_frequency,
            oracle=lambda x: max_affine_entropy_oracle(pieces_A, offsets, x, mu),
            local_step=lambda x, gradient, L: project_to_l2_ball(x - gradient / L, radius),
            aggregate_step=lambda gradient_sum, L: project_to_l2_ball(-gradient_sum / L, radius),
            should_stop=lambda current: (
                _piecewise_linear_dual_value(pieces_A, offsets, current.auxiliary_average, radius)
                if current.auxiliary_average is not None
                else -math.inf
            ) >= current.state.objective_value - epsilon,
            monotone_y=monotone_y,
            on_checkpoint=lambda current: emit_trace(
                iteration=current.iteration,
                stage=1,
                mu_value=mu,
                objective_value=current.state.objective_value,
                dual_weights=current.auxiliary_average,
            ),
        )

        if snapshot.auxiliary_average is None:
            raise RuntimeError("Piece-wise linear result requires averaged dual weights.")

        dual_value = _piecewise_linear_dual_value(pieces_A, offsets, snapshot.auxiliary_average, radius)
        gap = snapshot.state.objective_value - dual_value
        return PiecewiseLinearResult(
            x=snapshot.y.copy(),
            u=snapshot.auxiliary_average.copy(),
            objective_value=snapshot.state.objective_value,
            dual_value=dual_value,
            gap=gap,
            iterations=snapshot.iteration,
            predicted_iterations=predicted,
            mu=mu,
            lipschitz=lipschitz,
            operator_norm=operator_norm,
            radius=radius,
            check_frequency=check_frequency,
            converged=gap <= epsilon,
            elapsed_seconds=snapshot.elapsed_seconds,
            mu_gap_history=fixed_mu_gap_history(
                mu=mu,
                gap=gap,
                target_value=epsilon,
                iterations=snapshot.iteration,
            ),
        )

    current_x = initial_x
    current_mu = continuation.start_factor * mu
    total_iterations = 0
    total_elapsed = 0.0
    stages: list[ContinuationStage] = []
    latest_snapshot = None
    latest_dual_value = -math.inf
    latest_gap = math.inf
    latest_lipschitz = lipschitz

    stage_index = 0
    while True:
        stage_index += 1
        current_mu = max(mu, current_mu)
        final_stage = current_mu <= mu * (1.0 + 1.0e-15)
        latest_lipschitz = (operator_norm ** 2) / current_mu if operator_norm > 0.0 else 0.0
        stage_target = epsilon if final_stage else continuation.stage_factor * current_mu * dual_diameter
        stage_offset = total_iterations

        if trace_callback is not None and stage_offset == 0:
            initial_state = max_affine_entropy_oracle(pieces_A, offsets, current_x, current_mu)
            emit_trace(
                iteration=0,
                stage=stage_index,
                mu_value=current_mu,
                objective_value=initial_state.objective_value,
                dual_weights=initial_state.auxiliary,
            )

        if max_iterations is None:
            stage_predicted = optimization_iterations_for_target(
                lipschitz=latest_lipschitz,
                primal_diameter=primal_diameter,
                target=stage_target,
            )
            stage_iterations = _rounded_stage_iterations(stage_predicted, check_frequency)
        else:
            remaining_iterations = max_iterations - total_iterations
            if remaining_iterations < 1:
                break
            stage_iterations = remaining_iterations

        snapshot = run_accelerated_method(
            initial_x=current_x,
            lipschitz=latest_lipschitz,
            max_iterations=stage_iterations,
            check_frequency=check_frequency,
            oracle=lambda x, current_mu=current_mu: max_affine_entropy_oracle(pieces_A, offsets, x, current_mu),
            local_step=lambda x, gradient, L: project_to_l2_ball(x - gradient / L, radius),
            aggregate_step=lambda gradient_sum, L: project_to_l2_ball(-gradient_sum / L, radius),
            should_stop=lambda current, stage_target=stage_target: (
                _piecewise_linear_dual_value(pieces_A, offsets, current.auxiliary_average, radius)
                if current.auxiliary_average is not None
                else -math.inf
            ) >= current.state.objective_value - stage_target,
            monotone_y=monotone_y,
            on_checkpoint=lambda current, stage_offset=stage_offset, stage_index=stage_index, current_mu=current_mu: emit_trace(
                iteration=stage_offset + current.iteration,
                stage=stage_index,
                mu_value=current_mu,
                objective_value=current.state.objective_value,
                dual_weights=current.auxiliary_average,
            ),
        )
        if snapshot.auxiliary_average is None:
            raise RuntimeError("Piece-wise linear result requires averaged dual weights.")

        latest_snapshot = snapshot
        current_x = snapshot.y.copy()
        total_iterations += snapshot.iteration
        total_elapsed += snapshot.elapsed_seconds
        latest_dual_value = _piecewise_linear_dual_value(pieces_A, offsets, snapshot.auxiliary_average, radius)
        latest_gap = snapshot.state.objective_value - latest_dual_value
        target_met = latest_gap <= stage_target
        stages.append(
            ContinuationStage(
                index=stage_index,
                mu=current_mu,
                target_value=stage_target,
                achieved_value=latest_gap,
                iterations=snapshot.iteration,
                cumulative_iterations=total_iterations,
                target_met=target_met,
                final_stage=final_stage,
            )
        )

        if latest_gap <= epsilon:
            break
        if final_stage:
            break
        if continuation.max_stages is not None and stage_index >= continuation.max_stages:
            break
        if max_iterations is not None and total_iterations >= max_iterations:
            break
        if not target_met:
            break

        current_mu = continuation.decay * current_mu

    if latest_snapshot is None or latest_snapshot.auxiliary_average is None:
        raise RuntimeError("Continuation did not produce a piece-wise linear snapshot.")

    return PiecewiseLinearResult(
        x=latest_snapshot.y.copy(),
        u=latest_snapshot.auxiliary_average.copy(),
        objective_value=latest_snapshot.state.objective_value,
        dual_value=latest_dual_value,
        gap=latest_gap,
        iterations=total_iterations,
        predicted_iterations=predicted,
        mu=mu,
        lipschitz=latest_lipschitz,
        operator_norm=operator_norm,
        radius=radius,
        check_frequency=check_frequency,
        converged=latest_gap <= epsilon,
        elapsed_seconds=total_elapsed,
        continuation_stages=tuple(stages),
        mu_gap_history=continuation_mu_gap_history(stages),
    )


def piecewise_linear_default_grid() -> dict[float, dict[str, tuple[int, ...] | float]]:
    return {
        1.0e-2: {"m_values": (100, 300), "n_values": (50, 200), "radius": 1.0},
        1.0e-3: {"m_values": (100, 300), "n_values": (50, 200), "radius": 1.0},
        1.0e-4: {"m_values": (100, 300), "n_values": (50, 200), "radius": 1.0},
    }


def run_piecewise_linear_grid(
    *,
    base_seed: int = 0,
    epsilons: tuple[float, ...] | None = None,
    m_values: tuple[int, ...] | None = None,
    n_values: tuple[int, ...] | None = None,
    radius: float | None = None,
    check_frequency: int | None = None,
    max_iterations: int | None = None,
    continuation: ContinuationConfig | None = None,
    monotone_y: bool = False,
) -> list[PiecewiseLinearExperimentCell]:
    default_grid = piecewise_linear_default_grid()
    active_epsilons = tuple(default_grid) if epsilons is None else epsilons
    cells: list[PiecewiseLinearExperimentCell] = []

    for epsilon in active_epsilons:
        epsilon_m_values = default_grid[epsilon]["m_values"] if m_values is None else m_values
        epsilon_n_values = default_grid[epsilon]["n_values"] if n_values is None else n_values
        epsilon_radius = float(default_grid[epsilon]["radius"]) if radius is None else radius

        for m in epsilon_m_values:
            for n in epsilon_n_values:
                seed = hash((base_seed, epsilon, m, n, epsilon_radius)) & 0xFFFFFFFF
                rng = np.random.default_rng(seed)
                matrix_value = rng.uniform(-1.0, 1.0, size=(m, n))
                offsets = rng.uniform(-1.0, 1.0, size=m)
                result = solve_piecewise_linear_max_abs(
                    matrix_value,
                    offsets,
                    epsilon,
                    radius=epsilon_radius,
                    check_frequency=check_frequency,
                    max_iterations=max_iterations,
                    continuation=continuation,
                    monotone_y=monotone_y,
                )
                cells.append(
                    PiecewiseLinearExperimentCell(
                        epsilon=epsilon,
                        m=m,
                        n=n,
                        radius=epsilon_radius,
                        iterations=result.iterations,
                        predicted_iterations=result.predicted_iterations,
                        percent_of_predicted=100.0 * result.iterations / result.predicted_iterations,
                        objective_value=result.objective_value,
                        dual_value=result.dual_value,
                        gap=result.gap,
                        mu=result.mu,
                        lipschitz=result.lipschitz,
                        operator_norm=result.operator_norm,
                        converged=result.converged,
                        elapsed_seconds=result.elapsed_seconds,
                        mu_mode="continuation" if continuation is not None else "fixed",
                        stages=max(1, len(result.continuation_stages)),
                    )
                )

    return cells
