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
    OracleState,
    continuation_mu_gap_history,
    default_check_frequency,
    fixed_mu_gap_history,
    matrix,
    optimization_iterations_for_target,
    predicted_iterations,
    project_to_l2_ball,
    run_accelerated_method,
    theoretical_total_bound,
    vector,
)


@dataclass
class ContinuousLocationResult:
    x: FloatArray
    objective_value: float
    dual_value: float
    gap: float
    smoothed_value: float
    theoretical_gap_bound: float
    iterations: int
    predicted_iterations: int
    mu: float
    lipschitz: float
    operator_norm: float
    weight_sum: float
    radius: float
    check_frequency: int
    converged: bool
    elapsed_seconds: float
    continuation_stages: tuple[ContinuationStage, ...] = ()
    mu_gap_history: tuple[MuGapPoint, ...] = ()


@dataclass
class ContinuousLocationExperimentCell:
    epsilon: float
    num_cities: int
    dimension: int
    radius: float
    iterations: int
    predicted_iterations: int
    percent_of_predicted: float
    objective_value: float
    dual_value: float
    gap: float
    theoretical_gap_bound: float
    mu: float
    lipschitz: float
    operator_norm: float
    weight_sum: float
    converged: bool
    elapsed_seconds: float
    mu_mode: str
    stages: int


def _continuous_location_oracle(cities: FloatArray, weights: FloatArray, x: FloatArray, mu: float) -> OracleState:
    displacements = x[None, :] - cities
    distances = np.linalg.norm(displacements, axis=1)
    objective_value = float(np.dot(weights, distances))
    smoothed_value = 0.0
    gradient = np.zeros_like(x)
    dual_points = np.zeros_like(displacements)

    for index, distance in enumerate(distances):
        weight = float(weights[index])
        direction = displacements[index]
        if distance <= mu:
            smoothed_value += weight * distance * distance / (2.0 * mu)
            if distance > 0.0:
                dual_points[index] = direction / mu
                gradient += weight * dual_points[index]
        else:
            smoothed_value += weight * (distance - 0.5 * mu)
            dual_points[index] = direction / distance
            gradient += weight * dual_points[index]

    return OracleState(
        smoothed_value=smoothed_value,
        gradient=gradient,
        objective_value=objective_value,
        auxiliary=dual_points.reshape(-1),
    )


def _continuous_location_dual_value(
    cities: FloatArray,
    weights: FloatArray,
    dual_points_flat: FloatArray,
    radius: float,
) -> float:
    dual_points = dual_points_flat.reshape(cities.shape)
    weighted_sum = weights[:, None] * dual_points
    return float(-np.sum(weighted_sum * cities) - radius * np.linalg.norm(np.sum(weighted_sum, axis=0)))


def solve_continuous_location(
    cities: ArrayLike,
    epsilon: float,
    *,
    weights: ArrayLike | None = None,
    radius: float = 1.0,
    check_frequency: int | None = None,
    max_iterations: int | None = None,
    continuation: ContinuationConfig | None = None,
    monotone_y: bool = False,
    trace_callback: Callable[[GapTracePoint], None] | None = None,
) -> ContinuousLocationResult:
    city_matrix = matrix(cities, name="cities")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    num_cities, dimension = city_matrix.shape
    if num_cities < 1 or dimension < 1:
        raise ValueError("cities must contain at least one point.")

    if weights is None:
        weight_vector = np.ones(num_cities, dtype=np.float64)
    else:
        weight_vector = vector(weights, name="weights")
        if weight_vector.size != num_cities:
            raise ValueError("weights must have one entry per city.")
    if np.any(weight_vector <= 0.0):
        raise ValueError("weights must be positive.")

    weight_sum = float(np.sum(weight_vector))
    primal_diameter = 0.5 * radius * radius
    dual_diameter = 0.5 * weight_sum
    operator_norm = math.sqrt(weight_sum)
    mu = epsilon / weight_sum
    lipschitz = weight_sum / mu
    predicted = predicted_iterations(operator_norm, primal_diameter, dual_diameter, epsilon)

    if check_frequency is None:
        check_frequency = min(default_check_frequency(epsilon), predicted)
    if check_frequency <= 0:
        raise ValueError("check_frequency must be positive.")

    initial_x = np.zeros(dimension, dtype=np.float64)

    def emit_trace(
        *,
        iteration: int,
        stage: int,
        mu_value: float,
        objective_value: float,
        dual_points_flat: FloatArray | None,
    ) -> None:
        if trace_callback is None or dual_points_flat is None:
            return
        dual_value = _continuous_location_dual_value(city_matrix, weight_vector, dual_points_flat, radius)
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
            max_iterations = int(math.ceil(predicted / check_frequency) * check_frequency)
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive.")

        initial_state = _continuous_location_oracle(city_matrix, weight_vector, initial_x, mu)
        emit_trace(
            iteration=0,
            stage=1,
            mu_value=mu,
            objective_value=initial_state.objective_value,
            dual_points_flat=initial_state.auxiliary,
        )

        snapshot = run_accelerated_method(
            initial_x=initial_x,
            lipschitz=lipschitz,
            max_iterations=max_iterations,
            check_frequency=check_frequency,
            oracle=lambda x: _continuous_location_oracle(city_matrix, weight_vector, x, mu),
            local_step=lambda x, gradient, L: project_to_l2_ball(x - gradient / L, radius),
            aggregate_step=lambda gradient_sum, L: project_to_l2_ball(-gradient_sum / L, radius),
            should_stop=lambda current: (
                _continuous_location_dual_value(city_matrix, weight_vector, current.auxiliary_average, radius)
                if current.auxiliary_average is not None
                else -math.inf
            ) >= current.state.objective_value - epsilon,
            monotone_y=monotone_y,
            on_checkpoint=lambda current: emit_trace(
                iteration=current.iteration,
                stage=1,
                mu_value=mu,
                objective_value=current.state.objective_value,
                dual_points_flat=current.auxiliary_average,
            ),
        )
        if snapshot.auxiliary_average is None:
            raise RuntimeError("Continuous-location result requires averaged dual points.")

        dual_value = _continuous_location_dual_value(city_matrix, weight_vector, snapshot.auxiliary_average, radius)
        gap = snapshot.state.objective_value - dual_value
        bound = theoretical_total_bound(
            lipschitz=lipschitz,
            primal_diameter=primal_diameter,
            mu=mu,
            dual_diameter=dual_diameter,
            iterations=snapshot.iteration,
        )
        return ContinuousLocationResult(
            x=snapshot.y.copy(),
            objective_value=snapshot.state.objective_value,
            dual_value=dual_value,
            gap=gap,
            smoothed_value=snapshot.state.smoothed_value,
            theoretical_gap_bound=bound,
            iterations=snapshot.iteration,
            predicted_iterations=predicted,
            mu=mu,
            lipschitz=lipschitz,
            operator_norm=operator_norm,
            weight_sum=weight_sum,
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
    latest_bound = math.inf
    latest_dual_value = -math.inf
    latest_gap = math.inf
    latest_lipschitz = lipschitz

    stage_index = 0
    while True:
        stage_index += 1
        current_mu = max(mu, current_mu)
        final_stage = current_mu <= mu * (1.0 + 1.0e-15)
        latest_lipschitz = weight_sum / current_mu
        stage_target = epsilon if final_stage else continuation.stage_factor * current_mu * dual_diameter
        stage_offset = total_iterations

        if trace_callback is not None and stage_offset == 0:
            initial_state = _continuous_location_oracle(city_matrix, weight_vector, current_x, current_mu)
            emit_trace(
                iteration=0,
                stage=stage_index,
                mu_value=current_mu,
                objective_value=initial_state.objective_value,
                dual_points_flat=initial_state.auxiliary,
            )

        if max_iterations is None:
            stage_predicted = optimization_iterations_for_target(
                lipschitz=latest_lipschitz,
                primal_diameter=primal_diameter,
                target=stage_target,
            )
            stage_iterations = int(math.ceil(stage_predicted / check_frequency) * check_frequency)
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
            oracle=lambda x, current_mu=current_mu: _continuous_location_oracle(city_matrix, weight_vector, x, current_mu),
            local_step=lambda x, gradient, L: project_to_l2_ball(x - gradient / L, radius),
            aggregate_step=lambda gradient_sum, L: project_to_l2_ball(-gradient_sum / L, radius),
            should_stop=lambda current, stage_target=stage_target: (
                _continuous_location_dual_value(city_matrix, weight_vector, current.auxiliary_average, radius)
                if current.auxiliary_average is not None
                else -math.inf
            ) >= current.state.objective_value - stage_target,
            monotone_y=monotone_y,
            on_checkpoint=lambda current, stage_offset=stage_offset, stage_index=stage_index, current_mu=current_mu: emit_trace(
                iteration=stage_offset + current.iteration,
                stage=stage_index,
                mu_value=current_mu,
                objective_value=current.state.objective_value,
                dual_points_flat=current.auxiliary_average,
            ),
        )
        if snapshot.auxiliary_average is None:
            raise RuntimeError("Continuous-location result requires averaged dual points.")

        latest_snapshot = snapshot
        current_x = snapshot.y.copy()
        total_iterations += snapshot.iteration
        total_elapsed += snapshot.elapsed_seconds
        latest_dual_value = _continuous_location_dual_value(city_matrix, weight_vector, snapshot.auxiliary_average, radius)
        latest_gap = snapshot.state.objective_value - latest_dual_value
        latest_bound = theoretical_total_bound(
            lipschitz=latest_lipschitz,
            primal_diameter=primal_diameter,
            mu=current_mu,
            dual_diameter=dual_diameter,
            iterations=snapshot.iteration,
        )
        achieved_value = latest_gap
        target_met = achieved_value <= stage_target
        stages.append(
            ContinuationStage(
                index=stage_index,
                mu=current_mu,
                target_value=stage_target,
                achieved_value=achieved_value,
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

    if latest_snapshot is None:
        raise RuntimeError("Continuation did not produce a continuous-location snapshot.")

    return ContinuousLocationResult(
        x=latest_snapshot.y.copy(),
        objective_value=latest_snapshot.state.objective_value,
        dual_value=latest_dual_value,
        gap=latest_gap,
        smoothed_value=latest_snapshot.state.smoothed_value,
        theoretical_gap_bound=latest_bound,
        iterations=total_iterations,
        predicted_iterations=predicted,
        mu=mu,
        lipschitz=latest_lipschitz,
        operator_norm=operator_norm,
        weight_sum=weight_sum,
        radius=radius,
        check_frequency=check_frequency,
        converged=latest_gap <= epsilon,
        elapsed_seconds=total_elapsed,
        continuation_stages=tuple(stages),
        mu_gap_history=continuation_mu_gap_history(stages),
    )


def continuous_location_default_grid() -> dict[float, dict[str, tuple[int, ...] | float]]:
    return {
        1.0e-2: {"num_cities_values": (10, 50), "dimension_values": (2, 5), "radius": 1.0},
        1.0e-3: {"num_cities_values": (10, 50), "dimension_values": (2, 5), "radius": 1.0},
        1.0e-4: {"num_cities_values": (10, 50), "dimension_values": (2, 5), "radius": 1.0},
    }


def _sample_points_in_l2_ball(rng: np.random.Generator, count: int, dimension: int, radius: float) -> FloatArray:
    directions = rng.normal(size=(count, dimension))
    norms = np.linalg.norm(directions, axis=1)
    zero_mask = norms == 0.0
    if np.any(zero_mask):
        directions[zero_mask, 0] = 1.0
        norms[zero_mask] = 1.0
    directions = directions / norms[:, None]
    radii = radius * np.power(rng.random(count), 1.0 / dimension)
    return directions * radii[:, None]


def run_continuous_location_grid(
    *,
    base_seed: int = 0,
    epsilons: tuple[float, ...] | None = None,
    num_cities_values: tuple[int, ...] | None = None,
    dimension_values: tuple[int, ...] | None = None,
    radius: float | None = None,
    city_radius: float | None = None,
    check_frequency: int | None = None,
    max_iterations: int | None = None,
    continuation: ContinuationConfig | None = None,
    monotone_y: bool = False,
) -> list[ContinuousLocationExperimentCell]:
    default_grid = continuous_location_default_grid()
    active_epsilons = tuple(default_grid) if epsilons is None else epsilons
    cells: list[ContinuousLocationExperimentCell] = []

    for epsilon in active_epsilons:
        epsilon_num_cities = default_grid[epsilon]["num_cities_values"] if num_cities_values is None else num_cities_values
        epsilon_dimensions = default_grid[epsilon]["dimension_values"] if dimension_values is None else dimension_values
        epsilon_radius = float(default_grid[epsilon]["radius"]) if radius is None else radius
        epsilon_city_radius = 0.8 * epsilon_radius if city_radius is None else city_radius

        for num_cities in epsilon_num_cities:
            for dimension in epsilon_dimensions:
                seed = hash((base_seed, epsilon, num_cities, dimension, epsilon_radius)) & 0xFFFFFFFF
                rng = np.random.default_rng(seed)
                cities = _sample_points_in_l2_ball(rng, num_cities, dimension, epsilon_city_radius)
                weights = rng.uniform(0.5, 1.5, size=num_cities)
                result = solve_continuous_location(
                    cities,
                    epsilon,
                    weights=weights,
                    radius=epsilon_radius,
                    check_frequency=check_frequency,
                    max_iterations=max_iterations,
                    continuation=continuation,
                    monotone_y=monotone_y,
                )
                cells.append(
                    ContinuousLocationExperimentCell(
                        epsilon=epsilon,
                        num_cities=num_cities,
                        dimension=dimension,
                        radius=epsilon_radius,
                        iterations=result.iterations,
                        predicted_iterations=result.predicted_iterations,
                        percent_of_predicted=100.0 * result.iterations / result.predicted_iterations,
                        objective_value=result.objective_value,
                        dual_value=result.dual_value,
                        gap=result.gap,
                        theoretical_gap_bound=result.theoretical_gap_bound,
                        mu=result.mu,
                        lipschitz=result.lipschitz,
                        operator_norm=result.operator_norm,
                        weight_sum=result.weight_sum,
                        converged=result.converged,
                        elapsed_seconds=result.elapsed_seconds,
                        mu_mode="continuation" if continuation is not None else "fixed",
                        stages=max(1, len(result.continuation_stages)),
                    )
                )

    return cells
