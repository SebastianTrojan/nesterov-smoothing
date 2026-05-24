from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass
class OracleState:
    smoothed_value: float
    gradient: FloatArray
    objective_value: float
    auxiliary: FloatArray | None = None


@dataclass
class AcceleratedSnapshot:
    iteration: int
    y: FloatArray
    state: OracleState
    auxiliary_average: FloatArray | None
    elapsed_seconds: float


@dataclass
class ContinuationConfig:
    start_factor: float = 8.0
    decay: float = 0.5
    stage_factor: float = 1.0
    max_stages: int | None = None

    def __post_init__(self) -> None:
        if self.start_factor < 1.0:
            raise ValueError("start_factor must be at least 1.")
        if not (0.0 < self.decay < 1.0):
            raise ValueError("decay must belong to (0, 1).")
        if self.stage_factor <= 0.0:
            raise ValueError("stage_factor must be positive.")
        if self.max_stages is not None and self.max_stages < 1:
            raise ValueError("max_stages must be positive when provided.")


@dataclass
class ContinuationStage:
    index: int
    mu: float
    target_value: float
    achieved_value: float
    iterations: int
    cumulative_iterations: int
    target_met: bool
    final_stage: bool


def vector(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return array.astype(np.float64, copy=True)


def matrix(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return array.astype(np.float64, copy=True)


def project_to_simplex(x: ArrayLike) -> FloatArray:
    candidate = vector(x, name="x")
    if candidate.size == 0:
        raise ValueError("Simplex projection needs at least one coordinate.")
    sorted_values = np.sort(candidate)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, candidate.size + 1)
    support = np.nonzero(sorted_values - cumulative / indices > 0.0)[0]
    if support.size == 0:
        return np.full(candidate.size, 1.0 / candidate.size, dtype=np.float64)
    rho = int(support[-1])
    theta = cumulative[rho] / float(rho + 1)
    return np.maximum(candidate - theta, 0.0)


def project_to_l2_ball(x: ArrayLike, radius: float) -> FloatArray:
    candidate = vector(x, name="x")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    norm = float(np.linalg.norm(candidate))
    if norm <= radius:
        return candidate
    return (radius / norm) * candidate


def simplex_entropy_argmin(linear_term: ArrayLike, scale: float) -> FloatArray:
    gradient = vector(linear_term, name="linear_term")
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    shifted = -gradient / scale
    shifted -= float(np.max(shifted))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def simplex_l1_squared_step(x_bar: ArrayLike, gradient: ArrayLike, L: float) -> FloatArray:
    x_center = vector(x_bar, name="x_bar")
    grad = vector(gradient, name="gradient")
    if x_center.size != grad.size:
        raise ValueError("x_bar and gradient must have the same dimension.")
    if L <= 0.0:
        raise ValueError("L must be positive.")
    if not np.isclose(float(np.sum(x_center)), 1.0, atol=1.0e-10):
        raise ValueError("x_bar must belong to the simplex.")
    if np.any(x_center < -1.0e-12):
        raise ValueError("x_bar must be non-negative.")

    shifted_grad = grad - float(np.min(grad))
    max_grad = float(np.max(shifted_grad))
    if max_grad <= 0.0:
        return x_center.copy()

    tolerance = 1.0e-12 * max(1.0, max_grad)
    positive_levels = np.unique(shifted_grad[shifted_grad > tolerance])[::-1]
    mass_above = 0.0
    threshold = 0.0
    threshold_mask = np.zeros_like(x_center, dtype=bool)
    found = False

    for level_index, level in enumerate(positive_levels):
        current_mask = np.isclose(shifted_grad, level, rtol=0.0, atol=tolerance)
        current_mass = float(np.sum(x_center[current_mask]))
        candidate_mass = level / (4.0 * L)

        if mass_above - tolerance <= candidate_mass <= mass_above + current_mass + tolerance:
            threshold = float(level)
            threshold_mask = current_mask
            found = True
            break

        mass_through_level = mass_above + current_mass
        next_level = float(positive_levels[level_index + 1]) if level_index + 1 < positive_levels.size else 0.0
        candidate_threshold = 4.0 * L * mass_through_level
        if next_level + tolerance < candidate_threshold < level - tolerance:
            threshold = candidate_threshold
            threshold_mask = np.zeros_like(x_center, dtype=bool)
            found = True
            break

        mass_above = mass_through_level

    if not found:
        threshold = max_grad
        threshold_mask = np.isclose(shifted_grad, threshold, rtol=0.0, atol=tolerance)

    moved_mass = threshold / (4.0 * L)
    if moved_mass <= tolerance:
        return x_center.copy()

    x_next = x_center.copy()
    strict_donor_mask = shifted_grad > threshold + tolerance
    removed_mass = float(np.sum(x_next[strict_donor_mask]))
    x_next[strict_donor_mask] = 0.0

    remaining_to_remove = moved_mass - removed_mass
    if remaining_to_remove > tolerance:
        threshold_indices = np.flatnonzero(threshold_mask)
        for index in threshold_indices:
            take = min(float(x_next[index]), remaining_to_remove)
            x_next[index] -= take
            remaining_to_remove -= take
            if remaining_to_remove <= tolerance:
                break

    if remaining_to_remove > 1.0e-9:
        raise RuntimeError("Could not reconstruct the simplex step from the threshold solution.")

    recipient_index = int(np.argmin(shifted_grad))
    x_next[recipient_index] += moved_mass
    x_next = np.maximum(x_next, 0.0)
    residual = 1.0 - float(np.sum(x_next))
    x_next[recipient_index] += residual
    return x_next / float(np.sum(x_next))


def theoretical_total_bound(
    *,
    lipschitz: float,
    primal_diameter: float,
    mu: float,
    dual_diameter: float,
    iterations: int,
) -> float:
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    optimization_error = 4.0 * lipschitz * primal_diameter / (iterations * (iterations + 1))
    smoothing_error = mu * dual_diameter
    return optimization_error + smoothing_error


def optimization_error_bound(*, lipschitz: float, primal_diameter: float, iterations: int) -> float:
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    return 4.0 * lipschitz * primal_diameter / (iterations * (iterations + 1))


def predicted_iterations(operator_norm: float, primal_diameter: float, dual_diameter: float, epsilon: float) -> int:
    predicted = 4.0 * operator_norm * math.sqrt(primal_diameter * dual_diameter) / epsilon
    return max(1, int(math.ceil(predicted)))


def default_check_frequency(epsilon: float) -> int:
    return 100 if epsilon >= 1.0e-2 else 1000


def optimization_iterations_for_target(*, lipschitz: float, primal_diameter: float, target: float) -> int:
    if target <= 0.0:
        raise ValueError("target must be positive.")
    constant = 4.0 * lipschitz * primal_diameter
    if constant <= target:
        return 1
    discriminant = 1.0 + 4.0 * constant / target
    return max(1, int(math.ceil(0.5 * (-1.0 + math.sqrt(discriminant)))))


def run_accelerated_method(
    *,
    initial_x: FloatArray,
    lipschitz: float,
    max_iterations: int,
    check_frequency: int,
    oracle: Callable[[FloatArray], OracleState],
    local_step: Callable[[FloatArray, FloatArray, float], FloatArray],
    aggregate_step: Callable[[FloatArray, float], FloatArray],
    should_stop: Callable[[AcceleratedSnapshot], bool],
    monotone_y: bool = False,
) -> AcceleratedSnapshot:
    x_current = initial_x.copy()
    gradient_sum = np.zeros_like(initial_x)
    auxiliary_sum: FloatArray | None = None
    alpha_sum = 0.0
    last_snapshot: AcceleratedSnapshot | None = None
    started_at = time.perf_counter()
    previous_y = initial_x.copy()
    previous_y_state = oracle(previous_y)

    for iteration in range(1, max_iterations + 1):
        x_state = oracle(x_current)
        y_candidate = local_step(x_current, x_state.gradient, lipschitz)
        y_candidate_state = oracle(y_candidate)

        if monotone_y:
            candidates = (
                (previous_y, previous_y_state),
                (x_current, x_state),
                (y_candidate, y_candidate_state),
            )
            y_current, y_state = min(candidates, key=lambda item: item[1].objective_value)
            y_current = y_current.copy()
        else:
            y_current = y_candidate
            y_state = y_candidate_state

        alpha = 0.5 * float(iteration)
        alpha_sum += alpha
        gradient_sum += alpha * x_state.gradient

        if x_state.auxiliary is not None:
            if auxiliary_sum is None:
                auxiliary_sum = np.zeros_like(x_state.auxiliary)
            auxiliary_sum += alpha * x_state.auxiliary

        z_current = aggregate_step(gradient_sum, lipschitz)
        tau = 2.0 / float(iteration + 2)
        x_current = tau * z_current + (1.0 - tau) * y_current
        previous_y = y_current.copy()
        previous_y_state = y_state

        if iteration % check_frequency == 0 or iteration == max_iterations:
            auxiliary_average = None if auxiliary_sum is None else auxiliary_sum / alpha_sum
            snapshot = AcceleratedSnapshot(
                iteration=iteration,
                y=y_current.copy(),
                state=y_state,
                auxiliary_average=None if auxiliary_average is None else auxiliary_average.copy(),
                elapsed_seconds=time.perf_counter() - started_at,
            )
            last_snapshot = snapshot
            if should_stop(snapshot):
                return snapshot

    if last_snapshot is None:
        raise RuntimeError("The accelerated method did not produce any checkpoint.")
    return last_snapshot


def max_affine_entropy_oracle(pieces_A: FloatArray, offsets: FloatArray, x: FloatArray, mu: float) -> OracleState:
    scores = pieces_A @ x + offsets
    nonsmooth_value = float(np.max(scores))
    shift = nonsmooth_value
    scaled = np.exp((scores - shift) / mu)
    partition = float(np.sum(scaled))
    dual = scaled / partition
    smoothed_value = shift + mu * math.log(partition / pieces_A.shape[0])
    gradient = pieces_A.T @ dual
    return OracleState(smoothed_value=smoothed_value, gradient=gradient, objective_value=nonsmooth_value, auxiliary=dual)
