from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "BoxDomain",
    "CallableSmoothTerm",
    "EntropySmoothedMaxAffineObjective",
    "L2BallDomain",
    "LinearSmoothTerm",
    "PaperExperimentCell",
    "PaperMatrixGameResult",
    "NesterovResult",
    "SimplexDomain",
    "UnconstrainedDomain",
    "nesterov_accelerated_minimize",
    "paper_section6_grid",
    "run_paper_matrix_game_grid",
    "solve_max_affine",
    "solve_paper_matrix_game",
]


def _vector(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return array.astype(np.float64, copy=True)


def _matrix(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return array.astype(np.float64, copy=True)


def _project_to_simplex(x: FloatArray) -> FloatArray:
    if x.ndim != 1:
        raise ValueError("Simplex projection expects a 1D array.")
    n = x.size
    if n == 0:
        raise ValueError("Simplex projection needs at least one coordinate.")
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho_candidates = np.nonzero(u - cssv / np.arange(1, n + 1) > 0.0)[0]
    if rho_candidates.size == 0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    rho = int(rho_candidates[-1])
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(x - theta, 0.0)


class ProjectedDomain(Protocol):
    center: FloatArray
    prox_diameter: float
    sigma: float

    def project(self, x: ArrayLike) -> FloatArray:
        ...

    def linear_minimizer(self, gradient: ArrayLike) -> FloatArray:
        ...


@dataclass
class BoxDomain:
    lower: ArrayLike
    upper: ArrayLike
    sigma: float = 1.0

    def __post_init__(self) -> None:
        self.lower = _vector(self.lower, name="lower")
        self.upper = _vector(self.upper, name="upper")
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if np.any(self.lower > self.upper):
            raise ValueError("Each lower bound must be <= the matching upper bound.")
        self.center = 0.5 * (self.lower + self.upper)
        half_width = 0.5 * (self.upper - self.lower)
        self.prox_diameter = 0.5 * float(half_width @ half_width)

    def project(self, x: ArrayLike) -> FloatArray:
        vector = _vector(x, name="x")
        if vector.shape != self.center.shape:
            raise ValueError("x has the wrong dimension for this box.")
        return np.clip(vector, self.lower, self.upper)

    def linear_minimizer(self, gradient: ArrayLike) -> FloatArray:
        grad = _vector(gradient, name="gradient")
        if grad.shape != self.center.shape:
            raise ValueError("gradient has the wrong dimension for this box.")
        return np.where(grad > 0.0, self.lower, self.upper).astype(np.float64, copy=True)


@dataclass
class SimplexDomain:
    dimension: int
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")
        self.center = np.full(self.dimension, 1.0 / self.dimension, dtype=np.float64)
        self.prox_diameter = 0.5 * (1.0 - 1.0 / self.dimension)

    def project(self, x: ArrayLike) -> FloatArray:
        vector = _vector(x, name="x")
        if vector.size != self.dimension:
            raise ValueError("x has the wrong dimension for this simplex.")
        return _project_to_simplex(vector)

    def linear_minimizer(self, gradient: ArrayLike) -> FloatArray:
        grad = _vector(gradient, name="gradient")
        if grad.size != self.dimension:
            raise ValueError("gradient has the wrong dimension for this simplex.")
        result = np.zeros(self.dimension, dtype=np.float64)
        result[int(np.argmin(grad))] = 1.0
        return result


@dataclass
class L2BallDomain:
    radius: float
    dimension: int
    center_point: ArrayLike | None = None
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")
        if self.radius < 0.0:
            raise ValueError("radius must be non-negative.")
        if self.center_point is None:
            self.center = np.zeros(self.dimension, dtype=np.float64)
        else:
            self.center = _vector(self.center_point, name="center_point")
            if self.center.size != self.dimension:
                raise ValueError("center_point has the wrong dimension.")
        self.prox_diameter = 0.5 * self.radius * self.radius

    def project(self, x: ArrayLike) -> FloatArray:
        vector = _vector(x, name="x")
        if vector.size != self.dimension:
            raise ValueError("x has the wrong dimension for this ball.")
        displacement = vector - self.center
        norm = float(np.linalg.norm(displacement))
        if norm <= self.radius or norm == 0.0:
            return vector
        return self.center + (self.radius / norm) * displacement

    def linear_minimizer(self, gradient: ArrayLike) -> FloatArray:
        grad = _vector(gradient, name="gradient")
        if grad.size != self.dimension:
            raise ValueError("gradient has the wrong dimension for this ball.")
        norm = float(np.linalg.norm(grad))
        if norm == 0.0 or self.radius == 0.0:
            return self.center.copy()
        return self.center - (self.radius / norm) * grad


@dataclass
class UnconstrainedDomain:
    dimension: int
    center_point: ArrayLike | None = None
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")
        if self.center_point is None:
            self.center = np.zeros(self.dimension, dtype=np.float64)
        else:
            self.center = _vector(self.center_point, name="center_point")
            if self.center.size != self.dimension:
                raise ValueError("center_point has the wrong dimension.")
        self.prox_diameter = math.inf

    def project(self, x: ArrayLike) -> FloatArray:
        vector = _vector(x, name="x")
        if vector.size != self.dimension:
            raise ValueError("x has the wrong dimension for this domain.")
        return vector

    def linear_minimizer(self, gradient: ArrayLike) -> FloatArray:
        grad = _vector(gradient, name="gradient")
        if grad.size != self.dimension:
            raise ValueError("gradient has the wrong dimension for this domain.")
        if np.allclose(grad, 0.0):
            return self.center.copy()
        raise ValueError("Linear minimization over R^n is unbounded for non-zero gradients.")


class SmoothTerm(Protocol):
    lipschitz: float

    def value_grad(self, x: FloatArray) -> tuple[float, FloatArray]:
        ...


@dataclass
class ZeroSmoothTerm:
    dimension: int
    lipschitz: float = 0.0

    def value_grad(self, x: FloatArray) -> tuple[float, FloatArray]:
        return 0.0, np.zeros_like(x, dtype=np.float64)


@dataclass
class LinearSmoothTerm:
    coefficients: ArrayLike
    lipschitz: float = 0.0

    def __post_init__(self) -> None:
        self.coefficients = _vector(self.coefficients, name="coefficients")

    def value_grad(self, x: FloatArray) -> tuple[float, FloatArray]:
        if x.size != self.coefficients.size:
            raise ValueError("x has the wrong dimension for this linear term.")
        return float(self.coefficients @ x), self.coefficients.copy()


@dataclass
class CallableSmoothTerm:
    oracle: Callable[[FloatArray], tuple[float, ArrayLike]]
    lipschitz: float

    def value_grad(self, x: FloatArray) -> tuple[float, FloatArray]:
        value, grad = self.oracle(x.copy())
        return float(value), _vector(grad, name="smooth_grad")


@dataclass
class OracleState:
    smoothed_value: float
    grad: FloatArray
    dual: FloatArray
    nonsmooth_value: float


@dataclass
class PaperMatrixGameResult:
    x: FloatArray
    u: FloatArray
    primal_value: float
    dual_value: float
    gap: float
    iterations: int
    predicted_iterations: int
    mu: float
    lipschitz: float
    operator_norm: float
    check_frequency: int
    converged: bool
    elapsed_seconds: float


@dataclass
class PaperExperimentCell:
    epsilon: float
    m: int
    n: int
    iterations: int
    predicted_iterations: int
    percent_of_predicted: float
    primal_value: float
    dual_value: float
    gap: float
    mu: float
    lipschitz: float
    operator_norm: float
    converged: bool
    elapsed_seconds: float


@dataclass
class EntropySmoothedMaxAffineObjective:
    A: ArrayLike
    b: ArrayLike
    mu: float
    smooth_term: SmoothTerm | None = None

    def __post_init__(self) -> None:
        self.A = _matrix(self.A, name="A")
        self.b = _vector(self.b, name="b")
        if self.A.shape[0] == 0:
            raise ValueError("A must contain at least one affine piece.")
        if self.A.shape[0] != self.b.size:
            raise ValueError("A and b must agree on the number of affine pieces.")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive.")
        self.num_pieces, self.dimension = self.A.shape
        if self.smooth_term is None:
            self.smooth_term = ZeroSmoothTerm(self.dimension)
        self.sigma2 = 1.0
        self.D2 = math.log(self.num_pieces) if self.num_pieces > 1 else 0.0
        self.operator_norm = float(np.max(np.linalg.norm(self.A, axis=1)))
        self.smooth_lipschitz = float(self.smooth_term.lipschitz)
        if self.num_pieces == 1:
            self.lipschitz = self.smooth_lipschitz
        else:
            self.lipschitz = self.smooth_lipschitz + (self.operator_norm ** 2) / self.mu

    def oracle(self, x: ArrayLike) -> OracleState:
        vector = _vector(x, name="x")
        if vector.size != self.dimension:
            raise ValueError("x has the wrong dimension for this objective.")

        smooth_value, smooth_grad = self.smooth_term.value_grad(vector)
        smooth_grad = _vector(smooth_grad, name="smooth_grad")
        if smooth_grad.size != self.dimension:
            raise ValueError("smooth_grad has the wrong dimension.")

        scores = self.A @ vector + self.b
        nonsmooth_max = float(np.max(scores))
        nonsmooth_value = smooth_value + nonsmooth_max

        if self.num_pieces == 1:
            dual = np.array([1.0], dtype=np.float64)
            grad = smooth_grad + self.A[0]
            return OracleState(
                smoothed_value=nonsmooth_value,
                grad=grad,
                dual=dual,
                nonsmooth_value=nonsmooth_value,
            )

        shift = nonsmooth_max
        scaled = np.exp((scores - shift) / self.mu)
        partition = float(np.sum(scaled))
        dual = scaled / partition
        smoothed_max = shift + self.mu * math.log(partition / self.num_pieces)
        grad = smooth_grad + self.A.T @ dual

        return OracleState(
            smoothed_value=smooth_value + smoothed_max,
            grad=grad,
            dual=dual,
            nonsmooth_value=nonsmooth_value,
        )

    def dual_value(self, domain: ProjectedDomain, dual_variable: ArrayLike) -> float | None:
        linear_term = _extract_linear_term(self.smooth_term, self.dimension)
        if linear_term is None:
            return None

        dual = _vector(dual_variable, name="dual_variable")
        if dual.size != self.num_pieces:
            raise ValueError("dual_variable has the wrong dimension.")

        reduced_gradient = self.A.T @ dual + linear_term
        try:
            primal_argmin = domain.linear_minimizer(reduced_gradient)
        except ValueError:
            return None
        return float(self.b @ dual + reduced_gradient @ primal_argmin)


def _extract_linear_term(smooth_term: SmoothTerm | None, dimension: int) -> FloatArray | None:
    if smooth_term is None:
        return np.zeros(dimension, dtype=np.float64)
    if isinstance(smooth_term, ZeroSmoothTerm):
        return np.zeros(dimension, dtype=np.float64)
    if isinstance(smooth_term, LinearSmoothTerm):
        return smooth_term.coefficients.copy()
    return None


def _mu_for_iterations(
    *,
    iterations: int,
    operator_norm: float,
    D1: float,
    D2: float,
    sigma1: float,
    sigma2: float,
) -> float:
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if operator_norm == 0.0 or D2 == 0.0:
        return 1.0
    if not math.isfinite(D1):
        raise ValueError("Automatic smoothing needs a bounded domain with finite prox diameter.")
    return (
        2.0
        * operator_norm
        / iterations
        * math.sqrt(D1 / (sigma1 * sigma2 * D2))
    )


def _theorem3_gap_bound(
    *,
    iterations: int,
    operator_norm: float,
    D1: float,
    D2: float,
    sigma1: float,
    sigma2: float,
    smooth_lipschitz: float,
) -> float:
    if iterations < 1:
        return math.inf
    bound = 4.0 * smooth_lipschitz * D1 / (sigma1 * iterations * iterations)
    if operator_norm > 0.0 and D2 > 0.0:
        bound += (
            4.0
            * operator_norm
            * math.sqrt(D1 * D2 / (sigma1 * sigma2))
            / iterations
        )
    return bound


def _iterations_for_accuracy(
    *,
    desired_accuracy: float,
    operator_norm: float,
    D1: float,
    D2: float,
    sigma1: float,
    sigma2: float,
    smooth_lipschitz: float,
) -> int:
    if desired_accuracy <= 0.0:
        raise ValueError("desired_accuracy must be positive.")
    if not math.isfinite(D1):
        raise ValueError("desired_accuracy requires a bounded domain with finite prox diameter.")

    upper = 1
    while _theorem3_gap_bound(
        iterations=upper,
        operator_norm=operator_norm,
        D1=D1,
        D2=D2,
        sigma1=sigma1,
        sigma2=sigma2,
        smooth_lipschitz=smooth_lipschitz,
    ) > desired_accuracy:
        upper *= 2
        if upper > 1_000_000_000:
            raise RuntimeError("Could not bracket the iteration count for the requested accuracy.")

    lower = max(1, upper // 2)
    while lower < upper:
        middle = (lower + upper) // 2
        if _theorem3_gap_bound(
            iterations=middle,
            operator_norm=operator_norm,
            D1=D1,
            D2=D2,
            sigma1=sigma1,
            sigma2=sigma2,
            smooth_lipschitz=smooth_lipschitz,
        ) <= desired_accuracy:
            upper = middle
        else:
            lower = middle + 1
    return lower


@dataclass
class NesterovResult:
    x: FloatArray
    smoothed_value: float
    nonsmooth_value: float
    best_x: FloatArray
    best_nonsmooth_value: float
    dual_variable: FloatArray | None
    dual_value: float | None
    primal_dual_gap: float | None
    theoretical_gap_bound: float | None
    iterations: int
    mu: float
    lipschitz: float
    history: dict[str, list[float]] = field(default_factory=dict)


def nesterov_accelerated_minimize(
    objective: EntropySmoothedMaxAffineObjective,
    domain: ProjectedDomain,
    *,
    x0: ArrayLike | None = None,
    max_iters: int = 200,
    monotone: bool = False,
) -> NesterovResult:
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1.")

    L = float(objective.lipschitz)
    sigma1 = float(domain.sigma)
    D1 = float(domain.prox_diameter)

    if L == 0.0:
        linear_term = _extract_linear_term(objective.smooth_term, objective.dimension)
        if linear_term is None:
            raise ValueError(
                "The objective is flat under the provided smooth model. "
                "Pass a curved smooth term or solve the resulting linear program directly."
            )
        if objective.num_pieces > 1 and objective.operator_norm == 0.0:
            direct_gradient = linear_term
            dual = np.zeros(objective.num_pieces, dtype=np.float64)
            dual[int(np.argmax(objective.b))] = 1.0
        else:
            direct_gradient = linear_term + objective.A[0]
            dual = np.array([1.0], dtype=np.float64)
        x_star = domain.linear_minimizer(direct_gradient)
        state = objective.oracle(x_star)
        dual_value = objective.dual_value(domain, dual)
        gap = None if dual_value is None else state.nonsmooth_value - dual_value
        return NesterovResult(
            x=x_star,
            smoothed_value=state.smoothed_value,
            nonsmooth_value=state.nonsmooth_value,
            best_x=x_star.copy(),
            best_nonsmooth_value=state.nonsmooth_value,
            dual_variable=dual,
            dual_value=dual_value,
            primal_dual_gap=gap,
            theoretical_gap_bound=0.0,
            iterations=0,
            mu=objective.mu,
            lipschitz=L,
            history={"smoothed_value": [state.smoothed_value], "nonsmooth_value": [state.nonsmooth_value]},
        )

    x_current = domain.project(domain.center if x0 is None else x0)
    alpha_sum = 0.0
    gradient_sum = np.zeros_like(x_current, dtype=np.float64)
    dual_sum = np.zeros(objective.num_pieces, dtype=np.float64)

    previous_y: FloatArray | None = None
    previous_y_state: OracleState | None = None

    final_y = x_current.copy()
    final_state = objective.oracle(final_y)
    best_x = final_y.copy()
    best_nonsmooth_value = final_state.nonsmooth_value

    history = {
        "smoothed_value": [],
        "nonsmooth_value": [],
        "best_nonsmooth_value": [],
    }

    for k in range(max_iters):
        x_state = objective.oracle(x_current)
        y_candidate = domain.project(x_current - x_state.grad / L)
        y_candidate_state = objective.oracle(y_candidate)

        if monotone and previous_y is not None and previous_y_state is not None:
            choices = [
                (previous_y_state.nonsmooth_value, previous_y, previous_y_state),
                (x_state.nonsmooth_value, x_current, x_state),
                (y_candidate_state.nonsmooth_value, y_candidate, y_candidate_state),
            ]
            _, y_k, y_state = min(choices, key=lambda item: item[0])
            y_k = y_k.copy()
        else:
            y_k = y_candidate
            y_state = y_candidate_state

        alpha = 0.5 * float(k + 1)
        alpha_sum += alpha
        gradient_sum += alpha * x_state.grad
        dual_sum += alpha * x_state.dual

        z_k = domain.project(domain.center - gradient_sum / L)
        tau = 2.0 / float(k + 3)
        x_current = tau * z_k + (1.0 - tau) * y_k

        previous_y = y_k.copy()
        previous_y_state = y_state
        final_y = y_k.copy()
        final_state = y_state

        if y_state.nonsmooth_value < best_nonsmooth_value:
            best_nonsmooth_value = y_state.nonsmooth_value
            best_x = y_k.copy()

        history["smoothed_value"].append(y_state.smoothed_value)
        history["nonsmooth_value"].append(y_state.nonsmooth_value)
        history["best_nonsmooth_value"].append(best_nonsmooth_value)

    dual_average = dual_sum / alpha_sum
    dual_value = objective.dual_value(domain, dual_average)
    primal_dual_gap = None if dual_value is None else final_state.nonsmooth_value - dual_value

    theoretical_gap_bound = None
    if math.isfinite(D1):
        theoretical_gap_bound = 4.0 * objective.smooth_lipschitz * D1 / (sigma1 * max_iters * max_iters)
        if objective.operator_norm > 0.0 and objective.D2 > 0.0:
            theoretical_gap_bound += objective.mu * objective.D2
            theoretical_gap_bound += (
                4.0
                * objective.operator_norm
                * objective.operator_norm
                * D1
                / (objective.mu * sigma1 * objective.sigma2 * max_iters * max_iters)
            )

    return NesterovResult(
        x=final_y,
        smoothed_value=final_state.smoothed_value,
        nonsmooth_value=final_state.nonsmooth_value,
        best_x=best_x,
        best_nonsmooth_value=best_nonsmooth_value,
        dual_variable=dual_average,
        dual_value=dual_value,
        primal_dual_gap=primal_dual_gap,
        theoretical_gap_bound=theoretical_gap_bound,
        iterations=max_iters,
        mu=objective.mu,
        lipschitz=L,
        history=history,
    )


def _stable_logsumexp(values: FloatArray, scale: float) -> tuple[float, FloatArray]:
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    shift = float(np.max(values))
    shifted = np.exp((values - shift) / scale)
    partition = float(np.sum(shifted))
    return shift + scale * math.log(partition), shifted / partition


def _simplex_entropy_argmin(linear_term: FloatArray, scale: float) -> FloatArray:
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    shifted = -linear_term / scale
    max_shift = float(np.max(shifted))
    weights = np.exp(shifted - max_shift)
    return weights / float(np.sum(weights))


def _simplex_l1_squared_step(x_bar: ArrayLike, gradient: ArrayLike, L: float) -> FloatArray:
    x_center = _vector(x_bar, name="x_bar")
    grad = _vector(gradient, name="gradient")
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
    partial_from_level = 0.0
    threshold_mask = np.zeros_like(x_center, dtype=bool)
    found = False

    for level_index, level in enumerate(positive_levels):
        current_mask = np.isclose(shifted_grad, level, rtol=0.0, atol=tolerance)
        current_mass = float(np.sum(x_center[current_mask]))
        candidate_mass = level / (4.0 * L)

        if mass_above - tolerance <= candidate_mass <= mass_above + current_mass + tolerance:
            threshold = float(level)
            partial_from_level = max(0.0, candidate_mass - mass_above)
            threshold_mask = current_mask
            found = True
            break

        mass_through_level = mass_above + current_mass
        next_level = float(positive_levels[level_index + 1]) if level_index + 1 < positive_levels.size else 0.0
        candidate_threshold = 4.0 * L * mass_through_level
        if next_level + tolerance < candidate_threshold < level - tolerance:
            threshold = candidate_threshold
            partial_from_level = 0.0
            threshold_mask = np.zeros_like(x_center, dtype=bool)
            found = True
            break

        mass_above = mass_through_level

    if not found:
        threshold = max_grad
        threshold_mask = np.isclose(shifted_grad, threshold, rtol=0.0, atol=tolerance)
        partial_from_level = threshold / (4.0 * L) - float(np.sum(x_center[shifted_grad > threshold + tolerance]))
        partial_from_level = max(0.0, partial_from_level)

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


def solve_paper_matrix_game(
    A: ArrayLike,
    epsilon: float,
    *,
    check_frequency: int | None = None,
    max_iterations: int | None = None,
) -> PaperMatrixGameResult:
    matrix = _matrix(A, name="A")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    m, n = matrix.shape
    if m <= 0 or n <= 0:
        raise ValueError("A must have positive dimensions.")

    D1 = math.log(n)
    D2 = math.log(m)
    if D1 <= 0.0 or D2 <= 0.0:
        raise ValueError("The paper's entropy setup requires m >= 2 and n >= 2.")

    operator_norm = float(np.max(np.abs(matrix)))
    mu = epsilon / (2.0 * D2)
    L_mu = (operator_norm * operator_norm) / mu if operator_norm > 0.0 else 0.0

    predicted_real = 4.0 * operator_norm * math.sqrt(D1 * D2) / epsilon
    predicted_iterations = max(1, math.ceil(predicted_real - 1.0))

    if check_frequency is None:
        check_frequency = 100 if epsilon >= 1.0e-2 else 1000
    if check_frequency <= 0:
        raise ValueError("check_frequency must be positive.")

    if max_iterations is None:
        max_iterations = int(math.ceil(predicted_iterations / check_frequency) * check_frequency)
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    x_current = np.full(n, 1.0 / n, dtype=np.float64)
    accumulated_gradient = np.zeros(n, dtype=np.float64)
    accumulated_dual = np.zeros(m, dtype=np.float64)

    final_x = x_current.copy()
    final_u = np.full(m, 1.0 / m, dtype=np.float64)
    final_primal = float(np.max(matrix @ final_x))
    final_dual = float(np.min(matrix.T @ final_u))
    final_gap = final_primal - final_dual
    converged = False

    started_at = time.perf_counter()
    for k in range(max_iterations):
        scores = matrix @ x_current
        smoothed_value, dual = _stable_logsumexp(scores, mu)
        smoothed_value -= mu * D2
        gradient = matrix.T @ dual
        y_current = _simplex_l1_squared_step(x_current, gradient, L_mu)

        alpha = 0.5 * float(k + 1)
        accumulated_gradient += alpha * gradient
        accumulated_dual += alpha * dual

        if (k + 1) % check_frequency == 0 or k + 1 == max_iterations:
            averaging_factor = 0.25 * float(k + 1) * float(k + 2)
            u_hat = accumulated_dual / averaging_factor
            Ax = matrix @ y_current
            primal_value = float(np.max(Ax))
            dual_value = float(np.min(matrix.T @ u_hat))
            gap = primal_value - dual_value

            final_x = y_current.copy()
            final_u = u_hat
            final_primal = primal_value
            final_dual = dual_value
            final_gap = gap

            if gap <= epsilon:
                converged = True
                elapsed_seconds = time.perf_counter() - started_at
                return PaperMatrixGameResult(
                    x=final_x,
                    u=final_u,
                    primal_value=final_primal,
                    dual_value=final_dual,
                    gap=final_gap,
                    iterations=k + 1,
                    predicted_iterations=predicted_iterations,
                    mu=mu,
                    lipschitz=L_mu,
                    operator_norm=operator_norm,
                    check_frequency=check_frequency,
                    converged=converged,
                    elapsed_seconds=elapsed_seconds,
                )

        z_current = _simplex_entropy_argmin(accumulated_gradient, L_mu)
        tau = 2.0 / float(k + 3)
        x_current = tau * z_current + (1.0 - tau) * y_current

    elapsed_seconds = time.perf_counter() - started_at
    return PaperMatrixGameResult(
        x=final_x,
        u=final_u,
        primal_value=final_primal,
        dual_value=final_dual,
        gap=final_gap,
        iterations=max_iterations,
        predicted_iterations=predicted_iterations,
        mu=mu,
        lipschitz=L_mu,
        operator_norm=operator_norm,
        check_frequency=check_frequency,
        converged=converged,
        elapsed_seconds=elapsed_seconds,
    )


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
        if m_values is None:
            epsilon_m_values = section6_grid[epsilon]["m_values"]
        else:
            epsilon_m_values = m_values

        if n_values is None:
            epsilon_n_values = section6_grid[epsilon]["n_values"]
        else:
            epsilon_n_values = n_values

        for m in epsilon_m_values:
            for n in epsilon_n_values:
                seed = hash((base_seed, epsilon, m, n)) & 0xFFFFFFFF
                rng = np.random.default_rng(seed)
                matrix = rng.uniform(-1.0, 1.0, size=(m, n))
                result = solve_paper_matrix_game(
                    matrix,
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


def solve_max_affine(
    A: ArrayLike,
    b: ArrayLike,
    domain: ProjectedDomain,
    *,
    linear_term: ArrayLike | None = None,
    smooth_term: SmoothTerm | None = None,
    mu: float | None = None,
    max_iters: int | None = None,
    desired_accuracy: float | None = None,
    x0: ArrayLike | None = None,
    monotone: bool = False,
) -> NesterovResult:
    matrix = _matrix(A, name="A")
    offsets = _vector(b, name="b")
    if matrix.shape[0] != offsets.size:
        raise ValueError("A and b must agree on the number of affine pieces.")

    if linear_term is not None and smooth_term is not None:
        raise ValueError("Pass either linear_term or smooth_term, not both.")

    if smooth_term is None:
        if linear_term is None:
            smooth_term = ZeroSmoothTerm(matrix.shape[1])
        else:
            smooth_term = LinearSmoothTerm(linear_term)

    if max_iters is None:
        if desired_accuracy is None:
            max_iters = 200
        else:
            max_iters = _iterations_for_accuracy(
                desired_accuracy=desired_accuracy,
                operator_norm=float(np.max(np.linalg.norm(matrix, axis=1))),
                D1=float(domain.prox_diameter),
                D2=math.log(matrix.shape[0]) if matrix.shape[0] > 1 else 0.0,
                sigma1=float(domain.sigma),
                sigma2=1.0,
                smooth_lipschitz=float(smooth_term.lipschitz),
            )

    if mu is None:
        mu = _mu_for_iterations(
            iterations=max_iters,
            operator_norm=float(np.max(np.linalg.norm(matrix, axis=1))),
            D1=float(domain.prox_diameter),
            D2=math.log(matrix.shape[0]) if matrix.shape[0] > 1 else 0.0,
            sigma1=float(domain.sigma),
            sigma2=1.0,
        )

    objective = EntropySmoothedMaxAffineObjective(
        A=matrix,
        b=offsets,
        mu=mu,
        smooth_term=smooth_term,
    )
    return nesterov_accelerated_minimize(
        objective,
        domain,
        x0=x0,
        max_iters=max_iters,
        monotone=monotone,
    )


def _demo() -> None:
    A = np.array([[1.0], [-1.0]], dtype=np.float64)
    b = np.array([0.0, 0.0], dtype=np.float64)
    domain = BoxDomain(lower=np.array([-1.0]), upper=np.array([1.0]))
    result = solve_max_affine(A, b, domain, max_iters=300, monotone=True)

    print("Final iterate:", result.x)
    print("Best iterate :", result.best_x)
    print("f(best_x)    :", result.best_nonsmooth_value)
    print("Gap bound    :", result.theoretical_gap_bound)


if __name__ == "__main__":
    _demo()
