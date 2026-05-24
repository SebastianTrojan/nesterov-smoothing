from __future__ import annotations

import math
import time

import numpy as np
from numpy.typing import ArrayLike

from .common import FloatArray, matrix, vector
from .domains import EntropySimplexDomain, ProjectedDomain
from .objectives import EntropySmoothedMaxAffineObjective, OracleState, _extract_linear_term
from .results import NesterovResult, PaperMatrixGameResult
from .smooth_terms import LinearSmoothTerm, SmoothTerm, ZeroSmoothTerm


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


def _fixed_mu_gap_bound(
    objective: EntropySmoothedMaxAffineObjective,
    domain: ProjectedDomain,
    iterations: int,
) -> float | None:
    D1 = float(domain.prox_diameter)
    if not math.isfinite(D1):
        return None
    if iterations < 1:
        return math.inf
    sigma1 = float(domain.sigma)
    optimization_bound = 4.0 * float(objective.lipschitz) * D1 / (sigma1 * iterations * iterations)
    return optimization_bound + objective.mu * objective.D2


def _build_nesterov_result(
    *,
    objective: EntropySmoothedMaxAffineObjective,
    domain: ProjectedDomain,
    x: FloatArray,
    state: OracleState,
    best_x: FloatArray,
    best_nonsmooth_value: float,
    dual_sum: FloatArray | None,
    alpha_sum: float,
    iterations: int,
    history: dict[str, list[float]],
) -> NesterovResult:
    dual_average: FloatArray | None = None
    dual_value: float | None = None
    primal_dual_gap: float | None = None

    if dual_sum is not None and alpha_sum > 0.0:
        dual_average = dual_sum / alpha_sum
        dual_value = objective.dual_value(domain, dual_average)
        primal_dual_gap = None if dual_value is None else state.nonsmooth_value - dual_value

    return NesterovResult(
        x=x,
        smoothed_value=state.smoothed_value,
        nonsmooth_value=state.nonsmooth_value,
        best_x=best_x,
        best_nonsmooth_value=best_nonsmooth_value,
        dual_variable=dual_average,
        dual_value=dual_value,
        primal_dual_gap=primal_dual_gap,
        theoretical_gap_bound=_fixed_mu_gap_bound(objective, domain, iterations),
        iterations=iterations,
        mu=objective.mu,
        lipschitz=float(objective.lipschitz),
        history=history,
    )


def _zero_lipschitz_result(
    objective: EntropySmoothedMaxAffineObjective,
    domain: ProjectedDomain,
) -> NesterovResult:
    linear_term = _extract_linear_term(objective.smooth_term, objective.dimension)
    if linear_term is None:
        raise ValueError(
            "The objective is flat under the provided smooth model. "
            "Pass a linear smooth term or solve the resulting linear problem directly."
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
        lipschitz=0.0,
        history={"smoothed_value": [state.smoothed_value], "nonsmooth_value": [state.nonsmooth_value]},
    )


def nesterov_accelerated_minimize(
    objective: EntropySmoothedMaxAffineObjective,
    domain: ProjectedDomain,
    *,
    x0: ArrayLike | None = None,
    max_iters: int = 200,
    monotone: bool = False,
    desired_accuracy: float | None = None,
    check_frequency: int | None = None,
) -> NesterovResult:
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1.")
    if desired_accuracy is not None and desired_accuracy <= 0.0:
        raise ValueError("desired_accuracy must be positive.")

    should_check_gap = check_frequency is not None
    if check_frequency is None:
        check_frequency = 1
    if check_frequency <= 0:
        raise ValueError("check_frequency must be positive.")

    L = float(objective.lipschitz)
    if L == 0.0:
        return _zero_lipschitz_result(objective, domain)

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

        if should_check_gap and desired_accuracy is not None and ((k + 1) % check_frequency == 0 or k + 1 == max_iters):
            result = _build_nesterov_result(
                objective=objective,
                domain=domain,
                x=final_y,
                state=final_state,
                best_x=best_x,
                best_nonsmooth_value=best_nonsmooth_value,
                dual_sum=dual_sum,
                alpha_sum=alpha_sum,
                iterations=k + 1,
                history=history,
            )
            if result.primal_dual_gap is not None and result.primal_dual_gap <= desired_accuracy:
                return result

    return _build_nesterov_result(
        objective=objective,
        domain=domain,
        x=final_y,
        state=final_state,
        best_x=best_x,
        best_nonsmooth_value=best_nonsmooth_value,
        dual_sum=dual_sum,
        alpha_sum=alpha_sum,
        iterations=max_iters,
        history=history,
    )


def _nesterov_accelerated_minimize_entropy_simplex(
    objective: EntropySmoothedMaxAffineObjective,
    domain: EntropySimplexDomain,
    *,
    x0: ArrayLike | None = None,
    max_iters: int = 200,
    monotone: bool = False,
    desired_accuracy: float | None = None,
    check_frequency: int | None = None,
) -> NesterovResult:
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1.")
    if desired_accuracy is not None and desired_accuracy <= 0.0:
        raise ValueError("desired_accuracy must be positive.")

    should_check_gap = check_frequency is not None
    if check_frequency is None:
        check_frequency = 1
    if check_frequency <= 0:
        raise ValueError("check_frequency must be positive.")

    L = float(objective.lipschitz)
    if L == 0.0:
        return nesterov_accelerated_minimize(
            objective,
            domain,
            x0=x0,
            max_iters=max_iters,
            monotone=monotone,
            desired_accuracy=desired_accuracy,
            check_frequency=check_frequency,
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
        y_candidate = domain.prox_gradient_step(x_current, x_state.grad, L)
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

        z_k = domain.accumulated_model_minimizer(gradient_sum, L)
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

        if should_check_gap and desired_accuracy is not None and ((k + 1) % check_frequency == 0 or k + 1 == max_iters):
            result = _build_nesterov_result(
                objective=objective,
                domain=domain,
                x=final_y,
                state=final_state,
                best_x=best_x,
                best_nonsmooth_value=best_nonsmooth_value,
                dual_sum=dual_sum,
                alpha_sum=alpha_sum,
                iterations=k + 1,
                history=history,
            )
            if result.primal_dual_gap is not None and result.primal_dual_gap <= desired_accuracy:
                return result

    return _build_nesterov_result(
        objective=objective,
        domain=domain,
        x=final_y,
        state=final_state,
        best_x=best_x,
        best_nonsmooth_value=best_nonsmooth_value,
        dual_sum=dual_sum,
        alpha_sum=alpha_sum,
        iterations=max_iters,
        history=history,
    )


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
    check_frequency: int | None = None,
    x0: ArrayLike | None = None,
    monotone: bool = False,
    operator_norm: float | None = None,
    dual_diameter: float | None = None,
    dual_sigma: float = 1.0,
) -> NesterovResult:
    matrix_value = matrix(A, name="A")
    offsets = vector(b, name="b")
    if matrix_value.shape[0] != offsets.size:
        raise ValueError("A and b must agree on the number of affine pieces.")
    if linear_term is not None and smooth_term is not None:
        raise ValueError("Pass either linear_term or smooth_term, not both.")

    if smooth_term is None:
        smooth_term = ZeroSmoothTerm(matrix_value.shape[1]) if linear_term is None else LinearSmoothTerm(linear_term)

    if operator_norm is None:
        operator_norm = float(np.max(np.linalg.norm(matrix_value, axis=1)))
    if dual_diameter is None:
        dual_diameter = math.log(matrix_value.shape[0]) if matrix_value.shape[0] > 1 else 0.0

    if max_iters is None:
        if desired_accuracy is None:
            max_iters = 200
        else:
            max_iters = _iterations_for_accuracy(
                desired_accuracy=desired_accuracy,
                operator_norm=operator_norm,
                D1=float(domain.prox_diameter),
                D2=dual_diameter,
                sigma1=float(domain.sigma),
                sigma2=dual_sigma,
                smooth_lipschitz=float(smooth_term.lipschitz),
            )

    if mu is None:
        mu = _mu_for_iterations(
            iterations=max_iters,
            operator_norm=operator_norm,
            D1=float(domain.prox_diameter),
            D2=dual_diameter,
            sigma1=float(domain.sigma),
            sigma2=dual_sigma,
        )

    objective = EntropySmoothedMaxAffineObjective(
        A=matrix_value,
        b=offsets,
        mu=mu,
        smooth_term=smooth_term,
        operator_norm_override=operator_norm,
        dual_diameter=dual_diameter,
        dual_sigma=dual_sigma,
    )

    if isinstance(domain, EntropySimplexDomain):
        return _nesterov_accelerated_minimize_entropy_simplex(
            objective,
            domain,
            x0=x0,
            max_iters=max_iters,
            monotone=monotone,
            desired_accuracy=desired_accuracy,
            check_frequency=check_frequency,
        )

    return nesterov_accelerated_minimize(
        objective,
        domain,
        x0=x0,
        max_iters=max_iters,
        monotone=monotone,
        desired_accuracy=desired_accuracy,
        check_frequency=check_frequency,
    )


def solve_paper_matrix_game(
    A: ArrayLike,
    epsilon: float,
    *,
    check_frequency: int | None = None,
    max_iterations: int | None = None,
) -> PaperMatrixGameResult:
    matrix_value = matrix(A, name="A")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    m, n = matrix_value.shape
    if m <= 0 or n <= 0:
        raise ValueError("A must have positive dimensions.")

    D1 = math.log(n)
    D2 = math.log(m)
    if D1 <= 0.0 or D2 <= 0.0:
        raise ValueError("The paper's entropy setup requires m >= 2 and n >= 2.")

    operator_norm = float(np.max(np.abs(matrix_value)))
    mu = epsilon / (2.0 * D2)
    predicted_iterations = _iterations_for_accuracy(
        desired_accuracy=epsilon,
        operator_norm=operator_norm,
        D1=D1,
        D2=D2,
        sigma1=1.0,
        sigma2=1.0,
        smooth_lipschitz=0.0,
    )

    if check_frequency is None:
        check_frequency = 100 if epsilon >= 1.0e-2 else 1000
    if check_frequency <= 0:
        raise ValueError("check_frequency must be positive.")

    if max_iterations is None:
        max_iterations = int(math.ceil(predicted_iterations / check_frequency) * check_frequency)
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    started_at = time.perf_counter()
    result = solve_max_affine(
        A=matrix_value,
        b=np.zeros(m, dtype=np.float64),
        domain=EntropySimplexDomain(n),
        mu=mu,
        max_iters=max_iterations,
        desired_accuracy=epsilon,
        check_frequency=check_frequency,
        operator_norm=operator_norm,
        dual_diameter=D2,
        dual_sigma=1.0,
    )
    elapsed_seconds = time.perf_counter() - started_at

    dual = result.dual_variable if result.dual_variable is not None else np.full(m, 1.0 / m, dtype=np.float64)
    primal_value = result.nonsmooth_value
    dual_value = result.dual_value
    if dual_value is None:
        dual_value = float(np.min(matrix_value.T @ dual))
    gap = primal_value - dual_value

    return PaperMatrixGameResult(
        x=result.x.copy(),
        u=dual.copy(),
        primal_value=primal_value,
        dual_value=dual_value,
        gap=gap,
        iterations=result.iterations,
        predicted_iterations=predicted_iterations,
        mu=result.mu,
        lipschitz=result.lipschitz,
        operator_norm=operator_norm,
        check_frequency=check_frequency,
        converged=gap <= epsilon,
        elapsed_seconds=elapsed_seconds,
    )
