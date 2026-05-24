from __future__ import annotations

from dataclasses import dataclass, field

from .common import FloatArray


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
