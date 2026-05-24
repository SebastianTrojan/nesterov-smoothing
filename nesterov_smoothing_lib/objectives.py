from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .common import FloatArray, matrix, vector
from .domains import ProjectedDomain
from .smooth_terms import LinearSmoothTerm, SmoothTerm, ZeroSmoothTerm


@dataclass
class OracleState:
    smoothed_value: float
    grad: FloatArray
    dual: FloatArray
    nonsmooth_value: float


def _extract_linear_term(smooth_term: SmoothTerm | None, dimension: int) -> FloatArray | None:
    if smooth_term is None:
        return np.zeros(dimension, dtype=np.float64)
    if isinstance(smooth_term, ZeroSmoothTerm):
        return np.zeros(dimension, dtype=np.float64)
    if isinstance(smooth_term, LinearSmoothTerm):
        return smooth_term.coefficients.copy()
    return None


@dataclass
class EntropySmoothedMaxAffineObjective:
    A: ArrayLike
    b: ArrayLike
    mu: float
    smooth_term: SmoothTerm | None = None
    operator_norm_override: float | None = None
    dual_diameter: float | None = None
    dual_sigma: float = 1.0

    def __post_init__(self) -> None:
        self.A = matrix(self.A, name="A")
        self.b = vector(self.b, name="b")
        if self.A.shape[0] == 0:
            raise ValueError("A must contain at least one affine piece.")
        if self.A.shape[0] != self.b.size:
            raise ValueError("A and b must agree on the number of affine pieces.")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive.")
        if self.dual_sigma <= 0.0:
            raise ValueError("dual_sigma must be positive.")

        self.num_pieces, self.dimension = self.A.shape
        if self.smooth_term is None:
            self.smooth_term = ZeroSmoothTerm(self.dimension)

        self.sigma2 = float(self.dual_sigma)
        self.D2 = (
            float(self.dual_diameter)
            if self.dual_diameter is not None
            else (math.log(self.num_pieces) if self.num_pieces > 1 else 0.0)
        )
        if self.operator_norm_override is None:
            self.operator_norm = float(np.max(np.linalg.norm(self.A, axis=1)))
        else:
            if self.operator_norm_override < 0.0:
                raise ValueError("operator_norm_override must be non-negative.")
            self.operator_norm = float(self.operator_norm_override)

        self.smooth_lipschitz = float(self.smooth_term.lipschitz)
        if self.num_pieces == 1:
            self.lipschitz = self.smooth_lipschitz
        else:
            self.lipschitz = self.smooth_lipschitz + (self.operator_norm ** 2) / (self.mu * self.sigma2)

    def oracle(self, x: ArrayLike) -> OracleState:
        candidate = vector(x, name="x")
        if candidate.size != self.dimension:
            raise ValueError("x has the wrong dimension for this objective.")

        smooth_value, smooth_grad = self.smooth_term.value_grad(candidate)
        smooth_grad = vector(smooth_grad, name="smooth_grad")
        if smooth_grad.size != self.dimension:
            raise ValueError("smooth_grad has the wrong dimension.")

        scores = self.A @ candidate + self.b
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

        dual = vector(dual_variable, name="dual_variable")
        if dual.size != self.num_pieces:
            raise ValueError("dual_variable has the wrong dimension.")

        reduced_gradient = self.A.T @ dual + linear_term
        try:
            primal_argmin = domain.linear_minimizer(reduced_gradient)
        except ValueError:
            return None
        return float(self.b @ dual + reduced_gradient @ primal_argmin)
