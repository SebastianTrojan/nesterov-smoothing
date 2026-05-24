from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from .common import FloatArray, vector
from .geometry import _project_to_simplex, _simplex_entropy_argmin, _simplex_l1_squared_step


class ProjectedDomain(Protocol):
    center: FloatArray
    prox_diameter: float
    sigma: float

    def project(self, x: ArrayLike) -> FloatArray:
        ...

    def linear_minimizer(self, gradient: ArrayLike) -> FloatArray:
        ...


@dataclass
class EntropySimplexDomain:
    dimension: int
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")
        self.center = np.full(self.dimension, 1.0 / self.dimension, dtype=np.float64)
        self.prox_diameter = math.log(self.dimension) if self.dimension > 1 else 0.0

    def project(self, x: ArrayLike) -> FloatArray:
        candidate = vector(x, name="x")
        if candidate.size != self.dimension:
            raise ValueError("x has the wrong dimension for this simplex.")
        return _project_to_simplex(candidate)

    def linear_minimizer(self, gradient: ArrayLike) -> FloatArray:
        grad = vector(gradient, name="gradient")
        if grad.size != self.dimension:
            raise ValueError("gradient has the wrong dimension for this simplex.")
        result = np.zeros(self.dimension, dtype=np.float64)
        result[int(grad.argmin())] = 1.0
        return result

    def prox_gradient_step(self, x_bar: ArrayLike, gradient: ArrayLike, L: float) -> FloatArray:
        return _simplex_l1_squared_step(x_bar, gradient, L)

    def accumulated_model_minimizer(self, gradient_sum: ArrayLike, L: float) -> FloatArray:
        grad = vector(gradient_sum, name="gradient_sum")
        if grad.size != self.dimension:
            raise ValueError("gradient_sum has the wrong dimension for this simplex.")
        return _simplex_entropy_argmin(grad, L)
