from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from .common import FloatArray, vector


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
        self.coefficients = vector(self.coefficients, name="coefficients")

    def value_grad(self, x: FloatArray) -> tuple[float, FloatArray]:
        if x.size != self.coefficients.size:
            raise ValueError("x has the wrong dimension for this linear term.")
        return float(self.coefficients @ x), self.coefficients.copy()
