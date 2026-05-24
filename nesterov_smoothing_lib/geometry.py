from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .common import FloatArray, vector


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
