from __future__ import annotations

import numpy as np

from nesterov_smoothing_lib import *  # noqa: F401,F403
from nesterov_smoothing_lib import __all__ as _public_api

__all__ = list(_public_api)


def _demo() -> None:
    A = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    result = solve_paper_matrix_game(A, epsilon=1.0e-2, check_frequency=10)

    print("Iterations :", result.iterations)
    print("Predicted  :", result.predicted_iterations)
    print("Primal x   :", result.x)
    print("Dual u     :", result.u)
    print("Gap        :", result.gap)


if __name__ == "__main__":
    _demo()
