"""Transform the factor panel's decimal simple returns into additive log-return steps."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def log_return_steps(values: ArrayLike) -> NDArray[np.float64]:
    """For a cumulative log target, sum log(1+r) over future trading observations.

    Missing observations remain missing for the caller's history alignment. A simple
    return at or below -100%, or an infinite value, cannot produce a finite log return.
    """
    rows = np.asarray(values, dtype=np.float64)
    if np.any(rows <= -1.0) or np.any(np.isinf(rows)):
        raise ValueError("log_return history requires finite simple returns greater than -1")
    return np.log1p(rows)
