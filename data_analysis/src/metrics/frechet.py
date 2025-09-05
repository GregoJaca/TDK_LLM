# STATUS: PARTIAL
import numpy as np
from typing import Tuple, Optional, Dict

try:
    from frechetdist import frdist
    FRECHETDIST_AVAILABLE = True
except ImportError:
    FRECHETDIST_AVAILABLE = False

def _discrete_frechet(a, b):
    """A pure Python implementation of discrete Fréchet distance."""
    ca = np.full((len(a), len(b)), -1.0)

    def c(i, j):
        if ca[i, j] > -1.0:
            return ca[i, j]
        
        dist = np.linalg.norm(a[i] - b[j])
        if i == 0 and j == 0:
            ca[i, j] = dist
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), dist)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), dist)
        elif i > 0 and j > 0:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), dist)
        else:
            ca[i, j] = float("inf")
        return ca[i, j]

    return c(len(a) - 1, len(b) - 1)

def compare_trajectories(
    a: np.ndarray, 
    b: np.ndarray, 
    *, 
    return_timeseries: bool = True, 
    **kwargs
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Computes the Fréchet distance between two trajectories."""
    if FRECHETDIST_AVAILABLE:
        distance = frdist(a, b)
    else:
        # Using the pure Python fallback
        distance = _discrete_frechet(a, b)

    aggregates = {"frechet_distance": distance}

    # Fréchet is a scalar metric, so no timeseries is returned.
    timeseries = None

    return timeseries, aggregates
