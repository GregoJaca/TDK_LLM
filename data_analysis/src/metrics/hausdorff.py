# STATUS: DONE
import numpy as np
from typing import Tuple, Optional, Dict
from scipy.spatial.distance import directed_hausdorff

def compare_trajectories(
    a: np.ndarray, 
    b: np.ndarray, 
    *, 
    return_timeseries: bool = True, 
    **kwargs
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Computes the symmetric Hausdorff distance between two trajectories."""
    d_ab = directed_hausdorff(a, b)[0]
    d_ba = directed_hausdorff(b, a)[0]
    symmetric_d = max(d_ab, d_ba)

    aggregates = {"hausdorff_distance": symmetric_d}

    # Hausdorff is a scalar metric, so no timeseries is returned.
    # We could potentially return per-point nearest distances if needed for viz.
    timeseries = None

    return timeseries, aggregates
