# STATUS: PARTIAL
import numpy as np
from typing import Tuple, Optional, Dict
from scipy.spatial.distance import euclidean

try:
    from fastdtw import fastdtw
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False

try:
    from tslearn.metrics import dtw as tslearn_dtw
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False

def compare_trajectories(
    a: np.ndarray, 
    b: np.ndarray, 
    *, 
    return_timeseries: bool = True, 
    **kwargs
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Computes Dynamic Time Warping distance between two trajectories."""
    window = kwargs.get("window", 1)
    exact = kwargs.get("exact", False)

    if exact and TSLEARN_AVAILABLE:
        distance = tslearn_dtw(a, b)
        path = None # tslearn dtw does not return path by default
    elif FASTDTW_AVAILABLE:
        distance, path = fastdtw(a, b, radius=window, dist=euclidean)
    else:
        raise ImportError("Neither fastdtw nor tslearn is installed. Please install one to use DTW.")

    aggregates = {"dtw_distance": distance}

    if return_timeseries and path is not None:
        # Create a timeseries of distances along the alignment path
        timeseries = np.array([euclidean(a[i], b[j]) for i, j in path])
    else:
        timeseries = None

    return timeseries, aggregates
