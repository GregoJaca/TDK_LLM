# STATUS: PARTIAL
import numpy as np
from typing import Tuple, Optional, Dict
from config import CONFIG

def compare_trajectories(
    a: np.ndarray, 
    b: np.ndarray, 
    *, 
    return_timeseries: bool = True, 
    **kwargs
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Computes cosine distance between two trajectories, with optional shifts."""
    shifts = kwargs.get("shifts", CONFIG["metrics"]["cos"]["shifts"])
    
    # Ensure trajectories are normalized
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)

    all_distances = []
    for shift in shifts:
        if shift == 0:
            # No shift
            dist = 1 - np.sum(a * b, axis=-1)
            all_distances.append(dist)
            continue

        # Apply shift
        if shift > 0:
            shifted_a = a[:-shift]
            shifted_b = b[shift:]
        else: # shift < 0
            shifted_a = a[-shift:]
            shifted_b = b[:shift]
        
        dist = 1 - np.sum(shifted_a * shifted_b, axis=-1)
        
        # Pad to original length for consistent aggregation
        padded_dist = np.full(a.shape[0], np.nan)
        if shift > 0:
            padded_dist[shift:] = dist
        else:
            padded_dist[:len(dist)] = dist
        all_distances.append(padded_dist)

    # Aggregate across shifts (e.g., take the minimum distance at each timestep)
    # This part is a placeholder for a more configurable aggregation strategy
    final_distances = np.nanmin(np.array(all_distances), axis=0)

    aggregates = {
        "mean": np.nanmean(final_distances),
        "median": np.nanmedian(final_distances),
        "std": np.nanstd(final_distances)
    }

    timeseries = final_distances if return_timeseries else None
    return timeseries, aggregates
