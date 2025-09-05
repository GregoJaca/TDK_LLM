# STATUS: DONE
from typing import Protocol, Tuple, Optional, Dict
import numpy as np

class Metric(Protocol):
    def compare_trajectories(
        self, 
        a: np.ndarray, 
        b: np.ndarray, 
        *, 
        return_timeseries: bool = True, 
        **kwargs
    ) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        """Compares two trajectories and returns a timeseries of distances and an aggregate dictionary."""
        ...
