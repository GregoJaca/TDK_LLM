# STATUS: DONE
from typing import List, Tuple

def get_pairs(n_trajectories: int, mode: str, custom_pairs: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """Generates a list of pairs of trajectory indices to compare."""
    if mode == "ref0":
        if n_trajectories < 2:
            return []
        return [(0, i) for i in range(1, n_trajectories)]
    elif mode == "all":
        return [(i, j) for i in range(n_trajectories) for j in range(i + 1, n_trajectories)]
    elif mode == "custom":
        if custom_pairs is None:
            raise ValueError("Custom pairs must be provided for 'custom' mode.")
        return custom_pairs
    else:
        raise ValueError(f"Unknown pairing mode: {mode}")
