# STATUS: DONE
import torch
import numpy as np

def l2_normalize_per_timestep(x):
    if isinstance(x, np.ndarray):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    elif isinstance(x, torch.Tensor):
        return x / torch.linalg.norm(x, dim=-1, keepdim=True)
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")

def mean_subtract_global(x):
    if isinstance(x, np.ndarray):
        return x - x.mean(axis=tuple(range(x.ndim - 1)), keepdims=True)
    elif isinstance(x, torch.Tensor):
        return x - x.mean(dim=tuple(range(x.ndim - 1)), keepdim=True)
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")
