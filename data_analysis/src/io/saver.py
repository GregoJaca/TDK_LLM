# STATUS: DONE
import torch
import numpy as np

def save_tensor(tensor: torch.Tensor, path: str):
    torch.save(tensor, path)

def save_npz(data: dict, path: str):
    np.savez(path, **data)
