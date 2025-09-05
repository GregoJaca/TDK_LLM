# STATUS: DONE
import torch

def load_tensor(path: str) -> torch.Tensor:
    return torch.load(path, weights_only=True, map_location=torch.device('cpu'))

def assert_shape(tensor, shape):
    # Basic shape assertion, can be expanded
    assert tensor.shape == shape, f"Tensor shape mismatch: expected {shape}, got {tensor.shape}"
