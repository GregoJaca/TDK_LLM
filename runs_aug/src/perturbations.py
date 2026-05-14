import numpy as np
import torch

def build_simplex(num_points: int) -> np.ndarray:
    if num_points < 2:
        raise ValueError("num_points must be >= 2 for a simplex")
    eye = np.eye(num_points, dtype=np.float32)
    ones = np.ones((num_points, num_points), dtype=np.float32) / num_points
    vertices = eye - ones
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    return vertices

def select_subspace_indices(
    total_dim: int,
    num_points: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_points > total_dim:
        raise ValueError(
            f"num_conditions ({num_points}) must be <= subspace_dim ({total_dim})"
        )
    if mode == "first":
        return np.arange(num_points, dtype=np.int64)
    if mode == "random":
        return rng.choice(total_dim, size=num_points, replace=False).astype(np.int64)
    raise ValueError(f"Unknown subspace_mode: {mode}")

def generate_simplex_perturbations(
    base_embeds: torch.Tensor,
    n_conditions: int,
    radius: float,
    subspace_mode: str,
    target_tokens: int = None,
    seed: int = 42
) -> torch.Tensor:
    """
    Generates N perturbed embeddings using a regular simplex.
    base_embeds: [1, seq_len, embed_dim]
    target_tokens: integer M. If provided, perturbations are only applied to the first M tokens. 
                   If None, perturbations are applied to all tokens.
    Returns: [n_conditions, seq_len, embed_dim]
    """
    device = base_embeds.device
    dtype = base_embeds.dtype
    
    seq_len = base_embeds.shape[1]
    embed_dim = base_embeds.shape[2]
    
    if target_tokens is None:
        target_tokens = seq_len
        
    target_tokens = min(target_tokens, seq_len)
    
    total_perturbable_dim = target_tokens * embed_dim
    
    # Use provided seed for reproducible permutations
    rng = np.random.default_rng(seed)
    
    simplex = build_simplex(n_conditions)
    # The subspace we perturb has exactly `n_conditions` dimensions
    subspace_indices = select_subspace_indices(total_perturbable_dim, n_conditions, subspace_mode, rng)
    
    index_tensor = torch.as_tensor(subspace_indices, device=device, dtype=torch.long)
    
    perturbed_embeds = []
    
    for i in range(n_conditions):
        row = torch.as_tensor(simplex[i], device=device, dtype=dtype) * radius
        
        # We create a delta for the perturbable part
        delta_perturbable = torch.zeros((total_perturbable_dim,), device=device, dtype=dtype)
        delta_perturbable.index_copy_(0, index_tensor, row)
        delta_perturbable = delta_perturbable.view(1, target_tokens, embed_dim)
        
        # If there are remaining tokens, their delta is 0
        if target_tokens < seq_len:
            delta_remaining = torch.zeros((1, seq_len - target_tokens, embed_dim), device=device, dtype=dtype)
            delta = torch.cat([delta_perturbable, delta_remaining], dim=1)
        else:
            delta = delta_perturbable
            
        perturbed_embeds.append(base_embeds + delta)
        
    return torch.cat(perturbed_embeds, dim=0)
