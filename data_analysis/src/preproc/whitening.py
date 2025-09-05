# STATUS: DONE
import numpy as np
from config import CONFIG

def fit_whitening(X_flat: np.ndarray) -> dict:
    # Ensure input is 2D
    if X_flat.ndim != 2:
        raise ValueError(f"Input must be 2D (samples, features), but got {X_flat.shape}")

    # 1. Mean center the data
    mean = np.mean(X_flat, axis=0)
    X_centered = X_flat - mean

    # 2. Compute covariance matrix
    cov = np.cov(X_centered, rowvar=False)

    # 3. Eigen-decomposition of the covariance matrix
    # Use eigh for symmetric matrices
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 4. Compute the whitening matrix
    eps = CONFIG['reduction']['whiten']['eps']
    diag_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))
    W_inv_sqrt = eigvecs @ diag_inv_sqrt @ eigvecs.T

    return {"mean": mean, "W_inv_sqrt": W_inv_sqrt}

def apply_whitening(X: np.ndarray, whiten_params: dict) -> np.ndarray:
    # Input can be N-D, but last dimension must match feature dimension
    if X.shape[-1] != whiten_params["mean"].shape[0]:
        raise ValueError("Last dimension of X must match feature dimension of whitening parameters.")

    # Center and then apply whitening matrix
    X_centered = X - whiten_params["mean"]
    X_whitened = X_centered @ whiten_params["W_inv_sqrt"]

    return X_whitened
