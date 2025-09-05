# STATUS: PARTIAL
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from config import CONFIG
from src.reduce.base import Reducer

def fit_pca(X_flat, r):
    use_gpu = CONFIG["reduction"]["pca"]["use_gpu"] and torch.cuda.is_available()
    if use_gpu:
        # GPU SVD path
        try:
            device = torch.device("cuda")
            X_gpu = torch.from_numpy(X_flat).to(device)
            U, S, Vh = torch.linalg.svd(X_gpu, full_matrices=False)
            components = Vh[:r, :]
            mean = torch.mean(X_gpu, dim=0)
            # Calculate explained_variance_ratio for GPU SVD
            explained_variance_ratio = (S**2 / torch.sum(S**2)).cpu().numpy()
            return {"components": components.cpu().numpy(), "mean": mean.cpu().numpy(), "type": "gpu", "explained_variance_ratio": explained_variance_ratio}
        except Exception as e:
            print(f"GPU PCA failed with error: {e}. Falling back to CPU.")

    # CPU path (IncrementalPCA)
    pca = IncrementalPCA(n_components=r)
    pca.fit(X_flat)
    return {"components": pca.components_, "mean": pca.mean_, "type": "cpu", "model": pca, "explained_variance_ratio": pca.explained_variance_ratio_}

def transform_pca(X, pca_model):
    X_centered = X - pca_model["mean"]
    return X_centered @ pca_model["components"].T

class PCAReducer(Reducer):
    def __init__(self, r):
        self.r = r
        self.model = None

    def fit(self, X: np.ndarray):
        X_flat = X.reshape(-1, X.shape[-1])
        self.model = fit_pca(X_flat, self.r)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Fit must be called before transform.")
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_reduced_flat = transform_pca(X_flat, self.model)
        return X_reduced_flat.reshape(*original_shape[:-1], self.r)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
