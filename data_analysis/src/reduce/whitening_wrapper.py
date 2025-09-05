# STATUS: DONE
import numpy as np
from src.reduce.base import Reducer
from src.preproc.whitening import fit_whitening, apply_whitening

class WhiteningReducer(Reducer):
    def __init__(self):
        self.params = None

    def fit(self, X: np.ndarray):
        X_flat = X.reshape(-1, X.shape[-1])
        self.params = fit_whitening(X_flat)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Fit must be called before transform.")
        return apply_whitening(X, self.params)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
