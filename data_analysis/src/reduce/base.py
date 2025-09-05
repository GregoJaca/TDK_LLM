# STATUS: DONE
from typing import Protocol
import numpy as np

class Reducer(Protocol):
    def fit(self, X: np.ndarray):
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        ...
