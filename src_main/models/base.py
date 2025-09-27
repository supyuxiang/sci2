from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import joblib
import numpy as np


class BaseRegressor(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRegressor":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "BaseRegressor":
        return joblib.load(path)

