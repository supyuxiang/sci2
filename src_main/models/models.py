from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from .base import BaseRegressor


class SklearnRegressor(BaseRegressor):
    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnRegressor":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)


def build_regressor(kind: str, params: Dict[str, Any], multi_output: bool) -> BaseRegressor:
    kind = (kind or "").lower()
    if kind in ("linear", "linear_regression"):
        base = LinearRegression(**{k: v for k, v in (params or {}).items() if v is not None})
    elif kind in ("rf", "random_forest", "random_forest_regressor"):
        base = RandomForestRegressor(**{k: v for k, v in (params or {}).items() if v is not None})
    else:
        raise ValueError(f"Unsupported model type: {kind}")

    if multi_output:
        return SklearnRegressor(MultiOutputRegressor(base))
    return SklearnRegressor(base)

