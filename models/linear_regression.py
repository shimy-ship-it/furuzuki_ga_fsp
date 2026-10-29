from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LinearRegressionPredictor:
    """Linear regression model with feature standardization."""

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.model: Optional[Pipeline] = None
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionPredictor":
        if X.shape[1] == 0:
            raise ValueError("Feature matrix must contain at least one feature")

        self.feature_names = X.columns.tolist()
        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("regression", LinearRegression(fit_intercept=self.fit_intercept))
        ])
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        return self.model.predict(X[self.feature_names])

    def details(self) -> Dict[str, Any]:
        if self.model is None:
            return {}

        regression = self.model.named_steps["regression"]
        return {
            "model_type": "linear_regression",
            "framework": "scikit-learn",
            "feature_count": len(self.feature_names),
            "fit_intercept": self.fit_intercept,
            "coefficients": regression.coef_.astype(float).tolist(),
            "feature_coefficients": dict(zip(self.feature_names, regression.coef_.astype(float).tolist())),
            "intercept": float(regression.intercept_) if hasattr(regression, "intercept_") else None
        }
