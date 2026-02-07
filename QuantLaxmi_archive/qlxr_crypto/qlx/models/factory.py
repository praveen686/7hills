"""Model factory — config-driven pipeline construction.

One function, one source of truth.  No three-way copy-paste divergence.
The pipeline always includes imputation and scaling; the model is selected
by name from a registry.  Unknown model names raise immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_REGRESSION_MODELS: dict[str, type] = {
    "xgboost": XGBRegressor,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "linear": LinearRegression,
    "svr": SVR,
    "random_forest": RandomForestRegressor,
}

_CLASSIFICATION_MODELS: dict[str, type] = {
    "xgboost": XGBClassifier,
    "logistic": LogisticRegression,
    "svc": SVC,
    "random_forest": RandomForestClassifier,
}


def make_pipeline(
    model_name: str = "xgboost",
    task: str = "regression",
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Create an sklearn Pipeline: Imputer -> Scaler -> Model.

    Parameters
    ----------
    model_name : str
        Key into the model registry (e.g., "xgboost", "ridge").
    task : str
        "regression" or "classification".
    model_params : dict, optional
        Keyword arguments forwarded to the model constructor.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    model_params = model_params or {}

    registry = _REGRESSION_MODELS if task == "regression" else _CLASSIFICATION_MODELS

    if model_name not in registry:
        valid = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown {task} model '{model_name}'. Valid options: {valid}"
        )

    model_cls = registry[model_name]

    # Sensible defaults for XGBoost
    if model_name == "xgboost" and task == "regression":
        model_params.setdefault("objective", "reg:squarederror")
        model_params.setdefault("n_estimators", 200)

    model = model_cls(**model_params)

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


@dataclass(frozen=True)
class ModelFactory:
    """Frozen config object that can produce fresh pipelines on demand.

    Useful in walk-forward CV where each fold needs a fresh (unfitted)
    pipeline with identical hyperparameters.
    """

    model_name: str = "xgboost"
    task: str = "regression"
    model_params: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Pipeline:
        return make_pipeline(self.model_name, self.task, dict(self.model_params))
