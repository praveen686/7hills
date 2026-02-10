"""Ensemble of TFT models for variance reduction and uncertainty estimation.

Trains multiple TFT models with different random seeds or data splits,
then combines predictions via weighted averaging. Prediction disagreement
among ensemble members provides a natural measure of epistemic uncertainty.

Can be combined with :class:`ConformalPredictor` for calibrated intervals
around the ensemble mean.

References
----------
- Lakshminarayanan et al. (2017), "Simple and Scalable Predictive Uncertainty
  Estimation using Deep Ensembles"
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TFTEnsemble:
    """Ensemble of TFT models trained on different data splits/seeds.

    Reduces variance and provides uncertainty via prediction disagreement.

    Parameters
    ----------
    n_models : int
        Number of models in the ensemble.
    temperature : float
        Temperature for softmax weight computation from Sharpe ratios.
        Higher temperature = more uniform weights.
    """

    def __init__(self, n_models: int = 5, temperature: float = 1.0) -> None:
        if n_models < 1:
            raise ValueError(f"n_models must be >= 1, got {n_models}")
        self.n_models = n_models
        self.temperature = temperature
        self.models: list[Any] = []
        self.weights: Optional[np.ndarray] = None

    @property
    def is_trained(self) -> bool:
        """Whether the ensemble has been trained."""
        return len(self.models) == self.n_models

    def train_ensemble(
        self,
        train_fn: Callable[..., Any],
        train_data: Any,
        seeds: Optional[list[int]] = None,
    ) -> None:
        """Train n_models with different random seeds.

        Parameters
        ----------
        train_fn : callable
            Function that takes (data, seed) and returns a trained model.
            The model must support a predict(X) method returning np.ndarray.
        train_data : Any
            Data to pass to train_fn (can be any format your train_fn accepts).
        seeds : list[int], optional
            Random seeds for each model. If None, uses [42, 137, 256, 512, 1024, ...].
        """
        if seeds is None:
            seeds = [42 + i * 95 for i in range(self.n_models)]
        elif len(seeds) != self.n_models:
            raise ValueError(
                f"Expected {self.n_models} seeds, got {len(seeds)}"
            )

        self.models = []
        for i, seed in enumerate(seeds):
            logger.info("Training ensemble member %d/%d (seed=%d)", i + 1, self.n_models, seed)
            model = train_fn(train_data, seed)
            self.models.append(model)

        # Initialize uniform weights
        self.weights = np.ones(self.n_models) / self.n_models
        logger.info(
            "Ensemble training complete: %d models, uniform weights", self.n_models
        )

    def predict(self, X: Any) -> dict[str, Any]:
        """Ensemble prediction with uncertainty estimates.

        Parameters
        ----------
        X : Any
            Input data. Must be compatible with each model's predict() method.

        Returns
        -------
        dict with keys:
            'mean' : np.ndarray — weighted average of predictions
            'std' : np.ndarray — std of predictions (epistemic uncertainty)
            'predictions' : list[np.ndarray] — individual model predictions
            'disagreement' : np.ndarray — max - min of predictions
        """
        if not self.is_trained:
            raise RuntimeError(
                f"Ensemble not fully trained: {len(self.models)}/{self.n_models} models"
            )

        weights = self.weights if self.weights is not None else np.ones(self.n_models) / self.n_models

        # Collect predictions from all models
        all_preds = []
        for model in self.models:
            pred = np.asarray(model.predict(X), dtype=np.float64).ravel()
            all_preds.append(pred)

        # Stack: (n_models, n_samples)
        preds_stack = np.stack(all_preds, axis=0)

        # Weighted mean: (n_samples,)
        mean = np.average(preds_stack, axis=0, weights=weights)

        # Standard deviation across models (epistemic uncertainty)
        std = np.std(preds_stack, axis=0, ddof=1) if self.n_models > 1 else np.zeros_like(mean)

        # Disagreement: range of predictions
        disagreement = np.max(preds_stack, axis=0) - np.min(preds_stack, axis=0)

        return {
            "mean": mean,
            "std": std,
            "predictions": all_preds,
            "disagreement": disagreement,
        }

    def update_weights(
        self,
        recent_returns: np.ndarray,
    ) -> None:
        """Update model weights based on recent OOS Sharpe ratios.

        Exponentially weighted: w_i = exp(sharpe_i / temperature) / Z

        Parameters
        ----------
        recent_returns : np.ndarray
            Array of shape (n_models, n_days) containing strategy returns
            for each model on recent OOS data.
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained; cannot update weights")

        recent_returns = np.asarray(recent_returns, dtype=np.float64)
        if recent_returns.ndim == 1:
            # Single model case or already aggregated — can't compute per-model
            raise ValueError(
                "recent_returns must be 2D: (n_models, n_days)"
            )

        if recent_returns.shape[0] != self.n_models:
            raise ValueError(
                f"Expected {self.n_models} rows in recent_returns, "
                f"got {recent_returns.shape[0]}"
            )

        # Compute Sharpe for each model
        sharpes = np.zeros(self.n_models)
        for i in range(self.n_models):
            rets = recent_returns[i]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 2:
                sharpes[i] = 0.0
            else:
                mean_r = np.mean(valid)
                std_r = np.std(valid, ddof=1)
                sharpes[i] = (mean_r / max(std_r, 1e-8)) * math.sqrt(252)

        # Softmax weights: w_i = exp(sharpe_i / T) / sum(exp(sharpe_j / T))
        # Subtract max for numerical stability
        scaled = sharpes / max(self.temperature, 1e-8)
        scaled -= np.max(scaled)
        exp_weights = np.exp(scaled)
        self.weights = exp_weights / np.sum(exp_weights)

        logger.info(
            "Updated ensemble weights: sharpes=%s, weights=%s",
            np.round(sharpes, 3).tolist(),
            np.round(self.weights, 4).tolist(),
        )

    def predict_with_conformal(
        self,
        X: Any,
        conformal: "ConformalPredictor",
    ) -> dict[str, Any]:
        """Combined ensemble + conformal prediction.

        Returns ensemble mean + conformal intervals around it.

        Parameters
        ----------
        X : Any
            Input data.
        conformal : ConformalPredictor
            A calibrated conformal predictor.

        Returns
        -------
        dict with keys from predict() plus:
            'lower' : np.ndarray — conformal lower bound
            'upper' : np.ndarray — conformal upper bound
            'interval_width' : float — width of conformal interval
        """
        from quantlaxmi.models.ml.tft.conformal import ConformalPredictor

        if not conformal.is_calibrated:
            raise RuntimeError("ConformalPredictor must be calibrated first")

        result = self.predict(X)
        lower, upper = conformal.predict_interval(result["mean"])

        result["lower"] = lower
        result["upper"] = upper
        result["interval_width"] = conformal.interval_width()

        return result
