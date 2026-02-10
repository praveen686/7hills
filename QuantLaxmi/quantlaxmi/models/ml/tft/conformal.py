"""Split conformal prediction for TFT trading signals.

Provides calibrated prediction intervals without distributional assumptions.
Uses the inductive (split) conformal method:
  1. Train model on training set
  2. Compute nonconformity scores on calibration set
  3. Use quantile of scores for prediction intervals

The conformal predictor wraps around any point-prediction model and produces
valid coverage guarantees under exchangeability. For financial time series
this means coverage is approximate but still useful as an uncertainty filter.

References
----------
- Vovk, Gammerman, Shafer (2005), "Algorithmic Learning in a Random World"
- Lei et al. (2018), "Distribution-Free Predictive Inference for Regression"
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """Split conformal prediction for TFT trading signals.

    Provides calibrated prediction intervals without distributional assumptions.
    Uses the inductive (split) conformal method:
      1. Train model on training set
      2. Compute nonconformity scores on calibration set
      3. Use quantile of scores for prediction intervals

    Parameters
    ----------
    alpha : float
        Miscoverage rate. 0.1 means 90% nominal coverage.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.calibration_scores: Optional[np.ndarray] = None

    @property
    def is_calibrated(self) -> bool:
        """Whether calibrate() has been called with valid data."""
        return self.calibration_scores is not None and len(self.calibration_scores) > 0

    @property
    def quantile_threshold(self) -> float:
        """The conformal quantile q used for intervals.

        q = quantile(scores, ceil((n+1)(1-alpha)) / n)
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before accessing quantile_threshold")
        n = len(self.calibration_scores)
        # Finite-sample corrected quantile level
        level = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)  # clamp to valid quantile range
        return float(np.quantile(self.calibration_scores, level))

    def calibrate(self, predictions: np.ndarray, actuals: np.ndarray) -> None:
        """Compute nonconformity scores on calibration set.

        Score = |prediction - actual| (absolute residual).

        Parameters
        ----------
        predictions : np.ndarray
            Model predictions on the calibration set.
        actuals : np.ndarray
            True values on the calibration set.

        Raises
        ------
        ValueError
            If inputs have mismatched shapes or are empty.
        """
        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        actuals = np.asarray(actuals, dtype=np.float64).ravel()

        if len(predictions) != len(actuals):
            raise ValueError(
                f"Shape mismatch: predictions ({len(predictions)}) vs "
                f"actuals ({len(actuals)})"
            )
        if len(predictions) == 0:
            raise ValueError("Cannot calibrate with empty arrays")

        # Remove NaN pairs
        valid = ~(np.isnan(predictions) | np.isnan(actuals))
        predictions = predictions[valid]
        actuals = actuals[valid]

        if len(predictions) == 0:
            raise ValueError("No valid (non-NaN) calibration pairs")

        self.calibration_scores = np.abs(predictions - actuals)
        logger.info(
            "Conformal calibration: n=%d, median_score=%.6f, q(%.0f%%)=%.6f",
            len(self.calibration_scores),
            float(np.median(self.calibration_scores)),
            (1.0 - self.alpha) * 100,
            self.quantile_threshold,
        )

    def predict_interval(
        self, predictions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) prediction intervals.

        q = quantile(scores, ceil((n+1)(1-alpha))/n)
        interval = [pred - q, pred + q]

        Parameters
        ----------
        predictions : np.ndarray
            Point predictions from the model.

        Returns
        -------
        lower : np.ndarray
            Lower bound of prediction interval.
        upper : np.ndarray
            Upper bound of prediction interval.
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before predict_interval()")

        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        q = self.quantile_threshold
        lower = predictions - q
        upper = predictions + q
        return lower, upper

    def interval_width(self) -> float:
        """Width of the prediction interval (2 * quantile_threshold)."""
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before interval_width()")
        return 2.0 * self.quantile_threshold

    def is_confident(self, prediction: float, threshold: float = 0.5) -> bool:
        """Check if prediction interval is narrow enough to be actionable.

        Returns True if interval width < threshold * |prediction|.
        Used to gate trading: only trade when model is confident.

        Parameters
        ----------
        prediction : float
            The point prediction value.
        threshold : float
            Maximum ratio of interval_width / |prediction|.

        Returns
        -------
        bool
            True if the model is confident enough to trade.
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before is_confident()")

        width = self.interval_width()
        abs_pred = abs(prediction)

        # If prediction is essentially zero, the interval is never "narrow enough"
        if abs_pred < 1e-12:
            return False

        return width < threshold * abs_pred

    def coverage_score(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> float:
        """Compute empirical coverage on a test set.

        Parameters
        ----------
        predictions : np.ndarray
            Point predictions.
        actuals : np.ndarray
            True values.

        Returns
        -------
        float
            Fraction of actuals falling within predicted intervals.
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before coverage_score()")

        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        actuals = np.asarray(actuals, dtype=np.float64).ravel()

        lower, upper = self.predict_interval(predictions)
        covered = (actuals >= lower) & (actuals <= upper)
        return float(np.mean(covered))

    def adaptive_alpha(
        self,
        recent_coverage: float,
        target: float = 0.9,
        lr: float = 0.01,
    ) -> None:
        """Online alpha adjustment to maintain target coverage.

        If recent coverage < target, increase alpha (wider intervals).
        If recent coverage > target, decrease alpha (tighter intervals).

        This implements a simple online recalibration:
          alpha_new = alpha - lr * (recent_coverage - target)

        Parameters
        ----------
        recent_coverage : float
            Observed coverage fraction on recent data (e.g., last 100 predictions).
        target : float
            Desired coverage level (default 0.9 = 90%).
        lr : float
            Learning rate for alpha adjustment.
        """
        if not 0.0 <= recent_coverage <= 1.0:
            raise ValueError(
                f"recent_coverage must be in [0, 1], got {recent_coverage}"
            )

        # alpha = miscoverage rate: higher alpha -> more miscoverage -> narrower intervals
        # If coverage < target: we need wider intervals -> decrease alpha
        # If coverage > target: we can tighten intervals -> increase alpha
        # Formula: alpha_new = alpha + lr * (coverage - target)
        #   coverage < target => negative shift => alpha decreases => wider intervals
        #   coverage > target => positive shift => alpha increases => tighter intervals
        delta = lr * (recent_coverage - target)
        new_alpha = self.alpha + delta
        # Clamp to valid range
        new_alpha = max(0.001, min(0.999, new_alpha))

        logger.debug(
            "Adaptive alpha: %.4f -> %.4f (coverage=%.3f, target=%.3f)",
            self.alpha, new_alpha, recent_coverage, target,
        )
        self.alpha = new_alpha
