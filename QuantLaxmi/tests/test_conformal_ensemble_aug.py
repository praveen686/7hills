"""Tests for conformal prediction, TFT ensemble, and time series augmentation.

Covers:
- ConformalPredictor: calibration, interval coverage, adaptive alpha, is_confident
- TFTEnsemble: training, prediction, weight updates, conformal integration
- TimeSeriesAugmenter: jitter, scaling, magnitude_warping, window_slicing, mixup, augment_batch
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantlaxmi.models.ml.tft.conformal import ConformalPredictor
from quantlaxmi.models.ml.tft.ensemble import TFTEnsemble
from quantlaxmi.models.ml.tft.augmentation import TimeSeriesAugmenter


# ============================================================================
# Helpers
# ============================================================================


class _MockModel:
    """Minimal model for ensemble testing."""

    def __init__(self, bias: float = 0.0, seed: int = 42):
        self.bias = bias
        self.rng = np.random.default_rng(seed)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Return mean of features + bias
        if X.ndim == 1:
            return np.array([np.mean(X) + self.bias])
        return np.mean(X, axis=tuple(range(1, X.ndim))) + self.bias


def _mock_train_fn(data, seed):
    """Train function that returns a MockModel with seed-dependent bias."""
    return _MockModel(bias=seed * 0.001, seed=seed)


# ============================================================================
# Conformal Predictor Tests
# ============================================================================


class TestConformalPredictor:
    """Tests for ConformalPredictor."""

    def test_calibration_basic(self):
        """calibrate() stores nonconformity scores."""
        cp = ConformalPredictor(alpha=0.1)
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([1.1, 1.8, 3.2, 3.9, 5.3])
        cp.calibrate(preds, actual)
        assert cp.is_calibrated
        assert len(cp.calibration_scores) == 5
        np.testing.assert_array_almost_equal(
            cp.calibration_scores, np.abs(preds - actual)
        )

    def test_calibration_rejects_empty(self):
        """calibrate() raises on empty arrays."""
        cp = ConformalPredictor(alpha=0.1)
        with pytest.raises(ValueError, match="empty"):
            cp.calibrate(np.array([]), np.array([]))

    def test_calibration_rejects_shape_mismatch(self):
        """calibrate() raises on mismatched shapes."""
        cp = ConformalPredictor(alpha=0.1)
        with pytest.raises(ValueError, match="Shape mismatch"):
            cp.calibrate(np.array([1, 2, 3]), np.array([1, 2]))

    def test_interval_width_symmetric(self):
        """Prediction intervals are symmetric around the prediction."""
        cp = ConformalPredictor(alpha=0.1)
        preds = np.arange(100, dtype=float)
        actual = preds + np.random.default_rng(42).normal(0, 1, 100)
        cp.calibrate(preds, actual)

        test_preds = np.array([10.0, 20.0, 30.0])
        lower, upper = cp.predict_interval(test_preds)

        # Interval should be symmetric: pred - q, pred + q
        widths = upper - lower
        np.testing.assert_array_almost_equal(widths, widths[0] * np.ones(3))
        # Center should be the prediction
        centers = (upper + lower) / 2
        np.testing.assert_array_almost_equal(centers, test_preds)

    def test_coverage_approximately_correct(self):
        """Empirical coverage should be approximately 1-alpha on well-behaved data."""
        rng = np.random.default_rng(42)
        n_cal = 500
        n_test = 1000
        alpha = 0.1

        # Gaussian errors: predictions + N(0,1) noise
        cal_preds = rng.normal(0, 1, n_cal)
        cal_errors = rng.normal(0, 1, n_cal)
        cal_actual = cal_preds + cal_errors

        test_preds = rng.normal(0, 1, n_test)
        test_errors = rng.normal(0, 1, n_test)
        test_actual = test_preds + test_errors

        cp = ConformalPredictor(alpha=alpha)
        cp.calibrate(cal_preds, cal_actual)

        coverage = cp.coverage_score(test_preds, test_actual)
        # Coverage should be close to 1-alpha = 0.9
        # With 1000 test points, we allow some slack
        assert coverage >= 0.85, f"Coverage {coverage} too low (expected ~0.9)"
        assert coverage <= 0.98, f"Coverage {coverage} too high (expected ~0.9)"

    def test_is_confident_large_prediction(self):
        """is_confident returns True when prediction is large relative to interval."""
        cp = ConformalPredictor(alpha=0.1)
        # Small residuals -> narrow interval
        preds = np.ones(100)
        actual = preds + np.random.default_rng(42).normal(0, 0.01, 100)
        cp.calibrate(preds, actual)

        # Large prediction, narrow interval -> confident
        assert cp.is_confident(100.0, threshold=0.5)

    def test_is_confident_small_prediction(self):
        """is_confident returns False when prediction is small relative to interval."""
        cp = ConformalPredictor(alpha=0.1)
        # Large residuals -> wide interval
        preds = np.ones(100)
        actual = preds + np.random.default_rng(42).normal(0, 5.0, 100)
        cp.calibrate(preds, actual)

        # Small prediction, wide interval -> not confident
        assert not cp.is_confident(0.01, threshold=0.5)

    def test_is_confident_zero_prediction(self):
        """is_confident returns False for zero prediction."""
        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(np.ones(50), np.ones(50) + 0.1)
        assert not cp.is_confident(0.0, threshold=0.5)

    def test_adaptive_alpha_increases_when_undercovered(self):
        """adaptive_alpha decreases alpha (wider intervals) when coverage is below target."""
        cp = ConformalPredictor(alpha=0.1)
        original_alpha = cp.alpha
        # Coverage below target -> alpha should decrease
        cp.adaptive_alpha(recent_coverage=0.80, target=0.90, lr=0.1)
        assert cp.alpha < original_alpha, "Alpha should decrease when coverage is below target"

    def test_adaptive_alpha_decreases_when_overcovered(self):
        """adaptive_alpha increases alpha (tighter intervals) when coverage is above target."""
        cp = ConformalPredictor(alpha=0.3)
        original_alpha = cp.alpha
        # Coverage above target -> alpha should increase
        cp.adaptive_alpha(recent_coverage=0.98, target=0.90, lr=0.1)
        assert cp.alpha > original_alpha, "Alpha should increase when coverage is above target"

    def test_adaptive_alpha_clamped(self):
        """adaptive_alpha never pushes alpha outside (0, 1)."""
        cp = ConformalPredictor(alpha=0.01)
        # Aggressive decrease
        cp.adaptive_alpha(recent_coverage=0.0, target=0.9, lr=1.0)
        assert 0.0 < cp.alpha < 1.0

        cp2 = ConformalPredictor(alpha=0.99)
        cp2.adaptive_alpha(recent_coverage=1.0, target=0.9, lr=1.0)
        assert 0.0 < cp2.alpha < 1.0

    def test_invalid_alpha_raises(self):
        """ConformalPredictor rejects alpha outside (0, 1)."""
        with pytest.raises(ValueError):
            ConformalPredictor(alpha=0.0)
        with pytest.raises(ValueError):
            ConformalPredictor(alpha=1.0)
        with pytest.raises(ValueError):
            ConformalPredictor(alpha=-0.1)

    def test_predict_interval_before_calibration_raises(self):
        """predict_interval() before calibrate() raises RuntimeError."""
        cp = ConformalPredictor(alpha=0.1)
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict_interval(np.array([1.0]))

    def test_calibration_handles_nans(self):
        """calibrate() skips NaN pairs."""
        cp = ConformalPredictor(alpha=0.1)
        preds = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        actual = np.array([1.1, 2.0, np.nan, 3.9, np.nan])
        cp.calibrate(preds, actual)
        # Only (1.0, 1.1) and (4.0, 3.9) are valid
        assert len(cp.calibration_scores) == 2


# ============================================================================
# TFTEnsemble Tests
# ============================================================================


class TestTFTEnsemble:
    """Tests for TFTEnsemble."""

    def test_train_ensemble_creates_models(self):
        """train_ensemble() populates models list."""
        ens = TFTEnsemble(n_models=3)
        ens.train_ensemble(_mock_train_fn, None, seeds=[10, 20, 30])
        assert len(ens.models) == 3
        assert ens.is_trained

    def test_predict_mean_equals_average(self):
        """Ensemble mean should equal weighted average of individual predictions."""
        ens = TFTEnsemble(n_models=3)
        ens.train_ensemble(_mock_train_fn, None, seeds=[0, 100, 200])

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ens.predict(X)

        # With uniform weights, mean should be arithmetic average
        individual = np.stack(result["predictions"], axis=0)
        expected_mean = np.mean(individual, axis=0)
        np.testing.assert_array_almost_equal(result["mean"], expected_mean)

    def test_predict_std_positive(self):
        """Ensemble std should be > 0 when models disagree."""
        ens = TFTEnsemble(n_models=5)
        # Use different seeds so models have different biases
        ens.train_ensemble(_mock_train_fn, None, seeds=[0, 100, 200, 300, 400])

        X = np.array([[1.0, 2.0]])
        result = ens.predict(X)
        assert np.all(result["std"] > 0), "Std should be positive with diverse models"

    def test_predict_returns_all_keys(self):
        """predict() returns dict with all expected keys."""
        ens = TFTEnsemble(n_models=2)
        ens.train_ensemble(_mock_train_fn, None, seeds=[1, 2])

        X = np.array([[1.0]])
        result = ens.predict(X)

        assert "mean" in result
        assert "std" in result
        assert "predictions" in result
        assert "disagreement" in result
        assert len(result["predictions"]) == 2

    def test_weights_sum_to_one(self):
        """After training, weights sum to 1."""
        ens = TFTEnsemble(n_models=4)
        ens.train_ensemble(_mock_train_fn, None, seeds=[10, 20, 30, 40])
        np.testing.assert_almost_equal(np.sum(ens.weights), 1.0)

    def test_update_weights_favors_higher_sharpe(self):
        """update_weights gives more weight to the model with higher Sharpe."""
        ens = TFTEnsemble(n_models=3, temperature=1.0)
        ens.train_ensemble(_mock_train_fn, None, seeds=[1, 2, 3])

        # Model 0: high Sharpe (positive consistent returns)
        # Model 1: moderate Sharpe
        # Model 2: negative Sharpe
        rng = np.random.default_rng(42)
        returns = np.zeros((3, 100))
        returns[0] = 0.01 + rng.normal(0, 0.005, 100)  # high Sharpe
        returns[1] = 0.002 + rng.normal(0, 0.01, 100)   # moderate
        returns[2] = -0.005 + rng.normal(0, 0.01, 100)  # negative

        ens.update_weights(returns)

        assert ens.weights[0] > ens.weights[1], "Higher Sharpe model should get more weight"
        assert ens.weights[1] > ens.weights[2], "Moderate Sharpe > negative Sharpe"
        np.testing.assert_almost_equal(np.sum(ens.weights), 1.0)

    def test_update_weights_rejects_1d(self):
        """update_weights rejects 1D array."""
        ens = TFTEnsemble(n_models=2)
        ens.train_ensemble(_mock_train_fn, None, seeds=[1, 2])
        with pytest.raises(ValueError, match="2D"):
            ens.update_weights(np.array([0.01, 0.02, 0.03]))

    def test_predict_before_training_raises(self):
        """predict() before train_ensemble() raises RuntimeError."""
        ens = TFTEnsemble(n_models=3)
        with pytest.raises(RuntimeError, match="not fully trained"):
            ens.predict(np.array([[1.0]]))

    def test_predict_with_conformal(self):
        """predict_with_conformal returns ensemble + conformal keys."""
        ens = TFTEnsemble(n_models=3)
        ens.train_ensemble(_mock_train_fn, None, seeds=[1, 2, 3])

        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(np.arange(50, dtype=float), np.arange(50, dtype=float) + 0.1)

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ens.predict_with_conformal(X, cp)

        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert "interval_width" in result
        # lower < mean < upper
        assert np.all(result["lower"] <= result["mean"])
        assert np.all(result["mean"] <= result["upper"])

    def test_n_models_validation(self):
        """TFTEnsemble rejects n_models < 1."""
        with pytest.raises(ValueError):
            TFTEnsemble(n_models=0)

    def test_disagreement_nonnegative(self):
        """Disagreement (max - min) should be >= 0."""
        ens = TFTEnsemble(n_models=3)
        ens.train_ensemble(_mock_train_fn, None, seeds=[1, 2, 3])
        result = ens.predict(np.array([[5.0, 10.0]]))
        assert np.all(result["disagreement"] >= 0)


# ============================================================================
# TimeSeriesAugmenter Tests
# ============================================================================


class TestTimeSeriesAugmenter:
    """Tests for TimeSeriesAugmenter."""

    @pytest.fixture
    def augmenter(self):
        return TimeSeriesAugmenter(seed=42)

    @pytest.fixture
    def sample_2d(self):
        """Sample 2D array (seq_len=20, n_features=5)."""
        rng = np.random.default_rng(42)
        return rng.normal(0, 1, (20, 5))

    @pytest.fixture
    def sample_3d(self):
        """Sample 3D array (batch=8, seq_len=20, n_features=5)."""
        rng = np.random.default_rng(42)
        return rng.normal(0, 1, (8, 20, 5))

    def test_jitter_preserves_shape_2d(self, augmenter, sample_2d):
        """Jitter preserves 2D shape."""
        result = augmenter.jitter(sample_2d, sigma=0.01)
        assert result.shape == sample_2d.shape

    def test_jitter_preserves_shape_3d(self, augmenter, sample_3d):
        """Jitter preserves 3D shape."""
        result = augmenter.jitter(sample_3d, sigma=0.01)
        assert result.shape == sample_3d.shape

    def test_jitter_changes_values(self, augmenter, sample_2d):
        """Jitter should change values (not return identical copy)."""
        result = augmenter.jitter(sample_2d, sigma=0.1)
        assert not np.allclose(result, sample_2d)

    def test_jitter_small_sigma_close(self, augmenter, sample_2d):
        """With very small sigma, jittered values should be close to originals."""
        result = augmenter.jitter(sample_2d, sigma=1e-6)
        np.testing.assert_array_almost_equal(result, sample_2d, decimal=3)

    def test_scaling_preserves_shape(self, augmenter, sample_2d):
        """Scaling preserves shape."""
        result = augmenter.scaling(sample_2d, sigma=0.1)
        assert result.shape == sample_2d.shape

    def test_scaling_mean_factor_approximately_one(self, augmenter):
        """Scaling factors should have mean approximately 1 over many trials."""
        X = np.ones((50, 10))
        results = []
        for _ in range(200):
            scaled = augmenter.scaling(X, sigma=0.1)
            # Each column scaled by same factor -> take mean of each column
            results.append(np.mean(scaled, axis=0))
        avg_factor = np.mean(results, axis=0)
        np.testing.assert_array_almost_equal(avg_factor, 1.0, decimal=1)

    def test_magnitude_warping_preserves_shape(self, augmenter, sample_2d):
        """Magnitude warping preserves shape."""
        result = augmenter.magnitude_warping(sample_2d, sigma=0.2, knots=4)
        assert result.shape == sample_2d.shape

    def test_magnitude_warping_rejects_3d(self, augmenter, sample_3d):
        """Magnitude warping requires 2D input."""
        with pytest.raises(ValueError, match="2D"):
            augmenter.magnitude_warping(sample_3d)

    def test_window_slicing_preserves_shape(self, augmenter, sample_2d):
        """Window slicing preserves output shape."""
        result = augmenter.window_slicing(sample_2d, ratio=0.9)
        assert result.shape == sample_2d.shape

    def test_window_slicing_ratio_one_returns_copy(self, augmenter, sample_2d):
        """Window slicing with ratio=1.0 returns a copy."""
        result = augmenter.window_slicing(sample_2d, ratio=1.0)
        np.testing.assert_array_almost_equal(result, sample_2d)

    def test_mixup_convexity(self, augmenter):
        """Mixup output should be between the two inputs."""
        X1 = np.zeros((10, 3))
        X2 = np.ones((10, 3))
        X_mix, y_mix = augmenter.mixup(X1, X2, 0.0, 1.0, alpha=0.5)

        # Mixed X should be elementwise in [0, 1]
        assert np.all(X_mix >= 0.0)
        assert np.all(X_mix <= 1.0)
        # Mixed y should be in [0, 1]
        assert 0.0 <= y_mix <= 1.0

    def test_mixup_shape_mismatch_raises(self, augmenter):
        """Mixup rejects shape-mismatched inputs."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            augmenter.mixup(np.zeros((10, 3)), np.zeros((10, 4)), 0.0, 1.0)

    def test_augment_batch_output_shape(self, augmenter, sample_3d):
        """augment_batch output has correct shape: (n * (1 + n_aug), seq, feat)."""
        n_samples = sample_3d.shape[0]
        y = np.random.default_rng(42).normal(0, 1, n_samples)
        n_aug = 3

        X_aug, y_aug = augmenter.augment_batch(sample_3d, y, n_augmented=n_aug)

        expected_n = n_samples * (1 + n_aug)
        assert X_aug.shape[0] == expected_n, f"Expected {expected_n}, got {X_aug.shape[0]}"
        assert X_aug.shape[1:] == sample_3d.shape[1:]
        assert len(y_aug) == expected_n

    def test_augment_batch_preserves_originals(self, augmenter, sample_3d):
        """augment_batch keeps original samples as the first n rows."""
        n_samples = sample_3d.shape[0]
        y = np.random.default_rng(42).normal(0, 1, n_samples)

        X_aug, y_aug = augmenter.augment_batch(sample_3d, y, n_augmented=2)

        # First n_samples should be identical to originals
        np.testing.assert_array_equal(X_aug[:n_samples], sample_3d)
        np.testing.assert_array_equal(y_aug[:n_samples], y)

    def test_augment_batch_rejects_2d(self, augmenter):
        """augment_batch rejects 2D input."""
        with pytest.raises(ValueError, match="3D"):
            augmenter.augment_batch(np.zeros((10, 5)), np.zeros(10))

    def test_augment_batch_default_methods(self, augmenter, sample_3d):
        """augment_batch with default methods doesn't crash."""
        y = np.zeros(sample_3d.shape[0])
        X_aug, y_aug = augmenter.augment_batch(sample_3d, y, n_augmented=1)
        assert X_aug.shape[0] == sample_3d.shape[0] * 2

    def test_augment_batch_invalid_method_raises(self, augmenter, sample_3d):
        """augment_batch with invalid methods raises ValueError."""
        y = np.zeros(sample_3d.shape[0])
        with pytest.raises(ValueError, match="No valid methods"):
            augmenter.augment_batch(sample_3d, y, methods=["nonexistent"])

    def test_augmenter_reproducibility(self):
        """Same seed produces identical results."""
        X = np.random.default_rng(99).normal(0, 1, (20, 5))

        aug1 = TimeSeriesAugmenter(seed=123)
        aug2 = TimeSeriesAugmenter(seed=123)

        r1 = aug1.jitter(X, sigma=0.1)
        r2 = aug2.jitter(X, sigma=0.1)
        np.testing.assert_array_equal(r1, r2)
