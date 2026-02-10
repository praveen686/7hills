"""Data augmentation for financial time series.

Conservative augmentations that preserve temporal structure and statistical
properties of the original data. These are designed for sequential models
(TFT/LSTM) where the ordering of time steps matters.

Methods include:
- Jitter: Gaussian noise injection
- Scaling: random per-feature magnitude scaling
- Magnitude warping: smooth time-varying scaling via cubic spline
- Window slicing: random contiguous sub-window with resize
- Mixup: convex combination of two samples (inter-sample)
- Batch augmentation: apply random augmentations to a training batch

References
----------
- Um et al. (2017), "Data Augmentation of Wearable Sensor Data for Parkinson's
  Disease Monitoring using Convolutional Neural Networks"
- Zhang et al. (2018), "mixup: Beyond Empirical Risk Minimization"
- Iwana & Uchida (2021), "An Empirical Survey of Data Augmentation for
  Time Series Classification with Neural Networks"
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TimeSeriesAugmenter:
    """Data augmentation for financial time series.

    Conservative augmentations that preserve temporal structure.
    All methods accept and return numpy arrays.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def jitter(self, X: np.ndarray, sigma: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to features.

        Noise is scaled relative to the per-feature standard deviation
        of the input, so sigma=0.01 means 1% of the feature's variability.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (seq_len, n_features) or (batch, seq_len, n_features).
        sigma : float
            Noise scale relative to feature std.

        Returns
        -------
        np.ndarray
            Augmented array with same shape as X.
        """
        X = np.asarray(X, dtype=np.float64)
        # Compute feature-wise std over all but last axis
        if X.ndim == 2:
            feat_std = np.std(X, axis=0, ddof=0)
        elif X.ndim == 3:
            feat_std = np.std(X.reshape(-1, X.shape[-1]), axis=0, ddof=0)
        else:
            feat_std = np.std(X)

        feat_std = np.where(feat_std > 1e-12, feat_std, 1.0)
        noise = self.rng.normal(0, 1, size=X.shape) * sigma * feat_std
        return X + noise

    def scaling(self, X: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """Multiply each feature by random factor ~ N(1, sigma).

        Each feature gets its own scaling factor, which is constant across
        the time dimension (preserving temporal patterns).

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (seq_len, n_features) or (batch, seq_len, n_features).
        sigma : float
            Standard deviation of the scaling factor (mean=1).

        Returns
        -------
        np.ndarray
            Scaled array with same shape as X.
        """
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[-1]
        # One scale factor per feature
        factors = self.rng.normal(1.0, sigma, size=n_features)
        return X * factors

    def magnitude_warping(
        self, X: np.ndarray, sigma: float = 0.2, knots: int = 4
    ) -> np.ndarray:
        """Smooth magnitude warping via cubic spline interpolation.

        Creates a smooth time-varying scaling curve by interpolating
        random knot points with a cubic spline.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (seq_len, n_features).
        sigma : float
            Standard deviation of knot values around 1.0.
        knots : int
            Number of internal knot points (plus 2 endpoints).

        Returns
        -------
        np.ndarray
            Warped array with same shape as X.
        """
        from scipy.interpolate import CubicSpline

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                f"magnitude_warping expects 2D input (seq_len, n_features), "
                f"got shape {X.shape}"
            )

        seq_len, n_features = X.shape

        # Knot positions evenly spaced including endpoints
        n_knots = knots + 2
        knot_xs = np.linspace(0, seq_len - 1, n_knots)
        knot_ys = self.rng.normal(1.0, sigma, size=(n_knots, n_features))

        # Interpolate to full sequence length
        xs = np.arange(seq_len)
        warp_factors = np.zeros((seq_len, n_features))
        for f in range(n_features):
            cs = CubicSpline(knot_xs, knot_ys[:, f])
            warp_factors[:, f] = cs(xs)

        return X * warp_factors

    def window_slicing(self, X: np.ndarray, ratio: float = 0.9) -> np.ndarray:
        """Random contiguous slice, then resize to original length.

        Takes a random contiguous sub-window of length ratio*seq_len,
        then linearly interpolates back to the original length.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (seq_len, n_features).
        ratio : float
            Fraction of the sequence to keep (0 < ratio <= 1).

        Returns
        -------
        np.ndarray
            Sliced and resized array with same shape as X.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                f"window_slicing expects 2D input (seq_len, n_features), "
                f"got shape {X.shape}"
            )

        seq_len, n_features = X.shape
        slice_len = max(1, int(seq_len * ratio))

        if slice_len >= seq_len:
            return X.copy()

        # Random start position
        start = self.rng.integers(0, seq_len - slice_len + 1)
        sliced = X[start : start + slice_len]

        # Resize back to original length via linear interpolation
        old_xs = np.linspace(0, 1, slice_len)
        new_xs = np.linspace(0, 1, seq_len)
        resized = np.zeros((seq_len, n_features))
        for f in range(n_features):
            resized[:, f] = np.interp(new_xs, old_xs, sliced[:, f])

        return resized

    def mixup(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        y1: float,
        y2: float,
        alpha: float = 0.2,
    ) -> tuple[np.ndarray, float]:
        """Mixup: convex combination of two samples.

        lambda ~ Beta(alpha, alpha)
        X_mix = lambda * X1 + (1 - lambda) * X2
        y_mix = lambda * y1 + (1 - lambda) * y2

        Parameters
        ----------
        X1, X2 : np.ndarray
            Two input samples with the same shape.
        y1, y2 : float
            Target values for the two samples.
        alpha : float
            Beta distribution parameter. Smaller = more interpolation
            toward one of the endpoints.

        Returns
        -------
        X_mix : np.ndarray
            Mixed input.
        y_mix : float
            Mixed target.
        """
        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)

        if X1.shape != X2.shape:
            raise ValueError(
                f"Shape mismatch: X1 {X1.shape} vs X2 {X2.shape}"
            )

        lam = float(self.rng.beta(alpha, alpha))
        X_mix = lam * X1 + (1.0 - lam) * X2
        y_mix = lam * y1 + (1.0 - lam) * y2

        return X_mix, y_mix

    def augment_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_augmented: int = 2,
        methods: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations to a batch.

        For each original sample, generates n_augmented synthetic copies
        using randomly chosen augmentation methods.

        Parameters
        ----------
        X : np.ndarray
            Input batch of shape (n_samples, seq_len, n_features).
        y : np.ndarray
            Targets of shape (n_samples,).
        n_augmented : int
            Number of augmented copies per original sample.
        methods : list[str], optional
            Augmentation methods to use. Default: ['jitter', 'scaling'].
            Options: 'jitter', 'scaling', 'magnitude_warping', 'window_slicing'.

        Returns
        -------
        X_aug : np.ndarray
            Shape (n_samples * (1 + n_augmented), seq_len, n_features).
            Original samples followed by augmented samples.
        y_aug : np.ndarray
            Shape (n_samples * (1 + n_augmented),).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim != 3:
            raise ValueError(
                f"augment_batch expects 3D input (batch, seq_len, n_features), "
                f"got shape {X.shape}"
            )
        if len(y) != X.shape[0]:
            raise ValueError(
                f"Length mismatch: X has {X.shape[0]} samples, y has {len(y)}"
            )

        if methods is None:
            methods = ["jitter", "scaling"]

        method_map = {
            "jitter": self.jitter,
            "scaling": self.scaling,
            "magnitude_warping": self.magnitude_warping,
            "window_slicing": self.window_slicing,
        }

        valid_methods = [m for m in methods if m in method_map]
        if not valid_methods:
            raise ValueError(
                f"No valid methods in {methods}. "
                f"Choose from: {list(method_map.keys())}"
            )

        n_samples = X.shape[0]
        aug_X_list = [X]  # start with originals
        aug_y_list = [y]

        for _ in range(n_augmented):
            batch_aug = np.zeros_like(X)
            for i in range(n_samples):
                method_name = valid_methods[
                    self.rng.integers(0, len(valid_methods))
                ]
                method_fn = method_map[method_name]
                batch_aug[i] = method_fn(X[i])
            aug_X_list.append(batch_aug)
            # Augmented targets are same as originals (jitter/scaling don't
            # change the target; this is conservative for financial data)
            aug_y_list.append(y.copy())

        X_aug = np.concatenate(aug_X_list, axis=0)
        y_aug = np.concatenate(aug_y_list, axis=0)

        return X_aug, y_aug
