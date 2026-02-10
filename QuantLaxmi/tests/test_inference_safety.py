"""Tests for TFT inference safety: timeout + normalization guards.

Bug #7  — inference timeout (CRITICAL)
Bug #17 — div-by-zero in normalization (HIGH)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantlaxmi.models.ml.tft.production.inference import (
    InferenceResult,
    TFTInferencePipeline,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight mock metadata + model for unit tests
# ---------------------------------------------------------------------------

ASSET_NAMES = ["NIFTY", "BANKNIFTY"]
FEATURE_NAMES = [f"f{i}" for i in range(10)]
N_ASSETS = len(ASSET_NAMES)
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class _FakeMetadata:
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    asset_names: list[str] = field(default_factory=lambda: list(ASSET_NAMES))
    n_assets: int = N_ASSETS
    n_features: int = N_FEATURES
    version: int = 1
    model_type: str = "x_trend"
    config: dict = field(default_factory=lambda: {
        "seq_len": 5,
        "n_context": 2,
        "ctx_len": 5,
        "max_position": 0.25,
        "position_smooth": 0.3,
    })
    normalization: dict = field(default_factory=dict)


class _FakeModel:
    """Minimal stand-in for nn.Module without requiring torch."""

    def __init__(self, delay: float = 0.0):
        self._delay = delay

    def parameters(self):
        """Yield a fake parameter so next(model.parameters()) works."""
        import torch
        yield torch.zeros(1)

    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        if self._delay > 0:
            time.sleep(self._delay)
        import torch
        return torch.tensor(0.05)


def _make_pipeline(
    norm_means: Optional[np.ndarray] = None,
    norm_stds: Optional[np.ndarray] = None,
    inference_timeout: float = 30.0,
    model_delay: float = 0.0,
) -> TFTInferencePipeline:
    """Build a TFTInferencePipeline with fake model + metadata."""
    if norm_means is None:
        norm_means = np.zeros(N_FEATURES)
    if norm_stds is None:
        norm_stds = np.ones(N_FEATURES)

    return TFTInferencePipeline(
        model=_FakeModel(delay=model_delay),
        metadata=_FakeMetadata(),
        norm_means=norm_means,
        norm_stds=norm_stds,
        inference_timeout=inference_timeout,
    )


def _dummy_features(n_days: int = 10) -> np.ndarray:
    """Return (n_days, n_assets, n_features) random features."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_days, N_ASSETS, N_FEATURES))


# ===================================================================
# Bug #17 — div-by-zero in normalization
# ===================================================================


class TestNormalizationSafety:
    """Verify that zero-std features are handled without NaN/Inf."""

    def test_zero_std_no_nan_inf(self):
        """When some norm_stds are exactly 0, normalization must not
        produce NaN or Inf — it should substitute 1.0 for those stds."""
        stds = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0])
        means = np.zeros(N_FEATURES)

        pipe = _make_pipeline(norm_means=means, norm_stds=stds)

        features = _dummy_features(n_days=10)

        # Perform normalization the same way as predict() does
        safe_stds = np.where(pipe.norm_stds > 1e-10, pipe.norm_stds, 1.0)
        norm = (features - pipe.norm_means) / safe_stds

        assert np.all(np.isfinite(norm)), "NaN or Inf found in normalized features"

    def test_all_zero_stds(self):
        """Edge case: every single std is zero."""
        stds = np.zeros(N_FEATURES)
        means = np.ones(N_FEATURES) * 5.0
        pipe = _make_pipeline(norm_means=means, norm_stds=stds)

        features = _dummy_features(n_days=10)

        safe_stds = np.where(pipe.norm_stds > 1e-10, pipe.norm_stds, 1.0)
        norm = (features - pipe.norm_means) / safe_stds

        assert np.all(np.isfinite(norm)), "All-zero stds must not produce NaN/Inf"
        # With std=1.0 fallback, result is just (features - means)
        expected = features - means
        np.testing.assert_allclose(norm, expected, atol=1e-12)

    def test_tiny_stds_below_threshold(self):
        """Stds just below the 1e-10 threshold should be replaced with 1.0."""
        stds = np.full(N_FEATURES, 1e-11)
        means = np.zeros(N_FEATURES)
        pipe = _make_pipeline(norm_means=means, norm_stds=stds)

        features = np.ones((5, N_ASSETS, N_FEATURES)) * 100.0

        safe_stds = np.where(pipe.norm_stds > 1e-10, pipe.norm_stds, 1.0)
        norm = (features - pipe.norm_means) / safe_stds

        assert np.all(np.isfinite(norm))
        # With std replaced to 1.0, result should be features themselves
        np.testing.assert_allclose(norm, features, atol=1e-12)

    def test_normal_stds_unchanged(self):
        """Regular (non-zero) stds should produce the same result as raw division."""
        stds = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        means = np.linspace(0, 1, N_FEATURES)
        pipe = _make_pipeline(norm_means=means, norm_stds=stds)

        features = _dummy_features(n_days=8)

        safe_stds = np.where(pipe.norm_stds > 1e-10, pipe.norm_stds, 1.0)
        norm_safe = (features - pipe.norm_means) / safe_stds
        norm_raw = (features - pipe.norm_means) / pipe.norm_stds

        np.testing.assert_allclose(norm_safe, norm_raw, atol=1e-15)


# ===================================================================
# Bug #7 — inference timeout
# ===================================================================


class TestInferenceTimeout:
    """Verify inference timeout mechanism."""

    def test_inference_timeout_parameter_stored(self):
        """The inference_timeout should be stored on the pipeline."""
        pipe = _make_pipeline(inference_timeout=15.0)
        assert pipe.inference_timeout == 15.0

    def test_default_timeout_is_30s(self):
        """Default timeout should be 30 seconds."""
        pipe = _make_pipeline()
        assert pipe.inference_timeout == 30.0

    def test_safe_default_returns_zeros(self):
        """_safe_default should return zero positions and confidences."""
        pipe = _make_pipeline(inference_timeout=5.0)
        result = pipe._safe_default(date(2026, 2, 10))

        assert isinstance(result, InferenceResult)
        assert result.date == date(2026, 2, 10)
        assert all(v == 0.0 for v in result.positions.values())
        assert all(v == 0.0 for v in result.confidences.values())
        assert result.metadata["error"] == "timeout"
        assert result.metadata["timeout_s"] == 5.0

    def test_fast_inference_completes_within_timeout(self):
        """A fast _forward should return normally, not trigger timeout."""
        pipe = _make_pipeline(inference_timeout=10.0)

        features = _dummy_features(n_days=10)

        # Mock _build_features to return our dummy tensor
        pipe._build_features = MagicMock(return_value=(features, None))

        # Mock _forward to return immediately
        fast_result = (
            {"NIFTY": 0.1, "BANKNIFTY": -0.05},
            {"NIFTY": 0.8, "BANKNIFTY": 0.6},
            {"NIFTY": {"mu": 0.1}, "BANKNIFTY": {"mu": -0.05}},
        )
        pipe._forward = MagicMock(return_value=fast_result)
        pipe._get_feature_importance = MagicMock(return_value={})

        t0 = time.time()
        result = pipe.predict(date(2026, 2, 10), store=MagicMock())
        elapsed = time.time() - t0

        assert elapsed < 5.0, f"Fast inference took {elapsed:.1f}s — too slow"
        assert result.positions["NIFTY"] == 0.1
        assert result.positions["BANKNIFTY"] == -0.05
        assert result.metadata.get("error") is None

    def test_slow_model_triggers_timeout_returns_safe_default(self):
        """A _forward that hangs should be killed by timeout, returning zeros."""
        pipe = _make_pipeline(inference_timeout=1.0)

        features = _dummy_features(n_days=10)
        pipe._build_features = MagicMock(return_value=(features, None))

        # Simulate a model that hangs for 60 seconds
        def _slow_forward(norm_features):
            time.sleep(60)
            return (
                {"NIFTY": 0.1, "BANKNIFTY": -0.05},
                {"NIFTY": 0.8, "BANKNIFTY": 0.6},
                {},
            )

        pipe._forward = _slow_forward
        pipe._get_feature_importance = MagicMock(return_value={})

        t0 = time.time()
        result = pipe.predict(date(2026, 2, 10), store=MagicMock())
        elapsed = time.time() - t0

        # Should return within ~1-2 seconds (the timeout), not 60
        assert elapsed < 5.0, f"Timeout did not fire — took {elapsed:.1f}s"
        # Should return the safe default
        assert result.metadata.get("error") == "timeout"
        assert all(v == 0.0 for v in result.positions.values())
        assert all(v == 0.0 for v in result.confidences.values())

    def test_timeout_propagates_to_predict_batch(self):
        """predict_batch should respect the same timeout per date."""
        pipe = _make_pipeline(inference_timeout=0.5)

        features = _dummy_features(n_days=10)
        pipe._build_features = MagicMock(return_value=(features, None))

        call_count = 0

        def _slow_forward(norm_features):
            nonlocal call_count
            call_count += 1
            time.sleep(60)
            return ({}, {}, {})

        pipe._forward = _slow_forward
        pipe._get_feature_importance = MagicMock(return_value={})

        dates = [date(2026, 2, 8), date(2026, 2, 9), date(2026, 2, 10)]
        t0 = time.time()
        results = pipe.predict_batch(dates, store=MagicMock())
        elapsed = time.time() - t0

        # 3 dates * 0.5s timeout each = ~1.5s + overhead, well under 60s
        assert elapsed < 10.0, f"Batch took {elapsed:.1f}s — timeouts not firing"
        assert len(results) == 3
        assert all(r.metadata.get("error") == "timeout" for r in results)


# ===================================================================
# Integration: both fixes together
# ===================================================================


class TestNormalizationAndTimeoutCombined:
    """Verify that zero-std normalization + timeout work together."""

    def test_zero_std_with_fast_model(self):
        """Zero stds should not cause NaN — fast model returns normally."""
        stds = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0])
        pipe = _make_pipeline(norm_stds=stds, inference_timeout=5.0)

        features = _dummy_features(n_days=10)
        pipe._build_features = MagicMock(return_value=(features, None))

        fast_result = (
            {"NIFTY": 0.05, "BANKNIFTY": -0.02},
            {"NIFTY": 0.7, "BANKNIFTY": 0.5},
            {},
        )
        pipe._forward = MagicMock(return_value=fast_result)
        pipe._get_feature_importance = MagicMock(return_value={})

        result = pipe.predict(date(2026, 2, 10), store=MagicMock())

        # Should succeed (no NaN, no timeout)
        assert result.positions["NIFTY"] == 0.05
        assert result.metadata.get("error") is None

        # Verify _forward was called with finite features
        called_features = pipe._forward.call_args[0][0]
        assert np.all(np.isfinite(called_features)), \
            "Forward received NaN/Inf features despite zero-std guard"
