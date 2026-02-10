"""Tests for the Chronos foundation model wrapper and ensemble.

Covers:
- Module imports and graceful fallback
- ChronosForecaster predict shape and keys
- Trading signal bounds [-1, 1]
- Multi-asset predictions
- Walk-forward evaluation structure
- Ensemble weighting (static and dynamic)
- Panel conversion helpers
- Agreement / confidence metrics
- Edge cases (empty data, zero prices, NaN)

We mock the underlying Chronos pipeline to avoid downloading multi-GB
models in CI.  A separate marker ``@pytest.mark.slow`` guards tests that
actually load a model from HuggingFace.
"""

from __future__ import annotations

import importlib
import math
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Module under test ──────────────────────────────────────────────────────
from quantlaxmi.models.ml.foundation.chronos_wrapper import (
    ChronosConfig,
    ChronosForecaster,
    HAS_CHRONOS,
)
from quantlaxmi.models.ml.foundation.ensemble import (
    EnsembleConfig,
    FoundationEnsemble,
)

# Re-export check
from quantlaxmi.models.ml.foundation import (
    ChronosForecaster as CF_reexport,
    FoundationEnsemble as FE_reexport,
    HAS_CHRONOS as HC_reexport,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def price_series() -> np.ndarray:
    """Synthetic daily price series (random walk)."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.015, size=300)
    prices = 20000 * np.exp(np.cumsum(returns))
    return prices


@pytest.fixture
def mock_bolt_pipeline():
    """Mock Chronos2Pipeline that returns realistic-looking outputs."""
    pipeline = MagicMock()

    def _predict_quantiles(inputs, prediction_length=5, quantile_levels=None):
        import torch
        ql = quantile_levels or [0.1, 0.25, 0.5, 0.75, 0.9]
        batch = len(inputs) if isinstance(inputs, list) else 1
        H = prediction_length
        nq = len(ql)
        # Quantiles: ascending across quantile dimension
        base = 20100.0
        offsets = np.linspace(-200, 200, nq)
        q = np.zeros((batch, H, nq))
        for b in range(batch):
            for h in range(H):
                q[b, h, :] = base + offsets + h * 10
        mean = np.mean(q, axis=-1)  # (batch, H)
        return torch.tensor(q, dtype=torch.float32), torch.tensor(mean, dtype=torch.float32)

    pipeline.predict_quantiles = MagicMock(side_effect=_predict_quantiles)

    def _embed(inputs):
        import torch
        seq_len = inputs.shape[1] if inputs.ndim > 1 else inputs.shape[0]
        hidden_dim = 256
        emb = torch.randn(1, seq_len, hidden_dim)
        return emb, None

    pipeline.embed = MagicMock(side_effect=_embed)
    return pipeline


@pytest.fixture
def mock_forecaster(mock_bolt_pipeline) -> ChronosForecaster:
    """ChronosForecaster with mocked pipeline."""
    fc = ChronosForecaster.__new__(ChronosForecaster)
    fc.cfg = ChronosConfig(
        model_name="amazon/chronos-bolt-small",
        device="cpu",
        prediction_length=5,
        num_samples=100,
    )
    fc._pipeline = mock_bolt_pipeline
    fc._is_bolt = True
    fc._loaded = True
    return fc


# ═══════════════════════════════════════════════════════════════════════════
# 1. Module import tests
# ═══════════════════════════════════════════════════════════════════════════


class TestImports:
    def test_re_exports(self):
        """__init__.py re-exports the main classes."""
        assert CF_reexport is ChronosForecaster
        assert FE_reexport is FoundationEnsemble
        assert HC_reexport is HAS_CHRONOS

    def test_has_chronos_is_bool(self):
        assert isinstance(HAS_CHRONOS, bool)

    def test_chronos_available(self):
        """chronos-forecasting was installed successfully."""
        assert HAS_CHRONOS is True


# ═══════════════════════════════════════════════════════════════════════════
# 2. ChronosConfig tests
# ═══════════════════════════════════════════════════════════════════════════


class TestChronosConfig:
    def test_defaults(self):
        cfg = ChronosConfig()
        assert cfg.model_name == "amazon/chronos-t5-small"
        assert cfg.prediction_length == 5
        assert cfg.num_samples == 100
        assert 0.5 in cfg.quantile_levels
        assert 0.1 in cfg.quantile_levels

    def test_custom(self):
        cfg = ChronosConfig(model_name="amazon/chronos-t5-tiny", prediction_length=10)
        assert cfg.model_name == "amazon/chronos-t5-tiny"
        assert cfg.prediction_length == 10


# ═══════════════════════════════════════════════════════════════════════════
# 3. ChronosForecaster construction
# ═══════════════════════════════════════════════════════════════════════════


class TestForecasterInit:
    def test_default_init(self):
        fc = ChronosForecaster()
        assert not fc._loaded
        assert fc._is_bolt is False  # default model is T5
        assert fc.cfg.prediction_length == 5

    def test_init_with_config(self):
        cfg = ChronosConfig(model_name="amazon/chronos-t5-small", prediction_length=10)
        fc = ChronosForecaster(config=cfg)
        assert fc.cfg.model_name == "amazon/chronos-t5-small"
        assert fc._is_bolt is False

    def test_repr(self):
        fc = ChronosForecaster()
        r = repr(fc)
        assert "chronos-t5-small" in r
        assert "not loaded" in r


# ═══════════════════════════════════════════════════════════════════════════
# 4. Predict tests (mocked pipeline)
# ═══════════════════════════════════════════════════════════════════════════


class TestPredict:
    def test_predict_returns_expected_keys(self, mock_forecaster, price_series):
        out = mock_forecaster.predict(price_series)
        expected_keys = {"median", "mean", "quantile_10", "quantile_25",
                         "quantile_75", "quantile_90", "samples"}
        assert expected_keys == set(out.keys())

    def test_predict_shape(self, mock_forecaster, price_series):
        out = mock_forecaster.predict(price_series, prediction_length=5)
        assert out["median"].shape == (5,)
        assert out["mean"].shape == (5,)
        assert out["quantile_10"].shape == (5,)
        assert out["samples"].ndim >= 1

    def test_predict_quantile_ordering(self, mock_forecaster, price_series):
        """Lower quantiles should be <= higher quantiles."""
        out = mock_forecaster.predict(price_series)
        for h in range(len(out["median"])):
            assert out["quantile_10"][h] <= out["quantile_25"][h]
            assert out["quantile_25"][h] <= out["median"][h]
            assert out["median"][h] <= out["quantile_75"][h]
            assert out["quantile_75"][h] <= out["quantile_90"][h]

    def test_predict_custom_horizon(self, mock_forecaster, price_series):
        out = mock_forecaster.predict(price_series, prediction_length=10)
        assert out["median"].shape == (10,)

    def test_predict_short_context(self, mock_forecaster):
        """Should work with very short context."""
        short = np.array([100.0, 101.0, 99.5, 102.0, 100.5])
        out = mock_forecaster.predict(short)
        assert "median" in out


# ═══════════════════════════════════════════════════════════════════════════
# 5. Multi-asset prediction
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiAsset:
    def test_multi_asset_returns_all_assets(self, mock_forecaster, price_series):
        assets = {
            "NIFTY": price_series,
            "BANKNIFTY": price_series * 2.5,
        }
        results = mock_forecaster.predict_multi_asset(assets)
        assert "NIFTY" in results
        assert "BANKNIFTY" in results
        assert "median" in results["NIFTY"]
        assert "median" in results["BANKNIFTY"]

    def test_multi_asset_shapes(self, mock_forecaster, price_series):
        assets = {"A": price_series, "B": price_series[:100]}
        results = mock_forecaster.predict_multi_asset(assets, prediction_length=3)
        for name, res in results.items():
            assert res["median"].shape == (3,)

    def test_multi_asset_empty(self, mock_forecaster):
        results = mock_forecaster.predict_multi_asset({})
        assert results == {}


# ═══════════════════════════════════════════════════════════════════════════
# 6. Trading signal generation
# ═══════════════════════════════════════════════════════════════════════════


class TestTradingSignal:
    def test_signal_bounds(self, mock_forecaster, price_series):
        """Signal must be in [-1, 1]."""
        sig = mock_forecaster.generate_trading_signal(price_series)
        assert -1.0 <= sig <= 1.0

    def test_signal_is_float(self, mock_forecaster, price_series):
        sig = mock_forecaster.generate_trading_signal(price_series)
        assert isinstance(sig, float)

    def test_signal_zero_threshold(self, mock_forecaster, price_series):
        """With threshold=0, any signal should pass."""
        sig = mock_forecaster.generate_trading_signal(price_series, threshold=0.0)
        assert -1.0 <= sig <= 1.0

    def test_signal_high_threshold_returns_zero_or_strong(
        self, mock_forecaster, price_series
    ):
        """With threshold=0.99, signal should be either 0 or very strong."""
        sig = mock_forecaster.generate_trading_signal(price_series, threshold=0.99)
        assert sig == 0.0 or abs(sig) >= 0.99

    def test_signal_zero_price(self, mock_forecaster):
        """Zero last price should return 0.0 to avoid division by zero."""
        data = np.array([100.0, 50.0, 0.0])
        sig = mock_forecaster.generate_trading_signal(data)
        assert sig == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 7. Walk-forward evaluation
# ═══════════════════════════════════════════════════════════════════════════


class TestWalkForward:
    def test_wf_structure(self, mock_forecaster, price_series):
        result = mock_forecaster.walk_forward_evaluate(
            price_series,
            train_size=100,
            test_size=20,
            step_size=20,
            purge_gap=5,
            prediction_length=1,
        )
        assert "folds" in result
        assert "overall_mae" in result
        assert "overall_rmse" in result
        assert "directional_acc" in result
        assert "mean_sharpe" in result
        assert "num_folds" in result
        assert result["num_folds"] > 0

    def test_wf_fold_keys(self, mock_forecaster, price_series):
        result = mock_forecaster.walk_forward_evaluate(
            price_series, train_size=200, test_size=30, step_size=30
        )
        if result["num_folds"] > 0:
            fold = result["folds"][0]
            assert "train_end" in fold
            assert "test_start" in fold
            assert "mae" in fold
            assert "directional_acc" in fold
            assert "sharpe" in fold

    def test_wf_purge_gap(self, mock_forecaster, price_series):
        """Test start must be train_end + purge_gap."""
        result = mock_forecaster.walk_forward_evaluate(
            price_series, train_size=200, test_size=30, step_size=30, purge_gap=10
        )
        if result["num_folds"] > 0:
            fold = result["folds"][0]
            assert fold["test_start"] == fold["train_end"] + 10

    def test_wf_too_short_data(self, mock_forecaster):
        """Data too short for even one fold."""
        short = np.arange(50, dtype=np.float64)
        result = mock_forecaster.walk_forward_evaluate(
            short, train_size=200, test_size=42
        )
        assert result["num_folds"] == 0
        assert math.isnan(result["overall_mae"])

    def test_wf_metrics_finite(self, mock_forecaster, price_series):
        result = mock_forecaster.walk_forward_evaluate(
            price_series, train_size=100, test_size=20, step_size=50
        )
        if result["num_folds"] > 0:
            assert np.isfinite(result["overall_mae"])
            assert np.isfinite(result["overall_rmse"])


# ═══════════════════════════════════════════════════════════════════════════
# 8. Embedding extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestEmbed:
    def test_embed_shape(self, mock_forecaster, price_series):
        emb = mock_forecaster.embed(price_series)
        assert emb.ndim == 2  # (seq_len, hidden_dim)
        assert emb.shape[1] > 0  # hidden dim > 0

    def test_embed_returns_ndarray(self, mock_forecaster, price_series):
        emb = mock_forecaster.embed(price_series)
        assert isinstance(emb, np.ndarray)


# ═══════════════════════════════════════════════════════════════════════════
# 9. DataFrame-to-panel helper
# ═══════════════════════════════════════════════════════════════════════════


class TestDataFrameToPanel:
    def test_target_extracted(self):
        df = pd.DataFrame({
            "close": np.arange(100, dtype=np.float64),
            "feat_a": np.random.randn(100),
            "feat_b": np.random.randn(100),
        })
        panel = ChronosForecaster.dataframe_to_panel(df, target_col="close")
        assert "target" in panel
        assert len(panel["target"]) == 100

    def test_auto_selects_features(self):
        df = pd.DataFrame({
            "close": np.arange(100, dtype=np.float64),
            "high_var": np.random.randn(100) * 100,
            "low_var": np.random.randn(100) * 0.001,
        })
        panel = ChronosForecaster.dataframe_to_panel(df, top_n_features=1)
        # Should include the high-variance feature
        assert "high_var" in panel

    def test_skips_mostly_nan(self):
        df = pd.DataFrame({
            "close": np.arange(100, dtype=np.float64),
            "mostly_nan": np.full(100, np.nan),
        })
        panel = ChronosForecaster.dataframe_to_panel(
            df, feature_cols=["mostly_nan"]
        )
        assert "mostly_nan" not in panel

    def test_explicit_feature_cols(self):
        df = pd.DataFrame({
            "close": np.arange(50, dtype=np.float64),
            "x": np.random.randn(50),
            "y": np.random.randn(50),
            "z": np.random.randn(50),
        })
        panel = ChronosForecaster.dataframe_to_panel(
            df, feature_cols=["x", "z"]
        )
        assert "x" in panel
        assert "z" in panel
        assert "y" not in panel


# ═══════════════════════════════════════════════════════════════════════════
# 10. Ensemble tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsemble:
    def test_static_combine(self):
        ens = FoundationEnsemble(chronos_weight=0.3, tft_weight=0.7)
        sig = ens.combine(1.0, 0.0, use_dynamic=False)
        expected = 0.3 / 1.0 * 1.0 + 0.7 / 1.0 * 0.0
        assert abs(sig - expected) < 1e-9

    def test_combine_bounds(self):
        ens = FoundationEnsemble()
        for _ in range(100):
            c = np.random.uniform(-1, 1)
            t = np.random.uniform(-1, 1)
            sig = ens.combine(c, t, use_dynamic=False)
            assert -1.0 <= sig <= 1.0

    def test_dynamic_falls_back_without_history(self):
        ens = FoundationEnsemble(chronos_weight=0.4, tft_weight=0.6)
        w = ens.get_dynamic_weights()
        # With no history, falls back to static
        assert abs(w["chronos"] - 0.4) < 1e-9
        assert abs(w["tft"] - 0.6) < 1e-9

    def test_dynamic_weights_update(self):
        ens = FoundationEnsemble(
            config=EnsembleConfig(min_history=5, performance_window=20)
        )
        rng = np.random.default_rng(123)
        # Chronos does well, TFT does poorly
        for _ in range(15):
            ens.update_performance("chronos", rng.normal(0.005, 0.01))
            ens.update_performance("tft", rng.normal(-0.003, 0.01))

        w = ens.get_dynamic_weights()
        assert w["chronos"] > w["tft"], f"chronos={w['chronos']}, tft={w['tft']}"

    def test_dynamic_weights_sum_to_one(self):
        ens = FoundationEnsemble(
            config=EnsembleConfig(min_history=3, performance_window=10)
        )
        for i in range(10):
            ens.update_performance("chronos", 0.001 * i)
            ens.update_performance("tft", -0.001 * i)
        w = ens.get_dynamic_weights()
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_static_weights_normalized(self):
        ens = FoundationEnsemble(chronos_weight=3.0, tft_weight=7.0)
        w = ens.get_static_weights()
        assert abs(w["chronos"] - 0.3) < 1e-9
        assert abs(w["tft"] - 0.7) < 1e-9

    def test_combine_multi(self):
        ens = FoundationEnsemble()
        sig = ens.combine_multi(
            {"chronos": 0.8, "tft": -0.2, "xgb": 0.5},
            weights={"chronos": 1.0, "tft": 1.0, "xgb": 1.0},
        )
        expected = (0.8 + (-0.2) + 0.5) / 3.0
        assert abs(sig - expected) < 1e-9

    def test_combine_multi_equal_weights(self):
        ens = FoundationEnsemble()
        sig = ens.combine_multi({"a": 1.0, "b": -1.0})
        assert abs(sig - 0.0) < 1e-9

    def test_combine_multi_empty(self):
        ens = FoundationEnsemble()
        assert ens.combine_multi({}) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 11. Agreement and confidence
# ═══════════════════════════════════════════════════════════════════════════


class TestAgreement:
    def test_perfect_agreement(self):
        score = FoundationEnsemble.agreement_score(0.8, 0.8)
        assert abs(score - 1.0) < 1e-9

    def test_perfect_disagreement(self):
        score = FoundationEnsemble.agreement_score(1.0, -1.0)
        assert score < 0.01

    def test_both_zero(self):
        assert FoundationEnsemble.agreement_score(0.0, 0.0) == 1.0

    def test_agreement_bounds(self):
        for _ in range(200):
            c = np.random.uniform(-1, 1)
            t = np.random.uniform(-1, 1)
            score = FoundationEnsemble.agreement_score(c, t)
            assert 0.0 <= score <= 1.0

    def test_confidence_high_agreement(self):
        conf = FoundationEnsemble.confidence_from_agreement(0.8, 0.8)
        assert conf > 0.5

    def test_confidence_low_agreement(self):
        conf = FoundationEnsemble.confidence_from_agreement(0.8, -0.8)
        assert conf < 0.5

    def test_confidence_bounds(self):
        for _ in range(200):
            c = np.random.uniform(-1, 1)
            t = np.random.uniform(-1, 1)
            conf = FoundationEnsemble.confidence_from_agreement(c, t)
            assert 0.0 <= conf <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 12. Ensemble repr
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsembleRepr:
    def test_repr(self):
        ens = FoundationEnsemble(chronos_weight=0.4, tft_weight=0.6)
        r = repr(ens)
        assert "0.40" in r
        assert "0.60" in r
        assert "dynamic" in r.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 13. Fine-tuning (Chronos-2 only)
# ═══════════════════════════════════════════════════════════════════════════


class TestFineTune:
    def test_fine_tune_bolt(self, mock_forecaster):
        """Fine-tune should call pipeline.fit and return summary."""
        mock_forecaster._pipeline.fit = MagicMock(
            return_value=mock_forecaster._pipeline
        )
        train_data = [np.random.randn(200) for _ in range(5)]
        result = mock_forecaster.fine_tune(
            train_data, epochs=2, lr=1e-4, output_dir="/tmp/test_ft"
        )
        assert "num_series" in result
        assert result["num_series"] == 5
        assert "num_steps" in result
        assert result["num_steps"] > 0
        assert result["mode"] == "lora"
        mock_forecaster._pipeline.fit.assert_called_once()

    def test_fine_tune_t5_raises(self):
        """T5 models don't support .fit()."""
        fc = ChronosForecaster.__new__(ChronosForecaster)
        fc.cfg = ChronosConfig(model_name="amazon/chronos-t5-small")
        fc._pipeline = MagicMock()
        fc._is_bolt = False
        fc._loaded = True

        with pytest.raises(NotImplementedError, match="Chronos-2"):
            fc.fine_tune([np.random.randn(100)])


# ═══════════════════════════════════════════════════════════════════════════
# 14. Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_predict_single_value(self, mock_forecaster):
        out = mock_forecaster.predict(np.array([100.0]))
        assert "median" in out

    def test_predict_constant_series(self, mock_forecaster):
        out = mock_forecaster.predict(np.full(100, 50000.0))
        assert out["median"].shape == (5,)

    def test_context_length_trimming(self, mock_forecaster, price_series):
        mock_forecaster.cfg.context_length = 50
        out = mock_forecaster.predict(price_series)
        assert "median" in out
        # Verify the pipeline received only 50 values
        call_args = mock_forecaster._pipeline.predict_quantiles.call_args
        tensor_input = call_args[0][0]
        assert tensor_input.shape[1] == 50

    def test_ensemble_zero_weights(self):
        ens = FoundationEnsemble(chronos_weight=0.0, tft_weight=0.0)
        w = ens.get_static_weights()
        assert abs(w["chronos"] - 0.5) < 1e-9
        assert abs(w["tft"] - 0.5) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# 15. Integration: real model loading (slow, guarded)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestRealModel:
    """Tests that load actual Chronos models.  Skip in CI."""

    def test_load_t5_tiny(self):
        fc = ChronosForecaster(model_name="amazon/chronos-t5-tiny", device="cpu")
        fc._ensure_loaded()
        assert fc._loaded

    def test_real_predict(self):
        fc = ChronosForecaster(model_name="amazon/chronos-t5-tiny", device="cpu")
        prices = np.cumsum(np.random.randn(200)) + 20000
        out = fc.predict(prices, prediction_length=3)
        assert out["median"].shape == (3,)
        assert np.all(np.isfinite(out["median"]))
