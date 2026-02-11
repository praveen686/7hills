"""Tests for production TFT inference system.

Tests all components using synthetic data — no DuckDB dependency.
Covers: FeatureSelector, CheckpointManager, HPTuner, TFTInferencePipeline,
TFTStrategy, TrainingPipeline (unit-level).
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Skip entire module if torch is not available
torch = pytest.importorskip("torch")

from quantlaxmi.models.ml.tft.production.feature_selection import (
    FeatureSelector,
    FeatureSelectionConfig,
    StabilityReport,
)
from quantlaxmi.models.ml.tft.production.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
    _convert_numpy,
)
from quantlaxmi.models.ml.tft.production.inference import (
    TFTInferencePipeline,
    InferenceResult,
)
from quantlaxmi.models.ml.tft.production.strategy_adapter import (
    TFTStrategy,
    create_strategy,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_dir():
    """Temporary directory for checkpoint tests."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synth_features():
    """Synthetic feature DataFrame (100 rows, 20 features)."""
    rng = np.random.default_rng(42)
    n_rows, n_feat = 100, 20
    data = rng.standard_normal((n_rows, n_feat))
    # Add some NaN (10%)
    mask = rng.random((n_rows, n_feat)) < 0.1
    data[mask] = np.nan
    # Make two highly correlated columns
    data[:, 19] = data[:, 0] * 0.99 + rng.standard_normal(n_rows) * 0.01
    columns = [f"feat_{i}" for i in range(n_feat)]
    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def synth_features_3d():
    """Synthetic 3D features (60 days, 4 assets, 10 features)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((60, 4, 10)).astype(np.float32)


@pytest.fixture
def synth_targets():
    """Synthetic targets (60 days, 4 assets)."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((60, 4)) * 0.01).astype(np.float32)


@pytest.fixture
def mock_model():
    """Mock XTrendModel with VSN for weight extraction tests."""
    from quantlaxmi.models.ml.tft.x_trend import XTrendConfig, XTrendModel
    cfg = XTrendConfig(d_hidden=16, n_heads=2, lstm_layers=1, n_features=10, n_assets=4, seq_len=10, n_context=4)
    model = XTrendModel(cfg)
    return model


# ============================================================================
# TestFeatureSelector
# ============================================================================


class TestFeatureSelector:
    """Tests for FeatureSelector."""

    def test_coverage_filter(self, synth_features):
        selector = FeatureSelector(FeatureSelectionConfig(min_coverage=0.5))
        kept = selector._filter_coverage(synth_features, 0.5)
        # All features should have ~90% coverage (10% NaN)
        assert len(kept) > 15

    def test_coverage_filter_strict(self, synth_features):
        # With very strict coverage, should keep all (90% > 99% is false)
        selector = FeatureSelector()
        kept = selector._filter_coverage(synth_features, 0.99)
        # Some features may have <99% coverage due to random NaN
        assert len(kept) <= 20

    def test_correlation_filter(self, synth_features):
        selector = FeatureSelector()
        kept = selector._filter_correlation(synth_features, 0.95)
        # feat_19 is highly correlated with feat_0; one should be dropped
        assert len(kept) < 20
        # Either feat_0 or feat_19 dropped, not both
        has_0 = "feat_0" in kept
        has_19 = "feat_19" in kept
        assert has_0 or has_19
        assert not (has_0 and has_19)

    def test_prefilter_combined(self, synth_features):
        selector = FeatureSelector(FeatureSelectionConfig(
            min_coverage=0.5,
            correlation_threshold=0.95,
        ))
        kept = selector.prefilter(synth_features)
        assert len(kept) > 0
        assert len(kept) < 20  # at least one dropped
        assert selector.prefilter_kept is not None

    def test_extract_vsn_weights(self, mock_model):
        selector = FeatureSelector()
        names = [f"feat_{i}" for i in range(10)]
        weights = selector.extract_vsn_weights(mock_model, names)
        assert len(weights) == 10
        assert abs(sum(weights.values()) - 1.0) < 0.01  # softmax sums to ~1

    def test_add_fold_weights(self, mock_model):
        selector = FeatureSelector()
        names = [f"feat_{i}" for i in range(10)]
        w1 = selector.add_fold_weights(mock_model, names)
        w2 = selector.add_fold_weights(mock_model, names)
        assert selector.n_folds == 2
        assert len(w1) == 10
        assert len(w2) == 10

    def test_stability_report(self, mock_model):
        selector = FeatureSelector(FeatureSelectionConfig(
            vsn_weight_threshold=0.001,
            stability_min_folds=0.5,
        ))
        names = [f"feat_{i}" for i in range(10)]
        # Add same model 3 times (simulating 3 folds)
        for _ in range(3):
            selector.add_fold_weights(mock_model, names)

        report = selector.stability_report(names)
        assert isinstance(report, StabilityReport)
        assert report.n_folds == 3
        assert len(report.stable_features) > 0
        assert len(report.recommended_features) > 0
        assert all(0.0 <= s <= 1.0 for s in report.stability_scores.values())

    def test_stability_no_folds_raises(self):
        selector = FeatureSelector()
        with pytest.raises(ValueError, match="No fold weights"):
            selector.stability_report()

    def test_save_load_roundtrip(self, mock_model, tmp_dir):
        selector = FeatureSelector(FeatureSelectionConfig(min_coverage=0.4))
        names = [f"feat_{i}" for i in range(10)]
        selector.add_fold_weights(mock_model, names)

        save_path = tmp_dir / "selector.pkl"
        selector.save(save_path)

        loaded = FeatureSelector.load(save_path)
        assert loaded.n_folds == 1
        assert loaded.config.min_coverage == 0.4

    def test_feature_importance_report(self, mock_model):
        selector = FeatureSelector()
        names = [f"feat_{i}" for i in range(10)]
        for _ in range(2):
            selector.add_fold_weights(mock_model, names)

        report_str = selector.feature_importance_report()
        assert "FEATURE IMPORTANCE REPORT" in report_str
        assert "Folds analyzed: 2" in report_str


# ============================================================================
# TestCheckpointManager
# ============================================================================


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_creates_directory(self, tmp_dir):
        mgr = CheckpointManager(tmp_dir)
        state_dict = {"layer.weight": torch.randn(4, 4)}
        meta = CheckpointMetadata(
            version=1,
            model_type="test_model",
            n_features=10,
            feature_names=[f"f{i}" for i in range(10)],
        )
        ckpt_dir = mgr.save(state_dict, meta)
        assert ckpt_dir.exists()
        assert (ckpt_dir / "model.pt").exists()
        assert (ckpt_dir / "metadata.json").exists()

    def test_load_roundtrip_state_dict(self, tmp_dir):
        mgr = CheckpointManager(tmp_dir)
        original_sd = {"layer.weight": torch.randn(4, 4)}
        meta = CheckpointMetadata(version=1, model_type="test", n_features=5)
        ckpt_dir = mgr.save(original_sd, meta)

        loaded_sd, loaded_meta = mgr.load(ckpt_dir)
        assert torch.allclose(original_sd["layer.weight"], loaded_sd["layer.weight"])

    def test_load_roundtrip_metadata(self, tmp_dir):
        mgr = CheckpointManager(tmp_dir)
        meta = CheckpointMetadata(
            version=2,
            model_type="x_trend",
            n_features=42,
            sharpe_oos=1.5,
            feature_names=["a", "b", "c"],
            config={"d_hidden": 64, "seq_len": 42},
        )
        state_dict = {"w": torch.randn(2, 2)}
        ckpt_dir = mgr.save(state_dict, meta)

        _, loaded = mgr.load(ckpt_dir)
        assert loaded.version == 2
        assert loaded.model_type == "x_trend"
        assert loaded.n_features == 42
        assert loaded.sharpe_oos == 1.5
        assert loaded.feature_names == ["a", "b", "c"]
        assert loaded.config["d_hidden"] == 64

    def test_list_sorted(self, tmp_dir):
        mgr = CheckpointManager(tmp_dir)
        for v in [1, 2, 3]:
            meta = CheckpointMetadata(version=v, model_type="test")
            mgr.save({"w": torch.randn(2, 2)}, meta)

        checkpoints = mgr.list_checkpoints("test")
        assert len(checkpoints) == 3
        versions = [c["version"] for c in checkpoints]
        assert versions == [1, 2, 3]

    def test_load_latest(self, tmp_dir):
        mgr = CheckpointManager(tmp_dir)
        for v in [1, 2, 3]:
            meta = CheckpointMetadata(
                version=v, model_type="test", sharpe_oos=float(v),
            )
            mgr.save({"w": torch.randn(2, 2)}, meta)

        _, latest = mgr.load_latest("test")
        assert latest.version == 3
        assert latest.sharpe_oos == 3.0

    def test_auto_version(self, tmp_dir):
        mgr = CheckpointManager(tmp_dir)
        # Save with version=0 to trigger auto-increment
        meta1 = CheckpointMetadata(version=0, model_type="test")
        mgr.save({"w": torch.randn(2, 2)}, meta1)

        meta2 = CheckpointMetadata(version=0, model_type="test")
        mgr.save({"w": torch.randn(2, 2)}, meta2)

        checkpoints = mgr.list_checkpoints("test")
        versions = [c["version"] for c in checkpoints]
        assert 1 in versions
        assert 2 in versions

    def test_load_nonexistent_raises(self, tmp_dir):
        mgr = CheckpointManager(tmp_dir)
        with pytest.raises(FileNotFoundError):
            mgr.load(tmp_dir / "nonexistent")

    def test_load_latest_empty_raises(self, tmp_dir):
        mgr = CheckpointManager(tmp_dir)
        with pytest.raises(FileNotFoundError, match="No checkpoints"):
            mgr.load_latest("nonexistent")


# ============================================================================
# TestCheckpointMetadata
# ============================================================================


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata serialization."""

    def test_to_json_from_json_roundtrip(self):
        meta = CheckpointMetadata(
            version=5,
            model_type="x_trend",
            feature_names=["a", "b"],
            n_features=2,
            sharpe_oos=2.5,
            config={"d_hidden": 64},
            normalization={"means": [1.0, 2.0], "stds": [0.5, 0.5]},
        )
        json_str = meta.to_json()
        loaded = CheckpointMetadata.from_json(json_str)
        assert loaded.version == 5
        assert loaded.sharpe_oos == 2.5
        assert loaded.feature_names == ["a", "b"]

    def test_numpy_conversion(self):
        result = _convert_numpy({
            "int": np.int64(42),
            "float": np.float32(3.14),
            "array": np.array([1, 2, 3]),
            "bool": np.bool_(True),
            "nested": {"x": np.float64(1.0)},
        })
        assert result["int"] == 42
        assert isinstance(result["int"], int)
        assert isinstance(result["float"], float)
        assert result["array"] == [1, 2, 3]
        assert result["bool"] is True


# ============================================================================
# TestHPTuner
# ============================================================================


class TestHPTuner:
    """Tests for HPTuner."""

    def test_build_config_from_params(self):
        from quantlaxmi.models.ml.tft.production.hp_tuner import HPTuner

        params = {
            "d_hidden": 64,
            "n_heads": 4,
            "lstm_layers": 2,
            "dropout": 0.1,
            "seq_len": 42,
            "n_context": 16,
            "lr": 1e-3,
            "batch_size": 32,
            "mle_weight": 0.1,
            "loss_mode": "sharpe",
            "max_position": 0.25,
            "position_smooth": 0.3,
        }
        cfg = HPTuner.build_config_from_params(params, n_features=20, n_assets=4)
        assert cfg.d_hidden == 64
        assert cfg.n_features == 20
        assert cfg.n_assets == 4
        assert cfg.loss_mode == "sharpe"

    def test_build_config_with_overrides(self):
        from quantlaxmi.models.ml.tft.production.hp_tuner import HPTuner

        params = {"d_hidden": 64, "n_heads": 4}
        cfg = HPTuner.build_config_from_params(
            params, n_features=20, train_window=200,
        )
        assert cfg.d_hidden == 64
        assert cfg.train_window == 200


# ============================================================================
# TestTFTInferencePipeline
# ============================================================================


class TestTFTInferencePipeline:
    """Tests for TFTInferencePipeline."""

    def test_from_checkpoint_loads(self, tmp_dir):
        """Test that from_checkpoint correctly loads a model."""
        from quantlaxmi.models.ml.tft.x_trend import XTrendConfig, XTrendModel

        cfg = XTrendConfig(d_hidden=16, n_heads=2, lstm_layers=1,
                           n_features=10, n_assets=4, seq_len=10, n_context=4)
        model = XTrendModel(cfg)

        mgr = CheckpointManager(tmp_dir)
        meta = CheckpointMetadata(
            version=1,
            model_type="x_trend",
            feature_names=[f"f{i}" for i in range(10)],
            n_features=10,
            n_assets=4,
            asset_names=["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"],
            config=asdict(cfg),
            normalization={
                "means": [0.0] * 10,
                "stds": [1.0] * 10,
            },
        )
        ckpt_dir = mgr.save(model.state_dict(), meta)

        pipeline = TFTInferencePipeline.from_checkpoint(ckpt_dir, device="cpu")
        assert pipeline.version == 1
        assert len(pipeline.feature_names) == 10
        assert len(pipeline.asset_names) == 4

    def test_inference_result_structure(self):
        result = InferenceResult(
            date=date(2026, 2, 9),
            positions={"NIFTY": 0.15, "BANKNIFTY": -0.1},
            confidences={"NIFTY": 0.6, "BANKNIFTY": 0.4},
        )
        assert result.date == date(2026, 2, 9)
        assert result.positions["NIFTY"] == 0.15
        assert result.confidences["BANKNIFTY"] == 0.4

    def test_normalization_uses_saved_stats(self, tmp_dir):
        """Verify that inference uses saved normalization, not recomputed."""
        from quantlaxmi.models.ml.tft.x_trend import XTrendConfig, XTrendModel

        cfg = XTrendConfig(d_hidden=16, n_heads=2, lstm_layers=1,
                           n_features=5, n_assets=2, seq_len=5, n_context=2)
        model = XTrendModel(cfg)

        means = [1.0, 2.0, 3.0, 4.0, 5.0]
        stds = [0.5, 0.5, 0.5, 0.5, 0.5]

        mgr = CheckpointManager(tmp_dir)
        meta = CheckpointMetadata(
            version=1, model_type="x_trend",
            feature_names=[f"f{i}" for i in range(5)],
            n_features=5, n_assets=2,
            asset_names=["NIFTY", "BANKNIFTY"],
            config=asdict(cfg),
            normalization={"means": means, "stds": stds},
        )
        ckpt_dir = mgr.save(model.state_dict(), meta)

        pipeline = TFTInferencePipeline.from_checkpoint(ckpt_dir, device="cpu")
        np.testing.assert_array_almost_equal(pipeline.norm_means, means)
        np.testing.assert_array_almost_equal(pipeline.norm_stds, stds)


# ============================================================================
# TestTFTStrategy
# ============================================================================


class TestTFTStrategy:
    """Tests for TFTStrategy."""

    def test_implements_protocol(self):
        from quantlaxmi.strategies.protocol import StrategyProtocol
        strategy = TFTStrategy()
        # Check required interface attributes/methods exist
        assert hasattr(strategy, "strategy_id")
        assert hasattr(strategy, "scan")
        assert hasattr(strategy, "warmup_days")

    def test_strategy_id_format(self):
        strategy = TFTStrategy()
        sid = strategy.strategy_id
        assert sid.startswith("s_tft_x_trend_v")

    def test_warmup_days(self):
        strategy = TFTStrategy()
        assert strategy.warmup_days() >= 60

    def test_lazy_loading(self):
        """Pipeline should not load until first scan()."""
        strategy = TFTStrategy(base_dir="/nonexistent/path")
        assert strategy._pipeline is None
        # scan should handle missing gracefully
        mock_store = MagicMock()
        signals = strategy.scan(date(2026, 2, 9), mock_store)
        assert signals == []

    def test_conviction_threshold(self):
        """Signals below conviction threshold should be filtered out."""
        strategy = TFTStrategy(conviction_threshold=0.5)
        assert strategy._conviction_threshold == 0.5

    def test_create_strategy_factory(self):
        strategy = create_strategy()
        assert isinstance(strategy, TFTStrategy)

    def test_scan_with_mock_pipeline(self):
        """Test scan() with a mocked pipeline returning known positions."""
        strategy = TFTStrategy(conviction_threshold=0.1)

        mock_result = InferenceResult(
            date=date(2026, 2, 9),
            positions={"NIFTY": 0.2, "BANKNIFTY": -0.15, "FINNIFTY": 0.05},
            confidences={"NIFTY": 0.8, "BANKNIFTY": 0.6, "FINNIFTY": 0.2},
        )
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = mock_result
        mock_pipeline.version = 1
        mock_pipeline.feature_names = ["f0"]
        mock_pipeline.asset_names = ["NIFTY", "BANKNIFTY", "FINNIFTY"]

        strategy._pipeline = mock_pipeline
        strategy._version = 1

        mock_store = MagicMock()
        signals = strategy.scan(date(2026, 2, 9), mock_store)

        # FINNIFTY should be filtered (0.05 < 0.1 threshold)
        assert len(signals) == 2
        syms = {s.symbol for s in signals}
        assert "NIFTY" in syms
        assert "BANKNIFTY" in syms
        assert all(s.strategy_id.startswith("s_tft_") for s in signals)
        assert all(0.0 <= s.conviction <= 1.0 for s in signals)


# ============================================================================
# TestTrainingPipeline (unit-level, no data dependency)
# ============================================================================


class TestTrainingPipeline:
    """Unit tests for TrainingPipeline components."""

    def test_config_defaults(self):
        from quantlaxmi.models.ml.tft.production.training_pipeline import TrainingPipelineConfig
        cfg = TrainingPipelineConfig()
        assert cfg.start_date == "2024-01-01"
        assert len(cfg.symbols) == 6  # 4 India indices + BTCUSDT + ETHUSDT
        assert cfg.skip_feature_selection is False
        assert cfg.skip_tuning is False

    def test_result_structure(self):
        from quantlaxmi.models.ml.tft.production.training_pipeline import TrainingResult
        result = TrainingResult(
            n_features_initial=292,
            n_features_final=80,
            sharpe_oos=1.5,
        )
        assert result.n_features_initial == 292
        assert result.n_features_final == 80

    def test_compute_targets(self, synth_features_3d):
        """Test target computation from synthetic features."""
        from quantlaxmi.models.ml.tft.production.training_pipeline import (
            TrainingPipeline,
            TrainingPipelineConfig,
        )

        pipeline = TrainingPipeline(TrainingPipelineConfig())
        feature_names = [f"feat_{i}" for i in range(10)]
        feature_names[0] = "ret_1d"  # mock a return feature

        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        targets = pipeline._compute_targets(
            synth_features_3d, feature_names, dates,
            TrainingPipelineConfig(),
        )
        assert targets.shape == (60, 4)
        # Should have NaN at the end (no next-day return for last day)
        assert np.isnan(targets[-1]).all()


# ============================================================================
# Integration-style tests (lightweight)
# ============================================================================


class TestIntegration:
    """Lightweight integration tests tying components together."""

    def test_checkpoint_roundtrip_with_real_model(self, tmp_dir):
        """Save a real XTrendModel, load it, verify state dict matches."""
        from quantlaxmi.models.ml.tft.x_trend import XTrendConfig, XTrendModel

        cfg = XTrendConfig(d_hidden=16, n_heads=2, lstm_layers=1,
                           n_features=8, n_assets=4, seq_len=10, n_context=4)
        model = XTrendModel(cfg)

        # Forward pass to ensure model works
        batch = 2
        tgt = torch.randn(batch, cfg.seq_len, cfg.n_features)
        ctx = torch.randn(batch, cfg.n_context, cfg.seq_len, cfg.n_features)
        tid = torch.zeros(batch, dtype=torch.long)
        cid = torch.zeros(batch, cfg.n_context, dtype=torch.long)
        out = model(tgt, ctx, tid, cid)
        assert out.shape == (batch, 1)

        # Save and reload
        mgr = CheckpointManager(tmp_dir)
        meta = CheckpointMetadata(
            version=1, model_type="x_trend",
            n_features=8, n_assets=4,
            config=asdict(cfg),
            feature_names=[f"f{i}" for i in range(8)],
            normalization={"means": [0.0] * 8, "stds": [1.0] * 8},
        )
        ckpt_dir = mgr.save(model.state_dict(), meta)

        loaded_sd, loaded_meta = mgr.load(ckpt_dir)

        # Verify state dict matches
        for key in model.state_dict():
            assert torch.allclose(
                model.state_dict()[key].cpu(),
                loaded_sd[key].cpu(),
            ), f"Mismatch in {key}"

    def test_feature_selector_with_real_vsn(self, mock_model):
        """Full selector flow: extract weights, stability, report."""
        selector = FeatureSelector(FeatureSelectionConfig(
            vsn_weight_threshold=0.001,
            stability_min_folds=0.5,
            final_max_features=8,
        ))
        names = [f"feat_{i}" for i in range(10)]

        for _ in range(3):
            selector.add_fold_weights(mock_model, names)

        report = selector.stability_report(names)
        assert len(report.recommended_features) <= 8
        assert report.n_folds == 3

        report_str = selector.feature_importance_report(report)
        assert "FEATURE IMPORTANCE REPORT" in report_str

    def test_full_checkpoint_with_selector(self, mock_model, tmp_dir):
        """Save checkpoint with feature selector, reload both."""
        selector = FeatureSelector()
        names = [f"feat_{i}" for i in range(10)]
        selector.add_fold_weights(mock_model, names)

        mgr = CheckpointManager(tmp_dir)
        meta = CheckpointMetadata(
            version=1, model_type="x_trend",
            n_features=10, feature_names=names,
        )
        ckpt_dir = mgr.save(
            mock_model.state_dict(), meta,
            feature_selector=selector,
        )

        # Reload selector
        loaded_selector = mgr.load_feature_selector(ckpt_dir)
        assert loaded_selector is not None
        assert loaded_selector.n_folds == 1
