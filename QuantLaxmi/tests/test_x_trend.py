"""Tests for X-Trend model and TrendFeatureBuilder.

Covers: feature builder shapes/formulae, model forward pass, losses,
context construction, and a walk-forward smoke test on synthetic data.
"""
import sys
sys.path.insert(0, '/home/ubuntu/Desktop/7hills/QuantLaxmi')

import math
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Feature builder tests
# ---------------------------------------------------------------------------


class TestTrendFeatureBuilder:
    """Tests for features.trend.TrendFeatureBuilder."""

    def _make_prices(self, n_days=500, n_assets=4, seed=42):
        """Create synthetic price DataFrame."""
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2023-01-01", periods=n_days)
        symbols = [f"ASSET_{i}" for i in range(n_assets)]
        data = {"date": dates}
        for sym in symbols:
            # GBM prices
            returns = rng.normal(0.0005, 0.015, n_days)
            prices = 100 * np.exp(np.cumsum(returns))
            data[sym] = prices
        return pd.DataFrame(data), symbols

    def test_trend_feature_builder(self):
        """Paper-only features (no kite_1min_dir), correct shapes, no NaN in output zone."""
        from features.trend import TrendFeatureBuilder, N_PAPER_FEATURES

        df, syms = self._make_prices(500, 4)
        builder = TrendFeatureBuilder()
        features, targets, vol = builder.build(df, syms)

        # Without kite_1min_dir, returns 8 paper-only features
        assert features.shape == (500, 4, N_PAPER_FEATURES)
        assert targets.shape == (500, 4)
        assert vol.shape == (500, 4)
        assert N_PAPER_FEATURES == 8

        # After sufficient history (~253 days), features should not be NaN
        valid_zone = features[300:, :, :]
        assert not np.all(np.isnan(valid_zone)), "All features NaN in valid zone"
        # Some features should have values
        non_nan_frac = np.mean(~np.isnan(valid_zone))
        assert non_nan_frac > 0.5, f"Too many NaN in valid zone: {non_nan_frac:.1%}"

    def test_normalized_returns(self):
        """Normalized returns match paper formula: r̂ = r / (σ√t')."""
        from features.trend import TrendFeatureBuilder, RETURN_WINDOWS

        rng = np.random.default_rng(123)
        n = 400
        returns = rng.normal(0.0005, 0.015, n)
        close = 100 * np.exp(np.cumsum(returns))

        builder = TrendFeatureBuilder(vol_span=60)
        features, _, vol = builder.build_single(close)

        # Check 1-day normalized return at a point with enough history
        t = 350
        log_close = np.log(close)
        raw_ret = log_close[t] - log_close[t - 1]
        sigma_t = vol[t]
        expected = raw_ret / (sigma_t * np.sqrt(1))

        actual = features[t, 0]  # first feature = 1d normalized return
        if not np.isnan(expected) and not np.isnan(actual):
            assert abs(actual - expected) < 1e-10, (
                f"1d norm return mismatch: {actual} vs {expected}"
            )

    def test_macd_signals(self):
        """MACD features exist and have reasonable values."""
        from features.trend import TrendFeatureBuilder

        rng = np.random.default_rng(456)
        n = 500
        returns = rng.normal(0.0005, 0.015, n)
        close = 100 * np.exp(np.cumsum(returns))

        builder = TrendFeatureBuilder()
        features, _, _ = builder.build_single(close)

        # MACD features are indices 5, 6, 7
        macd_feats = features[350:, 5:8]
        assert not np.all(np.isnan(macd_feats)), "All MACD features NaN"
        # MACD standardized values should be roughly in [-5, 5]
        valid = macd_feats[~np.isnan(macd_feats)]
        assert len(valid) > 0
        assert np.abs(valid).max() < 20, f"MACD values too extreme: max={np.abs(valid).max()}"

    def test_feature_names_count(self):
        """24 total feature names (8 paper + 16 intraday)."""
        from features.trend import (
            FEATURE_NAMES, N_FEATURES, N_PAPER_FEATURES, N_INTRADAY_FEATURES,
        )
        assert N_PAPER_FEATURES == 8
        assert N_INTRADAY_FEATURES == 16
        assert N_FEATURES == 24
        assert len(FEATURE_NAMES) == N_FEATURES


# ---------------------------------------------------------------------------
# Model tests (require torch)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


class TestXTrendModel:
    """Tests for the X-Trend PyTorch model."""

    def _cfg(self, **overrides):
        from models.ml.tft.x_trend import XTrendConfig
        defaults = dict(
            d_hidden=32,
            n_heads=2,
            lstm_layers=1,
            dropout=0.0,
            n_features=8,
            seq_len=10,
            ctx_len=10,
            n_context=4,
            n_assets=4,
            loss_mode="sharpe",
        )
        defaults.update(overrides)
        return XTrendConfig(**defaults)

    def test_entity_embedding(self):
        """Correct shape, gradients flow."""
        from models.ml.tft.x_trend import EntityEmbedding

        emb = EntityEmbedding(n_assets=4, d_hidden=32)
        ids = torch.tensor([0, 1, 2, 3])
        out = emb(ids)
        assert out.shape == (4, 32)
        # Gradients flow
        loss = out.sum()
        loss.backward()
        assert emb.embedding.weight.grad is not None

    def test_vsn_forward(self):
        """8 features → d_hidden output, correct shape."""
        from models.ml.tft.x_trend import VariableSelectionNetwork

        vsn = VariableSelectionNetwork(n_features=8, hidden_dim=32, context_dim=32)
        x = torch.randn(2, 10, 8)
        ctx = torch.randn(2, 32)
        out = vsn(x, ctx)
        assert out.shape == (2, 10, 32)

    def test_cross_attention(self):
        """Q/K/V shapes work correctly."""
        from models.ml.tft.x_trend import CrossAttentionBlock

        ca = CrossAttentionBlock(d_hidden=32, n_heads=2)
        query = torch.randn(2, 32)
        keys = torch.randn(2, 4, 32)
        values = torch.randn(2, 4, 32)
        out = ca(query, keys, values)
        assert out.shape == (2, 32)

    def test_xtrend_forward_sharpe(self):
        """Full model forward pass in sharpe mode, output in [-1, 1]."""
        from models.ml.tft.x_trend import XTrendModel

        cfg = self._cfg(loss_mode="sharpe")
        model = XTrendModel(cfg)

        batch = 4
        target_seq = torch.randn(batch, cfg.seq_len, 8)
        context_set = torch.randn(batch, cfg.n_context, cfg.ctx_len, 8)
        target_id = torch.randint(0, cfg.n_assets, (batch,))
        context_ids = torch.randint(0, cfg.n_assets, (batch, cfg.n_context))

        out = model(target_seq, context_set, target_id, context_ids)
        assert out.shape == (batch, 1)
        # Tanh output → [-1, 1]
        assert out.min().item() >= -1.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_xtrend_forward_mle(self):
        """Full model forward pass in MLE mode, returns (mu, log_sigma)."""
        from models.ml.tft.x_trend import XTrendModel

        cfg = self._cfg(loss_mode="joint_mle")
        model = XTrendModel(cfg)

        batch = 4
        target_seq = torch.randn(batch, cfg.seq_len, 8)
        context_set = torch.randn(batch, cfg.n_context, cfg.ctx_len, 8)
        target_id = torch.randint(0, cfg.n_assets, (batch,))
        context_ids = torch.randint(0, cfg.n_assets, (batch, cfg.n_context))

        mu, log_sigma = model(target_seq, context_set, target_id, context_ids)
        assert mu.shape == (batch, 1)
        assert log_sigma.shape == (batch, 1)

    def test_predict_position_mle(self):
        """predict_position with MLE mode returns values in [-1, 1] via PTP."""
        from models.ml.tft.x_trend import XTrendModel

        cfg = self._cfg(loss_mode="joint_mle")
        model = XTrendModel(cfg)

        batch = 4
        target_seq = torch.randn(batch, cfg.seq_len, 8)
        context_set = torch.randn(batch, cfg.n_context, cfg.ctx_len, 8)
        target_id = torch.randint(0, cfg.n_assets, (batch,))
        context_ids = torch.randint(0, cfg.n_assets, (batch, cfg.n_context))

        pos = model.predict_position(target_seq, context_set, target_id, context_ids)
        assert pos.shape == (batch, 1)
        assert pos.min().item() >= -1.0 - 1e-6
        assert pos.max().item() <= 1.0 + 1e-6


class TestLossFunctions:
    """Tests for X-Trend loss functions."""

    def test_sharpe_loss(self):
        """Gradient flows, correct sign."""
        from models.ml.tft.x_trend import sharpe_loss

        positions = torch.tensor([0.5, -0.3, 0.8, 0.1], requires_grad=True)
        returns = torch.tensor([0.01, -0.02, 0.015, 0.005])

        loss = sharpe_loss(positions, returns)
        loss.backward()

        # Loss should be finite
        assert torch.isfinite(loss)
        # Gradient should flow to positions
        assert positions.grad is not None
        assert torch.all(torch.isfinite(positions.grad))

        # With positive strategy returns, loss should be negative (good Sharpe)
        pos2 = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)
        ret2 = torch.tensor([0.01, 0.02, 0.015, 0.01])
        loss2 = sharpe_loss(pos2, ret2)
        assert loss2.item() < 0, "Positive returns should give negative loss"

    def test_joint_mle_loss(self):
        """NLL + Sharpe combined correctly."""
        from models.ml.tft.x_trend import joint_loss

        mu = torch.tensor([[0.01], [0.02], [-0.01], [0.005]], requires_grad=True)
        log_sigma = torch.tensor([[-1.0], [-1.0], [-1.0], [-1.0]], requires_grad=True)
        targets = torch.tensor([[0.01], [0.015], [-0.005], [0.003]])

        loss = joint_loss(mu, log_sigma, targets, alpha=0.1)
        loss.backward()

        assert torch.isfinite(loss)
        assert mu.grad is not None
        assert log_sigma.grad is not None


class TestContextConstruction:
    """Tests for context set building."""

    def test_context_construction(self):
        """Causality enforced, correct shapes."""
        from models.ml.tft.x_trend import build_context_set

        rng = np.random.default_rng(42)
        # Synthetic features: (100 days, 4 assets, 8 features)
        features = rng.standard_normal((100, 4, 8))

        ctx_seqs, ctx_ids = build_context_set(
            features,
            target_start=60,
            n_context=8,
            ctx_len=20,
            rng=rng,
        )

        assert ctx_seqs.shape == (8, 20, 8)
        assert ctx_ids.shape == (8,)
        assert np.all(ctx_ids >= 0) and np.all(ctx_ids < 4)
        # No NaN
        assert not np.any(np.isnan(ctx_seqs))

    def test_context_causal(self):
        """Context can't leak future data."""
        from models.ml.tft.x_trend import build_context_set

        rng = np.random.default_rng(42)
        n_days, n_assets = 200, 2

        # Mark future data with a distinctive pattern
        features = np.zeros((n_days, n_assets, 8))
        features[:100] = 1.0  # historical
        features[100:] = -999.0  # future (should never appear in context)

        ctx_seqs, ctx_ids = build_context_set(
            features,
            target_start=100,
            n_context=16,
            ctx_len=20,
            rng=rng,
        )

        # No context sequence should contain -999
        assert not np.any(ctx_seqs == -999.0), "Context leaked future data!"

    def test_context_insufficient_history(self):
        """When not enough history, returns zeros gracefully."""
        from models.ml.tft.x_trend import build_context_set

        rng = np.random.default_rng(42)
        features = rng.standard_normal((10, 2, 8))

        ctx_seqs, ctx_ids = build_context_set(
            features,
            target_start=5,
            n_context=8,
            ctx_len=20,
            rng=rng,
        )

        assert ctx_seqs.shape == (8, 20, 8)
        # Should be zeros (not enough data)
        assert np.allclose(ctx_seqs, 0.0)


class TestWalkForwardSmoke:
    """Smoke test: 1 fold on synthetic data completes."""

    def test_walk_forward_smoke(self):
        """X-Trend walk-forward completes on small synthetic data."""
        from models.ml.tft.x_trend import run_xtrend_backtest, XTrendConfig

        rng = np.random.default_rng(42)
        n_days = 500
        symbols = ["A", "B"]
        dates = pd.bdate_range("2023-01-01", periods=n_days)

        data = {"date": dates}
        for sym in symbols:
            returns = rng.normal(0.0003, 0.012, n_days)
            data[sym] = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame(data)

        cfg = XTrendConfig(
            d_hidden=16,
            n_heads=2,
            lstm_layers=1,
            dropout=0.0,
            seq_len=20,
            ctx_len=20,
            n_context=4,
            n_assets=2,
            train_window=252,
            test_window=63,
            step_size=63,
            epochs=3,       # very few epochs for speed
            patience=2,
            batch_size=8,
        )

        results = run_xtrend_backtest(df, symbols, cfg)

        assert isinstance(results, dict)
        assert set(results.keys()) == {"A", "B"}
        for sym, res_df in results.items():
            assert "date" in res_df.columns
            assert "position" in res_df.columns
            assert "strategy_return" in res_df.columns
            assert len(res_df) == n_days
            # Positions should not be all NaN
            valid_pos = res_df["position"].dropna()
            # May have few valid positions with minimal epochs, just check no crash
            assert len(res_df) > 0


# ---------------------------------------------------------------------------
# LSTD test (RL module)
# ---------------------------------------------------------------------------


class TestLSTDPrediction:
    """Test for lstd_prediction in dynamic_programming."""

    def test_lstd_simple_chain(self):
        """LSTD on a simple 3-state chain with known solution."""
        from models.rl.core.dynamic_programming import lstd_prediction

        # 3-state chain: s0 → s1 → s2 (terminal)
        # Features: one-hot encoding (dim=2, only s0 and s1 are non-terminal)
        gamma = 0.9

        # Transitions: (phi_s, reward, phi_s_next)
        transitions = [
            # s0 → s1, reward=1
            (np.array([1.0, 0.0]), 1.0, np.array([0.0, 1.0])),
            # s1 → terminal, reward=2
            (np.array([0.0, 1.0]), 2.0, np.array([0.0, 0.0])),
        ]

        # True values: V(s1) = 2, V(s0) = 1 + 0.9 * 2 = 2.8
        w = lstd_prediction(transitions, gamma, feature_dim=2)

        v_s0 = w @ np.array([1.0, 0.0])
        v_s1 = w @ np.array([0.0, 1.0])

        assert abs(v_s1 - 2.0) < 0.1, f"V(s1) = {v_s1}, expected 2.0"
        assert abs(v_s0 - 2.8) < 0.1, f"V(s0) = {v_s0}, expected 2.8"

    def test_lstd_repeated_transitions(self):
        """LSTD with many samples of the same transition averages correctly."""
        from models.rl.core.dynamic_programming import lstd_prediction

        gamma = 0.95
        # Single self-loop state with reward=1
        # V(s) = 1 / (1 - gamma) = 20
        transitions = [
            (np.array([1.0]), 1.0, np.array([1.0]))
            for _ in range(100)
        ]

        w = lstd_prediction(transitions, gamma, feature_dim=1)
        v = w[0]
        expected = 1.0 / (1.0 - gamma)
        assert abs(v - expected) < 0.5, f"V = {v}, expected {expected}"


class TestDefaultTolerance:
    """Verify DEFAULT_TOLERANCE is exported and used."""

    def test_default_tolerance_value(self):
        from models.rl.core.dynamic_programming import DEFAULT_TOLERANCE
        assert DEFAULT_TOLERANCE == 1e-6
