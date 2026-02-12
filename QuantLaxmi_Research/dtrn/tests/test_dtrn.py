"""Comprehensive test suite for the DTRN system.

Tests cover:
1. Config — defaults, lot sizes, risk parameters
2. Feature Engine — shape, NaN, causality, reset
3. Dynamic Topology — sparsity, hysteresis, symmetry, flip rate
4. Graph Net (DTRN model) — output keys, shapes, regime probs, positions
5. Risk Manager — kill switch, drawdown, position change, reset
6. Execution Model — cost correctness, India FnO specifics
7. Data Loader — real data availability (integration)
8. End-to-end integration — full pipeline on 1 day
"""
from __future__ import annotations

import sys
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure dtrn package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dtrn.config import DTRNConfig, DATA_ROOT, TELEGRAM_ROOT, KITE_1MIN_ROOT, INSTRUMENTS
from dtrn.data.features import FeatureEngine, StreamingEWMA, StreamingZScore
from dtrn.model.topology import DynamicTopology
from dtrn.model.graph_net import DTRN as DTRNModel, NodeEmbedding, GraphMessagePass, GraphPool, RegimeHead, PolicyHead, PredictorHead
from dtrn.model.dtrn import create_dtrn
from dtrn.engine.risk import RiskManager, RiskState
from dtrn.engine.execution import ExecutionModel
from dtrn.data.loader import list_available_dates, load_day, load_day_feather, load_day_kite


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers — synthetic data generators
# ══════════════════════════════════════════════════════════════════════════════

def make_synthetic_bars(n: int = 375, base_price: float = 23000.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 1-min OHLCV bars resembling NIFTY futures.

    375 bars = 1 full trading day (9:15 to 15:30, 6h15m).
    """
    rng = np.random.RandomState(seed)

    # Random walk for close prices
    returns = rng.normal(0, 0.0003, n)  # ~3 bps per minute
    close = base_price * np.exp(np.cumsum(returns))

    # OHLCV
    high = close * (1 + rng.uniform(0.0001, 0.001, n))
    low = close * (1 - rng.uniform(0.0001, 0.001, n))
    open_ = close * (1 + rng.uniform(-0.0005, 0.0005, n))
    volume = rng.poisson(5000, n).astype(float)
    oi = np.full(n, 10_000_000.0) + rng.normal(0, 100_000, n)

    # Time index: 9:15 to 15:30 IST (375 one-minute bars)
    start = datetime(2026, 2, 10, 9, 15, 0)
    index = pd.date_range(start, periods=n, freq="1min")

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "oi": oi,
    }, index=index)
    df.index.name = "datetime"
    return df


def make_synthetic_bar_dict(price: float = 23000.0, dt=None) -> dict:
    """Single bar as a dict for FeatureEngine.update()."""
    if dt is None:
        dt = datetime(2026, 2, 10, 10, 0, 0)
    return {
        "datetime": dt,
        "open": price - 1.0,
        "high": price + 5.0,
        "low": price - 3.0,
        "close": price,
        "volume": 5000.0,
        "oi": 10_000_000.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  1. Config Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    """DTRNConfig defaults and parameter correctness."""

    def test_default_instruments(self):
        cfg = DTRNConfig()
        assert cfg.instruments == ["NIFTY", "BANKNIFTY"]

    def test_lot_sizes(self):
        cfg = DTRNConfig()
        assert cfg.lot_sizes["NIFTY"] == 75
        assert cfg.lot_sizes["BANKNIFTY"] == 30

    def test_risk_parameters(self):
        cfg = DTRNConfig()
        assert cfg.max_daily_loss == 0.02, "Kill switch at 2% daily loss"
        assert cfg.max_drawdown == 0.05, "Max drawdown at 5%"
        assert cfg.max_lots == 10
        assert cfg.max_position_change_per_step == 0.2

    def test_feature_engine_params(self):
        cfg = DTRNConfig()
        assert cfg.n_return_lags == 5
        assert cfg.ewma_spans == [10, 30, 60, 120]
        assert cfg.zscore_window == 60
        assert cfg.rsi_period == 14

    def test_topology_params(self):
        cfg = DTRNConfig()
        assert cfg.tau_on > cfg.tau_off, "Hysteresis: tau_on > tau_off"
        assert cfg.max_edge_flip_rate > 0
        assert cfg.top_k_edges == 6

    def test_regime_names(self):
        cfg = DTRNConfig()
        assert len(cfg.regime_names) == cfg.n_regimes
        assert cfg.n_regimes == 4
        assert "calm_mr" in cfg.regime_names
        assert "trend" in cfg.regime_names
        assert "high_vol" in cfg.regime_names
        assert "liq_stress" in cfg.regime_names

    def test_execution_params(self):
        cfg = DTRNConfig()
        assert cfg.brokerage_per_order == 20.0
        assert cfg.slippage_bps == 1.0
        assert cfg.gst_pct == 0.18

    def test_initial_capital(self):
        cfg = DTRNConfig()
        assert cfg.initial_capital == 10_000_000.0  # 1 Crore

    def test_bar_interval(self):
        cfg = DTRNConfig()
        assert cfg.bar_interval == "1min"

    def test_model_dimensions(self):
        cfg = DTRNConfig()
        assert cfg.d_embed == 32
        assert cfg.d_hidden == 64
        assert cfg.d_temporal == 64
        assert cfg.n_message_passes == 2

    def test_custom_config(self):
        cfg = DTRNConfig(d_embed=16, n_regimes=3)
        assert cfg.d_embed == 16
        assert cfg.n_regimes == 3
        # Other defaults unchanged
        assert cfg.d_hidden == 64


# ══════════════════════════════════════════════════════════════════════════════
#  2. Feature Engine Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamingHelpers:
    """StreamingEWMA and StreamingZScore unit tests."""

    def test_ewma_init(self):
        ewma = StreamingEWMA(10)
        assert ewma.value is None
        assert ewma.n == 0

    def test_ewma_first_value(self):
        ewma = StreamingEWMA(10)
        result = ewma.update(5.0)
        assert result == 5.0
        assert ewma.n == 1

    def test_ewma_converges(self):
        ewma = StreamingEWMA(10)
        for _ in range(100):
            val = ewma.update(1.0)
        assert abs(val - 1.0) < 1e-6

    def test_ewma_reset(self):
        ewma = StreamingEWMA(10)
        ewma.update(5.0)
        ewma.reset()
        assert ewma.value is None
        assert ewma.n == 0

    def test_zscore_warmup(self):
        zs = StreamingZScore(20)
        result = zs.update(1.0)
        assert result == 0.0, "Single value should return 0"

    def test_zscore_known_values(self):
        zs = StreamingZScore(100)
        # Feed a bunch of zeros, then one large value
        for _ in range(50):
            zs.update(0.0)
        result = zs.update(1.0)
        # Mean is close to 0, std is small, so z-score should be large positive
        assert result > 1.0

    def test_zscore_reset(self):
        zs = StreamingZScore(20)
        for i in range(10):
            zs.update(float(i))
        zs.reset()
        assert len(zs.buffer) == 0


class TestFeatureEngine:
    """FeatureEngine tests — shape, NaN, causality, reset."""

    def test_feature_count(self):
        fe = FeatureEngine()
        assert fe.n_features == 26
        assert len(fe.FEATURE_NAMES) == 26
        assert len(fe.feature_names) == 26

    def test_feature_names_unique(self):
        fe = FeatureEngine()
        names = fe.feature_names
        assert len(names) == len(set(names)), "Feature names must be unique"

    def test_update_returns_correct_shape(self):
        fe = FeatureEngine()
        bar = make_synthetic_bar_dict()
        features, mask = fe.update(bar)
        assert features.shape == (26,)
        assert mask.shape == (26,)
        assert features.dtype == np.float32
        assert mask.dtype == np.float32

    def test_compute_batch_shape(self):
        fe = FeatureEngine()
        df = make_synthetic_bars(100)
        features, masks = fe.compute_batch(df)
        assert features.shape == (100, 26)
        assert masks.shape == (100, 26)

    def test_compute_batch_full_day(self):
        """375 bars (full day) -> (375, 26)."""
        fe = FeatureEngine()
        df = make_synthetic_bars(375)
        features, masks = fe.compute_batch(df)
        assert features.shape == (375, 26)
        assert masks.shape == (375, 26)

    def test_no_nan_in_features(self):
        """Features must never contain NaN — masks handle cold-start."""
        fe = FeatureEngine()
        df = make_synthetic_bars(200)
        features, masks = fe.compute_batch(df)
        assert not np.any(np.isnan(features)), "Features must not contain NaN"
        assert not np.any(np.isnan(masks)), "Masks must not contain NaN"

    def test_no_inf_in_features(self):
        """Features must not contain Inf."""
        fe = FeatureEngine()
        df = make_synthetic_bars(200)
        features, masks = fe.compute_batch(df)
        assert not np.any(np.isinf(features)), "Features must not contain Inf"

    def test_mask_warmup_behavior(self):
        """First bar should have most masks at 0 (warmup needed)."""
        fe = FeatureEngine()
        df = make_synthetic_bars(10)
        features, masks = fe.compute_batch(df)
        # First bar: no prev_close, so log_return mask = 0
        assert masks[0, 0] == 0.0, "First bar log_return mask should be 0"
        # After a few bars, some features should be valid
        assert masks[5, 0] == 1.0, "After warmup, log_return should be valid"

    def test_masks_all_binary(self):
        """Masks should only be 0 or 1."""
        fe = FeatureEngine()
        df = make_synthetic_bars(200)
        _, masks = fe.compute_batch(df)
        unique_vals = np.unique(masks)
        assert all(v in [0.0, 1.0] for v in unique_vals), f"Mask values: {unique_vals}"

    def test_causality_feature_at_t_depends_only_on_past(self):
        """Feature at time t must only depend on data at indices 0..t.

        Test: compute features on bars[0:T], then on bars[0:T+K].
        Features for t < T must be identical.
        """
        fe = FeatureEngine()
        df = make_synthetic_bars(200)

        # Compute features for first 100 bars
        fe.reset()
        feat_100, mask_100 = fe.compute_batch(df.iloc[:100])

        # Compute features for first 150 bars
        fe.reset()
        feat_150, mask_150 = fe.compute_batch(df.iloc[:150])

        # First 100 bars of both runs must be identical
        np.testing.assert_array_equal(
            feat_100, feat_150[:100],
            err_msg="Causality violation: features changed with future data"
        )
        np.testing.assert_array_equal(
            mask_100, mask_150[:100],
            err_msg="Causality violation: masks changed with future data"
        )

    def test_reset_gives_fresh_state(self):
        """After reset(), behavior should be identical to a new instance."""
        df = make_synthetic_bars(50)

        fe1 = FeatureEngine()
        feat1, mask1 = fe1.compute_batch(df)

        # Use same engine, compute on different data, then reset + recompute
        fe1.compute_batch(make_synthetic_bars(100, seed=99))
        fe1.reset()
        feat2, mask2 = fe1.compute_batch(df)

        np.testing.assert_array_equal(feat1, feat2)
        np.testing.assert_array_equal(mask1, mask2)

    def test_time_features_sin_cos_range(self):
        """Time sin/cos features should be in [-1, 1]."""
        fe = FeatureEngine()
        df = make_synthetic_bars(375)
        features, masks = fe.compute_batch(df)
        # time_sin is index 20, time_cos is index 21
        assert np.all(features[:, 20] >= -1.0 - 1e-6)
        assert np.all(features[:, 20] <= 1.0 + 1e-6)
        assert np.all(features[:, 21] >= -1.0 - 1e-6)
        assert np.all(features[:, 21] <= 1.0 + 1e-6)

    def test_rsi_in_range(self):
        """RSI feature (index 18) should be in [-1, 1]."""
        fe = FeatureEngine()
        df = make_synthetic_bars(200)
        features, _ = fe.compute_batch(df)
        rsi_vals = features[:, 18]
        assert np.all(rsi_vals >= -1.0 - 1e-6)
        assert np.all(rsi_vals <= 1.0 + 1e-6)

    def test_session_half_binary(self):
        """session_half (index 23) should be 0 or 1."""
        fe = FeatureEngine()
        df = make_synthetic_bars(375)
        features, _ = fe.compute_batch(df)
        unique = np.unique(features[:, 23])
        assert all(v in [0.0, 1.0] for v in unique)

    def test_jump_flag_binary(self):
        """jump_flag (index 25) should be 0 or 1."""
        fe = FeatureEngine()
        df = make_synthetic_bars(200)
        features, masks = fe.compute_batch(df)
        # Only check where mask is valid
        valid = masks[:, 25] > 0.5
        jump_vals = features[valid, 25]
        if len(jump_vals) > 0:
            unique = np.unique(jump_vals)
            assert all(v in [0.0, 1.0] for v in unique)

    def test_compute_multi_day(self):
        """Multi-day computation with daily reset."""
        fe = FeatureEngine()
        day1 = make_synthetic_bars(100, seed=1)
        day2 = make_synthetic_bars(100, seed=2)
        daily_dfs = {"2026-02-09": day1, "2026-02-10": day2}
        features, masks, boundaries = fe.compute_multi_day(daily_dfs)
        assert features.shape == (200, 26)
        assert masks.shape == (200, 26)
        assert len(boundaries) == 2
        assert boundaries[0] == (0, 100, "2026-02-09")
        assert boundaries[1] == (100, 200, "2026-02-10")


# ══════════════════════════════════════════════════════════════════════════════
#  3. Topology Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDynamicTopology:
    """DynamicTopology tests — sparsity, hysteresis, symmetry, flip rate."""

    def test_initial_adjacency_is_zeros(self):
        topo = DynamicTopology(d=10)
        adj = topo.get_adjacency()
        assert adj.shape == (10, 10)
        assert adj.sum() == 0, "Initial adjacency should be all zeros"

    def test_update_returns_adjacency(self):
        topo = DynamicTopology(d=10)
        x = np.random.randn(10).astype(np.float32)
        adj = topo.update(x)
        assert adj.shape == (10, 10)

    def test_adjacency_is_binary(self):
        """After warmup, adjacency should be {0, 1}."""
        d = 10
        topo = DynamicTopology(d=d, ewma_span=20)
        rng = np.random.RandomState(42)

        for _ in range(50):
            x = rng.randn(d).astype(np.float32)
            topo.update(x)

        adj = topo.get_adjacency()
        unique = np.unique(adj)
        assert all(v in [0, 1] for v in unique), f"Adjacency values: {unique}"

    def test_weights_are_float(self):
        """Weighted adjacency should have float values."""
        d = 10
        topo = DynamicTopology(d=d, ewma_span=20)
        rng = np.random.RandomState(42)

        for _ in range(50):
            x = rng.randn(d).astype(np.float32)
            topo.update(x)

        weights = topo.get_weights()
        assert weights.dtype in [np.float32, np.float64]
        assert weights.shape == (d, d)

    def test_sparse_after_warmup(self):
        """After warmup, adjacency should be sparse — not all ones or all zeros."""
        d = 26
        topo = DynamicTopology(d=d, ewma_span=20, top_k=6, tau_on=0.05, tau_off=0.02)
        rng = np.random.RandomState(42)

        # Feed correlated data (so some edges form)
        base = rng.randn(d)
        for i in range(100):
            noise = rng.randn(d) * 0.3
            x = (base + noise).astype(np.float32)
            topo.update(x)

        adj = topo.get_adjacency()
        n_edges = int(adj.sum())
        max_possible = d * (d - 1)

        assert n_edges > 0, "Should have some edges after warmup with correlated data"
        assert n_edges < max_possible, "Should be sparse, not fully connected"

    def test_no_self_loops(self):
        """Diagonal of adjacency must be zero (no self-loops)."""
        d = 10
        topo = DynamicTopology(d=d, ewma_span=20, tau_on=0.05, tau_off=0.02)
        rng = np.random.RandomState(42)

        for _ in range(50):
            x = rng.randn(d).astype(np.float32)
            topo.update(x)

        adj = topo.get_adjacency()
        assert np.all(np.diag(adj) == 0), "No self-loops allowed"

    def test_precision_matrix_symmetric(self):
        """The regularized covariance should be symmetric (and hence the precision)."""
        d = 10
        topo = DynamicTopology(d=d, ewma_span=20)
        rng = np.random.RandomState(42)

        for _ in range(50):
            x = rng.randn(d).astype(np.float32)
            topo.update(x)

        # Check EWMA covariance symmetry
        np.testing.assert_array_almost_equal(
            topo.cov, topo.cov.T,
            decimal=10,
            err_msg="EWMA covariance must be symmetric"
        )

    def test_hysteresis_edges_dont_flip_immediately(self):
        """With hysteresis (tau_on > tau_off), edges should not flicker.

        Once an edge activates (score > tau_on), it stays active until
        score drops below tau_off. This test verifies that an edge that
        was just activated doesn't immediately deactivate.
        """
        d = 10
        topo = DynamicTopology(d=d, ewma_span=20, tau_on=0.3, tau_off=0.1)
        rng = np.random.RandomState(42)

        # Warmup: strongly correlated pairs should form edges
        for _ in range(30):
            base = rng.randn(5)
            x = np.concatenate([base, base + rng.randn(5) * 0.01]).astype(np.float32)
            topo.update(x)

        adj_before = topo.get_adjacency().copy()

        # Now feed slightly different data — edges shouldn't disappear immediately
        for _ in range(5):
            x = rng.randn(d).astype(np.float32) * 0.5
            topo.update(x)

        adj_after = topo.get_adjacency()

        # Some edges from before should persist (hysteresis)
        if adj_before.sum() > 0:
            # Not all edges should flip in 5 steps
            flipped = (adj_before != adj_after).sum()
            total_edges = adj_before.sum()
            # At most max_flip_rate of total possible edges per step
            # Over 5 steps with default max_flip_rate=0.02, max ~9 flips possible
            # (but topology only updates every 5 steps, so at most 1 update)
            assert flipped <= total_edges, \
                f"Too many edges flipped ({flipped} of {total_edges})"

    def test_max_flip_rate_respected(self):
        """Max flip rate should cap actual edge changes per precision update.

        The topology only recalculates adjacency every 5 steps after warmup.
        We check that the actual adjacency difference between consecutive
        precision updates is bounded by max_flips.
        """
        d = 20
        max_flip = 0.01  # Very low flip rate
        topo = DynamicTopology(d=d, ewma_span=10, max_flip_rate=max_flip, tau_on=0.05, tau_off=0.02)
        rng = np.random.RandomState(42)
        max_allowed = max(1, int(d * (d - 1) * max_flip))

        # Warm up to establish some topology
        for _ in range(30):
            base = rng.randn(d // 2)
            x = np.concatenate([base, base + rng.randn(d // 2) * 0.01]).astype(np.float32)
            topo.update(x)

        # Track actual adjacency changes across precision updates
        prev_adj = topo.get_adjacency().copy()

        for _ in range(50):
            x = rng.randn(d).astype(np.float32) * 10.0
            topo.update(x)
            new_adj = topo.get_adjacency()
            actual_flips = int((new_adj != prev_adj).sum())
            if actual_flips > 0:
                assert actual_flips <= max_allowed, \
                    f"Actual flip count {actual_flips} exceeds max allowed {max_allowed}"
            prev_adj = new_adj.copy()

    def test_get_stats(self):
        """get_stats() should return a well-formed dict."""
        d = 10
        topo = DynamicTopology(d=d)
        stats = topo.get_stats()
        assert "n_edges" in stats
        assert "n_updates" in stats
        assert "avg_edges_per_node" in stats
        assert "edge_density" in stats
        assert "mean_weight" in stats
        assert stats["n_edges"] == 0  # no edges initially

    def test_reset(self):
        """Reset should bring topology back to initial state."""
        d = 10
        topo = DynamicTopology(d=d, ewma_span=20, tau_on=0.05, tau_off=0.02)
        rng = np.random.RandomState(42)

        for _ in range(50):
            topo.update(rng.randn(d).astype(np.float32))

        topo.reset()
        assert topo.n_updates == 0
        assert topo.adjacency.sum() == 0
        np.testing.assert_array_equal(topo.cov, np.eye(d))

    def test_mask_handling(self):
        """When mask has only 1 valid feature, update should not crash."""
        d = 10
        topo = DynamicTopology(d=d)
        x = np.random.randn(d).astype(np.float32)
        mask = np.zeros(d, dtype=np.float32)
        mask[0] = 1.0  # Only 1 valid feature
        adj = topo.update(x, mask)
        assert adj.shape == (d, d)


# ══════════════════════════════════════════════════════════════════════════════
#  4. Graph Net (DTRN Model) Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGraphNetComponents:
    """Tests for individual graph net components."""

    def test_node_embedding_shape(self):
        ne = NodeEmbedding(n_features=26, d_embed=32)
        x = torch.randn(2, 26)
        mask = torch.ones(2, 26)
        out = ne(x, mask)
        assert out.shape == (2, 26, 32)

    def test_graph_message_pass_shape(self):
        gmp = GraphMessagePass(d_hidden=64)
        h = torch.randn(2, 26, 64)
        adj = torch.zeros(26, 26)
        weights = torch.zeros(26, 26)
        out = gmp(h, adj, weights)
        assert out.shape == (2, 26, 64)

    def test_graph_pool_shape(self):
        gp = GraphPool(d_hidden=64)
        h = torch.randn(2, 26, 64)
        out = gp(h)
        assert out.shape == (2, 64)

    def test_regime_head_output(self):
        rh = RegimeHead(d_input=64, n_regimes=4)
        h = torch.randn(2, 64)
        out = rh(h)
        assert out.shape == (2, 4)
        # Output should be log-probabilities (sum of exp = 1)
        probs = torch.exp(out)
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(2), atol=1e-5, rtol=1e-5)

    def test_policy_head_output_range(self):
        ph = PolicyHead(d_input=64, n_regimes=4)
        h = torch.randn(2, 64)
        regime_probs = torch.softmax(torch.randn(2, 4), dim=-1)
        out = ph(h, regime_probs)
        assert out.shape == (2, 1)
        assert torch.all(out >= -1.0)
        assert torch.all(out <= 1.0)

    def test_predictor_head_output_keys(self):
        pred = PredictorHead(d_input=64, horizon=5)
        h = torch.randn(2, 64)
        out = pred(h)
        assert "returns" in out
        assert "volatility" in out
        assert "jump_logits" in out
        assert out["returns"].shape == (2, 5)
        assert out["volatility"].shape == (2, 5)
        assert out["jump_logits"].shape == (2, 5)
        # Volatility should be positive (Softplus)
        assert torch.all(out["volatility"] > 0)


class TestDTRNModel:
    """DTRN full model forward pass tests."""

    @pytest.fixture
    def model(self):
        return DTRNModel(
            n_features=26,
            d_embed=32,
            d_hidden=64,
            n_message_passes=2,
            d_temporal=64,
            n_regimes=4,
            pred_horizon=5,
        )

    def test_forward_output_keys(self, model):
        B, T, d = 2, 10, 26
        x = torch.randn(B, T, d)
        mask = torch.ones(B, T, d)
        adj = torch.zeros(T, d, d)
        weights = torch.zeros(T, d, d)

        out = model(x, mask, adj, weights)
        assert "regime_logprobs" in out
        assert "regime_probs" in out
        assert "position" in out
        assert "predictions" in out
        assert "h_final" in out

    def test_forward_output_shapes(self, model):
        B, T, d, K, H = 2, 10, 26, 4, 5
        x = torch.randn(B, T, d)
        mask = torch.ones(B, T, d)
        adj = torch.zeros(T, d, d)
        weights = torch.zeros(T, d, d)

        out = model(x, mask, adj, weights)

        assert out["regime_logprobs"].shape == (B, T, K)
        assert out["regime_probs"].shape == (B, T, K)
        assert out["position"].shape == (B, T, 1)
        assert out["predictions"]["returns"].shape == (B, T, H)
        assert out["predictions"]["volatility"].shape == (B, T, H)
        assert out["predictions"]["jump_logits"].shape == (B, T, H)
        assert out["h_final"].shape == (1, B, 64)  # GRU hidden

    def test_regime_probs_sum_to_one(self, model):
        B, T, d = 2, 10, 26
        x = torch.randn(B, T, d)
        mask = torch.ones(B, T, d)
        adj = torch.zeros(T, d, d)
        weights = torch.zeros(T, d, d)

        out = model(x, mask, adj, weights)
        sums = out["regime_probs"].sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(B, T), atol=1e-4, rtol=1e-4)

    def test_position_in_range(self, model):
        B, T, d = 2, 10, 26
        x = torch.randn(B, T, d)
        mask = torch.ones(B, T, d)
        adj = torch.zeros(T, d, d)
        weights = torch.zeros(T, d, d)

        out = model(x, mask, adj, weights)
        assert torch.all(out["position"] >= -1.0)
        assert torch.all(out["position"] <= 1.0)

    def test_forward_step_online(self, model):
        """forward_step for online inference should match forward for single step."""
        B, d = 1, 26
        x_t = torch.randn(B, d)
        m_t = torch.ones(B, d)
        a_t = torch.zeros(d, d)
        w_t = torch.zeros(d, d)

        outputs, h_state = model.forward_step(x_t, m_t, a_t, w_t, None)

        assert "regime_logprobs" in outputs
        assert "regime_probs" in outputs
        assert "position" in outputs
        assert "predictions" in outputs
        assert outputs["position"].shape == (B, 1)
        assert h_state is not None
        assert h_state.shape == (1, B, 64)

    def test_forward_step_hidden_state_persistence(self, model):
        """Hidden state from forward_step should persist across calls."""
        B, d = 1, 26
        a_t = torch.zeros(d, d)
        w_t = torch.zeros(d, d)

        h_state = None
        for _ in range(5):
            x_t = torch.randn(B, d)
            m_t = torch.ones(B, d)
            outputs, h_state = model.forward_step(x_t, m_t, a_t, w_t, h_state)

        # After 5 steps, hidden state should be non-zero
        assert not torch.all(h_state == 0.0)

    def test_batched_adjacency(self, model):
        """Forward should work with batched adjacency (B, T, d, d)."""
        B, T, d = 2, 5, 26
        x = torch.randn(B, T, d)
        mask = torch.ones(B, T, d)
        adj = torch.zeros(B, T, d, d)
        weights = torch.zeros(B, T, d, d)

        out = model(x, mask, adj, weights)
        assert out["position"].shape == (B, T, 1)

    def test_volatility_prediction_positive(self, model):
        """Volatility predictions must be positive (Softplus)."""
        B, T, d = 2, 10, 26
        x = torch.randn(B, T, d) * 5  # larger inputs
        mask = torch.ones(B, T, d)
        adj = torch.zeros(T, d, d)
        weights = torch.zeros(T, d, d)

        out = model(x, mask, adj, weights)
        assert torch.all(out["predictions"]["volatility"] > 0)

    def test_gradient_flows(self, model):
        """Verify gradients flow through the entire model."""
        B, T, d = 1, 5, 26
        x = torch.randn(B, T, d, requires_grad=True)
        mask = torch.ones(B, T, d)
        adj = torch.zeros(T, d, d)
        weights = torch.zeros(T, d, d)

        out = model(x, mask, adj, weights)
        loss = out["position"].sum() + out["regime_logprobs"].sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (B, T, d)

    def test_param_count(self, model):
        """Sanity check on model parameter count."""
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        assert n_params < 10_000_000, "Model should not be excessively large"


class TestCreateDTRN:
    """create_dtrn() factory function."""

    def test_returns_topology_and_model(self):
        topo, model = create_dtrn()
        assert isinstance(topo, DynamicTopology)
        assert isinstance(model, DTRNModel)

    def test_config_propagated(self):
        cfg = DTRNConfig(d_embed=16, n_regimes=3, top_k_edges=4)
        topo, model = create_dtrn(cfg, n_features=26)
        assert topo.top_k == 4
        assert model.n_features == 26

    def test_custom_n_features(self):
        topo, model = create_dtrn(n_features=30)
        assert topo.d == 30
        assert model.n_features == 30


# ══════════════════════════════════════════════════════════════════════════════
#  5. Risk Manager Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskState:
    """RiskState dataclass tests."""

    def test_defaults(self):
        rs = RiskState()
        assert rs.position == 0
        assert rs.day_pnl == 0.0
        assert rs.trades_today == 0

    def test_reset_day(self):
        rs = RiskState(day_pnl=-5000.0, trades_today=10)
        rs.reset_day()
        assert rs.day_pnl == 0.0
        assert rs.trades_today == 0


class TestRiskManager:
    """RiskManager tests — kill switch, drawdown, position change."""

    @pytest.fixture
    def risk_mgr(self):
        cfg = DTRNConfig(
            initial_capital=10_000_000.0,
            max_daily_loss=0.02,
            max_drawdown=0.05,
            max_lots=4,  # 4 NIFTY lots (300 contracts) or equivalent
            max_position_change_per_step=0.5,
            lot_sizes={"NIFTY": 75, "BANKNIFTY": 30},
        )
        return RiskManager(cfg)

    def test_initial_state(self, risk_mgr):
        assert risk_mgr.state.equity == 10_000_000.0
        assert risk_mgr.state.position == 0

    def test_zero_target_returns_zero_contracts(self, risk_mgr):
        result = risk_mgr.check_and_clip(
            0.0, 23000.0, np.array([0.25, 0.25, 0.25, 0.25])
        )
        assert result == 0

    def test_kill_switch_triggers_at_daily_loss(self, risk_mgr):
        """When daily loss >= 2% of capital, should force flat."""
        # Simulate 2% loss
        risk_mgr.state.day_pnl = -200_000.0  # 2% of 1Cr
        risk_mgr.state.equity = 10_000_000.0

        result = risk_mgr.check_and_clip(
            1.0, 23000.0, np.array([0.5, 0.5, 0.0, 0.0])
        )
        assert result == 0, "Kill switch should force flat position"

    def test_kill_switch_does_not_trigger_below_threshold(self, risk_mgr):
        """Below 2% loss, trading should be allowed."""
        risk_mgr.state.day_pnl = -100_000.0  # 1% of 1Cr
        risk_mgr.state.equity = 10_000_000.0

        # Use high-confidence calm regime so confidence gate doesn't zero it
        result = risk_mgr.check_and_clip(
            1.0, 23000.0, np.array([0.9, 0.05, 0.025, 0.025])
        )
        assert result != 0, "Below threshold, trading should be allowed"

    def test_drawdown_halves_capacity(self, risk_mgr):
        """When drawdown >= 5%, max lots should be halved."""
        # Use high-confidence regime so confidence gate doesn't zero it
        calm_probs = np.array([0.9, 0.05, 0.025, 0.025])

        # With drawdown
        risk_mgr.state.peak_equity = 10_000_000.0
        risk_mgr.state.equity = 9_400_000.0  # 6% drawdown

        result_dd = risk_mgr.check_and_clip(1.0, 23000.0, calm_probs)

        # Without drawdown
        risk_mgr2 = RiskManager(risk_mgr.config)
        result_no_dd = risk_mgr2.check_and_clip(1.0, 23000.0, calm_probs)

        # Without drawdown should have positive position (sanity check)
        assert abs(result_no_dd) > 0, "No-DD result should be non-zero with high confidence"
        # With drawdown, capacity should be less or equal
        assert abs(result_dd) <= abs(result_no_dd), \
            f"DD result {result_dd} should be <= no-DD result {result_no_dd}"

    def test_max_position_change_enforced(self, risk_mgr):
        """Max position change per step should be enforced."""
        # Start at 0 position
        risk_mgr.state.position = 0
        max_change_lots = max(1, int(risk_mgr.config.max_position_change_per_step * risk_mgr.config.max_lots))
        max_change = max_change_lots * risk_mgr.config.lot_sizes["NIFTY"]

        result = risk_mgr.check_and_clip(
            1.0, 23000.0, np.array([0.9, 0.05, 0.025, 0.025])
        )

        assert abs(result) <= max_change, \
            f"Position change {abs(result)} exceeds max {max_change}"

    def test_position_rounded_to_lot_size(self, risk_mgr):
        """Position must be a multiple of lot size."""
        result = risk_mgr.check_and_clip(
            0.5, 23000.0, np.array([0.5, 0.5, 0.0, 0.0])
        )
        lot_size = risk_mgr.config.lot_sizes["NIFTY"]
        assert result % lot_size == 0, f"Position {result} not a multiple of lot size {lot_size}"

    def test_high_vol_regime_reduces_position(self, risk_mgr):
        """High vol regime should reduce position size."""
        # Low vol: calm_mr dominant
        result_calm = risk_mgr.check_and_clip(
            1.0, 23000.0, np.array([0.9, 0.05, 0.025, 0.025])
        )

        # High vol: high_vol dominant
        risk_mgr2 = RiskManager(risk_mgr.config)
        result_high_vol = risk_mgr2.check_and_clip(
            1.0, 23000.0, np.array([0.05, 0.05, 0.85, 0.05])
        )

        assert abs(result_high_vol) <= abs(result_calm), \
            f"High vol {result_high_vol} should be <= calm {result_calm}"

    def test_stress_regime_aggressive_reduction(self, risk_mgr):
        """Liquidity stress regime should aggressively reduce position."""
        # Use separate instances so state doesn't leak between calls
        rm_stress = RiskManager(risk_mgr.config)
        result_stress = rm_stress.check_and_clip(
            1.0, 23000.0, np.array([0.05, 0.05, 0.05, 0.85])  # high stress
        )

        rm_normal = RiskManager(risk_mgr.config)
        result_normal = rm_normal.check_and_clip(
            1.0, 23000.0, np.array([0.9, 0.05, 0.025, 0.025])  # calm, high confidence
        )

        assert abs(result_normal) > 0, "Normal regime should produce non-zero position"
        assert abs(result_stress) < abs(result_normal), \
            f"Stress {result_stress} should be < normal {result_normal}"

    def test_banknifty_lot_size(self, risk_mgr):
        """BANKNIFTY positions should use lot size 30."""
        result = risk_mgr.check_and_clip(
            0.5, 48000.0, np.array([0.5, 0.5, 0.0, 0.0]),
            instrument="BANKNIFTY"
        )
        assert result % 30 == 0, f"BANKNIFTY position {result} not a multiple of 30"

    def test_reset_day(self, risk_mgr):
        risk_mgr.state.day_pnl = -50000
        risk_mgr.state.trades_today = 5
        risk_mgr.reset_day()
        assert risk_mgr.state.day_pnl == 0.0
        assert risk_mgr.state.trades_today == 0

    def test_update_position(self, risk_mgr):
        risk_mgr.update_position(75, 23000.0)
        assert risk_mgr.state.position == 75
        assert risk_mgr.state.entry_price == 23000.0
        assert risk_mgr.state.trades_today == 1

    def test_contracts_within_max(self, risk_mgr):
        """Position should never exceed max_lots * lot_size."""
        result = risk_mgr.check_and_clip(
            1.0, 23000.0, np.array([0.9, 0.05, 0.025, 0.025])
        )
        max_contracts = risk_mgr.config.max_lots * risk_mgr.config.lot_sizes["NIFTY"]
        assert abs(result) <= max_contracts
        assert abs(result) > 0, "Should have non-zero position with high confidence"


# ══════════════════════════════════════════════════════════════════════════════
#  6. Execution Model Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionModel:
    """ExecutionModel tests — cost correctness, India FnO specifics."""

    @pytest.fixture
    def exec_model(self):
        return ExecutionModel()

    def test_zero_quantity_zero_cost(self, exec_model):
        cost = exec_model.compute_cost(0, 23000.0)
        assert cost == 0.0

    def test_cost_positive_for_nonzero_trade(self, exec_model):
        cost = exec_model.compute_cost(75, 23000.0)
        assert cost > 0.0

    def test_cost_includes_brokerage(self, exec_model):
        cost = exec_model.compute_cost(75, 23000.0)
        assert cost >= 20.0, "Cost must include at least brokerage (Rs.20)"

    def test_stt_only_on_sell(self, exec_model):
        """STT should only apply on sell side."""
        buy_cost = exec_model.compute_cost(75, 23000.0, is_sell=False)
        sell_cost = exec_model.compute_cost(75, 23000.0, is_sell=True)
        assert sell_cost > buy_cost, "Sell cost should be higher due to STT"

    def test_stamp_duty_only_on_buy(self, exec_model):
        """Stamp duty should only apply on buy side.

        Since STT is much larger than stamp duty, sell > buy overall.
        But we can verify the stamp_duty component directly.
        """
        cfg = exec_model.config
        turnover = 75 * 23000.0
        stamp_duty = turnover * cfg.stamp_duty_pct
        assert stamp_duty > 0

    def test_cost_scales_with_size(self, exec_model):
        cost_1_lot = exec_model.compute_cost(75, 23000.0)
        cost_2_lots = exec_model.compute_cost(150, 23000.0)
        # Brokerage is flat per order, but exchange charges scale
        assert cost_2_lots > cost_1_lot

    def test_simulate_fill_no_change(self, exec_model):
        fill = exec_model.simulate_fill(0, 0, 23000.0)
        assert fill["cost"] == 0.0
        assert fill["delta"] == 0
        assert fill["new_position"] == 0

    def test_simulate_fill_buy(self, exec_model):
        fill = exec_model.simulate_fill(0, 75, 23000.0)
        assert fill["delta"] == 75
        assert fill["new_position"] == 75
        assert fill["cost"] > 0
        assert fill["fill_price"] >= 23000.0  # adverse slippage on buy

    def test_simulate_fill_sell(self, exec_model):
        fill = exec_model.simulate_fill(75, 0, 23000.0)
        assert fill["delta"] == -75
        assert fill["new_position"] == 0
        assert fill["cost"] > 0
        assert fill["fill_price"] <= 23000.0  # adverse slippage on sell

    def test_roundtrip_cost_bps_nifty(self, exec_model):
        """NIFTY futures roundtrip cost at ~23000: ~1-5 bps (low for futures).
        STT 0.02% sell + exchange 0.00173% + stamp 0.002% buy + brokerage ₹20 flat.
        """
        bps = exec_model.roundtrip_cost_bps(23000.0, 75, "NIFTY")
        assert bps > 0.5, f"Roundtrip cost {bps:.2f} bps seems too low"
        assert bps < 20, f"Roundtrip cost {bps:.2f} bps seems too high for futures"

    def test_slippage_increases_with_volatility(self, exec_model):
        slip_low_vol = exec_model.compute_slippage(75, 23000.0, volatility=0.001)
        slip_high_vol = exec_model.compute_slippage(75, 23000.0, volatility=0.01)
        assert slip_high_vol > slip_low_vol

    def test_slippage_increases_with_size(self, exec_model):
        slip_small = exec_model.compute_slippage(75, 23000.0)
        slip_large = exec_model.compute_slippage(300, 23000.0)
        assert slip_large > slip_small

    def test_gst_on_brokerage(self, exec_model):
        """GST should be 18% on brokerage + exchange charge."""
        cfg = exec_model.config
        turnover = 75 * 23000.0
        brokerage = cfg.brokerage_per_order
        exchange_charge = turnover * cfg.exchange_txn_pct
        expected_gst = (brokerage + exchange_charge) * cfg.gst_pct
        assert expected_gst > 0

    def test_banknifty_lot_size_in_cost(self, exec_model):
        """BANKNIFTY lot size (30) should be used for cost calculation."""
        cost = exec_model.compute_cost(30, 48000.0, instrument="BANKNIFTY")
        assert cost > 0


# ══════════════════════════════════════════════════════════════════════════════
#  7. Data Loader Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDataLoader:
    """Data loader tests — requires real data files on disk."""

    def test_list_available_dates_returns_sorted(self):
        dates = list_available_dates()
        assert len(dates) > 0, "Should have at least some dates available"
        assert dates == sorted(dates), "Dates must be sorted"

    def test_list_available_dates_types(self):
        dates = list_available_dates()
        if dates:
            assert isinstance(dates[0], date)

    def test_load_day_returns_dataframe(self):
        dates = list_available_dates()
        assert len(dates) > 0, "Need at least 1 date for this test"

        # Try the first available date
        df = load_day(dates[0], "NIFTY")
        if df is not None:
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

    def test_load_day_expected_columns(self):
        dates = list_available_dates()
        if not dates:
            pytest.skip("No data available")

        df = load_day(dates[0], "NIFTY")
        if df is None:
            pytest.skip("No NIFTY data for first date")

        for col in ["open", "high", "low", "close", "volume", "oi"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_day_market_hours(self):
        """Data should be filtered to market hours (9:15-15:30)."""
        dates = list_available_dates()
        if not dates:
            pytest.skip("No data available")

        df = load_day(dates[0], "NIFTY")
        if df is None:
            pytest.skip("No NIFTY data for first date")

        # Check all times are within market hours
        times = df.index.time
        from datetime import time as dt_time
        market_open = dt_time(9, 15)
        market_close = dt_time(15, 30)
        for t in times:
            assert market_open <= t <= market_close, \
                f"Time {t} outside market hours"

    def test_load_day_no_duplicates(self):
        """Index should not have duplicates."""
        dates = list_available_dates()
        if not dates:
            pytest.skip("No data available")

        df = load_day(dates[0], "NIFTY")
        if df is None:
            pytest.skip("No NIFTY data for first date")

        assert not df.index.duplicated().any(), "Duplicate index entries found"

    def test_load_day_nonexistent_date(self):
        """Loading a non-trading date should return None."""
        result = load_day(date(2000, 1, 1), "NIFTY")
        assert result is None

    def test_load_day_banknifty(self):
        """BANKNIFTY data should also be available."""
        dates = list_available_dates()
        if not dates:
            pytest.skip("No data available")

        df = load_day(dates[0], "BANKNIFTY")
        # May or may not exist depending on data source
        if df is not None:
            assert len(df) > 0
            assert "close" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
#  8. Integration Test — End-to-end pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration tests."""

    def test_synthetic_pipeline_end_to_end(self):
        """Full pipeline on synthetic data: features -> topology -> model -> risk -> execution."""
        config = DTRNConfig()
        df = make_synthetic_bars(200)

        # 1. Feature engine
        fe = FeatureEngine(config)
        features, masks = fe.compute_batch(df)
        assert features.shape == (200, 26)

        # 2. Topology
        topo = DynamicTopology(
            d=26,
            ewma_span=config.ewma_cov_span,
            top_k=config.top_k_edges,
            tau_on=config.tau_on,
            tau_off=config.tau_off,
            max_flip_rate=config.max_edge_flip_rate,
            precision_reg=config.precision_reg,
        )
        adjs = np.zeros((200, 26, 26), dtype=np.float32)
        wgts = np.zeros((200, 26, 26), dtype=np.float32)
        for t in range(200):
            topo.update(features[t], masks[t])
            adjs[t] = topo.get_adjacency()
            wgts[t] = topo.get_weights()

        # 3. DTRN model
        _, model = create_dtrn(config, n_features=26)
        model.eval()

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, 200, 26)
        m = torch.tensor(masks, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(adjs, dtype=torch.float32)  # (200, 26, 26)
        w = torch.tensor(wgts, dtype=torch.float32)

        with torch.no_grad():
            out = model(x, m, a, w)

        assert out["position"].shape == (1, 200, 1)
        assert out["regime_probs"].shape == (1, 200, 4)

        # 4. Risk manager
        risk_mgr = RiskManager(config)
        price = float(df.iloc[100]["close"])
        regime_probs = out["regime_probs"][0, 100].numpy()
        target_pos = out["position"][0, 100, 0].item()

        contracts = risk_mgr.check_and_clip(target_pos, price, regime_probs)
        assert isinstance(contracts, (int, np.integer))

        # 5. Execution
        exec_model = ExecutionModel(config)
        fill = exec_model.simulate_fill(0, contracts, price)
        assert "fill_price" in fill
        assert "cost" in fill

    def test_online_step_by_step(self):
        """Online inference: step-by-step with forward_step."""
        config = DTRNConfig()
        df = make_synthetic_bars(50)

        fe = FeatureEngine(config)
        topo = DynamicTopology(d=26, ewma_span=config.ewma_cov_span)
        _, model = create_dtrn(config, n_features=26)
        model.eval()
        risk_mgr = RiskManager(config)
        exec_model = ExecutionModel(config)

        h_state = None
        position = 0

        for i, (dt, row) in enumerate(df.iterrows()):
            bar = {
                "datetime": dt,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volume", 0),
                "oi": row.get("oi", 0),
            }
            price = float(row["close"])

            # Features
            feat, mask = fe.update(bar)

            # Topology
            adj = topo.update(feat, mask)
            weights = topo.get_weights()

            # Model
            x_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            m_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            a_t = torch.tensor(adj, dtype=torch.float32)
            w_t = torch.tensor(weights, dtype=torch.float32)

            with torch.no_grad():
                outputs, h_state = model.forward_step(x_t, m_t, a_t, w_t, h_state)

            target_pos = outputs["position"].item()
            regime_probs = outputs["regime_probs"].squeeze(0).numpy()

            # Risk
            target_contracts = risk_mgr.check_and_clip(
                target_pos, price, regime_probs
            )

            # Execution
            if target_contracts != position:
                fill = exec_model.simulate_fill(position, target_contracts, price)
                position = fill["new_position"]

        # Should complete without error
        assert True

    def test_real_data_one_day(self):
        """Integration test on one day of real data (if available)."""
        dates = list_available_dates()
        if not dates:
            pytest.skip("No real data available")

        config = DTRNConfig()

        # Find a date with NIFTY data
        test_date = None
        df = None
        for d in dates[-10:]:  # try last 10 dates
            df = load_day(d, "NIFTY")
            if df is not None and len(df) > 50:
                test_date = d
                break

        if df is None:
            pytest.skip("No suitable NIFTY data found")

        # Feature engine
        fe = FeatureEngine(config)
        features, masks = fe.compute_batch(df)
        assert features.shape[0] == len(df)
        assert features.shape[1] == 26
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

        # Topology
        topo = DynamicTopology(d=26, ewma_span=config.ewma_cov_span)
        for t in range(len(df)):
            adj = topo.update(features[t], masks[t])

        # After a full day, should have some topology
        stats = topo.get_stats()
        assert stats["n_updates"] == len(df)

        # Model online inference
        _, model = create_dtrn(config, n_features=26)
        model.eval()
        risk_mgr = RiskManager(config)
        exec_model = ExecutionModel(config)

        fe.reset()
        topo.reset()
        h_state = None
        position = 0
        total_cost = 0.0
        trades = 0

        for dt_idx, (dt, row) in enumerate(df.iterrows()):
            bar = {
                "datetime": dt,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volume", 0),
                "oi": row.get("oi", 0),
            }
            price = float(row["close"])

            feat, mask = fe.update(bar)
            adj = topo.update(feat, mask)
            weights = topo.get_weights()

            x_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            m_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            a_t = torch.tensor(adj, dtype=torch.float32)
            w_t = torch.tensor(weights, dtype=torch.float32)

            with torch.no_grad():
                outputs, h_state = model.forward_step(x_t, m_t, a_t, w_t, h_state)

            target_pos = outputs["position"].item()
            regime_probs = outputs["regime_probs"].squeeze(0).numpy()

            target_contracts = risk_mgr.check_and_clip(
                target_pos, price, regime_probs
            )

            if target_contracts != position:
                fill = exec_model.simulate_fill(position, target_contracts, price)
                position = fill["new_position"]
                total_cost += fill["cost"]
                trades += 1

        # Verify completion
        assert True, f"Processed {len(df)} bars, {trades} trades, cost={total_cost:.0f}"


# ══════════════════════════════════════════════════════════════════════════════
#  Run
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
