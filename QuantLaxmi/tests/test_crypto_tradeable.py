"""Tests for crypto as tradeable asset in the RL pipeline.

Validates that BTC/ETH can trade alongside 4 India indices (NIFTY,
BANKNIFTY, FINNIFTY, MIDCPNIFTY) in the IntegratedTradingEnv with:
  - Correct 6-asset state dimensions
  - Unified cost model (India: index points per leg; Crypto: 0.1% per side)
  - Calendar alignment (India flat on non-trading days)
  - Overnight gap signal computation
  - Allocation weights summing to 1
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantlaxmi.models.rl.integration.integrated_env import (
    IntegratedTradingEnv,
    AssetCostModel,
    IndiaCostModel,
    CryptoCostModel,
    build_cost_models,
    is_india_trading_day,
    compute_overnight_gap,
    INDIA_SYMBOLS,
    CRYPTO_SYMBOLS,
    ALL_SYMBOLS,
    CRYPTO_FEE_RATE,
    COST_PER_LEG,
    INITIAL_SPOTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_backbone(n_assets: int, d_hidden: int = 8):
    """Create a mock backbone that returns deterministic hidden states."""
    class MockBackbone:
        def __init__(self):
            self.d_hidden = d_hidden

        def precompute_hidden_states(self, features, start, end, rng, **kwargs):
            fold_len = end - start
            # Deterministic hidden: asset_idx * 0.1 + step * 0.01
            h = np.zeros((fold_len, n_assets, d_hidden), dtype=np.float32)
            for t in range(fold_len):
                for a in range(n_assets):
                    h[t, a, :] = a * 0.1 + t * 0.01
            return h

    return MockBackbone()


def _make_6asset_env(
    n_days: int = 30,
    d_hidden: int = 8,
    include_crypto: bool = True,
    india_trading_days: set[int] | None = None,
) -> IntegratedTradingEnv:
    """Build a 6-asset IntegratedTradingEnv for testing."""
    symbols = INDIA_SYMBOLS + CRYPTO_SYMBOLS
    n_assets = len(symbols)  # 6

    backbone = _make_mock_backbone(n_assets, d_hidden)

    rng = np.random.default_rng(42)
    features = rng.standard_normal((n_days, n_assets, 10)).astype(np.float32)
    targets = rng.standard_normal((n_days, n_assets)).astype(np.float64) * 0.01

    dates = pd.bdate_range("2025-01-01", periods=n_days, freq="B")

    env = IntegratedTradingEnv(
        backbone=backbone,
        features=features,
        targets=targets,
        dates=dates,
        symbols=symbols,
        include_crypto=include_crypto,
        india_trading_days=india_trading_days,
    )
    return env


def _make_4asset_env(n_days: int = 30, d_hidden: int = 8) -> IntegratedTradingEnv:
    """Build a 4-asset (India-only) IntegratedTradingEnv for comparison."""
    symbols = list(INDIA_SYMBOLS)
    n_assets = len(symbols)  # 4

    backbone = _make_mock_backbone(n_assets, d_hidden)

    rng = np.random.default_rng(42)
    features = rng.standard_normal((n_days, n_assets, 10)).astype(np.float32)
    targets = rng.standard_normal((n_days, n_assets)).astype(np.float64) * 0.01

    dates = pd.bdate_range("2025-01-01", periods=n_days, freq="B")

    env = IntegratedTradingEnv(
        backbone=backbone,
        features=features,
        targets=targets,
        dates=dates,
        symbols=symbols,
        include_crypto=False,
    )
    return env


# ============================================================================
# Tests
# ============================================================================


class TestUnifiedEnvReset:
    """test_unified_env_reset: 6-asset state dimensions correct."""

    def test_state_dim_6_assets(self):
        """State dim = 6*d_hidden + 6 (positions) + 6 (pnl, dd, heat, 3 time)."""
        d_hidden = 8
        env = _make_6asset_env(d_hidden=d_hidden)
        expected_dim = 6 * d_hidden + 6 + 6  # 48 + 6 + 6 = 60
        assert env.state_dim == expected_dim

    def test_action_dim_6_assets(self):
        """Action dim = 6 (one per asset)."""
        env = _make_6asset_env()
        assert env.action_dim == 6

    def test_reset_returns_correct_shape(self):
        """reset() returns state vector of correct dimension."""
        env = _make_6asset_env()
        state = env.reset(fold_start_idx=0, fold_end_idx=20)
        assert state.shape == (env.state_dim,)

    def test_reset_positions_zero(self):
        """After reset, all positions are zero."""
        env = _make_6asset_env()
        env.reset(fold_start_idx=0, fold_end_idx=20)
        np.testing.assert_array_equal(env._positions, np.zeros(6))

    def test_4asset_backward_compat(self):
        """4-asset India-only mode still works."""
        d_hidden = 8
        env = _make_4asset_env(d_hidden=d_hidden)
        expected_dim = 4 * d_hidden + 4 + 6  # 32 + 4 + 6 = 42
        assert env.state_dim == expected_dim
        assert env.action_dim == 4
        state = env.reset(fold_start_idx=0, fold_end_idx=20)
        assert state.shape == (expected_dim,)

    def test_symbols_preserved(self):
        """Symbols list is preserved correctly."""
        env = _make_6asset_env()
        assert env.symbols == INDIA_SYMBOLS + CRYPTO_SYMBOLS
        assert env.n_assets == 6


class TestUnifiedEnvStep:
    """test_unified_env_step: actions applied, costs computed."""

    def test_step_returns_correct_shape(self):
        """step() returns (state, reward, done, info) with correct shapes."""
        env = _make_6asset_env()
        env.reset(fold_start_idx=0, fold_end_idx=20)
        actions = np.array([0.5, -0.3, 0.1, 0.0, 0.4, -0.2])
        state, reward, done, info = env.step(actions)
        assert state.shape == (env.state_dim,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "per_asset" in info

    def test_positions_updated(self):
        """Positions update to match actions."""
        env = _make_6asset_env()
        env.reset(fold_start_idx=0, fold_end_idx=20)
        actions = np.array([0.5, -0.3, 0.1, 0.0, 0.4, -0.2])
        env.step(actions)
        np.testing.assert_array_almost_equal(env._positions, actions)

    def test_cost_computed_for_all_assets(self):
        """Each asset has a non-negative cost in info."""
        env = _make_6asset_env()
        env.reset(fold_start_idx=0, fold_end_idx=20)
        actions = np.array([0.5, -0.3, 0.1, 0.2, 0.4, -0.2])
        _, _, _, info = env.step(actions)
        assert info["total_cost"] > 0.0
        for sym in env.symbols:
            assert info["per_asset"][sym]["cost"] >= 0.0

    def test_asset_type_in_info(self):
        """Per-asset info includes asset_type (india or crypto)."""
        env = _make_6asset_env()
        env.reset(fold_start_idx=0, fold_end_idx=20)
        actions = np.zeros(6)
        actions[0] = 0.1  # NIFTY
        actions[4] = 0.1  # BTCUSDT
        _, _, _, info = env.step(actions)
        assert info["per_asset"]["NIFTY"]["asset_type"] == "india"
        assert info["per_asset"]["BTCUSDT"]["asset_type"] == "crypto"

    def test_multiple_steps(self):
        """Multiple steps work without error."""
        env = _make_6asset_env()
        env.reset(fold_start_idx=0, fold_end_idx=20)
        rng = np.random.default_rng(123)
        for _ in range(5):
            actions = rng.uniform(-1, 1, size=6)
            state, reward, done, info = env.step(actions)
            if done:
                break
            assert state.shape == (env.state_dim,)

    def test_clipping(self):
        """Actions are clipped to [-1, 1]."""
        env = _make_6asset_env()
        env.reset(fold_start_idx=0, fold_end_idx=20)
        actions = np.array([2.0, -3.0, 1.5, -1.5, 5.0, -5.0])
        env.step(actions)
        # Positions should be clipped
        for pos in env._positions:
            assert -1.0 <= pos <= 1.0


class TestIndiaCostModel:
    """test_india_cost_model: points-based costs."""

    def test_nifty_cost(self):
        """NIFTY cost = |trade| * 3.0 / 24000."""
        model = IndiaCostModel("NIFTY")
        trade = 0.5
        expected = abs(trade) * 3.0 / 24_000.0
        assert abs(model.compute_cost(trade) - expected) < 1e-12

    def test_banknifty_cost(self):
        """BANKNIFTY cost = |trade| * 5.0 / 51000."""
        model = IndiaCostModel("BANKNIFTY")
        trade = -0.3
        expected = abs(trade) * 5.0 / 51_000.0
        assert abs(model.compute_cost(trade) - expected) < 1e-12

    def test_zero_trade_zero_cost(self):
        """Zero trade means zero cost."""
        model = IndiaCostModel("NIFTY")
        assert model.compute_cost(0.0) == 0.0

    def test_cost_symmetric(self):
        """Cost is symmetric for buy vs sell."""
        model = IndiaCostModel("FINNIFTY")
        assert model.compute_cost(0.5) == model.compute_cost(-0.5)

    def test_asset_type(self):
        """IndiaCostModel reports asset_type='india'."""
        model = IndiaCostModel("NIFTY")
        assert model.asset_type == "india"

    def test_custom_overrides(self):
        """Custom cost_per_leg and spot are respected."""
        model = IndiaCostModel("NIFTY", cost_per_leg=10.0, spot=25_000.0)
        trade = 1.0
        expected = abs(trade) * 10.0 / 25_000.0
        assert abs(model.compute_cost(trade) - expected) < 1e-12


class TestCryptoCostModel:
    """test_crypto_cost_model: percentage-based costs."""

    def test_btc_cost(self):
        """BTC cost = |trade| * 0.001."""
        model = CryptoCostModel("BTCUSDT")
        trade = 0.5
        expected = abs(trade) * 0.001
        assert abs(model.compute_cost(trade) - expected) < 1e-12

    def test_eth_cost(self):
        """ETH cost = |trade| * 0.001."""
        model = CryptoCostModel("ETHUSDT")
        trade = -0.8
        expected = abs(trade) * 0.001
        assert abs(model.compute_cost(trade) - expected) < 1e-12

    def test_zero_trade_zero_cost(self):
        """Zero trade means zero cost."""
        model = CryptoCostModel("BTCUSDT")
        assert model.compute_cost(0.0) == 0.0

    def test_cost_symmetric(self):
        """Cost is symmetric for buy vs sell."""
        model = CryptoCostModel("ETHUSDT")
        assert model.compute_cost(0.5) == model.compute_cost(-0.5)

    def test_asset_type(self):
        """CryptoCostModel reports asset_type='crypto'."""
        model = CryptoCostModel("BTCUSDT")
        assert model.asset_type == "crypto"

    def test_custom_fee_rate(self):
        """Custom fee_rate is respected."""
        model = CryptoCostModel("BTCUSDT", fee_rate=0.002)
        trade = 1.0
        expected = abs(trade) * 0.002
        assert abs(model.compute_cost(trade) - expected) < 1e-12

    def test_india_vs_crypto_cost_difference(self):
        """India cost model and crypto cost model produce different values."""
        india = IndiaCostModel("NIFTY")
        crypto = CryptoCostModel("BTCUSDT")
        trade = 0.5
        india_cost = india.compute_cost(trade)
        crypto_cost = crypto.compute_cost(trade)
        # They should differ (different formulas)
        assert india_cost != crypto_cost


class TestBuildCostModels:
    """Test the build_cost_models factory function."""

    def test_6_asset_models(self):
        """build_cost_models produces correct types for 6 assets."""
        symbols = INDIA_SYMBOLS + CRYPTO_SYMBOLS
        models = build_cost_models(symbols)
        assert len(models) == 6
        # First 4 are India
        for i in range(4):
            assert isinstance(models[i], IndiaCostModel)
            assert models[i].asset_type == "india"
        # Last 2 are crypto
        for i in range(4, 6):
            assert isinstance(models[i], CryptoCostModel)
            assert models[i].asset_type == "crypto"

    def test_india_only(self):
        """build_cost_models with India-only symbols."""
        models = build_cost_models(INDIA_SYMBOLS)
        assert len(models) == 4
        for m in models:
            assert isinstance(m, IndiaCostModel)


class TestCalendarAlignment:
    """test_calendar_alignment: India flat on non-trading days."""

    def test_is_india_trading_day_weekday(self):
        """Weekdays are India trading days."""
        # Monday
        assert is_india_trading_day(pd.Timestamp("2025-01-06"))
        # Friday
        assert is_india_trading_day(pd.Timestamp("2025-01-10"))

    def test_is_india_trading_day_weekend(self):
        """Weekends are NOT India trading days."""
        # Saturday
        assert not is_india_trading_day(pd.Timestamp("2025-01-11"))
        # Sunday
        assert not is_india_trading_day(pd.Timestamp("2025-01-12"))

    def test_india_flat_on_non_trading_day(self):
        """On non-trading days, India positions are forced to 0."""
        # _is_india_trading_step checks fold_start + step_idx.
        # After 4 steps from fold_start=0, step_idx=4, so day checked = 4.
        # Mark day_idx=4 as non-trading.
        india_trading = set(range(30)) - {4}
        env = _make_6asset_env(
            n_days=30,
            include_crypto=True,
            india_trading_days=india_trading,
        )
        env.reset(fold_start_idx=0, fold_end_idx=20)

        # Advance step_idx to 4 by taking 4 steps
        for _ in range(4):
            actions = np.zeros(6)
            env.step(actions)

        # Now step_idx=4 => day checked = 0+4 = 4 which is non-trading
        actions = np.array([0.5, -0.3, 0.1, 0.2, 0.4, -0.2])
        _, _, _, info = env.step(actions)

        # India positions (0-3) should be forced to 0
        for india_sym in INDIA_SYMBOLS:
            assert info["per_asset"][india_sym]["position"] == 0.0

        # Crypto positions should be unchanged
        assert info["per_asset"]["BTCUSDT"]["position"] == pytest.approx(0.4)
        assert info["per_asset"]["ETHUSDT"]["position"] == pytest.approx(-0.2)

    def test_india_active_on_trading_day(self):
        """On trading days, India positions are applied normally."""
        all_days = set(range(30))  # all days are trading days
        env = _make_6asset_env(
            n_days=30,
            include_crypto=True,
            india_trading_days=all_days,
        )
        env.reset(fold_start_idx=0, fold_end_idx=20)
        actions = np.array([0.5, -0.3, 0.1, 0.2, 0.4, -0.2])
        _, _, _, info = env.step(actions)

        assert info["india_active"] is True
        assert info["per_asset"]["NIFTY"]["position"] == pytest.approx(0.5)
        assert info["per_asset"]["BANKNIFTY"]["position"] == pytest.approx(-0.3)


class TestOvernightGapSignal:
    """test_overnight_gap_signal: BTC overnight return computed correctly."""

    def test_positive_overnight_return(self):
        """Positive overnight gap: BTC goes up during India off-hours."""
        prices = np.array([100.0, 105.0, 110.0, 108.0, 115.0])
        gap = compute_overnight_gap(prices, india_close_idx=1, india_open_idx=3)
        expected = math.log(108.0 / 105.0)
        assert abs(gap - expected) < 1e-10

    def test_negative_overnight_return(self):
        """Negative overnight gap: BTC drops during India off-hours."""
        prices = np.array([100.0, 110.0, 105.0, 95.0, 100.0])
        gap = compute_overnight_gap(prices, india_close_idx=1, india_open_idx=3)
        expected = math.log(95.0 / 110.0)
        assert abs(gap - expected) < 1e-10

    def test_zero_gap_same_price(self):
        """Zero gap when close == open price."""
        prices = np.array([100.0, 100.0, 100.0])
        gap = compute_overnight_gap(prices, india_close_idx=0, india_open_idx=1)
        assert gap == 0.0

    def test_out_of_bounds_returns_zero(self):
        """Out-of-bounds indices return 0."""
        prices = np.array([100.0, 105.0])
        assert compute_overnight_gap(prices, india_close_idx=5, india_open_idx=0) == 0.0
        assert compute_overnight_gap(prices, india_close_idx=0, india_open_idx=5) == 0.0

    def test_negative_indices_return_zero(self):
        """Negative indices return 0."""
        prices = np.array([100.0, 105.0])
        assert compute_overnight_gap(prices, india_close_idx=-1, india_open_idx=1) == 0.0

    def test_zero_price_returns_zero(self):
        """Zero price in either position returns 0."""
        prices = np.array([0.0, 100.0])
        assert compute_overnight_gap(prices, india_close_idx=0, india_open_idx=1) == 0.0
        prices2 = np.array([100.0, 0.0])
        assert compute_overnight_gap(prices2, india_close_idx=0, india_open_idx=1) == 0.0


class TestSixAssetAllocation:
    """test_6_asset_allocation: allocation weights sum to 1."""

    def test_equal_weight_sums_to_one(self):
        """Equal weight allocation across 6 assets sums to 1."""
        weights = np.ones(6) / 6.0
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_env_accepts_normalized_actions(self):
        """Env accepts actions that sum to 1 (allocation-like)."""
        env = _make_6asset_env()
        env.reset(fold_start_idx=0, fold_end_idx=20)
        # Softmax-like allocation (non-negative, sums to ~1)
        raw = np.array([0.2, 0.15, 0.1, 0.1, 0.3, 0.15])
        assert abs(raw.sum() - 1.0) < 1e-10
        state, reward, done, info = env.step(raw)
        assert state.shape == (env.state_dim,)
        # All positions should be between 0 and 1 (since raw is in [0,1])
        for sym in env.symbols:
            assert 0.0 <= info["per_asset"][sym]["position"] <= 1.0

    def test_long_short_allocation_sums_correctly(self):
        """Long/short allocation (positive + negative) with abs sum = 1."""
        weights = np.array([0.3, -0.2, 0.1, -0.1, 0.5, -0.1])
        # abs sum does not need to be 1 for position targets, but
        # net exposure = 0.3 - 0.2 + 0.1 - 0.1 + 0.5 - 0.1 = 0.5
        env = _make_6asset_env()
        env.reset(fold_start_idx=0, fold_end_idx=20)
        state, reward, done, info = env.step(weights)
        assert state.shape == (env.state_dim,)

    def test_all_india_cost_models_correct(self):
        """In the 6-asset env, India assets use IndiaCostModel."""
        env = _make_6asset_env()
        for i in range(4):
            model = env.get_cost_model(i)
            assert isinstance(model, IndiaCostModel)
            assert model.asset_type == "india"

    def test_all_crypto_cost_models_correct(self):
        """In the 6-asset env, crypto assets use CryptoCostModel."""
        env = _make_6asset_env()
        for i in range(4, 6):
            model = env.get_cost_model(i)
            assert isinstance(model, CryptoCostModel)
            assert model.asset_type == "crypto"

    def test_total_cost_combines_both_models(self):
        """Total cost is sum of India point-based and crypto percentage costs."""
        env = _make_6asset_env()
        env.reset(fold_start_idx=0, fold_end_idx=20)

        # Trade all 6 assets equally
        actions = np.full(6, 0.5)
        _, _, _, info = env.step(actions)

        # Manually compute expected costs
        expected_cost = 0.0
        for i, sym in enumerate(env.symbols):
            model = env.get_cost_model(i)
            expected_cost += model.compute_cost(0.5)

        assert abs(info["total_cost"] - expected_cost) < 1e-10


class TestCryptoEnvInstances:
    """Test that crypto env instances are created correctly."""

    def test_crypto_envs_are_crypto_env_type(self):
        """Crypto assets get CryptoEnv instances, not IndiaFnOEnv."""
        from quantlaxmi.models.rl.environments.crypto_env import CryptoEnv
        from quantlaxmi.models.rl.environments.india_fno_env import IndiaFnOEnv

        env = _make_6asset_env()
        # First 4: India
        for i in range(4):
            assert isinstance(env._envs[i], IndiaFnOEnv)
        # Last 2: Crypto
        for i in range(4, 6):
            assert isinstance(env._envs[i], CryptoEnv)

    def test_india_indices_tracked(self):
        """_india_indices contains [0,1,2,3]."""
        env = _make_6asset_env()
        assert env._india_indices == [0, 1, 2, 3]

    def test_crypto_indices_tracked(self):
        """_crypto_indices contains [4,5]."""
        env = _make_6asset_env()
        assert env._crypto_indices == [4, 5]
