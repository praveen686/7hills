"""Tests for CLRS — Crypto Liquidity Regime Strategy.

Tests cover:
  - VPIN computation (incremental + batch)
  - OFI computation
  - Hawkes process (event, intensity, calibration)
  - Kyle's Lambda
  - Amihud illiquidity
  - Funding PCA
  - Signal generation (all 4 channels)
  - State persistence (save/load round-trip)
  - Trade execution
  - Regime classification
"""

from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apps.crypto_flow.features import (
    HawkesState,
    OFIState,
    RegimeSnapshot,
    VPINState,
    amihud_illiquidity,
    calibrate_hawkes,
    compute_hawkes_intensity,
    compute_ofi_series,
    compute_vpin_series,
    funding_pca,
    kyles_lambda,
    rolling_funding_residual,
)
from apps.crypto_flow.scanner import (
    FundingMatrixBuilder,
    SymbolSnapshot,
    annualize_funding,
)
from apps.crypto_flow.signals import (
    SignalConfig,
    TradeSignal,
    generate_carry_signals,
    generate_cascade_signals,
    generate_reversion_signals,
)
from apps.crypto_flow.state import (
    PortfolioState,
    Position,
    compute_performance,
)
from apps.crypto_flow.strategy import (
    check_ttl_exits,
    execute_signals,
)


# ===========================================================================
# VPIN Tests
# ===========================================================================

class TestVPIN:
    """Tests for VPIN (Volume-synchronized Probability of Informed Trading)."""

    def test_vpin_state_initialization(self):
        state = VPINState(bucket_size=1000, n_buckets=10)
        assert state.bucket_size == 1000
        assert state.n_buckets == 10
        assert state._current_vol == 0.0

    def test_vpin_single_update_no_complete(self):
        """Small volume should not complete a bucket."""
        state = VPINState(bucket_size=10000, n_buckets=5)
        result = state.update(100.0, 500.0, 99.9, sigma=0.01)
        assert result is None  # bucket not complete

    def test_vpin_bucket_completion(self):
        """Feed enough volume to complete buckets and get a VPIN value."""
        state = VPINState(bucket_size=1000, n_buckets=5)

        # Feed 10 buckets worth of volume to get VPIN
        for i in range(100):
            price = 100 + np.sin(i * 0.1) * 2  # oscillating price
            prev_price = 100 + np.sin((i - 1) * 0.1) * 2
            result = state.update(price, 200.0, prev_price, sigma=0.02)

        # After enough volume, should have VPIN
        assert result is not None
        assert 0 <= result <= 1.0

    def test_vpin_informed_flow(self):
        """Strong directional moves should increase VPIN."""
        # Scenario 1: trending (high VPIN expected)
        state_trend = VPINState(bucket_size=500, n_buckets=10)
        prices = np.linspace(100, 120, 200)  # strong uptrend

        vpin_trend = None
        for i in range(1, len(prices)):
            v = state_trend.update(prices[i], 100.0, prices[i - 1], sigma=0.02)
            if v is not None:
                vpin_trend = v

        # Scenario 2: mean-reverting (lower VPIN expected)
        state_mr = VPINState(bucket_size=500, n_buckets=10)
        np.random.seed(42)
        prices_mr = 100 + np.random.randn(200) * 0.5

        vpin_mr = None
        for i in range(1, len(prices_mr)):
            v = state_mr.update(prices_mr[i], 100.0, prices_mr[i - 1], sigma=0.02)
            if v is not None:
                vpin_mr = v

        if vpin_trend is not None and vpin_mr is not None:
            assert vpin_trend > vpin_mr

    def test_compute_vpin_series(self):
        """Batch VPIN computation."""
        np.random.seed(42)
        n = 500
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.001))
        volumes = np.random.uniform(10000, 50000, n)

        vpin = compute_vpin_series(prices, volumes, bucket_size=20000, n_buckets=10)
        assert len(vpin) == n
        # Should have valid values after enough data
        valid = vpin[~np.isnan(vpin)]
        assert len(valid) > 0
        assert all(0 <= v <= 1.5 for v in valid)  # VPIN is bounded

    def test_vpin_zero_sigma(self):
        """Zero sigma should return None gracefully."""
        state = VPINState()
        result = state.update(100.0, 1000.0, 100.0, sigma=0.0)
        assert result is None


# ===========================================================================
# OFI Tests
# ===========================================================================

class TestOFI:
    """Tests for Order Flow Imbalance."""

    def test_ofi_first_tick(self):
        """First tick should return 0 (no previous state)."""
        state = OFIState()
        result = state.update(100.0, 100.1, 10.0, 10.0)
        assert result == 0.0

    def test_ofi_bid_increase(self):
        """Bid moving up should produce positive OFI."""
        state = OFIState(ema_alpha=1.0)  # no smoothing
        state.update(100.0, 100.1, 10.0, 10.0)  # initialize

        # Bid increases with size
        result = state.update(100.05, 100.1, 15.0, 10.0)
        assert result > 0  # buying pressure

    def test_ofi_ask_decrease(self):
        """Ask moving down should produce negative OFI (selling pressure)."""
        state = OFIState(ema_alpha=1.0)
        state.update(100.0, 100.1, 10.0, 10.0)

        # Ask drops (seller aggressive), bid unchanged
        result = state.update(100.0, 100.05, 10.0, 15.0)
        assert result < 0  # selling pressure

    def test_ofi_ema_smoothing(self):
        """EMA should smooth the raw OFI."""
        state = OFIState(ema_alpha=0.1)  # heavy smoothing
        state.update(100.0, 100.1, 10.0, 10.0)

        results = []
        for i in range(20):
            r = state.update(100.0 + i * 0.01, 100.1, 10.0 + i, 10.0)
            results.append(r)

        # Should be smoothly increasing
        diffs = np.diff(results)
        assert all(d >= 0 for d in diffs[1:])  # monotonic after first

    def test_compute_ofi_series(self):
        """Batch OFI computation."""
        n = 100
        bids = np.linspace(100, 101, n)  # trending up
        asks = bids + 0.1
        bid_qtys = np.random.uniform(5, 15, n)
        ask_qtys = np.random.uniform(5, 15, n)

        ofi = compute_ofi_series(bids, asks, bid_qtys, ask_qtys)
        assert len(ofi) == n
        # With trending bids, average OFI should be positive
        assert np.mean(ofi[10:]) > 0


# ===========================================================================
# Hawkes Process Tests
# ===========================================================================

class TestHawkes:
    """Tests for Hawkes self-exciting process."""

    def test_hawkes_baseline(self):
        """No events = baseline intensity."""
        state = HawkesState(mu=1.0, alpha=0.8, beta=1.5)
        assert state.intensity(0.0) == 1.0  # mu
        assert state.intensity_ratio(0.0) == 1.0

    def test_hawkes_excitation(self):
        """Events should increase intensity above baseline."""
        state = HawkesState(mu=1.0, alpha=0.8, beta=1.5)
        lam = state.event(1.0)
        assert lam > 1.0  # excited above baseline

    def test_hawkes_decay(self):
        """Intensity should decay back to baseline over time."""
        state = HawkesState(mu=1.0, alpha=0.8, beta=1.5)
        state.event(0.0)
        lam_immediately = state.intensity(0.01)
        lam_later = state.intensity(10.0)

        assert lam_immediately > lam_later
        assert abs(lam_later - 1.0) < 0.01  # nearly back to baseline

    def test_hawkes_cascade(self):
        """Rapid events should create cascade (high intensity ratio)."""
        state = HawkesState(mu=0.5, alpha=0.8, beta=1.0)

        # Rapid burst of events
        for t in np.arange(0, 5, 0.1):
            state.event(t)

        ratio = state.intensity_ratio(5.0)
        assert ratio > 3.0  # cascade territory

    def test_hawkes_branching_ratio(self):
        state = HawkesState(mu=1.0, alpha=0.8, beta=1.5)
        assert abs(state.branching_ratio - 0.8 / 1.5) < 1e-10

    def test_calibrate_hawkes(self):
        """Calibration should produce reasonable parameters."""
        # Generate synthetic Hawkes timestamps
        np.random.seed(42)
        timestamps = np.sort(np.random.uniform(0, 100, 200))

        mu, alpha, beta = calibrate_hawkes(timestamps)
        assert mu > 0
        assert alpha > 0
        assert beta > 0
        assert alpha / beta < 1.0  # subcritical

    def test_calibrate_hawkes_too_few(self):
        """Too few events should return defaults."""
        mu, alpha, beta = calibrate_hawkes(np.array([1.0, 2.0]))
        assert mu == 1.0
        assert alpha == 0.5
        assert beta == 1.5

    def test_compute_hawkes_intensity(self):
        """Batch intensity computation."""
        timestamps = np.array([1.0, 1.1, 1.2, 5.0, 10.0])
        eval_times = np.linspace(0, 15, 50)

        intensity = compute_hawkes_intensity(timestamps, eval_times)
        assert len(intensity) == 50
        # Intensity should spike around the cluster at t=1
        idx_at_1 = np.argmin(np.abs(eval_times - 1.3))
        idx_at_8 = np.argmin(np.abs(eval_times - 8.0))
        assert intensity[idx_at_1] > intensity[idx_at_8]


# ===========================================================================
# Kyle's Lambda Tests
# ===========================================================================

class TestKylesLambda:
    """Tests for price impact regression."""

    def test_kyles_lambda_basic(self):
        """Basic computation should produce values."""
        np.random.seed(42)
        n = 200
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.001))
        volumes = np.random.uniform(100, 1000, n)

        lam = kyles_lambda(prices, volumes, window=50)
        assert len(lam) == n

        # Should have NaN for warmup period, then values
        valid = lam[~np.isnan(lam)]
        assert len(valid) > 0

    def test_kyles_lambda_high_impact(self):
        """Correlated price-volume should have higher lambda."""
        np.random.seed(42)
        n = 300
        # Create price that moves proportionally to volume
        volumes = np.random.uniform(100, 1000, n)
        returns = volumes * 0.001 + np.random.randn(n) * 0.001
        prices = 100 * np.exp(np.cumsum(returns))

        lam = kyles_lambda(prices, volumes, window=100)
        valid = lam[~np.isnan(lam)]
        assert len(valid) > 0
        assert np.mean(valid) > 0  # positive lambda (price follows volume)


# ===========================================================================
# Amihud Illiquidity Tests
# ===========================================================================

class TestAmihud:
    def test_amihud_basic(self):
        np.random.seed(42)
        n = 100
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        volumes = np.random.uniform(1e6, 1e7, n)

        illiq = amihud_illiquidity(prices, volumes, window=20)
        assert len(illiq) == n
        valid = illiq[~np.isnan(illiq)]
        assert len(valid) > 0
        assert all(v >= 0 for v in valid)

    def test_amihud_low_volume_more_illiquid(self):
        """Lower volume should produce higher Amihud measure."""
        np.random.seed(42)
        n = 150
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))

        illiq_high_vol = amihud_illiquidity(prices, np.full(n, 1e8), window=20)
        illiq_low_vol = amihud_illiquidity(prices, np.full(n, 1e5), window=20)

        h = illiq_high_vol[~np.isnan(illiq_high_vol)]
        l = illiq_low_vol[~np.isnan(illiq_low_vol)]
        if len(h) > 0 and len(l) > 0:
            assert np.mean(l) > np.mean(h)


# ===========================================================================
# Funding PCA Tests
# ===========================================================================

class TestFundingPCA:
    """Tests for cross-sectional funding rate PCA."""

    def test_pca_basic(self):
        """PCA should decompose funding matrix."""
        np.random.seed(42)
        n_times, n_syms = 30, 10
        # Common factor + noise
        common = np.random.randn(n_times) * 0.001
        mat = np.outer(common, np.ones(n_syms)) + np.random.randn(n_times, n_syms) * 0.0001

        df = pd.DataFrame(mat, columns=[f"SYM{i}USDT" for i in range(n_syms)])
        loadings, scores, residuals = funding_pca(df, n_components=3)

        assert loadings.shape == (3, n_syms)
        assert scores.shape == (n_times, 3)
        assert residuals.shape == (n_times, n_syms)

    def test_pca_residuals_smaller_than_raw(self):
        """Residuals should be smaller than raw values (PCA captures common variance)."""
        np.random.seed(42)
        n_times, n_syms = 50, 15
        common = np.random.randn(n_times) * 0.01
        noise = np.random.randn(n_times, n_syms) * 0.001
        mat = np.outer(common, np.ones(n_syms)) + noise

        df = pd.DataFrame(mat)
        _, _, residuals = funding_pca(df, n_components=1)

        assert np.std(residuals) < np.std(mat)

    def test_pca_too_few_observations(self):
        """Should handle small matrices gracefully."""
        df = pd.DataFrame(np.random.randn(3, 5))
        loadings, scores, residuals = funding_pca(df, n_components=2)
        assert residuals.shape == (3, 5)

    def test_pca_with_nan(self):
        """Should handle NaN values."""
        mat = np.random.randn(20, 8)
        mat[5, 3] = np.nan
        mat[10, 7] = np.nan
        df = pd.DataFrame(mat)
        loadings, scores, residuals = funding_pca(df)
        assert not np.isnan(residuals).all()

    def test_rolling_residual(self):
        """Rolling PCA residual for one symbol."""
        np.random.seed(42)
        n_times, n_syms = 60, 10
        columns = [f"SYM{i}" for i in range(n_syms)]
        common = np.random.randn(n_times) * 0.01
        # SYM3 has extra alpha
        mat = np.outer(common, np.ones(n_syms)) + np.random.randn(n_times, n_syms) * 0.001
        mat[:, 3] += 0.005  # persistent mispricing

        df = pd.DataFrame(mat, columns=columns)
        res = rolling_funding_residual(df, "SYM3", window=20)
        assert len(res) == n_times
        # SYM3 should have positive residual (it's the mispriced one)
        valid = res[~np.isnan(res)]
        assert len(valid) > 0
        assert np.mean(valid) > 0


# ===========================================================================
# Regime Snapshot Tests
# ===========================================================================

class TestRegimeSnapshot:
    def test_cascade_detection(self):
        r = RegimeSnapshot("BTCUSDT", vpin=0.6, ofi=-0.5, hawkes_ratio=4.0,
                          kyles_lambda=0.001, funding_residual=0.0)
        assert r.is_cascade is True
        assert r.cascade_direction == "short"

    def test_safe_for_carry(self):
        r = RegimeSnapshot("ETHUSDT", vpin=0.2, ofi=0.1, hawkes_ratio=1.0,
                          kyles_lambda=0.0005, funding_residual=0.002)
        assert r.is_safe_for_carry is True

    def test_not_safe_high_vpin(self):
        r = RegimeSnapshot("ETHUSDT", vpin=0.6, ofi=0.1, hawkes_ratio=1.0,
                          kyles_lambda=0.0005, funding_residual=0.0)
        assert r.is_safe_for_carry is False

    def test_post_cascade_reversion(self):
        r = RegimeSnapshot("ETHUSDT", vpin=0.4, ofi=-0.5, hawkes_ratio=1.2,
                          kyles_lambda=0.001, funding_residual=0.0)
        assert r.post_cascade_reversion == "long"  # fade the sell


# ===========================================================================
# Signal Generation Tests
# ===========================================================================

class TestSignals:
    def _make_regime(self, symbol, vpin=0.2, ofi=0.0, hawkes=1.0, funding_res=0.0):
        return RegimeSnapshot(symbol, vpin=vpin, ofi=ofi, hawkes_ratio=hawkes,
                             kyles_lambda=0.0, funding_residual=funding_res)

    def _make_snap(self, symbol, ann_funding=25.0, volume=100e6):
        return SymbolSnapshot(
            symbol=symbol, funding_rate=ann_funding / (1095 * 100),
            ann_funding_pct=ann_funding, mark_price=100.0, index_price=100.0,
            next_funding_time_ms=0, time_to_funding_min=60.0,
            volume_24h_usd=volume,
        )

    def test_carry_entry_signal(self):
        """High funding + low VPIN should generate carry entry."""
        config = SignalConfig()
        regimes = {"ETHUSDT": self._make_regime("ETHUSDT", vpin=0.2)}
        snapshots = {"ETHUSDT": self._make_snap("ETHUSDT", ann_funding=25.0)}

        signals = generate_carry_signals(regimes, snapshots, set(), config)
        entries = [s for s in signals if s.direction == "long"]
        assert len(entries) == 1
        assert entries[0].signal_type == "carry_A"

    def test_carry_blocked_by_vpin(self):
        """High VPIN should block carry entry (when VPIN filter is enabled)."""
        # VPIN filter is disabled by default (0.99). Enable it to test the logic.
        config = SignalConfig(carry_vpin_max=0.40)
        regimes = {"ETHUSDT": self._make_regime("ETHUSDT", vpin=0.6)}
        snapshots = {"ETHUSDT": self._make_snap("ETHUSDT", ann_funding=25.0)}

        signals = generate_carry_signals(regimes, snapshots, set(), config)
        entries = [s for s in signals if s.direction == "long"]
        assert len(entries) == 0

    def test_carry_exit_on_vpin_spike(self):
        """VPIN spike should exit carry position (when VPIN filter is enabled)."""
        # VPIN exit is disabled by default (0.99). Enable it to test the logic.
        config = SignalConfig(carry_vpin_exit=0.55)
        regimes = {"ETHUSDT": self._make_regime("ETHUSDT", vpin=0.6)}
        snapshots = {"ETHUSDT": self._make_snap("ETHUSDT", ann_funding=25.0)}

        signals = generate_carry_signals(
            regimes, snapshots, {"ETHUSDT"}, config
        )
        exits = [s for s in signals if s.direction == "exit"]
        assert len(exits) == 1

    def test_tick_vpin_blocks_carry_entry(self):
        """Tick-sourced VPIN > 0.80 should block carry entry with default config."""
        config = SignalConfig()  # tick_vpin_max=0.80 by default
        # Tick VPIN = 0.85 > 0.80 → blocked
        regimes = {"ETHUSDT": RegimeSnapshot(
            "ETHUSDT", vpin=0.85, ofi=0.0, hawkes_ratio=1.0,
            kyles_lambda=0.0, funding_residual=0.0, source="tick",
        )}
        snapshots = {"ETHUSDT": self._make_snap("ETHUSDT", ann_funding=25.0)}
        signals = generate_carry_signals(regimes, snapshots, set(), config)
        entries = [s for s in signals if s.direction == "long"]
        assert len(entries) == 0

    def test_tick_vpin_allows_carry_entry(self):
        """Tick-sourced VPIN < 0.80 should allow carry entry."""
        config = SignalConfig()
        # Tick VPIN = 0.50 < 0.80 → allowed
        regimes = {"ETHUSDT": RegimeSnapshot(
            "ETHUSDT", vpin=0.50, ofi=0.0, hawkes_ratio=1.0,
            kyles_lambda=0.0, funding_residual=0.0, source="tick",
        )}
        snapshots = {"ETHUSDT": self._make_snap("ETHUSDT", ann_funding=25.0)}
        signals = generate_carry_signals(regimes, snapshots, set(), config)
        entries = [s for s in signals if s.direction == "long"]
        assert len(entries) == 1
        assert "tick" in entries[0].reason

    def test_tick_vpin_zero_not_filtered(self):
        """Tick VPIN = 0 (not warmed up) should NOT trigger filtering."""
        config = SignalConfig()
        # Tick VPIN = 0.0 → falls back to kline threshold (0.99) → not blocked
        regimes = {"ETHUSDT": RegimeSnapshot(
            "ETHUSDT", vpin=0.0, ofi=0.0, hawkes_ratio=1.0,
            kyles_lambda=0.0, funding_residual=0.0, source="tick",
        )}
        snapshots = {"ETHUSDT": self._make_snap("ETHUSDT", ann_funding=25.0)}
        signals = generate_carry_signals(regimes, snapshots, set(), config)
        entries = [s for s in signals if s.direction == "long"]
        assert len(entries) == 1

    def test_tick_vpin_exit_on_extreme_toxicity(self):
        """Tick VPIN > 0.90 should trigger exit on active carry position."""
        config = SignalConfig()
        regimes = {"ETHUSDT": RegimeSnapshot(
            "ETHUSDT", vpin=0.92, ofi=0.0, hawkes_ratio=1.0,
            kyles_lambda=0.0, funding_residual=0.0, source="tick",
        )}
        snapshots = {"ETHUSDT": self._make_snap("ETHUSDT", ann_funding=25.0)}
        signals = generate_carry_signals(
            regimes, snapshots, {"ETHUSDT"}, config
        )
        exits = [s for s in signals if s.direction == "exit"]
        assert len(exits) == 1
        assert "tick" in exits[0].reason

    def test_carry_exit_on_low_funding(self):
        """Low funding should exit carry position."""
        config = SignalConfig()
        regimes = {"ETHUSDT": self._make_regime("ETHUSDT", vpin=0.2)}
        snapshots = {"ETHUSDT": self._make_snap("ETHUSDT", ann_funding=2.0)}

        signals = generate_carry_signals(
            regimes, snapshots, {"ETHUSDT"}, config
        )
        exits = [s for s in signals if s.direction == "exit"]
        assert len(exits) == 1

    def test_cascade_entry_signal(self):
        """High Hawkes ratio + directional OFI should trigger cascade signal."""
        # Cascade disabled by default (max=0). Enable to test the logic.
        config = SignalConfig(max_cascade_positions=2)
        regimes = {"BTCUSDT": self._make_regime("BTCUSDT", hawkes=5.0, ofi=-0.7)}

        signals = generate_cascade_signals(regimes, set(), config)
        entries = [s for s in signals if s.direction in ("long", "short")]
        assert len(entries) == 1
        assert entries[0].direction == "short"  # follows OFI direction

    def test_cascade_exit_when_over(self):
        """Cascade signal should exit when Hawkes ratio drops."""
        config = SignalConfig(max_cascade_positions=2)
        regimes = {"BTCUSDT": self._make_regime("BTCUSDT", hawkes=1.5)}

        signals = generate_cascade_signals(regimes, {"BTCUSDT"}, config)
        exits = [s for s in signals if s.direction == "exit"]
        assert len(exits) == 1

    def test_reversion_signal(self):
        """Post-cascade state should trigger reversion signal."""
        # Reversion disabled by default (max=0). Enable to test the logic.
        config = SignalConfig(max_reversion_positions=2)
        regimes = {"ETHUSDT": self._make_regime("ETHUSDT", vpin=0.4, ofi=-0.5, hawkes=1.2)}

        signals = generate_reversion_signals(regimes, set(), config)
        entries = [s for s in signals if s.direction in ("long", "short")]
        assert len(entries) == 1
        assert entries[0].direction == "long"  # fade the sell

    def test_max_positions_respected(self):
        """Should not exceed max positions per signal type."""
        config = SignalConfig(max_carry_positions=2)
        regimes = {
            f"SYM{i}USDT": self._make_regime(f"SYM{i}USDT", vpin=0.1)
            for i in range(5)
        }
        snapshots = {
            f"SYM{i}USDT": self._make_snap(f"SYM{i}USDT", ann_funding=30.0)
            for i in range(5)
        }

        signals = generate_carry_signals(regimes, snapshots, set(), config)
        entries = [s for s in signals if s.direction == "long"]
        assert len(entries) <= 2


# ===========================================================================
# State Persistence Tests
# ===========================================================================

class TestState:
    def test_save_load_roundtrip(self):
        """State should survive save/load cycle."""
        state = PortfolioState(started_at="2025-01-01T00:00:00+00:00")
        state.equity = 1.05
        state.total_entries = 3
        state.carry_positions["BTCUSDT"] = Position(
            symbol="BTCUSDT", signal_type="carry_A", direction="long",
            entry_time="2025-01-01T00:00:00+00:00", entry_price=50000.0,
            notional_weight=0.5, accumulated_pnl=0.001,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state.save(path)

            loaded = PortfolioState.load(path)
            assert loaded.equity == 1.05
            assert loaded.total_entries == 3
            assert "BTCUSDT" in loaded.carry_positions
            assert loaded.carry_positions["BTCUSDT"].entry_price == 50000.0

    def test_fresh_state(self):
        """Loading non-existent file should return fresh state."""
        loaded = PortfolioState.load(Path("/nonexistent/path/state.json"))
        assert loaded.equity == 1.0
        assert loaded.n_positions == 0

    def test_all_positions(self):
        """all_positions should aggregate across signal types."""
        state = PortfolioState()
        state.carry_positions["A"] = Position("A", "carry_A", "long", "t")
        state.cascade_positions["B"] = Position("B", "cascade_C", "short", "t")
        assert len(state.all_positions) == 2
        assert state.n_positions == 2

    def test_trade_log(self):
        state = PortfolioState()
        state.log_trade("BTCUSDT", "carry_A", "enter", "long", "test")
        assert len(state.trade_log) == 1
        assert state.trade_log[0]["symbol"] == "BTCUSDT"

    def test_trade_log_cap(self):
        """Trade log should cap at 500 entries."""
        state = PortfolioState()
        for i in range(600):
            state.log_trade(f"SYM{i}", "carry_A", "enter", "long", "test")
        assert len(state.trade_log) == 500


# ===========================================================================
# Trade Execution Tests
# ===========================================================================

class TestExecution:
    def test_execute_entry(self):
        """Execute an entry signal."""
        state = PortfolioState(started_at="2025-01-01T00:00:00+00:00")
        config = SignalConfig(cost_per_leg_bps=10.0)

        snap = SymbolSnapshot(
            "ETHUSDT", 0.0001, 10.95, 3000.0, 3000.0, 0, 60.0, 100e6
        )
        signal = TradeSignal(
            "ETHUSDT", "carry_A", "long", 0.8, "test entry"
        )
        msgs = execute_signals(state, [signal], {"ETHUSDT": snap}, config)
        assert len(msgs) == 1
        assert "ENTER" in msgs[0]
        assert "ETHUSDT" in state.carry_positions
        assert state.total_entries == 1
        assert state.equity < 1.0  # cost deducted

    def test_execute_exit(self):
        """Execute an exit signal."""
        state = PortfolioState(started_at="2025-01-01T00:00:00+00:00")
        state.carry_positions["ETHUSDT"] = Position(
            "ETHUSDT", "carry_A", "long", "2025-01-01T00:00:00+00:00",
            3000.0, 0.5, 0.8, 0.001, 0.0001,
        )
        config = SignalConfig(cost_per_leg_bps=10.0)

        signal = TradeSignal("ETHUSDT", "carry_A", "exit", 0.0, "test exit")
        msgs = execute_signals(state, [signal], {}, config)
        assert len(msgs) == 1
        assert "EXIT" in msgs[0]
        assert "ETHUSDT" not in state.carry_positions

    def test_ttl_exits(self):
        """Cascade positions should exit after TTL."""
        state = PortfolioState()
        state.cascade_positions["BTCUSDT"] = Position(
            "BTCUSDT", "cascade_C", "short", "2025-01-01T00:00:00+00:00",
            hold_bars=10,
        )
        config = SignalConfig(cascade_hold_bars=6)

        exits = check_ttl_exits(state, config)
        assert len(exits) == 1
        assert exits[0].symbol == "BTCUSDT"
        assert exits[0].direction == "exit"


# ===========================================================================
# Performance Stats Tests
# ===========================================================================

class TestPerformance:
    def test_compute_performance(self):
        """Performance metrics from equity history."""
        history = [
            ["2025-01-01T00:00:00+00:00", 1.0, "start"],
            ["2025-01-02T00:00:00+00:00", 1.01, "trade"],
            ["2025-01-03T00:00:00+00:00", 1.005, "trade"],
            ["2025-01-04T00:00:00+00:00", 1.02, "trade"],
        ]
        perf = compute_performance(history)
        assert perf is not None
        assert perf.total_return_pct > 0
        assert perf.max_drawdown_pct > 0
        assert perf.days_running > 2

    def test_insufficient_data(self):
        assert compute_performance([]) is None
        assert compute_performance([["t", 1.0]]) is None


# ===========================================================================
# Funding Matrix Builder Tests
# ===========================================================================

class TestFundingMatrix:
    def test_add_observation(self):
        builder = FundingMatrixBuilder(max_rows=10)
        snaps = [
            SymbolSnapshot("BTCUSDT", 0.0001, 10.95, 50000, 50000, 0, 60, 1e9),
            SymbolSnapshot("ETHUSDT", 0.0002, 21.9, 3000, 3000, 0, 60, 5e8),
        ]
        builder.add_observation(snaps)
        df = builder.to_dataframe()
        assert len(df) == 1
        assert "BTCUSDT" in df.columns

    def test_rolling_window(self):
        builder = FundingMatrixBuilder(max_rows=5)
        for i in range(10):
            snaps = [SymbolSnapshot(f"SYM{i % 3}USDT", 0.0001 * (i + 1),
                                   10.0, 100, 100, 0, 60, 1e8)]
            builder.add_observation(snaps)
        df = builder.to_dataframe()
        assert len(df) <= 5

    def test_serialization(self):
        builder = FundingMatrixBuilder()
        snaps = [SymbolSnapshot("BTCUSDT", 0.0001, 10.95, 50000, 50000, 0, 60, 1e9)]
        builder.add_observation(snaps)

        d = builder.to_dict()
        restored = FundingMatrixBuilder.from_dict(d)
        df = restored.to_dataframe()
        assert len(df) == 1


# ===========================================================================
# Annualize Funding Tests
# ===========================================================================

class TestAnnualize:
    def test_annualize_positive(self):
        assert abs(annualize_funding(0.0001) - 10.95) < 0.01

    def test_annualize_zero(self):
        assert annualize_funding(0) == 0.0

    def test_annualize_negative(self):
        assert annualize_funding(-0.0001) < 0
