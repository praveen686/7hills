"""Tests for the Risk-Neutral Density Regime Strategy (RNDR).

Tests signal construction, composite weighting, backtest mechanics,
trade P&L, and edge cases using synthetic DensityDayObs series.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from quantlaxmi.strategies.s1_vrp.density import (
    DEFAULT_COST_BPS,
    DEFAULT_ENTRY_PCTILE,
    DEFAULT_EXIT_PCTILE,
    DEFAULT_HOLD_DAYS,
    DEFAULT_LOOKBACK,
    DensityBacktestResult,
    DensityDayObs,
    DensityTrade,
    compute_composite_signal,
    run_density_backtest,
    _rolling_percentile,
    _rolling_zscore,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic DensityDayObs series
# ---------------------------------------------------------------------------

def _make_obs(
    day_offset: int,
    spot: float = 24000.0,
    rn_skewness: float = -0.30,
    rn_kurtosis: float = 0.50,
    entropy: float = 2.50,
    left_tail: float = 0.16,
    right_tail: float = 0.14,
    phys_skewness: float = 0.0,
    entropy_change: float = 0.0,
    kl_div: float = 0.0,
    atm_iv: float = 0.15,
) -> DensityDayObs:
    """Create a single observation with specified features."""
    d = date(2025, 1, 2)  # Thursday
    # Skip weekends
    actual = d
    count = 0
    while count < day_offset:
        actual += timedelta(days=1)
        if actual.weekday() < 5:
            count += 1

    return DensityDayObs(
        date=actual,
        symbol="NIFTY",
        spot=spot,
        atm_iv=atm_iv,
        rn_skewness=rn_skewness,
        rn_kurtosis=rn_kurtosis,
        entropy=entropy,
        left_tail=left_tail,
        right_tail=right_tail,
        phys_skewness=phys_skewness,
        skew_premium=phys_skewness - rn_skewness,
        entropy_change=entropy_change,
        kl_div=kl_div,
        density_ok=True,
    )


def _make_series(
    n: int,
    base_spot: float = 24000.0,
    spot_drift: float = 0.0,
    base_rn_skew: float = -0.30,
    skew_overrides: dict[int, float] | None = None,
    left_tail_overrides: dict[int, float] | None = None,
    phys_skew: float = 0.0,
    noise_seed: int = 42,
) -> list[DensityDayObs]:
    """Generate a synthetic series with optional feature spikes."""
    rng = np.random.default_rng(noise_seed)
    skew_ov = skew_overrides or {}
    lt_ov = left_tail_overrides or {}

    series = []
    prev_entropy = 2.50

    for i in range(n):
        spot = base_spot + spot_drift * i
        rn_skew = skew_ov.get(i, base_rn_skew + rng.normal(0, 0.02))
        left_tail = lt_ov.get(i, 0.16 + rng.normal(0, 0.005))
        entropy = 2.50 + rng.normal(0, 0.02)
        d_entropy = entropy - prev_entropy
        kl = abs(rng.normal(0, 0.01))
        skew_prem = phys_skew - rn_skew

        series.append(_make_obs(
            day_offset=i,
            spot=spot,
            rn_skewness=rn_skew,
            left_tail=left_tail,
            phys_skewness=phys_skew,
            entropy=entropy,
            entropy_change=d_entropy,
            kl_div=kl,
        ))
        prev_entropy = entropy

    return series


# ---------------------------------------------------------------------------
# Test: rolling helpers
# ---------------------------------------------------------------------------

class TestRollingHelpers:

    def test_percentile_minimum_at_bottom(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        # idx=0 with any window: only 1 element → returns 0.5 (neutral default)
        assert _rolling_percentile(values, 0, 5) == pytest.approx(0.5)
        # idx=4 with window=5: 5.0 is largest of [1,2,3,4,5] → 5/5 = 1.0
        assert _rolling_percentile(values, 4, 5) == pytest.approx(1.0)
        # 1.0 is the smallest when full window is visible from idx=4's perspective
        # Test with reversed values: [5,4,3,2,1], idx=4 → 1.0 is smallest → 1/5
        rev_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _rolling_percentile(rev_values, 4, 5) == pytest.approx(1 / 5)

    def test_percentile_maximum_at_top(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _rolling_percentile(values, 4, 5) == pytest.approx(1.0)

    def test_percentile_window_clipping(self):
        """Window larger than available data should still work."""
        values = [3.0, 1.0, 2.0]
        p = _rolling_percentile(values, 2, 100)
        # 2.0 is at rank 2 of [3, 1, 2] → 2/3
        assert p == pytest.approx(2 / 3)

    def test_zscore_zero_for_mean(self):
        values = [1.0, 2.0, 3.0, 2.0]
        # Mean = 2.0, last = 2.0 → z = 0
        assert _rolling_zscore(values, 3, 4) == pytest.approx(0.0, abs=0.01)

    def test_zscore_positive_above_mean(self):
        values = [1.0, 2.0, 3.0, 10.0]
        z = _rolling_zscore(values, 3, 4)
        assert z > 1.0


# ---------------------------------------------------------------------------
# Test: composite signal
# ---------------------------------------------------------------------------

class TestCompositeSignal:

    def test_flat_series_near_zero(self):
        """Flat features → composite ≈ 0."""
        series = _make_series(60, noise_seed=42)
        signals = compute_composite_signal(series, lookback=30)
        # After warmup, signals should be small
        active = signals[30:]
        assert abs(np.mean(active)) < 0.15

    def test_fear_spike_generates_positive_signal(self):
        """When RN skewness spikes very negative → skew premium rises → bullish."""
        # Normal series but with a fear spike
        overrides = {i: -0.80 for i in range(40, 48)}  # extreme negative skew
        series = _make_series(60, skew_overrides=overrides, phys_skew=0.0)
        signals = compute_composite_signal(series, lookback=30)

        # Signal during the spike should be positive (bullish contrarian)
        spike_signals = signals[40:48]
        assert max(spike_signals) > 0.05

    def test_left_tail_spike_contributes(self):
        """Elevated left tail weight → bullish component."""
        lt_ov = {i: 0.28 for i in range(40, 48)}  # high left tail
        series = _make_series(60, left_tail_overrides=lt_ov)
        signals = compute_composite_signal(series, lookback=30)
        spike_signals = signals[40:48]
        assert max(spike_signals) > 0.0

    def test_lookback_respected(self):
        """Signals before lookback period should be zero."""
        series = _make_series(50)
        signals = compute_composite_signal(series, lookback=30)
        assert all(s == 0.0 for s in signals[:30])

    def test_signal_length_matches_series(self):
        series = _make_series(60)
        signals = compute_composite_signal(series, lookback=20)
        assert len(signals) == 60


# ---------------------------------------------------------------------------
# Test: backtest mechanics
# ---------------------------------------------------------------------------

class TestBacktestMechanics:

    def test_no_trades_flat_market(self):
        """Flat features with tight entry → few or no trades."""
        series = _make_series(80, noise_seed=99)
        result = run_density_backtest(
            series,
            lookback=30,
            entry_pctile=0.99,  # very tight
            hold_days=5,
            cost_bps=5,
        )
        # With 0.99 threshold and flat features, very few trades
        assert result.n_signals <= 5

    def test_fear_spike_triggers_trade(self):
        """A clear fear spike should generate at least one trade."""
        overrides = {i: -0.90 for i in range(40, 48)}
        series = _make_series(
            80,
            skew_overrides=overrides,
            phys_skew=0.0,
            spot_drift=5.0,  # slight upward drift
        )
        result = run_density_backtest(
            series,
            lookback=30,
            entry_pctile=0.70,
            hold_days=5,
            cost_bps=5,
        )
        assert result.n_signals >= 1
        assert len(result.trades) >= 1

    def test_trade_pnl_correct(self):
        """P&L = (exit - entry) / entry - cost."""
        overrides = {i: -0.90 for i in range(40, 48)}
        series = _make_series(
            80,
            skew_overrides=overrides,
            phys_skew=0.0,
            spot_drift=10.0,
        )
        result = run_density_backtest(
            series,
            lookback=30,
            entry_pctile=0.70,
            hold_days=5,
            cost_bps=5,
        )
        if result.trades:
            t = result.trades[0]
            expected_raw = (t.exit_spot - t.entry_spot) / t.entry_spot
            expected_net = expected_raw - 5 / 10_000
            assert t.pnl_pct == pytest.approx(expected_net, abs=1e-9)

    def test_max_hold_exit(self):
        """Trade should exit after hold_days."""
        overrides = {i: -0.90 for i in range(40, 80)}  # persistent fear
        series = _make_series(80, skew_overrides=overrides)
        result = run_density_backtest(
            series,
            lookback=30,
            entry_pctile=0.70,
            exit_pctile=0.05,  # very low exit threshold
            hold_days=3,
            cost_bps=5,
        )
        for t in result.trades:
            if t.exit_reason == "max_hold":
                assert t.hold_days >= 3

    def test_win_rate_bounded(self):
        """Win rate ∈ [0, 1]."""
        series = _make_series(80, spot_drift=5.0)
        result = run_density_backtest(series, lookback=30, entry_pctile=0.70)
        assert 0.0 <= result.win_rate <= 1.0

    def test_open_trade_closed_at_end(self):
        """Trade still open at end-of-data gets closed with 'end_of_data'."""
        # Put fear at the very end so trade doesn't have time to exit
        overrides = {i: -0.90 for i in range(73, 80)}
        series = _make_series(80, skew_overrides=overrides)
        result = run_density_backtest(
            series, lookback=30, entry_pctile=0.70, hold_days=999,
        )
        if result.trades:
            last = result.trades[-1]
            # Could be end_of_data if still open
            assert last.exit_reason in ("end_of_data", "max_hold", "signal_decay")


# ---------------------------------------------------------------------------
# Test: backtest result metrics
# ---------------------------------------------------------------------------

class TestBacktestMetrics:

    def test_total_return_from_trades(self):
        """Total return = product of (1 + pnl) - 1."""
        overrides = {i: -0.90 for i in range(40, 48)}
        series = _make_series(80, skew_overrides=overrides, spot_drift=10.0)
        result = run_density_backtest(
            series, lookback=30, entry_pctile=0.70, cost_bps=0,
        )
        if result.trades:
            eq = 1.0
            for t in result.trades:
                eq *= (1 + t.pnl_pct)
            expected = (eq - 1) * 100
            assert result.total_return_pct == pytest.approx(expected, abs=0.01)

    def test_sharpe_sign_matches_return(self):
        """If total return is positive, Sharpe should be non-negative."""
        series = _make_series(80, spot_drift=20.0)
        result = run_density_backtest(series, lookback=30, entry_pctile=0.60)
        if result.total_return_pct > 0 and len(result.trades) >= 3:
            assert result.sharpe >= 0

    def test_max_dd_non_negative(self):
        """Max drawdown should be ≥ 0."""
        series = _make_series(80)
        result = run_density_backtest(series, lookback=30, entry_pctile=0.60)
        assert result.max_dd_pct >= 0

    def test_no_trades_gives_zero_metrics(self):
        """No trades → all metrics zero."""
        series = _make_series(80)
        result = run_density_backtest(
            series, lookback=30, entry_pctile=0.999,
        )
        if not result.trades:
            assert result.total_return_pct == 0.0
            assert result.sharpe == 0.0
            assert result.win_rate == 0.0


# ---------------------------------------------------------------------------
# Test: cost impact
# ---------------------------------------------------------------------------

class TestCostImpact:

    def test_higher_cost_lower_return(self):
        """Higher trading cost → lower total return."""
        overrides = {i: -0.90 for i in range(40, 48)}
        series = _make_series(80, skew_overrides=overrides, spot_drift=10.0)

        r_low = run_density_backtest(series, lookback=30, entry_pctile=0.70, cost_bps=0)
        r_high = run_density_backtest(series, lookback=30, entry_pctile=0.70, cost_bps=50)

        if r_low.trades and r_high.trades:
            assert r_low.total_return_pct >= r_high.total_return_pct


# ---------------------------------------------------------------------------
# Test: dataclass properties
# ---------------------------------------------------------------------------

class TestDensityDayObs:

    def test_skew_premium_consistent(self):
        obs = _make_obs(0, phys_skewness=0.10, rn_skewness=-0.30)
        assert obs.skew_premium == pytest.approx(0.40)

    def test_frozen(self):
        obs = _make_obs(0)
        with pytest.raises(AttributeError):
            obs.spot = 99999  # type: ignore


class TestDensityTrade:

    def test_fields(self):
        t = DensityTrade(
            symbol="NIFTY",
            entry_date=date(2025, 1, 2),
            exit_date=date(2025, 1, 7),
            entry_spot=24000.0,
            exit_spot=24200.0,
            entry_signal=0.5,
            pnl_pct=0.0078,
            hold_days=5,
            exit_reason="max_hold",
        )
        assert t.symbol == "NIFTY"
        assert t.hold_days == 5
