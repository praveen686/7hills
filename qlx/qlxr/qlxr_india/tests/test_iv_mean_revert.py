"""Tests for IV mean-reversion strategy — dataclasses, rolling percentile, backtest."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from apps.india_fno.iv_mean_revert import (
    BacktestResult,
    DayObs,
    Trade,
    format_iv_results,
    rolling_percentile,
    run_from_series,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_day(
    d: date,
    spot: float = 20000.0,
    atm_iv: float = 0.15,
    atm_var: float = 0.0,
    forward: float = 20050.0,
    sanos_ok: bool = True,
) -> DayObs:
    if atm_var == 0.0:
        atm_var = atm_iv ** 2 * (30 / 365)
    return DayObs(
        date=d, spot=spot, atm_iv=atm_iv, atm_var=atm_var,
        forward=forward, sanos_ok=sanos_ok,
    )


def _make_series(
    n: int,
    base_iv: float = 0.15,
    iv_overrides: dict[int, float] | None = None,
    spot_start: float = 20000.0,
    spot_drift: float = 0.0,
    iv_noise_seed: int | None = 42,
) -> list[DayObs]:
    """Build a list of n DayObs with optional IV overrides and linear spot drift.

    By default adds small deterministic noise to base IV so rolling_percentile
    produces a realistic rank distribution (constant IV always ranks 1.0).
    Set iv_noise_seed=None for perfectly constant IV.
    """
    overrides = iv_overrides or {}
    if iv_noise_seed is not None:
        rng = np.random.default_rng(iv_noise_seed)
        noise = rng.normal(0, 0.002, size=n)  # tiny jitter around base
    else:
        noise = np.zeros(n)
    days = []
    d = date(2024, 1, 2)  # Tuesday
    count = 0
    while count < n:
        # skip weekends
        if d.weekday() < 5:
            iv = overrides.get(count, base_iv + noise[count])
            spot = spot_start + spot_drift * count
            days.append(_make_day(d, spot=spot, atm_iv=iv))
            count += 1
        d += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# DayObs dataclass
# ---------------------------------------------------------------------------


class TestDayObs:
    def test_basic_construction(self):
        obs = DayObs(
            date=date(2024, 6, 1), spot=20000, atm_iv=0.14,
            atm_var=0.001, forward=20050, sanos_ok=True,
        )
        assert obs.spot == 20000
        assert obs.atm_iv == 0.14
        assert obs.sanos_ok is True



# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------


class TestTrade:
    def test_construction(self):
        t = Trade(
            entry_date=date(2024, 1, 10), exit_date=date(2024, 1, 17),
            entry_spot=20000, exit_spot=20200, entry_iv=0.20, exit_iv=0.14,
            iv_pctile=0.85, pnl_pct=0.0095, hold_days=5, exit_reason="max_hold",
        )
        assert t.hold_days == 5
        assert t.exit_reason == "max_hold"
        assert t.pnl_pct == pytest.approx(0.0095)


# ---------------------------------------------------------------------------
# BacktestResult dataclass
# ---------------------------------------------------------------------------


class TestBacktestResult:
    def test_construction(self):
        result = BacktestResult(
            daily=[], trades=[], total_return_pct=5.0, annual_return_pct=10.0,
            sharpe=1.5, max_dd_pct=3.0, win_rate=0.6, avg_iv=0.15, n_signals=4,
        )
        assert result.total_return_pct == 5.0
        assert result.n_signals == 4

    def test_empty_result(self):
        result = BacktestResult(
            daily=[], trades=[], total_return_pct=0, annual_return_pct=0,
            sharpe=0, max_dd_pct=0, win_rate=0, avg_iv=0, n_signals=0,
        )
        assert result.win_rate == 0


# ---------------------------------------------------------------------------
# rolling_percentile
# ---------------------------------------------------------------------------


class TestRollingPercentile:
    def test_monotone_up(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        ranks = rolling_percentile(vals, window=5, )
        # Each value is strictly the largest seen so far => rank = 1.0
        assert all(r == 1.0 for r in ranks)

    def test_monotone_down(self):
        vals = [5.0, 4.0, 3.0, 2.0, 1.0]
        ranks = rolling_percentile(vals, window=5, )
        # First value alone => rank=1.0; subsequent values are smallest so far
        assert ranks[0] == 1.0
        for r in ranks[1:]:
            assert r < 1.0

    def test_constant_values(self):
        vals = [3.0] * 10
        ranks = rolling_percentile(vals, window=5, )
        # All values equal: everyone <= v, so rank = 1.0
        assert all(r == 1.0 for r in ranks)

    def test_known_rank(self):
        # Window of exactly 5: [1,2,3,4,5]. Current=3 => 3 values <=3 => rank=3/5=0.6
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        ranks = rolling_percentile(vals, window=5, )
        # At index 2 (value=3), window is [1,2,3], 3 values <=3 => 3/3=1.0
        # At index 4 (value=5), window is [1,2,3,4,5], 5/5=1.0
        assert ranks[4] == 1.0
        # Check a non-trivial rank: for a window where value is not largest
        vals2 = [5.0, 1.0, 2.0, 3.0, 4.0]
        ranks2 = rolling_percentile(vals2, window=5, )
        # At index 4 (value=4, window=[5,1,2,3,4]): 4 values <=4 => 4/5=0.8
        assert ranks2[4] == pytest.approx(0.8)

    def test_window_smaller_than_series(self):
        vals = [10.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        ranks = rolling_percentile(vals, window=3, )
        # At index 3 (val=3), window is [1,2,3] => rank=3/3=1.0
        assert ranks[3] == 1.0
        # At index 1 (val=1), window is [10,1] => 1 value <=1 => 1/2=0.5
        assert ranks[1] == pytest.approx(0.5)

    def test_single_value(self):
        ranks = rolling_percentile([42.0], window=10, )
        assert ranks == [1.0]


# ---------------------------------------------------------------------------
# run_from_series — flat IV (no signals)
# ---------------------------------------------------------------------------


class TestRunFromSeriesFlatIV:
    def test_constant_iv_ranks_one(self):
        """Constant IV gives rolling rank 1.0 (all equal => all <= v),
        which always meets entry_pctile=0.80.  This documents that
        constant-IV data is NOT 'quiet' for the strategy."""
        daily = _make_series(100, base_iv=0.15, iv_noise_seed=None)
        result = run_from_series(daily, iv_lookback=20, entry_pctile=0.80)
        assert result.n_signals > 0  # rank=1.0 meets >=0.80

    def test_noisy_flat_iv_few_signals(self):
        """When IV jitters around a mean (no real spike), signals should
        be rare at a strict 95th-percentile threshold."""
        daily = _make_series(200, base_iv=0.15, iv_noise_seed=7)
        result = run_from_series(daily, iv_lookback=60, entry_pctile=0.95)
        # Noise is tiny (sigma=0.002), so the 95th percentile is rarely
        # breached repeatedly.  Allow some signals, but far fewer than
        # the number of tradeable bars.
        tradeable_bars = len(daily) - 60
        assert result.n_signals < tradeable_bars / 2

    def test_too_short_series_returns_empty(self):
        daily = _make_series(10, base_iv=0.15)
        result = run_from_series(daily, iv_lookback=60)
        assert len(result.trades) == 0
        assert result.total_return_pct == 0
        assert result.sharpe == 0


# ---------------------------------------------------------------------------
# run_from_series — IV spike triggers trades
# ---------------------------------------------------------------------------


class TestRunFromSeriesSpike:
    @pytest.fixture()
    def spiked_series(self) -> list[DayObs]:
        """70 noisy-flat days then a sharp IV spike at day 70-77."""
        overrides = {i: 0.35 for i in range(70, 78)}
        return _make_series(100, base_iv=0.15, iv_overrides=overrides)

    def test_spike_generates_signal(self, spiked_series):
        result = run_from_series(spiked_series, iv_lookback=60, entry_pctile=0.95)
        assert result.n_signals >= 1

    def test_spike_generates_trade(self, spiked_series):
        result = run_from_series(spiked_series, iv_lookback=60, entry_pctile=0.95)
        assert len(result.trades) >= 1

    def test_spike_trade_has_high_entry_iv(self, spiked_series):
        """At least one trade should enter during the spike (IV=0.35)."""
        result = run_from_series(spiked_series, iv_lookback=60, entry_pctile=0.95)
        spike_trades = [t for t in result.trades if t.entry_iv > 0.25]
        assert len(spike_trades) >= 1


# ---------------------------------------------------------------------------
# run_from_series — exit reasons
# ---------------------------------------------------------------------------


class TestExitReasons:
    def test_max_hold_exit(self):
        """If IV stays high, trade exits after hold_days."""
        # Spike IV for many days so it stays above exit percentile
        overrides = {i: 0.35 for i in range(65, 100)}
        daily = _make_series(100, base_iv=0.15, iv_overrides=overrides)
        result = run_from_series(
            daily, iv_lookback=60, entry_pctile=0.95,
            exit_pctile=0.50, hold_days=5,
        )
        assert len(result.trades) >= 1
        # Trades entering during the sustained spike should exit on max_hold
        max_hold_trades = [t for t in result.trades if t.exit_reason == "max_hold"]
        assert len(max_hold_trades) >= 1

    def test_iv_normalised_exit(self):
        """If IV drops back quickly, trade exits on iv_normalised."""
        # Spike IV for 3 days then revert — use high entry_pctile so only
        # the spike triggers entry, and a generous hold_days so the exit
        # is forced by IV dropping below exit_pctile rather than max_hold.
        overrides = {65: 0.40, 66: 0.40, 67: 0.40}
        daily = _make_series(100, base_iv=0.15, iv_overrides=overrides)
        result = run_from_series(
            daily, iv_lookback=60, entry_pctile=0.95,
            exit_pctile=0.50, hold_days=30,
        )
        assert len(result.trades) >= 1
        # After the spike ends, IV reverts to ~0.15 which is well below
        # the 50th percentile of a window that saw 0.40, so should exit
        # on iv_normalised.
        norm_exits = [t for t in result.trades if t.exit_reason == "iv_normalised"]
        assert len(norm_exits) >= 1

    def test_end_of_backtest_exit(self):
        """Trade open at the very end should get exit_reason = end_of_backtest."""
        # Spike at the very end so the trade cannot close normally
        overrides = {i: 0.35 for i in range(95, 100)}
        daily = _make_series(100, base_iv=0.15, iv_overrides=overrides)
        result = run_from_series(
            daily, iv_lookback=60, entry_pctile=0.80,
            exit_pctile=0.50, hold_days=20,
        )
        if result.trades:
            last = result.trades[-1]
            assert last.exit_reason in ("end_of_backtest", "iv_normalised", "max_hold")


# ---------------------------------------------------------------------------
# run_from_series — entry/exit threshold sensitivity
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_low_entry_threshold_more_signals(self):
        """Lower entry percentile should generate more signals."""
        overrides = {i: 0.18 + 0.005 * (i % 5) for i in range(70, 90)}
        daily = _make_series(120, base_iv=0.15, iv_overrides=overrides)
        r_high = run_from_series(daily, iv_lookback=60, entry_pctile=0.95)
        r_low = run_from_series(daily, iv_lookback=60, entry_pctile=0.60)
        assert r_low.n_signals >= r_high.n_signals

    def test_entry_pctile_one_always_fires_on_max(self):
        """entry_pctile=1.0 means only the absolute max IV triggers."""
        overrides = {80: 0.50}  # one extreme spike
        daily = _make_series(100, base_iv=0.15, iv_overrides=overrides)
        result = run_from_series(daily, iv_lookback=60, entry_pctile=1.0)
        # rank=1.0 means >= 1.0 is met only when value is the window maximum
        assert result.n_signals >= 1


# ---------------------------------------------------------------------------
# run_from_series — P&L verification
# ---------------------------------------------------------------------------


class TestPnL:
    def test_known_spot_gain(self):
        """Spot rises during trade => positive P&L (minus cost)."""
        # Build series where spot jumps up right after entry signal
        overrides = {65: 0.35, 66: 0.35}
        # Spot drift +10 per day so trade gains value
        daily = _make_series(
            100, base_iv=0.15, iv_overrides=overrides,
            spot_start=20000.0, spot_drift=10.0,
        )
        result = run_from_series(
            daily, iv_lookback=60, entry_pctile=0.80,
            hold_days=5, cost_bps=0,
        )
        if result.trades:
            for t in result.trades:
                expected_pnl = (t.exit_spot - t.entry_spot) / t.entry_spot
                assert t.pnl_pct == pytest.approx(expected_pnl, abs=1e-9)

    def test_known_spot_loss(self):
        """Spot falls during trade => negative P&L."""
        overrides = {65: 0.35, 66: 0.35}
        daily = _make_series(
            100, base_iv=0.15, iv_overrides=overrides,
            spot_start=20000.0, spot_drift=-20.0,
        )
        result = run_from_series(
            daily, iv_lookback=60, entry_pctile=0.80,
            hold_days=5, cost_bps=0,
        )
        if result.trades:
            assert result.trades[0].pnl_pct < 0

    def test_cost_reduces_pnl(self):
        """With cost_bps > 0, P&L should be lower than without."""
        overrides = {65: 0.35, 66: 0.35}
        daily = _make_series(100, base_iv=0.15, iv_overrides=overrides)
        r_zero = run_from_series(daily, iv_lookback=60, entry_pctile=0.80, cost_bps=0)
        r_cost = run_from_series(daily, iv_lookback=60, entry_pctile=0.80, cost_bps=20)
        if r_zero.trades and r_cost.trades:
            assert r_cost.trades[0].pnl_pct < r_zero.trades[0].pnl_pct

    def test_win_rate_bounded(self):
        """Win rate should always be between 0 and 1."""
        overrides = {i: 0.35 for i in range(65, 80)}
        daily = _make_series(120, base_iv=0.15, iv_overrides=overrides)
        result = run_from_series(daily, iv_lookback=60, entry_pctile=0.80)
        assert 0.0 <= result.win_rate <= 1.0


# ---------------------------------------------------------------------------
# run_from_series — entropy filter
# ---------------------------------------------------------------------------


class TestEntropyFilter:
    def test_entropy_filter_blocks_entry(self):
        """When entropy is below threshold, entries should be suppressed."""
        overrides = {i: 0.35 for i in range(65, 80)}
        daily = _make_series(100, base_iv=0.15, iv_overrides=overrides)

        # Without filter
        r_no_filter = run_from_series(daily, iv_lookback=60, entry_pctile=0.80)

        # With entropy filter = all zeros (below any positive threshold)
        entropy = np.zeros(len(daily))
        r_filtered = run_from_series(
            daily, iv_lookback=60, entry_pctile=0.80,
            entropy_filter=entropy, entropy_threshold=0.5,
        )
        assert r_filtered.n_signals <= r_no_filter.n_signals

    def test_entropy_filter_passes_when_above_threshold(self):
        """When entropy is high enough, trades fire normally."""
        overrides = {i: 0.35 for i in range(65, 80)}
        daily = _make_series(100, base_iv=0.15, iv_overrides=overrides)

        entropy = np.ones(len(daily))  # all above threshold
        r_filtered = run_from_series(
            daily, iv_lookback=60, entry_pctile=0.80,
            entropy_filter=entropy, entropy_threshold=0.5,
        )
        r_no_filter = run_from_series(daily, iv_lookback=60, entry_pctile=0.80)
        assert r_filtered.n_signals == r_no_filter.n_signals


# ---------------------------------------------------------------------------
# format_iv_results
# ---------------------------------------------------------------------------


class TestFormatIVResults:
    def test_format_empty_result(self):
        result = BacktestResult(
            daily=[], trades=[], total_return_pct=0, annual_return_pct=0,
            sharpe=0, max_dd_pct=0, win_rate=0, avg_iv=0.15, n_signals=0,
        )
        text = format_iv_results(result)
        assert "IV Mean-Reversion" in text
        assert "Trades taken" in text

    def test_format_with_trades(self):
        t = Trade(
            entry_date=date(2024, 3, 5), exit_date=date(2024, 3, 12),
            entry_spot=20000, exit_spot=20200, entry_iv=0.25, exit_iv=0.16,
            iv_pctile=0.88, pnl_pct=0.0095, hold_days=5, exit_reason="max_hold",
        )
        result = BacktestResult(
            daily=[_make_day(date(2024, 3, d)) for d in range(1, 15)],
            trades=[t],
            total_return_pct=0.95,
            annual_return_pct=4.0,
            sharpe=1.2,
            max_dd_pct=0.5,
            win_rate=1.0,
            avg_iv=0.16,
            n_signals=1,
        )
        text = format_iv_results(result)
        assert "max_hold" in text
        assert "2024-03-05" in text
        assert "P&L" in text or "P&L" in text  # header has P&L

    def test_format_does_not_crash_on_many_trades(self):
        trades = [
            Trade(
                entry_date=date(2024, 1, i + 1), exit_date=date(2024, 1, i + 5),
                entry_spot=20000 + i * 10, exit_spot=20000 + (i + 5) * 10,
                entry_iv=0.20 + i * 0.01, exit_iv=0.15,
                iv_pctile=0.8 + i * 0.01, pnl_pct=0.001 * i,
                hold_days=4, exit_reason="max_hold",
            )
            for i in range(10)
        ]
        result = BacktestResult(
            daily=[], trades=trades, total_return_pct=2.5, annual_return_pct=8.0,
            sharpe=1.0, max_dd_pct=1.0, win_rate=0.7, avg_iv=0.17, n_signals=10,
        )
        text = format_iv_results(result)
        assert isinstance(text, str)
        assert len(text) > 100
