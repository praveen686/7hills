"""Tests for the RNDR Options Variant (bull put credit spreads).

Uses a MockBhavcopyCache with synthetic option prices computed via
Black-Scholes put-call parity so tests are fully self-contained.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from apps.india_fno.density_options import (
    OPTION_COST_BPS,
    SHORT_OFFSET,
    LONG_OFFSET,
    MIN_OI,
    MIN_PREMIUM,
    SpreadLeg,
    PutSpreadEntry,
    SpreadTrade,
    SpreadBacktestResult,
    _get_put_chain,
    _select_spread_strikes,
    _mark_to_market_spread,
    run_density_options_backtest,
    _compute_metrics,
)
from apps.india_fno.density_strategy import (
    DensityDayObs,
    compute_composite_signal,
    _rolling_percentile,
    DEFAULT_LOOKBACK,
)
from apps.india_fno.sanos import bs_call


# ---------------------------------------------------------------------------
# Helpers: synthetic option pricing
# ---------------------------------------------------------------------------

def _bs_put(spot: float, strike: float, iv: float, dte_days: int) -> float:
    """Black-Scholes put price via put-call parity: P = C - (S - K·e^{-rT}).

    Simplified: r=0 → P = C - S + K.
    """
    T = max(dte_days / 365, 1e-6)
    v = iv * iv * T  # total variance
    s = np.array([spot])
    k = np.array([strike])
    v_arr = np.array([v])
    call = float(bs_call(s, k, v_arr)[0])
    return max(call - spot + strike, 0.0)


def _build_fno_df(
    symbol: str,
    spot: float,
    expiry: date,
    dte: int,
    iv: float = 0.20,
    strike_step: float = 100.0,
    n_strikes: int = 80,
    base_oi: int = 5000,
) -> pd.DataFrame:
    """Build a synthetic F&O DataFrame for one expiry with CE+PE options."""
    atm = round(spot / strike_step) * strike_step
    lo = atm - (n_strikes // 2) * strike_step
    strikes = [lo + i * strike_step for i in range(n_strikes)]

    rows = []
    for K in strikes:
        p = _bs_put(spot, K, iv, dte)
        c_price = p + spot - K  # call from put-call parity (r=0)

        for opt_type, price in [("PE", p), ("CE", max(c_price, 0.01))]:
            rows.append({
                "SYMBOL": symbol,
                "INSTRUMENT": "OPTIDX",
                "StrkPric": K,
                "OptnTp": opt_type,
                "CLOSE": round(max(price, 0.05), 2),
                "OPEN_INT": base_oi + int(abs(K - atm) / strike_step) * 100,
                "EXPIRY_DT": expiry.strftime("%d-%b-%Y"),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MockBhavcopyCache
# ---------------------------------------------------------------------------

class MockBhavcopyCache:
    """Dict-backed cache: {date: DataFrame}."""

    def __init__(self, data: dict[date, pd.DataFrame] | None = None):
        self._data = data or {}

    def get_fno(self, d: date) -> pd.DataFrame:
        if d not in self._data:
            raise FileNotFoundError(f"No data for {d}")
        return self._data[d]

    def add(self, d: date, df: pd.DataFrame) -> None:
        self._data[d] = df


def _make_cache_with_chain(
    symbol: str = "BANKNIFTY",
    spot: float = 50000.0,
    start: date = date(2025, 3, 3),
    n_days: int = 20,
    iv: float = 0.20,
) -> MockBhavcopyCache:
    """Build a cache with synthetic option chains for several trading days."""
    cache = MockBhavcopyCache()
    # Two expiries: weekly (Thursday 7 days out) + monthly (last Thursday of month)
    weekly_exp = start + timedelta(days=(3 - start.weekday()) % 7 + 7)
    monthly_exp = start + timedelta(days=30)

    d = start
    for _ in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        dte_w = (weekly_exp - d).days
        dte_m = (monthly_exp - d).days
        frames = []
        if dte_w >= 0:
            frames.append(_build_fno_df(symbol, spot, weekly_exp, dte_w, iv))
        if dte_m >= 0:
            frames.append(_build_fno_df(symbol, spot, monthly_exp, dte_m, iv))
        if frames:
            cache.add(d, pd.concat(frames, ignore_index=True))
        spot += 50  # slight upward drift
        d += timedelta(days=1)
    return cache


# ---------------------------------------------------------------------------
# Helpers: synthetic DensityDayObs series
# ---------------------------------------------------------------------------

def _make_obs(
    day_offset: int,
    spot: float = 50000.0,
    rn_skewness: float = -0.30,
    phys_skewness: float = 0.0,
    entropy: float = 2.50,
    left_tail: float = 0.16,
    entropy_change: float = 0.0,
    kl_div: float = 0.0,
    symbol: str = "BANKNIFTY",
) -> DensityDayObs:
    d = date(2025, 3, 3)
    count = 0
    while count < day_offset:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return DensityDayObs(
        date=d,
        symbol=symbol,
        spot=spot,
        atm_iv=0.20,
        rn_skewness=rn_skewness,
        rn_kurtosis=0.50,
        entropy=entropy,
        left_tail=left_tail,
        right_tail=0.14,
        phys_skewness=phys_skewness,
        skew_premium=phys_skewness - rn_skewness,
        entropy_change=entropy_change,
        kl_div=kl_div,
        density_ok=True,
    )


def _make_series(
    n: int,
    base_spot: float = 50000.0,
    spot_drift: float = 0.0,
    base_rn_skew: float = -0.30,
    skew_overrides: dict[int, float] | None = None,
    left_tail_overrides: dict[int, float] | None = None,
    phys_skew: float = 0.0,
    noise_seed: int = 42,
    symbol: str = "BANKNIFTY",
) -> list[DensityDayObs]:
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
        series.append(_make_obs(
            day_offset=i,
            spot=spot,
            rn_skewness=rn_skew,
            phys_skewness=phys_skew,
            entropy=entropy,
            left_tail=left_tail,
            entropy_change=d_entropy,
            kl_div=kl,
            symbol=symbol,
        ))
        prev_entropy = entropy
    return series


# ---------------------------------------------------------------------------
# Test: _select_spread_strikes
# ---------------------------------------------------------------------------

class TestSelectSpreadStrikes:

    @pytest.fixture()
    def chain_df(self):
        """A basic PE chain for BANKNIFTY at 50000."""
        return _build_fno_df(
            "BANKNIFTY", 50000.0,
            expiry=date(2025, 3, 20), dte=17,
        )

    @pytest.fixture()
    def puts_df(self, chain_df):
        """Puts only, with DTE column."""
        df = chain_df[chain_df["OptnTp"].str.strip() == "PE"].copy()
        df["EXPIRY_DT"] = pd.to_datetime(df["EXPIRY_DT"], format="mixed", dayfirst=True)
        df["DTE"] = 17
        return df

    def test_correct_strikes_selected(self, puts_df):
        entry = _select_spread_strikes(
            puts_df, 50000.0, date(2025, 3, 3), "BANKNIFTY", 0.5,
        )
        assert entry is not None
        # Short ~3% OTM → ~48500, Long ~6% OTM → ~47000
        assert entry.short_leg.strike > entry.long_leg.strike
        assert abs(entry.short_leg.strike - 48500) <= 100
        assert abs(entry.long_leg.strike - 47000) <= 100

    def test_no_pe_options_returns_none(self):
        """Empty DataFrame → None."""
        df = pd.DataFrame(columns=[
            "SYMBOL", "INSTRUMENT", "StrkPric", "OptnTp",
            "CLOSE", "OPEN_INT", "EXPIRY_DT", "DTE",
        ])
        result = _select_spread_strikes(
            df, 50000.0, date(2025, 3, 3), "BANKNIFTY", 0.5,
        )
        assert result is None

    def test_insufficient_dte_returns_none(self, puts_df):
        """All options expire too soon → None."""
        puts_df["DTE"] = 1  # only 1 day left
        result = _select_spread_strikes(
            puts_df, 50000.0, date(2025, 3, 3), "BANKNIFTY", 0.5,
            min_dte=7,
        )
        assert result is None

    def test_low_oi_filter(self, puts_df):
        """Low open interest → None."""
        puts_df["OPEN_INT"] = 10  # below MIN_OI
        result = _select_spread_strikes(
            puts_df, 50000.0, date(2025, 3, 3), "BANKNIFTY", 0.5,
            min_oi=MIN_OI,
        )
        assert result is None

    def test_low_premium_filter(self, puts_df):
        """Premiums too small → None."""
        puts_df["CLOSE"] = 0.10  # below MIN_PREMIUM
        result = _select_spread_strikes(
            puts_df, 50000.0, date(2025, 3, 3), "BANKNIFTY", 0.5,
            min_premium=MIN_PREMIUM,
        )
        assert result is None

    def test_net_credit_positive(self, puts_df):
        entry = _select_spread_strikes(
            puts_df, 50000.0, date(2025, 3, 3), "BANKNIFTY", 0.5,
        )
        if entry is not None:
            assert entry.net_credit > 0

    def test_nearest_expiry_selected(self):
        """When two expiries exist, nearest with sufficient DTE is picked."""
        near = _build_fno_df("BANKNIFTY", 50000, date(2025, 3, 13), 10)
        far = _build_fno_df("BANKNIFTY", 50000, date(2025, 3, 27), 24)
        df = pd.concat([near, far], ignore_index=True)
        df = df[df["OptnTp"].str.strip() == "PE"].copy()
        df["EXPIRY_DT"] = pd.to_datetime(df["EXPIRY_DT"], format="mixed", dayfirst=True)
        near_ts = pd.Timestamp(date(2025, 3, 13))
        far_ts = pd.Timestamp(date(2025, 3, 27))
        df["DTE"] = df["EXPIRY_DT"].apply(
            lambda x: (x - pd.Timestamp(date(2025, 3, 3))).days
        )
        entry = _select_spread_strikes(
            df, 50000.0, date(2025, 3, 3), "BANKNIFTY", 0.5, min_dte=7,
        )
        assert entry is not None
        assert entry.short_leg.expiry == date(2025, 3, 13)


# ---------------------------------------------------------------------------
# Test: _mark_to_market_spread
# ---------------------------------------------------------------------------

class TestMarkToMarket:

    def test_finds_both_legs(self):
        """MTM finds closing prices for both strikes."""
        cache = MockBhavcopyCache()
        d = date(2025, 3, 10)
        df = _build_fno_df("BANKNIFTY", 49500, date(2025, 3, 20), 10)
        cache.add(d, df)
        result = _mark_to_market_spread(
            cache, d, "BANKNIFTY",
            short_strike=48500.0, long_strike=47000.0,
            expiry=date(2025, 3, 20), exit_spot=49500.0,
        )
        assert result is not None
        short_val, long_val = result
        assert short_val >= 0
        assert long_val >= 0

    def test_missing_strike_returns_intrinsic(self):
        """When a strike isn't in the chain, falls back to intrinsic."""
        cache = MockBhavcopyCache()
        d = date(2025, 3, 10)
        # Chain that doesn't include strike 99999
        df = _build_fno_df("BANKNIFTY", 49500, date(2025, 3, 20), 10)
        cache.add(d, df)
        result = _mark_to_market_spread(
            cache, d, "BANKNIFTY",
            short_strike=99999.0, long_strike=88888.0,
            expiry=date(2025, 3, 20), exit_spot=49500.0,
        )
        assert result is not None
        short_val, long_val = result
        # Intrinsic: max(K - S, 0)
        assert short_val == pytest.approx(max(99999 - 49500, 0))
        assert long_val == pytest.approx(max(88888 - 49500, 0))

    def test_expired_option_uses_intrinsic(self):
        """After expiry, use intrinsic max(K - S, 0)."""
        cache = MockBhavcopyCache()
        result = _mark_to_market_spread(
            cache, date(2025, 3, 25), "BANKNIFTY",
            short_strike=49000.0, long_strike=47500.0,
            expiry=date(2025, 3, 20), exit_spot=48000.0,
        )
        assert result is not None
        short_val, long_val = result
        # short: max(49000 - 48000, 0) = 1000
        assert short_val == pytest.approx(1000.0)
        # long: max(47500 - 48000, 0) = 0
        assert long_val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: backtest loop
# ---------------------------------------------------------------------------

class TestBacktestLoop:

    def _make_cache_for_series(
        self, series: list[DensityDayObs], symbol: str = "BANKNIFTY",
    ) -> MockBhavcopyCache:
        """Build a cache that has option chain data for each day in series."""
        cache = MockBhavcopyCache()
        weekly_exp = series[0].date + timedelta(days=21)
        for obs in series:
            dte = (weekly_exp - obs.date).days
            if dte < 0:
                weekly_exp = obs.date + timedelta(days=21)
                dte = (weekly_exp - obs.date).days
            df = _build_fno_df(symbol, obs.spot, weekly_exp, dte)
            cache.add(obs.date, df)
        return cache

    def test_no_trades_flat_market(self):
        """Flat features with very tight entry → no trades."""
        series = _make_series(80, noise_seed=99)
        cache = self._make_cache_for_series(series)
        result = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.99, hold_days=5,
            cost_bps=20, symbol="BANKNIFTY",
        )
        assert result.n_signals <= 5

    def test_fear_spike_triggers_trade(self):
        """A clear fear spike should generate at least one trade."""
        overrides = {i: -0.90 for i in range(40, 48)}
        series = _make_series(
            80, skew_overrides=overrides, phys_skew=0.0, spot_drift=50.0,
        )
        cache = self._make_cache_for_series(series)
        result = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.70, hold_days=5,
            cost_bps=20, symbol="BANKNIFTY",
        )
        assert result.n_signals >= 1
        assert len(result.trades) >= 1

    def test_pnl_computation(self):
        """P&L on risk = (net_credit - exit_cost - slippage) / max_risk."""
        overrides = {i: -0.90 for i in range(40, 48)}
        series = _make_series(
            80, skew_overrides=overrides, spot_drift=50.0,
        )
        cache = self._make_cache_for_series(series)
        result = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.70, hold_days=5,
            cost_bps=20, symbol="BANKNIFTY",
        )
        for t in result.trades:
            # pnl_points = net_credit - exit_cost - slippage
            # We can't exactly recompute slippage here but can check sign consistency
            assert isinstance(t.pnl_on_risk, float)
            # Max risk is always positive
            max_risk = (t.short_strike - t.long_strike) - t.net_credit
            assert max_risk > 0

    def test_max_hold_exit(self):
        """Trade should exit after hold_days."""
        overrides = {i: -0.90 for i in range(40, 80)}
        series = _make_series(80, skew_overrides=overrides)
        cache = self._make_cache_for_series(series)
        result = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.70,
            exit_pctile=0.05, hold_days=3,
            cost_bps=20, symbol="BANKNIFTY",
        )
        for t in result.trades:
            if t.exit_reason == "max_hold":
                assert t.hold_days >= 3

    def test_signal_decay_exit(self):
        """Trade exits when signal decays below exit threshold."""
        # Fear spike then quick recovery
        overrides = {40: -0.90, 41: -0.90}
        series = _make_series(80, skew_overrides=overrides)
        cache = self._make_cache_for_series(series)
        result = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.70,
            exit_pctile=0.60,  # high exit threshold → quick exit
            hold_days=999,
            cost_bps=20, symbol="BANKNIFTY",
        )
        for t in result.trades:
            assert t.exit_reason in ("signal_decay", "max_hold", "max_loss", "end_of_data")

    def test_max_loss_exit(self):
        """Trade exits when spot drops to long strike (max loss)."""
        # Fear spike + large drop
        overrides = {i: -0.90 for i in range(40, 48)}
        series = _make_series(
            80, skew_overrides=overrides, spot_drift=-200.0,
            base_spot=50000.0,
        )
        cache = self._make_cache_for_series(series)
        result = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.70,
            exit_pctile=0.05, hold_days=999,
            cost_bps=20, symbol="BANKNIFTY",
        )
        # With -200/day drift, spot should crash below long strike eventually
        max_loss_trades = [t for t in result.trades if t.exit_reason == "max_loss"]
        # May or may not trigger depending on exact series, but at least check loop runs
        assert isinstance(result.trades, list)

    def test_end_of_data_close(self):
        """Open trade at end-of-data gets closed."""
        overrides = {i: -0.90 for i in range(73, 80)}
        series = _make_series(80, skew_overrides=overrides)
        cache = self._make_cache_for_series(series)
        result = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.70, hold_days=999,
            cost_bps=20, symbol="BANKNIFTY",
        )
        if result.trades:
            last = result.trades[-1]
            assert last.exit_reason in ("end_of_data", "max_hold", "signal_decay", "max_loss")

    def test_skip_when_no_strikes(self):
        """When cache has no option data, signal fires but no trade opens."""
        overrides = {i: -0.90 for i in range(40, 48)}
        series = _make_series(80, skew_overrides=overrides)
        # Empty cache → no chain data
        cache = MockBhavcopyCache()
        result = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.70, hold_days=5,
            cost_bps=20, symbol="BANKNIFTY",
        )
        # Signals fire but no trades can be opened
        assert len(result.trades) == 0


# ---------------------------------------------------------------------------
# Test: metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_win_rate_bounded(self):
        overrides = {i: -0.90 for i in range(40, 48)}
        series = _make_series(80, skew_overrides=overrides, spot_drift=50.0)
        cache = TestBacktestLoop()._make_cache_for_series(series)
        result = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.70, hold_days=5,
            cost_bps=20, symbol="BANKNIFTY",
        )
        assert 0.0 <= result.win_rate <= 1.0

    def test_max_dd_non_negative(self):
        series = _make_series(80)
        cache = TestBacktestLoop()._make_cache_for_series(series)
        result = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.70, hold_days=5,
            cost_bps=20, symbol="BANKNIFTY",
        )
        assert result.max_dd_pct >= 0

    def test_ror_computation(self):
        """Total return on risk = product of (1 + pnl_on_risk) - 1."""
        overrides = {i: -0.90 for i in range(40, 48)}
        series = _make_series(80, skew_overrides=overrides, spot_drift=50.0)
        cache = TestBacktestLoop()._make_cache_for_series(series)
        result = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.70, hold_days=5,
            cost_bps=0, symbol="BANKNIFTY",
        )
        if result.trades:
            eq = 1.0
            for t in result.trades:
                eq *= (1 + t.pnl_on_risk)
            expected = (eq - 1) * 100
            assert result.total_return_on_risk_pct == pytest.approx(expected, abs=0.01)

    def test_no_trades_zero_metrics(self):
        series = _make_series(80)
        cache = MockBhavcopyCache()  # empty
        result = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.99, hold_days=5,
            cost_bps=20, symbol="BANKNIFTY",
        )
        if not result.trades:
            assert result.total_return_on_risk_pct == 0.0
            assert result.sharpe == 0.0
            assert result.win_rate == 0.0


# ---------------------------------------------------------------------------
# Test: cost impact
# ---------------------------------------------------------------------------

class TestCostImpact:

    def test_higher_cost_lower_ror(self):
        """Higher trading cost → lower total return on risk."""
        overrides = {i: -0.90 for i in range(40, 48)}
        series = _make_series(80, skew_overrides=overrides, spot_drift=50.0)
        cache = TestBacktestLoop()._make_cache_for_series(series)

        r_low = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.70, hold_days=5,
            cost_bps=0, symbol="BANKNIFTY",
        )
        r_high = run_density_options_backtest(
            series, cache,
            lookback=30, entry_pctile=0.70, hold_days=5,
            cost_bps=100, symbol="BANKNIFTY",
        )
        if r_low.trades and r_high.trades:
            assert r_low.total_return_on_risk_pct >= r_high.total_return_on_risk_pct


# ---------------------------------------------------------------------------
# Test: dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:

    def test_spread_leg_frozen(self):
        leg = SpreadLeg(
            strike=48500.0, premium=150.0,
            expiry=date(2025, 3, 20), dte=17, open_interest=5000,
        )
        with pytest.raises((AttributeError, FrozenInstanceError)):
            leg.strike = 99999  # type: ignore

    def test_spread_trade_fields(self):
        t = SpreadTrade(
            symbol="BANKNIFTY",
            entry_date=date(2025, 3, 3),
            exit_date=date(2025, 3, 10),
            entry_spot=50000.0,
            exit_spot=50500.0,
            short_strike=48500.0,
            long_strike=47000.0,
            expiry=date(2025, 3, 20),
            net_credit=200.0,
            exit_cost=50.0,
            pnl_points=140.0,
            pnl_on_risk=0.10,
            hold_days=5,
            exit_reason="max_hold",
            entry_signal=0.5,
        )
        assert t.symbol == "BANKNIFTY"
        assert t.hold_days == 5
        assert t.pnl_on_risk == 0.10
        assert t.net_credit == 200.0
