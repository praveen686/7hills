"""Phase 6 — Trade Analytics Service tests.

Tests MFM, MDA, efficiency, exit quality, duration, edge cases.
All tests are self-contained with mocked DuckDB stores.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from engine.services.trade_analytics import TradeAnalyticsService, TradeAnalytics
from engine.state import ClosedTrade


# ---------------------------------------------------------------------------
# Fake DuckDB store
# ---------------------------------------------------------------------------

class FakeStore:
    """Mock MarketDataStore that returns pre-configured DataFrames."""

    def __init__(self, data: dict[str, pd.DataFrame] | None = None):
        self._data = data or {}
        self._call_log: list[tuple[str, list | None]] = []

    def sql(self, query: str, params: list | None = None) -> pd.DataFrame:
        self._call_log.append((query, params))
        for key, df in self._data.items():
            if key in query:
                return df
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trade(
    direction: str = "long",
    entry_price: float = 100.0,
    exit_price: float = 105.0,
    pnl_pct: float = 0.05,
    entry_date: str = "2025-09-01",
    exit_date: str = "2025-09-05",
    instrument_type: str = "FUT",
    strategy_id: str = "s1",
    symbol: str = "NIFTY",
    trade_id: str = "s1:NIFTY:2025-09-01:0",
) -> ClosedTrade:
    return ClosedTrade(
        strategy_id=strategy_id,
        symbol=symbol,
        direction=direction,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        weight=0.1,
        pnl_pct=pnl_pct,
        instrument_type=instrument_type,
        trade_id=trade_id,
    )


def _ohlc_df(highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    """Build an index OHLC DataFrame matching the service's expected columns."""
    n = len(highs)
    return pd.DataFrame({
        "date": [f"2025-09-{i + 1:02d}" for i in range(n)],
        "high": highs,
        "low": lows,
        "close": closes,
    })


def _store_with_ohlc(
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> FakeStore:
    """Return a FakeStore that serves known OHLC for both futures and index queries."""
    df = _ohlc_df(highs, lows, closes)
    return FakeStore(data={
        "nfo_1min": df,
        "nse_index_close": df,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMFMLong:
    """test_mfm_long - long trade, MFM = (max_high - entry) / entry."""

    def test_mfm_long(self):
        store = _store_with_ohlc(
            highs=[102, 105, 103],
            lows=[98, 99, 97],
            closes=[101, 104, 102],
        )
        svc = TradeAnalyticsService(store=store)
        trade = _trade(direction="long", entry_price=100.0, pnl_pct=0.05)
        result = svc.analyze_trade(trade)

        assert result.price_path_available is True
        assert result.mfm == pytest.approx(0.05, abs=1e-9)  # (105 - 100) / 100


class TestMFMShort:
    """test_mfm_short - short trade, MFM = (entry - min_low) / entry."""

    def test_mfm_short(self):
        store = _store_with_ohlc(
            highs=[102, 105, 103],
            lows=[98, 99, 97],
            closes=[101, 104, 102],
        )
        svc = TradeAnalyticsService(store=store)
        trade = _trade(direction="short", entry_price=100.0, pnl_pct=0.03)
        result = svc.analyze_trade(trade)

        assert result.price_path_available is True
        assert result.mfm == pytest.approx(0.03, abs=1e-9)  # (100 - 97) / 100


class TestMDALong:
    """test_mda_long - long trade, MDA = (entry - min_low) / entry."""

    def test_mda_long(self):
        store = _store_with_ohlc(
            highs=[102, 105, 103],
            lows=[98, 99, 97],
            closes=[101, 104, 102],
        )
        svc = TradeAnalyticsService(store=store)
        trade = _trade(direction="long", entry_price=100.0, pnl_pct=0.05)
        result = svc.analyze_trade(trade)

        assert result.mda == pytest.approx(0.03, abs=1e-9)  # (100 - 97) / 100


class TestMDAShort:
    """test_mda_short - short trade, MDA = (max_high - entry) / entry."""

    def test_mda_short(self):
        store = _store_with_ohlc(
            highs=[102, 105, 103],
            lows=[98, 99, 97],
            closes=[101, 104, 102],
        )
        svc = TradeAnalyticsService(store=store)
        trade = _trade(direction="short", entry_price=100.0, pnl_pct=0.03)
        result = svc.analyze_trade(trade)

        assert result.mda == pytest.approx(0.05, abs=1e-9)  # (105 - 100) / 100


class TestEfficiencyNormal:
    """test_efficiency_normal - pnl_pct / mfm for a profitable trade."""

    def test_efficiency_normal(self):
        store = _store_with_ohlc(
            highs=[102, 110, 106],
            lows=[98, 99, 97],
            closes=[101, 108, 104],
        )
        svc = TradeAnalyticsService(store=store)
        # MFM = (110 - 100)/100 = 0.10, pnl_pct = 0.05
        trade = _trade(direction="long", entry_price=100.0, pnl_pct=0.05)
        result = svc.analyze_trade(trade)

        assert result.efficiency == pytest.approx(0.5, abs=1e-9)  # 0.05 / 0.10


class TestEfficiencyZeroMFM:
    """test_efficiency_zero_mfm - mfm=0 -> efficiency=0."""

    def test_efficiency_zero_mfm(self):
        # All highs at or below entry for long -> MFM = max(0, ...) = 0
        store = _store_with_ohlc(
            highs=[100, 100, 100],
            lows=[95, 96, 94],
            closes=[98, 99, 97],
        )
        svc = TradeAnalyticsService(store=store)
        trade = _trade(direction="long", entry_price=100.0, pnl_pct=-0.03)
        result = svc.analyze_trade(trade)

        assert result.mfm == 0.0
        assert result.efficiency == 0.0


class TestEfficiencyNegative:
    """test_efficiency_negative - losing trade gives negative efficiency."""

    def test_efficiency_negative(self):
        store = _store_with_ohlc(
            highs=[102, 105, 103],
            lows=[98, 99, 97],
            closes=[101, 104, 100],
        )
        svc = TradeAnalyticsService(store=store)
        # pnl_pct negative, mfm positive -> negative efficiency
        trade = _trade(direction="long", entry_price=100.0, pnl_pct=-0.02)
        result = svc.analyze_trade(trade)

        assert result.mfm == pytest.approx(0.05, abs=1e-9)
        assert result.efficiency == pytest.approx(-0.02 / 0.05, abs=1e-9)
        assert result.efficiency < 0


class TestExitQualityClamped:
    """test_exit_quality_clamped - exit_quality between 0 and 1."""

    def test_exit_quality_clamped(self):
        store = _store_with_ohlc(
            highs=[102, 110, 103],
            lows=[98, 99, 97],
            closes=[101, 108, 102],
        )
        svc = TradeAnalyticsService(store=store)
        # Long: optimal_exit=110, actual exit=105, mfm=(110-100)/100=0.10
        # exit_quality = 1 - |110 - 105| / (0.10 * 100) = 1 - 5/10 = 0.5
        trade = _trade(
            direction="long", entry_price=100.0, exit_price=105.0, pnl_pct=0.05,
        )
        result = svc.analyze_trade(trade)

        assert 0.0 <= result.exit_quality <= 1.0
        assert result.exit_quality == pytest.approx(0.5, abs=1e-9)


class TestDurationDays:
    """test_duration_days - (exit_date - entry_date).days."""

    def test_duration_days(self):
        store = FakeStore()  # No price data needed for duration
        svc = TradeAnalyticsService(store=store)
        trade = _trade(entry_date="2025-09-01", exit_date="2025-09-08")
        result = svc.analyze_trade(trade)

        assert result.duration_days == 7


class TestNoPriceData:
    """test_no_price_data - empty store -> price_path_available=False, metrics=0."""

    def test_no_price_data(self):
        store = FakeStore()  # Returns empty DataFrames for all queries
        svc = TradeAnalyticsService(store=store)
        trade = _trade()
        result = svc.analyze_trade(trade)

        assert result.price_path_available is False
        assert result.mfm == 0.0
        assert result.mda == 0.0
        assert result.efficiency == 0.0
        assert result.exit_quality == 0.0


class TestEntryPriceZero:
    """test_entry_price_zero - entry_price=0 -> skip, all metrics 0."""

    def test_entry_price_zero(self):
        store = _store_with_ohlc(
            highs=[102, 105, 103],
            lows=[98, 99, 97],
            closes=[101, 104, 102],
        )
        svc = TradeAnalyticsService(store=store)
        trade = _trade(entry_price=0.0, pnl_pct=0.0)
        result = svc.analyze_trade(trade)

        assert result.mfm == 0.0
        assert result.mda == 0.0
        assert result.efficiency == 0.0
        assert result.price_path_available is False


class TestMultiDayTrade:
    """test_multi_day_trade - 5-day trade with varying prices."""

    def test_multi_day_trade(self):
        store = _store_with_ohlc(
            highs=[102, 108, 106, 107, 112],
            lows=[98, 99, 95, 96, 100],
            closes=[101, 106, 100, 105, 110],
        )
        svc = TradeAnalyticsService(store=store)
        trade = _trade(
            direction="long",
            entry_price=100.0,
            exit_price=110.0,
            pnl_pct=0.10,
            entry_date="2025-09-01",
            exit_date="2025-09-05",
        )
        result = svc.analyze_trade(trade)

        assert result.duration_days == 4
        assert result.mfm == pytest.approx(0.12, abs=1e-9)  # (112 - 100) / 100
        assert result.mda == pytest.approx(0.05, abs=1e-9)   # (100 - 95) / 100
        assert result.price_path_available is True


class TestSameDayTrade:
    """test_same_day_trade - entry_date == exit_date."""

    def test_same_day_trade(self):
        store = _store_with_ohlc(
            highs=[105],
            lows=[96],
            closes=[102],
        )
        svc = TradeAnalyticsService(store=store)
        trade = _trade(
            entry_date="2025-09-01",
            exit_date="2025-09-01",
            entry_price=100.0,
            exit_price=102.0,
            pnl_pct=0.02,
        )
        result = svc.analyze_trade(trade)

        assert result.duration_days == 0
        assert result.mfm == pytest.approx(0.05, abs=1e-9)  # (105 - 100) / 100


class TestOptionsSpotProxy:
    """test_options_spot_proxy - instrument_type='CE' -> mfm_source='spot_proxy'."""

    def test_options_spot_proxy(self):
        # For CE instruments, the service calls _get_index_ohlc with source="spot_proxy"
        df = _ohlc_df(
            highs=[102, 108, 104],
            lows=[97, 98, 96],
            closes=[101, 106, 102],
        )
        store = FakeStore(data={"nse_index_close": df})
        svc = TradeAnalyticsService(store=store)
        trade = _trade(instrument_type="CE", entry_price=100.0, pnl_pct=0.05)
        result = svc.analyze_trade(trade)

        assert result.mfm_source == "spot_proxy"
        assert result.price_path_available is True


class TestDeterminism:
    """test_determinism - same inputs -> same outputs twice."""

    def test_determinism(self):
        store = _store_with_ohlc(
            highs=[102, 105, 103],
            lows=[98, 99, 97],
            closes=[101, 104, 102],
        )
        svc = TradeAnalyticsService(store=store)
        trade = _trade(direction="long", entry_price=100.0, pnl_pct=0.05)

        r1 = svc.analyze_trade(trade)
        r2 = svc.analyze_trade(trade)

        assert r1.mfm == r2.mfm
        assert r1.mda == r2.mda
        assert r1.efficiency == r2.efficiency
        assert r1.exit_quality == r2.exit_quality
        assert r1.duration_days == r2.duration_days
        assert r1.optimal_exit_price == r2.optimal_exit_price
        assert r1.worst_price == r2.worst_price
        assert r1.to_dict() == r2.to_dict()
