"""Tests for signal computation — delivery, OI, FII, composite."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from strategies.s9_momentum.signals import (
    CompositeSignal,
    DeliverySignal,
    FIIFlowSignal,
    OISignal,
    _safe_float,
    compute_composite_scores,
    compute_delivery_signals,
    compute_fii_flow_signal,
    compute_oi_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _equity_df(symbols: list[str], opens: list[float], closes: list[float],
               volumes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "SYMBOL": symbols,
        "OPEN": opens,
        "CLOSE": closes,
        "TOTTRDQTY": volumes,
    })


def _delivery_df(symbols: list[str], pcts: list[float],
                 traded: list[float] | None = None) -> pd.DataFrame:
    d = {"SYMBOL": symbols, "DELIVERY_PCT": pcts}
    if traded:
        d["TRADED_QTY"] = traded
    return pd.DataFrame(d)


def _fno_oi_df(symbols: list[str], oi: list[float],
               chg: list[float] | None = None,
               closes: list[float] | None = None) -> pd.DataFrame:
    d = {"SYMBOL": symbols, "OPEN_INT": oi}
    if chg:
        d["CHG_IN_OI"] = chg
    if closes:
        d["CLOSE"] = closes
    return pd.DataFrame(d)


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_normal(self):
        assert _safe_float(42.5) == 42.5

    def test_string_with_commas(self):
        assert _safe_float("1,234,567") == 1234567.0

    def test_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_empty_string(self):
        assert _safe_float("") == 0.0

    def test_dash(self):
        assert _safe_float("-") == 0.0

    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_custom_default(self):
        assert _safe_float("bad", default=-1.0) == -1.0


# ---------------------------------------------------------------------------
# Delivery Signals
# ---------------------------------------------------------------------------


class TestDeliverySignals:
    def _build_history(self, n_days: int = 21):
        """Build n_days of stable history + one spike day."""
        dates = [date(2026, 1, i + 1) for i in range(n_days)]
        target = dates[-1]

        equity_history = {}
        delivery_history = {}

        for i, d in enumerate(dates):
            if d == target:
                # Spike day: 3x delivery, 2x volume, close > open
                equity_history[d] = _equity_df(
                    ["RELIANCE", "TCS"],
                    [2700, 3400],  # open
                    [2850, 3350],  # close — RELIANCE up, TCS down
                    [20_000_000, 10_000_000],  # 2x volume
                )
                delivery_history[d] = _delivery_df(
                    ["RELIANCE", "TCS"],
                    [75.0, 80.0],  # ~3x the 25% average
                )
            else:
                equity_history[d] = _equity_df(
                    ["RELIANCE", "TCS"],
                    [2750, 3450],
                    [2760, 3440],
                    [10_000_000, 5_000_000],
                )
                delivery_history[d] = _delivery_df(
                    ["RELIANCE", "TCS"],
                    [25.0, 26.0],
                )

        return equity_history, delivery_history, target

    def test_accumulation_signal(self):
        eq, dl, target = self._build_history()
        signals = compute_delivery_signals(eq, dl, target)

        assert "RELIANCE" in signals
        rel = signals["RELIANCE"]
        assert rel.score == 1.0  # close > open → accumulation
        assert rel.delivery_ratio > 1.5
        assert rel.volume_ratio > 1.5

    def test_distribution_signal(self):
        eq, dl, target = self._build_history()
        # Make TCS close < open on target day
        eq[target] = _equity_df(
            ["RELIANCE", "TCS"],
            [2700, 3500],   # open
            [2850, 3350],   # close — TCS down
            [20_000_000, 10_000_000],
        )
        signals = compute_delivery_signals(eq, dl, target)

        assert "TCS" in signals
        tcs = signals["TCS"]
        assert tcs.score == -1.0  # close < open → distribution

    def test_no_signal_below_threshold(self):
        """Normal day: no spike → score = 0."""
        dates = [date(2026, 1, i + 1) for i in range(21)]
        target = dates[-1]

        equity_history = {}
        delivery_history = {}
        for d in dates:
            equity_history[d] = _equity_df(["RELIANCE"], [2750], [2760], [10_000_000])
            delivery_history[d] = _delivery_df(["RELIANCE"], [25.0])

        signals = compute_delivery_signals(equity_history, delivery_history, target)
        if "RELIANCE" in signals:
            assert signals["RELIANCE"].score == 0.0

    def test_missing_target_date(self):
        signals = compute_delivery_signals({}, {}, date(2026, 1, 1))
        assert signals == {}

    def test_symbol_filter(self):
        eq, dl, target = self._build_history()
        signals = compute_delivery_signals(eq, dl, target, symbols={"TCS"})
        assert "RELIANCE" not in signals


# ---------------------------------------------------------------------------
# OI Signals
# ---------------------------------------------------------------------------


class TestOISignals:
    def test_long_buildup(self):
        oi_today = _fno_oi_df(["RELIANCE"], [6_000_000])
        oi_prev = _fno_oi_df(["RELIANCE"], [5_000_000])
        eq_today = _equity_df(["RELIANCE"], [2800], [2900], [10_000_000])
        eq_prev = _equity_df(["RELIANCE"], [2750], [2800], [10_000_000])

        signals = compute_oi_signals(oi_today, oi_prev, eq_today, eq_prev)
        assert "RELIANCE" in signals
        assert signals["RELIANCE"].classification == "LONG_BUILDUP"
        assert signals["RELIANCE"].score == 1.0

    def test_short_buildup(self):
        oi_today = _fno_oi_df(["RELIANCE"], [6_000_000])
        oi_prev = _fno_oi_df(["RELIANCE"], [5_000_000])
        eq_today = _equity_df(["RELIANCE"], [2800], [2700], [10_000_000])  # price down
        eq_prev = _equity_df(["RELIANCE"], [2750], [2800], [10_000_000])

        signals = compute_oi_signals(oi_today, oi_prev, eq_today, eq_prev)
        assert signals["RELIANCE"].classification == "SHORT_BUILDUP"
        assert signals["RELIANCE"].score == -1.0

    def test_short_covering(self):
        oi_today = _fno_oi_df(["RELIANCE"], [4_000_000])  # OI down
        oi_prev = _fno_oi_df(["RELIANCE"], [5_000_000])
        eq_today = _equity_df(["RELIANCE"], [2800], [2900], [10_000_000])  # price up
        eq_prev = _equity_df(["RELIANCE"], [2750], [2800], [10_000_000])

        signals = compute_oi_signals(oi_today, oi_prev, eq_today, eq_prev)
        assert signals["RELIANCE"].classification == "SHORT_COVERING"
        assert signals["RELIANCE"].score == 0.5

    def test_long_unwinding(self):
        oi_today = _fno_oi_df(["RELIANCE"], [4_000_000])  # OI down
        oi_prev = _fno_oi_df(["RELIANCE"], [5_000_000])
        eq_today = _equity_df(["RELIANCE"], [2800], [2700], [10_000_000])  # price down
        eq_prev = _equity_df(["RELIANCE"], [2750], [2800], [10_000_000])

        signals = compute_oi_signals(oi_today, oi_prev, eq_today, eq_prev)
        assert signals["RELIANCE"].classification == "LONG_UNWINDING"
        assert signals["RELIANCE"].score == -0.5

    def test_empty_data(self):
        signals = compute_oi_signals(
            pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame(),
        )
        assert signals == {}


# ---------------------------------------------------------------------------
# FII Flow Signal
# ---------------------------------------------------------------------------


class TestFIIFlowSignal:
    def _fii_df(self, net: float) -> pd.DataFrame:
        return pd.DataFrame({
            "category": ["FII/FPI", "DII"],
            "buyValue": [10000.0, 8000.0],
            "sellValue": [10000.0 - net, 8000.0 + net],
            "netValue": [net, -net],
        })

    def test_bullish_flow(self):
        history = {
            date(2026, 2, 2): self._fii_df(500),
            date(2026, 2, 3): self._fii_df(300),
            date(2026, 2, 4): self._fii_df(200),
        }
        sig = compute_fii_flow_signal(history, date(2026, 2, 4))
        assert sig is not None
        assert sig.regime == "bullish"
        assert sig.score == 0.5
        assert sig.cumulative_net_inr_cr == 1000.0

    def test_bearish_flow(self):
        history = {
            date(2026, 2, 2): self._fii_df(-500),
            date(2026, 2, 3): self._fii_df(-300),
            date(2026, 2, 4): self._fii_df(-200),
        }
        sig = compute_fii_flow_signal(history, date(2026, 2, 4))
        assert sig is not None
        assert sig.regime == "bearish"
        assert sig.score == -0.5

    def test_empty_history(self):
        sig = compute_fii_flow_signal({}, date(2026, 2, 4))
        assert sig is None

    def test_partial_history(self):
        history = {date(2026, 2, 4): self._fii_df(100)}
        sig = compute_fii_flow_signal(history, date(2026, 2, 4))
        assert sig is not None
        assert sig.cumulative_net_inr_cr == 100.0


# ---------------------------------------------------------------------------
# Composite Scores
# ---------------------------------------------------------------------------


class TestCompositeScores:
    def test_basic_composite(self):
        del_signals = {
            "RELIANCE": DeliverySignal(
                symbol="RELIANCE", delivery_pct=75, avg_delivery_pct=25,
                delivery_ratio=3.0, volume=20e6, avg_volume=10e6,
                volume_ratio=2.0, close=2850, open_=2700, score=1.0,
            ),
        }
        oi_signals = {
            "RELIANCE": OISignal(
                symbol="RELIANCE", oi=6e6, oi_change=1e6,
                price_change_pct=3.5, classification="LONG_BUILDUP", score=1.0,
            ),
        }
        fii = FIIFlowSignal(cumulative_net_inr_cr=1000, regime="bullish", score=0.5)

        composites = compute_composite_scores(del_signals, oi_signals, fii)
        assert len(composites) == 1
        c = composites[0]
        assert c.symbol == "RELIANCE"
        # 1.0 * 1.0 + 0.8 * 1.0 + 0.5 * 0.5 = 2.05
        assert c.composite_score == pytest.approx(2.05, abs=0.01)

    def test_opposing_signals(self):
        del_signals = {
            "TCS": DeliverySignal(
                symbol="TCS", delivery_pct=75, avg_delivery_pct=25,
                delivery_ratio=3.0, volume=20e6, avg_volume=10e6,
                volume_ratio=2.0, close=3350, open_=3500, score=-1.0,
            ),
        }
        oi_signals = {
            "TCS": OISignal(
                symbol="TCS", oi=6e6, oi_change=1e6,
                price_change_pct=-3.0, classification="SHORT_BUILDUP", score=-1.0,
            ),
        }
        fii = FIIFlowSignal(cumulative_net_inr_cr=-500, regime="bearish", score=-0.5)

        composites = compute_composite_scores(del_signals, oi_signals, fii)
        c = composites[0]
        assert c.composite_score < 0  # bearish

    def test_no_fii_signal(self):
        del_signals = {
            "RELIANCE": DeliverySignal(
                symbol="RELIANCE", delivery_pct=75, avg_delivery_pct=25,
                delivery_ratio=3.0, volume=20e6, avg_volume=10e6,
                volume_ratio=2.0, close=2850, open_=2700, score=1.0,
            ),
        }
        composites = compute_composite_scores(del_signals, {}, None)
        assert len(composites) == 1
        # 1.0 * 1.0 + 0 + 0 = 1.0
        assert composites[0].composite_score == pytest.approx(1.0)

    def test_empty_signals(self):
        composites = compute_composite_scores({}, {}, None)
        assert composites == []

    def test_sorted_by_abs_score(self):
        del_signals = {
            "A": DeliverySignal("A", 75, 25, 3, 20e6, 10e6, 2, 110, 100, 1.0),
            "B": DeliverySignal("B", 75, 25, 3, 20e6, 10e6, 2, 90, 100, -1.0),
        }
        oi_signals = {
            "A": OISignal("A", 6e6, 1e6, 3.5, "LONG_BUILDUP", 1.0),
            # B has no OI signal → lower composite
        }
        composites = compute_composite_scores(del_signals, oi_signals, None)
        # A should be first (higher |score|)
        assert composites[0].symbol == "A"
