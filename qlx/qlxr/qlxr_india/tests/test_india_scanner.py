"""Integration tests for the India scanner — costs, state, scanner, backtest."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from apps.india_scanner.costs import DEFAULT_COSTS, IndiaCostModel
from apps.india_scanner.scanner import format_scan_results, run_daily_scan
from apps.india_scanner.signals import CompositeSignal
from apps.india_scanner.state import (
    ClosedPosition,
    PaperPosition,
    ScannerState,
    compute_performance,
)
from apps.india_scanner.universe import FNO_UNIVERSE, get_fno_symbols, get_lot_size


# ---------------------------------------------------------------------------
# Cost Model
# ---------------------------------------------------------------------------


class TestIndiaCostModel:
    def test_default_roundtrip_bps(self):
        # For Rs 1L trade: roundtrip should be ~25-35 bps
        bps = DEFAULT_COSTS.roundtrip_bps(100_000)
        assert 15 < bps < 50  # reasonable range

    def test_higher_value_lower_brokerage_impact(self):
        # Brokerage is flat Rs 20, so larger trades have less % impact
        bps_small = DEFAULT_COSTS.roundtrip_bps(10_000)
        bps_large = DEFAULT_COSTS.roundtrip_bps(1_000_000)
        assert bps_small > bps_large

    def test_one_way_positive(self):
        frac = DEFAULT_COSTS.delivery_cost_frac(100_000)
        assert frac > 0

    def test_zero_trade_value(self):
        # Should not crash
        frac = DEFAULT_COSTS.delivery_cost_frac(0)
        assert frac >= 0

    def test_custom_model(self):
        model = IndiaCostModel(stt_delivery_rate=0.002, slippage_bps=10)
        assert model.stt_delivery_rate == 0.002
        bps = model.roundtrip_bps(100_000)
        assert bps > DEFAULT_COSTS.roundtrip_bps(100_000)


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


class TestUniverse:
    def test_fno_universe_size(self):
        assert len(FNO_UNIVERSE) >= 100

    def test_get_fno_symbols_excludes_index(self):
        syms = get_fno_symbols(exclude_index=True)
        assert "NIFTY" not in syms
        assert "RELIANCE" in syms

    def test_get_fno_symbols_includes_index(self):
        syms = get_fno_symbols(exclude_index=False)
        assert "NIFTY" in syms

    def test_get_lot_size_known(self):
        assert get_lot_size("RELIANCE") == 250

    def test_get_lot_size_unknown(self):
        assert get_lot_size("UNKNOWNSYM") == 1

    def test_symbols_sorted(self):
        syms = get_fno_symbols()
        assert syms == sorted(syms)


# ---------------------------------------------------------------------------
# State Persistence
# ---------------------------------------------------------------------------


class TestScannerState:
    def test_save_load_roundtrip(self, tmp_path: Path):
        path = tmp_path / "state.json"
        state = ScannerState(
            equity=105_000,
            initial_equity=100_000,
            total_entries=3,
            total_exits=1,
            last_scan_date="2026-02-03",
            started_at="2026-01-01T00:00:00+00:00",
        )
        state.positions["RELIANCE"] = PaperPosition(
            symbol="RELIANCE",
            direction="long",
            entry_date="2026-02-01",
            entry_price=2800.0,
            composite_score=1.5,
            weight=0.2,
            hold_days=3,
            days_held=2,
        )
        state.save(path)

        loaded = ScannerState.load(path)
        assert loaded.equity == 105_000
        assert loaded.total_entries == 3
        assert "RELIANCE" in loaded.positions
        assert loaded.positions["RELIANCE"].direction == "long"
        assert loaded.positions["RELIANCE"].days_held == 2

    def test_load_nonexistent(self, tmp_path: Path):
        path = tmp_path / "nonexistent.json"
        state = ScannerState.load(path)
        assert state.equity == 100_000
        assert state.positions == {}

    def test_record_equity(self):
        state = ScannerState()
        state.record_equity("2026-02-03")
        assert len(state.equity_history) == 1
        assert state.equity_history[0][0] == "2026-02-03"

    def test_atomic_write(self, tmp_path: Path):
        """Verify tmp file is used for atomic write."""
        path = tmp_path / "state.json"
        state = ScannerState()
        state.save(path)
        assert path.exists()
        # tmp file should be cleaned up (renamed to final)
        assert not path.with_suffix(".tmp").exists()


class TestPaperPosition:
    def test_target_exit_not_reached(self):
        pos = PaperPosition(
            symbol="TCS", direction="long", entry_date="2026-02-01",
            entry_price=3500, composite_score=1.0, weight=0.2,
            hold_days=3, days_held=1,
        )
        assert not pos.target_exit_date_reached

    def test_target_exit_reached(self):
        pos = PaperPosition(
            symbol="TCS", direction="long", entry_date="2026-02-01",
            entry_price=3500, composite_score=1.0, weight=0.2,
            hold_days=3, days_held=3,
        )
        assert pos.target_exit_date_reached


# ---------------------------------------------------------------------------
# Performance Stats
# ---------------------------------------------------------------------------


class TestPerformanceStats:
    def test_compute_basic(self):
        equity_history = [
            ["2026-01-01T00:00:00+00:00", 100000],
            ["2026-01-02T00:00:00+00:00", 101000],
            ["2026-01-03T00:00:00+00:00", 102000],
            ["2026-01-04T00:00:00+00:00", 101500],
            ["2026-01-05T00:00:00+00:00", 103000],
        ]
        closed = [
            {"net_pnl_pct": 1.0},
            {"net_pnl_pct": -0.5},
            {"net_pnl_pct": 1.5},
        ]
        perf = compute_performance(equity_history, closed)
        assert perf is not None
        assert perf.total_trades == 3
        assert perf.winning_trades == 2
        assert perf.win_rate == pytest.approx(2 / 3)
        assert perf.total_return_pct > 0
        assert perf.max_drawdown_pct > 0

    def test_too_few_points(self):
        assert compute_performance([["2026-01-01T00:00:00", 100000]], []) is None

    def test_no_trades(self):
        equity_history = [
            ["2026-01-01T00:00:00+00:00", 100000],
            ["2026-01-02T00:00:00+00:00", 100000],
        ]
        perf = compute_performance(equity_history, [])
        assert perf is not None
        assert perf.total_trades == 0
        assert perf.win_rate == 0.0


# ---------------------------------------------------------------------------
# Scanner format
# ---------------------------------------------------------------------------


class TestFormatScanResults:
    def test_format_with_signals(self):
        signals = [
            CompositeSignal(
                symbol="RELIANCE", delivery_score=1.0, oi_score=1.0,
                fii_score=0.5, composite_score=2.05,
                delivery_signal=None, oi_signal=None,
            ),
            CompositeSignal(
                symbol="TCS", delivery_score=-1.0, oi_score=-1.0,
                fii_score=-0.5, composite_score=-2.05,
                delivery_signal=None, oi_signal=None,
            ),
        ]
        text = format_scan_results(signals, date(2026, 2, 3))
        assert "RELIANCE" in text
        assert "TCS" in text
        assert "2026-02-03" in text

    def test_format_empty(self):
        text = format_scan_results([], date(2026, 2, 3))
        assert "No actionable signals" in text
