"""Tests for the funding paper trading app.

Covers: state persistence, strategy signals, settlement detection,
equity tracking, funding smoothing, and annualization utility.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from apps.funding_paper.scanner import (
    FundingSnapshot,
    annualize_funding,
    SETTLEMENTS_PER_YEAR,
)
from apps.funding_paper.state import (
    PortfolioState,
    Position,
    compute_performance,
)
from apps.funding_paper.strategy import (
    StrategyConfig,
    execute_signals,
    generate_signals,
    record_funding_settlement,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _snap(symbol: str, rate: float = 0.0001, volume: float = 100e6) -> FundingSnapshot:
    """Helper to build a FundingSnapshot."""
    return FundingSnapshot(
        symbol=symbol,
        funding_rate=rate,
        ann_funding_pct=annualize_funding(rate),
        mark_price=100.0,
        index_price=100.0,
        next_funding_time_ms=0,
        time_to_funding_min=60.0,
        volume_24h_usd=volume,
    )


@pytest.fixture
def config() -> StrategyConfig:
    return StrategyConfig(
        entry_threshold_pct=20.0,
        exit_threshold_pct=3.0,
        max_positions=3,
        cost_per_leg_bps=10.0,
        min_volume_usd=50e6,
    )


@pytest.fixture
def fresh_state() -> PortfolioState:
    return PortfolioState(started_at="2025-01-01T00:00:00+00:00")


# ---------------------------------------------------------------------------
# Annualization
# ---------------------------------------------------------------------------


class TestAnnualization:
    def test_annualize_funding_formula(self):
        # 0.01% per 8h → 10.95% annualized
        assert annualize_funding(0.0001) == pytest.approx(0.0001 * 1095 * 100)

    def test_settlements_per_year(self):
        assert SETTLEMENTS_PER_YEAR == 3 * 365

    def test_annualize_negative(self):
        assert annualize_funding(-0.0005) < 0

    def test_annualize_zero(self):
        assert annualize_funding(0.0) == 0.0


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_save_and_load(self, tmp_path: Path, fresh_state: PortfolioState):
        state = fresh_state
        state.equity = 1.05
        state.total_entries = 3
        state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            entry_time="2025-01-01T00:00:00+00:00",
            entry_ann_funding=25.0,
            notional_weight=0.5,
            accumulated_funding=0.002,
            accumulated_cost=0.001,
            n_settlements=5,
        )

        path = tmp_path / "state.json"
        state.save(path)

        loaded = PortfolioState.load(path)
        assert loaded.equity == pytest.approx(1.05)
        assert loaded.total_entries == 3
        assert "BTCUSDT" in loaded.positions
        assert loaded.positions["BTCUSDT"].n_settlements == 5
        assert loaded.positions["BTCUSDT"].net_pnl == pytest.approx(0.001)

    def test_load_missing_file(self, tmp_path: Path):
        path = tmp_path / "nonexistent.json"
        state = PortfolioState.load(path)
        assert state.equity == 1.0
        assert len(state.positions) == 0
        assert state.started_at  # should be set

    def test_atomic_write(self, tmp_path: Path, fresh_state: PortfolioState):
        path = tmp_path / "state.json"
        fresh_state.save(path)
        # tmp file should not remain
        assert not path.with_suffix(".tmp").exists()
        assert path.exists()

    def test_equity_history_persists(self, tmp_path: Path, fresh_state: PortfolioState):
        state = fresh_state
        state.record_equity("trade")
        state.equity = 1.01
        state.record_equity("settlement")

        path = tmp_path / "state.json"
        state.save(path)
        loaded = PortfolioState.load(path)
        assert len(loaded.equity_history) == 2
        assert loaded.equity_history[0][2] == "trade"
        assert loaded.equity_history[1][2] == "settlement"

    def test_funding_rate_history_persists(self, tmp_path: Path, fresh_state: PortfolioState):
        state = fresh_state
        state.update_funding_rates({"BTCUSDT": 0.0001, "ETHUSDT": 0.0002})
        path = tmp_path / "state.json"
        state.save(path)

        loaded = PortfolioState.load(path)
        assert loaded.funding_rate_history["BTCUSDT"] == [0.0001]
        assert loaded.funding_rate_history["ETHUSDT"] == [0.0002]

    def test_last_settlement_hour_persists(self, tmp_path: Path, fresh_state: PortfolioState):
        state = fresh_state
        state.last_settlement_hour = 8
        path = tmp_path / "state.json"
        state.save(path)

        loaded = PortfolioState.load(path)
        assert loaded.last_settlement_hour == 8


# ---------------------------------------------------------------------------
# Settlement detection
# ---------------------------------------------------------------------------


class TestSettlementDetection:
    def _mock_utc(self, hour: int, minute: int):
        return datetime(2025, 6, 15, hour, minute, 0, tzinfo=timezone.utc)

    def test_detects_settlement_at_zero(self, fresh_state: PortfolioState):
        with patch("apps.funding_paper.state.datetime") as mock_dt:
            mock_dt.now.return_value = self._mock_utc(0, 5)
            mock_dt.fromisoformat = datetime.fromisoformat
            assert fresh_state.check_settlement() is True
            assert fresh_state.last_settlement_hour == 0

    def test_detects_settlement_at_eight(self, fresh_state: PortfolioState):
        with patch("apps.funding_paper.state.datetime") as mock_dt:
            mock_dt.now.return_value = self._mock_utc(8, 3)
            mock_dt.fromisoformat = datetime.fromisoformat
            assert fresh_state.check_settlement() is True

    def test_no_double_count(self, fresh_state: PortfolioState):
        with patch("apps.funding_paper.state.datetime") as mock_dt:
            mock_dt.now.return_value = self._mock_utc(8, 3)
            mock_dt.fromisoformat = datetime.fromisoformat
            assert fresh_state.check_settlement() is True
            assert fresh_state.check_settlement() is False  # same hour

    def test_no_settlement_at_non_settlement_hour(self, fresh_state: PortfolioState):
        with patch("apps.funding_paper.state.datetime") as mock_dt:
            mock_dt.now.return_value = self._mock_utc(12, 5)
            mock_dt.fromisoformat = datetime.fromisoformat
            assert fresh_state.check_settlement() is False

    def test_no_settlement_after_10_min(self, fresh_state: PortfolioState):
        with patch("apps.funding_paper.state.datetime") as mock_dt:
            mock_dt.now.return_value = self._mock_utc(8, 11)
            mock_dt.fromisoformat = datetime.fromisoformat
            assert fresh_state.check_settlement() is False

    def test_resets_between_settlements(self, fresh_state: PortfolioState):
        with patch("apps.funding_paper.state.datetime") as mock_dt:
            # Settlement at 08:05
            mock_dt.now.return_value = self._mock_utc(8, 5)
            mock_dt.fromisoformat = datetime.fromisoformat
            assert fresh_state.check_settlement() is True
            # Non-settlement hour resets
            mock_dt.now.return_value = self._mock_utc(12, 0)
            fresh_state.check_settlement()
            assert fresh_state.last_settlement_hour is None
            # Settlement at 16:03
            mock_dt.now.return_value = self._mock_utc(16, 3)
            assert fresh_state.check_settlement() is True


# ---------------------------------------------------------------------------
# Funding rate smoothing
# ---------------------------------------------------------------------------


class TestFundingSmoothing:
    def test_update_funding_rates(self, fresh_state: PortfolioState):
        fresh_state.update_funding_rates({"BTC": 0.001})
        fresh_state.update_funding_rates({"BTC": 0.002})
        fresh_state.update_funding_rates({"BTC": 0.003})
        assert fresh_state.funding_rate_history["BTC"] == [0.001, 0.002, 0.003]

    def test_smoothing_window_caps_at_3(self, fresh_state: PortfolioState):
        for i in range(5):
            fresh_state.update_funding_rates({"BTC": i * 0.001})
        assert len(fresh_state.funding_rate_history["BTC"]) == 3
        assert fresh_state.funding_rate_history["BTC"] == [0.002, 0.003, 0.004]

    def test_smoothed_ann_funding(self, fresh_state: PortfolioState):
        fresh_state.update_funding_rates({"BTC": 0.001})
        fresh_state.update_funding_rates({"BTC": 0.002})
        fresh_state.update_funding_rates({"BTC": 0.003})
        smoothed = fresh_state.smoothed_ann_funding("BTC")
        expected = annualize_funding(0.002)  # mean of [0.001, 0.002, 0.003]
        assert smoothed == pytest.approx(expected)

    def test_smoothed_unknown_symbol(self, fresh_state: PortfolioState):
        assert fresh_state.smoothed_ann_funding("UNKNOWN") is None


# ---------------------------------------------------------------------------
# Strategy signal generation
# ---------------------------------------------------------------------------


class TestSignalGeneration:
    def test_entry_when_funding_high(self, fresh_state, config):
        snaps = [_snap("BTCUSDT", rate=0.0003)]  # ~32.8% ann
        signals = generate_signals(fresh_state, snaps, config)
        entries = [s for s in signals if s.action == "enter"]
        assert len(entries) == 1
        assert entries[0].symbol == "BTCUSDT"

    def test_no_entry_below_threshold(self, fresh_state, config):
        snaps = [_snap("BTCUSDT", rate=0.00005)]  # ~5.5% ann
        signals = generate_signals(fresh_state, snaps, config)
        entries = [s for s in signals if s.action == "enter"]
        assert len(entries) == 0

    def test_no_entry_low_volume(self, fresh_state, config):
        snaps = [_snap("BTCUSDT", rate=0.0003, volume=10e6)]  # high funding, low vol
        signals = generate_signals(fresh_state, snaps, config)
        entries = [s for s in signals if s.action == "enter"]
        assert len(entries) == 0

    def test_max_positions_respected(self, fresh_state, config):
        # Fill 3 positions (max)
        for sym in ["AAAUSDT", "BBBUSDT", "CCCUSDT"]:
            fresh_state.positions[sym] = Position(
                symbol=sym,
                entry_time="2025-01-01T00:00:00+00:00",
                entry_ann_funding=25.0,
                notional_weight=0.33,
            )
        snaps = [
            _snap("AAAUSDT", rate=0.0003),
            _snap("BBBUSDT", rate=0.0003),
            _snap("CCCUSDT", rate=0.0003),
            _snap("DDDUSDT", rate=0.001),  # would love to enter, but full
        ]
        signals = generate_signals(fresh_state, snaps, config)
        entries = [s for s in signals if s.action == "enter"]
        assert len(entries) == 0

    def test_exit_when_funding_drops(self, fresh_state, config):
        fresh_state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            entry_time="2025-01-01T00:00:00+00:00",
            entry_ann_funding=25.0,
            notional_weight=1.0,
        )
        snaps = [_snap("BTCUSDT", rate=0.00001)]  # ~1.1% ann < 3% exit
        signals = generate_signals(fresh_state, snaps, config)
        exits = [s for s in signals if s.action == "exit"]
        assert len(exits) == 1

    def test_exit_on_delisted_symbol(self, fresh_state, config):
        fresh_state.positions["DEADUSDT"] = Position(
            symbol="DEADUSDT",
            entry_time="2025-01-01T00:00:00+00:00",
            entry_ann_funding=25.0,
            notional_weight=1.0,
        )
        snaps = []  # symbol gone
        signals = generate_signals(fresh_state, snaps, config)
        exits = [s for s in signals if s.action == "exit"]
        assert len(exits) == 1
        assert exits[0].reason == "Symbol delisted from premium index"

    def test_hold_when_funding_between_thresholds(self, fresh_state, config):
        fresh_state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            entry_time="2025-01-01T00:00:00+00:00",
            entry_ann_funding=25.0,
            notional_weight=1.0,
        )
        snaps = [_snap("BTCUSDT", rate=0.0001)]  # ~10.9% — above exit, below entry
        signals = generate_signals(fresh_state, snaps, config)
        holds = [s for s in signals if s.action == "hold"]
        assert len(holds) == 1


# ---------------------------------------------------------------------------
# Signal execution
# ---------------------------------------------------------------------------


class TestSignalExecution:
    def test_enter_deducts_cost(self, fresh_state, config):
        snaps = [_snap("BTCUSDT", rate=0.0003)]
        signals = generate_signals(fresh_state, snaps, config)
        msgs = execute_signals(fresh_state, signals, config)
        assert fresh_state.equity < 1.0  # cost deducted
        assert fresh_state.total_entries == 1
        assert "BTCUSDT" in fresh_state.positions
        assert len(msgs) >= 1

    def test_exit_deducts_cost_and_removes_position(self, fresh_state, config):
        fresh_state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            entry_time="2025-01-01T00:00:00+00:00",
            entry_ann_funding=25.0,
            notional_weight=1.0,
        )
        snaps = [_snap("BTCUSDT", rate=0.00001)]  # exit signal
        signals = generate_signals(fresh_state, snaps, config)
        execute_signals(fresh_state, signals, config)
        assert "BTCUSDT" not in fresh_state.positions
        assert fresh_state.total_exits == 1

    def test_weights_rebalanced_after_entry(self, fresh_state, config):
        snaps = [_snap("AAAUSDT", rate=0.0005), _snap("BBBUSDT", rate=0.0004)]
        signals = generate_signals(fresh_state, snaps, config)
        execute_signals(fresh_state, signals, config)
        # Both should have equal weight
        weights = [p.notional_weight for p in fresh_state.positions.values()]
        assert all(w == pytest.approx(0.5) for w in weights)

    def test_equity_recorded_on_trade(self, fresh_state, config):
        snaps = [_snap("BTCUSDT", rate=0.0003)]
        signals = generate_signals(fresh_state, snaps, config)
        execute_signals(fresh_state, signals, config)
        assert len(fresh_state.equity_history) >= 1
        assert fresh_state.equity_history[-1][2] == "trade"


# ---------------------------------------------------------------------------
# Funding settlement
# ---------------------------------------------------------------------------


class TestFundingSettlement:
    def test_credits_funding_to_positions(self, fresh_state):
        fresh_state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            entry_time="2025-01-01T00:00:00+00:00",
            entry_ann_funding=25.0,
            notional_weight=0.5,
        )
        snaps = [_snap("BTCUSDT", rate=0.0001)]
        msgs = record_funding_settlement(fresh_state, snaps)

        pos = fresh_state.positions["BTCUSDT"]
        expected_funding = 0.0001 * 0.5
        assert pos.accumulated_funding == pytest.approx(expected_funding)
        assert pos.n_settlements == 1
        assert fresh_state.total_funding_earned == pytest.approx(expected_funding)
        assert fresh_state.equity == pytest.approx(1.0 * (1 + expected_funding))
        assert len(msgs) == 1

    def test_missing_symbol_skipped(self, fresh_state):
        fresh_state.positions["DEADUSDT"] = Position(
            symbol="DEADUSDT",
            entry_time="2025-01-01T00:00:00+00:00",
            entry_ann_funding=25.0,
            notional_weight=1.0,
        )
        snaps = []  # no snapshot for DEADUSDT
        msgs = record_funding_settlement(fresh_state, snaps)
        assert len(msgs) == 0
        assert fresh_state.positions["DEADUSDT"].n_settlements == 0

    def test_records_equity_on_settlement(self, fresh_state):
        fresh_state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            entry_time="2025-01-01T00:00:00+00:00",
            entry_ann_funding=25.0,
            notional_weight=1.0,
        )
        snaps = [_snap("BTCUSDT", rate=0.0001)]
        record_funding_settlement(fresh_state, snaps)
        assert len(fresh_state.equity_history) == 1
        assert fresh_state.equity_history[0][2] == "settlement"


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


class TestPerformanceMetrics:
    def test_returns_none_with_insufficient_data(self):
        assert compute_performance([]) is None
        assert compute_performance([["2025-01-01T00:00:00+00:00", 1.0, "s"]]) is None

    def test_basic_metrics(self):
        history = [
            ["2025-01-01T00:00:00+00:00", 1.0, "trade"],
            ["2025-01-01T08:00:00+00:00", 1.001, "settlement"],
            ["2025-01-01T16:00:00+00:00", 1.002, "settlement"],
            ["2025-01-02T00:00:00+00:00", 1.003, "settlement"],
        ]
        perf = compute_performance(history)
        assert perf is not None
        assert perf.total_return_pct == pytest.approx(0.3, abs=0.1)
        assert perf.max_drawdown_pct == 0.0  # monotonically increasing
        assert perf.n_snapshots == 4
        assert perf.sharpe > 0  # positive returns

    def test_drawdown_detection(self):
        history = [
            ["2025-01-01T00:00:00+00:00", 1.0, "trade"],
            ["2025-01-01T08:00:00+00:00", 1.01, "settlement"],
            ["2025-01-01T16:00:00+00:00", 0.99, "settlement"],  # drawdown
            ["2025-01-02T00:00:00+00:00", 1.005, "settlement"],
        ]
        perf = compute_performance(history)
        assert perf is not None
        # Peak was 1.01, trough was 0.99 → DD ≈ 1.98%
        assert perf.max_drawdown_pct == pytest.approx(1.98, abs=0.1)
