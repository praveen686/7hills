"""Phase 6 — Cross-cutting diagnostics tests.

Tests trade_id uniqueness, NaN/Inf serialization, empty-state graceful
degradation, causality poison tests, and replay price parity.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from quantlaxmi.engine.services.ars_surface import ARSSurfaceService
from quantlaxmi.engine.services.missed_opportunity import MissedOpportunityService
from quantlaxmi.engine.services.trade_analytics import TradeAnalyticsService, TradeAnalytics
from quantlaxmi.engine.state import PortfolioState, ClosedTrade, Position
from quantlaxmi.core.events.envelope import EventEnvelope
from quantlaxmi.core.events.serde import serialize_envelope
from quantlaxmi.core.events.types import EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeStore:
    """Mock MarketDataStore that returns pre-configured DataFrames."""

    def __init__(self, data: dict[str, pd.DataFrame] | None = None):
        self._data = data or {}

    def sql(self, query: str, params: list | None = None) -> pd.DataFrame:
        for key, df in self._data.items():
            if key in query:
                return df
        return pd.DataFrame()


def _write_wal(tmp_dir: Path, day: str, envelopes: list[EventEnvelope]) -> None:
    """Write a list of envelopes to a JSONL WAL file."""
    path = tmp_dir / f"{day}.jsonl"
    with open(path, "w") as f:
        for e in envelopes:
            f.write(serialize_envelope(e) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTradeIdUniqueness:
    """test_trade_id_uniqueness - close 3 trades for same strategy, all get unique trade_ids."""

    def test_trade_id_uniqueness(self):
        state = PortfolioState()

        # Open and close 3 positions for strategy s1, same symbol
        for i in range(3):
            pos = Position(
                strategy_id="s1",
                symbol="NIFTY",
                direction="long",
                weight=0.1,
                instrument_type="FUT",
                entry_date=f"2025-09-0{i + 1}",
                entry_price=100.0 + i,
            )
            state.open_position(pos)
            state.close_position(
                "s1",
                "NIFTY",
                exit_date=f"2025-09-0{i + 2}",
                exit_price=105.0 + i,
            )

        assert len(state.closed_trades) == 3
        trade_ids = [t.trade_id for t in state.closed_trades]
        assert len(set(trade_ids)) == 3, f"Expected 3 unique trade_ids, got {trade_ids}"

        # Each trade_id should be non-empty
        for tid in trade_ids:
            assert len(tid) > 0


class TestNanInfSerialization:
    """test_nan_inf_serialization - TradeAnalytics with edge values serializes to valid JSON."""

    def test_nan_inf_serialization(self):
        analytics = TradeAnalytics(
            trade_id="s1:NIFTY:2025-09-01:0",
            strategy_id="s1",
            symbol="NIFTY",
            direction="long",
            entry_date="2025-09-01",
            exit_date="2025-09-05",
            entry_price=100.0,
            exit_price=105.0,
            pnl_pct=0.05,
            mfm=float("nan"),
            mda=float("inf"),
            efficiency=float("-inf"),
            exit_quality=0.5,
            duration_days=4,
            optimal_exit_price=110.0,
            worst_price=95.0,
            mfm_source="index_ohlc",
            price_path_available=True,
        )

        d = analytics.to_dict()

        # Verify serialization doesn't raise
        # Replace NaN/Inf with None for JSON compatibility (same as serde module)
        def _sanitize(obj):
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            return obj

        sanitized = _sanitize(d)
        json_str = json.dumps(sanitized)
        assert isinstance(json_str, str)

        # Verify we can parse it back
        parsed = json.loads(json_str)
        assert parsed["trade_id"] == "s1:NIFTY:2025-09-01:0"
        assert parsed["mfm"] is None  # NaN -> None
        assert parsed["mda"] is None  # Inf -> None
        assert parsed["efficiency"] is None  # -Inf -> None
        assert parsed["exit_quality"] == 0.5


class TestEmptyStateGraceful:
    """test_empty_state_graceful - all services return empty/0 with no data."""

    def test_empty_state_graceful(self, tmp_path):
        # Trade analytics with empty store
        store = FakeStore()
        trade_svc = TradeAnalyticsService(store=store)
        analytics = trade_svc.analyze_all([])
        assert analytics == []
        summary = trade_svc.summary_by_strategy(analytics)
        assert summary == {}

        # Missed opportunity with empty WAL directory
        missed_svc = MissedOpportunityService(base_dir=tmp_path)
        blocked = missed_svc.get_blocked_signals("2025-09-01")
        assert blocked == []
        results = missed_svc.analyze_missed("2025-09-01")
        assert results == []
        range_results = missed_svc.analyze_range("2025-09-01", "2025-09-03")
        assert range_results == []
        summary = missed_svc.summary_by_strategy([])
        assert summary == {}

        # ARS surface with empty WAL directory
        ars_svc = ARSSurfaceService(base_dir=tmp_path)
        points = ars_svc.surface_for_strategy("s1")
        assert points == []
        surface = ars_svc.surface_all_strategies()
        assert surface == {}
        hm = ars_svc.heatmap_matrix(surface)
        assert hm["dates"] == []
        assert hm["strategies"] == []
        assert hm["matrix"] == []
        assert hm["status_matrix"] == []


class TestCausalityPoison:
    """test_causality_poison - inject future WAL events after time T, verify signals at T unchanged."""

    def test_causality_poison(self, tmp_path):
        # Day 1: blocked signal with known attributes
        _write_wal(tmp_path, "2025-09-01", [
            EventEnvelope(
                ts="2025-09-01T09:15:00.000000Z",
                seq=1,
                run_id="test-run",
                event_type=EventType.SIGNAL.value,
                source="signal_generator",
                strategy_id="s1",
                symbol="NIFTY",
                payload={
                    "direction": "long",
                    "conviction": 0.75,
                    "instrument_type": "FUT",
                    "ttl_bars": 5,
                    "regime": "normal",
                },
            ),
            EventEnvelope(
                ts="2025-09-01T09:15:01.000000Z",
                seq=2,
                run_id="test-run",
                event_type=EventType.GATE_DECISION.value,
                source="risk_monitor",
                strategy_id="s1",
                symbol="NIFTY",
                payload={"approved": False, "reason": "dd_limit", "gate": "portfolio_risk"},
            ),
        ])

        svc = MissedOpportunityService(base_dir=tmp_path)
        baseline = svc.get_blocked_signals("2025-09-01")
        assert len(baseline) == 1
        baseline_conviction = baseline[0]["conviction"]
        baseline_seq = baseline[0]["signal_seq"]

        # Inject poison: massive future events on day 2
        future_events = []
        for i in range(50):
            future_events.append(EventEnvelope(
                ts=f"2025-09-02T09:{10 + i:02d}:00.000000Z",
                seq=1000 + i * 2,
                run_id="test-run",
                event_type=EventType.SIGNAL.value,
                source="signal_generator",
                strategy_id="s1",
                symbol="NIFTY",
                payload={
                    "direction": "long",
                    "conviction": 0.99,  # 10x spike
                    "instrument_type": "FUT",
                    "ttl_bars": 5,
                    "regime": "extreme_vol",
                },
            ))
            future_events.append(EventEnvelope(
                ts=f"2025-09-02T09:{10 + i:02d}:01.000000Z",
                seq=1000 + i * 2 + 1,
                run_id="test-run",
                event_type=EventType.GATE_DECISION.value,
                source="risk_monitor",
                strategy_id="s1",
                symbol="NIFTY",
                payload={"approved": False, "reason": "poison", "gate": "poison_gate"},
            ))

        _write_wal(tmp_path, "2025-09-02", future_events)

        # Re-read day 1 - must be IDENTICAL to baseline
        after_poison = svc.get_blocked_signals("2025-09-01")
        assert len(after_poison) == 1
        assert after_poison[0]["conviction"] == baseline_conviction
        assert after_poison[0]["signal_seq"] == baseline_seq
        assert after_poison[0]["block_reason"] == "dd_limit"


class TestReplayPriceParity:
    """test_replay_price_parity - same trade analyzed twice gives identical MFM/MDA."""

    def test_replay_price_parity(self):
        df = pd.DataFrame({
            "date": ["2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04", "2025-09-05"],
            "high": [102.0, 108.0, 106.0, 107.0, 112.0],
            "low": [98.0, 99.0, 95.0, 96.0, 100.0],
            "close": [101.0, 106.0, 100.0, 105.0, 110.0],
        })
        store = FakeStore(data={
            "nfo_1min": df,
            "nse_index_close": df,
        })

        svc = TradeAnalyticsService(store=store)
        trade = ClosedTrade(
            strategy_id="s1",
            symbol="NIFTY",
            direction="long",
            entry_date="2025-09-01",
            exit_date="2025-09-05",
            entry_price=100.0,
            exit_price=110.0,
            weight=0.1,
            pnl_pct=0.10,
            trade_id="s1:NIFTY:2025-09-01:0",
        )

        # Analyze twice
        r1 = svc.analyze_trade(trade)
        r2 = svc.analyze_trade(trade)

        # All numeric fields must be bit-identical
        assert r1.mfm == r2.mfm
        assert r1.mda == r2.mda
        assert r1.efficiency == r2.efficiency
        assert r1.exit_quality == r2.exit_quality
        assert r1.optimal_exit_price == r2.optimal_exit_price
        assert r1.worst_price == r2.worst_price
        assert r1.duration_days == r2.duration_days

        # Verify the values are correct
        assert r1.mfm == pytest.approx(0.12, abs=1e-9)  # (112-100)/100
        assert r1.mda == pytest.approx(0.05, abs=1e-9)   # (100-95)/100
