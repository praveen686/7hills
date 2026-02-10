"""Phase 7.1 — Missingness handling and DataQualityGate tests.

22 tests across 5 classes:
  TestDataQualityGate          (10) — gate logic, event emission, check independence
  TestGEXMinOIFilter           ( 4) — compute_gex OI filtering
  TestMissingnessEventPayload  ( 3) — serde, enum, WAL persistence
  TestWebSocketStaleness       ( 3) — staleness detection, heartbeat
  TestOrchestratorDQGateIntegration (2) — gate blocks/allows strategy
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlaxmi.engine.services.data_quality import DataQualityGate, DQGateResult, DQCheckResult
from quantlaxmi.engine.live.event_log import EventLogWriter, read_event_log
from quantlaxmi.core.events.types import EventType
from quantlaxmi.core.events.payloads import MissingnessPayload
from quantlaxmi.core.events.envelope import EventEnvelope
from quantlaxmi.strategies.s5_hawkes.analytics import compute_gex, MIN_GEX_OI
from quantlaxmi.engine.api.ws import (
    update_tick_ts,
    check_tick_staleness,
    _last_tick_ts,
    _STALENESS_THRESHOLD_S,
)


# ========================================================================
# TestDataQualityGate — 10 tests
# ========================================================================

class TestDataQualityGate:
    """Core DataQualityGate logic."""

    def test_complete_chain_passes(self, mock_chain_df):
        """22-strike chain with healthy OI and valid index close passes all checks."""
        gate = DataQualityGate()
        result = gate.check("NIFTY", chain_df=mock_chain_df, index_close=21500.0)
        assert result.passed is True
        assert len(result.failures) == 0

    def test_thin_chain_blocks(self, sparse_chain_df):
        """3-strike chain fails the min_strikes check (needs >= 5)."""
        gate = DataQualityGate()
        result = gate.check("NIFTY", chain_df=sparse_chain_df, index_close=21500.0)
        assert result.passed is False
        fail_types = [f.check_type for f in result.failures]
        assert "min_strikes" in fail_types

    def test_zero_oi_blocks(self, zero_oi_chain_df):
        """10-strike chain with zero OI everywhere fails min_oi check."""
        gate = DataQualityGate()
        result = gate.check("NIFTY", chain_df=zero_oi_chain_df, index_close=21500.0)
        assert result.passed is False
        fail_types = [f.check_type for f in result.failures]
        assert "min_oi" in fail_types

    def test_missing_close_blocks(self, mock_chain_df):
        """index_close=None triggers index_close failure."""
        gate = DataQualityGate()
        result = gate.check("NIFTY", chain_df=mock_chain_df, index_close=None)
        assert result.passed is False
        fail_types = [f.check_type for f in result.failures]
        assert "index_close" in fail_types

    def test_zero_close_blocks(self, mock_chain_df):
        """index_close=0.0 triggers index_close failure."""
        gate = DataQualityGate()
        result = gate.check("NIFTY", chain_df=mock_chain_df, index_close=0.0)
        assert result.passed is False
        fail_types = [f.check_type for f in result.failures]
        assert "index_close" in fail_types

    def test_stale_tick_blocks(self, mock_chain_df):
        """Tick 400s ago exceeds 300s threshold — gate fails."""
        gate = DataQualityGate()
        now = datetime.now(timezone.utc)
        stale_ts = now - timedelta(seconds=400)
        result = gate.check(
            "NIFTY",
            chain_df=mock_chain_df,
            index_close=21500.0,
            last_tick_ts=stale_ts,
            now=now,
        )
        assert result.passed is False
        fail_types = [f.check_type for f in result.failures]
        assert "tick_staleness" in fail_types

    def test_fresh_tick_passes(self, mock_chain_df):
        """Tick 100s ago is within 300s threshold — gate passes."""
        gate = DataQualityGate()
        now = datetime.now(timezone.utc)
        fresh_ts = now - timedelta(seconds=100)
        result = gate.check(
            "NIFTY",
            chain_df=mock_chain_df,
            index_close=21500.0,
            last_tick_ts=fresh_ts,
            now=now,
        )
        assert result.passed is True

    def test_no_chain_blocks(self):
        """chain_df=None triggers failure (no chain data)."""
        gate = DataQualityGate()
        result = gate.check("NIFTY", chain_df=None, index_close=21500.0)
        assert result.passed is False
        fail_types = [f.check_type for f in result.failures]
        assert "min_strikes" in fail_types

    def test_missingness_event_emitted(self, event_log, sparse_chain_df):
        """When gate fails with an event_log, MISSINGNESS events are emitted to WAL."""
        gate = DataQualityGate(event_log=event_log)
        result = gate.check("NIFTY", chain_df=sparse_chain_df, index_close=21500.0)
        assert result.passed is False
        # Flush and verify events were written
        event_log.flush()
        assert event_log.event_count > 0

    def test_all_checks_independent(self, mock_chain_df):
        """Each check_type appears independently in the results list."""
        gate = DataQualityGate()
        now = datetime.now(timezone.utc)
        fresh_ts = now - timedelta(seconds=50)
        result = gate.check(
            "NIFTY",
            chain_df=mock_chain_df,
            index_close=21500.0,
            last_tick_ts=fresh_ts,
            now=now,
        )
        check_types = [c.check_type for c in result.checks]
        # When all inputs are provided, we get exactly these 4 check types
        assert "index_close" in check_types
        assert "min_strikes" in check_types
        assert "min_oi" in check_types
        assert "tick_staleness" in check_types
        assert len(check_types) == 4


# ========================================================================
# TestGEXMinOIFilter — 4 tests
# ========================================================================

class TestGEXMinOIFilter:
    """compute_gex OI filter: < 5 valid-OI strikes returns neutral."""

    SPOT = 21250.0
    AS_OF = pd.Timestamp("2025-12-01")

    def test_zero_oi_neutral(self, zero_oi_chain_df):
        """Zero OI everywhere -> all strikes filtered -> neutral GEX."""
        gex = compute_gex(zero_oi_chain_df, self.SPOT, as_of_date=self.AS_OF)
        assert gex.net_gex_cr == 0.0
        assert gex.regime == "neutral"

    def test_low_oi_filtered(self, low_oi_chain_df):
        """All OI < 100 -> all strikes filtered -> neutral GEX."""
        gex = compute_gex(low_oi_chain_df, self.SPOT, as_of_date=self.AS_OF)
        assert gex.net_gex_cr == 0.0
        assert gex.regime == "neutral"

    def test_high_oi_passes(self, high_oi_chain_df):
        """All OI = 5000 -> 10 valid strikes -> non-neutral GEX."""
        gex = compute_gex(high_oi_chain_df, self.SPOT, as_of_date=self.AS_OF)
        assert gex.net_gex_cr != 0.0
        assert gex.regime in ("mean_revert", "momentum")

    def test_mixed_oi(self, mixed_oi_chain_df):
        """3 strikes below threshold, 7 above -> 7 valid >= 5 -> non-zero GEX."""
        gex = compute_gex(mixed_oi_chain_df, self.SPOT, as_of_date=self.AS_OF)
        assert gex.net_gex_cr != 0.0


# ========================================================================
# TestMissingnessEventPayload — 3 tests
# ========================================================================

class TestMissingnessEventPayload:
    """MissingnessPayload serde, enum, and WAL persistence."""

    def test_serde_roundtrip(self):
        """MissingnessPayload.to_dict() contains all fields."""
        payload = MissingnessPayload(
            check_type="min_strikes",
            symbol="NIFTY",
            detail="3 strikes (min 5)",
            severity="block",
            chain_strike_count=3,
            min_oi_found=0,
        )
        d = payload.to_dict()
        assert d["check_type"] == "min_strikes"
        assert d["symbol"] == "NIFTY"
        assert d["detail"] == "3 strikes (min 5)"
        assert d["severity"] == "block"
        assert d["chain_strike_count"] == 3
        assert d["min_oi_found"] == 0

    def test_enum_membership(self):
        """EventType.MISSINGNESS exists and equals 'missingness'."""
        assert EventType.MISSINGNESS.value == "missingness"
        assert EventType.MISSINGNESS == "missingness"

    def test_wal_persistence(self, tmp_path):
        """Emit a MISSINGNESS event via EventLogWriter, read back, verify payload."""
        log = EventLogWriter(
            base_dir=tmp_path / "wal_test",
            run_id="test-miss-wal",
            fsync_policy="none",
        )
        payload = MissingnessPayload(
            check_type="min_oi",
            symbol="BANKNIFTY",
            detail="Max OI 0 (min 100)",
            severity="block",
            chain_strike_count=10,
            min_oi_found=0,
        )
        log.emit(
            event_type=EventType.MISSINGNESS.value,
            source="data_quality_gate",
            payload=payload.to_dict(),
            strategy_id="",
            symbol="BANKNIFTY",
        )
        log.close()

        # Read back
        jsonl_files = sorted((tmp_path / "wal_test").glob("*.jsonl"))
        assert len(jsonl_files) == 1
        events = read_event_log(jsonl_files[0])
        assert len(events) == 1
        env = events[0]
        assert env.event_type == "missingness"
        assert env.payload["check_type"] == "min_oi"
        assert env.payload["symbol"] == "BANKNIFTY"
        assert env.payload["min_oi_found"] == 0


# ========================================================================
# TestWebSocketStaleness — 3 tests
# ========================================================================

class TestWebSocketStaleness:
    """WebSocket tick staleness tracking."""

    def test_staleness_detected(self):
        """Token with tick 400s ago is stale (> 300s threshold)."""
        token = 99990001
        _last_tick_ts[token] = datetime.now(timezone.utc) - timedelta(seconds=400)
        is_stale, staleness = check_tick_staleness(token)
        assert is_stale is True
        assert staleness >= 399.0  # allow small timing variance

    def test_within_threshold(self):
        """Token with tick 100s ago is not stale."""
        token = 99990002
        _last_tick_ts[token] = datetime.now(timezone.utc) - timedelta(seconds=100)
        is_stale, staleness = check_tick_staleness(token)
        assert is_stale is False
        assert 99.0 <= staleness <= 102.0

    def test_heartbeat_updates(self):
        """update_tick_ts records the current time for the token."""
        token = 99990003
        # Ensure no prior entry
        _last_tick_ts.pop(token, None)
        before = datetime.now(timezone.utc)
        update_tick_ts(token)
        after = datetime.now(timezone.utc)
        ts = _last_tick_ts[token]
        assert before <= ts <= after


# ========================================================================
# TestOrchestratorDQGateIntegration — 2 tests
# ========================================================================

class TestOrchestratorDQGateIntegration:
    """Gate blocks/allows strategy based on data quality."""

    def test_gate_blocks_strategy(self, sparse_chain_df):
        """Failed gate result blocks strategy execution."""
        gate = DataQualityGate()
        result = gate.check("NIFTY", chain_df=sparse_chain_df, index_close=21500.0)
        assert result.passed is False
        # Orchestrator would check result.passed before running strategy
        assert len(result.failures) > 0
        blocked = not result.passed
        assert blocked is True

    def test_gate_allows_strategy(self, mock_chain_df):
        """Passed gate result allows strategy execution."""
        gate = DataQualityGate()
        result = gate.check("NIFTY", chain_df=mock_chain_df, index_close=21500.0)
        assert result.passed is True
        assert len(result.failures) == 0
        allowed = result.passed
        assert allowed is True
