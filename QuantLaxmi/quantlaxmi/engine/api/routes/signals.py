"""Signals routes -- real signals from WAL event logs and orchestrator state.

GET /api/signals          -- all recent signals (WAL event logs + state)
GET /api/signals/today    -- signals from the most recent event log date
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/signals", tags=["signals"])

# WAL event logs are stored by date in data/events/
_EVENTS_DIR = Path(__file__).resolve().parents[3] / "data" / "events"


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

class SignalOut(BaseModel):
    id: str
    timestamp: str
    instrument: str
    symbol: str
    direction: str  # "BUY" | "SELL" | "HOLD"
    strength: float
    strategy_id: str
    strategy_name: str
    price: float
    target: float | None = None
    stop_loss: float | None = None
    components: dict[str, Any] | None = None
    reasoning: str = ""
    regime: str = ""
    approved: bool | None = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_STRATEGY_NAMES: dict[str, str] = {
    "s1_vrp": "VRP-RNDR Density",
    "s2_ramanujan": "Ramanujan Cycles",
    "s3_institutional": "Institutional Flow",
    "s4_iv_mr": "IV Mean Reversion",
    "s5_hawkes": "Hawkes Microstructure",
    "s6_multi_factor": "Multi-Factor ML",
    "s7_regime": "Regime Switch",
    "s8_expiry_theta": "Expiry Theta",
    "s9_momentum": "Cross-Section Momentum",
    "s10_gamma_scalp": "Gamma Scalp",
    "s11_pairs": "Pairs StatArb",
}


def _direction_label(direction: str) -> str:
    """Map internal direction labels to frontend-compatible values."""
    mapping = {
        "long": "BUY",
        "short": "SELL",
        "short_vol": "SELL",
        "flat": "HOLD",
    }
    return mapping.get(direction.lower(), direction.upper())


def _strategy_name(strategy_id: str) -> str:
    """Map strategy_id to human-readable name."""
    return _STRATEGY_NAMES.get(strategy_id, strategy_id)


def _load_wal_signals(events_dir: Path, max_files: int = 30) -> list[dict]:
    """Load signal events from WAL .jsonl files.

    Reads all .jsonl event log files from the events directory,
    parses each line as JSON, and extracts signal-type events.
    Also correlates with subsequent gate_decision events to show
    whether the signal was approved.
    """
    signals: list[dict] = []
    if not events_dir.exists():
        logger.debug("Events directory does not exist: %s", events_dir)
        return signals

    # Read most recent files first
    files = sorted(events_dir.glob("*.jsonl"), reverse=True)[:max_files]
    if not files:
        logger.debug("No .jsonl files found in %s", events_dir)
        return signals

    for path in files:
        try:
            events = _parse_events_file(path)
            signals.extend(events)
        except Exception as exc:
            logger.warning("Failed to parse events file %s: %s", path.name, exc)

    return signals


def _parse_events_file(path: Path) -> list[dict]:
    """Parse a single .jsonl event log file into signal dicts."""
    signals: list[dict] = []
    all_events: list[dict] = []

    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            evt = json.loads(line)
            all_events.append(evt)
        except json.JSONDecodeError:
            continue

    # Build a map of signal_seq -> gate_decision for approval status
    gate_decisions: dict[int, dict] = {}
    for evt in all_events:
        if evt.get("event_type") == "gate_decision":
            payload = evt.get("payload", {})
            sig_seq = payload.get("signal_seq")
            if sig_seq is not None:
                gate_decisions[sig_seq] = payload

    # Extract signal events
    for evt in all_events:
        if evt.get("event_type") != "signal":
            continue

        payload = evt.get("payload", {})
        strategy_id = evt.get("strategy_id", "")
        symbol = evt.get("symbol", "")
        ts = evt.get("ts", "")
        seq = evt.get("seq")

        direction = payload.get("direction", "")
        conviction = float(payload.get("conviction", 0))
        instrument_type = payload.get("instrument_type", "FUT")
        reasoning = payload.get("reasoning", "")
        regime = payload.get("regime", "")
        components = payload.get("components", {})
        strike = payload.get("strike", 0)

        # Look up gate decision for this signal
        gate = gate_decisions.get(seq, {})
        approved = gate.get("approved") if gate else None

        # Derive price from components or order context
        price = 0.0
        if "spot" in components:
            price = float(components["spot"])
        elif strike:
            price = float(strike)

        signals.append({
            "id": f"wal_{strategy_id}_{symbol}_{ts}",
            "timestamp": ts,
            "instrument": instrument_type,
            "symbol": symbol,
            "direction": _direction_label(direction),
            "strength": min(1.0, conviction),
            "strategy_id": strategy_id,
            "strategy_name": _strategy_name(strategy_id),
            "price": price,
            "target": None,
            "stop_loss": None,
            "components": components,
            "reasoning": reasoning,
            "regime": regime,
            "approved": approved,
        })

    return signals


def _signals_from_state(state: Any) -> list[dict]:
    """Extract signals from PortfolioState.

    The orchestrator stores recent signals in state metadata.
    We also synthesize signals from active positions (latest entries).
    """
    signals: list[dict] = []

    # From recent trades (entries within last few scans)
    for trade in reversed(state.closed_trades[-100:]):
        pnl_pct = getattr(trade, "pnl_pct", 0) if hasattr(trade, "pnl_pct") else trade.get("pnl_pct", 0)
        signals.append({
            "id": f"t_{trade.symbol}_{trade.entry_date}",
            "timestamp": trade.entry_date,
            "instrument": trade.instrument_type,
            "symbol": trade.symbol,
            "direction": _direction_label(trade.direction),
            "strength": min(1.0, abs(pnl_pct) / 5.0),
            "strategy_id": trade.strategy_id,
            "strategy_name": _strategy_name(trade.strategy_id),
            "price": trade.entry_price,
            "target": None,
            "stop_loss": None,
            "components": None,
            "reasoning": "",
            "regime": "",
            "approved": True,  # was executed
        })

    # From active positions (latest entries as signals)
    for pos in state.active_positions():
        signals.append({
            "id": f"p_{pos.symbol}_{pos.entry_date}",
            "timestamp": pos.entry_date,
            "instrument": pos.instrument_type,
            "symbol": pos.symbol,
            "direction": _direction_label(pos.direction),
            "strength": min(1.0, abs(pos.weight) * 2),
            "strategy_id": pos.strategy_id,
            "strategy_name": _strategy_name(pos.strategy_id),
            "price": pos.entry_price,
            "target": None,
            "stop_loss": None,
            "components": None,
            "reasoning": "",
            "regime": "",
            "approved": True,  # is active
        })

    return signals


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.get("", response_model=list[SignalOut])
async def get_signals(request: Request) -> list[SignalOut]:
    """Return recent signals from WAL event logs and portfolio state.

    Primary source: WAL event log .jsonl files in data/events/
    Secondary source: PortfolioState closed trades and active positions
    """
    all_signals: list[dict] = []

    # Primary: WAL event logs
    events_dir = _EVENTS_DIR
    # Also check if app.state has a wal_query with a configured base_dir
    if hasattr(request.app.state, "wal_query"):
        wal_svc = request.app.state.wal_query
        if hasattr(wal_svc, "_base_dir"):
            events_dir = wal_svc._base_dir

    wal_signals = _load_wal_signals(events_dir)
    all_signals.extend(wal_signals)

    # Secondary: state-derived signals
    state = request.app.state.engine
    state_signals = _signals_from_state(state)
    all_signals.extend(state_signals)

    # Deduplicate by id, keep first (WAL takes priority)
    seen: set[str] = set()
    unique: list[dict] = []
    for s in all_signals:
        if s["id"] not in seen:
            seen.add(s["id"])
            unique.append(s)

    # Sort by timestamp descending
    unique.sort(key=lambda s: s["timestamp"], reverse=True)
    return [SignalOut(**s) for s in unique[:100]]


@router.get("/today", response_model=list[SignalOut])
async def get_today_signals(request: Request) -> list[SignalOut]:
    """Return signals from the most recent event log date.

    Finds the newest .jsonl file in the events directory and
    returns only signals from that date.
    """
    all_signals: list[dict] = []

    # WAL event logs
    events_dir = _EVENTS_DIR
    if hasattr(request.app.state, "wal_query"):
        wal_svc = request.app.state.wal_query
        if hasattr(wal_svc, "_base_dir"):
            events_dir = wal_svc._base_dir

    wal_signals = _load_wal_signals(events_dir, max_files=1)
    all_signals.extend(wal_signals)

    # State-derived for today
    state = request.app.state.engine
    state_signals = _signals_from_state(state)

    # Filter state signals to last scan date
    last_date = state.last_scan_date
    if last_date:
        state_signals = [s for s in state_signals if s["timestamp"] == last_date]
    all_signals.extend(state_signals)

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict] = []
    for s in all_signals:
        if s["id"] not in seen:
            seen.add(s["id"])
            unique.append(s)

    unique.sort(key=lambda s: s["timestamp"], reverse=True)
    return [SignalOut(**s) for s in unique[:50]]
