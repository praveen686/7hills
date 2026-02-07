"""Typed payload schemas for BRAHMASTRA events.

Each payload is a frozen dataclass with a ``to_dict()`` method that
produces a canonical dict (stable key order, no surprises).

Payloads are the *content* inside an EventEnvelope.  The envelope
carries metadata (ts, seq, run_id); the payload carries domain data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TickPayload:
    """Raw tick from KiteTicker or replay."""

    instrument_token: int
    ltp: float
    volume: int = 0
    oi: int = 0
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Bar1mPayload:
    """1-minute OHLCV bar."""

    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int = 0
    vwap: float = 0.0
    bar_ts: str = ""  # bar open timestamp (ISO)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Strategy Decisions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalPayload:
    """Strategy signal — pre-gate, full context.

    Logs every generated signal including those later blocked by gates.
    Fields aligned with Strategy Contracts (STRATEGY_CONTRACTS_TIER1_V1.md).
    """

    direction: str                    # "long", "short", "flat"
    conviction: float                 # [0, 1]
    instrument_type: str              # "FUT", "CE", "PE", "SPREAD"
    strike: float = 0.0
    expiry: str = ""
    ttl_bars: int = 5
    regime: str = ""                  # regime at signal time
    components: dict = field(default_factory=dict)  # strategy-specific context
    reasoning: str = ""               # human-readable explanation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GateDecisionPayload:
    """Risk gate decision — pass/fail with reasons and thresholds.

    Links to the upstream SignalEvent via signal_seq.
    """

    signal_seq: int                   # seq of the upstream SignalEvent
    gate: str                         # GateResult value
    approved: bool
    adjusted_weight: float
    reason: str = ""
    vpin: float = 0.0                 # VPIN at decision time
    portfolio_dd: float = 0.0         # portfolio DD at decision time
    strategy_dd: float = 0.0          # strategy DD at decision time
    total_exposure: float = 0.0       # total exposure at decision time

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Execution Lifecycle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderPayload:
    """Order lifecycle event.

    Covers: submit, ack, reject, cancel, replace.
    """

    order_id: str
    action: str                       # "submit", "ack", "reject", "cancel", "replace"
    side: str                         # "buy", "sell"
    order_type: str                   # "market", "limit", "sl"
    quantity: int = 0
    price: float = 0.0
    product_type: str = "NRML"
    rejection_reason: str = ""
    broker_order_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FillPayload:
    """Execution fill event.

    Covers: partial fill, full fill.
    """

    order_id: str
    fill_id: str
    side: str                         # "buy", "sell"
    quantity: int
    price: float
    fees: float = 0.0                 # total fees for this fill
    is_partial: bool = False
    broker_fill_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskAlertPayload:
    """Risk alert — breaker transitions, VPIN toxic, DD breach.

    Emitted on state transitions only (not every check).
    """

    alert_type: str                   # "vpin_toxic", "dd_portfolio", "dd_strategy",
                                      # "concentration", "exposure", "breaker_on", "breaker_off"
    previous_state: str = ""
    new_state: str = ""
    threshold: float = 0.0
    current_value: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MissingnessPayload:
    """Data quality gate failure — pre-signal validation.

    Emitted when the DataQualityGate blocks a strategy due to
    insufficient or stale market data.
    """

    check_type: str               # "min_strikes", "min_oi", "tick_staleness", "index_close"
    symbol: str
    detail: str
    severity: str                 # "warning", "block"
    chain_strike_count: int = 0
    min_oi_found: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SnapshotPayload:
    """Periodic portfolio + risk snapshot.

    Emitted at fixed cadence (default 60s) for audit trail.
    """

    equity: float
    peak_equity: float
    portfolio_dd: float
    total_exposure: float
    vpin: float
    position_count: int
    strategy_equity: dict = field(default_factory=dict)
    strategy_dd: dict = field(default_factory=dict)
    active_breakers: list = field(default_factory=list)
    regime: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
