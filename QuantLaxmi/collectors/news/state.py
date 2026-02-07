"""Persistent paper trading state for India news sentiment strategy.

Uses atomic JSON writes (tempfile + os.replace) for crash safety.
Tracks active trades, closed trades, equity curve, and performance.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = Path("data/india_news_state.json")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ActiveTrade:
    """An open paper trade."""

    symbol: str
    direction: str       # "long" or "short"
    entry_date: str      # ISO YYYY-MM-DD
    entry_price: float
    score: float         # sentiment score at entry
    confidence: float
    event_type: str
    hold_days: int = 0
    current_price: float | None = None

    def to_dict(self) -> dict:
        d = {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "score": self.score,
            "confidence": self.confidence,
            "event_type": self.event_type,
            "hold_days": self.hold_days,
        }
        if self.current_price is not None:
            d["current_price"] = self.current_price
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ActiveTrade:
        return cls(
            symbol=d["symbol"],
            direction=d["direction"],
            entry_date=d["entry_date"],
            entry_price=float(d["entry_price"]),
            score=float(d["score"]),
            confidence=float(d["confidence"]),
            event_type=d.get("event_type", "general"),
            hold_days=int(d.get("hold_days", 0)),
            current_price=float(d["current_price"]) if d.get("current_price") is not None else None,
        )


@dataclass
class ClosedTrade:
    """A completed paper trade."""

    symbol: str
    direction: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    score: float
    confidence: float
    event_type: str
    pnl_pct: float       # after costs
    hold_days: int
    exit_reason: str      # "ttl", "signal_flip", "manual"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "score": self.score,
            "confidence": self.confidence,
            "event_type": self.event_type,
            "pnl_pct": self.pnl_pct,
            "hold_days": self.hold_days,
            "exit_reason": self.exit_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClosedTrade:
        return cls(
            symbol=d["symbol"],
            direction=d["direction"],
            entry_date=d["entry_date"],
            exit_date=d["exit_date"],
            entry_price=float(d["entry_price"]),
            exit_price=float(d["exit_price"]),
            score=float(d["score"]),
            confidence=float(d["confidence"]),
            event_type=d.get("event_type", "general"),
            pnl_pct=float(d["pnl_pct"]),
            hold_days=int(d["hold_days"]),
            exit_reason=d["exit_reason"],
        )


# ---------------------------------------------------------------------------
# Paper trading state
# ---------------------------------------------------------------------------

@dataclass
class IndiaNewsTradingState:
    """Full paper trading state for India news sentiment strategy."""

    active_trades: list[ActiveTrade] = field(default_factory=list)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    equity: float = 1.0
    started_at: str = ""
    last_scan_date: str = ""

    # Config (persisted for self-describing state)
    max_positions: int = 5
    ttl_days: int = 3
    cost_bps: float = 30.0
    confidence_threshold: float = 0.70
    score_threshold: float = 0.50

    # --- Serialization ---

    def to_dict(self) -> dict:
        return {
            "active_trades": [t.to_dict() for t in self.active_trades],
            "closed_trades": [t.to_dict() for t in self.closed_trades],
            "equity": self.equity,
            "started_at": self.started_at,
            "last_scan_date": self.last_scan_date,
            "config": {
                "max_positions": self.max_positions,
                "ttl_days": self.ttl_days,
                "cost_bps": self.cost_bps,
                "confidence_threshold": self.confidence_threshold,
                "score_threshold": self.score_threshold,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> IndiaNewsTradingState:
        cfg = d.get("config", {})
        return cls(
            active_trades=[ActiveTrade.from_dict(t) for t in d.get("active_trades", [])],
            closed_trades=[ClosedTrade.from_dict(t) for t in d.get("closed_trades", [])],
            equity=float(d.get("equity", 1.0)),
            started_at=d.get("started_at", ""),
            last_scan_date=d.get("last_scan_date", ""),
            max_positions=int(cfg.get("max_positions", 5)),
            ttl_days=int(cfg.get("ttl_days", 3)),
            cost_bps=float(cfg.get("cost_bps", 30.0)),
            confidence_threshold=float(cfg.get("confidence_threshold", 0.70)),
            score_threshold=float(cfg.get("score_threshold", 0.50)),
        )

    def save(self, path: Path = DEFAULT_STATE_FILE) -> None:
        """Atomic save to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_FILE) -> IndiaNewsTradingState:
        """Load from JSON, returning fresh state if file doesn't exist."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load state from %s: %s", path, e)
            return cls()

    # --- Position management ---

    def active_symbols(self) -> set[str]:
        """Return set of symbols with active trades."""
        return {t.symbol for t in self.active_trades}

    def available_slots(self) -> int:
        """How many more positions can we open."""
        return max(0, self.max_positions - len(self.active_trades))

    def enter_trade(
        self,
        symbol: str,
        direction: str,
        price: float,
        date_str: str,
        score: float,
        confidence: float,
        event_type: str,
    ) -> ActiveTrade:
        """Open a new paper trade."""
        if symbol in self.active_symbols():
            raise ValueError(f"Already have active trade in {symbol}")
        if self.available_slots() <= 0:
            raise ValueError("Max positions reached")

        trade = ActiveTrade(
            symbol=symbol,
            direction=direction,
            entry_date=date_str,
            entry_price=price,
            score=score,
            confidence=confidence,
            event_type=event_type,
            hold_days=0,
        )
        self.active_trades.append(trade)
        logger.info("ENTER %s %s @ %.2f (score=%.2f, conf=%.2f, %s)",
                     direction, symbol, price, score, confidence, event_type)
        return trade

    def exit_trade(
        self,
        symbol: str,
        price: float,
        date_str: str,
        reason: str,
    ) -> ClosedTrade:
        """Close an active trade and record it."""
        trade_idx = None
        for i, t in enumerate(self.active_trades):
            if t.symbol == symbol:
                trade_idx = i
                break

        if trade_idx is None:
            raise ValueError(f"No active trade for {symbol}")

        active = self.active_trades.pop(trade_idx)
        cost_frac = self.cost_bps / 10_000

        if active.direction == "long":
            raw_pnl = (price - active.entry_price) / active.entry_price
        else:
            raw_pnl = (active.entry_price - price) / active.entry_price

        pnl_pct = raw_pnl - cost_frac

        closed = ClosedTrade(
            symbol=active.symbol,
            direction=active.direction,
            entry_date=active.entry_date,
            exit_date=date_str,
            entry_price=active.entry_price,
            exit_price=price,
            score=active.score,
            confidence=active.confidence,
            event_type=active.event_type,
            pnl_pct=pnl_pct,
            hold_days=active.hold_days,
            exit_reason=reason,
        )
        self.closed_trades.append(closed)

        # Position sizing: equal weight across max_positions
        position_weight = 1.0 / self.max_positions
        self.equity *= (1 + pnl_pct * position_weight)

        logger.info("EXIT %s %s @ %.2f → %.2f (%.2f%%, %dd, %s)",
                     active.direction, symbol, active.entry_price, price,
                     pnl_pct * 100, active.hold_days, reason)
        return closed

    def increment_hold_days(self) -> None:
        """Increment hold_days for all active trades."""
        for t in self.active_trades:
            t.hold_days += 1

    def expired_trades(self) -> list[ActiveTrade]:
        """Return active trades that have exceeded TTL."""
        return [t for t in self.active_trades if t.hold_days >= self.ttl_days]

    # --- Performance metrics ---

    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.pnl_pct > 0)
        return wins / len(self.closed_trades)

    def total_return_pct(self) -> float:
        return (self.equity - 1.0) * 100

    def avg_pnl_pct(self) -> float:
        if not self.closed_trades:
            return 0.0
        return sum(t.pnl_pct for t in self.closed_trades) / len(self.closed_trades) * 100
