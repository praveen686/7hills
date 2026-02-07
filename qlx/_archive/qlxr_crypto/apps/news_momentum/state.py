"""State persistence for news momentum paper trader.

Tracks active trades, trade history, and performance metrics.
Uses atomic JSON writes (tmp + rename) for crash safety.
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

DEFAULT_STATE_FILE = Path("data/news_momentum_state.json")


@dataclass
class ActiveTrade:
    """A currently active momentum trade."""

    symbol: str
    direction: str             # "long" or "short"
    entry_time: str            # ISO format
    entry_price: float         # mark price at entry
    ttl_minutes: float         # time to live
    score: float
    confidence: float
    n_headlines: int
    headlines: list[str]

    def age_minutes(self) -> float:
        entry_dt = datetime.fromisoformat(self.entry_time)
        now = datetime.now(timezone.utc)
        return (now - entry_dt).total_seconds() / 60

    def is_expired(self) -> bool:
        return self.age_minutes() >= self.ttl_minutes

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "ttl_minutes": self.ttl_minutes,
            "score": self.score,
            "confidence": self.confidence,
            "n_headlines": self.n_headlines,
            "headlines": self.headlines,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ActiveTrade:
        return cls(**d)


@dataclass
class ClosedTrade:
    """A completed trade with P&L."""

    symbol: str
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl_pct: float            # percentage P&L
    score: float
    reason: str                # "ttl_expired", "signal_flip", etc.

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl_pct": self.pnl_pct,
            "score": self.score,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClosedTrade:
        return cls(**d)


@dataclass
class TradingState:
    """Full state of the news momentum paper trader."""

    active_trades: dict[str, ActiveTrade] = field(default_factory=dict)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    headlines_seen: list[str] = field(default_factory=list)
    total_trades: int = 0
    started_at: str = ""

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()

    def save(self, path: Path = DEFAULT_STATE_FILE) -> None:
        """Atomic save: write tmp + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_trades": {
                sym: t.to_dict() for sym, t in self.active_trades.items()
            },
            "closed_trades": [t.to_dict() for t in self.closed_trades[-200:]],
            "headlines_seen": self.headlines_seen[-500:],
            "total_trades": self.total_trades,
            "started_at": self.started_at,
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_FILE) -> TradingState:
        """Load state from disk, or return fresh state."""
        if not path.exists():
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            state = cls(
                active_trades={
                    sym: ActiveTrade.from_dict(t)
                    for sym, t in data.get("active_trades", {}).items()
                },
                closed_trades=[
                    ClosedTrade.from_dict(t)
                    for t in data.get("closed_trades", [])
                ],
                headlines_seen=data.get("headlines_seen", []),
                total_trades=data.get("total_trades", 0),
                started_at=data.get("started_at", ""),
            )
            logger.info(
                "Loaded state: %d active, %d closed trades",
                len(state.active_trades),
                len(state.closed_trades),
            )
            return state
        except Exception as e:
            logger.warning("Failed to load state: %s, starting fresh", e)
            return cls()

    def record_entry(
        self,
        symbol: str,
        direction: str,
        price: float,
        ttl_minutes: float,
        score: float,
        confidence: float,
        n_headlines: int,
        headlines: list[str],
    ) -> ActiveTrade:
        """Record a new trade entry."""
        trade = ActiveTrade(
            symbol=symbol,
            direction=direction,
            entry_time=datetime.now(timezone.utc).isoformat(),
            entry_price=price,
            ttl_minutes=ttl_minutes,
            score=score,
            confidence=confidence,
            n_headlines=n_headlines,
            headlines=headlines[:3],
        )
        self.active_trades[symbol] = trade
        self.total_trades += 1
        return trade

    def record_exit(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
    ) -> ClosedTrade | None:
        """Record a trade exit and compute P&L."""
        trade = self.active_trades.pop(symbol, None)
        if trade is None:
            return None

        if trade.direction == "long":
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
        else:
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price * 100

        closed = ClosedTrade(
            symbol=symbol,
            direction=trade.direction,
            entry_time=trade.entry_time,
            exit_time=datetime.now(timezone.utc).isoformat(),
            entry_price=trade.entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            score=trade.score,
            reason=reason,
        )
        self.closed_trades.append(closed)
        return closed

    def win_rate(self) -> float:
        """Percentage of profitable closed trades."""
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.pnl_pct > 0)
        return wins / len(self.closed_trades) * 100

    def avg_pnl(self) -> float:
        """Average P&L of closed trades in percent."""
        if not self.closed_trades:
            return 0.0
        return sum(t.pnl_pct for t in self.closed_trades) / len(self.closed_trades)

    def total_pnl(self) -> float:
        """Total cumulative P&L in percent (simple sum)."""
        return sum(t.pnl_pct for t in self.closed_trades)
