"""Brahmastra portfolio state — atomic JSON persistence.

Tracks combined portfolio equity, per-strategy state, positions,
and trade history across all strategies.  Follows the atomic JSON
pattern from paper_state.py: tempfile + os.replace for crash safety.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = Path("data/brahmastra_state.json")


@dataclass
class Position:
    """Active position in the portfolio."""

    strategy_id: str
    symbol: str
    direction: str         # "long" or "short"
    weight: float          # portfolio weight
    instrument_type: str   # "FUT", "CE", "PE", "SPREAD"
    entry_date: str        # ISO date
    entry_price: float
    strike: float = 0.0
    expiry: str = ""
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "weight": self.weight,
            "instrument_type": self.instrument_type,
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "strike": self.strike,
            "expiry": self.expiry,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Position:
        return cls(
            strategy_id=d["strategy_id"],
            symbol=d["symbol"],
            direction=d["direction"],
            weight=float(d["weight"]),
            instrument_type=d.get("instrument_type", "FUT"),
            entry_date=d["entry_date"],
            entry_price=float(d["entry_price"]),
            strike=float(d.get("strike", 0.0)),
            expiry=d.get("expiry", ""),
            current_price=float(d.get("current_price", 0.0)),
            unrealized_pnl=float(d.get("unrealized_pnl", 0.0)),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ClosedTrade:
    """Completed trade record."""

    strategy_id: str
    symbol: str
    direction: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    weight: float
    pnl_pct: float
    instrument_type: str = "FUT"
    exit_reason: str = ""
    trade_id: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "weight": self.weight,
            "pnl_pct": self.pnl_pct,
            "instrument_type": self.instrument_type,
            "exit_reason": self.exit_reason,
            "trade_id": self.trade_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClosedTrade:
        return cls(
            strategy_id=d["strategy_id"],
            symbol=d["symbol"],
            direction=d["direction"],
            entry_date=d["entry_date"],
            exit_date=d["exit_date"],
            entry_price=float(d["entry_price"]),
            exit_price=float(d["exit_price"]),
            weight=float(d["weight"]),
            pnl_pct=float(d["pnl_pct"]),
            instrument_type=d.get("instrument_type", "FUT"),
            exit_reason=d.get("exit_reason", ""),
            trade_id=d.get("trade_id", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class BrahmastraState:
    """Complete Brahmastra portfolio state."""

    # Portfolio tracking
    equity: float = 1.0
    peak_equity: float = 1.0
    cash: float = 1.0

    # Per-strategy equity tracking
    strategy_equity: dict[str, float] = field(default_factory=dict)
    strategy_peaks: dict[str, float] = field(default_factory=dict)

    # Active positions: key = "strategy_id:symbol"
    positions: dict[str, Position] = field(default_factory=dict)

    # Trade history
    closed_trades: list[ClosedTrade] = field(default_factory=list)

    # Scan tracking
    last_scan_date: str = ""
    last_scan_time: str = ""
    scan_count: int = 0

    # Risk state
    circuit_breaker_active: bool = False
    last_vpin: float = 0.0
    last_vix: float = 0.0
    last_regime: str = "normal"

    # Equity history — list of {"date": str, "equity": float} snapshots
    equity_history: list[dict] = field(default_factory=list)
    equity_at_open: float = 1.0

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def position_key(self, strategy_id: str, symbol: str) -> str:
        return f"{strategy_id}:{symbol}"

    def open_position(self, pos: Position) -> None:
        key = self.position_key(pos.strategy_id, pos.symbol)
        self.positions[key] = pos
        self.cash -= pos.weight
        logger.info(
            "OPEN %s %s %s weight=%.4f @ %.2f",
            pos.strategy_id, pos.symbol, pos.direction, pos.weight, pos.entry_price,
        )

    def close_position(
        self,
        strategy_id: str,
        symbol: str,
        exit_date: str,
        exit_price: float,
        exit_reason: str = "",
    ) -> ClosedTrade | None:
        key = self.position_key(strategy_id, symbol)
        pos = self.positions.pop(key, None)
        if pos is None:
            return None

        # Compute P&L
        if pos.direction == "long":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        # Update equity
        equity_change = pos.weight * pnl_pct
        self.equity += equity_change
        self.cash += pos.weight + equity_change
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Update strategy equity
        strat_eq = self.strategy_equity.get(strategy_id, 1.0)
        strat_eq += equity_change
        self.strategy_equity[strategy_id] = strat_eq
        if strat_eq > self.strategy_peaks.get(strategy_id, 1.0):
            self.strategy_peaks[strategy_id] = strat_eq

        # Generate unique trade_id: "{strategy_id}:{symbol}:{entry_date}:{idx}"
        idx = len([t for t in self.closed_trades if t.strategy_id == strategy_id])
        trade_id = f"{strategy_id}:{symbol}:{pos.entry_date}:{idx}"

        trade = ClosedTrade(
            strategy_id=strategy_id,
            symbol=symbol,
            direction=pos.direction,
            entry_date=pos.entry_date,
            exit_date=exit_date,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            weight=pos.weight,
            pnl_pct=pnl_pct,
            instrument_type=pos.instrument_type,
            exit_reason=exit_reason,
            trade_id=trade_id,
            metadata=pos.metadata,
        )
        self.closed_trades.append(trade)

        logger.info(
            "CLOSE %s %s %s pnl=%.2f%% reason=%s",
            strategy_id, symbol, pos.direction, pnl_pct * 100, exit_reason,
        )
        return trade

    def get_position(self, strategy_id: str, symbol: str) -> Position | None:
        key = self.position_key(strategy_id, symbol)
        return self.positions.get(key)

    def active_positions(self) -> list[Position]:
        return list(self.positions.values())

    def positions_by_strategy(self, strategy_id: str) -> list[Position]:
        return [p for p in self.positions.values() if p.strategy_id == strategy_id]

    def archive_stale_positions(
        self, current_date: date | str, max_age_days: int = 30,
    ) -> list[ClosedTrade]:
        """Force-close positions exceeding TTL.

        Parameters
        ----------
        current_date : date or str
            Today's date (ISO format if str).
        max_age_days : int
            Maximum position age before forced closure.

        Returns
        -------
        List of ClosedTrade records for archived positions.
        """
        if isinstance(current_date, str):
            current_date = date.fromisoformat(current_date)

        stale_keys: list[str] = []
        for key, pos in self.positions.items():
            entry = date.fromisoformat(pos.entry_date)
            age = (current_date - entry).days
            if age > max_age_days:
                stale_keys.append(key)

        archived: list[ClosedTrade] = []
        for key in stale_keys:
            pos = self.positions[key]
            # Use entry_price as exit_price (no market data for stale positions)
            exit_price = pos.current_price if pos.current_price > 0 else pos.entry_price
            trade = self.close_position(
                strategy_id=pos.strategy_id,
                symbol=pos.symbol,
                exit_date=current_date.isoformat(),
                exit_price=exit_price,
                exit_reason="ttl_expired",
            )
            if trade:
                archived.append(trade)

        return archived

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def portfolio_dd(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    def strategy_dd(self, strategy_id: str) -> float:
        eq = self.strategy_equity.get(strategy_id, 1.0)
        peak = self.strategy_peaks.get(strategy_id, 1.0)
        if peak <= 0:
            return 0.0
        return (peak - eq) / peak

    @property
    def total_exposure(self) -> float:
        return sum(abs(p.weight) for p in self.positions.values())

    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.pnl_pct > 0)
        return wins / len(self.closed_trades)

    def total_return_pct(self) -> float:
        return (self.equity - 1.0) * 100

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "cash": self.cash,
            "strategy_equity": self.strategy_equity,
            "strategy_peaks": self.strategy_peaks,
            "positions": {k: p.to_dict() for k, p in self.positions.items()},
            "closed_trades": [t.to_dict() for t in self.closed_trades],
            "last_scan_date": self.last_scan_date,
            "last_scan_time": self.last_scan_time,
            "scan_count": self.scan_count,
            "circuit_breaker_active": self.circuit_breaker_active,
            "last_vpin": self.last_vpin,
            "last_vix": self.last_vix,
            "last_regime": self.last_regime,
            "equity_history": self.equity_history,
            "equity_at_open": self.equity_at_open,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BrahmastraState:
        state = cls(
            equity=float(d.get("equity", 1.0)),
            peak_equity=float(d.get("peak_equity", 1.0)),
            cash=float(d.get("cash", 1.0)),
            strategy_equity=d.get("strategy_equity", {}),
            strategy_peaks=d.get("strategy_peaks", {}),
            closed_trades=[ClosedTrade.from_dict(t) for t in d.get("closed_trades", [])],
            last_scan_date=d.get("last_scan_date", ""),
            last_scan_time=d.get("last_scan_time", ""),
            scan_count=int(d.get("scan_count", 0)),
            circuit_breaker_active=bool(d.get("circuit_breaker_active", False)),
            last_vpin=float(d.get("last_vpin", 0.0)),
            last_vix=float(d.get("last_vix", 0.0)),
            last_regime=d.get("last_regime", "normal"),
            equity_history=d.get("equity_history", []),
            equity_at_open=float(d.get("equity_at_open", d.get("equity", 1.0))),
        )
        # Deserialize positions
        for k, p in d.get("positions", {}).items():
            state.positions[k] = Position.from_dict(p)
        return state

    def save(self, path: Path = DEFAULT_STATE_FILE) -> None:
        """Atomic save to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_FILE) -> BrahmastraState:
        """Load from JSON, returning fresh state if file doesn't exist."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load Brahmastra state from %s: %s", path, e)
            return cls()
