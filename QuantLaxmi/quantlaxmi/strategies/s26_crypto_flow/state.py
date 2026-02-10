"""Persistent state for the CLRS paper trader.

Tracks 4 position pools (carry, residual, cascade, reversion),
portfolio equity, and rolling performance metrics.
Uses atomic JSON persistence (tmp + rename) for crash safety.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from quantlaxmi.strategies.s26_crypto_flow.scanner import FundingMatrixBuilder

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A single CLRS position."""

    symbol: str
    signal_type: str           # carry_A, residual_B, cascade_C, revert_D
    direction: str             # long or short
    entry_time: str            # ISO format
    entry_price: float = 0.0
    notional_weight: float = 0.0
    strength: float = 0.0
    accumulated_pnl: float = 0.0
    accumulated_cost: float = 0.0
    n_settlements: int = 0
    hold_bars: int = 0         # for cascade/reversion signals with TTL
    reason: str = ""

    @property
    def net_pnl(self) -> float:
        return self.accumulated_pnl - self.accumulated_cost


@dataclass(frozen=True)
class PerformanceStats:
    """Rolling performance metrics."""

    days_running: float
    total_return_pct: float
    ann_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    n_snapshots: int
    win_rate: float = 0.0
    total_trades: int = 0


def compute_performance(equity_history: list[list]) -> PerformanceStats | None:
    """Compute performance from equity history."""
    if len(equity_history) < 2:
        return None

    equities = [e[1] for e in equity_history]
    t0 = datetime.fromisoformat(equity_history[0][0])
    t1 = datetime.fromisoformat(equity_history[-1][0])
    days = (t1 - t0).total_seconds() / 86400

    if days < 0.01:
        return None

    total_ret = equities[-1] / equities[0] - 1
    ann_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0.0

    # Max drawdown
    peak = equities[0]
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe from returns
    returns = []
    for i in range(1, len(equities)):
        r = equities[i] / equities[i - 1] - 1
        returns.append(r)

    if len(returns) >= 2:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 1e-10
        # Annualize: assume ~288 scans/day (5-min intervals)
        sharpe = (mean_r / std_r) * math.sqrt(288 * 365)
    else:
        sharpe = 0.0

    return PerformanceStats(
        days_running=days,
        total_return_pct=total_ret * 100,
        ann_return_pct=ann_ret * 100,
        max_drawdown_pct=max_dd * 100,
        sharpe=sharpe,
        n_snapshots=len(equity_history),
    )


@dataclass
class PortfolioState:
    """Full CLRS portfolio state."""

    # Positions by signal type
    carry_positions: dict[str, Position] = field(default_factory=dict)
    residual_positions: dict[str, Position] = field(default_factory=dict)
    cascade_positions: dict[str, Position] = field(default_factory=dict)
    reversion_positions: dict[str, Position] = field(default_factory=dict)

    # Portfolio metrics
    equity: float = 1.0
    total_funding_earned: float = 0.0
    total_costs_paid: float = 0.0
    total_entries: int = 0
    total_exits: int = 0
    total_wins: int = 0

    # Timing
    last_scan_time: str = ""
    started_at: str = ""
    last_settlement_hour: int | None = None

    # History
    equity_history: list[list] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)

    # Funding matrix state for PCA
    funding_matrix_state: dict = field(default_factory=dict)

    _SETTLEMENT_HOURS: frozenset[int] = field(
        default=frozenset({0, 8, 16}), init=False, repr=False,
    )

    @property
    def all_positions(self) -> dict[str, Position]:
        """All positions across all signal types."""
        return {
            **self.carry_positions,
            **self.residual_positions,
            **self.cascade_positions,
            **self.reversion_positions,
        }

    @property
    def n_positions(self) -> int:
        return (len(self.carry_positions) + len(self.residual_positions)
                + len(self.cascade_positions) + len(self.reversion_positions))

    def positions_by_type(self, signal_type: str) -> dict[str, Position]:
        """Get positions dict for a signal type."""
        return {
            "carry_A": self.carry_positions,
            "residual_B": self.residual_positions,
            "cascade_C": self.cascade_positions,
            "revert_D": self.reversion_positions,
        }.get(signal_type, {})

    def check_settlement(self) -> bool:
        """Detect 8h funding settlement."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        if hour in self._SETTLEMENT_HOURS and now.minute < 10:
            if self.last_settlement_hour != hour:
                self.last_settlement_hour = hour
                return True
        if hour not in self._SETTLEMENT_HOURS:
            self.last_settlement_hour = None
        return False

    def record_equity(self, event: str = "scan") -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.equity_history.append([now, self.equity, event])

    def log_trade(self, symbol: str, signal_type: str, action: str,
                  direction: str, reason: str, pnl: float = 0.0) -> None:
        """Log a trade event."""
        self.trade_log.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal_type": signal_type,
            "action": action,
            "direction": direction,
            "reason": reason,
            "pnl": pnl,
            "equity": self.equity,
        })
        # Keep last 500 trades
        if len(self.trade_log) > 500:
            self.trade_log = self.trade_log[-500:]

    def save(self, path: Path) -> None:
        """Atomic save to JSON."""
        data = {
            "carry_positions": {s: asdict(p) for s, p in self.carry_positions.items()},
            "residual_positions": {s: asdict(p) for s, p in self.residual_positions.items()},
            "cascade_positions": {s: asdict(p) for s, p in self.cascade_positions.items()},
            "reversion_positions": {s: asdict(p) for s, p in self.reversion_positions.items()},
            "equity": self.equity,
            "total_funding_earned": self.total_funding_earned,
            "total_costs_paid": self.total_costs_paid,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "total_wins": self.total_wins,
            "last_scan_time": self.last_scan_time,
            "started_at": self.started_at,
            "last_settlement_hour": self.last_settlement_hour,
            "equity_history": self.equity_history,
            "trade_log": self.trade_log,
            "funding_matrix_state": self.funding_matrix_state,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> PortfolioState:
        """Load from JSON, or return fresh state."""
        if not path.exists():
            now = datetime.now(timezone.utc).isoformat()
            return cls(started_at=now)

        data = json.loads(path.read_text())

        def _load_positions(key: str) -> dict[str, Position]:
            return {sym: Position(**pd) for sym, pd in data.get(key, {}).items()}

        return cls(
            carry_positions=_load_positions("carry_positions"),
            residual_positions=_load_positions("residual_positions"),
            cascade_positions=_load_positions("cascade_positions"),
            reversion_positions=_load_positions("reversion_positions"),
            equity=data.get("equity", 1.0),
            total_funding_earned=data.get("total_funding_earned", 0.0),
            total_costs_paid=data.get("total_costs_paid", 0.0),
            total_entries=data.get("total_entries", 0),
            total_exits=data.get("total_exits", 0),
            total_wins=data.get("total_wins", 0),
            last_scan_time=data.get("last_scan_time", ""),
            started_at=data.get("started_at", ""),
            last_settlement_hour=data.get("last_settlement_hour"),
            equity_history=data.get("equity_history", []),
            trade_log=data.get("trade_log", []),
            funding_matrix_state=data.get("funding_matrix_state", {}),
        )
