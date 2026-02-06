"""Persistent state for the India scanner paper trader.

Positions and portfolio state are saved to a JSON file so the engine
survives restarts without losing track of open positions.

Same atomic-write pattern as apps/funding_paper/state.py.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """A single paper trading position."""

    symbol: str
    direction: str              # "long" or "short"
    entry_date: str             # YYYY-MM-DD
    entry_price: float          # entry price (next-day open)
    composite_score: float      # signal score at entry
    weight: float               # portfolio weight fraction
    hold_days: int              # target hold period
    days_held: int = 0          # days held so far

    @property
    def target_exit_date_reached(self) -> bool:
        return self.days_held >= self.hold_days


@dataclass(frozen=True)
class ClosedPosition:
    """A completed trade."""

    symbol: str
    direction: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    weight: float
    pnl_pct: float              # (exit - entry) / entry, signed for direction
    cost_pct: float             # roundtrip cost as %
    net_pnl_pct: float          # pnl_pct - cost_pct
    hold_days: int


@dataclass(frozen=True)
class PerformanceStats:
    """Summary performance metrics."""

    total_trades: int
    winning_trades: int
    win_rate: float
    avg_pnl_pct: float
    total_return_pct: float
    ann_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    days_running: float


def compute_performance(
    equity_history: list[list],
    closed_trades: list[dict],
) -> PerformanceStats | None:
    """Compute performance from equity curve and closed trades."""
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

    # Sharpe from daily returns (annualize with sqrt(252))
    returns = []
    for i in range(1, len(equities)):
        r = equities[i] / equities[i - 1] - 1
        returns.append(r)

    if len(returns) >= 2:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 1e-10
        sharpe = (mean_r / std_r) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Trade stats
    n_trades = len(closed_trades)
    winners = sum(1 for t in closed_trades if t.get("net_pnl_pct", 0) > 0)
    avg_pnl = (
        sum(t.get("net_pnl_pct", 0) for t in closed_trades) / n_trades
        if n_trades > 0 else 0.0
    )

    return PerformanceStats(
        total_trades=n_trades,
        winning_trades=winners,
        win_rate=winners / n_trades if n_trades > 0 else 0.0,
        avg_pnl_pct=avg_pnl,
        total_return_pct=total_ret * 100,
        ann_return_pct=ann_ret * 100,
        max_drawdown_pct=max_dd * 100,
        sharpe=sharpe,
        days_running=days,
    )


@dataclass
class ScannerState:
    """Full scanner portfolio state."""

    positions: dict[str, PaperPosition] = field(default_factory=dict)
    closed_trades: list[dict] = field(default_factory=list)
    equity: float = 100_000.0       # start with Rs 1 lakh notional
    initial_equity: float = 100_000.0
    total_entries: int = 0
    total_exits: int = 0
    last_scan_date: str = ""
    started_at: str = ""
    equity_history: list[list] = field(default_factory=list)

    def record_equity(self, d: str) -> None:
        """Record equity snapshot."""
        self.equity_history.append([d, self.equity])

    def save(self, path: Path) -> None:
        """Atomic save to JSON."""
        data = {
            "positions": {sym: asdict(pos) for sym, pos in self.positions.items()},
            "closed_trades": self.closed_trades,
            "equity": self.equity,
            "initial_equity": self.initial_equity,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "last_scan_date": self.last_scan_date,
            "started_at": self.started_at,
            "equity_history": self.equity_history,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(path)
        logger.debug("State saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> ScannerState:
        """Load state from JSON, or return fresh state."""
        if not path.exists():
            now = datetime.now(timezone.utc).isoformat()
            return cls(started_at=now)

        data = json.loads(path.read_text())
        positions = {}
        for sym, pos_data in data.get("positions", {}).items():
            positions[sym] = PaperPosition(**pos_data)

        return cls(
            positions=positions,
            closed_trades=data.get("closed_trades", []),
            equity=data.get("equity", 100_000.0),
            initial_equity=data.get("initial_equity", 100_000.0),
            total_entries=data.get("total_entries", 0),
            total_exits=data.get("total_exits", 0),
            last_scan_date=data.get("last_scan_date", ""),
            started_at=data.get("started_at", ""),
            equity_history=data.get("equity_history", []),
        )
