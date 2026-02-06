"""Persistent state for the funding paper trader.

Positions and portfolio state are saved to a JSON file so the engine
survives restarts without losing track of open positions.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Number of funding rate observations to keep for smoothing (3 = ~24h at 8h settlement)
SMOOTHING_WINDOW = 3


@dataclass
class Position:
    """A single funding-harvesting position (long spot + short perp)."""

    symbol: str
    entry_time: str                # ISO format
    entry_ann_funding: float       # annualized % at entry
    notional_weight: float         # fraction of portfolio
    accumulated_funding: float = 0.0  # cumulative funding earned
    accumulated_cost: float = 0.0     # cumulative costs paid
    n_settlements: int = 0            # funding settlements while in position

    @property
    def net_pnl(self) -> float:
        return self.accumulated_funding - self.accumulated_cost


@dataclass(frozen=True)
class PerformanceStats:
    """Rolling performance metrics computed from equity history."""

    days_running: float
    total_return_pct: float
    ann_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    n_snapshots: int


def compute_performance(equity_history: list[list]) -> PerformanceStats | None:
    """Compute performance metrics from equity history.

    equity_history: list of [iso_timestamp, equity_value] pairs.
    """
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

    # Sharpe from 8h returns (3 per day → annualize with sqrt(3*365))
    returns = []
    for i in range(1, len(equities)):
        r = equities[i] / equities[i - 1] - 1
        returns.append(r)

    if len(returns) >= 2:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 1e-10
        sharpe = (mean_r / std_r) * math.sqrt(3 * 365)
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
    """Full portfolio state."""

    positions: dict[str, Position] = field(default_factory=dict)
    equity: float = 1.0
    total_funding_earned: float = 0.0
    total_costs_paid: float = 0.0
    total_entries: int = 0
    total_exits: int = 0
    last_scan_time: str = ""
    started_at: str = ""
    equity_history: list[list] = field(default_factory=list)
    # Funding rate history: {symbol: [rate1, rate2, ...]} for smoothing
    # Each rate is the raw 8h funding rate observed at a scan
    funding_rate_history: dict[str, list[float]] = field(default_factory=dict)
    last_settlement_hour: int | None = None

    _SETTLEMENT_HOURS: frozenset[int] = field(
        default=frozenset({0, 8, 16}), init=False, repr=False,
    )

    def check_settlement(self) -> bool:
        """Detect if a funding settlement just occurred.

        Tracks the last settlement hour to avoid double-counting.
        Persisted across restarts via save/load.
        """
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
        """Record current equity with timestamp. Called on settlements and trades."""
        now = datetime.now(timezone.utc).isoformat()
        self.equity_history.append([now, self.equity, event])

    def update_funding_rates(self, rates: dict[str, float]) -> None:
        """Record latest funding rate observations for smoothing.

        rates: {symbol: raw_8h_funding_rate}
        """
        for sym, rate in rates.items():
            hist = self.funding_rate_history.setdefault(sym, [])
            hist.append(rate)
            # Keep only the last SMOOTHING_WINDOW observations
            if len(hist) > SMOOTHING_WINDOW:
                self.funding_rate_history[sym] = hist[-SMOOTHING_WINDOW:]

    def smoothed_ann_funding(self, symbol: str) -> float | None:
        """Get smoothed annualized funding rate for a symbol.

        Returns None if no history available.
        """
        hist = self.funding_rate_history.get(symbol)
        if not hist:
            return None
        from apps.funding_paper.scanner import annualize_funding

        avg_rate = sum(hist) / len(hist)
        return annualize_funding(avg_rate)

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        data = {
            "positions": {sym: asdict(pos) for sym, pos in self.positions.items()},
            "equity": self.equity,
            "total_funding_earned": self.total_funding_earned,
            "total_costs_paid": self.total_costs_paid,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "last_scan_time": self.last_scan_time,
            "started_at": self.started_at,
            "equity_history": self.equity_history,
            "funding_rate_history": self.funding_rate_history,
            "last_settlement_hour": self.last_settlement_hour,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(path)
        logger.debug("State saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> PortfolioState:
        """Load state from JSON file, or return fresh state."""
        if not path.exists():
            now = datetime.now(timezone.utc).isoformat()
            return cls(started_at=now)

        data = json.loads(path.read_text())
        positions = {}
        for sym, pos_data in data.get("positions", {}).items():
            positions[sym] = Position(**pos_data)

        return cls(
            positions=positions,
            equity=data.get("equity", 1.0),
            total_funding_earned=data.get("total_funding_earned", 0.0),
            total_costs_paid=data.get("total_costs_paid", 0.0),
            total_entries=data.get("total_entries", 0),
            total_exits=data.get("total_exits", 0),
            last_scan_time=data.get("last_scan_time", ""),
            started_at=data.get("started_at", ""),
            equity_history=data.get("equity_history", []),
            funding_rate_history=data.get("funding_rate_history", {}),
            last_settlement_hour=data.get("last_settlement_hour"),
        )
