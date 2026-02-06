"""Paper trading state persistence for microstructure strategies.

Atomic JSON state (tempfile + rename), same pattern as other paper traders.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

DEFAULT_STATE_FILE = Path("data/micro_paper_state.json")


@dataclass
class PaperPosition:
    """Active futures position."""

    symbol: str
    direction: str           # "long" or "short"
    entry_time: str          # ISO timestamp
    entry_price: float       # Futures price at entry
    entry_spot: float        # Spot at entry
    conviction: float        # Signal conviction at entry
    gex_regime: str          # GEX regime at entry
    reasoning: str           # Signal reasoning at entry
    cost_bps: float = 5.0    # Round-trip cost in bps (index futures)

    def unrealized_pnl(self, current_price: float) -> float:
        """Return unrealized P&L as percentage."""
        if self.direction == "long":
            return (current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - current_price) / self.entry_price * 100

    def age_minutes(self) -> float:
        now = datetime.now(IST)
        entry = datetime.fromisoformat(self.entry_time)
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=IST)
        return (now - entry).total_seconds() / 60


@dataclass
class ClosedTrade:
    """Completed trade."""

    symbol: str
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    entry_spot: float
    exit_spot: float
    pnl_pct: float           # Net P&L after costs
    cost_bps: float
    conviction: float
    gex_regime: str
    exit_reason: str
    reasoning: str


@dataclass
class AnalyticsLog:
    """Log entry for each analytics snapshot."""

    timestamp: str
    symbol: str
    spot: float
    futures: float
    net_gex: float
    gex_regime: str
    gex_flip: float
    pcr_oi: float
    basis_pts: float
    basis_zscore: float
    iv_near: float
    iv_far: float
    iv_slope: float
    oi_flow_score: float
    combined_score: float
    signal_direction: str


@dataclass
class MicroPaperState:
    """Full paper trading state."""

    positions: dict[str, PaperPosition] = field(default_factory=dict)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    analytics_log: list[AnalyticsLog] = field(default_factory=list)
    total_entries: int = 0

    def record_entry(
        self, symbol: str, direction: str, futures_price: float,
        spot: float, conviction: float, gex_regime: str, reasoning: str,
        cost_bps: float = 5.0,
    ) -> PaperPosition:
        now = datetime.now(IST).isoformat()
        pos = PaperPosition(
            symbol=symbol,
            direction=direction,
            entry_time=now,
            entry_price=futures_price,
            entry_spot=spot,
            conviction=conviction,
            gex_regime=gex_regime,
            reasoning=reasoning,
            cost_bps=cost_bps,
        )
        self.positions[symbol] = pos
        self.total_entries += 1
        return pos

    def record_exit(
        self, symbol: str, futures_price: float, spot: float,
        reason: str,
    ) -> ClosedTrade | None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None

        raw_pnl = pos.unrealized_pnl(futures_price)
        net_pnl = raw_pnl - pos.cost_bps / 100  # cost in %

        now = datetime.now(IST).isoformat()
        trade = ClosedTrade(
            symbol=symbol,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=now,
            entry_price=pos.entry_price,
            exit_price=futures_price,
            entry_spot=pos.entry_spot,
            exit_spot=spot,
            pnl_pct=net_pnl,
            cost_bps=pos.cost_bps,
            conviction=pos.conviction,
            gex_regime=pos.gex_regime,
            exit_reason=reason,
            reasoning=pos.reasoning,
        )
        self.closed_trades.append(trade)
        return trade

    def log_analytics(self, entry: AnalyticsLog) -> None:
        self.analytics_log.append(entry)
        # Keep last 500 entries
        if len(self.analytics_log) > 500:
            self.analytics_log = self.analytics_log[-500:]

    def total_pnl(self) -> float:
        return sum(t.pnl_pct for t in self.closed_trades)

    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.pnl_pct > 0)
        return wins / len(self.closed_trades) * 100

    def avg_pnl(self) -> float:
        if not self.closed_trades:
            return 0.0
        return self.total_pnl() / len(self.closed_trades)

    # ----- Persistence -----

    def save(self, path: Path = DEFAULT_STATE_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "closed_trades": [asdict(t) for t in self.closed_trades],
            "analytics_log": [asdict(a) for a in self.analytics_log[-200:]],
            "total_entries": self.total_entries,
        }
        class _Encoder(json.JSONEncoder):
            def default(self, o):
                import numpy as np
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return super().default(o)

        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, cls=_Encoder)
            os.rename(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_FILE) -> MicroPaperState:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            state = cls()
            state.positions = {
                k: PaperPosition(**v) for k, v in data.get("positions", {}).items()
            }
            state.closed_trades = [
                ClosedTrade(**t) for t in data.get("closed_trades", [])
            ]
            state.analytics_log = [
                AnalyticsLog(**a) for a in data.get("analytics_log", [])
            ]
            state.total_entries = data.get("total_entries", 0)
            logger.info(
                "Loaded state: %d positions, %d closed trades",
                len(state.positions), len(state.closed_trades),
            )
            return state
        except Exception as e:
            logger.warning("Failed to load state from %s: %s", path, e)
            return cls()
