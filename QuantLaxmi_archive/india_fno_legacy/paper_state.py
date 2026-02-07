"""Persistent state for IV mean-reversion paper trading.

Supports multiple indices: NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY.
Stores daily IV observations, current positions, closed trades, and equity.
Uses atomic JSON writes (tempfile + os.replace) for crash safety.
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

DEFAULT_STATE_FILE = Path("data/iv_paper_state.json")

# Supported indices (excludes NIFTYNXT50 which has poor backtest results)
TRADEABLE_INDICES = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class IVObservation:
    """Single day's IV observation for an index (serializable)."""
    date: str               # ISO format YYYY-MM-DD
    spot: float
    atm_iv: float           # annualized
    atm_var: float
    forward: float
    sanos_ok: bool
    symbol: str = "NIFTY"   # which index

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "spot": self.spot,
            "atm_iv": self.atm_iv,
            "atm_var": self.atm_var,
            "forward": self.forward,
            "sanos_ok": self.sanos_ok,
            "symbol": self.symbol,
        }

    @classmethod
    def from_dict(cls, d: dict) -> IVObservation:
        return cls(
            date=d["date"],
            spot=float(d["spot"]),
            atm_iv=float(d["atm_iv"]),
            atm_var=float(d["atm_var"]),
            forward=float(d["forward"]),
            sanos_ok=bool(d["sanos_ok"]),
            symbol=d.get("symbol", "NIFTY"),
        )


@dataclass
class PaperPosition:
    """Active index futures position."""
    symbol: str              # NIFTY, BANKNIFTY, etc.
    entry_date: str          # ISO format
    entry_spot: float
    entry_iv: float
    iv_pctile: float
    hold_days: int = 0       # trading days held
    size_weight: float = 1.0 # signal-strength sizing (0.0–1.0 within 1/N allocation)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date,
            "entry_spot": self.entry_spot,
            "entry_iv": self.entry_iv,
            "iv_pctile": self.iv_pctile,
            "hold_days": self.hold_days,
            "size_weight": self.size_weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PaperPosition:
        return cls(
            symbol=d.get("symbol", "NIFTY"),
            entry_date=d["entry_date"],
            entry_spot=float(d["entry_spot"]),
            entry_iv=float(d["entry_iv"]),
            iv_pctile=float(d["iv_pctile"]),
            hold_days=int(d["hold_days"]),
            size_weight=float(d.get("size_weight", 1.0)),
        )


@dataclass
class ClosedTrade:
    """Completed trade record."""
    symbol: str
    entry_date: str
    exit_date: str
    entry_spot: float
    exit_spot: float
    entry_iv: float
    exit_iv: float
    iv_pctile: float
    pnl_pct: float           # after costs
    hold_days: int
    exit_reason: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_spot": self.entry_spot,
            "exit_spot": self.exit_spot,
            "entry_iv": self.entry_iv,
            "exit_iv": self.exit_iv,
            "iv_pctile": self.iv_pctile,
            "pnl_pct": self.pnl_pct,
            "hold_days": self.hold_days,
            "exit_reason": self.exit_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClosedTrade:
        return cls(
            symbol=d.get("symbol", "NIFTY"),
            entry_date=d["entry_date"],
            exit_date=d["exit_date"],
            entry_spot=float(d["entry_spot"]),
            exit_spot=float(d["exit_spot"]),
            entry_iv=float(d["entry_iv"]),
            exit_iv=float(d["exit_iv"]),
            iv_pctile=float(d["iv_pctile"]),
            pnl_pct=float(d["pnl_pct"]),
            hold_days=int(d["hold_days"]),
            exit_reason=d["exit_reason"],
        )


# ---------------------------------------------------------------------------
# Multi-index paper trading state
# ---------------------------------------------------------------------------

@dataclass
class MultiIndexPaperState:
    """Complete multi-index paper trading state."""
    # Per-index IV history: symbol -> list of observations
    iv_histories: dict[str, list[IVObservation]] = field(default_factory=dict)

    # Per-index positions: symbol -> position (or None)
    positions: dict[str, PaperPosition | None] = field(default_factory=dict)

    # All closed trades (across all indices)
    closed_trades: list[ClosedTrade] = field(default_factory=list)

    # Global equity (trades affect this)
    equity: float = 1.0
    started_at: str = ""
    last_scan_date: str = ""

    # Indices being tracked
    symbols: list[str] = field(default_factory=lambda: TRADEABLE_INDICES.copy())

    # Strategy config (persisted so state is self-describing)
    iv_lookback: int = 30
    entry_pctile: float = 0.80
    exit_pctile: float = 0.50
    max_hold_days: int = 5
    cost_bps: float = 5.0

    def __post_init__(self):
        # Ensure all symbols have histories and position slots
        for sym in self.symbols:
            if sym not in self.iv_histories:
                self.iv_histories[sym] = []
            if sym not in self.positions:
                self.positions[sym] = None

    # --- Serialization ---

    def to_dict(self) -> dict:
        return {
            "iv_histories": {
                sym: [o.to_dict() for o in obs_list]
                for sym, obs_list in self.iv_histories.items()
            },
            "positions": {
                sym: pos.to_dict() if pos else None
                for sym, pos in self.positions.items()
            },
            "closed_trades": [t.to_dict() for t in self.closed_trades],
            "equity": self.equity,
            "started_at": self.started_at,
            "last_scan_date": self.last_scan_date,
            "symbols": self.symbols,
            "config": {
                "iv_lookback": self.iv_lookback,
                "entry_pctile": self.entry_pctile,
                "exit_pctile": self.exit_pctile,
                "max_hold_days": self.max_hold_days,
                "cost_bps": self.cost_bps,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> MultiIndexPaperState:
        cfg = d.get("config", {})
        symbols = d.get("symbols", TRADEABLE_INDICES.copy())

        iv_histories = {}
        for sym, obs_list in d.get("iv_histories", {}).items():
            iv_histories[sym] = [IVObservation.from_dict(o) for o in obs_list]

        positions = {}
        for sym, pos_data in d.get("positions", {}).items():
            positions[sym] = PaperPosition.from_dict(pos_data) if pos_data else None

        state = cls(
            iv_histories=iv_histories,
            positions=positions,
            closed_trades=[ClosedTrade.from_dict(t) for t in d.get("closed_trades", [])],
            equity=float(d.get("equity", 1.0)),
            started_at=d.get("started_at", ""),
            last_scan_date=d.get("last_scan_date", ""),
            symbols=symbols,
            iv_lookback=int(cfg.get("iv_lookback", 30)),
            entry_pctile=float(cfg.get("entry_pctile", 0.80)),
            exit_pctile=float(cfg.get("exit_pctile", 0.50)),
            max_hold_days=int(cfg.get("max_hold_days", 5)),
            cost_bps=float(cfg.get("cost_bps", 5.0)),
        )
        return state

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
    def load(cls, path: Path = DEFAULT_STATE_FILE) -> MultiIndexPaperState:
        """Load from JSON, returning fresh state if file doesn't exist."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            # Check if this is a multi-index state or legacy single-index state
            if "iv_histories" in data:
                return cls.from_dict(data)
            else:
                # Migrate from legacy single-index state
                return cls._migrate_from_legacy(data)
        except Exception as e:
            logger.warning("Failed to load state from %s: %s", path, e)
            return cls()

    @classmethod
    def _migrate_from_legacy(cls, d: dict) -> MultiIndexPaperState:
        """Migrate from legacy single-index PaperState format."""
        cfg = d.get("config", {})

        # Convert single iv_history to multi-index format
        iv_histories = {"NIFTY": [IVObservation.from_dict(o) for o in d.get("iv_history", [])]}

        # Convert single position
        positions = {"NIFTY": None}
        if d.get("position"):
            pos_data = d["position"]
            pos_data["symbol"] = "NIFTY"
            positions["NIFTY"] = PaperPosition.from_dict(pos_data)

        # Convert trades (add symbol field)
        closed_trades = []
        for t in d.get("closed_trades", []):
            t["symbol"] = "NIFTY"
            closed_trades.append(ClosedTrade.from_dict(t))

        state = cls(
            iv_histories=iv_histories,
            positions=positions,
            closed_trades=closed_trades,
            equity=float(d.get("equity", 1.0)),
            started_at=d.get("started_at", ""),
            last_scan_date=d.get("last_scan_date", ""),
            symbols=TRADEABLE_INDICES.copy(),
            iv_lookback=int(cfg.get("iv_lookback", 30)),
            entry_pctile=float(cfg.get("entry_pctile", 0.80)),
            exit_pctile=float(cfg.get("exit_pctile", 0.50)),
            max_hold_days=int(cfg.get("max_hold_days", 5)),
            cost_bps=float(cfg.get("cost_bps", 5.0)),
        )
        logger.info("Migrated legacy NIFTY-only state to multi-index format")
        return state

    # --- IV history management ---

    def append_observation(self, obs: IVObservation) -> None:
        """Add a new daily observation for an index. Skips if date already exists."""
        symbol = obs.symbol
        if symbol not in self.iv_histories:
            self.iv_histories[symbol] = []

        history = self.iv_histories[symbol]
        if history and history[-1].date >= obs.date:
            if history[-1].date == obs.date:
                logger.debug("%s: Observation for %s already exists, updating", symbol, obs.date)
                history[-1] = obs
                return
            logger.warning("%s: Observation for %s is before latest %s, skipping",
                           symbol, obs.date, history[-1].date)
            return
        history.append(obs)

    def current_ivs(self, symbol: str) -> list[float]:
        """Return ATM IV values from history for a given index."""
        return [o.atm_iv for o in self.iv_histories.get(symbol, [])]

    def percentile_rank(self, symbol: str, window: int | None = None) -> float:
        """Compute current IV percentile rank within lookback window for an index."""
        history = self.iv_histories.get(symbol, [])
        if not history:
            return 0.0
        w = window or self.iv_lookback
        ivs = [o.atm_iv for o in history]
        lookback = ivs[max(0, len(ivs) - w):]
        current = ivs[-1]
        return sum(1 for x in lookback if x <= current) / len(lookback)

    def have_enough_history(self, symbol: str) -> bool:
        """Check if we have enough IV history for signal generation."""
        return len(self.iv_histories.get(symbol, [])) >= self.iv_lookback

    def latest_observation(self, symbol: str) -> IVObservation | None:
        """Get the most recent observation for an index."""
        history = self.iv_histories.get(symbol, [])
        return history[-1] if history else None

    # --- Position management ---

    @staticmethod
    def compute_size_weight(pctile: float, entry_threshold: float) -> float:
        """Compute signal-strength sizing weight (0.0–1.0).

        Proportional to how extreme the IV spike is above the entry threshold.
        At exactly the entry threshold → 0.25 (minimum size).
        At 100th percentile → 1.0 (full size within 1/N allocation).
        """
        if pctile <= entry_threshold:
            return 0.25
        return min(1.0, 0.25 + 0.75 * (pctile - entry_threshold) / (1.0 - entry_threshold))

    def enter_position(self, symbol: str, obs: IVObservation, pctile: float) -> PaperPosition:
        """Enter a long futures position for an index.

        Position size scales with IV percentile extremity:
        just above entry threshold → 25% of 1/N allocation,
        99th percentile → 100% of 1/N allocation.
        """
        if self.positions.get(symbol) is not None:
            raise ValueError(f"Already in a position for {symbol}")

        size_weight = self.compute_size_weight(pctile, self.entry_pctile)

        pos = PaperPosition(
            symbol=symbol,
            entry_date=obs.date,
            entry_spot=obs.spot,
            entry_iv=obs.atm_iv,
            iv_pctile=pctile,
            hold_days=0,
            size_weight=size_weight,
        )
        self.positions[symbol] = pos
        logger.info("ENTER long %s @ %.1f (IV=%.1f%%, pctile=%.2f, size=%.0f%%)",
                    symbol, obs.spot, obs.atm_iv * 100, pctile, size_weight * 100)
        return pos

    def exit_position(self, symbol: str, obs: IVObservation, reason: str) -> ClosedTrade:
        """Exit the current position for an index and record the trade."""
        pos = self.positions.get(symbol)
        if pos is None:
            raise ValueError(f"No position to exit for {symbol}")

        cost_frac = self.cost_bps / 10_000
        pnl_pct = (obs.spot - pos.entry_spot) / pos.entry_spot - cost_frac

        trade = ClosedTrade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=obs.date,
            entry_spot=pos.entry_spot,
            exit_spot=obs.spot,
            entry_iv=pos.entry_iv,
            exit_iv=obs.atm_iv,
            iv_pctile=pos.iv_pctile,
            pnl_pct=pnl_pct,
            hold_days=pos.hold_days,
            exit_reason=reason,
        )
        self.closed_trades.append(trade)

        # Position sizing: 1/N base allocation × signal-strength weight
        n_indices = len(self.symbols)
        position_weight = (1.0 / n_indices) * pos.size_weight
        self.equity *= (1 + pnl_pct * position_weight)

        self.positions[symbol] = None

        logger.info("EXIT %s %s: %.1f → %.1f (%.2f%%, %dd, size=%.0f%%, %s)",
                    symbol, reason, pos.entry_spot, obs.spot,
                    pnl_pct * 100, pos.hold_days, pos.size_weight * 100, reason)
        return trade

    def increment_hold(self, symbol: str) -> None:
        """Increment hold_days for an index's position."""
        if self.positions.get(symbol) is not None:
            self.positions[symbol].hold_days += 1

    # --- Signal logic ---

    def check_signal(self, symbol: str) -> str:
        """Check for entry/exit signals for a given index.

        Returns:
            "enter" — IV above entry threshold, no position
            "exit_hold" — max hold days reached
            "exit_iv" — IV dropped below exit threshold
            "hold" — in position, hold conditions met
            "wait" — no position, IV below threshold
            "warmup" — not enough history
        """
        if not self.have_enough_history(symbol):
            return "warmup"

        pctile = self.percentile_rank(symbol)
        pos = self.positions.get(symbol)

        if pos is None:
            if pctile >= self.entry_pctile:
                return "enter"
            return "wait"
        else:
            if pos.hold_days >= self.max_hold_days:
                return "exit_hold"
            if pctile < self.exit_pctile:
                return "exit_iv"
            return "hold"

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

    def trades_by_symbol(self) -> dict[str, list[ClosedTrade]]:
        """Group closed trades by symbol."""
        by_sym: dict[str, list[ClosedTrade]] = {sym: [] for sym in self.symbols}
        for t in self.closed_trades:
            if t.symbol in by_sym:
                by_sym[t.symbol].append(t)
        return by_sym

    def active_position_count(self) -> int:
        """Count how many indices have active positions."""
        return sum(1 for pos in self.positions.values() if pos is not None)


# ---------------------------------------------------------------------------
# Legacy single-index state (for backwards compatibility)
# ---------------------------------------------------------------------------

@dataclass
class PaperState:
    """Legacy single-index paper trading state (NIFTY only).

    Deprecated: Use MultiIndexPaperState instead.
    """
    iv_history: list[IVObservation] = field(default_factory=list)
    position: PaperPosition | None = None
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    equity: float = 1.0
    started_at: str = ""
    last_scan_date: str = ""

    # Strategy config (persisted so state is self-describing)
    iv_lookback: int = 30
    entry_pctile: float = 0.80
    exit_pctile: float = 0.50
    max_hold_days: int = 5
    cost_bps: float = 5.0

    # --- Serialization ---

    def to_dict(self) -> dict:
        return {
            "iv_history": [o.to_dict() for o in self.iv_history],
            "position": self.position.to_dict() if self.position else None,
            "closed_trades": [t.to_dict() for t in self.closed_trades],
            "equity": self.equity,
            "started_at": self.started_at,
            "last_scan_date": self.last_scan_date,
            "config": {
                "iv_lookback": self.iv_lookback,
                "entry_pctile": self.entry_pctile,
                "exit_pctile": self.exit_pctile,
                "max_hold_days": self.max_hold_days,
                "cost_bps": self.cost_bps,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> PaperState:
        cfg = d.get("config", {})
        state = cls(
            iv_history=[IVObservation.from_dict(o) for o in d.get("iv_history", [])],
            position=(PaperPosition.from_dict(d["position"])
                      if d.get("position") else None),
            closed_trades=[ClosedTrade.from_dict(t) for t in d.get("closed_trades", [])],
            equity=float(d.get("equity", 1.0)),
            started_at=d.get("started_at", ""),
            last_scan_date=d.get("last_scan_date", ""),
            iv_lookback=int(cfg.get("iv_lookback", 30)),
            entry_pctile=float(cfg.get("entry_pctile", 0.80)),
            exit_pctile=float(cfg.get("exit_pctile", 0.50)),
            max_hold_days=int(cfg.get("max_hold_days", 5)),
            cost_bps=float(cfg.get("cost_bps", 5.0)),
        )
        return state

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
    def load(cls, path: Path = DEFAULT_STATE_FILE) -> PaperState:
        """Load from JSON, returning fresh state if file doesn't exist."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load state from %s: %s", path, e)
            return cls()

    # --- IV history management ---

    def append_observation(self, obs: IVObservation) -> None:
        """Add a new daily observation. Skips if date already exists."""
        if self.iv_history and self.iv_history[-1].date >= obs.date:
            if self.iv_history[-1].date == obs.date:
                logger.debug("Observation for %s already exists, updating", obs.date)
                self.iv_history[-1] = obs
                return
            logger.warning("Observation for %s is before latest %s, skipping",
                           obs.date, self.iv_history[-1].date)
            return
        self.iv_history.append(obs)

    def current_ivs(self) -> list[float]:
        """Return ATM IV values from history."""
        return [o.atm_iv for o in self.iv_history]

    def percentile_rank(self, window: int | None = None) -> float:
        """Compute current IV percentile rank within lookback window."""
        if not self.iv_history:
            return 0.0
        w = window or self.iv_lookback
        ivs = self.current_ivs()
        lookback = ivs[max(0, len(ivs) - w):]
        current = ivs[-1]
        return sum(1 for x in lookback if x <= current) / len(lookback)

    def have_enough_history(self) -> bool:
        """Check if we have enough IV history for signal generation."""
        return len(self.iv_history) >= self.iv_lookback

    # --- Position management ---

    def enter_position(self, obs: IVObservation, pctile: float) -> PaperPosition:
        """Enter a long NIFTY futures position."""
        if self.position is not None:
            raise ValueError("Already in a position")

        self.position = PaperPosition(
            symbol="NIFTY",
            entry_date=obs.date,
            entry_spot=obs.spot,
            entry_iv=obs.atm_iv,
            iv_pctile=pctile,
            hold_days=0,
        )
        logger.info("ENTER long NIFTY @ %.1f (IV=%.1f%%, pctile=%.2f)",
                     obs.spot, obs.atm_iv * 100, pctile)
        return self.position

    def exit_position(self, obs: IVObservation, reason: str) -> ClosedTrade:
        """Exit the current position and record the trade."""
        if self.position is None:
            raise ValueError("No position to exit")

        pos = self.position
        cost_frac = self.cost_bps / 10_000
        pnl_pct = (obs.spot - pos.entry_spot) / pos.entry_spot - cost_frac

        trade = ClosedTrade(
            symbol="NIFTY",
            entry_date=pos.entry_date,
            exit_date=obs.date,
            entry_spot=pos.entry_spot,
            exit_spot=obs.spot,
            entry_iv=pos.entry_iv,
            exit_iv=obs.atm_iv,
            iv_pctile=pos.iv_pctile,
            pnl_pct=pnl_pct,
            hold_days=pos.hold_days,
            exit_reason=reason,
        )
        self.closed_trades.append(trade)
        self.equity *= (1 + pnl_pct)
        self.position = None

        logger.info("EXIT %s: %.1f → %.1f (%.2f%%, %dd, %s)",
                     reason, pos.entry_spot, obs.spot,
                     pnl_pct * 100, pos.hold_days, reason)
        return trade

    def increment_hold(self) -> None:
        """Increment hold_days for current position."""
        if self.position is not None:
            self.position.hold_days += 1

    # --- Signal logic ---

    def check_signal(self) -> str:
        """Check for entry/exit signals based on current IV percentile.

        Returns:
            "enter" — IV above entry threshold, no position
            "exit_hold" — max hold days reached
            "exit_iv" — IV dropped below exit threshold
            "hold" — in position, hold conditions met
            "wait" — no position, IV below threshold
            "warmup" — not enough history
        """
        if not self.have_enough_history():
            return "warmup"

        pctile = self.percentile_rank()

        if self.position is None:
            if pctile >= self.entry_pctile:
                return "enter"
            return "wait"
        else:
            if self.position.hold_days >= self.max_hold_days:
                return "exit_hold"
            if pctile < self.exit_pctile:
                return "exit_iv"
            return "hold"

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
