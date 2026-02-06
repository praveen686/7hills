"""Persistent state for RNDR (Risk-Neutral Density Regime) paper trading.

Tracks density observations, composite signals, positions, and trades for
BANKNIFTY, MIDCPNIFTY, and FINNIFTY (NIFTY excluded — too efficient).

Uses atomic JSON writes (tempfile + os.replace) for crash safety.
Mirrors the pattern of ``paper_state.MultiIndexPaperState``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from apps.india_fno.density_strategy import (
    DEFAULT_COST_BPS,
    DEFAULT_ENTRY_PCTILE,
    DEFAULT_EXIT_PCTILE,
    DEFAULT_HOLD_DAYS,
    DEFAULT_LOOKBACK,
    DEFAULT_PHYS_WINDOW,
    DensityDayObs,
    _rolling_percentile,
    compute_composite_signal,
)

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = Path("data/rndr_paper_state.json")

# NIFTY excluded: too efficient, negative across all configs
RNDR_SYMBOLS = ["BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DensityObservation:
    """Single day's density observation for an index (serializable)."""

    date: str  # ISO YYYY-MM-DD
    symbol: str
    spot: float
    atm_iv: float
    rn_skewness: float
    rn_kurtosis: float
    entropy: float
    left_tail: float
    right_tail: float
    phys_skewness: float
    skew_premium: float
    entropy_change: float
    kl_div: float
    density_ok: bool

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "symbol": self.symbol,
            "spot": self.spot,
            "atm_iv": self.atm_iv,
            "rn_skewness": self.rn_skewness,
            "rn_kurtosis": self.rn_kurtosis,
            "entropy": self.entropy,
            "left_tail": self.left_tail,
            "right_tail": self.right_tail,
            "phys_skewness": self.phys_skewness,
            "skew_premium": self.skew_premium,
            "entropy_change": self.entropy_change,
            "kl_div": self.kl_div,
            "density_ok": bool(self.density_ok),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DensityObservation:
        return cls(
            date=d["date"],
            symbol=d["symbol"],
            spot=float(d["spot"]),
            atm_iv=float(d["atm_iv"]),
            rn_skewness=float(d["rn_skewness"]),
            rn_kurtosis=float(d["rn_kurtosis"]),
            entropy=float(d["entropy"]),
            left_tail=float(d["left_tail"]),
            right_tail=float(d["right_tail"]),
            phys_skewness=float(d["phys_skewness"]),
            skew_premium=float(d["skew_premium"]),
            entropy_change=float(d["entropy_change"]),
            kl_div=float(d["kl_div"]),
            density_ok=bool(d["density_ok"]),
        )

    def to_day_obs(self) -> DensityDayObs:
        """Convert back to frozen DensityDayObs for signal computation."""
        return DensityDayObs(
            date=date.fromisoformat(self.date),
            symbol=self.symbol,
            spot=self.spot,
            atm_iv=self.atm_iv,
            rn_skewness=self.rn_skewness,
            rn_kurtosis=self.rn_kurtosis,
            entropy=self.entropy,
            left_tail=self.left_tail,
            right_tail=self.right_tail,
            phys_skewness=self.phys_skewness,
            skew_premium=self.skew_premium,
            entropy_change=self.entropy_change,
            kl_div=self.kl_div,
            density_ok=self.density_ok,
        )

    @classmethod
    def from_day_obs(cls, obs: DensityDayObs) -> DensityObservation:
        """Create from a DensityDayObs (date → ISO string).

        Explicit float/bool casts to convert numpy scalars to native Python
        types for JSON serialization.
        """
        return cls(
            date=obs.date.isoformat(),
            symbol=obs.symbol,
            spot=float(obs.spot),
            atm_iv=float(obs.atm_iv),
            rn_skewness=float(obs.rn_skewness),
            rn_kurtosis=float(obs.rn_kurtosis),
            entropy=float(obs.entropy),
            left_tail=float(obs.left_tail),
            right_tail=float(obs.right_tail),
            phys_skewness=float(obs.phys_skewness),
            skew_premium=float(obs.skew_premium),
            entropy_change=float(obs.entropy_change),
            kl_div=float(obs.kl_div),
            density_ok=bool(obs.density_ok),
        )


@dataclass
class RNDRPosition:
    """Active index futures position."""

    symbol: str
    entry_date: str  # ISO
    entry_spot: float
    entry_signal: float
    signal_pctile: float
    hold_days: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date,
            "entry_spot": self.entry_spot,
            "entry_signal": self.entry_signal,
            "signal_pctile": self.signal_pctile,
            "hold_days": self.hold_days,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RNDRPosition:
        return cls(
            symbol=d["symbol"],
            entry_date=d["entry_date"],
            entry_spot=float(d["entry_spot"]),
            entry_signal=float(d["entry_signal"]),
            signal_pctile=float(d["signal_pctile"]),
            hold_days=int(d["hold_days"]),
        )


@dataclass
class RNDRClosedTrade:
    """Completed trade record."""

    symbol: str
    entry_date: str
    exit_date: str
    entry_spot: float
    exit_spot: float
    entry_signal: float
    pnl_pct: float
    hold_days: int
    exit_reason: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_spot": self.entry_spot,
            "exit_spot": self.exit_spot,
            "entry_signal": self.entry_signal,
            "pnl_pct": self.pnl_pct,
            "hold_days": self.hold_days,
            "exit_reason": self.exit_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RNDRClosedTrade:
        return cls(
            symbol=d["symbol"],
            entry_date=d["entry_date"],
            exit_date=d["exit_date"],
            entry_spot=float(d["entry_spot"]),
            exit_spot=float(d["exit_spot"]),
            entry_signal=float(d["entry_signal"]),
            pnl_pct=float(d["pnl_pct"]),
            hold_days=int(d["hold_days"]),
            exit_reason=d["exit_reason"],
        )


# ---------------------------------------------------------------------------
# Main state
# ---------------------------------------------------------------------------

@dataclass
class DensityPaperState:
    """Complete RNDR paper trading state."""

    # Per-index density history: symbol -> list of observations
    density_histories: dict[str, list[DensityObservation]] = field(default_factory=dict)

    # Previous day's raw density for KL divergence: symbol -> list of 500 floats
    prev_densities: dict[str, list[float]] = field(default_factory=dict)

    # Per-index positions: symbol -> position (or None)
    positions: dict[str, RNDRPosition | None] = field(default_factory=dict)

    # All closed trades
    closed_trades: list[RNDRClosedTrade] = field(default_factory=list)

    # Global equity
    equity: float = 1.0
    started_at: str = ""
    last_scan_date: str = ""

    # Indices being tracked
    symbols: list[str] = field(default_factory=lambda: RNDR_SYMBOLS.copy())

    # Strategy config
    lookback: int = DEFAULT_LOOKBACK
    entry_pctile: float = DEFAULT_ENTRY_PCTILE
    exit_pctile: float = DEFAULT_EXIT_PCTILE
    max_hold_days: int = DEFAULT_HOLD_DAYS
    cost_bps: float = DEFAULT_COST_BPS
    phys_window: int = DEFAULT_PHYS_WINDOW

    def __post_init__(self):
        for sym in self.symbols:
            if sym not in self.density_histories:
                self.density_histories[sym] = []
            if sym not in self.positions:
                self.positions[sym] = None

    # --- Serialization ---

    def to_dict(self) -> dict:
        return {
            "density_histories": {
                sym: [o.to_dict() for o in obs_list]
                for sym, obs_list in self.density_histories.items()
            },
            "prev_densities": {
                sym: vals
                for sym, vals in self.prev_densities.items()
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
                "lookback": self.lookback,
                "entry_pctile": self.entry_pctile,
                "exit_pctile": self.exit_pctile,
                "max_hold_days": self.max_hold_days,
                "cost_bps": self.cost_bps,
                "phys_window": self.phys_window,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> DensityPaperState:
        cfg = d.get("config", {})
        symbols = d.get("symbols", RNDR_SYMBOLS.copy())

        density_histories = {}
        for sym, obs_list in d.get("density_histories", {}).items():
            density_histories[sym] = [DensityObservation.from_dict(o) for o in obs_list]

        prev_densities = {}
        for sym, vals in d.get("prev_densities", {}).items():
            prev_densities[sym] = [float(v) for v in vals]

        positions = {}
        for sym, pos_data in d.get("positions", {}).items():
            positions[sym] = RNDRPosition.from_dict(pos_data) if pos_data else None

        return cls(
            density_histories=density_histories,
            prev_densities=prev_densities,
            positions=positions,
            closed_trades=[RNDRClosedTrade.from_dict(t) for t in d.get("closed_trades", [])],
            equity=float(d.get("equity", 1.0)),
            started_at=d.get("started_at", ""),
            last_scan_date=d.get("last_scan_date", ""),
            symbols=symbols,
            lookback=int(cfg.get("lookback", DEFAULT_LOOKBACK)),
            entry_pctile=float(cfg.get("entry_pctile", DEFAULT_ENTRY_PCTILE)),
            exit_pctile=float(cfg.get("exit_pctile", DEFAULT_EXIT_PCTILE)),
            max_hold_days=int(cfg.get("max_hold_days", DEFAULT_HOLD_DAYS)),
            cost_bps=float(cfg.get("cost_bps", DEFAULT_COST_BPS)),
            phys_window=int(cfg.get("phys_window", DEFAULT_PHYS_WINDOW)),
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
    def load(cls, path: Path = DEFAULT_STATE_FILE) -> DensityPaperState:
        """Load from JSON, returning fresh state if file doesn't exist."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load RNDR state from %s: %s", path, e)
            return cls()

    # --- History management ---

    def append_observation(self, obs: DensityObservation) -> None:
        """Add a new daily observation for an index. Skips if date exists."""
        symbol = obs.symbol
        if symbol not in self.density_histories:
            self.density_histories[symbol] = []

        history = self.density_histories[symbol]
        if history and history[-1].date >= obs.date:
            if history[-1].date == obs.date:
                logger.debug("%s: Observation for %s already exists, updating", symbol, obs.date)
                history[-1] = obs
                return
            logger.warning("%s: Observation for %s is before latest %s, skipping",
                           symbol, obs.date, history[-1].date)
            return
        history.append(obs)

    def get_density_day_obs(self, symbol: str) -> list[DensityDayObs]:
        """Convert stored observations back to DensityDayObs for signal computation."""
        return [o.to_day_obs() for o in self.density_histories.get(symbol, [])]

    def have_enough_history(self, symbol: str) -> bool:
        """Check if we have enough history for signal generation."""
        return len(self.density_histories.get(symbol, [])) >= self.lookback

    def latest_observation(self, symbol: str) -> DensityObservation | None:
        history = self.density_histories.get(symbol, [])
        return history[-1] if history else None

    # --- Signal logic ---

    def check_signal(self, symbol: str) -> tuple[str, float, float]:
        """Check for entry/exit signals based on composite density signal.

        Returns:
            (signal_type, composite_value, signal_pctile)
            signal_type: "enter"/"exit_hold"/"exit_signal"/"hold"/"wait"/"warmup"
        """
        if not self.have_enough_history(symbol):
            return "warmup", 0.0, 0.0

        series = self.get_density_day_obs(symbol)
        signals = compute_composite_signal(series, self.lookback)

        if not signals:
            return "warmup", 0.0, 0.0

        idx = len(signals) - 1
        composite = signals[idx]
        sig_pctile = _rolling_percentile(signals, idx, self.lookback)

        pos = self.positions.get(symbol)

        if pos is None:
            if sig_pctile >= self.entry_pctile and composite > 0:
                return "enter", composite, sig_pctile
            return "wait", composite, sig_pctile
        else:
            if pos.hold_days >= self.max_hold_days:
                return "exit_hold", composite, sig_pctile
            if sig_pctile < self.exit_pctile:
                return "exit_signal", composite, sig_pctile
            return "hold", composite, sig_pctile

    # --- Position management ---

    def enter_position(self, symbol: str, obs: DensityObservation,
                       composite: float, sig_pctile: float) -> RNDRPosition:
        """Enter a long futures position."""
        if self.positions.get(symbol) is not None:
            raise ValueError(f"Already in a position for {symbol}")

        pos = RNDRPosition(
            symbol=symbol,
            entry_date=obs.date,
            entry_spot=obs.spot,
            entry_signal=composite,
            signal_pctile=sig_pctile,
            hold_days=0,
        )
        self.positions[symbol] = pos
        logger.info("ENTER long %s @ %.1f (signal=%.3f, pctile=%.2f)",
                     symbol, obs.spot, composite, sig_pctile)
        return pos

    def exit_position(self, symbol: str, obs: DensityObservation,
                      reason: str) -> RNDRClosedTrade:
        """Exit the current position and record the trade."""
        pos = self.positions.get(symbol)
        if pos is None:
            raise ValueError(f"No position to exit for {symbol}")

        cost_frac = self.cost_bps / 10_000
        pnl_pct = (obs.spot - pos.entry_spot) / pos.entry_spot - cost_frac

        trade = RNDRClosedTrade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=obs.date,
            entry_spot=pos.entry_spot,
            exit_spot=obs.spot,
            entry_signal=pos.entry_signal,
            pnl_pct=pnl_pct,
            hold_days=pos.hold_days,
            exit_reason=reason,
        )
        self.closed_trades.append(trade)

        # 1/N allocation across indices
        n_indices = len(self.symbols)
        position_weight = 1.0 / n_indices
        self.equity *= (1 + pnl_pct * position_weight)

        self.positions[symbol] = None

        logger.info("EXIT %s %s: %.1f -> %.1f (%.2f%%, %dd, %s)",
                     symbol, reason, pos.entry_spot, obs.spot,
                     pnl_pct * 100, pos.hold_days, reason)
        return trade

    def increment_hold(self, symbol: str) -> None:
        """Increment hold_days for an index's position."""
        if self.positions.get(symbol) is not None:
            self.positions[symbol].hold_days += 1

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

    def trades_by_symbol(self) -> dict[str, list[RNDRClosedTrade]]:
        """Group closed trades by symbol."""
        by_sym: dict[str, list[RNDRClosedTrade]] = {sym: [] for sym in self.symbols}
        for t in self.closed_trades:
            if t.symbol in by_sym:
                by_sym[t.symbol].append(t)
        return by_sym

    def active_position_count(self) -> int:
        return sum(1 for pos in self.positions.values() if pos is not None)
