"""State manager — unified view of all strategy and portfolio state.

Reads and aggregates:
  1. The main brahmastra_state.json (portfolio-level equity, positions, trades).
  2. All per-strategy state JSON files from data/strategy_state/.
  3. Engine-level real-time state (risk snapshots, signal history).

The API layer queries the StateManager for a single combined view
instead of reading scattered files.  The manager periodically reloads
from disk so it picks up changes from batch processes or manual edits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from engine.state import BrahmastraState, DEFAULT_STATE_FILE

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Directory containing per-strategy JSON state files
_STRATEGY_STATE_DIR = Path("data/strategy_state")

# How often to reload state from disk (seconds)
_RELOAD_INTERVAL_SEC = 10.0


@dataclass
class StrategyStateView:
    """Summarized view of a single strategy's state file."""

    strategy_id: str
    file_path: str
    loaded_at: float               # monotonic time of last load
    positions: list[dict]          # active positions
    closed_trades_count: int
    equity: float
    win_rate: float
    total_return_pct: float
    last_scan_date: str
    raw: dict                      # full deserialized JSON


@dataclass
class CombinedState:
    """Complete state snapshot for the API layer.

    Aggregates portfolio-level, per-strategy, and engine-level data.
    """

    # Portfolio level
    equity: float
    peak_equity: float
    portfolio_dd: float
    total_exposure: float
    cash: float
    positions: list[dict]
    closed_trades_count: int
    win_rate: float
    total_return_pct: float
    last_scan_date: str
    last_scan_time: str
    regime: str
    vix: float
    vpin: float
    circuit_breaker_active: bool

    # Per-strategy
    strategies: dict[str, StrategyStateView]
    strategy_equity: dict[str, float]

    # Engine real-time (optional, filled when engine is running)
    engine_running: bool = False
    engine_stats: dict = field(default_factory=dict)
    latest_risk_snapshot: dict = field(default_factory=dict)
    signal_history: list[dict] = field(default_factory=list)

    # Meta
    generated_at: str = ""


class StateManager:
    """Unified state aggregator for the BRAHMASTRA system.

    Parameters
    ----------
    state_file : Path
        Path to the main brahmastra_state.json.
    strategy_state_dir : Path
        Directory containing per-strategy JSON state files.
    reload_interval : float
        Seconds between automatic state reloads from disk.
    """

    def __init__(
        self,
        state_file: Path = DEFAULT_STATE_FILE,
        strategy_state_dir: Path = _STRATEGY_STATE_DIR,
        reload_interval: float = _RELOAD_INTERVAL_SEC,
    ) -> None:
        self._state_file = state_file
        self._strategy_dir = strategy_state_dir
        self._reload_interval = reload_interval

        # Cached state
        self._brahmastra_state: BrahmastraState | None = None
        self._strategy_states: dict[str, StrategyStateView] = {}
        self._last_reload: float = 0.0

        # Signal history (ring buffer, last N signals from the live session)
        self._signal_history: list[dict] = []
        self._max_signal_history = 200

        # Engine stats reference (set by the main engine loop)
        self._engine_stats: dict = {}
        self._risk_snapshot: dict = {}
        self._engine_running = False

        # Control
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the periodic state reload loop."""
        self._running = True
        await self._reload()
        self._task = asyncio.create_task(self._reload_loop(), name="state_manager")
        logger.info("StateManager started (reload every %.0fs)", self._reload_interval)

    async def stop(self) -> None:
        """Stop the reload loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("StateManager stopped")

    async def _reload_loop(self) -> None:
        """Periodically reload state from disk."""
        while self._running:
            try:
                await asyncio.sleep(self._reload_interval)
            except asyncio.CancelledError:
                break
            await self._reload()

    async def _reload(self) -> None:
        """Reload all state files from disk."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._reload_sync)
        self._last_reload = time.monotonic()

    def _reload_sync(self) -> None:
        """Synchronous reload (runs in executor)."""
        # 1. Main Brahmastra state
        try:
            self._brahmastra_state = BrahmastraState.load(self._state_file)
        except Exception as e:
            logger.warning("Failed to load Brahmastra state: %s", e)
            if self._brahmastra_state is None:
                self._brahmastra_state = BrahmastraState()

        # 2. Per-strategy state files
        self._strategy_states.clear()

        if self._strategy_dir.exists() and self._strategy_dir.is_dir():
            for state_file in sorted(self._strategy_dir.glob("*.json")):
                try:
                    data = json.loads(state_file.read_text())
                    view = self._parse_strategy_state(state_file, data)
                    self._strategy_states[view.strategy_id] = view
                except Exception as e:
                    logger.warning(
                        "Failed to load strategy state %s: %s", state_file.name, e,
                    )

    def _parse_strategy_state(
        self,
        file_path: Path,
        data: dict,
    ) -> StrategyStateView:
        """Parse a strategy state JSON into a StrategyStateView."""
        # Strategy ID from filename (e.g. "s1_vrp.json" → "s1_vrp")
        strategy_id = data.get("strategy_id", file_path.stem)

        # Extract positions
        positions = []
        raw_positions = data.get("positions", {})
        if isinstance(raw_positions, dict):
            for sym, pos in raw_positions.items():
                if isinstance(pos, dict):
                    pos["symbol"] = pos.get("symbol", sym)
                    positions.append(pos)
        elif isinstance(raw_positions, list):
            positions = raw_positions

        # Extract metrics
        closed = data.get("closed_trades", [])
        closed_count = len(closed) if isinstance(closed, list) else 0

        equity = float(data.get("equity", 1.0))

        # Win rate
        if closed_count > 0 and isinstance(closed, list):
            wins = sum(
                1 for t in closed
                if isinstance(t, dict) and float(t.get("pnl_pct", t.get("net_pnl_pct", 0))) > 0
            )
            win_rate = wins / closed_count
        else:
            win_rate = 0.0

        # Total return
        initial = float(data.get("initial_equity", 1.0))
        if initial > 0:
            total_ret = (equity / initial - 1.0) * 100
        else:
            total_ret = 0.0

        return StrategyStateView(
            strategy_id=strategy_id,
            file_path=str(file_path),
            loaded_at=time.monotonic(),
            positions=positions,
            closed_trades_count=closed_count,
            equity=equity,
            win_rate=win_rate,
            total_return_pct=total_ret,
            last_scan_date=data.get("last_scan_date", ""),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Combined state view
    # ------------------------------------------------------------------

    def get_combined_state(self) -> CombinedState:
        """Build and return the unified state snapshot.

        This is the main API entry point for the dashboard / REST endpoint.
        """
        state = self._brahmastra_state or BrahmastraState()

        positions = [p.to_dict() for p in state.active_positions()]

        return CombinedState(
            # Portfolio
            equity=state.equity,
            peak_equity=state.peak_equity,
            portfolio_dd=state.portfolio_dd,
            total_exposure=state.total_exposure,
            cash=state.cash,
            positions=positions,
            closed_trades_count=len(state.closed_trades),
            win_rate=state.win_rate(),
            total_return_pct=state.total_return_pct(),
            last_scan_date=state.last_scan_date,
            last_scan_time=state.last_scan_time,
            regime=state.last_regime,
            vix=state.last_vix,
            vpin=state.last_vpin,
            circuit_breaker_active=state.circuit_breaker_active,
            # Strategies
            strategies=dict(self._strategy_states),
            strategy_equity=dict(state.strategy_equity),
            # Engine
            engine_running=self._engine_running,
            engine_stats=dict(self._engine_stats),
            latest_risk_snapshot=dict(self._risk_snapshot),
            signal_history=list(self._signal_history),
            # Meta
            generated_at=datetime.now(IST).isoformat(),
        )

    def get_strategy_state(self, strategy_id: str) -> StrategyStateView | None:
        """Get state view for a specific strategy."""
        return self._strategy_states.get(strategy_id)

    def list_strategies(self) -> list[str]:
        """List all known strategy IDs."""
        ids = set(self._strategy_states.keys())
        if self._brahmastra_state:
            ids.update(self._brahmastra_state.strategy_equity.keys())
        return sorted(ids)

    # ------------------------------------------------------------------
    # Engine integration (called by the main engine loop)
    # ------------------------------------------------------------------

    def set_engine_running(self, running: bool) -> None:
        """Mark whether the live engine is running."""
        self._engine_running = running

    def update_engine_stats(self, stats: dict) -> None:
        """Update engine-level stats (tick handler, signal gen, risk monitor)."""
        self._engine_stats = stats

    def update_risk_snapshot(self, snapshot: dict) -> None:
        """Update the latest risk snapshot from the risk monitor."""
        self._risk_snapshot = snapshot

    def record_signal(self, signal_data: dict) -> None:
        """Record a signal in the rolling history buffer."""
        self._signal_history.append(signal_data)
        if len(self._signal_history) > self._max_signal_history:
            self._signal_history = self._signal_history[-self._max_signal_history:]

    # ------------------------------------------------------------------
    # Serialization for API responses
    # ------------------------------------------------------------------

    def to_api_dict(self) -> dict:
        """Serialize the combined state to a JSON-safe dict.

        This is the payload returned by the ``/api/state`` endpoint.
        """
        combined = self.get_combined_state()

        strategies_dict = {}
        for sid, sv in combined.strategies.items():
            strategies_dict[sid] = {
                "strategy_id": sv.strategy_id,
                "positions": sv.positions,
                "closed_trades_count": sv.closed_trades_count,
                "equity": sv.equity,
                "win_rate": round(sv.win_rate, 4),
                "total_return_pct": round(sv.total_return_pct, 2),
                "last_scan_date": sv.last_scan_date,
            }

        return {
            "portfolio": {
                "equity": round(combined.equity, 6),
                "peak_equity": round(combined.peak_equity, 6),
                "portfolio_dd": round(combined.portfolio_dd, 4),
                "total_exposure": round(combined.total_exposure, 4),
                "cash": round(combined.cash, 6),
                "positions": combined.positions,
                "closed_trades_count": combined.closed_trades_count,
                "win_rate": round(combined.win_rate, 4),
                "total_return_pct": round(combined.total_return_pct, 2),
                "last_scan_date": combined.last_scan_date,
                "last_scan_time": combined.last_scan_time,
            },
            "market": {
                "regime": combined.regime,
                "vix": round(combined.vix, 1),
                "vpin": round(combined.vpin, 4),
            },
            "risk": {
                "circuit_breaker_active": combined.circuit_breaker_active,
                "latest_snapshot": combined.latest_risk_snapshot,
            },
            "strategies": strategies_dict,
            "strategy_equity": {
                k: round(v, 6) for k, v in combined.strategy_equity.items()
            },
            "engine": {
                "running": combined.engine_running,
                "stats": combined.engine_stats,
            },
            "signal_history": combined.signal_history[-20:],
            "generated_at": combined.generated_at,
        }

    def stats(self) -> dict:
        """Return manager-level diagnostics."""
        return {
            "state_file": str(self._state_file),
            "strategy_dir": str(self._strategy_dir),
            "strategies_loaded": len(self._strategy_states),
            "signal_history_size": len(self._signal_history),
            "last_reload_age_sec": round(time.monotonic() - self._last_reload, 1),
            "engine_running": self._engine_running,
        }
