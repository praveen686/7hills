"""Strategy state reader — loads per-strategy JSON state files.

Reads strategy state files from ``data/strategy_state/`` and returns
uniform dicts suitable for the REST API.  Each strategy may persist
its own state file with positions, closed trades, equity, and metadata.

File convention:
    data/strategy_state/{strategy_id}.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY_DIR = Path("data/strategy_state")


class StrategyReader:
    """Read and normalise per-strategy JSON state files."""

    def __init__(self, strategy_dir: Path | None = None) -> None:
        self._dir = strategy_dir or DEFAULT_STRATEGY_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_ids(self) -> list[str]:
        """Return sorted list of strategy IDs that have state files."""
        if not self._dir.exists():
            return []
        return sorted(
            p.stem for p in self._dir.glob("*.json") if p.is_file()
        )

    def read(self, strategy_id: str) -> dict[str, Any] | None:
        """Read a single strategy state file and return a normalised dict.

        Returns None if the file does not exist or cannot be parsed.
        """
        path = self._dir / f"{strategy_id}.json"
        raw = self._load(path)
        if raw is None:
            return None
        return self._normalise(strategy_id, raw, path)

    def read_all(self) -> list[dict[str, Any]]:
        """Read all strategy state files."""
        results: list[dict[str, Any]] = []
        for sid in self.list_ids():
            entry = self.read(sid)
            if entry is not None:
                results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read strategy state %s: %s", path, exc)
            return None

    def _normalise(
        self,
        strategy_id: str,
        raw: dict[str, Any],
        path: Path,
    ) -> dict[str, Any]:
        """Convert raw JSON into a uniform API-friendly dict."""
        closed_trades = raw.get("closed_trades", [])
        positions = raw.get("positions", {})
        equity = float(raw.get("equity", 1.0))

        # Compute win rate
        n_closed = len(closed_trades)
        wins = sum(1 for t in closed_trades if t.get("pnl_pct", 0) > 0)
        win_rate = (wins / n_closed * 100) if n_closed > 0 else 0.0

        # Staleness from file modification time
        status = self._staleness(path)

        # Flatten positions dict into a list
        position_list: list[dict[str, Any]] = []
        if isinstance(positions, dict):
            for key, pos in positions.items():
                if isinstance(pos, dict):
                    pos_entry = {"key": key, **pos}
                    position_list.append(pos_entry)
        elif isinstance(positions, list):
            position_list = positions

        return {
            "strategy_id": strategy_id,
            "name": raw.get("name", strategy_id),
            "status": status,
            "equity": round(equity, 6),
            "return_pct": round((equity - 1.0) * 100, 4),
            "n_open": len(position_list),
            "n_closed": n_closed,
            "win_rate": round(win_rate, 1),
            "positions": position_list,
            "recent_trades": [
                {
                    "symbol": t.get("symbol", ""),
                    "direction": t.get("direction", ""),
                    "entry_date": t.get("entry_date", ""),
                    "exit_date": t.get("exit_date", ""),
                    "pnl_pct": round(float(t.get("pnl_pct", 0)) * 100, 4),
                    "exit_reason": t.get("exit_reason", ""),
                }
                for t in closed_trades[-20:]
            ],
            "metadata": {
                k: v
                for k, v in raw.items()
                if k not in ("closed_trades", "positions", "equity", "name")
            },
        }

    @staticmethod
    def _staleness(path: Path) -> str:
        """Determine file staleness: running / stale / stopped."""
        try:
            mtime = path.stat().st_mtime
            age = datetime.now(timezone.utc).timestamp() - mtime
        except OSError:
            return "stopped"
        if age < 900:
            return "running"
        if age < 3600:
            return "stale"
        return "stopped"
