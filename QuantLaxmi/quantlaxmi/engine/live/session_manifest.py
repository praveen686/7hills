"""SessionManifest — per-run metadata and summary.

One manifest per run, written to data/sessions/run_{run_id}.json.

Lifecycle:
  1. start() — writes the start header (code version, config, start time)
  2. During run — updated incrementally (optional)
  3. finalize() — writes end-of-session summary (PnL, DD, trades, blocks)

The manifest enables:
  - Session discovery (ls data/sessions/ to find all runs)
  - Replay verification (compare replay summary vs original)
  - Operational audit (what config was in force)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from quantlaxmi.data._paths import SESSIONS_DIR

logger = logging.getLogger(__name__)


class SessionManifest:
    """Per-run session metadata and summary writer.

    Parameters
    ----------
    base_dir : Path
        Directory for session manifests (e.g. ``data/sessions/``).
    run_id : str
        Stable identifier for this session.
    """

    def __init__(
        self,
        base_dir: Path | str = SESSIONS_DIR,
        run_id: str = "",
    ):
        self._base_dir = Path(base_dir)
        self._run_id = run_id
        self._manifest: dict = {}
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        safe_id = self._run_id.replace(":", "-")
        return self._base_dir / f"run_{safe_id}.json"

    def start(
        self,
        strategies: list[str],
        risk_limits: dict,
        data_sources: list[str] | None = None,
        code_version: str = "",
    ) -> None:
        """Write the session start header."""
        self._manifest = {
            "run_id": self._run_id,
            "status": "running",
            "start_ts": datetime.now(timezone.utc).isoformat(),
            "end_ts": None,
            "code_version": code_version,
            "platform": {
                "python": platform.python_version(),
                "os": platform.system(),
                "machine": platform.machine(),
            },
            "config": {
                "strategies": strategies,
                "risk_limits": risk_limits,
                "data_sources": data_sources or [],
            },
            "summary": None,
        }
        self._save()
        logger.info("Session manifest started: %s", self.path)

    def finalize(
        self,
        total_signals: int = 0,
        total_trades: int = 0,
        total_blocks: int = 0,
        final_equity: float = 1.0,
        peak_equity: float = 1.0,
        max_dd: float = 0.0,
        total_pnl_pct: float = 0.0,
        event_count: int = 0,
        error: str = "",
    ) -> None:
        """Write the end-of-session summary and mark complete."""
        self._manifest["status"] = "error" if error else "complete"
        self._manifest["end_ts"] = datetime.now(timezone.utc).isoformat()
        self._manifest["summary"] = {
            "total_signals": total_signals,
            "total_trades": total_trades,
            "total_blocks": total_blocks,
            "final_equity": final_equity,
            "peak_equity": peak_equity,
            "max_dd": max_dd,
            "total_pnl_pct": total_pnl_pct,
            "event_count": event_count,
            "error": error,
        }
        self._save()
        logger.info("Session manifest finalized: %s (%s)", self.path, self._manifest["status"])

    def _save(self) -> None:
        """Atomic write via tempfile + rename."""
        fd, tmp = tempfile.mkstemp(dir=self._base_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._manifest, f, indent=2, sort_keys=True, default=str)
            os.replace(tmp, str(self.path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load(self) -> dict:
        """Read back the manifest (for verification)."""
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    @property
    def run_id(self) -> str:
        return self._run_id
