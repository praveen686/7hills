"""Base strategy class with common logic.

Provides JSON state persistence, structured logging, and the
``StrategyProtocol`` interface.  Concrete strategies subclass this
and implement ``_scan_impl``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

from data.store import MarketDataStore
from strategies.protocol import Signal

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base for all BRAHMASTRA strategies.

    Subclasses must implement:
      - ``strategy_id`` (property) — unique identifier
      - ``warmup_days()`` — minimum historical days
      - ``_scan_impl(d, store)`` — the actual scanning logic
    """

    def __init__(self, state_dir: Path | str = Path("data/strategy_state")):
        self._state_dir = Path(state_dir)
        self._state: dict = {}
        self._load_state()

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        ...

    @abstractmethod
    def warmup_days(self) -> int:
        ...

    @abstractmethod
    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        """Produce signals for a single date.

        Implementations must be fully causal — no future data.
        """
        ...

    def scan(self, d: date, store: MarketDataStore) -> list[Signal]:
        """Public entry point — delegates to ``_scan_impl`` with logging."""
        signals = self._scan_impl(d, store)
        if signals:
            logger.info(
                "[%s] %s: %d signal(s) — %s",
                self.strategy_id,
                d.isoformat(),
                len(signals),
                ", ".join(
                    f"{s.symbol} {s.direction} {s.conviction:.2f}"
                    for s in signals
                ),
            )
        return signals

    # ------------------------------------------------------------------
    # State persistence (atomic JSON, same pattern as paper_state.py)
    # ------------------------------------------------------------------

    @property
    def _state_file(self) -> Path:
        return self._state_dir / f"{self.strategy_id}.json"

    def _load_state(self) -> None:
        if self._state_file.exists():
            try:
                self._state = json.loads(self._state_file.read_text())
            except Exception as e:
                logger.warning("Failed to load state for %s: %s", self.strategy_id, e)
                self._state = {}

    def _save_state(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._state_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
            os.replace(tmp, str(self._state_file))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def get_state(self, key: str, default=None):
        return self._state.get(key, default)

    def set_state(self, key: str, value) -> None:
        self._state[key] = value
        self._save_state()

    def update_state(self, updates: dict) -> None:
        self._state.update(updates)
        self._save_state()
