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

from quantlaxmi.data._paths import STRATEGY_STATE
from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.strategies.protocol import Signal

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base for all BRAHMASTRA strategies.

    Subclasses must implement:
      - ``strategy_id`` (property) — unique identifier
      - ``warmup_days()`` — minimum historical days
      - ``_scan_impl(d, store)`` — the actual scanning logic
    """

    def __init__(self, state_dir: Path | str | None = None):
        self._state_dir = Path(state_dir) if state_dir else STRATEGY_STATE
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

    # ------------------------------------------------------------------
    # CUSUM event filter (AFML integration)
    # ------------------------------------------------------------------

    def cusum_events(
        self, close: "pd.Series", threshold_mult: float = 1.0
    ) -> "pd.DatetimeIndex":
        """Get CUSUM-filtered event timestamps for adaptive sampling.

        Uses AFML's symmetric CUSUM filter with an adaptive threshold
        based on daily volatility. Events fire on statistically
        significant mean shifts — sparse in calm markets, frequent
        in volatile regimes.

        Parameters
        ----------
        close : pd.Series
            Price series with DatetimeIndex.
        threshold_mult : float
            Multiplier on daily vol for the CUSUM threshold.
            Higher = fewer events, lower = more events.

        Returns
        -------
        pd.DatetimeIndex
            Timestamps at which CUSUM events fired.
        """
        from quantlaxmi.models.afml import cusum_filter, get_daily_vol

        daily_vol = get_daily_vol(close)
        threshold = daily_vol * threshold_mult
        return cusum_filter(close, threshold=threshold)
