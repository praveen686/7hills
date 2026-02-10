"""Strategy protocol and signal dataclass for BRAHMASTRA.

Every strategy in the system must implement ``StrategyProtocol``.  The
protocol is intentionally minimal: each strategy scans a date and returns
a list of ``Signal`` objects that the orchestrator routes through the
meta-allocator and risk manager before execution.

Signals are frozen dataclasses — immutable once created.  This prevents
downstream components from accidentally mutating upstream state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

from quantlaxmi.data.store import MarketDataStore


@dataclass(frozen=True)
class Signal:
    """A strategy's output for a single instrument.

    Attributes
    ----------
    strategy_id : str
        Identifier of the emitting strategy (e.g. "s1_vrp").
    symbol : str
        Underlying symbol (e.g. "NIFTY", "BANKNIFTY", "RELIANCE").
    direction : str
        One of "long", "short", "flat".
    conviction : float
        Signal strength in [0, 1].  Higher = more confident.
    instrument_type : str
        "FUT" for futures, "CE" / "PE" for single-leg options,
        "SPREAD" for multi-leg option positions.
    strike : float
        Relevant strike price (0.0 for futures).
    expiry : str
        Expiry date as ISO string (empty for default nearest).
    ttl_bars : int
        Expected holding period in bars (days for daily strategies).
    metadata : dict
        Strategy-specific context (e.g. IV percentile, composite score).
    """

    strategy_id: str
    symbol: str
    direction: str
    conviction: float
    instrument_type: str = "FUT"
    strike: float = 0.0
    expiry: str = ""
    ttl_bars: int = 5
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "flat"):
            raise ValueError(f"Invalid direction: {self.direction!r}")
        if not 0.0 <= self.conviction <= 1.0:
            raise ValueError(f"Conviction must be in [0, 1], got {self.conviction}")
        if self.instrument_type not in ("FUT", "CE", "PE", "SPREAD"):
            raise ValueError(f"Invalid instrument_type: {self.instrument_type!r}")


@runtime_checkable
class StrategyProtocol(Protocol):
    """Contract that every BRAHMASTRA strategy must implement.

    The ``scan`` method is called once per trading day (in paper/live mode)
    or once per historical date (in backtest mode).  It must return only
    causal signals — no lookahead.
    """

    @property
    def strategy_id(self) -> str:
        """Unique identifier for this strategy."""
        ...

    def scan(self, d: date, store: MarketDataStore) -> list[Signal]:
        """Scan a single date and return zero or more signals.

        Parameters
        ----------
        d : date
            The trading date to scan.
        store : MarketDataStore
            Read-only access to market data.

        Returns
        -------
        list[Signal]
            Zero or more signals.  An empty list means no trade today.
        """
        ...

    def warmup_days(self) -> int:
        """Minimum historical days required before the strategy can emit signals."""
        ...
