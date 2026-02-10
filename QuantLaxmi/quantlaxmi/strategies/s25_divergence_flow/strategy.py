"""S25: Divergence Flow Field — Conservation-Law Flow Strategy.

Uses Helmholtz decomposition of NSE 4-party participant OI flows to
detect information asymmetry pressure gradients before they materialize
in price.

Edge: conservation law (zero-sum constraint) in NSE derivatives creates
a measurable divergence between informed (FII+Pro) and uninformed
(Client+DII) positioning flows. The divergence predicts 3-5 day returns.

Instruments: NIFTY, BANKNIFTY index futures.
Holding period: 3-10 days.
Expected Sharpe: 2.0-2.5 (after costs).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.features.divergence_flow import DivergenceFlowBuilder, DFFConfig
from quantlaxmi.strategies.base import BaseStrategy
from quantlaxmi.strategies.protocol import Signal

logger = logging.getLogger(__name__)

SYMBOLS = ["NIFTY", "BANKNIFTY"]

# Signal thresholds
DEFAULT_ENTRY_THRESHOLD = 0.5     # |composite| > 0.5 to enter
DEFAULT_EXIT_THRESHOLD = 0.3      # |composite| < 0.3 to exit
DEFAULT_MAX_HOLD_DAYS = 10        # Force exit after 10 days
DEFAULT_SIGNAL_SCALE = 2.0        # Normalize composite to ±1 range
DEFAULT_MAX_CONVICTION = 0.8      # Cap conviction


class S25DFFStrategy(BaseStrategy):
    """S25: Divergence Flow Field — Information Flow Strategy.

    Parameters
    ----------
    symbols : list[str], optional
        Index names to trade. Default: ["NIFTY", "BANKNIFTY"].
    entry_threshold : float
        Minimum |composite signal| to enter a position.
    exit_threshold : float
        Close position when |composite signal| falls below this.
    max_hold_days : int
        Maximum holding period before forced exit.
    signal_scale : float
        Divisor to normalize composite signal to ±1 range.
    dff_config : DFFConfig, optional
        Configuration for the DivergenceFlowBuilder.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
        exit_threshold: float = DEFAULT_EXIT_THRESHOLD,
        max_hold_days: int = DEFAULT_MAX_HOLD_DAYS,
        signal_scale: float = DEFAULT_SIGNAL_SCALE,
        max_conviction: float = DEFAULT_MAX_CONVICTION,
        dff_config: DFFConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._symbols = symbols or SYMBOLS
        self._entry_threshold = entry_threshold
        self._exit_threshold = exit_threshold
        self._max_hold_days = max_hold_days
        self._signal_scale = signal_scale
        self._max_conviction = max_conviction
        self._builder = DivergenceFlowBuilder(config=dff_config)
        # Cache: {symbol: features_df} — rebuild on each scan day
        self._features_cache: dict[str, tuple[date, "pd.DataFrame"]] = {}

    @property
    def strategy_id(self) -> str:
        return "s25_dff"

    def warmup_days(self) -> int:
        return 30  # need ~21 days for z-scores + buffer

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        """Scan for DFF signals on date d.

        Uses participant OI data up to and including date d-1 to generate
        signals for trading on date d. This is causal: participant OI for
        date t is published EOD on date t, so at the open of t+1 it's known.

        The composite signal S(t-1) determines our position for day t:
        - S > entry_threshold -> long
        - S < -entry_threshold -> short
        - |S| < exit_threshold -> flat (or close existing position)
        """
        import pandas as pd  # lazy import to keep module-level clean

        signals: list[Signal] = []

        # Build features up to d-1 (causal: we can't use today's data)
        # Use a lookback window for feature computation
        lookback_start = (d - timedelta(days=120)).isoformat()
        prev_day = (d - timedelta(days=1)).isoformat()

        try:
            features = self._builder.build(lookback_start, prev_day, store=store)
        except Exception as e:
            logger.debug("DFF feature build failed for %s: %s", d, e)
            return signals

        if features.empty:
            return signals

        # Get the most recent feature row (this is d-1's data)
        latest = features.iloc[-1]
        composite = latest.get("dff_composite", 0.0)

        if pd.isna(composite):
            return signals

        for symbol in self._symbols:
            sig = self._scan_symbol(d, symbol, composite, latest)
            if sig is not None:
                signals.append(sig)

        return signals

    def _scan_symbol(
        self, d: date, symbol: str, composite: float, latest: "pd.Series"
    ) -> Signal | None:
        """Generate signal for a single symbol.

        State tracking per symbol:
        - position_<symbol>: {"direction": "long"/"short", "entry_date": "YYYY-MM-DD", "entry_signal": float}
        """
        import pandas as pd

        pos_key = f"position_{symbol}"
        pos = self.get_state(pos_key)

        abs_signal = abs(composite)
        direction = "long" if composite > 0 else "short"
        conviction = min(abs_signal / self._signal_scale, self._max_conviction)

        # Check for forced exit (max hold days)
        if pos is not None:
            entry_date = date.fromisoformat(pos["entry_date"])
            days_held = (d - entry_date).days
            if days_held >= self._max_hold_days:
                self.set_state(pos_key, None)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction="flat",
                    conviction=0.0,
                    instrument_type="FUT",
                    ttl_bars=0,
                    metadata={"exit_reason": "max_hold", "days_held": days_held},
                )

        # Check for exit signal
        if pos is not None and abs_signal < self._exit_threshold:
            self.set_state(pos_key, None)
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction="flat",
                conviction=0.0,
                instrument_type="FUT",
                ttl_bars=0,
                metadata={
                    "exit_reason": "signal_decay",
                    "composite": round(composite, 4),
                },
            )

        # Check for direction flip
        if pos is not None and pos["direction"] != direction and abs_signal >= self._entry_threshold:
            # Exit first, re-enter on next scan
            self.set_state(pos_key, None)
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction="flat",
                conviction=0.0,
                instrument_type="FUT",
                ttl_bars=0,
                metadata={"exit_reason": "direction_flip"},
            )

        # Already in position with same direction — hold
        if pos is not None:
            return None

        # New entry
        if abs_signal >= self._entry_threshold:
            self.set_state(pos_key, {
                "entry_date": d.isoformat(),
                "direction": direction,
                "entry_signal": round(composite, 4),
            })

            # Build metadata with DFF diagnostics
            metadata = {
                "composite": round(composite, 4),
                "d_hat": round(float(latest.get("dff_d_hat", 0)), 6),
                "r_hat": round(float(latest.get("dff_r_hat", 0)), 6),
                "z_d": round(float(latest.get("dff_z_d", 0)), 4),
                "z_r": round(float(latest.get("dff_z_r", 0)), 4),
                "interaction": round(float(latest.get("dff_interaction", 0)), 4),
                "energy": round(float(latest.get("dff_energy", 0)), 2),
                "regime": int(latest.get("dff_regime", 0)),
                "momentum": round(float(latest.get("dff_momentum", 0)), 6),
            }

            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction=direction,
                conviction=conviction,
                instrument_type="FUT",
                ttl_bars=min(self._max_hold_days, 5),  # Expected 3-5 days
                metadata=metadata,
            )

        return None


def create_strategy() -> S25DFFStrategy:
    """Factory for registry auto-discovery."""
    return S25DFFStrategy()
