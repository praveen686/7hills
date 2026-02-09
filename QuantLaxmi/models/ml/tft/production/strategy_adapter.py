"""TFT Strategy Adapter — wraps TFTInferencePipeline as a BaseStrategy.

Provides the StrategyProtocol interface so the TFT model can participate
in the orchestrator alongside other strategies (S1-S24).

Features:
- Lazy loading: model loads on first scan() call
- Conviction threshold: only emit signals above threshold
- Multi-asset: one signal per asset per day
- Signal metadata: raw position, confidence, model version

Usage
-----
    strategy = TFTStrategy()
    signals = strategy.scan(date(2026, 2, 6), store)

    # Or via registry
    from strategies.registry import default_registry
    default_registry.register(TFTStrategy())
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


class TFTStrategy:
    """BaseStrategy wrapper for TFT inference pipeline.

    Implements StrategyProtocol: strategy_id, scan(), warmup_days().

    Parameters
    ----------
    checkpoint_dir : str or Path, optional
        Specific checkpoint to load. If None, loads latest.
    model_type : str
        Model type for latest loading.
    conviction_threshold : float
        Minimum |position| to emit a signal. Below this, returns flat.
    base_dir : str
        Checkpoint base directory.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        model_type: str = "x_trend",
        conviction_threshold: float = 0.3,
        base_dir: str = "checkpoints",
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._model_type = model_type
        self._conviction_threshold = conviction_threshold
        self._base_dir = base_dir
        self._pipeline = None  # Lazy loaded
        self._version = 0

    @property
    def strategy_id(self) -> str:
        """Unique identifier for this strategy."""
        return f"s_tft_x_trend_v{self._version}"

    def scan(self, d: date, store: "MarketDataStore") -> list:
        """Scan a single date and return signals.

        Parameters
        ----------
        d : date
            Trading date.
        store : MarketDataStore

        Returns
        -------
        list[Signal]
        """
        from strategies.protocol import Signal

        # Lazy load pipeline
        if self._pipeline is None:
            self._load_pipeline()

        if self._pipeline is None:
            logger.warning("TFTStrategy: pipeline not available, returning empty")
            return []

        try:
            result = self._pipeline.predict(d, store)
        except Exception as e:
            logger.error("TFTStrategy prediction failed for %s: %s", d, e)
            return []

        signals = []
        for asset_name, position in result.positions.items():
            confidence = result.confidences.get(asset_name, 0.0)

            # Apply conviction threshold
            if abs(position) < self._conviction_threshold:
                continue

            direction = "long" if position > 0 else "short"
            conviction = min(abs(position) / 0.25, 1.0)  # normalize to [0, 1]

            # Map asset name to symbol
            symbol = self._asset_to_symbol(asset_name)

            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction=direction,
                conviction=conviction,
                instrument_type="FUT",
                ttl_bars=5,
                metadata={
                    "raw_position": float(position),
                    "confidence": float(confidence),
                    "model_type": self._model_type,
                    "checkpoint_version": self._version,
                },
            )
            signals.append(signal)

        logger.info(
            "TFTStrategy.scan(%s): %d signals from %d assets",
            d, len(signals), len(result.positions),
        )
        return signals

    def warmup_days(self) -> int:
        """Minimum historical days for the strategy."""
        return 120  # lookback for feature construction

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """Load the inference pipeline (lazy)."""
        try:
            from .inference import TFTInferencePipeline

            if self._checkpoint_dir:
                self._pipeline = TFTInferencePipeline.from_checkpoint(
                    self._checkpoint_dir
                )
            else:
                self._pipeline = TFTInferencePipeline.from_latest(
                    self._model_type, self._base_dir,
                )

            self._version = self._pipeline.version
            logger.info(
                "TFTStrategy loaded pipeline v%d (%d features, %d assets)",
                self._version,
                len(self._pipeline.feature_names),
                len(self._pipeline.asset_names),
            )
        except FileNotFoundError:
            logger.warning(
                "TFTStrategy: no checkpoint found for %s in %s",
                self._model_type, self._base_dir,
            )
            self._pipeline = None
        except Exception as e:
            logger.error("TFTStrategy: failed to load pipeline: %s", e)
            self._pipeline = None

    @staticmethod
    def _asset_to_symbol(asset_name: str) -> str:
        """Map asset name to trading symbol."""
        mapping = {
            "NIFTY": "NIFTY",
            "BANKNIFTY": "BANKNIFTY",
            "FINNIFTY": "FINNIFTY",
            "MIDCPNIFTY": "MIDCPNIFTY",
        }
        return mapping.get(asset_name.upper(), asset_name.upper())


def create_strategy() -> TFTStrategy:
    """Factory for StrategyRegistry auto-discovery."""
    return TFTStrategy()
