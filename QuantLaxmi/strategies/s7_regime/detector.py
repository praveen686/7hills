"""S7: Information-Theoretic Regime Detection.

Classifies market into TRENDING / MEAN_REVERTING / RANDOM using:
  - Shannon entropy (distribution shape) from core/features/information.py
  - Mutual Information (predictability) from core/features/information.py
  - VPIN (microstructure toxicity) from core/features/microstructure.py

These features already exist — this module is a coordinator that
computes them and classifies the regime for downstream sub-strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.features.information import price_entropy, mutual_information

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    RANDOM = "random"


@dataclass(frozen=True)
class RegimeObservation:
    """Single regime classification observation."""

    entropy: float            # [0, 1] — 1=random, 0=predictable
    mutual_info: float        # nats — higher = more predictable
    vpin: float               # [0, 1] — higher = more toxic flow
    regime: MarketRegime
    confidence: float         # [0, 1]


# Thresholds (calibrated from backtest data)
ENTROPY_HIGH = 0.85     # above this = random
ENTROPY_LOW = 0.65      # below this = trending/structured
MI_HIGH = 0.10          # above this = predictable (trending)
MI_LOW = 0.03           # below this = no memory (random)
VPIN_TOXIC = 0.70       # above this = universal kill


def classify_regime(
    prices: np.ndarray,
    vpin: float = 0.0,
    entropy_window: int = 100,
    word_length: int = 2,
) -> RegimeObservation:
    """Classify market regime from price history and VPIN.

    Parameters
    ----------
    prices : np.ndarray
        Recent close prices (at least entropy_window bars).
    vpin : float
        Current VPIN reading (0-1). If > VPIN_TOXIC, regime = RANDOM
        regardless of other indicators (universal kill).
    entropy_window : int
        Window for entropy and MI computation.
    word_length : int
        Binary word length for entropy computation.

    Returns
    -------
    RegimeObservation with classification and confidence.
    """
    if len(prices) < entropy_window:
        return RegimeObservation(
            entropy=0.5, mutual_info=0.0, vpin=vpin,
            regime=MarketRegime.RANDOM, confidence=0.0,
        )

    ent = price_entropy(prices, word_length=word_length, window=entropy_window)
    mi = mutual_information(prices, word_length=word_length, window=entropy_window)

    # VPIN override: toxic flow → no trades
    if vpin > VPIN_TOXIC:
        return RegimeObservation(
            entropy=ent, mutual_info=mi, vpin=vpin,
            regime=MarketRegime.RANDOM, confidence=0.9,
        )

    # Classification logic
    if ent < ENTROPY_LOW and mi > MI_HIGH:
        # Low entropy + high MI = structured, predictable = TRENDING
        conf = min(1.0, (ENTROPY_LOW - ent) / 0.2 + (mi - MI_HIGH) / 0.1)
        return RegimeObservation(
            entropy=ent, mutual_info=mi, vpin=vpin,
            regime=MarketRegime.TRENDING, confidence=min(1.0, conf),
        )

    if ent > ENTROPY_HIGH and mi < MI_LOW:
        # High entropy + low MI = random walk
        conf = min(1.0, (ent - ENTROPY_HIGH) / 0.15 + (MI_LOW - mi) / 0.05)
        return RegimeObservation(
            entropy=ent, mutual_info=mi, vpin=vpin,
            regime=MarketRegime.RANDOM, confidence=min(1.0, conf),
        )

    # Middle ground → mean-reverting
    # When entropy is moderate and MI is moderate, markets oscillate
    conf = 1.0 - abs(ent - 0.75) / 0.25
    return RegimeObservation(
        entropy=ent, mutual_info=mi, vpin=vpin,
        regime=MarketRegime.MEAN_REVERTING, confidence=max(0.3, min(1.0, conf)),
    )


def classify_regime_with_hysteresis(
    prices: np.ndarray,
    vpin: float = 0.0,
    previous_regime: MarketRegime | None = None,
    bars_in_regime: int = 0,
    min_hold: int = 3,
    entropy_window: int = 100,
    word_length: int = 2,
) -> RegimeObservation:
    """Classify regime with hysteresis to suppress spurious flips.

    If the raw classification differs from the previous regime and the
    previous regime has been held for fewer than ``min_hold`` bars, the
    flip is suppressed and the confidence is reduced by 0.5×.

    Parameters
    ----------
    prices : np.ndarray
        Recent close prices.
    vpin : float
        Current VPIN.
    previous_regime : MarketRegime or None
        Previous regime classification. If None, no hysteresis applied.
    bars_in_regime : int
        How many bars the previous regime has been held.
    min_hold : int
        Minimum bars before a flip is allowed.

    Returns
    -------
    RegimeObservation — with potentially suppressed regime and reduced confidence.
    """
    raw = classify_regime(prices, vpin, entropy_window, word_length)

    # No hysteresis if no previous regime
    if previous_regime is None:
        return raw

    # Same regime — no suppression needed
    if raw.regime == previous_regime:
        return raw

    # Different regime — check hysteresis
    if bars_in_regime < min_hold:
        # Suppress flip: keep previous regime, reduce confidence
        return RegimeObservation(
            entropy=raw.entropy,
            mutual_info=raw.mutual_info,
            vpin=raw.vpin,
            regime=previous_regime,
            confidence=raw.confidence * 0.5,
        )

    # Held long enough — allow the flip
    return raw


def reset_regime_state() -> dict:
    """Return a fresh regime state dict for tracking hysteresis.

    Returns
    -------
    dict with keys: regime, bars_in_regime, confidence.
    """
    return {
        "regime": MarketRegime.RANDOM,
        "bars_in_regime": 0,
        "confidence": 0.0,
    }


def rolling_regime(
    prices: np.ndarray,
    vpins: np.ndarray | None = None,
    window: int = 100,
    word_length: int = 2,
) -> list[RegimeObservation]:
    """Compute rolling regime classification.

    Parameters
    ----------
    prices : np.ndarray
        Full price series.
    vpins : np.ndarray or None
        VPIN series (same length as prices). None = assume 0.
    window : int
        Rolling window size.

    Returns
    -------
    List of RegimeObservation, one per bar (first `window-1` are RANDOM with 0 confidence).
    """
    n = len(prices)
    results: list[RegimeObservation] = []

    for i in range(n):
        if i < window - 1:
            results.append(RegimeObservation(
                entropy=0.5, mutual_info=0.0, vpin=0.0,
                regime=MarketRegime.RANDOM, confidence=0.0,
            ))
            continue

        window_prices = prices[i - window + 1: i + 1]
        vpin = float(vpins[i]) if vpins is not None and i < len(vpins) else 0.0

        obs = classify_regime(
            window_prices, vpin=vpin,
            entropy_window=window, word_length=word_length,
        )
        results.append(obs)

    return results
