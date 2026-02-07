"""Combined microstructure signal generator.

Takes analytics from each snapshot and produces a trade signal:
  direction (long/short/flat), conviction (0-1), and reasoning.

Signal weights:
  - GEX regime:      30%  (strongest edge — dealer mechanics are physical)
  - OI delta flow:    25%  (institutional positioning)
  - IV term struct:   20%  (panic/complacency)
  - Basis:            15%  (leverage positioning)
  - PCR:              10%  (sentiment extreme)

Trade rules:
  - Only trade when EMA-smoothed score > threshold (avoid noise)
  - Require 2 consecutive raw signals in same direction before entry
  - GEX regime determines HOW we trade, not just direction:
    - mean_revert regime: fade moves (sell rallies, buy dips)
    - momentum regime: follow breakouts
  - Max pain used as a target, not a signal

IMPORTANT: Module-level EMA state persists across calls. Call
``reset_signal_state()`` between backtest runs or CV folds to
prevent state leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from strategies.s5_hawkes.analytics import MicrostructureSnapshot

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Output of the signal generator."""

    symbol: str
    direction: str          # "long", "short", or "flat"
    conviction: float       # 0.0 to 1.0
    gex_regime: str         # "mean_revert" or "momentum"
    components: dict        # Individual signal scores
    reasoning: str          # Human-readable explanation
    spot: float
    futures: float
    max_pain: float
    raw_score: float = 0.0        # Raw combined score (before EMA)
    smoothed_score: float = 0.0   # EMA-smoothed score
    consecutive: int = 0          # Consecutive scans in same direction


# ---------------------------------------------------------------------------
# EMA state per symbol (persists across calls within same process)
# ---------------------------------------------------------------------------
_ema_state: dict[str, float] = {}          # symbol → EMA of combined score
_direction_streak: dict[str, int] = {}     # symbol → consecutive scans in same dir
_last_raw_dir: dict[str, str] = {}         # symbol → last raw direction
_flat_streak: dict[str, int] = {}          # symbol → consecutive flat scans

EMA_ALPHA = 0.3          # Smoothing factor (lower = slower, less noise)
MIN_CONSECUTIVE = 2      # Require N consecutive raw signals before entry


def reset_signal_state() -> None:
    """Reset all signal EMA state (for testing or fresh start)."""
    _ema_state.clear()
    _direction_streak.clear()
    _last_raw_dir.clear()
    _flat_streak.clear()


# ---------------------------------------------------------------------------
# Individual signal scorers (each returns -1 to +1, bullish = positive)
# ---------------------------------------------------------------------------

def _score_gex(snap: MicrostructureSnapshot) -> float:
    """GEX regime score.

    In mean-revert regime (positive GEX), dealers dampen moves.
    If spot < GEX flip → expect reversion up → bullish.
    If spot > GEX flip → expect reversion down → bearish.

    In momentum regime (negative GEX), dealers amplify moves.
    Recent price direction matters more — we combine with OI flow.
    GEX itself just tells us the regime, not the direction.
    """
    gex = snap.gex
    if gex.regime == "mean_revert":
        # Spot below flip → bullish reversion expected
        dist = (gex.gex_flip_strike - snap.spot) / snap.spot
        # Clamp to [-1, 1], positive = bullish
        return max(-1.0, min(1.0, dist * 20))  # 5% distance → full signal
    else:
        # Momentum regime — direction comes from other signals, GEX just amplifies
        return 0.0


def _score_oi_flow(snap: MicrostructureSnapshot) -> float:
    """OI flow score. Uses the pre-computed score from analytics."""
    if snap.oi_flow is None:
        return 0.0
    return snap.oi_flow.score


def _score_iv_term(snap: MicrostructureSnapshot) -> float:
    """IV term structure score.

    Inverted (near >> far) → panic → contrarian bullish.
    Steep (near << far) → complacent → mild bearish.
    """
    iv = snap.iv_term
    if iv.signal == "inverted":
        # Panic: stronger inversion → stronger bullish signal
        return min(1.0, (iv.slope - 1.0) * 2)
    elif iv.signal == "steep":
        return max(-1.0, (1.0 - iv.slope) * 2)
    return 0.0


def _score_basis(snap: MicrostructureSnapshot) -> float:
    """Basis score.

    Overleveraged longs (high premium) → contrarian bearish.
    Overleveraged shorts (discount) → contrarian bullish.
    """
    b = snap.basis
    if b.signal == "overleveraged_long":
        return max(-1.0, -b.basis_zscore / 3.0)
    elif b.signal == "overleveraged_short":
        return min(1.0, -b.basis_zscore / 3.0)
    return 0.0


def _score_pcr(snap: MicrostructureSnapshot) -> float:
    """PCR score.

    Extreme fear (high PCR) → contrarian bullish.
    Extreme greed (low PCR) → contrarian bearish.
    """
    pcr = snap.pcr
    if pcr.signal == "extreme_fear":
        return min(1.0, (pcr.pcr_oi - 1.0) * 2)
    elif pcr.signal == "extreme_greed":
        return max(-1.0, (0.8 - pcr.pcr_oi) * 2)
    return 0.0


# ---------------------------------------------------------------------------
# Combined signal
# ---------------------------------------------------------------------------

# Signal weights
WEIGHTS = {
    "gex": 0.30,
    "oi_flow": 0.25,
    "iv_term": 0.20,
    "basis": 0.15,
    "pcr": 0.10,
}

# Minimum EMA-smoothed score to generate a trade signal
# Note: in momentum regime GEX=0 and IV term often=0, so max possible is ~0.35
ENTRY_THRESHOLD = 0.25
# Score below which we consider the signal "flat" (used for decay exit)
EXIT_THRESHOLD = 0.15
# Number of consecutive flat scans before triggering decay exit
FLAT_DECAY_SCANS = 5


def generate_signal(
    snap: MicrostructureSnapshot,
    entry_threshold: float = ENTRY_THRESHOLD,
) -> TradeSignal:
    """Generate a combined trade signal from a microstructure snapshot.

    Uses EMA smoothing on the combined score to filter out noise from
    volatile components (especially OI flow between 3-min snapshots).
    Requires MIN_CONSECUTIVE raw signals in the same direction before entry.

    Returns a TradeSignal with direction, conviction, and reasoning.
    """
    sym = snap.symbol
    scores = {
        "gex": _score_gex(snap),
        "oi_flow": _score_oi_flow(snap),
        "iv_term": _score_iv_term(snap),
        "basis": _score_basis(snap),
        "pcr": _score_pcr(snap),
    }

    # Weighted sum (raw)
    raw_combined = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)

    # EMA smooth (warm-start: first observation seeds the EMA)
    if sym not in _ema_state:
        smoothed = raw_combined  # no history → use raw as initial EMA
    else:
        smoothed = EMA_ALPHA * raw_combined + (1 - EMA_ALPHA) * _ema_state[sym]
    _ema_state[sym] = smoothed

    # Track direction streak (raw signal consistency)
    raw_dir = "long" if raw_combined > 0.1 else ("short" if raw_combined < -0.1 else "flat")

    if raw_dir == _last_raw_dir.get(sym):
        _direction_streak[sym] = _direction_streak.get(sym, 0) + 1
    else:
        _direction_streak[sym] = 1 if raw_dir != "flat" else 0
    _last_raw_dir[sym] = raw_dir

    # Track flat streak (for decay exits)
    if abs(smoothed) < EXIT_THRESHOLD:
        _flat_streak[sym] = _flat_streak.get(sym, 0) + 1
    else:
        _flat_streak[sym] = 0

    consecutive = _direction_streak.get(sym, 0)

    # Active signals (non-zero) for reasoning
    active = {k: v for k, v in scores.items() if abs(v) > 0.05}

    # Direction and conviction from SMOOTHED score + consecutive check
    if abs(smoothed) >= entry_threshold and consecutive >= MIN_CONSECUTIVE:
        direction = "long" if smoothed > 0 else "short"
        conviction = min(1.0, abs(smoothed) / 0.5)
    else:
        direction = "flat"
        conviction = 0.0

    # Build reasoning
    parts = []
    for k, v in sorted(active.items(), key=lambda x: abs(x[1]), reverse=True):
        side = "bull" if v > 0 else "bear"
        parts.append(f"{k}={v:+.2f}({side})")

    reasoning = (
        f"ema={smoothed:+.3f} raw={raw_combined:+.3f} streak={consecutive} "
        f"[{', '.join(parts) or 'no active signals'}] "
        f"regime={snap.gex.regime}"
    )

    return TradeSignal(
        symbol=snap.symbol,
        direction=direction,
        conviction=conviction,
        gex_regime=snap.gex.regime,
        components=scores,
        reasoning=reasoning,
        spot=snap.spot,
        futures=snap.futures,
        max_pain=snap.max_pain.max_pain_strike,
        raw_score=raw_combined,
        smoothed_score=smoothed,
        consecutive=consecutive,
    )
