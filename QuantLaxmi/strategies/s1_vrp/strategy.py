"""S1 VRP-RNDR Strategy wrapper.

Wraps the density_strategy.py (RNDR futures) and density_options.py
(bull put spreads) into the unified StrategyProtocol.

Edge: risk-neutral density features detect overpriced crash fear.
Sharpe: 5.59 (options variant), 1.67 (futures variant).
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta

from core.market.store import MarketDataStore
from core.strategy.base import BaseStrategy
from core.strategy.protocol import Signal

logger = logging.getLogger(__name__)

# Composite signal weights (from density_strategy.py)
W_SKEW_PREMIUM = 0.40
W_LEFT_TAIL = 0.25
W_ENTROPY = 0.20
W_KL_DIRECTION = 0.15

DEFAULT_LOOKBACK = 30
DEFAULT_ENTRY_PCTILE = 0.75
DEFAULT_EXIT_PCTILE = 0.40
DEFAULT_HOLD_DAYS = 5
DEFAULT_PHYS_WINDOW = 20
SYMBOLS = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]


class S1VRPStrategy(BaseStrategy):
    """S1: Volatility Risk Premium via Risk-Neutral Density features.

    On each scan day, calibrates SANOS for each index, computes
    density features (skew premium, left tail, entropy, KL divergence),
    and emits long signals when the composite percentile exceeds threshold.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        lookback: int = DEFAULT_LOOKBACK,
        entry_pctile: float = DEFAULT_ENTRY_PCTILE,
        exit_pctile: float = DEFAULT_EXIT_PCTILE,
        hold_days: int = DEFAULT_HOLD_DAYS,
        phys_window: int = DEFAULT_PHYS_WINDOW,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._symbols = symbols or SYMBOLS
        self._lookback = lookback
        self._entry_pctile = entry_pctile
        self._exit_pctile = exit_pctile
        self._hold_days = hold_days
        self._phys_window = phys_window

    @property
    def strategy_id(self) -> str:
        return "s1_vrp"

    def warmup_days(self) -> int:
        return self._lookback + self._phys_window + 5

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        from strategies.s1_vrp.density import (
            _calibrate_density,
            compute_composite_signal,
            _rolling_percentile,
        )
        from core.pricing.risk_neutral import (
            extract_density,
            kl_divergence,
            physical_skewness,
        )

        signals: list[Signal] = []

        for symbol in self._symbols:
            try:
                sig = self._scan_symbol(d, store, symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.debug("S1 scan failed for %s %s: %s", symbol, d, e)

        return signals

    def _scan_symbol(self, d: date, store: MarketDataStore, symbol: str) -> Signal | None:
        from strategies.s1_vrp.density import _calibrate_density
        from core.pricing.risk_neutral import (
            extract_density,
            kl_divergence,
            physical_skewness,
        )
        import numpy as np

        # Load persisted history for this symbol
        history_key = f"density_history_{symbol}"
        history = self.get_state(history_key, [])

        # Calibrate today
        snap, result, spot, atm_iv = _calibrate_density(store, d, symbol)
        if snap is None or not snap.density_ok:
            return None

        # Track spots for physical skewness
        spots_key = f"spots_{symbol}"
        spots = self.get_state(spots_key, [])
        spots.append({"date": d.isoformat(), "spot": spot})

        # Physical skewness from trailing log returns
        phys_skew = 0.0
        if len(spots) > self._phys_window:
            recent = spots[-self._phys_window - 1:]
            log_rets = []
            for j in range(1, len(recent)):
                d1 = date.fromisoformat(recent[j - 1]["date"])
                d2 = date.fromisoformat(recent[j]["date"])
                gap = (d2 - d1).days
                if gap <= 4:
                    log_rets.append(math.log(recent[j]["spot"] / recent[j - 1]["spot"]))
            if len(log_rets) >= self._phys_window // 2:
                phys_skew = physical_skewness(np.array(log_rets))

        # KL divergence vs yesterday
        K, q = extract_density(result, 0)
        dK = K[1] - K[0]
        kl = 0.0
        d_entropy = 0.0
        prev_key = f"prev_density_{symbol}"
        prev = self.get_state(prev_key)
        if prev is not None and len(prev.get("q", [])) == len(q):
            prev_q = np.array(prev["q"])
            kl = kl_divergence(q, prev_q, dK)
            d_entropy = snap.entropy - prev.get("entropy", snap.entropy)

        skew_premium = phys_skew - snap.rn_skewness

        # Build today's observation
        obs = {
            "date": d.isoformat(),
            "skew_premium": skew_premium,
            "left_tail": snap.left_tail,
            "entropy_change": d_entropy,
            "kl_div": kl,
            "rn_skewness": snap.rn_skewness,
            "spot": spot,
            "atm_iv": atm_iv,
        }
        history.append(obs)

        # Save state
        self.update_state({
            history_key: history[-200:],  # keep last 200 days
            spots_key: spots[-200:],
            prev_key: {"q": q.tolist(), "entropy": snap.entropy},
        })

        # Need enough history for percentile ranking
        if len(history) < self._lookback:
            return None

        # Compute composite signal
        recent = history[-self._lookback:]
        skew_premia = [o["skew_premium"] for o in recent]
        left_tails = [o["left_tail"] for o in recent]
        entropy_changes = [o["entropy_change"] for o in recent]
        kl_divs = [o["kl_div"] for o in recent]

        # Percentile rank of today within lookback
        def _pctile(values):
            current = values[-1]
            return sum(1 for v in values if v <= current) / len(values)

        srp_pctile = _pctile(skew_premia)
        lt_pctile = _pctile(left_tails)

        # Z-scores
        def _zscore(values):
            arr = np.array(values)
            if len(arr) < 3:
                return 0.0
            mu, std = np.mean(arr), np.std(arr, ddof=1)
            return float((arr[-1] - mu) / std) if std > 1e-12 else 0.0

        ent_z = _zscore(entropy_changes)
        kl_z = _zscore(kl_divs)

        # Skew direction for KL component
        rn_skews = [o["rn_skewness"] for o in recent]
        skew_dir = -1.0 if len(rn_skews) >= 2 and rn_skews[-1] < rn_skews[-2] else 1.0
        kl_directional = kl_z * skew_dir

        composite = (
            W_SKEW_PREMIUM * (2 * srp_pctile - 1)
            + W_LEFT_TAIL * (2 * lt_pctile - 1)
            + W_ENTROPY * (-ent_z / 3.0)
            + W_KL_DIRECTION * (kl_directional / 3.0)
        )

        # Percentile-rank composite within lookback of composites
        composites_key = f"composites_{symbol}"
        composites = self.get_state(composites_key, [])
        composites.append(composite)
        self.set_state(composites_key, composites[-200:])

        if len(composites) < self._lookback:
            return None

        sig_pctile = _pctile(composites[-self._lookback:])

        # Check existing position
        pos_key = f"position_{symbol}"
        pos = self.get_state(pos_key)

        if pos is not None:
            # In a position — check exit
            hold = pos.get("hold_days", 0) + 1
            pos["hold_days"] = hold

            should_exit = hold >= self._hold_days or sig_pctile < self._exit_pctile
            if should_exit:
                self.set_state(pos_key, None)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction="flat",
                    conviction=0.0,
                    instrument_type="FUT",
                    ttl_bars=0,
                    metadata={"exit_reason": "max_hold" if hold >= self._hold_days else "signal_decay"},
                )
            else:
                self.set_state(pos_key, pos)
                return None  # hold

        # No position — check entry
        if sig_pctile >= self._entry_pctile and composite > 0:
            conviction = min(1.0, abs(composite) / 0.5)
            self.set_state(pos_key, {
                "entry_date": d.isoformat(),
                "entry_spot": spot,
                "hold_days": 0,
                "composite": composite,
            })
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction="long",
                conviction=conviction,
                instrument_type="FUT",
                ttl_bars=self._hold_days,
                metadata={
                    "composite": round(composite, 4),
                    "sig_pctile": round(sig_pctile, 4),
                    "skew_premium": round(skew_premium, 4),
                    "left_tail": round(snap.left_tail, 4),
                    "atm_iv": round(atm_iv, 4),
                },
            )

        return None


def create_strategy() -> S1VRPStrategy:
    """Factory for registry auto-discovery."""
    return S1VRPStrategy()
