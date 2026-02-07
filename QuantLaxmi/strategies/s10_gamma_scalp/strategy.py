"""S10: Gamma Scalping Strategy.

Concept: Delta-neutral long gamma. Buy ATM straddle when IV is cheap
(percentile < 0.20), delta-hedge with futures.

P&L: Gamma/2 × (dS)² − Theta×dt
Profits when realized vol > implied vol.

Entry: IV_percentile < 0.20 AND VRP < -0.02 AND DTE >= 14
Hedge: Rebalance when |delta| > 0.30, max 2 hedges/day

Reuse: iv_engine.py GPU Greeks, SANOS surface, RealizedVol from core/features/iv.py
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from core.market.store import MarketDataStore
from core.strategy.base import BaseStrategy
from core.strategy.protocol import Signal

logger = logging.getLogger(__name__)

SYMBOLS = ["NIFTY", "BANKNIFTY"]


class S10GammaScalpStrategy(BaseStrategy):
    """S10: Gamma scalping — long realized vol vs implied vol."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        iv_pctile_threshold: float = 0.20,
        vrp_threshold: float = -0.02,
        min_dte: int = 14,
        delta_rebalance: float = 0.30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._symbols = symbols or SYMBOLS
        self._iv_pctile = iv_pctile_threshold
        self._vrp_threshold = vrp_threshold
        self._min_dte = min_dte
        self._delta_rebal = delta_rebalance

    @property
    def strategy_id(self) -> str:
        return "s10_gamma_scalp"

    def warmup_days(self) -> int:
        return 60  # need enough for IV percentile and realized vol

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        signals: list[Signal] = []

        for symbol in self._symbols:
            try:
                sig = self._scan_symbol(d, store, symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.debug("S10 scan failed for %s %s: %s", symbol, d, e)

        return signals

    def _scan_symbol(self, d: date, store: MarketDataStore, symbol: str) -> Signal | None:
        from core.pricing.sanos import fit_sanos, prepare_nifty_chain
        from strategies.s9_momentum.data import get_fno

        # Load FnO data
        try:
            fno = get_fno(store, d)
            if fno.empty:
                return None
        except Exception:
            return None

        chain_data = prepare_nifty_chain(fno, symbol=symbol, max_expiries=2)
        if chain_data is None:
            return None

        spot = chain_data["spot"]
        atm_vars = chain_data["atm_variances"]
        atm_iv = math.sqrt(max(atm_vars[0], 1e-12))

        # Track IV history for percentile
        iv_key = f"iv_history_{symbol}"
        iv_history = self.get_state(iv_key, [])
        iv_history.append({"date": d.isoformat(), "atm_iv": atm_iv, "spot": spot})
        self.set_state(iv_key, iv_history[-200:])

        if len(iv_history) < 60:
            return None

        # Compute IV percentile
        ivs = [h["atm_iv"] for h in iv_history[-60:]]
        current_iv = ivs[-1]
        iv_pctile = sum(1 for x in ivs if x <= current_iv) / len(ivs)

        # Compute realized vol (20-day close-to-close)
        spots = [h["spot"] for h in iv_history[-21:]]
        if len(spots) >= 21:
            log_rets = [math.log(spots[i] / spots[i - 1]) for i in range(1, len(spots))]
            realized_vol = np.std(log_rets, ddof=1) * math.sqrt(252)
        else:
            realized_vol = atm_iv  # no edge estimate

        # VRP = realized − implied
        vrp = realized_vol - atm_iv

        # DTE check
        try:
            expiry_str = chain_data["expiry_labels"][0]
            from datetime import datetime
            exp_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            dte = (exp_dt - d).days
        except Exception:
            dte = 7

        # Check position state
        pos_key = f"position_{symbol}"
        pos = self.get_state(pos_key)

        if pos is not None:
            # In position — check exit (hold until DTE < 3 or IV percentile > 0.50)
            if dte < 3 or iv_pctile > 0.50:
                self.set_state(pos_key, None)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction="flat",
                    conviction=0.0,
                    instrument_type="SPREAD",
                    metadata={
                        "exit_reason": "dte_low" if dte < 3 else "iv_normalized",
                        "iv_pctile": round(iv_pctile, 3),
                    },
                )
            return None  # hold

        # Entry conditions
        if iv_pctile > self._iv_pctile:
            return None  # IV not cheap enough

        if vrp > self._vrp_threshold:
            return None  # VRP not favorable

        if dte < self._min_dte:
            return None  # expiry too close

        # Enter: buy ATM straddle (delta-neutral at entry)
        conviction = min(1.0, (self._iv_pctile - iv_pctile) / 0.15 * 0.8)
        conviction = max(0.3, conviction)

        atm_strike = round(spot / 50) * 50 if symbol == "NIFTY" else round(spot / 100) * 100

        self.set_state(pos_key, {
            "entry_date": d.isoformat(),
            "entry_spot": spot,
            "entry_iv": atm_iv,
            "entry_rv": realized_vol,
            "atm_strike": atm_strike,
            "dte": dte,
        })

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction="long",
            conviction=conviction,
            instrument_type="SPREAD",
            strike=atm_strike,
            expiry=chain_data["expiry_labels"][0] if chain_data["expiry_labels"] else "",
            ttl_bars=min(dte, 14),
            metadata={
                "structure": "atm_straddle",
                "iv_pctile": round(iv_pctile, 3),
                "atm_iv": round(atm_iv, 4),
                "realized_vol": round(realized_vol, 4),
                "vrp": round(vrp, 4),
                "dte": dte,
            },
        )


def create_strategy() -> S10GammaScalpStrategy:
    return S10GammaScalpStrategy()
