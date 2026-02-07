"""S4 IV Mean-Reversion Strategy wrapper.

Wraps iv_mean_revert.py into the unified StrategyProtocol.

Edge: ATM IV spikes above rolling percentile → long futures (fear = bounce).
Sharpe: 3.07
"""

from __future__ import annotations

import logging
import math
from datetime import date

from qlx.data.store import MarketDataStore
from qlx.strategy.base import BaseStrategy
from qlx.strategy.protocol import Signal

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK = 30
DEFAULT_ENTRY_PCTILE = 0.80
DEFAULT_EXIT_PCTILE = 0.50
DEFAULT_HOLD_DAYS = 5
SYMBOLS = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]


class S4IVMeanRevertStrategy(BaseStrategy):
    """S4: IV Mean-Reversion via SANOS ATM IV percentile.

    On each scan day, calibrates SANOS for each index, computes ATM IV,
    and emits long signal when IV exceeds rolling percentile threshold.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        iv_lookback: int = DEFAULT_LOOKBACK,
        entry_pctile: float = DEFAULT_ENTRY_PCTILE,
        exit_pctile: float = DEFAULT_EXIT_PCTILE,
        hold_days: int = DEFAULT_HOLD_DAYS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._symbols = symbols or SYMBOLS
        self._iv_lookback = iv_lookback
        self._entry_pctile = entry_pctile
        self._exit_pctile = exit_pctile
        self._hold_days = hold_days

    @property
    def strategy_id(self) -> str:
        return "s4_iv_mr"

    def warmup_days(self) -> int:
        return self._iv_lookback + 5

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        signals: list[Signal] = []

        for symbol in self._symbols:
            try:
                sig = self._scan_symbol(d, store, symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.debug("S4 scan failed for %s %s: %s", symbol, d, e)

        return signals

    def _scan_symbol(self, d: date, store: MarketDataStore, symbol: str) -> Signal | None:
        import numpy as np
        from qlx.pricing.sanos import fit_sanos, prepare_nifty_chain
        from strategies.s9_momentum.data import get_fno

        # Calibrate SANOS for today
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
        forward = chain_data["forward"]
        atm_vars = chain_data["atm_variances"]
        atm_iv = math.sqrt(max(atm_vars[0], 1e-12))

        try:
            result = fit_sanos(
                market_strikes=chain_data["market_strikes"],
                market_calls=chain_data["market_calls"],
                market_spreads=chain_data.get("market_spreads"),
                atm_variances=atm_vars,
                expiry_labels=chain_data["expiry_labels"],
                eta=0.50,
                n_model_strikes=100,
                K_min=0.7,
                K_max=1.5,
            )
            if result.lp_success:
                atm_strike = np.array([1.0])
                exp_str = result.expiry_labels[0]
                try:
                    from datetime import datetime
                    exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    T = max((exp_dt - d).days / 365.0, 1 / 365.0)
                except Exception:
                    T = atm_vars[0] / max(atm_iv ** 2, 1e-6) if atm_iv > 0 else 7 / 365.0
                iv_arr = result.iv(0, atm_strike, T)
                atm_iv = float(iv_arr[0])
        except Exception as e:
            logger.debug("SANOS failed for %s %s: %s", symbol, d, e)

        # Update IV history
        history_key = f"iv_history_{symbol}"
        history = self.get_state(history_key, [])
        history.append({
            "date": d.isoformat(),
            "atm_iv": atm_iv,
            "spot": spot,
            "forward": forward,
        })
        self.set_state(history_key, history[-200:])

        # Need enough data for percentile
        if len(history) < self._iv_lookback:
            return None

        # Compute IV percentile
        ivs = [h["atm_iv"] for h in history[-self._iv_lookback:]]
        current_iv = ivs[-1]
        pctile = sum(1 for x in ivs if x <= current_iv) / len(ivs)

        # Check existing position
        pos_key = f"position_{symbol}"
        pos = self.get_state(pos_key)

        if pos is not None:
            hold = pos.get("hold_days", 0) + 1
            pos["hold_days"] = hold

            should_exit = hold >= self._hold_days or pctile < self._exit_pctile
            if should_exit:
                self.set_state(pos_key, None)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction="flat",
                    conviction=0.0,
                    instrument_type="FUT",
                    ttl_bars=0,
                    metadata={"exit_reason": "max_hold" if hold >= self._hold_days else "iv_normalised"},
                )
            else:
                self.set_state(pos_key, pos)
                return None

        # Check entry
        if pctile >= self._entry_pctile:
            # Signal strength sizing (same as paper_state.py)
            if pctile <= self._entry_pctile:
                size_weight = 0.25
            else:
                size_weight = min(
                    1.0,
                    0.25 + 0.75 * (pctile - self._entry_pctile) / (1.0 - self._entry_pctile),
                )

            self.set_state(pos_key, {
                "entry_date": d.isoformat(),
                "entry_spot": spot,
                "entry_iv": atm_iv,
                "iv_pctile": pctile,
                "hold_days": 0,
            })

            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction="long",
                conviction=size_weight,
                instrument_type="FUT",
                ttl_bars=self._hold_days,
                metadata={
                    "atm_iv": round(atm_iv, 4),
                    "iv_pctile": round(pctile, 4),
                    "spot": round(spot, 2),
                },
            )

        return None


def create_strategy() -> S4IVMeanRevertStrategy:
    """Factory for registry auto-discovery."""
    return S4IVMeanRevertStrategy()
