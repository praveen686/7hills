"""S9: Cross-Sectional Stock FnO Momentum Strategy.

Concept: Long-short stock futures portfolio based on cross-sectional ranking:
  - Delivery spike z-score (institutional accumulation signal — India-specific)
  - OI-price concordance (long buildup vs unwinding)
  - Relative strength (multi-timeframe percentile rank)
  - Sector rotation score

Execution: Weekly rebalance, stock futures (STF), T+1, beta-neutral.

Reuse: apps/india_scanner/signals.py delivery + OI signals,
       apps/india_scanner/data.py data loaders.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.strategies.base import BaseStrategy
from quantlaxmi.strategies.protocol import Signal

logger = logging.getLogger(__name__)


@dataclass
class StockScore:
    """Cross-sectional score for a single stock."""

    symbol: str
    delivery_z: float     # delivery spike z-score
    oi_price_conc: float  # OI-price concordance score
    rel_strength: float   # relative strength percentile
    composite: float      # weighted composite
    rank: int = 0


# Signal weights
W_DELIVERY = 0.35
W_OI_PRICE = 0.25
W_REL_STRENGTH = 0.25
W_SECTOR = 0.15


class S9MomentumStrategy(BaseStrategy):
    """S9: Cross-sectional stock FnO momentum."""

    def __init__(
        self,
        top_n: int = 5,
        rebalance_day: int = 0,  # 0=Monday
        lookback: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._top_n = top_n
        self._rebalance_day = rebalance_day
        self._lookback = lookback

    @property
    def strategy_id(self) -> str:
        return "s9_momentum"

    def warmup_days(self) -> int:
        return self._lookback + 10

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        # Only rebalance on specified weekday
        if d.weekday() != self._rebalance_day:
            return []

        signals: list[Signal] = []

        try:
            scores = self._rank_stocks(d, store)
        except Exception as e:
            logger.debug("S9 ranking failed for %s: %s", d, e)
            return []

        if not scores:
            return []

        # Top N → long, bottom N → short
        sorted_scores = sorted(scores, key=lambda s: s.composite, reverse=True)
        longs = sorted_scores[:self._top_n]
        shorts = sorted_scores[-self._top_n:]

        # Close existing positions not in new set
        long_syms = {s.symbol for s in longs}
        short_syms = {s.symbol for s in shorts}
        existing = self.get_state("positions", {})

        for sym, pos in list(existing.items()):
            if pos["direction"] == "long" and sym not in long_syms:
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=sym, direction="flat", conviction=0.0,
                    instrument_type="FUT",
                    metadata={"exit_reason": "rebalance"},
                ))
                del existing[sym]
            elif pos["direction"] == "short" and sym not in short_syms:
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=sym, direction="flat", conviction=0.0,
                    instrument_type="FUT",
                    metadata={"exit_reason": "rebalance"},
                ))
                del existing[sym]

        # Open new positions
        for stock in longs:
            if stock.symbol not in existing:
                conv = min(1.0, max(0.3, abs(stock.composite)))
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=stock.symbol,
                    direction="long",
                    conviction=conv,
                    instrument_type="FUT",
                    ttl_bars=5,
                    metadata={
                        "delivery_z": round(stock.delivery_z, 2),
                        "oi_price": round(stock.oi_price_conc, 2),
                        "rel_strength": round(stock.rel_strength, 2),
                        "composite": round(stock.composite, 4),
                        "rank": stock.rank,
                    },
                ))
                existing[stock.symbol] = {"direction": "long", "entry_date": d.isoformat()}

        for stock in shorts:
            if stock.symbol not in existing:
                conv = min(1.0, max(0.3, abs(stock.composite)))
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=stock.symbol,
                    direction="short",
                    conviction=conv,
                    instrument_type="FUT",
                    ttl_bars=5,
                    metadata={
                        "delivery_z": round(stock.delivery_z, 2),
                        "oi_price": round(stock.oi_price_conc, 2),
                        "rel_strength": round(stock.rel_strength, 2),
                        "composite": round(stock.composite, 4),
                        "rank": stock.rank,
                    },
                ))
                existing[stock.symbol] = {"direction": "short", "entry_date": d.isoformat()}

        self.set_state("positions", existing)
        return signals

    def _rank_stocks(self, d: date, store: MarketDataStore) -> list[StockScore]:
        """Rank all FnO stocks on composite momentum score."""
        d_str = d.isoformat()

        # Get delivery data
        try:
            delivery = store.sql(
                "SELECT symbol, delivery_qty, traded_qty "
                "FROM nse_cm_bhavcopy WHERE date = ?",
                [d_str],
            )
        except Exception:
            delivery = None

        # Get FnO bhavcopy for OI
        try:
            fno = store.sql(
                "SELECT \"TckrSymb\" as symbol, \"OpnIntrst\" as oi, "
                "\"ClsPric\" as close, \"PrvsClsgPric\" as prev_close "
                "FROM nse_fo_bhavcopy "
                "WHERE date = ? AND \"FinInstrmTp\" = 'STF'",
                [d_str],
            )
        except Exception:
            fno = None

        if fno is None or fno.empty:
            return []

        # Get historical delivery for z-score
        lookback_start = (d - timedelta(days=self._lookback * 2)).isoformat()
        try:
            hist_delivery = store.sql(
                "SELECT date, symbol, delivery_qty, traded_qty "
                "FROM nse_cm_bhavcopy "
                "WHERE date BETWEEN ? AND ? "
                "ORDER BY date",
                [lookback_start, d_str],
            )
        except Exception:
            hist_delivery = None

        scores: list[StockScore] = []
        symbols = fno["symbol"].unique()

        for sym in symbols:
            try:
                score = self._score_stock(sym, d, fno, delivery, hist_delivery)
                if score is not None:
                    scores.append(score)
            except Exception:
                continue

        # Assign ranks
        sorted_scores = sorted(scores, key=lambda s: s.composite, reverse=True)
        for i, s in enumerate(sorted_scores):
            s.rank = i + 1

        return sorted_scores

    def _score_stock(
        self, symbol: str, d: date, fno, delivery, hist_delivery,
    ) -> StockScore | None:
        """Compute composite score for a single stock."""
        # Delivery z-score
        delivery_z = 0.0
        if delivery is not None and not delivery.empty and hist_delivery is not None:
            sym_del = delivery[delivery["symbol"].str.contains(symbol[:8], na=False)]
            if not sym_del.empty:
                today_del = float(sym_del["delivery_qty"].iloc[0]) if "delivery_qty" in sym_del.columns else 0
                today_traded = float(sym_del["traded_qty"].iloc[0]) if "traded_qty" in sym_del.columns else 1

                # Historical delivery ratio for z-score
                hist_sym = hist_delivery[hist_delivery["symbol"].str.contains(symbol[:8], na=False)]
                if not hist_sym.empty and "delivery_qty" in hist_sym.columns and "traded_qty" in hist_sym.columns:
                    hist_ratios = (hist_sym["delivery_qty"] / hist_sym["traded_qty"].replace(0, 1)).values
                    hist_ratios = hist_ratios[np.isfinite(hist_ratios)]
                    if len(hist_ratios) > 5:
                        current_ratio = today_del / max(today_traded, 1)
                        mu = np.mean(hist_ratios)
                        std = np.std(hist_ratios, ddof=1)
                        if std > 0:
                            delivery_z = (current_ratio - mu) / std

        # OI-price concordance
        oi_price_conc = 0.0
        sym_fno = fno[fno["symbol"] == symbol]
        if not sym_fno.empty:
            close_val = float(sym_fno["close"].iloc[0])
            prev_close = float(sym_fno["prev_close"].iloc[0]) if "prev_close" in sym_fno.columns else close_val
            oi_val = float(sym_fno["oi"].iloc[0]) if "oi" in sym_fno.columns else 0

            price_change = (close_val - prev_close) / prev_close if prev_close > 0 else 0
            # Positive OI + positive price = long buildup (bullish)
            # Positive OI + negative price = short buildup (bearish)
            oi_sign = 1.0 if oi_val > 0 else -1.0
            oi_price_conc = price_change * oi_sign * 100  # scale

        # Relative strength (simple: 20-day return percentile)
        rel_strength = 0.5  # default if no history
        prices_key = f"prices_{symbol}"
        prices = self.get_state(prices_key, [])
        if not sym_fno.empty:
            prices.append(float(sym_fno["close"].iloc[0]))
            self.set_state(prices_key, prices[-100:])

        if len(prices) >= self._lookback:
            ret_20d = (prices[-1] / prices[-self._lookback] - 1)
            # Cross-sectional rank will be done at portfolio level
            rel_strength = ret_20d

        composite = (
            W_DELIVERY * delivery_z
            + W_OI_PRICE * oi_price_conc
            + W_REL_STRENGTH * rel_strength * 10  # scale
            + W_SECTOR * 0.0  # sector rotation TBD
        )

        return StockScore(
            symbol=symbol,
            delivery_z=delivery_z,
            oi_price_conc=oi_price_conc,
            rel_strength=rel_strength,
            composite=composite,
        )


def create_strategy() -> S9MomentumStrategy:
    return S9MomentumStrategy()
