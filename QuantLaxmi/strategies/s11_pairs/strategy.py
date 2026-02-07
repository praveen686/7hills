"""S11: Statistical Arbitrage / Pairs Trading Strategy.

Concept: Engle-Granger cointegration on FnO stock pairs. Trade spread
mean-reversion.

Filter: ADF p < 0.05, half-life in [3, 20], Hurst < 0.45
Entry:  z-score > 2 or < -2
Exit:   |z| < 0.5
Sizing: Dollar-neutral pairs, max 3 concurrent

Note: Limited by 133-day data window. Start now, improve as data accumulates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from core.data.store import MarketDataStore
from core.strategy.base import BaseStrategy
from core.strategy.protocol import Signal

logger = logging.getLogger(__name__)

MAX_CONCURRENT_PAIRS = 3
Z_ENTRY = 2.0
Z_EXIT = 0.5
ADF_P_THRESHOLD = 0.05
HALF_LIFE_MIN = 3
HALF_LIFE_MAX = 20
HURST_MAX = 0.45


@dataclass
class PairCandidate:
    """A cointegrated pair with statistics."""

    symbol_a: str
    symbol_b: str
    hedge_ratio: float     # beta: units of B per unit of A
    adf_pvalue: float      # ADF test p-value
    half_life: float       # mean-reversion half-life in days
    hurst: float           # Hurst exponent
    current_z: float       # current spread z-score
    spread_std: float


class S11PairsStrategy(BaseStrategy):
    """S11: Statistical arbitrage / pairs trading."""

    def __init__(
        self,
        lookback: int = 60,
        rebalance_day: int = 0,  # Monday
        max_pairs: int = MAX_CONCURRENT_PAIRS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._lookback = lookback
        self._rebalance_day = rebalance_day
        self._max_pairs = max_pairs

    @property
    def strategy_id(self) -> str:
        return "s11_pairs"

    def warmup_days(self) -> int:
        return self._lookback + 10

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        signals: list[Signal] = []

        # Check existing pair positions for exit
        positions = self.get_state("pair_positions", {})
        for pair_key, pos in list(positions.items()):
            z = self._current_zscore(d, store, pos)
            if z is not None and abs(z) < Z_EXIT:
                # Mean reverted — close both legs
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=pos["symbol_a"],
                    direction="flat", conviction=0.0,
                    instrument_type="FUT",
                    metadata={"exit_reason": "spread_reverted", "z": round(z, 2)},
                ))
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=pos["symbol_b"],
                    direction="flat", conviction=0.0,
                    instrument_type="FUT",
                    metadata={"exit_reason": "spread_reverted", "z": round(z, 2)},
                ))
                del positions[pair_key]

        self.set_state("pair_positions", positions)

        # Only scan for new pairs on rebalance day
        if d.weekday() != self._rebalance_day:
            return signals

        if len(positions) >= self._max_pairs:
            return signals

        # Find cointegrated pairs
        try:
            candidates = self._find_pairs(d, store)
        except Exception as e:
            logger.debug("S11 pair finding failed for %s: %s", d, e)
            return signals

        # Enter best new pairs
        for pair in candidates:
            if len(positions) >= self._max_pairs:
                break

            pair_key = f"{pair.symbol_a}:{pair.symbol_b}"
            if pair_key in positions:
                continue

            if abs(pair.current_z) < Z_ENTRY:
                continue  # not extreme enough

            # z > 2 → spread too wide → short A, long B
            # z < -2 → spread too narrow → long A, short B
            if pair.current_z > Z_ENTRY:
                dir_a, dir_b = "short", "long"
            else:
                dir_a, dir_b = "long", "short"

            conv = min(1.0, abs(pair.current_z) / 3.0)

            signals.append(Signal(
                strategy_id=self.strategy_id,
                symbol=pair.symbol_a,
                direction=dir_a,
                conviction=conv,
                instrument_type="FUT",
                ttl_bars=int(pair.half_life * 2),
                metadata={
                    "pair": pair.symbol_b,
                    "hedge_ratio": round(pair.hedge_ratio, 4),
                    "z_score": round(pair.current_z, 2),
                    "half_life": round(pair.half_life, 1),
                    "hurst": round(pair.hurst, 3),
                },
            ))
            signals.append(Signal(
                strategy_id=self.strategy_id,
                symbol=pair.symbol_b,
                direction=dir_b,
                conviction=conv,
                instrument_type="FUT",
                ttl_bars=int(pair.half_life * 2),
                metadata={
                    "pair": pair.symbol_a,
                    "hedge_ratio": round(1.0 / pair.hedge_ratio, 4),
                    "z_score": round(pair.current_z, 2),
                },
            ))

            positions[pair_key] = {
                "symbol_a": pair.symbol_a,
                "symbol_b": pair.symbol_b,
                "hedge_ratio": pair.hedge_ratio,
                "spread_mean": float(np.mean(self._get_spread(d, store, pair))),
                "spread_std": pair.spread_std,
                "entry_date": d.isoformat(),
                "entry_z": pair.current_z,
            }

        self.set_state("pair_positions", positions)
        return signals

    def _find_pairs(self, d: date, store: MarketDataStore) -> list[PairCandidate]:
        """Find cointegrated stock pairs from FnO universe."""
        d_str = d.isoformat()
        lookback_start = (d - timedelta(days=self._lookback * 2)).isoformat()

        # Get all stock futures with enough history
        try:
            prices_df = store.sql(
                "SELECT date, \"TckrSymb\" as symbol, \"ClsPric\" as close "
                "FROM nse_fo_bhavcopy "
                "WHERE date BETWEEN ? AND ? AND \"FinInstrmTp\" = 'STF' "
                "ORDER BY date, symbol",
                [lookback_start, d_str],
            )
        except Exception:
            return []

        if prices_df.empty:
            return []

        # Pivot to get price matrix
        pivot = prices_df.pivot_table(index="date", columns="symbol", values="close")
        pivot = pivot.dropna(axis=1, thresh=self._lookback)

        if pivot.shape[1] < 2:
            return []

        symbols = list(pivot.columns)
        candidates: list[PairCandidate] = []

        # Test pairs (limit to avoid O(n²) explosion)
        max_symbols = min(30, len(symbols))
        test_symbols = symbols[:max_symbols]

        for i in range(len(test_symbols)):
            for j in range(i + 1, len(test_symbols)):
                sym_a, sym_b = test_symbols[i], test_symbols[j]
                pa = pivot[sym_a].dropna().values
                pb = pivot[sym_b].dropna().values
                n = min(len(pa), len(pb))
                if n < self._lookback:
                    continue
                pa, pb = pa[-n:], pb[-n:]

                pair = self._test_cointegration(sym_a, sym_b, pa, pb)
                if pair is not None:
                    candidates.append(pair)

        # Sort by Hurst (lower = more mean-reverting)
        candidates.sort(key=lambda p: p.hurst)
        return candidates[:10]

    def _test_cointegration(
        self, sym_a: str, sym_b: str,
        pa: np.ndarray, pb: np.ndarray,
    ) -> PairCandidate | None:
        """Test if two price series are cointegrated.

        Uses all-but-last bar for OLS hedge ratio estimation,
        then applies to full series (including current bar) for z-score.
        This avoids look-ahead bias in the regression coefficient.
        """
        n = len(pa)
        if n < 3:
            return None

        # OLS regression on historical bars (exclude current bar)
        pa_hist, pb_hist = pa[:-1], pb[:-1]
        pb_mean = np.mean(pb_hist)
        pa_mean = np.mean(pa_hist)
        cov = np.sum((pa_hist - pa_mean) * (pb_hist - pb_mean))
        var = np.sum((pb_hist - pb_mean) ** 2)
        if var < 1e-12:
            return None
        beta = cov / var

        # Compute spread using estimated beta on full series
        spread = pa - beta * pb
        # Statistics from historical spread only (exclude current bar)
        spread_hist = spread[:-1]
        spread_mean = np.mean(spread_hist)
        spread_std = np.std(spread_hist, ddof=1)
        if spread_std < 1e-8:
            return None

        # ADF test (simplified: check if spread mean-reverts)
        # Regression: d(spread) = phi * spread(-1) + noise
        ds = np.diff(spread)
        s_lag = spread[:-1]
        if len(ds) < 10:
            return None

        # OLS for phi
        phi = np.sum(ds * s_lag) / np.sum(s_lag ** 2)
        resid = ds - phi * s_lag
        se = np.sqrt(np.sum(resid ** 2) / (len(resid) - 1)) / np.sqrt(np.sum(s_lag ** 2))

        if se < 1e-12:
            return None
        t_stat = phi / se

        # ADF critical values (approximate, n≈60):
        # 1%: -3.51, 5%: -2.89, 10%: -2.58
        # Convert t-stat to approximate p-value
        if t_stat < -3.51:
            p_value = 0.01
        elif t_stat < -2.89:
            p_value = 0.05
        elif t_stat < -2.58:
            p_value = 0.10
        else:
            p_value = 0.50  # not cointegrated

        if p_value > ADF_P_THRESHOLD:
            return None

        # Half-life = -ln(2) / ln(1 + phi)
        if phi >= 0:
            return None  # not mean-reverting
        half_life = -math.log(2) / math.log(1 + phi)
        if half_life < HALF_LIFE_MIN or half_life > HALF_LIFE_MAX:
            return None

        # Hurst exponent (R/S method, simplified)
        hurst = self._hurst_exponent(spread)
        if hurst > HURST_MAX:
            return None

        current_z = (spread[-1] - spread_mean) / spread_std

        return PairCandidate(
            symbol_a=sym_a,
            symbol_b=sym_b,
            hedge_ratio=beta,
            adf_pvalue=p_value,
            half_life=half_life,
            hurst=hurst,
            current_z=current_z,
            spread_std=spread_std,
        )

    @staticmethod
    def _hurst_exponent(ts: np.ndarray) -> float:
        """Estimate Hurst exponent via R/S analysis."""
        n = len(ts)
        if n < 20:
            return 0.5

        max_k = min(n // 2, 50)
        lags = range(10, max_k)
        rs_values = []

        for lag in lags:
            chunks = [ts[i:i + lag] for i in range(0, n - lag + 1, lag)]
            rs_list = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                mean_c = np.mean(chunk)
                deviations = np.cumsum(chunk - mean_c)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(chunk, ddof=1)
                if S > 0:
                    rs_list.append(R / S)

            if rs_list:
                rs_values.append((np.log(lag), np.log(np.mean(rs_list))))

        if len(rs_values) < 3:
            return 0.5

        x = np.array([v[0] for v in rs_values])
        y = np.array([v[1] for v in rs_values])

        # OLS slope = Hurst exponent
        x_mean = np.mean(x)
        slope = np.sum((x - x_mean) * (y - np.mean(y))) / np.sum((x - x_mean) ** 2)
        return max(0.0, min(1.0, slope))

    def _current_zscore(self, d: date, store: MarketDataStore, pos: dict) -> float | None:
        """Get current z-score for an existing pair position."""
        d_str = d.isoformat()
        try:
            df = store.sql(
                "SELECT \"TckrSymb\" as symbol, \"ClsPric\" as close "
                "FROM nse_fo_bhavcopy "
                "WHERE date = ? AND \"TckrSymb\" IN (?, ?) AND \"FinInstrmTp\" = 'STF'",
                [d_str, pos["symbol_a"], pos["symbol_b"]],
            )
            if len(df) < 2:
                return None

            price_a = float(df[df["symbol"] == pos["symbol_a"]]["close"].iloc[0])
            price_b = float(df[df["symbol"] == pos["symbol_b"]]["close"].iloc[0])
            spread = price_a - pos["hedge_ratio"] * price_b
            return (spread - pos["spread_mean"]) / pos["spread_std"]
        except Exception:
            return None

    def _get_spread(self, d: date, store: MarketDataStore, pair: PairCandidate) -> np.ndarray:
        """Compute spread = price_A - beta * price_B over lookback window."""
        d_str = d.isoformat()
        lookback_start = (d - timedelta(days=self._lookback * 2)).isoformat()
        try:
            prices_df = store.sql(
                "SELECT date, \"TckrSymb\" as symbol, \"ClsPric\" as close "
                "FROM nse_fo_bhavcopy "
                "WHERE date BETWEEN ? AND ? AND \"TckrSymb\" IN (?, ?) "
                "AND \"FinInstrmTp\" = 'STF' ORDER BY date",
                [lookback_start, d_str, pair.symbol_a, pair.symbol_b],
            )
            if prices_df.empty:
                return np.array([0.0])
            pivot = prices_df.pivot_table(index="date", columns="symbol", values="close")
            pivot = pivot.dropna()
            if pivot.empty or pair.symbol_a not in pivot.columns or pair.symbol_b not in pivot.columns:
                return np.array([0.0])
            pa = pivot[pair.symbol_a].values
            pb = pivot[pair.symbol_b].values
            return pa - pair.hedge_ratio * pb
        except Exception:
            return np.array([0.0])


def create_strategy() -> S11PairsStrategy:
    return S11PairsStrategy()
