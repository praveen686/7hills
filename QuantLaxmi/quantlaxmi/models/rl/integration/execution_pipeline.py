"""NEW-3: RL Optimal Execution calibrated from kite_depth + Hawkes optimal stopping.

Implements three components for production-grade execution:

1. **ExecutionCalibrator**: Estimates Almgren-Chriss market impact parameters
   (alpha_perm, beta_temp, sigma, spread, depth, fill_rate_k) from kite_depth
   L5 order book snapshots stored in DuckDB.

2. **OptimalExecutionPipeline**: Walk-forward train/evaluate loop that
   calibrates per fold, trains an OptimalExecutionAgent for 5000 episodes,
   benchmarks vs TWAP and Almgren-Chriss, and reports implementation shortfall.

3. **HawkesOptimalStopping**: Finite-horizon MDP for S5 Hawkes strategy exit
   timing.  Discretised (intensity_bin, signal_bin, days_held) state space
   solved via value_iteration from quantlaxmi.models.rl.core.dynamic_programming.

References:
    Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions"
    Bertsimas & Lo (1998), "Optimal Control of Execution Costs"
    Rao & Jelvis Ch 10.2 (Optimal Execution), Ch 5 (Backward Induction)
    Hawkes (1971), "Spectra of some self-exciting and mutually exciting
        point processes"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from quantlaxmi.data.store import MarketDataStore
except ImportError:
    MarketDataStore = None  # type: ignore[misc,assignment]

from quantlaxmi.models.rl.agents.execution_agent import OptimalExecutionAgent
from quantlaxmi.models.rl.environments.execution_env import ExecutionEnv, LOBSimulator
from quantlaxmi.models.rl.finance.optimal_execution import AlmgrenChrissSolution
from quantlaxmi.models.rl.core.dynamic_programming import value_iteration
from quantlaxmi.models.rl.core.markov_process import (
    FiniteMarkovDecisionProcess,
    NonTerminal,
)

__all__ = [
    "ExecutionCalibrator",
    "OptimalExecutionPipeline",
    "HawkesOptimalStopping",
]


# ============================================================================
# Default impact parameters (used when kite_depth data is sparse)
# ============================================================================

_DEFAULT_PARAMS: Dict[str, float] = {
    "alpha_perm": 5e-5,      # permanent impact per share (NIFTY-scale)
    "beta_temp": 2e-4,       # temporary impact per share
    "sigma": 0.0012,         # per-step price volatility (~30 bps intraday)
    "spread": 0.5,           # average bid-ask spread in index points
    "depth_mean": 500.0,     # average depth at best level (lots)
    "fill_rate_k": 1.2,      # exponential fill rate decay
}

# Minimum number of L5 snapshots required for reliable calibration
_MIN_DEPTH_ROWS = 200


# ============================================================================
# ExecutionCalibrator
# ============================================================================


@dataclass
class CalibrationResult:
    """Container for calibrated market impact parameters."""

    alpha_perm: float
    beta_temp: float
    sigma: float
    spread: float
    depth_mean: float
    fill_rate_k: float
    n_snapshots: int
    ticker: str
    start_date: str
    end_date: str
    used_defaults: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_perm": self.alpha_perm,
            "beta_temp": self.beta_temp,
            "sigma": self.sigma,
            "spread": self.spread,
            "depth_mean": self.depth_mean,
            "fill_rate_k": self.fill_rate_k,
            "n_snapshots": self.n_snapshots,
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "used_defaults": self.used_defaults,
        }


class ExecutionCalibrator:
    """Calibrate Almgren-Chriss impact parameters from kite_depth L5 data.

    Loads kite_depth parquet data (41-column schema: bid/ask price/qty levels
    1-5, last_price, volume, etc.) from the MarketDataStore and estimates:

    - **alpha_perm** (permanent impact): OLS of delta-mid on cumulative volume.
    - **beta_temp** (temporary impact): OLS of intra-snapshot price deviation
      on instantaneous trade imbalance.
    - **sigma**: per-step mid-price volatility (std of log-returns).
    - **spread**: average best-bid to best-ask.
    - **depth_mean**: average quantity at best bid + best ask levels.
    - **fill_rate_k**: exponential fill rate decay estimated from depth profile.

    Falls back to ``_DEFAULT_PARAMS`` when data is insufficient.
    """

    def __init__(self, store: Any) -> None:
        """
        Parameters
        ----------
        store : MarketDataStore
            Data store with sql() method for DuckDB queries.
        """
        self._store = store

    def estimate_impact_params(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> CalibrationResult:
        """Calibrate impact parameters for a specific ticker and date range.

        Parameters
        ----------
        ticker : str
            Symbol key in kite_depth, e.g. "NIFTY_FUT", "BANKNIFTY_FUT".
        start_date : str
            Start date inclusive, "YYYY-MM-DD".
        end_date : str
            End date inclusive, "YYYY-MM-DD".

        Returns
        -------
        CalibrationResult with all parameters and metadata.
        """
        df = self._load_depth_data(ticker, start_date, end_date)

        if df is None or len(df) < _MIN_DEPTH_ROWS:
            logger.warning(
                "Insufficient kite_depth data for %s (%s to %s): %d rows. "
                "Using default parameters.",
                ticker,
                start_date,
                end_date,
                0 if df is None else len(df),
            )
            return CalibrationResult(
                **_DEFAULT_PARAMS,
                n_snapshots=0 if df is None else len(df),
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                used_defaults=True,
            )

        spread = self._estimate_spread(df)
        depth_mean = self._estimate_depth(df)
        sigma = self._estimate_volatility(df)
        alpha_perm = self._estimate_permanent_impact(df)
        beta_temp = self._estimate_temporary_impact(df)
        fill_rate_k = self._estimate_fill_rate(df)

        return CalibrationResult(
            alpha_perm=alpha_perm,
            beta_temp=beta_temp,
            sigma=sigma,
            spread=spread,
            depth_mean=depth_mean,
            fill_rate_k=fill_rate_k,
            n_snapshots=len(df),
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            used_defaults=False,
        )

    def _load_depth_data(
        self, ticker: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Load kite_depth L5 data from DuckDB via store.sql().

        The kite_depth parquet files are stored under zerodha/5level/{date}/
        with the DEPTH_SCHEMA (41 columns).  We query for a specific symbol
        in the given date range.

        Attempts two strategies:
        1. Query the registered 'kite_depth' view if the store has it.
        2. Direct read_parquet from the default zerodha/5level/ path.

        Returns None if no data found.
        """
        try:
            # Strategy 1: try querying via kite_depth view or direct parquet
            # The kite_depth collector stores files at:
            #   data/zerodha/5level/{date}/{SYMBOL}.parquet
            # We try a direct glob read if the view doesn't exist.
            query = """
                SELECT
                    timestamp_ms,
                    last_price,
                    volume,
                    total_buy_qty,
                    total_sell_qty,
                    bid_price_1, bid_qty_1,
                    bid_price_2, bid_qty_2,
                    bid_price_3, bid_qty_3,
                    bid_price_4, bid_qty_4,
                    bid_price_5, bid_qty_5,
                    ask_price_1, ask_qty_1,
                    ask_price_2, ask_qty_2,
                    ask_price_3, ask_qty_3,
                    ask_price_4, ask_qty_4,
                    ask_price_5, ask_qty_5
                FROM read_parquet(
                    'data/zerodha/5level/*/{}*.parquet',
                    hive_partitioning=false,
                    union_by_name=true
                )
                WHERE symbol = ?
                ORDER BY timestamp_ms
            """.format(ticker)  # ticker in glob, also filtered in WHERE
            df = self._store.sql(query, [ticker])
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            logger.debug("Direct parquet read failed: %s", e)

        try:
            # Strategy 2: try a kite_depth view if registered
            query = """
                SELECT
                    timestamp_ms,
                    last_price,
                    volume,
                    total_buy_qty,
                    total_sell_qty,
                    bid_price_1, bid_qty_1,
                    bid_price_2, bid_qty_2,
                    bid_price_3, bid_qty_3,
                    bid_price_4, bid_qty_4,
                    bid_price_5, bid_qty_5,
                    ask_price_1, ask_qty_1,
                    ask_price_2, ask_qty_2,
                    ask_price_3, ask_qty_3,
                    ask_price_4, ask_qty_4,
                    ask_price_5, ask_qty_5
                FROM kite_depth
                WHERE symbol = ?
                ORDER BY timestamp_ms
            """
            df = self._store.sql(query, [ticker])
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            logger.debug("kite_depth view query failed: %s", e)

        return None

    def _estimate_spread(self, df: pd.DataFrame) -> float:
        """Average bid-ask spread from L1 prices."""
        spreads = df["ask_price_1"] - df["bid_price_1"]
        # Filter out zero/negative spreads (data quality)
        valid = spreads[spreads > 0.0]
        if len(valid) < 10:
            return _DEFAULT_PARAMS["spread"]
        return float(valid.median())

    def _estimate_depth(self, df: pd.DataFrame) -> float:
        """Average depth at best bid + best ask (L1 quantity)."""
        depth = df["bid_qty_1"] + df["ask_qty_1"]
        valid = depth[depth > 0]
        if len(valid) < 10:
            return _DEFAULT_PARAMS["depth_mean"]
        return float(valid.median())

    def _estimate_volatility(self, df: pd.DataFrame) -> float:
        """Per-step mid-price volatility (std of log-returns).

        Uses mid = (bid_1 + ask_1) / 2 to compute log-returns
        between consecutive snapshots.
        """
        mid = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
        mid = mid[mid > 0.0]
        if len(mid) < 20:
            return _DEFAULT_PARAMS["sigma"]
        log_ret = np.log(mid.values[1:] / mid.values[:-1])
        # Filter extreme returns (data gaps / market open/close)
        log_ret = log_ret[np.abs(log_ret) < 0.01]
        if len(log_ret) < 10:
            return _DEFAULT_PARAMS["sigma"]
        return float(np.std(log_ret, ddof=1))

    def _estimate_permanent_impact(self, df: pd.DataFrame) -> float:
        """Estimate permanent impact alpha via OLS: delta_mid ~ alpha * cum_volume.

        Permanent impact is the irreversible price shift caused by informed
        order flow.  We regress the change in mid-price on the net cumulative
        buy-sell imbalance (proxy for informed flow).

        delta_mid_k = alpha * net_flow_k + noise
        where net_flow_k = total_buy_qty_k - total_sell_qty_k
        """
        mid = (df["bid_price_1"].values + df["ask_price_1"].values) / 2.0
        valid_mask = mid > 0.0
        mid = mid[valid_mask]

        buy_qty = df["total_buy_qty"].values[valid_mask].astype(np.float64)
        sell_qty = df["total_sell_qty"].values[valid_mask].astype(np.float64)

        if len(mid) < 50:
            return _DEFAULT_PARAMS["alpha_perm"]

        delta_mid = np.diff(mid)
        net_flow = np.diff(buy_qty - sell_qty)

        # Remove zero-flow observations
        nonzero = np.abs(net_flow) > 1e-6
        delta_mid = delta_mid[nonzero]
        net_flow = net_flow[nonzero]

        if len(net_flow) < 30:
            return _DEFAULT_PARAMS["alpha_perm"]

        # OLS: delta_mid = alpha * net_flow
        # alpha = (X^T X)^{-1} X^T y
        xtx = float(np.dot(net_flow, net_flow))
        if xtx < 1e-12:
            return _DEFAULT_PARAMS["alpha_perm"]
        xty = float(np.dot(net_flow, delta_mid))
        alpha = xty / xtx

        # Permanent impact should be positive (buying pushes price up)
        # and small.  Clamp to reasonable range.
        alpha = max(1e-8, min(abs(alpha), 1e-2))
        return alpha

    def _estimate_temporary_impact(self, df: pd.DataFrame) -> float:
        """Estimate temporary impact beta via volume-weighted price deviation.

        Temporary impact = how much the execution price deviates from mid
        per unit of trade size.  We use the L5 order book to compute
        the volume-weighted average fill price for a hypothetical market
        order and regress the deviation on trade size.

        For each snapshot, simulate filling the best-ask side:
          vwap = sum(ask_price_k * min(qty_k, remaining)) / filled
          deviation = vwap - mid
        Regress deviation on filled quantity: beta = deviation / qty.
        """
        deviations: list[float] = []
        sizes: list[float] = []

        ask_prices = np.column_stack([
            df[f"ask_price_{i}"].values for i in range(1, 6)
        ])
        ask_qtys = np.column_stack([
            df[f"ask_qty_{i}"].values for i in range(1, 6)
        ])
        bid_prices = np.column_stack([
            df[f"bid_price_{i}"].values for i in range(1, 6)
        ])

        n = len(df)
        for row_idx in range(n):
            mid = (bid_prices[row_idx, 0] + ask_prices[row_idx, 0]) / 2.0
            if mid <= 0:
                continue

            # Simulate eating through the ask side
            total_cost = 0.0
            filled = 0.0
            for level in range(5):
                p = ask_prices[row_idx, level]
                q = ask_qtys[row_idx, level]
                if p <= 0 or q <= 0:
                    break
                total_cost += p * q
                filled += q

            if filled < 1.0:
                continue

            vwap = total_cost / filled
            dev = vwap - mid
            deviations.append(dev)
            sizes.append(filled)

        if len(deviations) < 30:
            return _DEFAULT_PARAMS["beta_temp"]

        dev_arr = np.array(deviations)
        size_arr = np.array(sizes)

        # OLS: deviation = beta * size
        xtx = float(np.dot(size_arr, size_arr))
        if xtx < 1e-12:
            return _DEFAULT_PARAMS["beta_temp"]
        xty = float(np.dot(size_arr, dev_arr))
        beta = xty / xtx

        # Beta should be positive and reasonable
        beta = max(1e-8, min(abs(beta), 1e-1))
        return beta

    def _estimate_fill_rate(self, df: pd.DataFrame) -> float:
        """Estimate exponential fill rate parameter from depth profile.

        The fill probability at distance d from mid is modelled as:
            P(fill) = 1 - exp(-k * (1 - d/spread))

        We estimate k from the ratio of depth at level 1 vs level 5.
        If depth decays rapidly away from best, fill rate is high (large k).
        """
        bid_q1 = df["bid_qty_1"].values.astype(np.float64)
        bid_q5 = df["bid_qty_5"].values.astype(np.float64)
        ask_q1 = df["ask_qty_1"].values.astype(np.float64)
        ask_q5 = df["ask_qty_5"].values.astype(np.float64)

        # Average depth at level 1 and level 5
        d1 = float(np.median(bid_q1[bid_q1 > 0])) if np.any(bid_q1 > 0) else 100.0
        d1 += float(np.median(ask_q1[ask_q1 > 0])) if np.any(ask_q1 > 0) else 100.0
        d1 /= 2.0

        d5 = float(np.median(bid_q5[bid_q5 > 0])) if np.any(bid_q5 > 0) else 100.0
        d5 += float(np.median(ask_q5[ask_q5 > 0])) if np.any(ask_q5 > 0) else 100.0
        d5 /= 2.0

        if d1 < 1.0 or d5 < 1.0:
            return _DEFAULT_PARAMS["fill_rate_k"]

        # depth_ratio ~ exp(-k * level_distance / num_levels)
        # d5/d1 ~ exp(-k * 4/5) => k = -5/4 * ln(d5/d1)
        ratio = d5 / d1
        if ratio <= 0 or ratio >= 1.0:
            # Depth doesn't decay or is inverted — use default
            return _DEFAULT_PARAMS["fill_rate_k"]

        k = -5.0 / 4.0 * math.log(ratio)
        k = max(0.1, min(k, 10.0))
        return k


# ============================================================================
# OptimalExecutionPipeline
# ============================================================================


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""

    fold_idx: int
    calibration: CalibrationResult
    rl_shortfall: float
    twap_shortfall: float
    ac_shortfall: float
    improvement_vs_twap_pct: float
    improvement_vs_ac_pct: float
    train_episodes: int
    avg_train_reward: float


@dataclass
class PipelineResult:
    """Aggregate results from the walk-forward execution pipeline."""

    folds: List[FoldResult]
    avg_rl_shortfall: float
    avg_twap_shortfall: float
    avg_ac_shortfall: float
    avg_improvement_vs_twap_pct: float
    avg_improvement_vs_ac_pct: float
    sharpe_rl: float
    sharpe_twap: float


class OptimalExecutionPipeline:
    """Walk-forward RL execution pipeline calibrated from kite_depth.

    For each fold:
      1. Calibrate impact parameters from kite_depth data in the train window.
      2. Create an ExecutionEnv with calibrated parameters.
      3. Train an OptimalExecutionAgent for ``train_episodes`` episodes.
      4. Evaluate OOS: compare RL vs TWAP vs Almgren-Chriss.

    Walk-forward windows: train_days calibration, then eval_days OOS.

    Parameters
    ----------
    store : MarketDataStore
        Data store for kite_depth queries.
    ticker : str
        Futures symbol for calibration (e.g. "NIFTY_FUT").
    total_shares : int
        Parent order size.
    num_steps : int
        Execution time horizon (number of intervals).
    train_episodes : int
        RL training episodes per fold.
    eval_episodes : int
        Benchmark evaluation episodes per fold.
    risk_aversion : float
        Almgren-Chriss risk aversion lambda.
    """

    def __init__(
        self,
        store: Any,
        ticker: str = "NIFTY_FUT",
        total_shares: int = 500,
        num_steps: int = 50,
        train_episodes: int = 5000,
        eval_episodes: int = 200,
        risk_aversion: float = 1e-6,
    ) -> None:
        self._store = store
        self._ticker = ticker
        self._total_shares = total_shares
        self._num_steps = num_steps
        self._train_episodes = train_episodes
        self._eval_episodes = eval_episodes
        self._risk_aversion = risk_aversion
        self._calibrator = ExecutionCalibrator(store)

    def run(
        self,
        dates: Sequence[str],
        train_days: int = 20,
        eval_days: int = 5,
        step_days: int = 5,
    ) -> PipelineResult:
        """Run walk-forward execution pipeline.

        Parameters
        ----------
        dates : Sequence[str]
            Sorted list of trading dates (YYYY-MM-DD) covering the full period.
        train_days : int
            Number of dates in each calibration window.
        eval_days : int
            Number of dates in each evaluation window.
        step_days : int
            Step size between fold starts.

        Returns
        -------
        PipelineResult with per-fold and aggregate metrics.
        """
        dates_list = list(dates)
        n_dates = len(dates_list)

        folds: List[FoldResult] = []
        fold_idx = 0

        start = 0
        while start + train_days + eval_days <= n_dates:
            train_start = dates_list[start]
            train_end = dates_list[start + train_days - 1]
            eval_start = dates_list[start + train_days]
            eval_end_idx = min(start + train_days + eval_days - 1, n_dates - 1)
            eval_end = dates_list[eval_end_idx]

            logger.info(
                "Fold %d: calibrate [%s, %s], evaluate [%s, %s]",
                fold_idx, train_start, train_end, eval_start, eval_end,
            )

            # 1. Calibrate
            cal = self._calibrator.estimate_impact_params(
                self._ticker, train_start, train_end
            )

            # 2. Create env with calibrated params
            env = ExecutionEnv(
                total_shares=self._total_shares,
                num_steps=self._num_steps,
                price_init=100.0,
                sigma=cal.sigma,
                spread=cal.spread,
                depth_mean=cal.depth_mean,
                alpha_perm=cal.alpha_perm,
                beta_temp=cal.beta_temp,
                side="sell",
            )

            # 3. Train RL agent
            agent = OptimalExecutionAgent(
                instrument=self._ticker.split("_")[0],
                algo="actor_critic",
                hidden_layers=(128, 64),
                learning_rate=3e-4,
                risk_aversion=self._risk_aversion,
            )
            train_result = agent.train(env, num_episodes=self._train_episodes)
            avg_train_reward = train_result.get("avg_reward", 0.0)

            # 4. Evaluate: RL vs TWAP
            bench_result = agent.benchmark_vs_twap(
                env, num_episodes=self._eval_episodes
            )
            rl_shortfall = bench_result["rl_shortfall"]
            twap_shortfall = bench_result["twap_shortfall"]
            improvement_twap = bench_result["improvement_pct"]

            # 5. Evaluate: Almgren-Chriss analytical benchmark
            ac = AlmgrenChrissSolution(
                total_shares=self._total_shares,
                num_steps=self._num_steps,
                sigma=cal.sigma,
                eta=cal.beta_temp,
                gamma_perm=cal.alpha_perm,
                risk_aversion=self._risk_aversion,
            )
            ac_expected_cost = ac.expected_cost()
            improvement_ac = 0.0
            if abs(ac_expected_cost) > 1e-10:
                improvement_ac = (
                    (ac_expected_cost - rl_shortfall)
                    / abs(ac_expected_cost)
                    * 100.0
                )

            fold = FoldResult(
                fold_idx=fold_idx,
                calibration=cal,
                rl_shortfall=rl_shortfall,
                twap_shortfall=twap_shortfall,
                ac_shortfall=ac_expected_cost,
                improvement_vs_twap_pct=improvement_twap,
                improvement_vs_ac_pct=improvement_ac,
                train_episodes=self._train_episodes,
                avg_train_reward=avg_train_reward,
            )
            folds.append(fold)
            logger.info(
                "Fold %d: RL=%.4f  TWAP=%.4f  AC=%.4f  "
                "improv_TWAP=%.1f%%  improv_AC=%.1f%%",
                fold_idx,
                rl_shortfall,
                twap_shortfall,
                ac_expected_cost,
                improvement_twap,
                improvement_ac,
            )

            fold_idx += 1
            start += step_days

        if not folds:
            logger.warning("No folds generated — insufficient dates.")
            return PipelineResult(
                folds=[],
                avg_rl_shortfall=0.0,
                avg_twap_shortfall=0.0,
                avg_ac_shortfall=0.0,
                avg_improvement_vs_twap_pct=0.0,
                avg_improvement_vs_ac_pct=0.0,
                sharpe_rl=0.0,
                sharpe_twap=0.0,
            )

        # Aggregate
        rl_costs = np.array([f.rl_shortfall for f in folds])
        twap_costs = np.array([f.twap_shortfall for f in folds])
        ac_costs = np.array([f.ac_shortfall for f in folds])

        avg_rl = float(np.mean(rl_costs))
        avg_twap = float(np.mean(twap_costs))
        avg_ac = float(np.mean(ac_costs))
        avg_imp_twap = float(np.mean([f.improvement_vs_twap_pct for f in folds]))
        avg_imp_ac = float(np.mean([f.improvement_vs_ac_pct for f in folds]))

        # Sharpe of cost savings: daily cost saving = twap_cost - rl_cost
        savings = twap_costs - rl_costs
        sharpe_rl = 0.0
        if len(savings) > 1:
            s_mean = float(np.mean(savings))
            s_std = float(np.std(savings, ddof=1))
            if s_std > 1e-12:
                sharpe_rl = s_mean / s_std * math.sqrt(252)

        # TWAP Sharpe (baseline: TWAP savings over worst-case = AC)
        twap_savings = ac_costs - twap_costs
        sharpe_twap = 0.0
        if len(twap_savings) > 1:
            t_mean = float(np.mean(twap_savings))
            t_std = float(np.std(twap_savings, ddof=1))
            if t_std > 1e-12:
                sharpe_twap = t_mean / t_std * math.sqrt(252)

        return PipelineResult(
            folds=folds,
            avg_rl_shortfall=avg_rl,
            avg_twap_shortfall=avg_twap,
            avg_ac_shortfall=avg_ac,
            avg_improvement_vs_twap_pct=avg_imp_twap,
            avg_improvement_vs_ac_pct=avg_imp_ac,
            sharpe_rl=sharpe_rl,
            sharpe_twap=sharpe_twap,
        )


# ============================================================================
# HawkesOptimalStopping — S5 strategy exit MDP
# ============================================================================


# State = (intensity_bin, signal_bin, days_held)
_HawkesState = Tuple[int, int, int]
_HawkesAction = str  # "hold" or "exit"


class HawkesOptimalStopping:
    """Finite-horizon MDP for optimal exit timing in S5 Hawkes strategy.

    The S5 Hawkes microstructure strategy generates entry signals from
    GEX+OI+IV+basis+PCR.  Once entered, the key decision is **when to
    exit** to maximise risk-adjusted PnL.  This module solves the exit
    MDP exactly via value iteration.

    State space (discretised):
        - **intensity_bin** (0..n_intensity-1): current Hawkes process
          intensity level.  High intensity = clustered events (regime
          momentum).  Intensity decays exponentially: lambda_t+1 =
          mu + (lambda_t - mu) * exp(-beta).
        - **signal_bin** (0..n_signal-1): current signal strength from
          the S5 combined score (GEX+OI+IV+basis+PCR).
        - **days_held** (0..max_days-1): number of days position has
          been held.

    Actions: {"hold", "exit"}

    Transitions (for "hold"):
        - intensity decays: bin moves towards zero (exponential decay).
        - signal strength may persist or mean-revert (Markov chain on bins).
        - days_held increments by 1.
        - At max_days, forced exit.

    Reward:
        - "hold": 0 (no realised PnL)
        - "exit": expected PnL proportional to signal_strength * direction,
          penalised by holding cost (time decay, opportunity cost).

    Solved via value_iteration from quantlaxmi.models.rl.core.dynamic_programming
    using FiniteMarkovDecisionProcess.

    Parameters
    ----------
    n_intensity_bins : int
        Number of discretised intensity levels.
    n_signal_bins : int
        Number of discretised signal strength levels.
    max_days : int
        Maximum holding period (forced exit at this horizon).
    hawkes_mu : float
        Base intensity of the Hawkes process.
    hawkes_beta : float
        Decay rate of the Hawkes process.
    signal_persistence : float
        Probability that signal stays in the same bin (AR-like).
    exit_reward_scale : float
        Scaling factor for exit reward.
    holding_cost_per_day : float
        Daily cost of holding (spread, financing, opportunity cost).
    gamma : float
        Discount factor for the MDP.
    """

    def __init__(
        self,
        n_intensity_bins: int = 10,
        n_signal_bins: int = 5,
        max_days: int = 20,
        hawkes_mu: float = 0.3,
        hawkes_beta: float = 0.5,
        signal_persistence: float = 0.6,
        exit_reward_scale: float = 1.0,
        holding_cost_per_day: float = 0.002,
        gamma: float = 0.99,
    ) -> None:
        self.n_intensity_bins = n_intensity_bins
        self.n_signal_bins = n_signal_bins
        self.max_days = max_days
        self.hawkes_mu = hawkes_mu
        self.hawkes_beta = hawkes_beta
        self.signal_persistence = signal_persistence
        self.exit_reward_scale = exit_reward_scale
        self.holding_cost_per_day = holding_cost_per_day
        self.gamma = gamma

        # Bin edges for intensity: [0, max_intensity] linearly
        self._max_intensity = 2.0  # max lambda for discretisation
        self._intensity_edges = np.linspace(
            0.0, self._max_intensity, n_intensity_bins + 1
        )
        self._intensity_centers = (
            self._intensity_edges[:-1] + self._intensity_edges[1:]
        ) / 2.0

        # Bin edges for signal: [-1, 1] linearly (combined S5 score range)
        self._signal_edges = np.linspace(-1.0, 1.0, n_signal_bins + 1)
        self._signal_centers = (
            self._signal_edges[:-1] + self._signal_edges[1:]
        ) / 2.0

        # Solve the MDP
        self._value_fn: Optional[Dict[NonTerminal[_HawkesState], float]] = None
        self._policy: Optional[Dict[_HawkesState, _HawkesAction]] = None
        self._solve()

    def _intensity_to_bin(self, intensity: float) -> int:
        """Map continuous intensity to bin index."""
        clamped = max(0.0, min(intensity, self._max_intensity - 1e-9))
        idx = int(
            clamped / (self._max_intensity / self.n_intensity_bins)
        )
        return min(idx, self.n_intensity_bins - 1)

    def _signal_to_bin(self, signal_strength: float) -> int:
        """Map continuous signal [-1, 1] to bin index."""
        clamped = max(-1.0, min(signal_strength, 1.0 - 1e-9))
        idx = int((clamped + 1.0) / (2.0 / self.n_signal_bins))
        return min(idx, self.n_signal_bins - 1)

    def _next_intensity_bin_probs(self, i_bin: int) -> Dict[int, float]:
        """Compute transition probabilities for intensity bin.

        Hawkes intensity decays: lambda_{t+1} = mu + (lambda_t - mu) * exp(-beta).
        We add noise by spreading probability across nearby bins.
        """
        lam_now = self._intensity_centers[i_bin]
        lam_next = self.hawkes_mu + (
            lam_now - self.hawkes_mu
        ) * math.exp(-self.hawkes_beta)

        target_bin = self._intensity_to_bin(lam_next)

        # Spread probability: 60% on target, 20% each on neighbours
        probs: Dict[int, float] = {}
        probs[target_bin] = 0.60

        if target_bin > 0:
            probs[target_bin - 1] = 0.20
        else:
            probs[target_bin] = probs.get(target_bin, 0.0) + 0.20

        if target_bin < self.n_intensity_bins - 1:
            probs[target_bin + 1] = 0.20
        else:
            probs[target_bin] = probs.get(target_bin, 0.0) + 0.20

        return probs

    def _next_signal_bin_probs(self, s_bin: int) -> Dict[int, float]:
        """Compute transition probabilities for signal bin.

        Signal has persistence (AR-1 like): stays with prob signal_persistence,
        moves to adjacent bins with remaining probability (mean-reverting
        towards center bin).
        """
        probs: Dict[int, float] = {}
        p_stay = self.signal_persistence
        p_move = 1.0 - p_stay

        center_bin = self.n_signal_bins // 2
        probs[s_bin] = p_stay

        # Mean-revert towards center
        if s_bin < center_bin:
            target = min(s_bin + 1, self.n_signal_bins - 1)
        elif s_bin > center_bin:
            target = max(s_bin - 1, 0)
        else:
            # At center: can go either way
            if self.n_signal_bins > 1:
                left = max(s_bin - 1, 0)
                right = min(s_bin + 1, self.n_signal_bins - 1)
                if left == right:
                    probs[left] = probs.get(left, 0.0) + p_move
                else:
                    probs[left] = probs.get(left, 0.0) + p_move / 2.0
                    probs[right] = probs.get(right, 0.0) + p_move / 2.0
                return probs
            else:
                return {0: 1.0}

        probs[target] = probs.get(target, 0.0) + p_move
        return probs

    def _build_transition_map(
        self,
    ) -> Mapping[_HawkesState, Mapping[_HawkesAction, Mapping[
        Tuple[_HawkesState, float], float
    ] | None]]:
        """Build the full transition map for FiniteMarkovDecisionProcess.

        States: (i_bin, s_bin, d) for i in [0..n_i-1], s in [0..n_s-1],
                d in [0..max_days-1]
        Terminal: "EXITED" state (absorbing)

        Actions at non-terminal:
          - "hold": intensity/signal transition + days_held += 1;
            if days_held reaches max_days, forced exit.
          - "exit": transition to EXITED with reward = exit_pnl.
        """
        EXITED = (-1, -1, -1)
        transition_map: Dict[
            _HawkesState,
            Dict[_HawkesAction, Dict[Tuple[_HawkesState, float], float] | None],
        ] = {}

        for i_bin in range(self.n_intensity_bins):
            for s_bin in range(self.n_signal_bins):
                for d in range(self.max_days):
                    state = (i_bin, s_bin, d)

                    signal_val = self._signal_centers[s_bin]
                    intensity_val = self._intensity_centers[i_bin]

                    # Exit reward: proportional to signal * intensity
                    # High intensity + strong signal = good exit
                    # Penalise by holding cost accumulated
                    exit_pnl = (
                        self.exit_reward_scale
                        * signal_val
                        * (0.5 + 0.5 * intensity_val / self._max_intensity)
                        - self.holding_cost_per_day * d
                    )

                    # --- "exit" action ---
                    exit_transitions: Dict[Tuple[_HawkesState, float], float] = {
                        (EXITED, exit_pnl): 1.0,
                    }

                    # --- "hold" action ---
                    hold_transitions: Dict[Tuple[_HawkesState, float], float] = {}
                    hold_reward = -self.holding_cost_per_day  # cost per day

                    next_d = d + 1

                    if next_d >= self.max_days:
                        # Forced exit at max_days
                        forced_exit_pnl = (
                            self.exit_reward_scale
                            * signal_val
                            * (0.5 + 0.5 * intensity_val / self._max_intensity)
                            - self.holding_cost_per_day * next_d
                        )
                        hold_transitions[(EXITED, forced_exit_pnl)] = 1.0
                    else:
                        # Normal transition
                        i_probs = self._next_intensity_bin_probs(i_bin)
                        s_probs = self._next_signal_bin_probs(s_bin)

                        for next_i, p_i in i_probs.items():
                            for next_s, p_s in s_probs.items():
                                next_state = (next_i, next_s, next_d)
                                prob = p_i * p_s
                                if prob > 1e-12:
                                    key = (next_state, hold_reward)
                                    hold_transitions[key] = (
                                        hold_transitions.get(key, 0.0) + prob
                                    )

                    transition_map[state] = {
                        "hold": hold_transitions,
                        "exit": exit_transitions,
                    }

        # Terminal EXITED state: no actions available
        transition_map[EXITED] = {}

        return transition_map

    def _solve(self) -> None:
        """Solve the Hawkes exit MDP via value iteration."""
        transition_map = self._build_transition_map()
        mdp = FiniteMarkovDecisionProcess(
            transition_map=transition_map,
            gamma=self.gamma,
        )

        vf, policy = value_iteration(mdp, gamma=self.gamma, tolerance=1e-6)

        # Extract policy as a simple dict
        self._policy = {}
        for nt in mdp.non_terminal_states:
            s = nt.state
            if s == (-1, -1, -1):
                continue
            action_dist = policy.act(nt)
            self._policy[s] = action_dist.sample()

        # Store value function
        self._value_fn = {nt: vf[nt] for nt in mdp.non_terminal_states}

    def get_optimal_action(
        self,
        intensity: float,
        signal_strength: float,
        days_held: int,
    ) -> str:
        """Look up the optimal action for a given state.

        Parameters
        ----------
        intensity : float
            Current Hawkes process intensity (non-negative).
        signal_strength : float
            Current S5 combined signal score in [-1, 1].
        days_held : int
            Number of days the current position has been held.

        Returns
        -------
        "hold" or "exit"
        """
        if days_held >= self.max_days:
            return "exit"

        i_bin = self._intensity_to_bin(intensity)
        s_bin = self._signal_to_bin(signal_strength)
        d = min(days_held, self.max_days - 1)

        state = (i_bin, s_bin, d)

        if self._policy is not None and state in self._policy:
            return self._policy[state]

        # Fallback: exit if signal is weak or days are high
        if abs(signal_strength) < 0.1 or days_held > self.max_days * 0.7:
            return "exit"
        return "hold"

    def get_state_value(
        self,
        intensity: float,
        signal_strength: float,
        days_held: int,
    ) -> float:
        """Return V*(state) for the given continuous state.

        Parameters
        ----------
        intensity : float
            Current Hawkes intensity.
        signal_strength : float
            S5 signal score [-1, 1].
        days_held : int
            Days position held.

        Returns
        -------
        float: optimal expected remaining value.
        """
        if days_held >= self.max_days:
            return 0.0

        i_bin = self._intensity_to_bin(intensity)
        s_bin = self._signal_to_bin(signal_strength)
        d = min(days_held, self.max_days - 1)

        state = (i_bin, s_bin, d)
        nt = NonTerminal(state)

        if self._value_fn is not None and nt in self._value_fn:
            return self._value_fn[nt]
        return 0.0

    def policy_summary(self) -> pd.DataFrame:
        """Return the full policy as a DataFrame for inspection.

        Columns: intensity_bin, signal_bin, days_held, action,
                 intensity_center, signal_center, value
        """
        if self._policy is None:
            return pd.DataFrame()

        rows: list[dict] = []
        for (i_bin, s_bin, d), action in sorted(self._policy.items()):
            if i_bin < 0:
                continue  # skip EXITED state
            nt = NonTerminal((i_bin, s_bin, d))
            val = self._value_fn.get(nt, 0.0) if self._value_fn else 0.0
            rows.append({
                "intensity_bin": i_bin,
                "signal_bin": s_bin,
                "days_held": d,
                "action": action,
                "intensity_center": self._intensity_centers[i_bin],
                "signal_center": self._signal_centers[s_bin],
                "value": val,
            })

        return pd.DataFrame(rows)

    def exit_boundary(self) -> Dict[int, Dict[int, int]]:
        """Compute the exit boundary: earliest day to exit for each (i_bin, s_bin).

        Returns
        -------
        Dict mapping intensity_bin -> {signal_bin -> first_exit_day}.
        If the policy always holds until max_days, returns max_days.
        """
        if self._policy is None:
            return {}

        boundary: Dict[int, Dict[int, int]] = {}
        for i_bin in range(self.n_intensity_bins):
            boundary[i_bin] = {}
            for s_bin in range(self.n_signal_bins):
                first_exit = self.max_days
                for d in range(self.max_days):
                    state = (i_bin, s_bin, d)
                    if self._policy.get(state) == "exit":
                        first_exit = d
                        break
                boundary[i_bin][s_bin] = first_exit

        return boundary
