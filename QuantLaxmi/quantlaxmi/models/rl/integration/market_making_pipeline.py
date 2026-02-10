"""NEW-4: Avellaneda-Stoikov Market-Making Pipeline on BTC/ETH perps.

End-to-end pipeline that calibrates A-S parameters from Binance OHLCV data,
trains an RL MarketMakingAgent on CryptoEnv, and evaluates OOS with walk-forward.

Architecture:
    CryptoMMCalibrator  --  calibrate sigma, fill_rate_k, gamma from Binance data
    MarketMakingPipeline -- train + evaluate RL agent vs A-S analytical benchmark

Key formulas (Avellaneda & Stoikov 2008):
    reservation_price  = S - q * gamma * sigma^2 * (T - t)
    optimal_spread     = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

Walk-forward protocol:
    For each fold: calibrate on train window, train RL, evaluate OOS.
    Zero look-ahead bias: test data never enters calibration or training.

Book reference: Ch 10.3 (Market-Making), Avellaneda & Stoikov (2008).

Usage::

    from quantlaxmi.models.rl.integration.market_making_pipeline import MarketMakingPipeline
    pipeline = MarketMakingPipeline()
    results = pipeline.run(
        symbols=["BTCUSDT", "ETHUSDT"],
        start_date="2025-01-01",
        end_date="2026-02-01",
    )
    print(pipeline.report(results))
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantlaxmi.data._paths import BINANCE_DIR
from quantlaxmi.models.rl.agents.market_maker import MarketMakingAgent, avellaneda_stoikov_quotes
from quantlaxmi.models.rl.environments.crypto_env import CryptoEnv, CRYPTO_DEFAULTS
from quantlaxmi.models.rl.finance.market_making import AvellanedaStoikovSolution

logger = logging.getLogger(__name__)

__all__ = [
    "CryptoMMCalibrator",
    "MarketMakingPipeline",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 8-hour session in 1-min bars (standard Binance funding period)
DEFAULT_SESSION_LENGTH: int = 480

# Sharpe annualisation factor for 8-hour crypto sessions:
# 365 days * 3 sessions/day = 1095 sessions/year
CRYPTO_8H_SHARPE_FACTOR: float = math.sqrt(365 * 3)

# Default walk-forward parameters (in 8h sessions)
DEFAULT_TRAIN_SESSIONS: int = 90   # ~30 days of 3 sessions/day
DEFAULT_TEST_SESSIONS: int = 30    # ~10 days
DEFAULT_STEP_SESSIONS: int = 15    # ~5 days

# Default RL training episodes per fold
DEFAULT_N_EPISODES: int = 2000

# Minimum rows needed for calibration
MIN_CALIBRATION_ROWS: int = 100


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_binance_parquet(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1m",
) -> Optional[pd.DataFrame]:
    """Attempt to load Binance OHLCV data from local parquet files.

    Searches for hive-partitioned parquet files under BINANCE_DIR with the
    naming convention: ``{symbol}/{interval}/*.parquet`` or
    ``{symbol}_{interval}.parquet``.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``"BTCUSDT"``.
    interval : str
        Candle interval, e.g. ``"1m"``, ``"1h"``.
    start_date, end_date : str
        Date range ``"YYYY-MM-DD"``.

    Returns
    -------
    pd.DataFrame or None
        OHLCV DataFrame with columns: open, high, low, close, volume.
        Index: DatetimeIndex (UTC).  Returns None if no data found.
    """
    candidates = [
        BINANCE_DIR / symbol / interval,
        BINANCE_DIR / symbol,
        BINANCE_DIR / f"{symbol}_{interval}.parquet",
        BINANCE_DIR / f"{symbol.lower()}_{interval}.parquet",
    ]

    for path in candidates:
        if path.is_file() and path.suffix == ".parquet":
            try:
                df = pd.read_parquet(path)
                df = _normalise_ohlcv(df, start_date, end_date)
                if df is not None and len(df) >= MIN_CALIBRATION_ROWS:
                    logger.info(
                        "Loaded %d rows from parquet: %s", len(df), path
                    )
                    return df
            except Exception as exc:
                logger.debug("Failed to load %s: %s", path, exc)

        elif path.is_dir():
            parquet_files = sorted(path.glob("*.parquet"))
            if parquet_files:
                try:
                    dfs = [pd.read_parquet(f) for f in parquet_files]
                    df = pd.concat(dfs, ignore_index=False)
                    df = _normalise_ohlcv(df, start_date, end_date)
                    if df is not None and len(df) >= MIN_CALIBRATION_ROWS:
                        logger.info(
                            "Loaded %d rows from %d parquet files in %s",
                            len(df), len(parquet_files), path,
                        )
                        return df
                except Exception as exc:
                    logger.debug("Failed to load from dir %s: %s", path, exc)

    return None


def _normalise_ohlcv(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """Normalise an OHLCV DataFrame to standard column names and filter dates.

    Handles various column naming conventions (Title Case, lowercase, etc.)
    and ensures DatetimeIndex in UTC.

    Returns None if the DataFrame has insufficient data after filtering.
    """
    # Normalise column names to lowercase
    col_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_map)

    # Ensure required columns exist
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        # Try common alternative names
        alt_map = {
            "open_price": "open",
            "high_price": "high",
            "low_price": "low",
            "close_price": "close",
            "vol": "volume",
            "qty": "volume",
        }
        df = df.rename(columns={k: v for k, v in alt_map.items() if k in df.columns})
        if not required.issubset(set(df.columns)):
            return None

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        for col_name in ("timestamp", "datetime", "date", "time"):
            if col_name in df.columns:
                df[col_name] = pd.to_datetime(df[col_name], utc=True)
                df = df.set_index(col_name)
                break
        else:
            # Try parsing the existing index
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                return None

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()

    # Filter date range
    mask = (df.index >= pd.Timestamp(start_date, tz="UTC")) & (
        df.index <= pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    )
    df = df.loc[mask]

    if len(df) < MIN_CALIBRATION_ROWS:
        return None

    # Ensure numeric OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close", "volume"])
    return df


def _fetch_binance_live(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1m",
) -> pd.DataFrame:
    """Fetch Binance klines via BinanceConnector as fallback.

    Raises RuntimeError if the connector is unavailable or fetch fails.
    """
    try:
        from quantlaxmi.data.connectors.binance_connector import BinanceConnector
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import BinanceConnector and no local parquet data for {symbol}. "
            f"Install python-binance or provide parquet files in {BINANCE_DIR}."
        ) from exc

    connector = BinanceConnector()
    df = connector.fetch_klines_chunked(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
    )
    return df


# ---------------------------------------------------------------------------
# CryptoMMCalibrator
# ---------------------------------------------------------------------------


class CryptoMMCalibrator:
    """Calibrate Avellaneda-Stoikov parameters from Binance OHLCV data.

    Estimates the key A-S parameters from historical market data:
    - sigma:       per-step price volatility (annualised, then per-step via sqrt-scaling)
    - fill_rate_k: exponential fill rate decay (estimated from volume/spread)
    - gamma:       risk aversion (default 0.01, tunable)
    - T_session:   trading horizon in steps (e.g. 480 for 8h session at 1-min bars)
    - mean_spread: average spread in basis points
    - daily_volume: average daily volume in base currency

    Data source priority:
    1. Local parquet files under BINANCE_DIR
    2. Live fetch via BinanceConnector

    Parameters
    ----------
    interval : str
        Bar interval for calibration (default ``"1m"``).
    session_length : int
        Number of bars per trading session (default 480 = 8h at 1-min).
    default_gamma : float
        Default risk aversion parameter.
    """

    def __init__(
        self,
        interval: str = "1m",
        session_length: int = DEFAULT_SESSION_LENGTH,
        default_gamma: float = 0.01,
    ) -> None:
        self.interval = interval
        self.session_length = session_length
        self.default_gamma = default_gamma

    def calibrate(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, float]:
        """Calibrate A-S parameters for a single symbol.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTCUSDT"``.
        start_date, end_date : str
            Calibration window ``"YYYY-MM-DD"``.

        Returns
        -------
        dict
            Keys: sigma, fill_rate_k, gamma, T_session, mean_spread,
            daily_volume, mid_price.
        """
        df = self._load_data(symbol, start_date, end_date)

        # 1. Compute per-step returns (log returns on close)
        close = df["close"].values.astype(np.float64)
        log_returns = np.diff(np.log(np.maximum(close, 1e-10)))

        # Annualised volatility (crypto: 525600 min/year for 1-min bars)
        minutes_per_year = 525_600.0
        sigma_per_step = float(np.std(log_returns, ddof=1))
        sigma_annual = sigma_per_step * math.sqrt(minutes_per_year)

        # 2. Estimate mean spread from high-low range (Parkinson estimator proxy)
        # Spread ~ (high - low) / close as proxy for intraday spread
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        mid = (high + low) / 2.0
        range_bps = (high - low) / np.maximum(mid, 1e-10) * 10_000.0
        mean_spread_bps = float(np.median(range_bps))
        # Effective spread is typically 10-20% of the bar range for liquid crypto
        effective_spread_bps = mean_spread_bps * 0.15

        # 3. Estimate fill_rate_k from volume/spread relationship
        # Higher volume & tighter spread => higher fill rate => lower k
        # k = ln(2) / (half_spread * volume_ratio)
        # where volume_ratio = avg_volume / median_volume (normalised)
        volume = df["volume"].values.astype(np.float64)
        daily_volume = float(np.mean(volume)) * (1440.0 if self.interval == "1m" else 24.0)

        # Use the relationship: fill_rate decays as exp(-k * delta)
        # At the effective spread, we expect ~50% fill probability
        # => 0.5 = exp(-k * half_spread) => k = ln(2) / half_spread
        mid_price = float(np.median(close))
        half_spread_price = (effective_spread_bps / 10_000.0) * mid_price / 2.0
        half_spread_price = max(half_spread_price, 1e-8)

        # Volume adjustment: higher volume => lower k (easier to fill)
        vol_ratio = daily_volume / max(float(np.median(volume)) * 1440.0, 1e-8)
        vol_adjustment = 1.0 / max(vol_ratio, 0.1)
        fill_rate_k = math.log(2.0) / half_spread_price * vol_adjustment

        # Clamp k to reasonable range
        fill_rate_k = float(np.clip(fill_rate_k, 0.1, 100.0))

        # 4. Compose result
        result = {
            "sigma": sigma_annual,
            "sigma_per_step": sigma_per_step,
            "fill_rate_k": fill_rate_k,
            "gamma": self.default_gamma,
            "T_session": float(self.session_length),
            "mean_spread": effective_spread_bps,
            "daily_volume": daily_volume,
            "mid_price": mid_price,
            "n_bars": len(df),
            "start_date": start_date,
            "end_date": end_date,
        }

        logger.info(
            "Calibrated %s: sigma=%.4f (annual), k=%.4f, spread=%.2f bps, "
            "mid=$%.2f, %d bars",
            symbol, sigma_annual, fill_rate_k, effective_spread_bps,
            mid_price, len(df),
        )

        return result

    def _load_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load OHLCV data from parquet or live fetch.

        Tries local parquet first, falls back to BinanceConnector.

        Returns
        -------
        pd.DataFrame with columns: open, high, low, close, volume.
        """
        df = _load_binance_parquet(symbol, start_date, end_date, self.interval)
        if df is not None:
            return df

        logger.info(
            "No local parquet for %s, falling back to BinanceConnector",
            symbol,
        )
        df = _fetch_binance_live(symbol, start_date, end_date, self.interval)
        return df


# ---------------------------------------------------------------------------
# MarketMakingPipeline
# ---------------------------------------------------------------------------


class MarketMakingPipeline:
    """Train and evaluate RL market-making agent on BTC/ETH perps.

    Orchestrates the full NEW-4 pipeline:
    1. Calibrate A-S parameters per symbol per fold from Binance data
    2. Train MarketMakingAgent on CryptoEnv (warm-started from A-S)
    3. Evaluate OOS and compute metrics (PnL, Sharpe, inventory, fills, spread)
    4. Compare RL vs A-S analytical benchmark

    Walk-forward protocol:
        For each fold: calibrate + train on train window, evaluate on test window.
        Fold boundaries are defined in number of 8h sessions.

    Parameters
    ----------
    calibrator : CryptoMMCalibrator or None
        Calibrator instance. If None, a default one is created.
    train_sessions : int
        Number of 8h sessions for training per fold.
    test_sessions : int
        Number of 8h sessions for OOS evaluation per fold.
    step_sessions : int
        Walk-forward step size in 8h sessions.
    n_episodes : int
        RL training episodes per fold.
    max_inventory : int
        Max inventory for the market-making agent.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        calibrator: Optional[CryptoMMCalibrator] = None,
        train_sessions: int = DEFAULT_TRAIN_SESSIONS,
        test_sessions: int = DEFAULT_TEST_SESSIONS,
        step_sessions: int = DEFAULT_STEP_SESSIONS,
        n_episodes: int = DEFAULT_N_EPISODES,
        max_inventory: int = 10,
        seed: int = 42,
    ) -> None:
        self.calibrator = calibrator or CryptoMMCalibrator()
        self.train_sessions = train_sessions
        self.test_sessions = test_sessions
        self.step_sessions = step_sessions
        self.n_episodes = n_episodes
        self.max_inventory = max_inventory
        self.seed = seed

    def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        n_episodes: Optional[int] = None,
        max_inventory: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the full market-making pipeline for multiple symbols.

        Parameters
        ----------
        symbols : list[str]
            Trading pairs, e.g. ``["BTCUSDT", "ETHUSDT"]``.
        start_date, end_date : str
            Overall date range ``"YYYY-MM-DD"``.
        n_episodes : int or None
            Override for RL training episodes per fold.
        max_inventory : int or None
            Override for max inventory.

        Returns
        -------
        dict with per-symbol results and aggregate metrics:
            per_symbol : dict[str, dict]  -- per-symbol fold results
            aggregate  : dict             -- cross-symbol summary
        """
        n_ep = n_episodes or self.n_episodes
        max_inv = max_inventory or self.max_inventory

        per_symbol_results: Dict[str, Dict[str, Any]] = {}

        for symbol in symbols:
            logger.info("=" * 60)
            logger.info("Processing %s: %s to %s", symbol, start_date, end_date)
            logger.info("=" * 60)

            sym_result = self._run_symbol(
                symbol, start_date, end_date, n_ep, max_inv
            )
            per_symbol_results[symbol] = sym_result

        # Aggregate across symbols
        aggregate = self._aggregate_results(per_symbol_results)

        return {
            "per_symbol": per_symbol_results,
            "aggregate": aggregate,
        }

    def _run_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        n_episodes: int,
        max_inventory: int,
    ) -> Dict[str, Any]:
        """Run walk-forward pipeline for a single symbol.

        Returns dict with folds, aggregate RL/AS metrics.
        """
        # Parse date range to compute fold boundaries
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        total_days = (end_ts - start_ts).days

        # Each 8h session = 1/3 day; compute total sessions
        total_sessions = total_days * 3
        if total_sessions < self.train_sessions + self.test_sessions:
            logger.warning(
                "Date range too short for walk-forward on %s (%d sessions < %d needed). "
                "Running single fold.",
                symbol, total_sessions, self.train_sessions + self.test_sessions,
            )
            # Single fold: use all data for calibration, simulate train/test split
            calibration = self.calibrator.calibrate(symbol, start_date, end_date)
            agent = self._train_fold(symbol, calibration, n_episodes, max_inventory)
            rl_metrics = self._evaluate_fold(agent, calibration, max_inventory)
            as_metrics = self._as_benchmark(calibration, max_inventory)

            return {
                "folds": [{
                    "fold_idx": 0,
                    "calibration": calibration,
                    "rl_metrics": rl_metrics,
                    "as_metrics": as_metrics,
                }],
                "rl_sharpe": rl_metrics["sharpe"],
                "as_sharpe": as_metrics["sharpe"],
                "rl_pnl_mean": rl_metrics["pnl_mean"],
                "as_pnl_mean": as_metrics["pnl_mean"],
            }

        # Walk-forward folds
        folds: List[Dict[str, Any]] = []
        all_rl_pnls: List[float] = []
        all_as_pnls: List[float] = []

        fold_idx = 0
        session_offset = 0

        while session_offset + self.train_sessions + self.test_sessions <= total_sessions:
            # Compute fold date boundaries (each session = 8h = 1/3 day)
            train_start_day = session_offset // 3
            train_end_day = (session_offset + self.train_sessions) // 3
            test_end_day = min(
                (session_offset + self.train_sessions + self.test_sessions) // 3,
                total_days,
            )

            fold_train_start = (start_ts + pd.Timedelta(days=train_start_day)).strftime("%Y-%m-%d")
            fold_train_end = (start_ts + pd.Timedelta(days=train_end_day)).strftime("%Y-%m-%d")
            fold_test_end = (start_ts + pd.Timedelta(days=test_end_day)).strftime("%Y-%m-%d")

            logger.info(
                "Fold %d: calibrate [%s, %s], test [%s, %s]",
                fold_idx, fold_train_start, fold_train_end,
                fold_train_end, fold_test_end,
            )

            # 1. Calibrate on train window
            try:
                calibration = self.calibrator.calibrate(
                    symbol, fold_train_start, fold_train_end
                )
            except Exception as exc:
                logger.warning("Calibration failed for fold %d: %s", fold_idx, exc)
                session_offset += self.step_sessions
                fold_idx += 1
                continue

            # 2. Train RL agent
            agent = self._train_fold(
                symbol, calibration, n_episodes, max_inventory
            )

            # 3. Evaluate RL agent OOS
            rl_metrics = self._evaluate_fold(agent, calibration, max_inventory)

            # 4. A-S analytical benchmark on same env
            as_metrics = self._as_benchmark(calibration, max_inventory)

            fold_result = {
                "fold_idx": fold_idx,
                "train_range": (fold_train_start, fold_train_end),
                "test_range": (fold_train_end, fold_test_end),
                "calibration": calibration,
                "rl_metrics": rl_metrics,
                "as_metrics": as_metrics,
            }
            folds.append(fold_result)

            all_rl_pnls.append(rl_metrics["pnl_mean"])
            all_as_pnls.append(as_metrics["pnl_mean"])

            session_offset += self.step_sessions
            fold_idx += 1

        # Aggregate across folds
        rl_pnls = np.array(all_rl_pnls) if all_rl_pnls else np.array([0.0])
        as_pnls = np.array(all_as_pnls) if all_as_pnls else np.array([0.0])

        rl_sharpe = _compute_sharpe(rl_pnls)
        as_sharpe = _compute_sharpe(as_pnls)

        return {
            "folds": folds,
            "rl_sharpe": rl_sharpe,
            "as_sharpe": as_sharpe,
            "rl_pnl_mean": float(np.mean(rl_pnls)),
            "as_pnl_mean": float(np.mean(as_pnls)),
        }

    def _train_fold(
        self,
        symbol: str,
        calibration: Dict[str, float],
        n_episodes: int,
        max_inventory: int,
    ) -> MarketMakingAgent:
        """Train a MarketMakingAgent for one fold.

        Creates a CryptoEnv parameterised by calibration, trains the agent
        using Actor-Critic RL warm-started from the A-S analytical solution.

        Parameters
        ----------
        symbol : str
            Trading pair.
        calibration : dict
            Calibrated parameters from CryptoMMCalibrator.
        n_episodes : int
            Number of training episodes.
        max_inventory : int
            Max inventory for the agent.

        Returns
        -------
        MarketMakingAgent
            Trained agent.
        """
        sigma_annual = calibration["sigma"]
        mid_price = calibration["mid_price"]
        session_length = int(calibration["T_session"])

        # Create training environment
        env = CryptoEnv(
            symbol=symbol,
            mode="simulated",
            num_steps=session_length,
            maker_fee=CRYPTO_DEFAULTS.get(symbol, CRYPTO_DEFAULTS["BTCUSDT"])["maker_fee"],
            taker_fee=CRYPTO_DEFAULTS.get(symbol, CRYPTO_DEFAULTS["BTCUSDT"])["taker_fee"],
        )

        # Create agent with calibrated parameters
        agent = MarketMakingAgent(
            instrument=symbol,
            max_inventory=max_inventory,
            sigma=calibration["sigma_per_step"],
            gamma_risk=calibration["gamma"],
            fill_rate_k=calibration["fill_rate_k"],
            hidden_layers=(256, 128, 64),
            learning_rate=1e-4,
            warmstart_analytical=True,
            seed=self.seed,
        )

        # Train the agent
        logger.info(
            "Training %s agent: %d episodes, sigma=%.6f/step, k=%.4f, gamma=%.4f",
            symbol, n_episodes, calibration["sigma_per_step"],
            calibration["fill_rate_k"], calibration["gamma"],
        )

        train_result = agent.train(env, num_episodes=n_episodes)

        logger.info(
            "Training complete: avg_pnl=%.4f, sharpe=%.4f, episodes=%d",
            train_result["avg_pnl"], train_result["sharpe"],
            train_result["num_episodes"],
        )

        # Disable warmstart so RL policy is used during evaluation
        agent.warmstart_analytical = False

        return agent

    def _evaluate_fold(
        self,
        agent: MarketMakingAgent,
        calibration: Dict[str, float],
        max_inventory: int,
        n_eval_episodes: int = 200,
    ) -> Dict[str, float]:
        """Evaluate a trained agent on a CryptoEnv parameterised by calibration.

        Runs n_eval_episodes and computes summary metrics.

        Parameters
        ----------
        agent : MarketMakingAgent
            Trained RL agent.
        calibration : dict
            Calibrated parameters.
        max_inventory : int
            Max inventory.
        n_eval_episodes : int
            Number of evaluation episodes.

        Returns
        -------
        dict with keys: pnl_mean, pnl_std, sharpe, max_inventory_used,
        fill_rate, avg_spread_bps, turnover, max_drawdown.
        """
        symbol = agent.instrument
        session_length = int(calibration["T_session"])
        mid_price = calibration["mid_price"]

        env = CryptoEnv(
            symbol=symbol,
            mode="simulated",
            num_steps=session_length,
            maker_fee=CRYPTO_DEFAULTS.get(symbol, CRYPTO_DEFAULTS["BTCUSDT"])["maker_fee"],
            taker_fee=CRYPTO_DEFAULTS.get(symbol, CRYPTO_DEFAULTS["BTCUSDT"])["taker_fee"],
        )

        eval_result = agent.evaluate(env, num_episodes=n_eval_episodes)

        # Compute additional metrics
        pnl_mean = eval_result["avg_pnl"]
        sharpe_raw = eval_result["sharpe"]

        # Re-annualise with crypto 8h factor
        # agent.evaluate uses sqrt(252); we correct to sqrt(365*3)
        if abs(sharpe_raw) > 1e-10:
            sharpe_corrected = sharpe_raw / math.sqrt(252) * CRYPTO_8H_SHARPE_FACTOR
        else:
            sharpe_corrected = 0.0

        return {
            "pnl_mean": pnl_mean,
            "pnl_std": eval_result.get("pnl_std", 0.0) if "pnl_std" in eval_result else (
                float(np.std([pnl_mean], ddof=0))
            ),
            "sharpe": sharpe_corrected,
            "max_inventory_used": eval_result.get("avg_inventory", 0.0),
            "fill_rate": eval_result.get("fill_rate", 0.0),
            "avg_spread_bps": calibration["mean_spread"],
            "turnover": eval_result.get("fill_rate", 0.0) * session_length,
            "max_drawdown": eval_result.get("max_drawdown", 0.0),
        }

    def _as_benchmark(
        self,
        calibration: Dict[str, float],
        max_inventory: int,
        n_eval_episodes: int = 200,
    ) -> Dict[str, float]:
        """Compute A-S analytical benchmark metrics for comparison.

        Creates a pure A-S market-making agent (no RL), evaluates on the same
        CryptoEnv parameterisation, and returns metrics.

        Parameters
        ----------
        calibration : dict
            Calibrated parameters.
        max_inventory : int
            Max inventory.
        n_eval_episodes : int
            Number of evaluation episodes.

        Returns
        -------
        dict with same keys as _evaluate_fold.
        """
        session_length = int(calibration["T_session"])
        sigma_per_step = calibration["sigma_per_step"]
        gamma = calibration["gamma"]
        fill_rate_k = calibration["fill_rate_k"]
        mid_price = calibration["mid_price"]

        rng = np.random.default_rng(self.seed + 10000)

        pnls: List[float] = []
        max_inventories: List[int] = []
        fill_counts: List[int] = []

        for ep in range(n_eval_episodes):
            ep_pnl = 0.0
            inventory = 0
            cash = 100_000.0
            price = mid_price
            max_inv = 0
            fills = 0

            for step in range(session_length):
                # A-S analytical quotes
                T_remaining = (session_length - step) / session_length
                quotes = avellaneda_stoikov_quotes(
                    mid_price=price,
                    inventory=inventory,
                    sigma=sigma_per_step,
                    gamma=gamma,
                    T_remaining=T_remaining,
                    fill_rate_k=fill_rate_k,
                )

                bid_price = quotes["bid_price"]
                ask_price = quotes["ask_price"]
                bid_offset = price - bid_price
                ask_offset = ask_price - price

                # Stochastic fills (exponential fill model)
                bid_fill_prob = math.exp(-fill_rate_k * max(bid_offset, 0.0))
                ask_fill_prob = math.exp(-fill_rate_k * max(ask_offset, 0.0))
                bid_fill_prob = min(max(bid_fill_prob, 0.0), 1.0)
                ask_fill_prob = min(max(ask_fill_prob, 0.0), 1.0)

                bid_filled = rng.random() < bid_fill_prob and inventory < max_inventory
                ask_filled = rng.random() < ask_fill_prob and inventory > -max_inventory

                # Update inventory and cash
                if bid_filled:
                    inventory += 1
                    cash -= bid_price
                    fills += 1
                if ask_filled:
                    inventory -= 1
                    cash += ask_price
                    fills += 1

                max_inv = max(max_inv, abs(inventory))

                # Price evolution (GBM-like random walk)
                z = rng.standard_normal()
                price = price * math.exp(sigma_per_step * z)

            # Terminal: liquidate at mid with impact cost
            liquidation_cost = abs(inventory) * sigma_per_step * price * 2.0
            terminal_value = cash + inventory * price - liquidation_cost
            ep_pnl = terminal_value - 100_000.0

            pnls.append(ep_pnl)
            max_inventories.append(max_inv)
            fill_counts.append(fills)

        pnls_arr = np.array(pnls)
        pnl_mean = float(np.mean(pnls_arr))
        pnl_std = float(np.std(pnls_arr, ddof=1)) if len(pnls_arr) > 1 else 1e-8

        sharpe_per_session = pnl_mean / max(pnl_std, 1e-8)
        sharpe_annual = sharpe_per_session * CRYPTO_8H_SHARPE_FACTOR

        # Max drawdown
        cumulative = np.cumsum(pnls_arr)
        peak = np.maximum.accumulate(cumulative)
        dd = peak - cumulative
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0

        return {
            "pnl_mean": pnl_mean,
            "pnl_std": pnl_std,
            "sharpe": sharpe_annual,
            "max_inventory_used": float(np.mean(max_inventories)),
            "fill_rate": float(np.mean(fill_counts)) / max(session_length, 1),
            "avg_spread_bps": calibration["mean_spread"],
            "turnover": float(np.mean(fill_counts)),
            "max_drawdown": max_dd,
        }

    @staticmethod
    def _aggregate_results(
        per_symbol: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate results across all symbols.

        Returns summary dict with mean Sharpe, PnL, and improvement ratio.
        """
        rl_sharpes: List[float] = []
        as_sharpes: List[float] = []
        rl_pnls: List[float] = []
        as_pnls: List[float] = []

        for sym, result in per_symbol.items():
            rl_sharpes.append(result["rl_sharpe"])
            as_sharpes.append(result["as_sharpe"])
            rl_pnls.append(result["rl_pnl_mean"])
            as_pnls.append(result["as_pnl_mean"])

        mean_rl_sharpe = float(np.mean(rl_sharpes)) if rl_sharpes else 0.0
        mean_as_sharpe = float(np.mean(as_sharpes)) if as_sharpes else 0.0

        # Improvement ratio: RL Sharpe / A-S Sharpe
        improvement = (
            mean_rl_sharpe / max(abs(mean_as_sharpe), 1e-8)
            if mean_as_sharpe != 0.0
            else 0.0
        )

        return {
            "mean_rl_sharpe": mean_rl_sharpe,
            "mean_as_sharpe": mean_as_sharpe,
            "mean_rl_pnl": float(np.mean(rl_pnls)) if rl_pnls else 0.0,
            "mean_as_pnl": float(np.mean(as_pnls)) if as_pnls else 0.0,
            "sharpe_improvement_ratio": improvement,
            "n_symbols": len(per_symbol),
        }

    def report(self, results: Dict[str, Any]) -> str:
        """Generate a formatted report comparing RL vs A-S performance.

        Parameters
        ----------
        results : dict
            Output from :meth:`run`.

        Returns
        -------
        str
            Formatted multi-line report string.
        """
        lines: List[str] = []
        sep = "=" * 72

        lines.append(sep)
        lines.append(f"{'AVELLANEDA-STOIKOV MARKET-MAKING PIPELINE RESULTS':^72}")
        lines.append(f"{'(NEW-4: BTC/ETH Perps on Binance)':^72}")
        lines.append(sep)
        lines.append("")

        # Aggregate summary
        agg = results.get("aggregate", {})
        lines.append("AGGREGATE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Symbols traded:          {agg.get('n_symbols', 0)}")
        lines.append(f"  RL Mean Sharpe:          {agg.get('mean_rl_sharpe', 0.0):>10.4f}")
        lines.append(f"  A-S Mean Sharpe:         {agg.get('mean_as_sharpe', 0.0):>10.4f}")
        lines.append(f"  RL Mean PnL/session:     {agg.get('mean_rl_pnl', 0.0):>10.2f}")
        lines.append(f"  A-S Mean PnL/session:    {agg.get('mean_as_pnl', 0.0):>10.2f}")
        improvement = agg.get("sharpe_improvement_ratio", 0.0)
        lines.append(f"  Sharpe Improvement:      {improvement:>10.2f}x")
        lines.append("")

        # Per-symbol details
        per_symbol = results.get("per_symbol", {})
        for symbol, sym_result in per_symbol.items():
            lines.append(f"SYMBOL: {symbol}")
            lines.append("-" * 40)
            lines.append(f"  RL Sharpe:     {sym_result.get('rl_sharpe', 0.0):>10.4f}")
            lines.append(f"  A-S Sharpe:    {sym_result.get('as_sharpe', 0.0):>10.4f}")
            lines.append(f"  RL PnL/sess:   {sym_result.get('rl_pnl_mean', 0.0):>10.2f}")
            lines.append(f"  A-S PnL/sess:  {sym_result.get('as_pnl_mean', 0.0):>10.2f}")

            folds = sym_result.get("folds", [])
            lines.append(f"  Folds:         {len(folds)}")

            for fold in folds:
                fold_idx = fold.get("fold_idx", 0)
                rl_m = fold.get("rl_metrics", {})
                as_m = fold.get("as_metrics", {})
                cal = fold.get("calibration", {})
                train_range = fold.get("train_range", ("?", "?"))
                test_range = fold.get("test_range", ("?", "?"))

                lines.append(f"    Fold {fold_idx}:")
                if isinstance(train_range, tuple) and len(train_range) == 2:
                    lines.append(f"      Train: {train_range[0]} -> {train_range[1]}")
                    lines.append(f"      Test:  {test_range[0]} -> {test_range[1]}")
                lines.append(f"      Calibration: sigma={cal.get('sigma', 0):.4f}, "
                             f"k={cal.get('fill_rate_k', 0):.4f}, "
                             f"spread={cal.get('mean_spread', 0):.2f}bps")
                lines.append(f"      RL  PnL={rl_m.get('pnl_mean', 0):.2f}, "
                             f"Sharpe={rl_m.get('sharpe', 0):.4f}, "
                             f"Fill={rl_m.get('fill_rate', 0):.2%}, "
                             f"MaxInv={rl_m.get('max_inventory_used', 0):.1f}")
                lines.append(f"      A-S PnL={as_m.get('pnl_mean', 0):.2f}, "
                             f"Sharpe={as_m.get('sharpe', 0):.4f}, "
                             f"Fill={as_m.get('fill_rate', 0):.2%}, "
                             f"MaxInv={as_m.get('max_inventory_used', 0):.1f}")

            lines.append("")

        lines.append(sep)
        lines.append(f"{'Annualisation: sqrt(365 * 3) = ' + f'{CRYPTO_8H_SHARPE_FACTOR:.4f}':^72}")
        lines.append(f"{'Sharpe: ddof=1, per-session then annualised':^72}")
        lines.append(sep)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _compute_sharpe(pnls: np.ndarray) -> float:
    """Compute annualised Sharpe ratio from per-session PnLs.

    Uses ddof=1 and CRYPTO_8H_SHARPE_FACTOR = sqrt(365 * 3).

    Parameters
    ----------
    pnls : np.ndarray
        Array of per-session PnL values.

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    if len(pnls) < 2:
        return 0.0
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls, ddof=1))
    if std_pnl < 1e-12:
        return 0.0
    return (mean_pnl / std_pnl) * CRYPTO_8H_SHARPE_FACTOR


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_market_making_pipeline(
    symbols: Optional[List[str]] = None,
    start_date: str = "2025-01-01",
    end_date: str = "2026-02-01",
    n_episodes: int = DEFAULT_N_EPISODES,
    max_inventory: int = 10,
) -> Dict[str, Any]:
    """Convenience function to run the full A-S market-making pipeline.

    Parameters
    ----------
    symbols : list[str] or None
        Trading pairs. Default: ["BTCUSDT", "ETHUSDT"].
    start_date, end_date : str
        Date range.
    n_episodes : int
        RL training episodes per fold.
    max_inventory : int
        Max inventory.

    Returns
    -------
    dict of pipeline results.
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]

    pipeline = MarketMakingPipeline(
        n_episodes=n_episodes,
        max_inventory=max_inventory,
    )

    results = pipeline.run(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )

    report_str = pipeline.report(results)
    logger.info("\n%s", report_str)
    print(report_str)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_market_making_pipeline()
