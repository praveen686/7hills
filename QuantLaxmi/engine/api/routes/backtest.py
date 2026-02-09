"""Backtest routes — strategy-specific backtests via dispatcher.

POST /api/backtest/run           — launch a new backtest
GET  /api/backtest/{id}/status   — poll backtest progress / results
GET  /api/backtest/strategies    — list available strategies with default params
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


# ------------------------------------------------------------------
# Enums and models
# ------------------------------------------------------------------

class BacktestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BacktestRequest(BaseModel):
    strategy_id: str
    start_date: str = Field(..., description="ISO date YYYY-MM-DD")
    end_date: str = Field(..., description="ISO date YYYY-MM-DD")
    initial_capital: float = 10_000_000  # ₹1 Crore default
    params: dict = Field(default_factory=dict, description="Strategy-specific parameters")


class EquityPointOut(BaseModel):
    date: str
    equity: float
    drawdown: float | None = None
    benchmark: float | None = None


class DrawdownPointOut(BaseModel):
    date: str
    drawdown: float


class MonthlyReturnOut(BaseModel):
    year: int
    month: int
    return_pct: float


class BacktestResultOut(BaseModel):
    total_return: float | None = None
    cagr: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float | None = None
    n_trades: int | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    avg_trade_pnl: float | None = None
    total_costs: float | None = None
    final_equity: float | None = None
    equity_curve: list[EquityPointOut] | None = None
    drawdown_curve: list[DrawdownPointOut] | None = None
    monthly_returns: list[MonthlyReturnOut] | None = None


class BacktestStatusOut(BaseModel):
    backtest_id: str
    status: BacktestStatus
    strategy_id: str
    start_date: str
    end_date: str
    created_at: str
    completed_at: str | None = None
    error: str | None = None
    result: BacktestResultOut | None = None


class BacktestLaunchOut(BaseModel):
    backtest_id: str
    status: BacktestStatus
    message: str


class StrategyParamInfo(BaseModel):
    strategy_id: str
    name: str
    default_params: dict


# ------------------------------------------------------------------
# Strategy catalog with default parameters
# ------------------------------------------------------------------

STRATEGY_DEFAULTS: dict[str, dict] = {
    "s1_vrp": {
        "name": "S1 VRP-RNDR",
        "params": {"entry_pctile": 0.80, "hold_days": 5, "variant": "options"},
    },
    "s2_ramanujan": {
        "name": "S2 Ramanujan Cycles",
        "params": {"symbol": "NIFTY", "max_period": 64},
    },
    "s3_institutional": {
        "name": "S3 Institutional Flow",
        "params": {"top_n": 10},
    },
    "s4_iv_mr": {
        "name": "S4 IV Mean-Reversion",
        "params": {"iv_lookback": 30, "entry_pctile": 0.80},
    },
    "s5_hawkes": {
        "name": "S5 Hawkes Microstructure",
        "params": {"lookback": 60, "entry_pctile": 0.80, "feature": "hawkes_ratio"},
    },
    "s6_multi_factor": {
        "name": "S6 Multi-Factor XGBoost",
        "params": {"target_index": "Nifty 50", "train_window": 60},
    },
    "s7_regime": {
        "name": "S7 Regime Switch",
        "params": {"symbol": "NIFTY", "lookback": 100, "supertrend_mult": 3.0, "cost_bps": 10.0},
    },
    "s8_expiry_theta": {
        "name": "S8 Expiry-Day Theta",
        "params": {"symbol": "NIFTY", "short_otm_pct": 0.015, "max_vix": 18.0},
    },
    "s9_momentum": {
        "name": "S9 Cross-Section Momentum",
        "params": {"top_n": 5, "cost_bps": 10.0},
    },
    "s10_gamma_scalp": {
        "name": "S10 Gamma Scalping",
        "params": {"symbol": "NIFTY", "iv_pctile_threshold": 0.20, "min_dte": 14},
    },
    "s11_pairs": {
        "name": "S11 Statistical Pairs",
        "params": {"lookback": 60, "z_entry": 2.0, "cost_bps": 10.0},
    },
}


# ------------------------------------------------------------------
# In-memory backtest tracker (shared via app.state)
# ------------------------------------------------------------------

class BacktestTracker:
    """Thread-safe tracker for async backtest jobs with disk persistence."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._persist_dir = persist_dir
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # -- persistence helpers --------------------------------------------------

    def _load_from_disk(self) -> None:
        """Load completed/failed backtest results from JSON files on startup."""
        if self._persist_dir is None:
            return
        import json as _json
        for p in sorted(self._persist_dir.glob("*.json")):
            try:
                data = _json.loads(p.read_text())
                job_id = data.get("backtest_id", p.stem)
                # Restore enum from string
                data["status"] = BacktestStatus(data["status"])
                self._jobs[job_id] = data
            except Exception as exc:
                logger.warning("Failed to load backtest result %s: %s", p, exc)

    def _persist(self, job_id: str) -> None:
        """Write a single job record to disk (completed or failed only)."""
        if self._persist_dir is None:
            return
        import json as _json
        job = self._jobs.get(job_id)
        if job is None:
            return
        status = job["status"]
        if status not in (BacktestStatus.COMPLETED, BacktestStatus.FAILED):
            return
        # Build serialisable copy (drop non-JSON-safe fields)
        rec = {
            "backtest_id": job["backtest_id"],
            "status": job["status"].value,
            "strategy_id": job["strategy_id"],
            "start_date": job["start_date"],
            "end_date": job["end_date"],
            "created_at": job["created_at"],
            "completed_at": job["completed_at"],
            "error": job["error"],
            "result": job["result"],
        }
        out = self._persist_dir / f"{job_id}.json"
        out.write_text(_json.dumps(rec, default=str))

    # -- public API -----------------------------------------------------------

    async def create(self, params: BacktestRequest) -> str:
        job_id = uuid.uuid4().hex[:12]
        async with self._lock:
            self._jobs[job_id] = {
                "backtest_id": job_id,
                "status": BacktestStatus.PENDING,
                "strategy_id": params.strategy_id,
                "start_date": params.start_date,
                "end_date": params.end_date,
                "params": params,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "error": None,
                "result": None,
            }
        return job_id

    async def set_running(self, job_id: str) -> None:
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = BacktestStatus.RUNNING

    async def set_completed(self, job_id: str, result: dict[str, Any]) -> None:
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = BacktestStatus.COMPLETED
                self._jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                self._jobs[job_id]["result"] = result
                self._persist(job_id)

    async def set_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = BacktestStatus.FAILED
                self._jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                self._jobs[job_id]["error"] = error
                self._persist(job_id)

    async def get(self, job_id: str) -> dict[str, Any] | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_all(self) -> list[dict[str, Any]]:
        """Return all tracked jobs (for history endpoint)."""
        async with self._lock:
            return list(self._jobs.values())


# ------------------------------------------------------------------
# Strategy-specific backtest functions
# ------------------------------------------------------------------

def _run_s1(store, start: date, end: date, p: dict) -> dict:
    """S1 VRP-RNDR: density-based directional or options spreads."""
    variant = p.get("variant", "options")
    entry_pctile = float(p.get("entry_pctile", 0.80))
    hold_days = int(p.get("hold_days", 5))

    if variant == "options":
        from strategies.s1_vrp.options import run_multi_index_options_backtest
        results = run_multi_index_options_backtest(
            store, start, end, entry_pctile=entry_pctile, hold_days=hold_days,
        )
        # Aggregate: pick the best-performing index
        best = None
        for sym, r in results.items():
            if best is None or r.sharpe > best.sharpe:
                best = r
        if best is None:
            return {"trades": 0, "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0, "win_rate": 0}
        daily_rets = _trades_to_daily_returns(best.daily, best.trades, "pnl_on_risk", pct=False)
        return {
            "trades": best.n_signals,
            "total_return_pct": best.total_return_on_risk_pct,
            "sharpe": best.sharpe,
            "max_dd_pct": best.max_dd_pct,
            "win_rate": best.win_rate,
            "daily_returns": daily_rets,
        }
    else:
        from strategies.s1_vrp.density import run_multi_index_density_backtest
        results = run_multi_index_density_backtest(
            store, start, end, entry_pctile=entry_pctile, hold_days=hold_days,
        )
        best = None
        for sym, r in results.items():
            if best is None or r.sharpe > best.sharpe:
                best = r
        if best is None:
            return {"trades": 0, "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0, "win_rate": 0}
        daily_rets = _trades_to_daily_returns(best.daily, best.trades, "pnl_pct", pct=True)
        return {
            "trades": best.n_signals,
            "total_return_pct": best.total_return_pct,
            "sharpe": best.sharpe,
            "max_dd_pct": best.max_dd_pct,
            "win_rate": best.win_rate,
            "daily_returns": daily_rets,
        }


def _run_s2(store, start: date, end: date, p: dict) -> dict:
    """S2 Ramanujan Cycles: intraday phase-based trading.

    Adapts run_research to return structured results instead of just printing.
    """
    from features.ramanujan import ramanujan_periodogram, ramanujan_sum

    symbol = p.get("symbol", "NIFTY")
    max_period = int(p.get("max_period", 64))

    dates = store.available_dates("nfo_1min")
    dates = [d for d in dates if start <= d <= end]
    dates.sort()

    daily_returns: list[float] = []
    total_trades = 0
    total_wins = 0

    for d in dates:
        try:
            df = store.sql(
                "SELECT * FROM nfo_1min WHERE date = ? AND name = ? "
                "AND instrument_type = 'FUT' ORDER BY timestamp",
                [d.isoformat(), symbol],
            )
            if df is None or df.empty or len(df) < max_period * 2:
                daily_returns.append(0.0)
                continue

            if "expiry" in df.columns:
                df["_exp"] = pd.to_datetime(df["expiry"], format="mixed", errors="coerce")
                min_exp = df["_exp"].min()
                df = df[df["_exp"] == min_exp].drop(columns=["_exp"])

            close = df["close"].values.astype(np.float64)

            # Causal detrend
            window = min(20, len(close) // 5)
            window = max(window, 2)
            trend = pd.Series(close).rolling(window, min_periods=1).mean().values
            detrended = close - trend

            mid = max(len(detrended) // 2, max_period)
            if mid >= len(detrended):
                daily_returns.append(0.0)
                continue

            # Causal periodogram at midpoint
            segment = detrended[max(0, mid - max_period + 1):mid + 1]
            if len(segment) < max_period:
                daily_returns.append(0.0)
                continue

            energies = ramanujan_periodogram(segment, max_period)
            energies[0] = 0
            best_idx = np.argmax(energies)
            if energies[best_idx] <= 0:
                daily_returns.append(0.0)
                continue

            primary = int(best_idx + 1)

            # Causal phase via quadrature
            import math
            length = primary * 2
            n = len(detrended)
            cos_filt = np.array([math.cos(2 * math.pi * k / primary) for k in range(length)])
            sin_filt = np.array([math.sin(2 * math.pi * k / primary) for k in range(length)])
            hann = np.hanning(length)
            cos_filt *= hann
            sin_filt *= hann
            cos_out = np.convolve(detrended, cos_filt[::-1], mode='full')[:n]
            sin_out = np.convolve(detrended, sin_filt[::-1], mode='full')[:n]
            phase = np.arctan2(sin_out, cos_out)

            # Trade in second half
            trade_start = mid + primary * 2
            position = 0
            entry_price = 0.0
            day_pnl = 0.0
            n_trades = 0
            wins = 0

            for j in range(max(trade_start, 1), len(close)):
                if position == 0 and phase[j - 1] < -1.5 and phase[j] >= -1.5:
                    position = 1
                    entry_price = close[j]
                elif position == 1 and phase[j - 1] < 1.0 and phase[j] >= 1.0:
                    pnl = (close[j] - entry_price) / entry_price
                    day_pnl += pnl
                    n_trades += 1
                    if pnl > 0:
                        wins += 1
                    position = 0

            if position == 1:
                pnl = (close[-1] - entry_price) / entry_price
                day_pnl += pnl
                n_trades += 1
                if pnl > 0:
                    wins += 1

            daily_returns.append(day_pnl)
            total_trades += n_trades
            total_wins += wins

        except Exception:
            daily_returns.append(0.0)

    return _compute_metrics_from_daily(daily_returns, total_trades, total_wins)


def _run_s3(store, start: date, end: date, p: dict) -> dict:
    """S3 Institutional Flow: signal quality evaluation with forward returns."""
    top_n = int(p.get("top_n", 10))
    cost_bps = float(p.get("cost_bps", 10.0))
    cost_frac = cost_bps / 10_000

    from strategies.s9_momentum.scanner import run_daily_scan
    from strategies.s9_momentum.data import available_dates, is_trading_day

    dates = available_dates(store, "nse_delivery")
    dates = [d for d in dates if start <= d <= end]
    dates.sort()

    daily_returns: list[float] = []
    total_trades = 0
    total_wins = 0

    for i, d in enumerate(dates[:-5]):
        try:
            signals = run_daily_scan(d, store=store, top_n=top_n)
        except Exception:
            daily_returns.append(0.0)
            continue

        if not signals:
            daily_returns.append(0.0)
            continue

        # 1-day forward return from top signal
        next_d = d + timedelta(days=1)
        while not is_trading_day(next_d) and (next_d - d).days < 10:
            next_d += timedelta(days=1)

        day_ret = 0.0
        for sig in signals[:3]:  # top 3
            entry_price = _get_cm_close(store, sig.symbol, d)
            exit_price = _get_cm_close(store, sig.symbol, next_d)
            if entry_price and exit_price and entry_price > 0:
                direction = 1 if sig.composite_score > 0 else -1
                ret = direction * (exit_price / entry_price - 1) - cost_frac
                day_ret += ret / 3
                total_trades += 1
                if ret > 0:
                    total_wins += 1

        daily_returns.append(day_ret)

    return _compute_metrics_from_daily(daily_returns, total_trades, total_wins)


def _get_cm_close(store, symbol: str, d: date) -> float | None:
    """Helper: get CM bhavcopy close for S3."""
    df = store.sql(
        "SELECT * FROM nse_cm_bhavcopy WHERE date = ?", [d.isoformat()]
    )
    if df is None or df.empty:
        return None
    for col in ["TckrSymb", "SYMBOL", "symbol"]:
        if col in df.columns:
            row = df[df[col] == symbol]
            if not row.empty:
                for pcol in ["ClsPric", "CLOSE", "close"]:
                    if pcol in row.columns:
                        val = row[pcol].iloc[0]
                        if pd.notna(val) and float(val) > 0:
                            return float(val)
    return None


def _run_s4(store, start: date, end: date, p: dict) -> dict:
    """S4 IV Mean-Reversion: SANOS-calibrated IV percentile strategy."""
    iv_lookback = int(p.get("iv_lookback", 30))
    entry_pctile = float(p.get("entry_pctile", 0.80))

    from strategies.s4_iv_mr.engine import (
        build_iv_series,
        run_from_series,
        SUPPORTED_INDICES,
    )

    # Build IV series and run for each index, pick best
    best_result = None
    best_sharpe = -999.0

    for symbol in SUPPORTED_INDICES:
        try:
            daily = build_iv_series(store, start, end, symbol=symbol)
            result = run_from_series(
                daily, iv_lookback=iv_lookback, entry_pctile=entry_pctile, symbol=symbol,
            )
            if result.sharpe > best_sharpe:
                best_sharpe = result.sharpe
                best_result = result
        except Exception as exc:
            logger.debug("S4 backtest failed for %s: %s", symbol, exc)

    if best_result is None:
        return {"trades": 0, "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0, "win_rate": 0}

    daily_rets = _trades_to_daily_returns(best_result.daily, best_result.trades, "pnl_pct", pct=True)
    return {
        "trades": best_result.n_signals,
        "total_return_pct": best_result.total_return_pct,
        "sharpe": best_result.sharpe,
        "max_dd_pct": best_result.max_dd_pct,
        "win_rate": best_result.win_rate,
        "daily_returns": daily_rets,
    }


def _run_s5(store, start: date, end: date, p: dict) -> dict:
    """S5 Hawkes Microstructure: feature-based next-day signal backtest.

    Uses CPU-only Hawkes estimation (no GPU required for backtest mode).
    """
    lookback = int(p.get("lookback", 60))
    entry_pctile = float(p.get("entry_pctile", 0.80))
    feature = p.get("feature", "hawkes_ratio")
    cost_bps = float(p.get("cost_bps", 5.0))
    token = int(p.get("token", 256265))  # NIFTY

    from strategies.s5_hawkes.research import _load_day_data, _backtest_feature

    dates = store.available_dates("ticks")
    dates = sorted(d for d in dates if start <= d <= end)

    if len(dates) < lookback + 20:
        return {"trades": 0, "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0, "win_rate": 0}

    # Load tick data and compute Hawkes features (CPU only, sequential)
    daily_features: list[dict] = []
    for d in dates:
        try:
            rec = _load_day_data(d.isoformat(), token)
            if rec is not None:
                rec.pop("prices", None)  # drop raw prices to save memory
                daily_features.append(rec)
        except Exception:
            continue

    if len(daily_features) < lookback + 10:
        return {"trades": 0, "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0, "win_rate": 0}

    daily_features.sort(key=lambda x: x["date"])
    df = pd.DataFrame(daily_features)

    result = _backtest_feature(
        df, feature, lookback=lookback, entry_pctile=entry_pctile,
        hold_days=1, cost_bps=cost_bps,
    )

    return result


def _run_s6(store, start: date, end: date, p: dict) -> dict:
    """S6 Multi-Factor XGBoost: walk-forward prediction backtest."""
    target_index = p.get("target_index", "Nifty 50")
    train_window = int(p.get("train_window", 60))
    test_window = int(p.get("test_window", 5))

    from strategies.s6_multi_factor.research import _build_feature_matrix, _walk_forward_backtest

    X, y, feature_names = _build_feature_matrix(
        store, start, end, target_index=target_index,
    )
    if X.empty:
        return {"trades": 0, "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0, "win_rate": 0}

    result = _walk_forward_backtest(X, y, train_window=train_window, test_window=test_window)

    if "error" in result:
        return {"trades": 0, "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0, "win_rate": 0}

    return {
        "trades": result.get("n_predictions", 0),
        "total_return_pct": result.get("cum_return", 0) * 100,
        "sharpe": result.get("sharpe", 0),
        "max_dd_pct": abs(result.get("max_dd", 0)) * 100,
        "win_rate": result.get("hit_rate", 0),
        "daily_returns": result.get("daily_returns", []),
    }


def _run_s7(store, start: date, end: date, p: dict) -> dict:
    """S7 Regime Switch: entropy/VPIN regime detection with sub-strategies."""
    from strategies.s7_regime.research import backtest_regime_switch

    symbol = p.get("symbol", "NIFTY")
    lookback = int(p.get("lookback", 100))
    supertrend_mult = float(p.get("supertrend_mult", 3.0))
    cost_bps = float(p.get("cost_bps", 10.0))

    return backtest_regime_switch(
        store, start, end, symbol=symbol,
        lookback=lookback, supertrend_mult=supertrend_mult, cost_bps=cost_bps,
    )


def _run_s8(store, start: date, end: date, p: dict) -> dict:
    """S8 Expiry-Day Theta: iron condors on expiry using actual option prices."""
    from strategies.s8_expiry_theta.strategy import backtest_expiry_theta

    symbol = p.get("symbol", "NIFTY")
    short_otm_pct = float(p.get("short_otm_pct", 0.015))
    max_vix = float(p.get("max_vix", 18.0))

    return backtest_expiry_theta(
        store, start, end, symbol=symbol,
        short_otm_pct=short_otm_pct, max_vix=max_vix,
    )


def _run_s9(store, start: date, end: date, p: dict) -> dict:
    """S9 Cross-Sectional Momentum: weekly rebalance stock portfolio."""
    from strategies.s9_momentum.research import backtest_momentum

    top_n = int(p.get("top_n", 5))
    cost_bps = float(p.get("cost_bps", 10.0))

    return backtest_momentum(store, start, end, top_n=top_n, cost_bps=cost_bps)


def _run_s10(store, start: date, end: date, p: dict) -> dict:
    """S10 Gamma Scalping: buy straddles in low-IV with actual option prices."""
    from strategies.s10_gamma_scalp.research import backtest_gamma_scalp

    symbol = p.get("symbol", "NIFTY")
    iv_pctile_threshold = float(p.get("iv_pctile_threshold", 0.20))
    min_dte = int(p.get("min_dte", 14))

    return backtest_gamma_scalp(
        store, start, end, symbol=symbol,
        iv_pctile_threshold=iv_pctile_threshold, min_dte=min_dte,
    )


def _run_s11(store, start: date, end: date, p: dict) -> dict:
    """S11 Statistical Pairs: cointegration-based spread trading."""
    from strategies.s11_pairs.research import backtest_pairs

    lookback = int(p.get("lookback", 60))
    z_entry = float(p.get("z_entry", 2.0))
    cost_bps = float(p.get("cost_bps", 10.0))

    return backtest_pairs(
        store, start, end, lookback=lookback, z_entry=z_entry, cost_bps=cost_bps,
    )


# Strategy dispatcher
_STRATEGY_RUNNERS = {
    "s1_vrp": _run_s1,
    "s2_ramanujan": _run_s2,
    "s3_institutional": _run_s3,
    "s4_iv_mr": _run_s4,
    "s5_hawkes": _run_s5,
    "s6_multi_factor": _run_s6,
    "s7_regime": _run_s7,
    "s8_expiry_theta": _run_s8,
    "s9_momentum": _run_s9,
    "s10_gamma_scalp": _run_s10,
    "s11_pairs": _run_s11,
}


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _trades_to_daily_returns(
    daily_obs: list, trades: list, pnl_field: str = "pnl_pct", *, pct: bool = True,
) -> list[float]:
    """Convert a list of trade objects to a daily return series.

    Maps each trade's PnL to its exit_date, sums per date, fills zeros
    for non-trade dates.  Uses the ``daily`` observation list for date alignment.

    Args:
        daily_obs: List of day observation objects (must have .date attribute).
        trades: List of trade objects (must have .exit_date and the pnl_field).
        pnl_field: Attribute name on each trade holding the PnL value.
        pct: If True, the pnl_field is in percentage (divide by 100 to get fraction).
    """
    dates = [obs.date for obs in daily_obs]
    date_pnl: dict[date, float] = {}
    for t in trades:
        d = t.exit_date
        pnl = getattr(t, pnl_field, 0)
        if pct:
            pnl = pnl / 100  # convert percentage to fraction
        date_pnl[d] = date_pnl.get(d, 0.0) + pnl
    return [date_pnl.get(d, 0.0) for d in dates]


def _safe(v: float, decimals: int = 6) -> float | None:
    """Sanitize float for JSON (handle inf/nan)."""
    f = float(v)
    if not np.isfinite(f):
        return None
    return round(f, decimals)


def _compute_metrics_from_daily(
    daily_returns: list[float],
    total_trades: int,
    total_wins: int,
) -> dict:
    """Compute standard backtest metrics from daily return series."""
    if not daily_returns or all(r == 0 for r in daily_returns):
        return {
            "trades": total_trades, "total_return_pct": 0, "sharpe": 0,
            "max_dd_pct": 0, "win_rate": 0, "daily_returns": [],
        }

    rets = np.array(daily_returns, dtype=float)
    equity = np.cumprod(1 + rets)
    total_ret = (equity[-1] - 1) * 100
    std = np.std(rets, ddof=1)
    sharpe = float(np.mean(rets) / std * np.sqrt(252)) if std > 0 else 0

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(np.max(dd)) * 100

    win_rate = total_wins / total_trades if total_trades > 0 else 0

    return {
        "trades": total_trades,
        "total_return_pct": total_ret,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "win_rate": win_rate,
        "wins": total_wins,
        "daily_returns": daily_returns,
    }


def _build_equity_curve(
    raw: dict,
    start: date,
    end: date,
    initial_capital: float,
    store=None,
) -> dict:
    """Build equity curve, drawdown curve, and monthly returns from strategy result.

    Tries to use daily_returns if available; falls back to synthetic curve from
    total return distributed linearly over available trading dates.
    """
    daily_rets = raw.get("daily_returns", [])
    total_return_pct = raw.get("total_return_pct", 0)
    n_trades = raw.get("trades", 0)
    win_rate = raw.get("win_rate", 0)
    sharpe = raw.get("sharpe", 0)
    max_dd_pct = raw.get("max_dd_pct", 0)

    equity_curve: list[dict] = []
    drawdown_curve: list[dict] = []
    monthly_returns: list[dict] = []

    if daily_rets:
        rets = np.array(daily_rets, dtype=float)
        equity = np.cumprod(1 + rets) * initial_capital

        # Generate trading dates
        trading_dates = _generate_trading_dates(start, end, len(rets), store)
        n = min(len(trading_dates), len(equity))

        peak = np.maximum.accumulate(equity[:n])
        dd = (equity[:n] - peak) / np.where(peak > 0, peak, 1)

        for i in range(n):
            d_str = trading_dates[i].isoformat()
            equity_curve.append({
                "date": d_str,
                "equity": round(float(equity[i]), 2),
                "drawdown": round(float(dd[i]), 6),
            })
            drawdown_curve.append({
                "date": d_str,
                "drawdown": round(float(dd[i]), 6),
            })

        # Monthly returns
        monthly_groups: dict[tuple[int, int], float] = {}
        for i in range(min(n, len(rets))):
            ym = (trading_dates[i].year, trading_dates[i].month)
            monthly_groups.setdefault(ym, 0.0)
            monthly_groups[ym] += float(rets[i])
        for (y, m), r in sorted(monthly_groups.items()):
            monthly_returns.append({"year": y, "month": m, "return_pct": round(r * 100, 4)})

        final_equity = round(float(equity[n - 1]), 2) if n > 0 else initial_capital

    elif total_return_pct != 0:
        # Fallback: generate synthetic equity curve from total return
        trading_dates = _generate_trading_dates(start, end, 0, store)
        n_days = len(trading_dates) or 1
        daily_ret = (1 + total_return_pct / 100) ** (1 / n_days) - 1

        equity_val = initial_capital
        peak_val = initial_capital
        for d in trading_dates:
            equity_val *= (1 + daily_ret)
            peak_val = max(peak_val, equity_val)
            dd_val = (equity_val - peak_val) / peak_val if peak_val > 0 else 0
            equity_curve.append({
                "date": d.isoformat(),
                "equity": round(equity_val, 2),
                "drawdown": round(dd_val, 6),
            })
            drawdown_curve.append({"date": d.isoformat(), "drawdown": round(dd_val, 6)})

        final_equity = round(equity_val, 2)
    else:
        final_equity = initial_capital

    # Compute sortino from daily returns
    sortino = 0.0
    if daily_rets:
        rets = np.array(daily_rets, dtype=float)
        down = rets[rets < 0]
        if len(down) > 1:
            down_std = np.std(down, ddof=1)
            if down_std > 0:
                sortino = float(np.mean(rets) / down_std * np.sqrt(252))

    # Compute profit factor from trade details if available
    profit_factor = 0.0
    trade_details = raw.get("trade_details", [])
    if trade_details:
        gains = sum(
            (getattr(t, "pnl_pct", 0) if hasattr(t, "pnl_pct") else t.get("pnl_pct", 0))
            for t in trade_details
            if (getattr(t, "pnl_pct", 0) if hasattr(t, "pnl_pct") else t.get("pnl_pct", 0)) > 0
        )
        losses = abs(sum(
            (getattr(t, "pnl_pct", 0) if hasattr(t, "pnl_pct") else t.get("pnl_pct", 0))
            for t in trade_details
            if (getattr(t, "pnl_pct", 0) if hasattr(t, "pnl_pct") else t.get("pnl_pct", 0)) < 0
        ))
        if losses > 0:
            profit_factor = gains / losses

    avg_trade_pnl = total_return_pct / n_trades / 100 if n_trades > 0 else 0

    # CAGR: annualized compound growth rate
    n_years = len(equity_curve) / 252 if equity_curve else 0
    if n_years > 0.01 and final_equity > 0 and initial_capital > 0:
        cagr = (final_equity / initial_capital) ** (1 / n_years) - 1
    else:
        cagr = 0

    return {
        "total_return": _safe(total_return_pct / 100),
        "cagr": _safe(cagr, 6),
        "sharpe_ratio": _safe(sharpe, 4),
        "sortino_ratio": _safe(sortino, 4),
        "max_drawdown": _safe(max_dd_pct / 100),
        "n_trades": n_trades,
        "win_rate": _safe(win_rate * 100 if win_rate <= 1.0 else win_rate, 2),
        "profit_factor": _safe(profit_factor, 4),
        "avg_trade_pnl": _safe(avg_trade_pnl),
        "total_costs": 0,
        "final_equity": final_equity,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "monthly_returns": monthly_returns,
    }


def _generate_trading_dates(
    start: date, end: date, target_count: int, store=None,
) -> list[date]:
    """Generate list of trading dates between start and end.

    When target_count > 0 (i.e. we have daily_returns to align), prefer
    the data source whose count is closest.  Falls back to business days.
    """
    # Always generate business days as baseline
    biz_dates: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            biz_dates.append(d)
        d += timedelta(days=1)

    if store is not None:
        try:
            available = store.available_dates("nse_index_close")
            store_dates = sorted(d for d in available if start <= d <= end)
            if store_dates:
                # If we have a target count, pick whichever source is closer
                if target_count > 0:
                    store_diff = abs(len(store_dates) - target_count)
                    biz_diff = abs(len(biz_dates) - target_count)
                    return store_dates if store_diff <= biz_diff else biz_dates
                return store_dates
        except Exception:
            pass

    return biz_dates


# ------------------------------------------------------------------
# Background task
# ------------------------------------------------------------------

async def _run_backtest_task(
    job_id: str,
    params: BacktestRequest,
    tracker: BacktestTracker,
    app_state: Any,
) -> None:
    """Execute strategy-specific backtest in background and update tracker."""
    await tracker.set_running(job_id)

    try:
        store = app_state.market_data_service.store
        start = date.fromisoformat(params.start_date)
        end = date.fromisoformat(params.end_date)

        runner = _STRATEGY_RUNNERS.get(params.strategy_id)
        if runner is None:
            await tracker.set_failed(
                job_id,
                f"Strategy '{params.strategy_id}' has no backtest implementation. "
                f"Available: {list(_STRATEGY_RUNNERS.keys())}",
            )
            return

        # Merge default params with user overrides
        defaults = STRATEGY_DEFAULTS.get(params.strategy_id, {}).get("params", {})
        merged_params = {**defaults, **params.params}

        logger.info(
            "Backtest %s: running %s [%s → %s] params=%s",
            job_id, params.strategy_id, start, end, merged_params,
        )

        # Run in executor to avoid blocking event loop
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            raw = await loop.run_in_executor(
                pool, lambda: runner(store, start, end, merged_params)
            )

        logger.info(
            "Backtest %s: raw result — trades=%d sharpe=%.2f return=%.2f%%",
            job_id, raw.get("trades", 0), raw.get("sharpe", 0), raw.get("total_return_pct", 0),
        )

        # Normalize and build equity curve
        result_dict = _build_equity_curve(raw, start, end, params.initial_capital, store)

        await tracker.set_completed(job_id, result_dict)
        logger.info("Backtest %s completed successfully", job_id)

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Backtest %s failed: %s\n%s", job_id, exc, tb)
        await tracker.set_failed(job_id, str(exc))


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.get("/strategies", response_model=list[StrategyParamInfo])
async def list_backtest_strategies() -> list[StrategyParamInfo]:
    """List all strategies available for backtesting with their default parameters."""
    return [
        StrategyParamInfo(
            strategy_id=sid,
            name=info["name"],
            default_params=info["params"],
        )
        for sid, info in STRATEGY_DEFAULTS.items()
    ]


@router.post("/run", response_model=BacktestLaunchOut, status_code=202)
async def run_backtest_endpoint(
    body: BacktestRequest,
    request: Request,
) -> BacktestLaunchOut:
    """Launch an asynchronous backtest job.

    Returns immediately with a backtest_id.  Poll
    ``GET /api/backtest/{id}/status`` for progress and results.
    """
    # Validate dates
    try:
        start = date.fromisoformat(body.start_date)
        end = date.fromisoformat(body.end_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {exc}") from exc

    if start >= end:
        raise HTTPException(status_code=400, detail="start_date must be before end_date.")

    # Validate strategy exists
    if body.strategy_id not in _STRATEGY_RUNNERS:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{body.strategy_id}' not found. "
                   f"Available: {list(_STRATEGY_RUNNERS.keys())}",
        )

    tracker: BacktestTracker = request.app.state.backtest_tracker
    job_id = await tracker.create(body)

    # Fire and forget
    asyncio.create_task(
        _run_backtest_task(job_id, body, tracker, request.app.state),
        name=f"backtest-{job_id}",
    )

    return BacktestLaunchOut(
        backtest_id=job_id,
        status=BacktestStatus.PENDING,
        message=f"Backtest launched. Poll GET /api/backtest/{job_id}/status for results.",
    )


@router.get("/history")
async def list_backtests(request: Request) -> list[BacktestStatusOut]:
    """List all backtest jobs (including persisted completed/failed ones)."""
    tracker: BacktestTracker = request.app.state.backtest_tracker
    jobs = await tracker.list_all()
    out: list[BacktestStatusOut] = []
    for job in jobs:
        result_out = BacktestResultOut(**job["result"]) if job.get("result") else None
        out.append(BacktestStatusOut(
            backtest_id=job["backtest_id"],
            status=job["status"],
            strategy_id=job["strategy_id"],
            start_date=job["start_date"],
            end_date=job["end_date"],
            created_at=job["created_at"],
            completed_at=job.get("completed_at"),
            error=job.get("error"),
            result=result_out,
        ))
    # Most recent first
    out.sort(key=lambda x: x.created_at, reverse=True)
    return out


@router.get("/{backtest_id}/status", response_model=BacktestStatusOut)
async def get_backtest_status(
    backtest_id: str,
    request: Request,
) -> BacktestStatusOut:
    """Poll the status and results of a running or completed backtest."""
    tracker: BacktestTracker = request.app.state.backtest_tracker
    job = await tracker.get(backtest_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Backtest '{backtest_id}' not found.",
        )

    result_out = None
    if job["result"] is not None:
        result_out = BacktestResultOut(**job["result"])

    return BacktestStatusOut(
        backtest_id=job["backtest_id"],
        status=job["status"],
        strategy_id=job["strategy_id"],
        start_date=job["start_date"],
        end_date=job["end_date"],
        created_at=job["created_at"],
        completed_at=job["completed_at"],
        error=job["error"],
        result=result_out,
    )
