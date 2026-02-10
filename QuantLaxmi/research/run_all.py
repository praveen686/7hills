"""AlphaForge — Master Backtest Runner & Scorecard Generator.

Runs all AlphaForge strategies and produces a consolidated scorecard
with honest statistics (Sharpe, max drawdown, Calmar, win rate, etc.).

Usage:
    python -m research.run_all [--strategies hmm,ofi,skew,mtfm,vrp]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Strategy imports
# ---------------------------------------------------------------------------
from quantlaxmi.strategies.s13_hmm_regime.strategy import (
    HMMRegimeConfig,
    run_backtest as run_hmm,
)
from quantlaxmi.strategies.s14_ofi_intraday.strategy import (
    OFIConfig,
    run_ofi_backtest as run_ofi,
)
from quantlaxmi.strategies.s15_skew_mr.strategy import (
    SkewConfig,
    run_skew_backtest as run_skew,
)
from quantlaxmi.models.ml.tft.momentum_tfm import (
    MomentumTFMConfig,
    run_backtest as run_mtfm,
)
from quantlaxmi.strategies.s16_vrp_enhanced.strategy import (
    VRPConfig,
    run_vrp_backtest as run_vrp,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Scorecard data class
# ---------------------------------------------------------------------------
@dataclass
class StrategyResult:
    name: str
    sharpe: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    calmar: float
    n_trades: int
    win_rate_pct: float
    avg_trade_return_pct: float
    profit_factor: float
    avg_hold_days: float
    backtest_days: int
    status: str = "OK"
    notes: str = ""
    elapsed_sec: float = 0.0


def compute_stats_from_equity(equity: pd.Series) -> dict:
    """Compute standard statistics from an equity curve (cumulative returns)."""
    if equity is None or len(equity) < 2:
        return {}

    # Daily returns
    daily_ret = equity.pct_change().fillna(0.0)
    n_days = len(daily_ret)

    # Sharpe: ddof=1, sqrt(252), all days including flat
    mean_r = daily_ret.mean()
    std_r = daily_ret.std(ddof=1)
    sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0

    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()

    # Annualized return
    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    ann_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1.0

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_ret * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "backtest_days": n_days,
    }


# ---------------------------------------------------------------------------
# Individual strategy runners
# ---------------------------------------------------------------------------

def _run_hmm(name: str = "NIFTY", start: str = "2025-08-06",
             end: str = "2026-02-06") -> StrategyResult:
    """Run HMM regime-switching strategy."""
    t0 = time.time()
    try:
        cfg = HMMRegimeConfig()
        result = run_hmm(
            index_name=name,
            start_date=start,
            end_date=end,
            config=cfg,
        )
        elapsed = time.time() - t0

        # Extract stats from result
        trades = result.trades if hasattr(result, "trades") else []
        equity = result.equity_curve if hasattr(result, "equity_curve") else None
        n_trades = len(trades)

        if equity is not None and len(equity) > 1:
            stats = compute_stats_from_equity(equity)
        else:
            stats = {
                "sharpe": getattr(result, "sharpe", 0.0),
                "total_return_pct": getattr(result, "total_return_pct", 0.0),
                "annualized_return_pct": 0.0,
                "max_drawdown_pct": getattr(result, "max_drawdown_pct", 0.0),
                "calmar": 0.0,
                "backtest_days": getattr(result, "n_days", 0),
            }

        # Win rate
        if n_trades > 0:
            if hasattr(trades[0], "pnl"):
                wins = sum(1 for t in trades if t.pnl > 0)
                pnls = [t.pnl for t in trades]
            elif isinstance(trades[0], dict):
                wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
                pnls = [t.get("pnl", 0) for t in trades]
            else:
                wins = 0
                pnls = []
            wr = wins / n_trades * 100
            avg_pnl = np.mean(pnls) if pnls else 0
            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            wr = 0.0
            avg_pnl = 0.0
            pf = 0.0

        return StrategyResult(
            name="HMM Regime",
            sharpe=stats.get("sharpe", 0.0),
            total_return_pct=stats.get("total_return_pct", 0.0),
            annualized_return_pct=stats.get("annualized_return_pct", 0.0),
            max_drawdown_pct=stats.get("max_drawdown_pct", 0.0),
            calmar=stats.get("calmar", 0.0),
            n_trades=n_trades,
            win_rate_pct=round(wr, 1),
            avg_trade_return_pct=round(avg_pnl, 4),
            profit_factor=round(pf, 2),
            avg_hold_days=0.0,
            backtest_days=stats.get("backtest_days", 0),
            elapsed_sec=round(elapsed, 1),
        )
    except Exception as e:
        return StrategyResult(
            name="HMM Regime", sharpe=0, total_return_pct=0,
            annualized_return_pct=0, max_drawdown_pct=0, calmar=0,
            n_trades=0, win_rate_pct=0, avg_trade_return_pct=0,
            profit_factor=0, avg_hold_days=0, backtest_days=0,
            status="ERROR", notes=str(e)[:200],
            elapsed_sec=round(time.time() - t0, 1),
        )


def _run_ofi(name: str = "NIFTY", start: str = "2025-08-06",
             end: str = "2026-02-06") -> StrategyResult:
    """Run OFI intraday strategy."""
    t0 = time.time()
    try:
        cfg = OFIConfig()
        result = run_ofi(
            index_name=name,
            start_date=start,
            end_date=end,
            config=cfg,
        )
        elapsed = time.time() - t0
        return StrategyResult(
            name="OFI Intraday",
            sharpe=getattr(result, "sharpe", 0.0),
            total_return_pct=getattr(result, "total_return_pct", 0.0),
            annualized_return_pct=0.0,
            max_drawdown_pct=getattr(result, "max_drawdown_pct", 0.0),
            calmar=0.0,
            n_trades=getattr(result, "n_trades", 0),
            win_rate_pct=getattr(result, "win_rate_pct", 0.0),
            avg_trade_return_pct=getattr(result, "avg_trade_pnl_pts", 0.0),
            profit_factor=getattr(result, "profit_factor", 0.0),
            avg_hold_days=0.0,
            backtest_days=getattr(result, "n_days", 0),
            elapsed_sec=round(elapsed, 1),
        )
    except Exception as e:
        return StrategyResult(
            name="OFI Intraday", sharpe=0, total_return_pct=0,
            annualized_return_pct=0, max_drawdown_pct=0, calmar=0,
            n_trades=0, win_rate_pct=0, avg_trade_return_pct=0,
            profit_factor=0, avg_hold_days=0, backtest_days=0,
            status="ERROR", notes=str(e)[:200],
            elapsed_sec=round(time.time() - t0, 1),
        )


def _run_skew(name: str = "NIFTY", start: str = "2025-08-06",
              end: str = "2026-02-06") -> StrategyResult:
    """Run Skew Mean-Reversion strategy."""
    t0 = time.time()
    try:
        cfg = SkewConfig()
        result = run_skew(
            index_name=name,
            start_date=start,
            end_date=end,
            config=cfg,
        )
        elapsed = time.time() - t0
        return StrategyResult(
            name="Skew MR",
            sharpe=getattr(result, "sharpe", 0.0),
            total_return_pct=getattr(result, "total_return_pct", 0.0),
            annualized_return_pct=0.0,
            max_drawdown_pct=getattr(result, "max_drawdown_pct", 0.0),
            calmar=0.0,
            n_trades=getattr(result, "n_trades", 0),
            win_rate_pct=getattr(result, "win_rate_pct", 0.0),
            avg_trade_return_pct=0.0,
            profit_factor=getattr(result, "profit_factor", 0.0),
            avg_hold_days=getattr(result, "avg_hold_days", 0.0),
            backtest_days=getattr(result, "n_days", 0),
            elapsed_sec=round(elapsed, 1),
        )
    except Exception as e:
        return StrategyResult(
            name="Skew MR", sharpe=0, total_return_pct=0,
            annualized_return_pct=0, max_drawdown_pct=0, calmar=0,
            n_trades=0, win_rate_pct=0, avg_trade_return_pct=0,
            profit_factor=0, avg_hold_days=0, backtest_days=0,
            status="ERROR", notes=str(e)[:200],
            elapsed_sec=round(time.time() - t0, 1),
        )


def _run_mtfm(name: str = "NIFTY", start: str = "2025-08-06",
              end: str = "2026-02-06") -> StrategyResult:
    """Run Momentum Transformer strategy."""
    t0 = time.time()
    try:
        cfg = MomentumTFMConfig()
        result = run_mtfm(
            index_name=name,
            start_date=start,
            end_date=end,
            config=cfg,
        )
        elapsed = time.time() - t0
        return StrategyResult(
            name="Momentum TFM",
            sharpe=getattr(result, "sharpe", 0.0),
            total_return_pct=getattr(result, "total_return_pct", 0.0),
            annualized_return_pct=0.0,
            max_drawdown_pct=getattr(result, "max_drawdown_pct", 0.0),
            calmar=0.0,
            n_trades=getattr(result, "n_trades", 0),
            win_rate_pct=getattr(result, "win_rate_pct", 0.0),
            avg_trade_return_pct=0.0,
            profit_factor=getattr(result, "profit_factor", 0.0),
            avg_hold_days=0.0,
            backtest_days=getattr(result, "n_days", 0),
            elapsed_sec=round(elapsed, 1),
        )
    except Exception as e:
        return StrategyResult(
            name="Momentum TFM", sharpe=0, total_return_pct=0,
            annualized_return_pct=0, max_drawdown_pct=0, calmar=0,
            n_trades=0, win_rate_pct=0, avg_trade_return_pct=0,
            profit_factor=0, avg_hold_days=0, backtest_days=0,
            status="ERROR", notes=str(e)[:200],
            elapsed_sec=round(time.time() - t0, 1),
        )


def _run_vrp(name: str = "NIFTY", start: str = "2025-08-06",
             end: str = "2026-02-06") -> StrategyResult:
    """Run Enhanced VRP strategy."""
    t0 = time.time()
    try:
        cfg = VRPConfig()
        result = run_vrp(
            index_name=name,
            start_date=start,
            end_date=end,
            config=cfg,
        )
        elapsed = time.time() - t0
        return StrategyResult(
            name="Enhanced VRP",
            sharpe=getattr(result, "sharpe", 0.0),
            total_return_pct=getattr(result, "total_return_pct", 0.0),
            annualized_return_pct=0.0,
            max_drawdown_pct=getattr(result, "max_drawdown_pct", 0.0),
            calmar=0.0,
            n_trades=getattr(result, "n_trades", 0),
            win_rate_pct=getattr(result, "win_rate_pct", 0.0),
            avg_trade_return_pct=0.0,
            profit_factor=getattr(result, "profit_factor", 0.0),
            avg_hold_days=getattr(result, "avg_hold_days", 0.0),
            backtest_days=getattr(result, "n_days", 0),
            elapsed_sec=round(elapsed, 1),
        )
    except Exception as e:
        return StrategyResult(
            name="Enhanced VRP", sharpe=0, total_return_pct=0,
            annualized_return_pct=0, max_drawdown_pct=0, calmar=0,
            n_trades=0, win_rate_pct=0, avg_trade_return_pct=0,
            profit_factor=0, avg_hold_days=0, backtest_days=0,
            status="ERROR", notes=str(e)[:200],
            elapsed_sec=round(time.time() - t0, 1),
        )


# ---------------------------------------------------------------------------
# Scorecard generation
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "hmm": _run_hmm,
    "ofi": _run_ofi,
    "skew": _run_skew,
    "mtfm": _run_mtfm,
    "vrp": _run_vrp,
}


def generate_scorecard(results: list[StrategyResult]) -> str:
    """Generate a markdown scorecard from strategy results."""
    today = date.today().isoformat()
    lines = [
        f"# AlphaForge Scorecard — {today}",
        "",
        "## Summary",
        "",
        "| Strategy | Sharpe | Total Ret% | Ann Ret% | Max DD% | Calmar | Trades | Win% | PF | Status |",
        "|----------|--------|-----------|----------|---------|--------|--------|------|-----|--------|",
    ]

    for r in sorted(results, key=lambda x: x.sharpe, reverse=True):
        lines.append(
            f"| {r.name} | {r.sharpe:.2f} | {r.total_return_pct:.1f}% | "
            f"{r.annualized_return_pct:.1f}% | {r.max_drawdown_pct:.1f}% | "
            f"{r.calmar:.2f} | {r.n_trades} | {r.win_rate_pct:.0f}% | "
            f"{r.profit_factor:.2f} | {r.status} |"
        )

    lines.extend([
        "",
        "## Details",
        "",
    ])

    for r in results:
        lines.extend([
            f"### {r.name}",
            f"- **Sharpe**: {r.sharpe:.3f}",
            f"- **Total Return**: {r.total_return_pct:.2f}%",
            f"- **Annualized Return**: {r.annualized_return_pct:.2f}%",
            f"- **Max Drawdown**: {r.max_drawdown_pct:.2f}%",
            f"- **Calmar Ratio**: {r.calmar:.3f}",
            f"- **Trades**: {r.n_trades}",
            f"- **Win Rate**: {r.win_rate_pct:.1f}%",
            f"- **Profit Factor**: {r.profit_factor:.2f}",
            f"- **Avg Hold**: {r.avg_hold_days:.1f} days",
            f"- **Backtest Days**: {r.backtest_days}",
            f"- **Runtime**: {r.elapsed_sec:.1f}s",
        ])
        if r.notes:
            lines.append(f"- **Notes**: {r.notes}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Methodology",
        "- Sharpe: ddof=1, sqrt(252), all daily returns including flat days",
        "- Costs: 3 pts/leg (NIFTY), 5 pts/leg (BANKNIFTY) — per leg, NOT roundtrip",
        "- Execution: T+1 (signal at close day T, execute at close day T+1)",
        "- No look-ahead bias — all signals fully causal",
        "- Walk-forward validation (no in-sample optimization reported as OOS)",
        "",
        f"Generated: {datetime.now().isoformat()}",
    ])

    return "\n".join(lines)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="AlphaForge Master Runner")
    parser.add_argument(
        "--strategies",
        type=str,
        default="hmm,ofi,skew,mtfm,vrp",
        help="Comma-separated strategy keys: hmm,ofi,skew,mtfm,vrp",
    )
    parser.add_argument("--index", type=str, default="NIFTY")
    parser.add_argument("--start", type=str, default="2025-08-06")
    parser.add_argument("--end", type=str, default="2026-02-06")
    args = parser.parse_args()

    strats = [s.strip() for s in args.strategies.split(",")]
    results: list[StrategyResult] = []

    for key in strats:
        fn = STRATEGY_MAP.get(key)
        if fn is None:
            logger.warning("Unknown strategy key: %s", key)
            continue
        logger.info("=" * 60)
        logger.info("Running: %s", key.upper())
        logger.info("=" * 60)
        r = fn(name=args.index, start=args.start, end=args.end)
        results.append(r)
        logger.info(
            "  => %s: Sharpe=%.3f, Return=%.2f%%, Trades=%d, Status=%s (%.1fs)",
            r.name, r.sharpe, r.total_return_pct, r.n_trades, r.status,
            r.elapsed_sec,
        )

    # Write scorecard
    scorecard = generate_scorecard(results)
    sc_path = RESULTS_DIR / f"SCORECARD_{date.today().isoformat()}.md"
    sc_path.write_text(scorecard)
    logger.info("Scorecard written to %s", sc_path)

    # Write JSON for programmatic access
    json_path = RESULTS_DIR / f"results_{date.today().isoformat()}.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    logger.info("JSON results written to %s", json_path)

    # Print summary to stdout
    print("\n" + scorecard)
    return results


if __name__ == "__main__":
    main()
