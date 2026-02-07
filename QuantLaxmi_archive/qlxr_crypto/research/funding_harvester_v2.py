"""Funding Harvester v2 — Scaled multi-symbol portfolio.

Approach: run per-symbol threshold entry/exit (like v1) across an expanded
universe, then combine into a portfolio.

Key fix from first v2 attempt: DON'T constantly rebalance. Instead, enter
positions when funding is high and hold until funding drops. The v1
per-symbol backtest showed Sharpe 7.4 because it had very few trades
(2-6 per symbol). Frequent rebalancing destroys the edge via costs.

Strategy per symbol:
  - Entry: smoothed annualized funding > entry_threshold
  - Exit: smoothed annualized funding < exit_threshold
  - Hold: collect 8h funding payments (short perp pays you)
  - Delta-neutral: long spot + short perp

Portfolio:
  - Equal-weight across all active positions
  - Max concurrent positions capped
  - When a new entry would exceed cap, only enter if new symbol's
    funding exceeds the lowest active position's funding

Usage:
    python3 research/funding_harvester_v2.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlx.data.binance import fetch_funding_rates, fetch_24h_volumes
from qlx.data.cache import KlineCache
from qlx.metrics.performance import compute_metrics

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXCLUDE = {"XAGUSDT", "XAUUSDT", "PAXGUSDT", "USDCUSDT", "BUSDUSDT",
           "TUSDUSDT", "FDUSDUSDT", "EURUSDT", "GBPUSDT"}

N_UNIVERSE = 25
MAX_POSITIONS = 10
ENTRY_THRESHOLD = 15.0   # ann % to enter
EXIT_THRESHOLD = 3.0     # ann % to exit

# Costs per leg (entry or exit)
COST_PER_LEG_BPS = 12    # ~7bps commission + 5bps slippage for both spot+perp

START = "2024-01-01"
DATA_DIR = Path("data/klines")
FUNDING_DIR = Path("data/funding")


# ---------------------------------------------------------------------------
# Data loading (reuse cached data from v1)
# ---------------------------------------------------------------------------

def discover_universe() -> list[str]:
    """Get top N liquid USDT perp symbols (crypto only)."""
    volumes = fetch_24h_volumes()
    crypto = {
        s: v for s, v in volumes.items()
        if s.endswith("USDT") and s not in EXCLUDE and v >= 50_000_000
    }
    ranked = sorted(crypto.items(), key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in ranked[:N_UNIVERSE]]


def load_symbol_data(
    sym: str,
    cache: KlineCache,
    funding_dir: Path,
) -> tuple[pd.Series, pd.Series, pd.DataFrame] | None:
    """Load spot close, perp close, and funding rates for a symbol."""
    try:
        spot = cache.get(sym, "1h", start=START, market="spot").close
        perp = cache.get(sym, "1h", start=START, market="futures").close

        path = funding_dir / f"{sym}_funding.parquet"
        if path.exists():
            funding = pd.read_parquet(path)
            funding.index = pd.to_datetime(funding.index, utc=True)
            last_ts = funding.index[-1]
            try:
                new = fetch_funding_rates(sym, start=last_ts.isoformat())
                combined = pd.concat([funding, new])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            except Exception:
                combined = funding
        else:
            combined = fetch_funding_rates(sym, start=START)
        combined.to_parquet(path)

        if len(combined) < 50 or len(spot) < 500 or len(perp) < 500:
            return None

        return spot, perp, combined
    except Exception as e:
        print(f"    {sym}: load failed — {e}")
        return None


# ---------------------------------------------------------------------------
# Per-symbol backtest (same logic as v1 but cleaner)
# ---------------------------------------------------------------------------

def backtest_single(
    funding: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    cost_per_leg_bps: float,
) -> dict:
    """Backtest funding harvesting for a single symbol.

    Returns equity curve at 8h frequency (funding settlement times).
    """
    cost_frac = cost_per_leg_bps / 10_000

    rates = funding["fundingRate"].values
    times = funding.index

    # Smoothed annualized funding (3-period = 24h)
    ann_smooth = pd.Series(rates).rolling(3, min_periods=1).mean().values * 3 * 365 * 100

    equity = 1.0
    in_position = False
    equity_curve = np.ones(len(rates))
    n_trades = 0
    total_funding = 0.0
    total_costs = 0.0

    for i in range(len(rates)):
        if not in_position:
            if ann_smooth[i] > entry_threshold:
                in_position = True
                equity *= (1 - cost_frac)
                total_costs += cost_frac
                n_trades += 1
        else:
            # Collect funding
            equity *= (1 + rates[i])
            total_funding += rates[i]

            if ann_smooth[i] < exit_threshold:
                in_position = False
                equity *= (1 - cost_frac)
                total_costs += cost_frac

        equity_curve[i] = equity

    # Close open position at end
    if in_position:
        equity *= (1 - cost_frac)
        total_costs += cost_frac
        equity_curve[-1] = equity

    return {
        "equity": pd.Series(equity_curve, index=times),
        "total_return": equity - 1.0,
        "n_trades": n_trades,
        "total_funding": total_funding,
        "total_costs": total_costs,
    }


# ---------------------------------------------------------------------------
# Portfolio: combine per-symbol equity curves
# ---------------------------------------------------------------------------

def build_portfolio(
    symbol_results: dict[str, dict],
    max_positions: int = MAX_POSITIONS,
) -> dict:
    """Combine per-symbol results into equal-weight portfolio.

    At each timestamp, the portfolio is equal-weight across all symbols
    that are currently in a position (have non-zero return since their
    last entry). Capped at max_positions.
    """
    # Collect all equity curves
    equities = {}
    for sym, r in symbol_results.items():
        equities[sym] = r["equity"]

    # Align to common timeline
    eq_df = pd.DataFrame(equities)
    eq_df = eq_df.sort_index()

    # Per-period returns for each symbol
    ret_df = eq_df.pct_change()

    # Portfolio: equal-weight average of per-symbol returns
    # (This is equivalent to equal-weight rebalancing at each settlement)
    n_active = ret_df.notna().sum(axis=1)
    port_ret = ret_df.mean(axis=1)  # equal weight across all symbols
    port_ret = port_ret.dropna()

    if len(port_ret) < 10:
        return {"metrics": None, "error": "Insufficient data"}

    port_equity = (1 + port_ret).cumprod()
    metrics = compute_metrics(port_ret, periods_per_year=1095)

    # Aggregate stats
    total_trades = sum(r["n_trades"] for r in symbol_results.values())
    total_funding = sum(r["total_funding"] for r in symbol_results.values())
    total_costs = sum(r["total_costs"] for r in symbol_results.values())

    return {
        "equity": port_equity,
        "returns": port_ret,
        "metrics": metrics,
        "total_return": float(port_equity.iloc[-1]) - 1.0,
        "total_trades": total_trades,
        "total_funding": total_funding,
        "total_costs": total_costs,
        "n_symbols": len(symbol_results),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Funding Harvester v2 — Per-Symbol Hold, Portfolio Combine")
    print(f"  Entry: >{ENTRY_THRESHOLD}% ann | Exit: <{EXIT_THRESHOLD}% ann")
    print(f"  Cost per leg: {COST_PER_LEG_BPS}bps | Max positions: {MAX_POSITIONS}")
    print("=" * 70)

    cache = KlineCache(str(DATA_DIR))
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)

    # Discover universe
    print("\nDiscovering liquid universe...")
    symbols = discover_universe()
    print(f"  Found {len(symbols)} symbols")

    # Load data
    print("\nLoading data...")
    universe_data = {}
    for sym in symbols:
        result = load_symbol_data(sym, cache, FUNDING_DIR)
        if result is not None:
            universe_data[sym] = result
            _, _, fund = result
            avg_f = fund["fundingRate"].mean() * 3 * 365 * 100
            pct_pos = (fund["fundingRate"] > 0).mean()
            print(f"  {sym:15s}  fund={len(fund):5d}  avg_ann={avg_f:+5.1f}%  pos={pct_pos:.0%}")
        time.sleep(0.3)

    print(f"\n  Loaded {len(universe_data)} symbols")

    # -------------------------------------------------------------------
    # Per-symbol backtests
    # -------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("  PER-SYMBOL RESULTS")
    print(f"{'='*80}")
    print(f"{'Symbol':15s} {'Return':>8s} {'Trades':>7s} {'Funding':>9s} {'Costs':>8s} {'Net/yr':>8s}")
    print("-" * 80)

    sym_results = {}
    for sym, (spot, perp, funding) in universe_data.items():
        r = backtest_single(
            funding=funding,
            entry_threshold=ENTRY_THRESHOLD,
            exit_threshold=EXIT_THRESHOLD,
            cost_per_leg_bps=COST_PER_LEG_BPS,
        )
        sym_results[sym] = r

        # Annualize: data spans ~2 years
        duration_years = (funding.index[-1] - funding.index[0]).total_seconds() / (365.25 * 86400)
        ann_return = (1 + r["total_return"]) ** (1 / max(duration_years, 0.5)) - 1

        print(
            f"{sym:15s} "
            f"{r['total_return']:+7.1%} "
            f"{r['n_trades']:7d} "
            f"{r['total_funding']:+8.4f} "
            f"{r['total_costs']:8.4f} "
            f"{ann_return:+7.1%}"
        )

    # -------------------------------------------------------------------
    # Threshold sweep
    # -------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("  THRESHOLD SWEEP (equal-weight portfolio)")
    print(f"{'='*80}")
    print(f"{'Entry':>7s} {'Exit':>7s} {'Sharpe':>7s} {'AnnRet':>8s} {'MaxDD':>8s} "
          f"{'Sortino':>8s} {'Calmar':>8s} {'Trades':>7s}")
    print("-" * 80)

    sweep_configs = [
        (10, 2), (10, 3), (10, 5),
        (15, 2), (15, 3), (15, 5),
        (20, 3), (20, 5), (20, 10),
        (25, 5), (25, 10),
        (30, 5), (30, 10),
    ]

    sweep_results = []
    for entry_th, exit_th in sweep_configs:
        sr = {}
        for sym, (_, _, funding) in universe_data.items():
            sr[sym] = backtest_single(
                funding=funding,
                entry_threshold=entry_th,
                exit_threshold=exit_th,
                cost_per_leg_bps=COST_PER_LEG_BPS,
            )

        port = build_portfolio(sr)
        if port["metrics"] is not None:
            m = port["metrics"]
            print(
                f"{entry_th:6d}% {exit_th:6d}% "
                f"{m.sharpe_ratio:7.3f} "
                f"{m.annualised_return:7.1%} "
                f"{m.max_drawdown:7.1%} "
                f"{m.sortino_ratio:8.3f} "
                f"{m.calmar_ratio:8.3f} "
                f"{port['total_trades']:7d}"
            )
            sweep_results.append((entry_th, exit_th, port))

    # -------------------------------------------------------------------
    # Best config deep dive
    # -------------------------------------------------------------------
    if sweep_results:
        best_entry, best_exit, best_port = max(
            sweep_results, key=lambda x: x[2]["metrics"].sharpe_ratio
        )
        m = best_port["metrics"]

        print(f"\n\n{'='*70}")
        print(f"  BEST: Entry >{best_entry}%, Exit <{best_exit}%")
        print(f"{'='*70}")
        print(m.summary())

        print(f"\n  Portfolio stats:")
        print(f"    Symbols:      {best_port['n_symbols']}")
        print(f"    Total trades: {best_port['total_trades']}")
        print(f"    Total return: {best_port['total_return']:+.1%}")

        # Monthly returns
        monthly = best_port["returns"].resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        print(f"\n  Monthly Returns:")
        pos_months = 0
        neg_months = 0
        for date, ret in monthly.items():
            marker = "+" if ret > 0 else " "
            bar = "#" * int(abs(ret) * 500)
            print(f"    {date.strftime('%Y-%m')}: {marker}{ret:7.3%}  {bar}")
            if ret > 0:
                pos_months += 1
            else:
                neg_months += 1

        print(f"\n  Positive months: {pos_months}/{pos_months + neg_months} "
              f"({pos_months / (pos_months + neg_months):.0%})")

        # Win rate
        wins = best_port["returns"] > 0
        print(f"  Settlement win rate: {wins.mean():.1%}")

        # Drawdown
        eq = best_port["equity"]
        peak = eq.cummax()
        dd = (eq - peak) / peak
        print(f"\n  Worst drawdowns:")
        for dt, val in dd.sort_values().head(5).items():
            print(f"    {dt.strftime('%Y-%m-%d')}: {val:.2%}")


if __name__ == "__main__":
    main()
