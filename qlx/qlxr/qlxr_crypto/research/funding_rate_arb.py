"""Funding rate arbitrage research — spot long + perp short carry trade.

The most proven mechanical alpha source in crypto:
  - Perpetual futures consistently trade at a premium to spot
  - Longs pay shorts a funding rate (every 8h on Binance)
  - Strategy: long spot + short perp = delta-neutral, collect funding
  - Entry when annualised funding > threshold, exit when it drops

This is NOT directional — it's a carry trade that captures the cost
of leverage paid by speculative longs.

Pipeline:
  1. Download spot + perp 1h klines and historical funding rates
  2. Compute basis (perp premium), annualized carry
  3. Backtest: enter when carry > entry_threshold, exit when < exit_threshold
  4. Realistic costs: spot + perp commissions, basis slippage
  5. Report per-symbol and aggregate metrics

Usage:
    python3 research/funding_rate_arb.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlx.data.binance import fetch_klines, fetch_funding_rates
from qlx.data.cache import KlineCache
from qlx.metrics.performance import compute_metrics

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
]

START = "2024-01-01"

# Cost assumptions (conservative for retail)
SPOT_COMMISSION_BPS = 10    # maker/taker on spot
PERP_COMMISSION_BPS = 4     # maker rate on futures
SLIPPAGE_BPS = 5            # per side, per leg
ENTRY_COST_BPS = (SPOT_COMMISSION_BPS + PERP_COMMISSION_BPS) / 2 + SLIPPAGE_BPS * 2  # both legs
EXIT_COST_BPS = ENTRY_COST_BPS

# Strategy thresholds (annualized funding rate)
ENTRY_THRESHOLD_PCT = 15.0   # enter when ann. funding > 15%
EXIT_THRESHOLD_PCT = 3.0     # exit when ann. funding < 3%
MAX_BASIS_BPS = 200          # don't enter if basis > 2% (too expensive)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_funding_rates(symbol: str, cache_dir: Path) -> pd.DataFrame:
    """Load funding rates from cache or download."""
    path = cache_dir / f"{symbol}_funding.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        # Fetch incremental
        last_ts = df.index[-1]
        try:
            new = fetch_funding_rates(symbol, start=last_ts.isoformat())
            combined = pd.concat([df, new])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        except Exception:
            combined = df
    else:
        combined = fetch_funding_rates(symbol, start=START)

    combined.to_parquet(path)
    return combined


def load_perp_klines(symbol: str, cache: KlineCache) -> pd.DataFrame:
    """Load perp klines via cache."""
    ohlcv = cache.get(symbol, "1h", start=START, market="futures")
    return ohlcv.df


# ---------------------------------------------------------------------------
# Strategy logic
# ---------------------------------------------------------------------------

def compute_carry_signals(
    spot_close: pd.Series,
    perp_close: pd.Series,
    funding: pd.DataFrame,
) -> pd.DataFrame:
    """Compute basis, carry, and entry/exit signals at 8h frequency.

    Returns DataFrame indexed at funding settlement times with:
    - basis_bps: (perp/spot - 1) in bps
    - funding_rate: raw 8h funding rate
    - ann_funding_pct: annualized (funding_rate * 3 * 365 * 100)
    - signal: 1 = in position, 0 = flat
    """
    # Align to funding settlement times (every 8h)
    fund_times = funding.index

    # For each funding time, find the nearest spot/perp close
    # (within the same hour)
    basis_bps_list = []
    for ft in fund_times:
        # Find the closest hour bar <= funding time
        spot_slice = spot_close[spot_close.index <= ft]
        perp_slice = perp_close[perp_close.index <= ft]
        if len(spot_slice) == 0 or len(perp_slice) == 0:
            basis_bps_list.append(np.nan)
            continue
        s = spot_slice.iloc[-1]
        p = perp_slice.iloc[-1]
        if s > 0:
            basis_bps_list.append((p / s - 1.0) * 10_000)
        else:
            basis_bps_list.append(np.nan)

    signals = pd.DataFrame(index=fund_times)
    signals["basis_bps"] = basis_bps_list
    signals["funding_rate"] = funding["fundingRate"]
    signals["ann_funding_pct"] = signals["funding_rate"] * 3 * 365 * 100

    # Rolling average of annualized funding (smooth out single-period spikes)
    signals["ann_funding_smooth"] = signals["ann_funding_pct"].rolling(3, min_periods=1).mean()

    return signals


def backtest_funding_arb(
    signals: pd.DataFrame,
    spot_close: pd.Series,
    perp_close: pd.Series,
    entry_threshold: float = ENTRY_THRESHOLD_PCT,
    exit_threshold: float = EXIT_THRESHOLD_PCT,
    max_basis_bps: float = MAX_BASIS_BPS,
    entry_cost_bps: float = ENTRY_COST_BPS,
    exit_cost_bps: float = EXIT_COST_BPS,
) -> dict:
    """Backtest the funding rate harvesting strategy.

    Position: long spot + short perp (delta neutral).
    PnL per period = funding_collected - basis_change - costs.

    Returns dict with equity curve, metrics, trade log.
    """
    in_position = False
    equity = 1.0
    equity_curve = []
    equity_dates = []
    trades = []
    entry_basis = 0.0
    total_funding_collected = 0.0
    total_costs = 0.0

    for i in range(len(signals)):
        row = signals.iloc[i]
        dt = signals.index[i]
        ann_f = row["ann_funding_smooth"]
        basis = row["basis_bps"]
        raw_funding = row["funding_rate"]

        if pd.isna(ann_f) or pd.isna(basis):
            equity_curve.append(equity)
            equity_dates.append(dt)
            continue

        if not in_position:
            # Entry condition
            if ann_f > entry_threshold and abs(basis) < max_basis_bps:
                in_position = True
                entry_basis = basis
                entry_dt = dt
                cost = entry_cost_bps / 10_000
                equity *= (1 - cost)
                total_costs += cost
        else:
            # Collect funding: short perp receives positive funding, pays negative
            # Raw funding rate is what longs pay shorts
            funding_pnl = raw_funding  # as fraction of position
            equity *= (1 + funding_pnl)
            total_funding_collected += funding_pnl

            # Basis change PnL: we're short perp, so basis decrease = profit
            # (Our position loses if basis widens, gains if it narrows)
            prev_basis = signals.iloc[i - 1]["basis_bps"] if i > 0 else basis
            basis_change_frac = (basis - prev_basis) / 10_000
            # Short perp: we lose when perp rises relative to spot
            equity *= (1 - basis_change_frac)

            # Exit condition
            if ann_f < exit_threshold:
                cost = exit_cost_bps / 10_000
                equity *= (1 - cost)
                total_costs += cost
                trades.append({
                    "entry_time": entry_dt,
                    "exit_time": dt,
                    "entry_basis_bps": entry_basis,
                    "exit_basis_bps": basis,
                    "periods_held": (dt - entry_dt).total_seconds() / 3600,
                })
                in_position = False

        equity_curve.append(equity)
        equity_dates.append(dt)

    # Close any open position at end
    if in_position:
        cost = exit_cost_bps / 10_000
        equity *= (1 - cost)
        total_costs += cost
        equity_curve[-1] = equity
        trades.append({
            "entry_time": entry_dt,
            "exit_time": signals.index[-1],
            "entry_basis_bps": entry_basis,
            "exit_basis_bps": signals.iloc[-1]["basis_bps"],
            "periods_held": (signals.index[-1] - entry_dt).total_seconds() / 3600,
        })

    eq_series = pd.Series(equity_curve, index=equity_dates)
    returns = eq_series.pct_change().dropna()

    # Time in market
    in_market_bars = sum(1 for i in range(len(signals)) if i > 0 and
                        equity_curve[i] != equity_curve[i-1] if i < len(equity_curve))

    return {
        "equity": eq_series,
        "returns": returns,
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
        "total_funding_collected": total_funding_collected,
        "total_costs": total_costs,
        "total_return": equity - 1.0,
        "n_trades": len(trades),
        "time_in_market_pct": in_market_bars / max(len(signals), 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Funding Rate Arbitrage Research")
    print(f"  Entry: >{ENTRY_THRESHOLD_PCT}% ann | Exit: <{EXIT_THRESHOLD_PCT}% ann")
    print(f"  Costs: {ENTRY_COST_BPS:.0f}bps entry + {EXIT_COST_BPS:.0f}bps exit per leg")
    print("=" * 70)

    cache = KlineCache("data/klines")
    funding_dir = Path("data/funding")
    funding_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    agg_returns = None

    for sym in SYMBOLS:
        print(f"\n{'—'*60}")
        print(f"  {sym}")
        print(f"{'—'*60}")

        try:
            # Load spot klines (from existing cache)
            spot_ohlcv = cache.get(sym, "1h", start=START, market="spot")
            spot_close = spot_ohlcv.close
            print(f"  Spot:    {len(spot_close)} bars")

            # Load perp klines
            perp_ohlcv = cache.get(sym, "1h", start=START, market="futures")
            perp_close = perp_ohlcv.close
            print(f"  Perp:    {len(perp_close)} bars")

            # Load funding rates
            funding = load_funding_rates(sym, funding_dir)
            print(f"  Funding: {len(funding)} settlements")

            if len(funding) < 10:
                print(f"  SKIP: insufficient funding data")
                continue

            # Compute signals
            signals = compute_carry_signals(spot_close, perp_close, funding)
            signals = signals.dropna(subset=["ann_funding_pct"])
            print(f"  Signals: {len(signals)} periods")

            # Stats
            avg_funding = signals["ann_funding_pct"].mean()
            median_funding = signals["ann_funding_pct"].median()
            pct_positive = (signals["funding_rate"] > 0).mean()
            avg_basis = signals["basis_bps"].mean()
            print(f"  Avg ann funding:    {avg_funding:.1f}%")
            print(f"  Median ann funding: {median_funding:.1f}%")
            print(f"  Funding positive:   {pct_positive:.1%}")
            print(f"  Avg basis:          {avg_basis:.1f} bps")

            # Backtest
            result = backtest_funding_arb(signals, spot_close, perp_close)

            if len(result["returns"]) > 10:
                # 3 funding periods per day = 1095 per year
                metrics = compute_metrics(result["returns"], periods_per_year=1095)
                result["metrics"] = metrics

                print(f"\n  Results:")
                print(f"    Total return:   {result['total_return']:+.1%}")
                print(f"    Sharpe:         {metrics.sharpe_ratio:.3f}")
                print(f"    Ann return:     {metrics.annualised_return:.1%}")
                print(f"    Max DD:         {metrics.max_drawdown:.1%}")
                print(f"    Sortino:        {metrics.sortino_ratio:.3f}")
                print(f"    Trades:         {result['n_trades']}")
                print(f"    Funding earned: {result['total_funding_collected']:.4f}")
                print(f"    Costs paid:     {result['total_costs']:.4f}")
            else:
                print(f"  SKIP: not enough return data")
                result["metrics"] = None

            result["symbol"] = sym
            result["avg_funding"] = avg_funding
            result["avg_basis"] = avg_basis
            all_results.append(result)

            # Aggregate portfolio: equal-weight across symbols
            if agg_returns is None:
                agg_returns = result["returns"].copy()
            else:
                # Align and average
                combined = pd.concat([agg_returns, result["returns"]], axis=1)
                agg_returns = combined.mean(axis=1)

            time.sleep(0.3)  # Rate limit courtesy

        except Exception as e:
            print(f"  ERROR: {e}")

    # -------------------------------------------------------------------
    # Universe summary
    # -------------------------------------------------------------------
    good_results = [r for r in all_results if r.get("metrics") is not None]

    if not good_results:
        print("\nNo valid results to report.")
        return

    print(f"\n\n{'='*90}")
    print("  UNIVERSE SUMMARY — Funding Rate Arbitrage")
    print(f"{'='*90}")
    print(
        f"{'Symbol':12s} {'Sharpe':>7s} {'AnnRet':>8s} {'MaxDD':>8s} "
        f"{'Sortino':>8s} {'Trades':>7s} {'AvgFund':>8s} {'AvgBasis':>8s}"
    )
    print("-" * 90)

    for r in sorted(good_results, key=lambda x: x["metrics"].sharpe_ratio, reverse=True):
        m = r["metrics"]
        print(
            f"{r['symbol']:12s} "
            f"{m.sharpe_ratio:7.3f} "
            f"{m.annualised_return:7.1%} "
            f"{m.max_drawdown:7.1%} "
            f"{m.sortino_ratio:8.3f} "
            f"{r['n_trades']:7d} "
            f"{r['avg_funding']:7.1f}% "
            f"{r['avg_basis']:7.1f}bp"
        )

    # Average
    avg_sharpe = np.mean([r["metrics"].sharpe_ratio for r in good_results])
    avg_return = np.mean([r["metrics"].annualised_return for r in good_results])
    avg_dd = np.mean([r["metrics"].max_drawdown for r in good_results])
    print("-" * 90)
    print(f"{'AVERAGE':12s} {avg_sharpe:7.3f} {avg_return:7.1%} {avg_dd:7.1%}")

    # Portfolio (equal-weight across all symbols)
    if agg_returns is not None and len(agg_returns) > 10:
        port_metrics = compute_metrics(agg_returns.dropna(), periods_per_year=1095)
        print(f"\n{'PORTFOLIO (EW)':12s} {port_metrics.sharpe_ratio:7.3f} "
              f"{port_metrics.annualised_return:7.1%} {port_metrics.max_drawdown:7.1%}")

    # -------------------------------------------------------------------
    # Sweep: different thresholds
    # -------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("  THRESHOLD SWEEP (aggregate across universe)")
    print(f"{'='*80}")
    print(f"{'Entry%':>7s} {'Exit%':>7s} {'Sharpe':>7s} {'AnnRet':>8s} {'MaxDD':>8s} {'Trades':>7s}")
    print("-" * 80)

    # Precompute signals for all symbols
    all_signals = {}
    for r in all_results:
        sym = r["symbol"]
        if sym in [rr["symbol"] for rr in good_results]:
            # Re-fetch from our computed data
            spot_close = cache.get(sym, "1h", start=START, market="spot").close
            perp_close = cache.get(sym, "1h", start=START, market="futures").close
            funding = load_funding_rates(sym, funding_dir)
            signals = compute_carry_signals(spot_close, perp_close, funding)
            all_signals[sym] = (signals.dropna(subset=["ann_funding_pct"]), spot_close, perp_close)

    threshold_configs = [
        (5,  2),
        (10, 3),
        (15, 3),
        (15, 5),
        (20, 5),
        (20, 10),
        (25, 5),
        (30, 10),
    ]

    for entry_th, exit_th in threshold_configs:
        sweep_returns = None
        total_trades = 0
        for sym, (sig, sc, pc) in all_signals.items():
            r = backtest_funding_arb(sig, sc, pc,
                                     entry_threshold=entry_th,
                                     exit_threshold=exit_th)
            total_trades += r["n_trades"]
            if len(r["returns"]) > 0:
                if sweep_returns is None:
                    sweep_returns = r["returns"].copy()
                else:
                    combined = pd.concat([sweep_returns, r["returns"]], axis=1)
                    sweep_returns = combined.mean(axis=1)

        if sweep_returns is not None and len(sweep_returns) > 10:
            sm = compute_metrics(sweep_returns.dropna(), periods_per_year=1095)
            print(
                f"{entry_th:6d}% {exit_th:6d}% "
                f"{sm.sharpe_ratio:7.3f} "
                f"{sm.annualised_return:7.1%} "
                f"{sm.max_drawdown:7.1%} "
                f"{total_trades:7d}"
            )
        else:
            print(f"{entry_th:6d}% {exit_th:6d}%  (insufficient data)")


if __name__ == "__main__":
    main()
