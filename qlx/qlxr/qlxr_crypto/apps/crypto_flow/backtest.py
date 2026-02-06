"""CLRS Backtester — historical simulation using klines and funding data.

Fetches historical data from Binance REST and runs the strategy logic
offline.  Uses the same features (VPIN, OFI proxy, Kyle's Lambda) and
signal rules as the live paper trader, but in batch mode.

Usage:
    python -m apps.crypto_flow.backtest --symbol BTCUSDT --days 30
    python -m apps.crypto_flow.backtest --multi --days 90

Note: Hawkes calibration in backtest uses kline trade counts as a proxy
for actual trade timestamps (which aren't available historically at scale).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from apps.crypto_flow.features import (
    VPINState,
    amihud_illiquidity,
    compute_vpin_series,
    funding_pca,
    kyles_lambda,
)
from apps.crypto_flow.scanner import annualize_funding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

FAPI_BASE = "https://fapi.binance.com"


def fetch_historical_klines(
    symbol: str,
    interval: str = "1h",
    days: int = 90,
) -> pd.DataFrame:
    """Fetch historical klines with pagination."""
    import requests

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    url = f"{FAPI_BASE}/fapi/v1/klines"
    all_rows = []
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 1500,
        "startTime": int(start.timestamp() * 1000),
    }

    while True:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        last_close = rows[-1][6]
        if last_close >= int(end.timestamp() * 1000):
            break
        params["startTime"] = last_close + 1
        if len(rows) < 1500:
            break

    df = pd.DataFrame(all_rows, columns=[
        "Open_time", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "QuoteVolume", "Trade_count",
        "Taker_buy_base", "Taker_buy_quote", "Ignore",
    ])
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms", utc=True)
    for col in ("Open", "High", "Low", "Close", "Volume", "QuoteVolume",
                "Trade_count", "Taker_buy_base", "Taker_buy_quote"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("Open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def fetch_historical_funding(
    symbol: str,
    days: int = 90,
) -> pd.DataFrame:
    """Fetch historical funding rates with pagination."""
    import requests

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    url = f"{FAPI_BASE}/fapi/v1/fundingRate"
    all_rows = []
    params = {
        "symbol": symbol,
        "limit": 1000,
        "startTime": int(start.timestamp() * 1000),
    }

    while True:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < 1000:
            break
        params["startTime"] = rows[-1]["fundingTime"] + 1

    if not all_rows:
        return pd.DataFrame(columns=["fundingRate"])

    df = pd.DataFrame(all_rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.set_index("fundingTime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["fundingRate"]]


# ---------------------------------------------------------------------------
# Single-symbol VPIN-filtered carry backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Backtest parameters."""

    # VPIN — disabled by default (hourly VPIN is too noisy, causes whipsaw).
    # Enable only with tick-level data or very conservative thresholds.
    vpin_bucket_size: float = 50_000
    vpin_n_buckets: int = 50
    vpin_entry_max: float = 0.99       # effectively disabled
    vpin_exit_threshold: float = 0.99  # effectively disabled

    # Carry thresholds
    carry_entry_ann_pct: float = 20.0  # enter when smoothed ann funding > 20%
    carry_exit_ann_pct: float = 3.0    # exit when < 3%

    # Kyle's Lambda
    kyle_window: int = 50

    # Cost
    cost_per_leg_bps: float = 8.0

    # Funding smoothing (3 = ~24h at 8h settlement)
    funding_smooth_window: int = 3


@dataclass
class BacktestResult:
    """Result of a single-symbol backtest."""

    symbol: str
    total_return_pct: float
    ann_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    n_trades: int
    n_settlements: int
    funding_earned: float
    costs_paid: float
    avg_hold_days: float
    equity_curve: list[float] = field(default_factory=list)
    dates: list = field(default_factory=list)


def run_single_symbol_backtest(
    symbol: str,
    days: int = 90,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run VPIN-filtered carry backtest on one symbol.

    Logic:
    1. Compute VPIN from kline data
    2. Map funding settlement times to kline bars
    3. Enter carry when: funding > threshold AND VPIN < threshold
    4. Exit when: funding < exit_threshold OR VPIN > exit_threshold
    5. Credit funding at each 8h settlement while in position
    """
    if config is None:
        config = BacktestConfig()

    logger.info("Backtest %s: fetching %d days of data...", symbol, days)

    # Fetch data
    klines = fetch_historical_klines(symbol, interval="1h", days=days)
    funding = fetch_historical_funding(symbol, days=days)

    if len(klines) < 100 or len(funding) < 10:
        logger.warning("%s: insufficient data (klines=%d, funding=%d)",
                      symbol, len(klines), len(funding))
        return BacktestResult(symbol=symbol, total_return_pct=0, ann_return_pct=0,
                             max_drawdown_pct=0, sharpe=0, n_trades=0,
                             n_settlements=0, funding_earned=0, costs_paid=0,
                             avg_hold_days=0)

    # Compute VPIN
    prices = klines["Close"].values
    volumes_usd = klines["QuoteVolume"].values
    vpin = compute_vpin_series(
        prices, volumes_usd,
        bucket_size=config.vpin_bucket_size,
        n_buckets=config.vpin_n_buckets,
    )

    # Kyle's Lambda
    lam = kyles_lambda(prices, volumes_usd, window=config.kyle_window)

    # Map funding to nearest kline bar
    funding_ann = {}
    fund_history = []
    for ts, row in funding.iterrows():
        rate = row["fundingRate"]
        ann = annualize_funding(rate)
        fund_history.append(rate)
        # Smooth
        if len(fund_history) >= config.funding_smooth_window:
            smooth_rate = np.mean(fund_history[-config.funding_smooth_window:])
        else:
            smooth_rate = rate
        smooth_ann = annualize_funding(smooth_rate)

        # Find nearest kline bar
        idx = klines.index.searchsorted(ts)
        if 0 <= idx < len(klines):
            funding_ann[idx] = {"raw": ann, "smooth": smooth_ann, "rate": rate}

    # Simulate
    equity = 1.0
    in_position = False
    entry_bar = 0
    cost_frac = config.cost_per_leg_bps / 10_000
    n_trades = 0
    n_settlements = 0
    funding_earned = 0.0
    costs_paid = 0.0
    hold_durations = []

    equity_curve = []
    dates = []

    for i in range(len(klines)):
        # Check for funding settlement at this bar
        if i in funding_ann and in_position:
            rate = funding_ann[i]["rate"]
            funding_earned += rate
            equity *= (1 + rate)
            n_settlements += 1

        # Get current VPIN and funding
        current_vpin = vpin[i] if not np.isnan(vpin[i]) else 0.5

        # Find most recent funding observation
        recent_funding = 0.0
        for j in sorted(funding_ann.keys(), reverse=True):
            if j <= i:
                recent_funding = funding_ann[j]["smooth"]
                break

        # Entry logic
        if not in_position:
            if (recent_funding > config.carry_entry_ann_pct
                    and current_vpin < config.vpin_entry_max):
                # Enter carry
                in_position = True
                entry_bar = i
                equity *= (1 - cost_frac)
                costs_paid += cost_frac
                n_trades += 1

        # Exit logic
        elif in_position:
            should_exit = False
            if current_vpin > config.vpin_exit_threshold:
                should_exit = True
            elif recent_funding < config.carry_exit_ann_pct:
                should_exit = True

            if should_exit:
                equity *= (1 - cost_frac)
                costs_paid += cost_frac
                in_position = False
                hold_bars = i - entry_bar
                hold_durations.append(hold_bars / 24)  # convert to days

        equity_curve.append(equity)
        dates.append(klines.index[i])

    # Close any open position at end
    if in_position:
        equity *= (1 - cost_frac)
        costs_paid += cost_frac
        hold_durations.append((len(klines) - entry_bar) / 24)

    # Compute metrics
    total_ret = equity - 1.0
    eq_arr = np.array(equity_curve)
    n_days = days
    ann_ret = (equity ** (365 / max(n_days, 1)) - 1) * 100

    # Max drawdown
    peak = eq_arr[0]
    max_dd = 0.0
    for v in eq_arr:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe
    returns = np.diff(eq_arr) / eq_arr[:-1]
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(24 * 365)
    else:
        sharpe = 0.0

    avg_hold = np.mean(hold_durations) if hold_durations else 0.0

    return BacktestResult(
        symbol=symbol,
        total_return_pct=total_ret * 100,
        ann_return_pct=ann_ret,
        max_drawdown_pct=max_dd * 100,
        sharpe=sharpe,
        n_trades=n_trades,
        n_settlements=n_settlements,
        funding_earned=funding_earned,
        costs_paid=costs_paid,
        avg_hold_days=avg_hold,
        equity_curve=equity_curve,
        dates=[d.isoformat() for d in dates],
    )


# ---------------------------------------------------------------------------
# Multi-symbol cross-sectional backtest
# ---------------------------------------------------------------------------

def scan_high_funding_symbols(
    min_ann_pct: float = 15.0,
    min_volume_usd: float = 10_000_000,
    top_n: int = 20,
) -> list[str]:
    """Scan Binance for symbols with high POSITIVE funding rates.

    Carry strategy = short perp + long spot.  We EARN when funding is POSITIVE
    (longs pay shorts).  Negative funding means we'd PAY, so we filter those out.

    Current rate is just a snapshot — many of these symbols spike briefly.
    We scan broadly and let the backtest engine decide when to enter/exit
    based on historical rates.
    """
    import requests

    EXCLUDE = {
        "USDCUSDT", "BUSDUSDT", "FDUSDUSDT", "TUSDUSDT",
        "XAUUSDT", "XAGUSDT", "EURUSDT", "GBPUSDT", "PAXGUSDT",
    }

    # Get funding rates
    resp = requests.get(f"{FAPI_BASE}/fapi/v1/premiumIndex", timeout=15)
    resp.raise_for_status()
    premium = resp.json()

    # Get volumes
    resp2 = requests.get(f"{FAPI_BASE}/fapi/v1/ticker/24hr", timeout=15)
    resp2.raise_for_status()
    volumes = {t["symbol"]: float(t.get("quoteVolume", 0)) for t in resp2.json()}

    candidates = []
    for item in premium:
        sym = item.get("symbol", "")
        if not sym.endswith("USDT") or sym in EXCLUDE:
            continue
        try:
            rate = float(item.get("lastFundingRate", 0))
            ann = rate * 1095 * 100  # SIGNED annualized % (positive = we earn)
            vol = volumes.get(sym, 0)
            if vol >= min_volume_usd:
                candidates.append((sym, ann, vol, rate))
        except (ValueError, TypeError):
            pass

    # Sort by POSITIVE funding rate descending (we want symbols that pay US)
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Take top N with positive funding above threshold
    positive = [(s, a, v, r) for s, a, v, r in candidates if a >= min_ann_pct]
    selected = [c[0] for c in positive[:top_n]]

    # If not enough positive, also include symbols with high absolute rates
    # (they may have oscillated positive historically)
    if len(selected) < top_n:
        remaining = top_n - len(selected)
        by_abs = sorted(candidates, key=lambda x: abs(x[1]), reverse=True)
        for sym, ann, vol, rate in by_abs:
            if sym not in selected and abs(ann) >= min_ann_pct:
                selected.append(sym)
                if len(selected) >= top_n:
                    break

    print(f"  Scanned {len(premium)} symbols, {len(positive)} with positive "
          f"funding >{min_ann_pct}%")
    print(f"  Selected {len(selected)} symbols for backtest:")
    sym_map = {c[0]: c for c in candidates}
    for sym in selected:
        if sym in sym_map:
            _, ann, vol, _ = sym_map[sym]
            print(f"    {sym:15s}  ann={ann:+7.1f}%  vol=${vol/1e6:.0f}M")

    return selected


def run_multi_symbol_backtest(
    symbols: list[str] | None = None,
    days: int = 90,
    config: BacktestConfig | None = None,
    top_n: int = 20,
) -> list[BacktestResult]:
    """Run backtest across multiple symbols.

    If symbols not provided, scans for high-funding symbols (NOT top volume).
    """
    if config is None:
        config = BacktestConfig()

    if symbols is None:
        symbols = scan_high_funding_symbols(
            min_ann_pct=config.carry_entry_ann_pct * 0.75,  # scan slightly below entry
            min_volume_usd=10_000_000,
            top_n=top_n,
        )

    if not symbols:
        print("  No symbols found above funding threshold.")
        return []

    print(f"\n  Running backtest on {len(symbols)} symbols, {days} days...")
    print(f"  {'Symbol':15s} {'Return':>8s} {'Sharpe':>8s} {'Trades':>7s} "
          f"{'Settle':>7s} {'Funding':>10s} {'MaxDD':>7s} {'Hold':>6s}")
    print(f"  {'-' * 72}")

    results = []
    for sym in symbols:
        try:
            result = run_single_symbol_backtest(sym, days=days, config=config)
            results.append(result)
            if result.n_trades > 0:
                print(f"  {sym:15s} {result.total_return_pct:+7.2f}% "
                      f"{result.sharpe:+7.2f} {result.n_trades:7d} "
                      f"{result.n_settlements:7d} {result.funding_earned:+10.6f} "
                      f"{result.max_drawdown_pct:6.2f}% {result.avg_hold_days:5.1f}d")
            else:
                print(f"  {sym:15s}    (no trades — funding below entry threshold)")
        except Exception as e:
            logger.warning("Backtest failed for %s: %s", sym, e)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CLRS Backtester")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single symbol to backtest (default: auto-scan)")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--multi", action="store_true",
                        help="Run multi-symbol backtest (scans by funding rate)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top high-funding symbols (default: 10)")
    parser.add_argument("--carry-entry", type=float, default=20.0,
                        help="Entry threshold: ann funding %% (default: 20)")
    parser.add_argument("--carry-exit", type=float, default=3.0,
                        help="Exit threshold: ann funding %% (default: 3)")
    parser.add_argument("--cost-bps", type=float, default=8.0,
                        help="Cost per leg in bps (default: 8)")
    parser.add_argument("--vpin-entry", type=float, default=0.99,
                        help="VPIN entry threshold (default: 0.99 = disabled)")
    parser.add_argument("--vpin-exit", type=float, default=0.99,
                        help="VPIN exit threshold (default: 0.99 = disabled)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    config = BacktestConfig(
        vpin_entry_max=args.vpin_entry,
        vpin_exit_threshold=args.vpin_exit,
        carry_entry_ann_pct=args.carry_entry,
        carry_exit_ann_pct=args.carry_exit,
        cost_per_leg_bps=args.cost_bps,
    )

    vpin_status = "disabled" if config.vpin_entry_max >= 0.95 else f"entry<{config.vpin_entry_max}, exit>{config.vpin_exit_threshold}"
    print(f"\nCLRS Backtest — {args.days} days")
    print(f"Carry: entry>{config.carry_entry_ann_pct}%, exit<{config.carry_exit_ann_pct}%")
    print(f"Cost: {config.cost_per_leg_bps}bps/leg | VPIN: {vpin_status}")
    print("=" * 78)

    if args.multi or args.symbol is None:
        # Default: scan for high-funding symbols
        results = run_multi_symbol_backtest(days=args.days, config=config,
                                            top_n=args.top_n)
        if results:
            active = [r for r in results if r.n_trades > 0]
            profitable = [r for r in active if r.total_return_pct > 0]

            print(f"\n{'=' * 78}")
            print(f"SUMMARY ({len(active)} active / {len(results)} scanned)")
            print(f"{'=' * 78}")

            if active:
                avg_ret = sum(r.total_return_pct for r in active) / len(active)
                avg_sharpe = sum(r.sharpe for r in active) / len(active)
                total_funding = sum(r.funding_earned for r in active)
                total_trades = sum(r.n_trades for r in active)
                worst_dd = max(r.max_drawdown_pct for r in active)
                best = max(active, key=lambda r: r.total_return_pct)

                ann_factor = 365 / max(args.days, 1)
                avg_ann = ((1 + avg_ret / 100) ** ann_factor - 1) * 100

                print(f"  Avg return:     {avg_ret:+.2f}% ({args.days}d)")
                print(f"  Annualized:     {avg_ann:+.1f}%")
                print(f"  Avg Sharpe:     {avg_sharpe:.2f}")
                print(f"  Worst max DD:   {worst_dd:.2f}%")
                print(f"  Profitable:     {len(profitable)}/{len(active)} symbols")
                print(f"  Total trades:   {total_trades}")
                print(f"  Total funding:  {total_funding:+.6f}")
                print(f"  Best symbol:    {best.symbol} {best.total_return_pct:+.2f}% "
                      f"(Sharpe {best.sharpe:.1f})")
            else:
                print("  No symbols had trades above entry threshold.")
                print("  Try lowering --carry-entry or --top-n.")
    else:
        result = run_single_symbol_backtest(args.symbol, days=args.days, config=config)
        print(f"\n{result.symbol} Results:")
        print(f"  Total return: {result.total_return_pct:+.2f}%")
        print(f"  Annualized:   {result.ann_return_pct:+.2f}%")
        print(f"  Max drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"  Sharpe ratio: {result.sharpe:.2f}")
        print(f"  Trades:       {result.n_trades}")
        print(f"  Settlements:  {result.n_settlements}")
        print(f"  Funding earn: {result.funding_earned:+.6f}")
        print(f"  Costs paid:   {result.costs_paid:.6f}")
        print(f"  Avg hold:     {result.avg_hold_days:.1f} days")


if __name__ == "__main__":
    main()
