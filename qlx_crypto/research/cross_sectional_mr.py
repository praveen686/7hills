"""Cross-sectional allocation strategies on crypto universe.

Lesson from v1: pure long-short mean-reversion failed because shorting
crypto in a bull market (2024-2025) is catastrophic.

This v2 tests LONG-ONLY allocation strategies that beat equal-weight B&H:
  1. Equal-weight rebalancing (true benchmark — sells winners, buys losers)
  2. Inverse-volatility weighting (risk parity lite)
  3. Mean-reversion tilt (overweight oversold, underweight overbought)
  4. Momentum tilt (overweight winners, underweight losers)
  5. Combo: momentum selection + MR entry timing (buy dips in uptrends)
  6. Conditional long-short (shorts only when market is weak)

All strategies are long-only except #6, use 10bps roundtrip costs.

Usage:
    python3 research/cross_sectional_mr.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

COST_BPS = 10
WARMUP = 180  # bars of warmup (> 168h z-score window)


# ---------------------------------------------------------------------------
# Allocation strategies
# ---------------------------------------------------------------------------

def alloc_equal_weight(
    closes: pd.DataFrame,
    idx: int,
    **_kwargs,
) -> dict[str, float]:
    """1/N equal weight — the simplest active rebalancing."""
    n = len(closes.columns)
    return {sym: 1.0 / n for sym in closes.columns}


def alloc_inverse_vol(
    closes: pd.DataFrame,
    idx: int,
    vol_window: int = 168,
    **_kwargs,
) -> dict[str, float]:
    """Weight inversely proportional to recent realised volatility."""
    rets = closes.iloc[max(0, idx - vol_window):idx + 1].pct_change().dropna()
    if len(rets) < 20:
        return alloc_equal_weight(closes, idx)

    vols = rets.std()
    vols = vols.replace(0, np.nan).dropna()
    if len(vols) == 0:
        return alloc_equal_weight(closes, idx)

    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return {sym: float(weights.get(sym, 0)) for sym in closes.columns}


def alloc_mr_tilt(
    closes: pd.DataFrame,
    idx: int,
    z_windows: tuple[int, ...] = (24, 72, 168),
    tilt_strength: float = 0.5,
    **_kwargs,
) -> dict[str, float]:
    """Equal weight + tilt toward oversold (low z-score).

    tilt_strength controls how much we deviate from EW.
    0 = pure EW, 1 = maximum tilt.  Weights clipped to [0, 2/N].
    """
    n = len(closes.columns)
    base_w = 1.0 / n

    z_scores = {}
    for sym in closes.columns:
        c = closes[sym].iloc[:idx + 1]
        zs = []
        for w in z_windows:
            if len(c) < w + 1:
                continue
            ma = c.rolling(w).mean().iloc[-1]
            std = c.rolling(w).std().iloc[-1]
            if std > 0:
                zs.append((c.iloc[-1] - ma) / std)
        z_scores[sym] = np.mean(zs) if zs else 0.0

    if not z_scores:
        return alloc_equal_weight(closes, idx)

    # Tilt: negative z → more weight, positive z → less weight
    z_arr = pd.Series(z_scores)
    # Normalise z-scores to [-1, 1] range using rank
    ranked = z_arr.rank(pct=True)  # 0 = lowest z, 1 = highest z
    # Tilt: center rank around 0, multiply by strength
    # rank=0 (most oversold) → tilt = +tilt_strength
    # rank=1 (most overbought) → tilt = -tilt_strength
    tilt = (0.5 - ranked) * 2.0 * tilt_strength

    weights = {}
    for sym in closes.columns:
        w = base_w * (1.0 + tilt.get(sym, 0.0))
        w = max(0.0, min(w, 2.0 * base_w))  # clip to [0, 2/N]
        weights[sym] = w

    # Renormalise to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {sym: w / total for sym, w in weights.items()}
    else:
        return alloc_equal_weight(closes, idx)

    return weights


def alloc_momentum_tilt(
    closes: pd.DataFrame,
    idx: int,
    lookback: int = 168,
    tilt_strength: float = 0.5,
    **_kwargs,
) -> dict[str, float]:
    """Equal weight + tilt toward recent winners (momentum).

    Overweight symbols with positive momentum, underweight those with
    negative momentum.  Weights clipped to [0, 2/N].
    """
    n = len(closes.columns)
    base_w = 1.0 / n

    if idx < lookback:
        return alloc_equal_weight(closes, idx)

    rets = {}
    for sym in closes.columns:
        c = closes[sym]
        rets[sym] = (c.iloc[idx] / c.iloc[idx - lookback]) - 1.0

    r_arr = pd.Series(rets)
    ranked = r_arr.rank(pct=True)
    tilt = (ranked - 0.5) * 2.0 * tilt_strength

    weights = {}
    for sym in closes.columns:
        w = base_w * (1.0 + tilt.get(sym, 0.0))
        w = max(0.0, min(w, 2.0 * base_w))
        weights[sym] = w

    total = sum(weights.values())
    if total > 0:
        weights = {sym: w / total for sym, w in weights.items()}
    else:
        return alloc_equal_weight(closes, idx)

    return weights


def alloc_combo(
    closes: pd.DataFrame,
    idx: int,
    mom_lookback: int = 168,
    z_window: int = 24,
    n_select: int = 5,
    **_kwargs,
) -> dict[str, float]:
    """Momentum selection + mean-reversion entry timing.

    1. Select top N symbols by medium-term momentum (168h return).
    2. Among those, tilt weight toward the most short-term oversold (24h z-score).

    The idea: buy what's trending up, but enter when it dips.
    """
    if idx < mom_lookback:
        return alloc_equal_weight(closes, idx)

    # Momentum ranking
    mom = {}
    for sym in closes.columns:
        c = closes[sym]
        mom[sym] = (c.iloc[idx] / c.iloc[idx - mom_lookback]) - 1.0

    mom_ranked = pd.Series(mom).sort_values(ascending=False)
    selected = mom_ranked.index[:n_select].tolist()

    if not selected:
        return alloc_equal_weight(closes, idx)

    # Z-score for entry timing among selected
    z_scores = {}
    for sym in selected:
        c = closes[sym].iloc[:idx + 1]
        if len(c) < z_window + 1:
            z_scores[sym] = 0.0
            continue
        ma = c.rolling(z_window).mean().iloc[-1]
        std = c.rolling(z_window).std().iloc[-1]
        z_scores[sym] = (c.iloc[-1] - ma) / std if std > 0 else 0.0

    # Tilt: lower z-score → more weight (buy the dip in uptrend)
    z_arr = pd.Series(z_scores)
    ranked = z_arr.rank(pct=True)
    tilt = (0.5 - ranked) * 1.0  # moderate tilt

    base_w = 1.0 / len(selected)
    weights = {sym: 0.0 for sym in closes.columns}
    for sym in selected:
        w = base_w * (1.0 + tilt.get(sym, 0.0))
        w = max(0.0, min(w, 2.0 * base_w))
        weights[sym] = w

    total = sum(weights.values())
    if total > 0:
        weights = {sym: w / total for sym, w in weights.items()}

    return weights


def alloc_conditional_ls(
    closes: pd.DataFrame,
    idx: int,
    market_proxy: str = "BTCUSDT",
    market_lookback: int = 168,
    n_long: int = 3,
    n_short: int = 3,
    z_windows: tuple[int, ...] = (24, 72, 168),
    **_kwargs,
) -> dict[str, float]:
    """Long-short only when market is weak, long-only when strong.

    Market regime: BTC 168h return. Positive = bull, negative = bear.
    Bull: long-only top momentum (no shorts).
    Bear: long oversold + short overbought.
    """
    if idx < market_lookback:
        return alloc_equal_weight(closes, idx)

    # Market regime
    btc = closes[market_proxy]
    market_ret = (btc.iloc[idx] / btc.iloc[idx - market_lookback]) - 1.0
    is_bull = market_ret > 0

    # Compute z-scores
    z_scores = {}
    for sym in closes.columns:
        c = closes[sym].iloc[:idx + 1]
        zs = []
        for w in z_windows:
            if len(c) < w + 1:
                continue
            ma = c.rolling(w).mean().iloc[-1]
            std = c.rolling(w).std().iloc[-1]
            if std > 0:
                zs.append((c.iloc[-1] - ma) / std)
        z_scores[sym] = np.mean(zs) if zs else 0.0

    z_arr = pd.Series(z_scores).sort_values()

    if is_bull:
        # Bull market: long-only, tilt toward oversold (buy dips)
        n = len(closes.columns)
        base_w = 1.0 / n
        ranked = z_arr.rank(pct=True)
        tilt = (0.5 - ranked) * 0.5

        weights = {}
        for sym in closes.columns:
            w = base_w * (1.0 + tilt.get(sym, 0.0))
            weights[sym] = max(0.0, w)

        total = sum(weights.values())
        if total > 0:
            weights = {sym: w / total for sym, w in weights.items()}
    else:
        # Bear market: long oversold, short overbought
        longs = z_arr.index[:n_long].tolist()
        shorts = z_arr.index[-n_short:].tolist()
        shorts = [s for s in shorts if s not in longs]

        total_pos = len(longs) + len(shorts)
        if total_pos == 0:
            return {sym: 0.0 for sym in closes.columns}

        w = 1.0 / total_pos
        weights = {sym: 0.0 for sym in closes.columns}
        for sym in longs:
            weights[sym] = w
        for sym in shorts:
            weights[sym] = -w

    return weights


# ---------------------------------------------------------------------------
# Backtest engine (generic for any allocation function)
# ---------------------------------------------------------------------------

def backtest_strategy(
    closes: pd.DataFrame,
    alloc_fn,
    rebal_hours: int = 24,
    cost_bps: float = 10,
    warmup: int = WARMUP,
    **alloc_kwargs,
) -> dict:
    """Generic rebalancing backtest.

    At each rebal point, calls alloc_fn(closes, idx, **alloc_kwargs) to get
    target weights, then computes period returns with transaction costs.
    """
    one_way_frac = cost_bps / 10_000 / 2

    rebal_indices = list(range(warmup, len(closes), rebal_hours))
    period_returns = []
    period_dates = []
    prev_weights: dict[str, float] = {sym: 0.0 for sym in closes.columns}
    total_turnover = 0.0
    n_rebalances = 0

    for i, rebal_idx in enumerate(rebal_indices):
        next_rebal = rebal_indices[i + 1] if i + 1 < len(rebal_indices) else len(closes) - 1
        if next_rebal <= rebal_idx or next_rebal >= len(closes):
            break

        # Get target allocation
        target = alloc_fn(closes, rebal_idx, **alloc_kwargs)

        # Period return
        close_start = closes.iloc[rebal_idx]
        close_end = closes.iloc[next_rebal]
        sym_ret = (close_end - close_start) / close_start

        period_gross = sum(
            target.get(sym, 0.0) * sym_ret[sym]
            for sym in closes.columns
            if pd.notna(sym_ret[sym])
        )

        # Turnover cost
        turnover = sum(
            abs(target.get(sym, 0.0) - prev_weights.get(sym, 0.0))
            for sym in closes.columns
        )
        period_cost = turnover * one_way_frac
        total_turnover += turnover

        period_returns.append(period_gross - period_cost)
        period_dates.append(closes.index[next_rebal])
        prev_weights = target.copy()
        n_rebalances += 1

    if not period_returns:
        return {"metrics": None, "error": "No periods"}

    ret_series = pd.Series(period_returns, index=period_dates, name="strategy")
    equity = (1 + ret_series).cumprod()

    periods_per_year = 8760.0 / rebal_hours
    metrics = compute_metrics(ret_series, periods_per_year=periods_per_year)

    return {
        "equity": equity,
        "net_returns": ret_series,
        "metrics": metrics,
        "n_rebalances": n_rebalances,
        "total_turnover": total_turnover,
        "avg_turnover": total_turnover / max(n_rebalances, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Cross-Sectional Allocation Strategies — Crypto Universe")
    print(f"  Cost: {COST_BPS}bps RT | Warmup: {WARMUP}h")
    print("=" * 70)

    cache = KlineCache("data/klines")

    print("\nLoading universe...")
    universe = {}
    for sym in SYMBOLS:
        ohlcv = cache.get(sym, "1h", start="2024-01-01", market="spot")
        universe[sym] = ohlcv.df
        print(f"  {sym}: {len(ohlcv)} bars")

    closes = pd.DataFrame({sym: df["Close"] for sym, df in universe.items()})
    closes = closes.dropna()
    print(f"\n  Aligned: {len(closes)} common bars "
          f"({closes.index[0].date()} → {closes.index[-1].date()})")

    # -------------------------------------------------------------------
    # Strategy sweep
    # -------------------------------------------------------------------
    strategies = [
        # (label, alloc_fn, rebal_hours, kwargs)
        ("EW Rebal 24h",         alloc_equal_weight,  24, {}),
        ("EW Rebal 168h",        alloc_equal_weight, 168, {}),
        ("InvVol 24h",           alloc_inverse_vol,   24, {"vol_window": 168}),
        ("InvVol 168h",          alloc_inverse_vol,  168, {"vol_window": 168}),
        ("MR Tilt 0.3",          alloc_mr_tilt,       24, {"tilt_strength": 0.3}),
        ("MR Tilt 0.5",          alloc_mr_tilt,       24, {"tilt_strength": 0.5}),
        ("MR Tilt 0.7",          alloc_mr_tilt,       24, {"tilt_strength": 0.7}),
        ("MR Tilt 0.5 (48h)",    alloc_mr_tilt,       48, {"tilt_strength": 0.5}),
        ("Mom Tilt 0.3",         alloc_momentum_tilt, 24, {"tilt_strength": 0.3, "lookback": 168}),
        ("Mom Tilt 0.5",         alloc_momentum_tilt, 24, {"tilt_strength": 0.5, "lookback": 168}),
        ("Mom Tilt 0.5 (72h lb)",alloc_momentum_tilt, 24, {"tilt_strength": 0.5, "lookback": 72}),
        ("Combo top5",           alloc_combo,         24, {"n_select": 5, "mom_lookback": 168}),
        ("Combo top3",           alloc_combo,         24, {"n_select": 3, "mom_lookback": 168}),
        ("Combo top5 (72h mom)", alloc_combo,         24, {"n_select": 5, "mom_lookback": 72}),
        ("Cond L/S (BTC regime)",alloc_conditional_ls,24, {}),
    ]

    results = []
    for label, fn, rebal, kwargs in strategies:
        r = backtest_strategy(
            closes=closes,
            alloc_fn=fn,
            rebal_hours=rebal,
            cost_bps=COST_BPS,
            **kwargs,
        )
        results.append((label, r))
        m = r["metrics"]
        print(f"  {label:28s}  Sharpe={m.sharpe_ratio:+.3f}  Ret={m.annualised_return:+.1%}  DD={m.max_drawdown:.1%}")

    # Buy-and-hold benchmark (no rebalancing)
    bnh_ret = closes.pct_change().iloc[WARMUP:].mean(axis=1)
    bnh_equity = (1 + bnh_ret).cumprod()
    bnh_metrics = compute_metrics(bnh_ret.dropna(), periods_per_year=8760)

    # -------------------------------------------------------------------
    # Results table
    # -------------------------------------------------------------------
    print(f"\n\n{'='*105}")
    print("  STRATEGY COMPARISON  (sorted by Sharpe)")
    print(f"{'='*105}")
    print(
        f"{'Strategy':30s} {'Sharpe':>7s} {'AnnRet':>8s} {'MaxDD':>8s} "
        f"{'Sortino':>8s} {'Calmar':>8s} {'Rebals':>6s} {'AvgTO':>6s}"
    )
    print("-" * 105)

    for label, r in sorted(results, key=lambda x: x[1]["metrics"].sharpe_ratio, reverse=True):
        m = r["metrics"]
        print(
            f"{label:30s} "
            f"{m.sharpe_ratio:7.3f} "
            f"{m.annualised_return:7.1%} "
            f"{m.max_drawdown:7.1%} "
            f"{m.sortino_ratio:8.3f} "
            f"{m.calmar_ratio:8.3f} "
            f"{r['n_rebalances']:6d} "
            f"{r['avg_turnover']:6.3f}"
        )

    print("-" * 105)
    print(
        f"{'BUY & HOLD (EW, no rebal)':30s} "
        f"{bnh_metrics.sharpe_ratio:7.3f} "
        f"{bnh_metrics.annualised_return:7.1%} "
        f"{bnh_metrics.max_drawdown:7.1%} "
        f"{bnh_metrics.sortino_ratio:8.3f} "
        f"{bnh_metrics.calmar_ratio:8.3f}"
    )

    # -------------------------------------------------------------------
    # Deep dive on best strategy
    # -------------------------------------------------------------------
    best_label, best_r = max(results, key=lambda x: x[1]["metrics"].sharpe_ratio)
    print(f"\n\n{'='*70}")
    print(f"  BEST: {best_label}")
    print(f"{'='*70}")
    print(best_r["metrics"].summary())

    # Excess return over BnH
    strat_total = best_r["metrics"].total_return
    bnh_total = bnh_metrics.total_return
    print(f"\n  Excess return over B&H: {strat_total - bnh_total:+.1%}")

    # Monthly returns
    monthly = best_r["net_returns"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    print(f"\n  Monthly returns:")
    for date, ret in monthly.items():
        marker = "+" if ret > 0 else " "
        bar = "#" * int(abs(ret) * 100)
        print(f"    {date.strftime('%Y-%m')}: {marker}{ret:7.2%}  {bar}")

    # Win rate
    wins = best_r["net_returns"] > 0
    print(f"\n  Win rate: {wins.mean():.1%} ({wins.sum()}/{len(wins)} periods)")

    pos_ret = best_r["net_returns"][best_r["net_returns"] > 0]
    neg_ret = best_r["net_returns"][best_r["net_returns"] <= 0]
    if len(pos_ret) > 0 and len(neg_ret) > 0:
        print(f"  Avg win:  {pos_ret.mean():.3%}")
        print(f"  Avg loss: {neg_ret.mean():.3%}")
        print(f"  Win/loss: {abs(pos_ret.mean() / neg_ret.mean()):.2f}")

    # Drawdown
    equity = best_r["equity"]
    peak = equity.cummax()
    dd = (equity - peak) / peak
    print(f"\n  Worst drawdown periods:")
    for dt, val in dd.sort_values().head(5).items():
        print(f"    {dt.strftime('%Y-%m-%d')}: {val:.1%}")

    # -------------------------------------------------------------------
    # Strategy-vs-benchmark rolling Sharpe
    # -------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  ROLLING 90-DAY SHARPE (best strategy vs B&H)")
    print(f"{'='*70}")
    # Compute rolling Sharpe for best strategy
    window = 90  # 90 rebalance periods
    roll_ret = best_r["net_returns"]
    if len(roll_ret) > window:
        roll_mean = roll_ret.rolling(window).mean()
        roll_std = roll_ret.rolling(window).std()
        roll_sharpe = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(8760 / 24)
        # Sample every 30 periods
        for dt, sh in roll_sharpe.dropna().iloc[::30].items():
            bar = "#" * max(0, int(sh * 5)) if sh > 0 else "-" * max(0, int(-sh * 5))
            print(f"    {dt.strftime('%Y-%m-%d')}: {sh:+6.2f}  {bar}")


if __name__ == "__main__":
    main()
