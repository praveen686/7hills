"""S5: Tick Microstructure Research — GPU-accelerated VPIN, entropy, Hawkes.

Architecture:
  - Worker processes (CPU): load tick data from DuckDB, compute Hawkes params
  - Main process (GPU): run VPIN + entropy on returned price arrays via T4
  - This avoids CUDA fork issues (can't reinit CUDA in forked subprocess)

Correlates daily microstructure features with next-day index returns.

Usage:
    python -m apps.india_fno.research.s5_tick_microstructure
    python -m apps.india_fno.research.s5_tick_microstructure --start 2025-09-01
    python -m apps.india_fno.research.s5_tick_microstructure --token 256265
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch

from apps.india_scanner.data import is_trading_day
from qlx.data.store import MarketDataStore

NIFTY_TOKEN = 256265
BANKNIFTY_TOKEN = 260105

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.float32


# ---------------------------------------------------------------------------
# GPU feature kernels (run in MAIN process only)
# ---------------------------------------------------------------------------

def _gpu_vpin(
    prices: np.ndarray,
    bucket_size: float = 100.0,
    n_buckets: int = 20,
    sigma_window: int = 100,
) -> float:
    """Compute final VPIN value using GPU for vectorized math.

    GPU handles: log returns, rolling sigma, BVC norm_cdf.
    CPU handles: sequential bucket accumulation (inherently serial).
    """
    n = len(prices)
    if n < sigma_window + 10:
        return float("nan")

    p = torch.tensor(prices, dtype=_DTYPE, device=_DEVICE)

    # Log returns on GPU
    log_ret = torch.zeros(n, dtype=_DTYPE, device=_DEVICE)
    log_ret[1:] = torch.log(torch.clamp(p[1:], min=1e-8) / torch.clamp(p[:-1], min=1e-8))

    # Rolling sigma via cumsum trick (fully vectorized on GPU)
    lr_sq = log_ret ** 2
    cumsum = torch.cumsum(lr_sq, dim=0)
    cumsum_lr = torch.cumsum(log_ret, dim=0)

    # Vectorized rolling variance: avoid Python loop over sigma_window..n
    padded_cumsum = torch.cat([torch.zeros(1, dtype=_DTYPE, device=_DEVICE), cumsum])
    padded_cumsum_lr = torch.cat([torch.zeros(1, dtype=_DTYPE, device=_DEVICE), cumsum_lr])
    idx = torch.arange(sigma_window, n, device=_DEVICE)
    s2 = (padded_cumsum[idx + 1] - padded_cumsum[idx + 1 - sigma_window]) / sigma_window
    m = (padded_cumsum_lr[idx + 1] - padded_cumsum_lr[idx + 1 - sigma_window]) / sigma_window
    var = s2 - m * m
    sigma = torch.full((n,), 0.01, dtype=_DTYPE, device=_DEVICE)
    sigma[sigma_window:] = torch.sqrt(torch.clamp(var, min=1e-16))
    sigma = torch.where(sigma > 1e-8, sigma, torch.tensor(0.01, dtype=_DTYPE, device=_DEVICE))

    # BVC: buy fraction = Phi(log_ret / sigma) on GPU
    buy_frac = 0.5 * (1.0 + torch.erf(log_ret / (sigma * math.sqrt(2.0))))

    # Transfer to CPU for sequential bucket accumulation
    buy_vol = buy_frac.cpu().numpy()
    sell_vol = (1.0 - buy_frac).cpu().numpy()

    current_buy = 0.0
    current_sell = 0.0
    current_vol = 0.0
    completed: list[float] = []
    last_vpin = float("nan")

    for i in range(1, n):
        bv = float(buy_vol[i])
        sv = float(sell_vol[i])

        current_buy += bv
        current_sell += sv
        current_vol += 1.0

        while current_vol >= bucket_size:
            overflow = current_vol - bucket_size
            frac = 1.0 - overflow
            bucket_imb = abs(
                current_buy - bv * (1 - frac) - (current_sell - sv * (1 - frac))
            )
            completed.append(bucket_imb)
            if len(completed) > n_buckets:
                completed = completed[-n_buckets:]
            current_buy = bv * (1 - frac)
            current_sell = sv * (1 - frac)
            current_vol = overflow
            if len(completed) >= n_buckets:
                last_vpin = sum(completed) / (n_buckets * bucket_size)

    return last_vpin


def _gpu_entropy(
    prices: np.ndarray,
    window: int = 100,
    n_bins: int = 10,
) -> float:
    """Compute final tick entropy value using GPU for log returns."""
    n = len(prices)
    if n < window + 10:
        return float("nan")

    p = torch.tensor(prices, dtype=_DTYPE, device=_DEVICE)
    log_ret = torch.zeros(n, dtype=_DTYPE, device=_DEVICE)
    log_ret[1:] = torch.log(torch.clamp(p[1:], min=1e-8) / torch.clamp(p[:-1], min=1e-8))

    rets = log_ret[-window:].cpu().numpy()
    counts, _ = np.histogram(rets, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ---------------------------------------------------------------------------
# CPU-only Hawkes estimation (safe for worker processes)
# ---------------------------------------------------------------------------

def _hawkes_params(
    timestamps: np.ndarray,
) -> tuple[float, float, float, float]:
    """Estimate Hawkes process parameters from trade arrival times.

    Returns (mean_intensity, mu, alpha, beta).
    """
    if len(timestamps) < 20:
        return float("nan"), 1.0, 0.5, 1.5

    ts = timestamps.astype(np.float64)
    if ts[0] > 1e15:
        ts = ts / 1e9
    elif ts[0] > 1e12:
        ts = ts / 1e3
    ts = ts - ts[0]

    dt = np.diff(ts)
    dt = dt[dt > 0]
    if len(dt) < 10:
        return float("nan"), 1.0, 0.5, 1.5

    lambda_bar = len(timestamps) / max(ts[-1], 1e-6)

    bin_size = 10.0
    n_bins = max(int(ts[-1] / bin_size), 5)
    counts = np.histogram(ts, bins=n_bins)[0].astype(float)
    mean_c = np.mean(counts)
    var_c = np.var(counts)

    if mean_c > 0:
        ratio = var_c / mean_c
        n_est = max(0.01, min(0.95, 1 - 1 / math.sqrt(max(ratio, 1.01))))
    else:
        n_est = 0.3

    if len(counts) > 2:
        acf1 = abs(np.corrcoef(counts[:-1], counts[1:])[0, 1])
        acf1 = max(0.01, min(0.99, acf1))
        beta = -math.log(acf1) / bin_size
        beta = max(0.1, min(10.0, beta))
    else:
        beta = 1.5

    mu = lambda_bar * (1 - n_est)
    alpha = n_est * beta

    # Compute mean intensity over eval points
    eval_times = np.arange(0, ts[-1], 1.0)
    intensity = np.full(len(eval_times), mu)
    sum_kernel = 0.0
    last_time = 0.0
    event_idx = 0
    for i, t in enumerate(eval_times):
        while event_idx < len(ts) and ts[event_idx] <= t:
            dt_e = ts[event_idx] - last_time
            sum_kernel = sum_kernel * math.exp(-beta * dt_e) + 1.0
            last_time = ts[event_idx]
            event_idx += 1
        dt_e = t - last_time
        intensity[i] = mu + alpha * sum_kernel * math.exp(-beta * dt_e)

    valid = intensity[~np.isnan(intensity)]
    mean_intensity = float(valid.mean()) if len(valid) > 0 else float("nan")
    return mean_intensity, mu, alpha, beta


# ---------------------------------------------------------------------------
# Worker function (CPU only — loads data + computes Hawkes)
# ---------------------------------------------------------------------------

def _load_day_data(
    d_iso: str,
    token: int,
) -> dict | None:
    """Load tick data and compute CPU-only features for one day.

    Returns raw prices + Hawkes results for GPU processing in main process.
    """
    d = date.fromisoformat(d_iso)
    store = MarketDataStore()
    try:
        df = store.sql(
            "SELECT timestamp, ltp FROM ticks "
            "WHERE date = ? AND instrument_token = ? "
            "AND timestamp >= '2000-01-01' AND ltp > 0.05 "
            "ORDER BY timestamp",
            [d_iso, token],
        )
        if df is None or df.empty or len(df) < 200:
            return None

        prices = df["ltp"].values.astype(np.float64)
        timestamps = df["timestamp"].values

        # CPU-only: Hawkes params
        mean_intensity, mu, alpha, beta = _hawkes_params(timestamps)
        hawkes_ratio = alpha / beta if beta > 0 else float("nan")

        day_close = float(prices[-1])
        day_open = float(prices[0])
        day_return = (day_close - day_open) / day_open

        # Next-day return from nse_index_close
        next_d = d + timedelta(days=1)
        while not is_trading_day(next_d):
            next_d += timedelta(days=1)
            if (next_d - d).days > 10:
                break

        current_close = None
        next_close = None
        for target, label in [(d, "current"), (next_d, "next")]:
            idf = store.sql(
                "SELECT * FROM nse_index_close WHERE date = ?",
                [target.isoformat()],
            )
            if idf is not None and not idf.empty:
                for col in idf.columns:
                    if "close" in col.lower() or "closing" in col.lower():
                        for idx_col in idf.columns:
                            if "index" in idx_col.lower() or "name" in idx_col.lower():
                                row = idf[idf[idx_col].astype(str).str.contains(
                                    "Nifty 50", case=False, na=False
                                )]
                                if not row.empty:
                                    val = row[col].iloc[0]
                                    if pd.notna(val) and float(val) > 0:
                                        if label == "current":
                                            current_close = float(val)
                                        else:
                                            next_close = float(val)
                                        break
                        break

        if current_close and next_close:
            next_day_return = (next_close - current_close) / current_close
        else:
            next_day_return = float("nan")

        return {
            "date": d,
            "prices": prices,  # for GPU processing in main
            "n_ticks": len(prices),
            "hawkes_mean": mean_intensity,
            "hawkes_ratio": hawkes_ratio,
            "mu": float(mu),
            "alpha": float(alpha),
            "beta": float(beta),
            "day_return": day_return,
            "next_day_return": next_day_return,
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Main research runner
# ---------------------------------------------------------------------------

def run_research(
    store: MarketDataStore,
    start: date,
    end: date,
    token: int = NIFTY_TOKEN,
    max_workers: int = 8,
) -> None:
    """Run microstructure research.

    Workers load data + compute Hawkes (CPU). Main process runs VPIN + entropy (GPU).
    """
    dates = store.available_dates("ticks")
    dates = sorted(d for d in dates if start <= d <= end)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\nS5 Tick Microstructure Research (GPU: {gpu_name})")
    print(f"  Token: {token}, {len(dates)} dates, {max_workers} workers\n")

    t0 = time.time()

    # Phase 1: parallel data loading + Hawkes (CPU workers)
    loaded: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_load_day_data, d.isoformat(), token): d
            for d in dates
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is not None:
                loaded.append(result)
            if done % 50 == 0:
                print(f"  loaded {done}/{len(dates)} ({len(loaded)} valid, "
                      f"{time.time()-t0:.1f}s)")

    load_time = time.time() - t0
    print(f"  Data loaded: {len(loaded)} days in {load_time:.1f}s")

    # Phase 2: GPU VPIN + entropy (main process, sequential but fast)
    t1 = time.time()
    daily_features: list[dict] = []
    for i, rec in enumerate(loaded):
        prices = rec.pop("prices")
        rec["vpin"] = _gpu_vpin(prices)
        rec["entropy"] = _gpu_entropy(prices)
        daily_features.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  GPU features: {i+1}/{len(loaded)} ({time.time()-t1:.1f}s)")

    gpu_time = time.time() - t1
    elapsed = time.time() - t0
    daily_features.sort(key=lambda x: x["date"])
    print(f"  GPU phase: {gpu_time:.1f}s | Total: {elapsed:.1f}s\n")

    if not daily_features:
        print("  No valid data found.")
        return

    df = pd.DataFrame(daily_features)
    _print_results(df)


def _backtest_feature(
    df: pd.DataFrame,
    feature: str,
    lookback: int = 60,
    entry_pctile: float = 0.80,
    hold_days: int = 1,
    cost_bps: float = 5,
) -> dict:
    """Backtest a single microstructure feature as a next-day signal.

    Direction is determined CAUSALLY: at each step, the trailing IC from the
    past `lookback` days determines whether high feature → long or short.
    Sharpe is computed from ALL daily returns (including zero-trade days).
    """
    from scipy.stats import spearmanr

    valid = df.dropna(subset=[feature, "next_day_return"]).copy()
    valid = valid.sort_values("date").reset_index(drop=True)

    if len(valid) < lookback + 10:
        return {"trades": 0}

    cost_frac = cost_bps / 10_000
    trades = []
    daily_pnl = []  # ALL days from lookback onward (including zero-trade days)
    direction_counts = {1: 0, -1: 0}

    for i in range(lookback, len(valid)):
        # Causal IC: correlate features[i-lookback..i-1] with returns[i-lookback..i-1]
        # At time i, next_day_return[j] for j < i is already realized
        trail = slice(max(0, i - lookback), i)
        trail_feat = valid[feature].iloc[trail].values.astype(float)
        trail_ret = valid["next_day_return"].iloc[trail].values.astype(float)
        mask = ~(np.isnan(trail_feat) | np.isnan(trail_ret))
        if mask.sum() > 10:
            ic, _ = spearmanr(trail_feat[mask], trail_ret[mask])
            direction = 1 if ic > 0 else -1
        else:
            direction = 1  # default long if insufficient data

        window = valid[feature].iloc[max(0, i - lookback + 1):i + 1]
        threshold = window.quantile(entry_pctile)
        val = valid[feature].iloc[i]

        if val >= threshold:
            ret = valid["next_day_return"].iloc[i] * direction - cost_frac
            trades.append({
                "date": valid["date"].iloc[i],
                "feature_val": val,
                "ret": ret,
                "direction": direction,
            })
            daily_pnl.append(ret)
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        else:
            daily_pnl.append(0.0)

    if not trades:
        return {"trades": 0}

    pnl_arr = np.array(daily_pnl)
    trade_rets = np.array([t["ret"] for t in trades])
    cumulative = np.cumsum(pnl_arr)
    total_return = float(cumulative[-1])

    # Sharpe from ALL daily returns (including zero-trade days) with ddof=1
    if len(pnl_arr) > 1:
        std = np.std(pnl_arr, ddof=1)
        if std > 0:
            sharpe = float(np.mean(pnl_arr) / std * np.sqrt(252))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

    wins = sum(1 for t in trades if t["ret"] > 0)

    # Dominant direction used across trades
    dom_dir = 1 if direction_counts.get(1, 0) >= direction_counts.get(-1, 0) else -1

    return {
        "trades": len(trades),
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate": wins / len(trades),
        "avg_ret_pct": float(np.mean(trade_rets)) * 100,
        "direction": dom_dir,
    }


def _print_results(df: pd.DataFrame) -> None:
    """Print research results with feature analysis and signal backtests."""
    from scipy.stats import spearmanr

    print("=" * 70)
    print("S5 TICK MICROSTRUCTURE RESULTS")
    print("=" * 70)

    features = ["vpin", "entropy", "hawkes_mean", "hawkes_ratio"]

    # --- Feature statistics ---
    print("\n  Feature Statistics:")
    for feat in features:
        valid = df[feat].dropna()
        if valid.empty:
            continue
        print(f"    {feat:<15} mean={valid.mean():.4f}, std={valid.std():.4f}, "
              f"min={valid.min():.4f}, max={valid.max():.4f}")

    # --- Predictive correlations (full-sample, informational only) ---
    print("\n  Predictive Correlations (vs next-day return):")
    print("    (full-sample IC shown for diagnostics; backtests use causal rolling IC)")
    valid_df = df.dropna(subset=["next_day_return"])

    n_tests = len(features)
    bonferroni = 0.05 / n_tests

    if len(valid_df) > 10:
        for feat in features:
            feat_valid = valid_df.dropna(subset=[feat])
            if len(feat_valid) < 10:
                continue
            ic, pval = spearmanr(feat_valid[feat], feat_valid["next_day_return"])
            sig = "**" if pval < bonferroni else "*" if pval < 0.05 else " "
            print(f"    {feat:<15} IC={ic:+.4f} (p={pval:.4f}) {sig}"
                  f"  {'(Bonf. sig)' if pval < bonferroni else ''}")

        print("\n  Quintile Spreads (full-sample, in-sample — not achievable OOS):")
        for feat in features:
            feat_valid = valid_df.dropna(subset=[feat])
            if len(feat_valid) < 20:
                continue
            feat_valid = feat_valid.sort_values(feat)
            n_q = 5
            q_size = len(feat_valid) // n_q
            q_returns = []
            for q in range(n_q):
                s = q * q_size
                e = s + q_size if q < n_q - 1 else len(feat_valid)
                q_ret = feat_valid.iloc[s:e]["next_day_return"].mean() * 100
                q_returns.append(q_ret)
            spread = q_returns[-1] - q_returns[0]
            q_str = " | ".join(f"Q{i+1}={r:+.3f}%" for i, r in enumerate(q_returns))
            print(f"    {feat:<15} spread={spread:+.3f}%  [{q_str}]")
    else:
        print("    Insufficient data for correlation analysis.")

    # --- Signal backtests ---
    print(f"\n{'='*70}")
    print("SIGNAL BACKTESTS (causal rolling IC → direction, feature > rolling pctile → entry)")
    print("=" * 70)

    lookbacks = [30, 60]
    entry_pctiles = [0.75, 0.80]

    print(f"\n  {'Feature':<15} {'Dir':>4} {'LB':>4} {'Entry':>6} | "
          f"{'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'WinRate':>8} {'Trades':>7}")
    print("  " + "-" * 80)

    best_sharpe = -999.0
    best_config = ""

    for feat in features:
        for lb in lookbacks:
            for ep in entry_pctiles:
                result = _backtest_feature(
                    df, feat, lookback=lb, entry_pctile=ep, cost_bps=5,
                )
                if result["trades"] < 3:
                    continue

                dir_label = "long" if result.get("direction", 1) == 1 else "short"
                print(f"  {feat:<15} {dir_label:>4} {lb:4d} {ep:6.2f} | "
                      f"{result['sharpe']:7.2f} {result['total_return_pct']:+7.2f}% "
                      f"{result['max_dd_pct']:6.2f}% {result['win_rate']*100:7.1f}% "
                      f"{result['trades']:7d}")

                if result["sharpe"] > best_sharpe and result["trades"] >= 5:
                    best_sharpe = result["sharpe"]
                    best_config = (f"{feat} dir={dir_label} lb={lb} "
                                   f"entry={ep}")

    # --- Composite signal (causal: IC direction from trailing window) ---
    print(f"\n  {'--- Composite signals ---':^80}")

    composite_df = df.dropna(subset=features + ["next_day_return"]).copy()
    composite_df = composite_df.sort_values("date").reset_index(drop=True)

    if len(composite_df) > 70:
        for lb in lookbacks:
            for ep in entry_pctiles:
                # Build causal composite: trailing IC determines z-score direction
                composite_vals = np.full(len(composite_df), np.nan)
                for i in range(lb, len(composite_df)):
                    signal_val = 0.0
                    n_valid = 0
                    for feat in features:
                        # Causal trailing IC for direction
                        trail = slice(max(0, i - lb), i)
                        tf = composite_df[feat].iloc[trail].values.astype(float)
                        tr = composite_df["next_day_return"].iloc[trail].values.astype(float)
                        mask = ~(np.isnan(tf) | np.isnan(tr))
                        if mask.sum() > 10:
                            ic, _ = spearmanr(tf[mask], tr[mask])
                            direction = 1 if ic > 0 else -1
                        else:
                            direction = 1
                        # Z-score within trailing window
                        window = composite_df[feat].iloc[max(0, i - lb + 1):i + 1]
                        mu = window.mean()
                        std = window.std()
                        z = (composite_df[feat].iloc[i] - mu) / std if std > 1e-12 else 0.0
                        signal_val += z * direction
                        n_valid += 1
                    composite_vals[i] = signal_val / max(n_valid, 1)

                composite_df["_composite"] = composite_vals

                result = _backtest_feature(
                    composite_df, "_composite", lookback=lb, entry_pctile=ep,
                    cost_bps=5,
                )
                if result["trades"] >= 3:
                    dir_label = "long" if result.get("direction", 1) == 1 else "short"
                    print(f"  {'composite':<15} {dir_label:>4} {lb:4d} {ep:6.2f} | "
                          f"{result['sharpe']:7.2f} {result['total_return_pct']:+7.2f}% "
                          f"{result['max_dd_pct']:6.2f}% {result['win_rate']*100:7.1f}% "
                          f"{result['trades']:7d}")

                    if result["sharpe"] > best_sharpe and result["trades"] >= 5:
                        best_sharpe = result["sharpe"]
                        best_config = f"composite dir={dir_label} lb={lb} entry={ep}"

    if best_sharpe > -999:
        print(f"\n  Best: {best_config} (Sharpe={best_sharpe:.2f})")

    print()


def main() -> None:
    from apps.india_fno.research import tee_to_results

    parser = argparse.ArgumentParser(description="S5 Tick Microstructure Research (GPU)")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--token", type=int, default=NIFTY_TOKEN,
                        help=f"Instrument token (default: {NIFTY_TOKEN} = NIFTY)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    with tee_to_results("s5_tick_microstructure"):
        store = MarketDataStore()
        if args.start:
            start = date.fromisoformat(args.start)
        else:
            dates = store.available_dates("ticks")
            start = min(dates) if dates else date(2025, 8, 1)
        end = date.fromisoformat(args.end) if args.end else date.today()

        run_research(store, start, end, token=args.token, max_workers=args.workers)
        store.close()


if __name__ == "__main__":
    main()
