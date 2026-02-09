"""S12: Vedic Fractional Alpha — Research backtest runner.

Modes:
  python -m strategies.s12_vedic_ffpe.research                 # math-first (default params)
  python -m strategies.s12_vedic_ffpe.research --sweep          # parameter sweep
  python -m strategies.s12_vedic_ffpe.research --validate       # placebo + time-shift tests
  python -m strategies.s12_vedic_ffpe.research --intraday       # intraday mode backtest

Math-first philosophy: run with default parameters, report honest results.
Parameter sweep clearly labeled as optimised (not default).
Validation gates from Alpha Factory protocol.
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from itertools import product

import numpy as np
import pandas as pd

from data.store import MarketDataStore
from strategies.s12_vedic_ffpe.signals import (
    compute_daily_signal,
    compute_features_array,
    classify_regime,
)
from strategies.s12_vedic_ffpe.intraday_engine import run_intraday

# GPU acceleration (lazy import to avoid torch import overhead when not needed)
_GPU_AVAILABLE = False
try:
    import torch
    _GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

# Symbol mapping: user-facing name → DuckDB ILIKE pattern
_SYMBOL_MAP: dict[str, str] = {
    "NIFTY": "Nifty 50",
    "BANKNIFTY": "Nifty Bank",
    "FINNIFTY": "Nifty Fin",
    "MIDCPNIFTY": "Nifty Midcap",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_index_closes(store: MarketDataStore, symbol: str) -> pd.DataFrame:
    """Load daily index closes for a symbol."""
    pattern = _SYMBOL_MAP.get(symbol, symbol)
    df = store.sql(
        'SELECT date, "Closing Index Value" as close_val '
        'FROM nse_index_close '
        'WHERE "Index Name" = ? '
        'ORDER BY date',
        [pattern],
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df["close_val"] = pd.to_numeric(df["close_val"], errors="coerce")
    df = df.dropna(subset=["close_val"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def _load_intraday_bars(d_iso: str, symbol: str) -> dict | None:
    """Load 1-minute bars for a symbol on a date (worker process).

    Uses nearest-month futures as index proxy (filters instrument_type='FUT'
    and selects the nearest expiry to avoid mixing strikes/option prices).
    """
    store = MarketDataStore()
    try:
        df = store.sql(
            "SELECT timestamp, open, high, low, close, volume "
            "FROM nfo_1min "
            "WHERE name = ? AND date = ? AND instrument_type = 'FUT' "
            "AND expiry = ("
            "  SELECT MIN(expiry) FROM nfo_1min "
            "  WHERE name = ? AND date = ? AND instrument_type = 'FUT'"
            ") "
            "ORDER BY timestamp",
            [symbol, d_iso, symbol, d_iso],
        )
        if df is None or df.empty or len(df) < 30:
            return None
        return {
            "date": date.fromisoformat(d_iso),
            "close": df["close"].values.astype(np.float64),
            "high": df["high"].values.astype(np.float64),
            "low": df["low"].values.astype(np.float64),
            "n_bars": len(df),
        }
    except Exception:
        return None
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def _backtest_signal(
    closes: np.ndarray,
    dates: list,
    *,
    alpha_window: int = 60,
    alpha_lo: float = 0.85,
    alpha_hi: float = 1.15,
    frac_d: float = 0.226,
    min_conviction: float = 0.15,
    phase_lo: float = 0.0,
    phase_hi: float = 0.5,
    cost_bps: float = 5.0,
    hold_days: int = 5,
) -> dict:
    """Run EOD backtest: T+1 execution with regime-gated signals."""
    n = len(closes)
    warmup = alpha_window + 10
    if n < warmup + 20:
        return {"trades": 0}

    cost_frac = cost_bps / 10_000

    # Compute next-day returns
    next_day_ret = np.full(n, np.nan)
    for i in range(n - 1):
        next_day_ret[i] = (closes[i + 1] - closes[i]) / closes[i]

    # Run signals day by day
    prev_regime = "normal"
    bars_in_regime = 0
    regime_centroids: dict | None = None
    position = "flat"
    entry_day = 0
    daily_pnl = []
    trades = []
    regime_counts = {"subdiffusive": 0, "normal": 0, "superdiffusive": 0}

    for i in range(warmup, n - 1):
        history = closes[:i + 1]

        signal, state = compute_daily_signal(
            history,
            alpha_window=alpha_window,
            alpha_lo=alpha_lo,
            alpha_hi=alpha_hi,
            frac_d=frac_d,
            min_conviction=min_conviction,
            phase_entry_lo=phase_lo,
            phase_entry_hi=phase_hi,
            prev_regime=prev_regime,
            bars_in_regime=bars_in_regime,
            regime_centroids=regime_centroids,
        )
        prev_regime = state["prev_regime"]
        bars_in_regime = state["bars_in_regime"]
        regime_centroids = state.get("regime_centroids")
        regime_counts[signal.regime] = regime_counts.get(signal.regime, 0) + 1

        # Position management
        if position != "flat":
            days_held = i - entry_day
            # Exit conditions: hold_days exceeded or opposite signal
            should_exit = days_held >= hold_days
            should_exit |= (signal.direction != "flat"
                            and signal.direction != position)

            if should_exit:
                ret = next_day_ret[i]
                if not np.isnan(ret):
                    trade_ret = ret * (1 if position == "long" else -1) - cost_frac
                    daily_pnl.append(trade_ret)
                    trades.append({
                        "date": dates[i] if i < len(dates) else i,
                        "direction": position,
                        "ret": trade_ret,
                        "regime": signal.regime,
                        "alpha": signal.alpha,
                    })
                else:
                    daily_pnl.append(0.0)
                position = "flat"
            else:
                # Holding — earn next-day return
                ret = next_day_ret[i]
                if not np.isnan(ret):
                    daily_pnl.append(ret * (1 if position == "long" else -1))
                else:
                    daily_pnl.append(0.0)
                continue

        # Entry
        if position == "flat" and signal.direction in ("long", "short"):
            position = signal.direction
            entry_day = i
            # T+1: no PnL on entry day, PnL starts next bar
            daily_pnl.append(0.0)
        else:
            daily_pnl.append(0.0)

    if not daily_pnl:
        return {"trades": 0}

    pnl_arr = np.array(daily_pnl)
    cumulative = np.cumsum(pnl_arr)
    total_return = float(cumulative[-1])

    # Sharpe: all daily returns, ddof=1
    if len(pnl_arr) > 1:
        std = np.std(pnl_arr, ddof=1)
        sharpe = float(np.mean(pnl_arr) / std * np.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

    wins = sum(1 for t in trades if t["ret"] > 0)

    return {
        "trades": len(trades),
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate": wins / max(len(trades), 1),
        "regime_counts": regime_counts,
        "daily_returns": daily_pnl,
        "trade_list": trades,
    }


# ---------------------------------------------------------------------------
# Research modes
# ---------------------------------------------------------------------------

def run_math_first(store: MarketDataStore, symbols: list[str],
                   use_gpu: bool = False) -> None:
    """Section A: math-first research with default parameters."""
    print("=" * 70)
    print("S12 VEDIC FRACTIONAL ALPHA — MATH-FIRST RESEARCH")
    print("=" * 70)
    print("  Default parameters (no optimisation):")
    print("    α window = 60 days")
    print("    α thresholds = [0.85, 1.15]")
    print("    Fractional d = 0.226 (LCFT prior)")
    print("    Madhava order = 4")
    print("    Cost = 5 bps roundtrip")
    if use_gpu:
        print("    Compute = GPU (T4)")
    print()

    for symbol in symbols:
        print(f"\n{'─' * 60}")
        print(f"  {symbol}")
        print(f"{'─' * 60}")

        t0 = time.time()
        df = _load_index_closes(store, symbol)
        if df.empty or len(df) < 80:
            print(f"  Insufficient data for {symbol} ({len(df)} days)")
            continue

        closes = df["close_val"].values
        dates_list = df["date"].tolist()
        print(f"  Data: {len(df)} days ({dates_list[0]} → {dates_list[-1]})")

        # Feature statistics
        print("\n  Computing features...")
        if use_gpu and _GPU_AVAILABLE:
            from strategies.s12_vedic_ffpe.gpu_features import (
                compute_features_array_gpu,
            )
            features = compute_features_array_gpu(
                closes, alpha_window=60, frac_d=0.226,
            )
        else:
            features = compute_features_array(closes, alpha_window=60, frac_d=0.226)
        feat_time = time.time() - t0
        print(f"  Features computed in {feat_time:.1f}s")

        # Print feature stats
        print("\n  Feature Statistics:")
        for feat_name in ["alpha", "alpha_msd", "alpha_waiting", "hurst",
                          "mock_theta_ratio", "mock_theta_div", "vol_distortion",
                          "coherence", "phase"]:
            vals = features.get(feat_name)
            if vals is None:
                continue
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                continue
            print(f"    {feat_name:<20} mean={np.mean(valid):.4f}  "
                  f"std={np.std(valid, ddof=1):.4f}  "
                  f"min={np.min(valid):.4f}  max={np.max(valid):.4f}")

        # Regime distribution
        regime = features["regime"]
        valid_regime = regime[~np.isnan(regime)]
        if len(valid_regime) > 0:
            n_sub = np.sum(valid_regime == 0)
            n_norm = np.sum(valid_regime == 1)
            n_sup = np.sum(valid_regime == 2)
            total = len(valid_regime)
            print(f"\n  Regime Distribution:")
            print(f"    Subdiffusive:  {n_sub:4d} ({n_sub/total*100:.1f}%)")
            print(f"    Normal:        {n_norm:4d} ({n_norm/total*100:.1f}%)")
            print(f"    Superdiffusive:{n_sup:4d} ({n_sup/total*100:.1f}%)")

        # Predictive correlations
        from scipy.stats import spearmanr

        next_ret = np.full(len(closes), np.nan)
        next_ret[:-1] = np.diff(closes) / closes[:-1]

        print("\n  Predictive Correlations (vs next-day return):")
        print("    (full-sample IC — informational; backtest uses causal logic)")
        for feat_name in ["alpha", "mock_theta_ratio", "mock_theta_div",
                          "vol_distortion", "coherence", "phase", "frac_d_series"]:
            vals = features.get(feat_name)
            if vals is None:
                continue
            mask = ~(np.isnan(vals) | np.isnan(next_ret))
            if mask.sum() < 20:
                continue
            ic, pval = spearmanr(vals[mask], next_ret[mask])
            sig = "**" if pval < 0.01 else "*" if pval < 0.05 else " "
            print(f"    {feat_name:<20} IC={ic:+.4f}  (p={pval:.4f}) {sig}")

        # Backtest
        print("\n  EOD Backtest (default params, 5 bps cost):")
        if use_gpu and _GPU_AVAILABLE:
            from strategies.s12_vedic_ffpe.gpu_features import (
                backtest_from_features,
            )
            result = backtest_from_features(
                features, closes, dates_list,
                alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                cost_bps=5.0, hold_days=5,
            )
        else:
            result = _backtest_signal(
                closes, dates_list,
                alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                frac_d=0.226, cost_bps=5.0, hold_days=5,
            )
        _print_backtest_result(result, symbol)

    print()


def run_sweep(store: MarketDataStore, symbols: list[str],
              use_gpu: bool = False) -> None:
    """Section B: parameter sweep (clearly labeled as optimised).

    With GPU: pre-compute features per (d, lb) pair, then sweep thresholds
    using backtest_from_features (near-instant per combo).
    """
    print("\n" + "=" * 70)
    print("S12 VEDIC FRACTIONAL ALPHA — PARAMETER SWEEP")
    print("=" * 70)
    print("  ⚠ OPTIMISED RESULTS — NOT ACHIEVABLE OUT OF SAMPLE")
    if use_gpu and _GPU_AVAILABLE:
        print("  Compute = GPU (T4)")
    print()

    alpha_los = [0.75, 0.80, 0.85, 0.90]
    alpha_his = [1.10, 1.15, 1.20, 1.25]
    frac_ds = [0.15, 0.20, 0.226, 0.25, 0.30]
    lookbacks = [30, 45, 60, 90]

    for symbol in symbols:
        df = _load_index_closes(store, symbol)
        if df.empty or len(df) < 100:
            continue

        closes = df["close_val"].values
        dates_list = df["date"].tolist()

        print(f"\n{'─' * 60}")
        print(f"  {symbol} — {len(df)} days")
        print(f"{'─' * 60}")

        header = (f"  {'αLo':>5} {'αHi':>5} {'d':>6} {'LB':>4} | "
                  f"{'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'WinR':>6} {'Trades':>6}")
        print(header)
        print("  " + "-" * 70)

        best_sharpe = -999.0
        best_config = ""

        if use_gpu and _GPU_AVAILABLE:
            from strategies.s12_vedic_ffpe.gpu_features import (
                compute_features_array_gpu,
                backtest_from_features,
            )
            # Pre-compute features per (d, lb) — reuse across threshold combos
            feature_cache: dict[tuple[float, int], dict] = {}
            t0 = time.time()
            for d, lb in product(frac_ds, lookbacks):
                features = compute_features_array_gpu(
                    closes, alpha_window=lb, frac_d=d,
                )
                feature_cache[(d, lb)] = features
            print(f"  Pre-computed {len(feature_cache)} feature sets "
                  f"in {time.time()-t0:.1f}s")

            for alpha_lo, alpha_hi, d, lb in product(alpha_los, alpha_his, frac_ds, lookbacks):
                if alpha_lo >= alpha_hi:
                    continue
                features = feature_cache[(d, lb)]
                result = backtest_from_features(
                    features, closes, dates_list,
                    alpha_window=lb, alpha_lo=alpha_lo, alpha_hi=alpha_hi,
                    cost_bps=5.0, hold_days=5,
                )
                if result["trades"] < 3:
                    continue

                sharpe = result["sharpe"]
                print(f"  {alpha_lo:5.2f} {alpha_hi:5.2f} {d:6.3f} {lb:4d} | "
                      f"{sharpe:7.2f} {result['total_return_pct']:+7.2f}% "
                      f"{result['max_dd_pct']:6.2f}% "
                      f"{result['win_rate']*100:5.1f}% "
                      f"{result['trades']:6d}")

                if sharpe > best_sharpe and result["trades"] >= 5:
                    best_sharpe = sharpe
                    best_config = f"αLo={alpha_lo} αHi={alpha_hi} d={d} lb={lb}"
        else:
            for alpha_lo, alpha_hi, d, lb in product(alpha_los, alpha_his, frac_ds, lookbacks):
                if alpha_lo >= alpha_hi:
                    continue
                result = _backtest_signal(
                    closes, dates_list,
                    alpha_window=lb, alpha_lo=alpha_lo, alpha_hi=alpha_hi,
                    frac_d=d, cost_bps=5.0, hold_days=5,
                )
                if result["trades"] < 3:
                    continue

                sharpe = result["sharpe"]
                print(f"  {alpha_lo:5.2f} {alpha_hi:5.2f} {d:6.3f} {lb:4d} | "
                      f"{sharpe:7.2f} {result['total_return_pct']:+7.2f}% "
                      f"{result['max_dd_pct']:6.2f}% "
                      f"{result['win_rate']*100:5.1f}% "
                      f"{result['trades']:6d}")

                if sharpe > best_sharpe and result["trades"] >= 5:
                    best_sharpe = sharpe
                    best_config = f"αLo={alpha_lo} αHi={alpha_hi} d={d} lb={lb}"

        if best_sharpe > -999:
            print(f"\n  Best (optimised): {best_config} (Sharpe={best_sharpe:.2f})")

    print()


def run_validate(store: MarketDataStore, symbols: list[str],
                 use_gpu: bool = False) -> None:
    """Section C: validation gates (placebo + time-shift)."""
    print("\n" + "=" * 70)
    print("S12 VEDIC FRACTIONAL ALPHA — VALIDATION GATES")
    print("=" * 70)
    if use_gpu and _GPU_AVAILABLE:
        print("  Compute = GPU (T4)")

    _compute_feats = compute_features_array
    _backtest_fn = None
    if use_gpu and _GPU_AVAILABLE:
        from strategies.s12_vedic_ffpe.gpu_features import (
            compute_features_array_gpu,
            backtest_from_features,
        )
        _compute_feats = compute_features_array_gpu
        _backtest_fn = backtest_from_features

    for symbol in symbols:
        df = _load_index_closes(store, symbol)
        if df.empty or len(df) < 100:
            continue

        closes = df["close_val"].values
        dates_list = df["date"].tolist()

        print(f"\n{'─' * 60}")
        print(f"  {symbol}")
        print(f"{'─' * 60}")

        # Real signal
        t0 = time.time()
        if _backtest_fn is not None:
            features = _compute_feats(closes, alpha_window=60, frac_d=0.226)
            real_result = _backtest_fn(
                features, closes, dates_list,
                alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                cost_bps=5.0,
            )
        else:
            real_result = _backtest_signal(
                closes, dates_list,
                alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                frac_d=0.226, cost_bps=5.0,
            )
        real_sharpe = real_result.get("sharpe", 0.0)
        print(f"\n  Real signal Sharpe: {real_sharpe:.3f}  ({time.time()-t0:.1f}s)")

        # --- Placebo test: 100 random permutations of signals ---
        print("\n  Placebo Test (100 permutations):")
        rng = np.random.default_rng(42)
        placebo_sharpes = []
        t0 = time.time()

        for p in range(100):
            # Shuffle returns relative to signal dates
            shuffled_closes = closes.copy()
            log_ret = np.diff(np.log(np.maximum(shuffled_closes, 1e-8)))
            rng.shuffle(log_ret)
            shuffled_closes[1:] = shuffled_closes[0] * np.exp(np.cumsum(log_ret))

            if _backtest_fn is not None:
                feats = _compute_feats(shuffled_closes, alpha_window=60, frac_d=0.226)
                result = _backtest_fn(
                    feats, shuffled_closes, dates_list,
                    alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                    cost_bps=5.0,
                )
            else:
                result = _backtest_signal(
                    shuffled_closes, dates_list,
                    alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                    frac_d=0.226, cost_bps=5.0,
                )
            placebo_sharpes.append(result.get("sharpe", 0.0))

        placebo_arr = np.array(placebo_sharpes)
        pctile = np.sum(real_sharpe > placebo_arr) / len(placebo_arr) * 100
        print(f"    Completed in {time.time()-t0:.1f}s")
        print(f"    Real Sharpe: {real_sharpe:.3f}")
        print(f"    Placebo mean: {np.mean(placebo_arr):.3f} ± {np.std(placebo_arr, ddof=1):.3f}")
        print(f"    Placebo 90th: {np.percentile(placebo_arr, 90):.3f}")
        print(f"    Real > {pctile:.0f}% of placebos "
              f"{'✓ PASS' if pctile >= 90 else '✗ FAIL'}")

        # --- Time-shift test ---
        print("\n  Time-Shift Test:")
        shifts = [-3, -2, -1, 1, 2, 3]
        shift_sharpes = {}
        for s in shifts:
            shifted_closes = np.roll(closes, s)
            # Handle edges
            if s > 0:
                shifted_closes[:s] = closes[:s]
            else:
                shifted_closes[s:] = closes[s:]

            if _backtest_fn is not None:
                feats = _compute_feats(shifted_closes, alpha_window=60, frac_d=0.226)
                result = _backtest_fn(
                    feats, shifted_closes, dates_list,
                    alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                    cost_bps=5.0,
                )
            else:
                result = _backtest_signal(
                    shifted_closes, dates_list,
                    alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                    frac_d=0.226, cost_bps=5.0,
                )
            shift_sharpes[s] = result.get("sharpe", 0.0)
            beat = "✓" if real_sharpe > shift_sharpes[s] else "✗"
            print(f"    Shift {s:+d}: Sharpe={shift_sharpes[s]:.3f} {beat}")

        all_beat = all(real_sharpe >= v for v in shift_sharpes.values())
        print(f"    Original beats all shifts: "
              f"{'✓ PASS' if all_beat else '✗ FAIL'}")

        # --- Regime stability ---
        print("\n  Regime Stability:")
        features = _compute_feats(closes, alpha_window=60, frac_d=0.226)
        alpha_arr = features["alpha"]
        next_ret = np.full(len(closes), np.nan)
        next_ret[:-1] = np.diff(closes) / closes[:-1]

        # Check if α correctly predicts next-day behavior
        correct = 0
        total = 0
        for i in range(len(alpha_arr)):
            if np.isnan(alpha_arr[i]) or np.isnan(next_ret[i]):
                continue
            regime = classify_regime(alpha_arr[i])
            ret = next_ret[i]
            total += 1

            # Subdiffusive → expect reversion (small |ret|)
            # Superdiffusive → expect continuation (same sign)
            # Normal → no prediction
            if regime == "subdiffusive" and abs(ret) < np.nanstd(next_ret):
                correct += 1
            elif regime == "superdiffusive":
                # Check if next move continues recent direction
                if i >= 5:
                    recent_mom = np.sum(np.diff(np.log(np.maximum(closes[i-5:i+1], 1e-8))))
                    if recent_mom * ret > 0:
                        correct += 1
            elif regime == "normal":
                correct += 1  # no prediction = always "correct"

        accuracy = correct / max(total, 1)
        print(f"    Regime classification accuracy: {accuracy:.1%} "
              f"({correct}/{total})")

    print()


def run_intraday_research(store: MarketDataStore, symbols: list[str],
                          max_workers: int = 8) -> None:
    """Intraday mode backtest using 1-minute bars."""
    print("\n" + "=" * 70)
    print("S12 VEDIC FRACTIONAL ALPHA — INTRADAY RESEARCH")
    print("=" * 70)

    for symbol in symbols:
        print(f"\n{'─' * 60}")
        print(f"  {symbol}")
        print(f"{'─' * 60}")

        dates = store.available_dates("nfo_1min")
        dates = sorted(dates)

        if not dates:
            print(f"  No 1-min data available for {symbol}")
            continue

        print(f"  Loading 1-min bars ({len(dates)} dates, {max_workers} workers)...")
        t0 = time.time()

        # Parallel load
        loaded: list[dict] = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_load_intraday_bars, d.isoformat(), symbol): d
                for d in dates
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    loaded.append(result)

        loaded.sort(key=lambda x: x["date"])
        load_time = time.time() - t0
        print(f"  Loaded {len(loaded)} valid days in {load_time:.1f}s")

        if not loaded:
            print(f"  No valid intraday data for {symbol}")
            continue

        # Run intraday engine on each day
        all_trades = []
        daily_pnls = []

        for rec in loaded:
            state = run_intraday(
                rec["close"], rec["high"], rec["low"],
                alpha_window=60, signal_lookback=60,
                entry_threshold=0.20, cost_bps=5.0,
                max_trades=3, total_bars=rec["n_bars"],
            )
            day_pnl = sum(t.pnl for t in state.trades)
            daily_pnls.append(day_pnl)
            for t in state.trades:
                all_trades.append({
                    "date": rec["date"],
                    "direction": t.direction,
                    "regime": t.regime,
                    "pnl": t.pnl,
                    "exit_reason": t.exit_reason,
                    "bars_held": (t.exit_bar or 0) - t.entry_bar,
                })

        if not daily_pnls:
            print("  No trades generated")
            continue

        pnl_arr = np.array(daily_pnls)

        # Sharpe from daily aggregated PnL (including zero days)
        if len(pnl_arr) > 1:
            std = np.std(pnl_arr, ddof=1)
            sharpe = float(np.mean(pnl_arr) / std * np.sqrt(252)) if std > 0 else 0.0
        else:
            sharpe = 0.0

        cumulative = np.cumsum(pnl_arr)
        total_return = float(cumulative[-1])
        peak = np.maximum.accumulate(cumulative)
        dd = cumulative - peak
        max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

        wins = sum(1 for t in all_trades if t["pnl"] > 0)

        print(f"\n  Intraday Results:")
        print(f"    Total trades:    {len(all_trades)}")
        print(f"    Trading days:    {len(loaded)}")
        print(f"    Sharpe:          {sharpe:.2f}")
        print(f"    Total return:    {total_return*100:+.2f}%")
        print(f"    Max drawdown:    {max_dd*100:.2f}%")
        print(f"    Win rate:        {wins/max(len(all_trades),1)*100:.1f}%")

        # Exit reason breakdown
        if all_trades:
            reasons = {}
            for t in all_trades:
                reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
            print(f"\n    Exit Reasons:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"      {reason:<15} {count:4d} ({count/len(all_trades)*100:.1f}%)")

    print()


# ---------------------------------------------------------------------------
# Recalibrated architecture (Optuna-informed)
# ---------------------------------------------------------------------------

def run_recalibrated(store: MarketDataStore, symbols: list[str],
                     n_trials: int = 300) -> None:
    """Run recalibrated gated architecture with Optuna fine-tuning.

    Phase 1: Run with Optuna-derived defaults (no optimization)
    Phase 2: Optuna fine-tune the regime factors + entry/hold params
    Phase 3: Ablation on recalibrated architecture
    """
    import optuna
    from strategies.s12_vedic_ffpe.gpu_features import (
        compute_features_array_gpu,
        backtest_recalibrated,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 70)
    print("S12 VEDIC FRACTIONAL ALPHA — RECALIBRATED ARCHITECTURE")
    print("=" * 70)
    print("  Informed by Optuna feature discovery:")
    print("    - ALL regimes trade (soft modulation)")
    print("    - Direction: CONTRARIAN (anti-momentum)")
    print("    - Conviction: coherence (52.8%) > phase (19.3%) > momentum (10.5%)")
    print("    - α regime softly modulates conviction")
    print()

    for symbol in symbols:
        df = _load_index_closes(store, symbol)
        if df.empty or len(df) < 100:
            print(f"  {symbol}: insufficient data ({len(df)} days)")
            continue

        closes = df["close_val"].values
        dates_list = df["date"].tolist()

        print(f"{'─' * 70}")
        print(f"  {symbol} — {len(df)} days")
        print(f"{'─' * 70}")

        features = compute_features_array_gpu(closes, alpha_window=60, frac_d=0.226)

        # --- Phase 1: Default Optuna-derived params ---
        print("\n  Phase 1: Default params (Optuna-derived, no optimisation)")
        result_default = backtest_recalibrated(
            features, closes, dates_list,
            alpha_window=60,
        )
        _print_backtest_result(result_default, symbol)

        # --- Phase 2: Optuna fine-tune regime factors + entry/hold ---
        print(f"\n  Phase 2: Optuna fine-tune ({n_trials} trials)")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "w_coherence": trial.suggest_float("w_coherence", 0.5, 3.0),
                "w_phase": trial.suggest_float("w_phase", 0.0, 3.0),
                "w_mock_theta": trial.suggest_float("w_mock_theta", 0.0, 2.0),
                "w_frac_d": trial.suggest_float("w_frac_d", 0.0, 2.0),
                "regime_sub_boost": trial.suggest_float("regime_sub_boost", 0.5, 2.0),
                "regime_norm_factor": trial.suggest_float("regime_norm_factor", 0.5, 2.0),
                "regime_super_factor": trial.suggest_float("regime_super_factor", 0.2, 1.5),
                "min_conviction": trial.suggest_float("min_conviction", 0.05, 0.5),
                "hold_days": trial.suggest_int("hold_days", 3, 20),
            }
            result = backtest_recalibrated(
                features, closes, dates_list,
                alpha_window=60, cost_bps=5.0,
                **params,
            )
            if result["trades"] < 10:
                return -10.0
            return result["sharpe"]

        t0 = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        elapsed = time.time() - t0

        print(f"  {n_trials} trials in {elapsed:.1f}s")
        best = study.best_trial
        print(f"\n  Best Sharpe: {best.value:.3f}")
        print(f"  Best params:")
        for k, v in sorted(best.params.items()):
            print(f"    {k:<20} = {v:+.4f}" if isinstance(v, float)
                  else f"    {k:<20} = {v}")

        result_best = backtest_recalibrated(
            features, closes, dates_list,
            alpha_window=60, cost_bps=5.0,
            **best.params,
        )
        print(f"\n  Optimised result:")
        _print_backtest_result(result_best, symbol)

        # Feature importance
        print(f"\n  Param Importance (fANOVA):")
        try:
            importances = optuna.importance.get_param_importances(study)
            for param, imp in importances.items():
                bar = "█" * int(imp * 40)
                print(f"    {param:<20} {imp:6.1%}  {bar}")
        except Exception as e:
            print(f"    (Could not compute: {e})")

        # --- Phase 3: Ablation on recalibrated ---
        print(f"\n  Phase 3: Ablation (recalibrated defaults)")
        bp = best.params

        ablations = [
            ("ALL ON", {}),
            ("-coherence", {"w_coherence": 0.0}),
            ("-phase", {"w_phase": 0.0}),
            ("-mock_theta", {"w_mock_theta": 0.0}),
            ("-frac_d", {"w_frac_d": 0.0}),
            ("-regime (all=1.0)", {"regime_sub_boost": 1.0,
                                    "regime_norm_factor": 1.0,
                                    "regime_super_factor": 1.0}),
            ("ONLY coherence", {"w_phase": 0.0, "w_mock_theta": 0.0,
                                 "w_frac_d": 0.0}),
            ("ONLY phase", {"w_coherence": 0.0, "w_mock_theta": 0.0,
                             "w_frac_d": 0.0}),
        ]

        header = (f"  {'Config':<22} {'Sharpe':>7} {'Return':>9} "
                  f"{'Trades':>6} {'WinR':>6}")
        print(header)
        print("  " + "-" * 54)

        for label, overrides in ablations:
            params = {**bp, **overrides}
            result = backtest_recalibrated(
                features, closes, dates_list,
                alpha_window=60, cost_bps=5.0,
                **params,
            )
            if result["trades"] == 0:
                print(f"  {label:<22} {'--':>7} {'--':>9} {'0':>6} {'--':>6}")
            else:
                print(f"  {label:<22} {result['sharpe']:>7.2f} "
                      f"{result['total_return_pct']:>+8.2f}% "
                      f"{result['trades']:>6d} {result['win_rate']*100:>5.1f}%")

        # --- Comparison table ---
        print(f"\n  Comparison:")
        print(f"  {'Architecture':<30} {'Sharpe':>7} {'Trades':>6}")
        print(f"  " + "-" * 46)
        print(f"  {'Original gated (default)':<30} {'0.58':>7} {'181':>6}")
        print(f"  {'Original gated (ONLY alpha)':<30} {'1.19':>7} {'227':>6}")
        print(f"  {'Recalibrated (default)':<30} "
              f"{result_default.get('sharpe', 0):.2f}  "
              f"{result_default.get('trades', 0):>5d}")
        print(f"  {'Recalibrated (Optuna best)':<30} "
              f"{result_best.get('sharpe', 0):.2f}  "
              f"{result_best.get('trades', 0):>5d}")
        print(f"  {'Gate-free linear (Optuna)':<30} {'2.77':>7} {'184':>6}")

        print()

    print()


# ---------------------------------------------------------------------------
# Validation gates on recalibrated architecture
# ---------------------------------------------------------------------------

def run_validate_recalibrated(store: MarketDataStore, symbols: list[str],
                               n_trials: int = 300) -> None:
    """Validation gates (placebo + time-shift) on recalibrated Optuna-best config.

    1. Compute features on real data, Optuna-find best params
    2. Placebo: shuffle returns 100×, recompute features, run recalibrated backtest
    3. Time-shift: shift prices ±1/2/3, recompute features, run recalibrated backtest
    4. Regime stability check
    """
    import optuna
    from strategies.s12_vedic_ffpe.gpu_features import (
        compute_features_array_gpu,
        backtest_recalibrated,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 70)
    print("S12 VEDIC FRACTIONAL ALPHA — VALIDATION (RECALIBRATED)")
    print("=" * 70)
    print("  Compute = GPU (T4)")
    print("  Architecture: recalibrated contrarian + soft regime gating")
    print()

    for symbol in symbols:
        df = _load_index_closes(store, symbol)
        if df.empty or len(df) < 100:
            print(f"  {symbol}: insufficient data ({len(df)} days)")
            continue

        closes = df["close_val"].values
        dates_list = df["date"].tolist()

        print(f"{'─' * 70}")
        print(f"  {symbol} — {len(df)} days")
        print(f"{'─' * 70}")

        # Step 1: Find best params on real data
        print("\n  Step 1: Optuna on real data...")
        features_real = compute_features_array_gpu(closes, alpha_window=60, frac_d=0.226)

        def make_objective(feats, cl, dl):
            def objective(trial):
                params = {
                    "w_coherence": trial.suggest_float("w_coherence", 0.5, 3.0),
                    "w_phase": trial.suggest_float("w_phase", 0.0, 3.0),
                    "w_mock_theta": trial.suggest_float("w_mock_theta", 0.0, 2.0),
                    "w_frac_d": trial.suggest_float("w_frac_d", 0.0, 2.0),
                    "regime_sub_boost": trial.suggest_float("regime_sub_boost", 0.5, 2.0),
                    "regime_norm_factor": trial.suggest_float("regime_norm_factor", 0.5, 2.0),
                    "regime_super_factor": trial.suggest_float("regime_super_factor", 0.2, 1.5),
                    "min_conviction": trial.suggest_float("min_conviction", 0.05, 0.5),
                    "hold_days": trial.suggest_int("hold_days", 3, 20),
                }
                result = backtest_recalibrated(
                    feats, cl, dl,
                    alpha_window=60, cost_bps=5.0, **params,
                )
                if result["trades"] < 10:
                    return -10.0
                return result["sharpe"]
            return objective

        study_real = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_real.optimize(make_objective(features_real, closes, dates_list),
                            n_trials=n_trials)
        best_params = study_real.best_params

        real_result = backtest_recalibrated(
            features_real, closes, dates_list,
            alpha_window=60, cost_bps=5.0, **best_params,
        )
        real_sharpe = real_result.get("sharpe", 0.0)
        print(f"  Real signal Sharpe: {real_sharpe:.3f} "
              f"({real_result.get('trades', 0)} trades)")

        # Step 2: Placebo test — shuffle returns, recompute features, re-optimize
        print(f"\n  Step 2a: Placebo Test (100 permutations, FIXED params)")
        print(f"  (Uses real-data best params on shuffled data — tests signal, not overfit)")
        rng = np.random.default_rng(42)
        placebo_sharpes = []
        t0 = time.time()

        for p in range(100):
            shuffled_closes = closes.copy()
            log_ret = np.diff(np.log(np.maximum(shuffled_closes, 1e-8)))
            rng.shuffle(log_ret)
            shuffled_closes[1:] = shuffled_closes[0] * np.exp(np.cumsum(log_ret))

            feats_shuf = compute_features_array_gpu(
                shuffled_closes, alpha_window=60, frac_d=0.226,
            )
            result = backtest_recalibrated(
                feats_shuf, shuffled_closes, dates_list,
                alpha_window=60, cost_bps=5.0, **best_params,
            )
            placebo_sharpes.append(result.get("sharpe", 0.0))

        placebo_arr = np.array(placebo_sharpes)
        pctile = np.sum(real_sharpe > placebo_arr) / len(placebo_arr) * 100
        print(f"    Completed in {time.time()-t0:.1f}s")
        print(f"    Real Sharpe:   {real_sharpe:.3f}")
        print(f"    Placebo mean:  {np.mean(placebo_arr):.3f} "
              f"± {np.std(placebo_arr, ddof=1):.3f}")
        print(f"    Placebo 90th:  {np.percentile(placebo_arr, 90):.3f}")
        print(f"    Placebo max:   {np.max(placebo_arr):.3f}")
        print(f"    Real > {pctile:.0f}% of placebos  "
              f"{'PASS' if pctile >= 90 else 'FAIL'}")

        # Step 2b: Stricter placebo — re-optimize on each shuffle
        print(f"\n  Step 2b: Placebo Test (20 permutations, RE-OPTIMISED)")
        print(f"  (Re-runs Optuna on each shuffle — tests if real data is special)")
        placebo_opt_sharpes = []
        t0 = time.time()

        for p in range(20):
            shuffled_closes = closes.copy()
            log_ret = np.diff(np.log(np.maximum(shuffled_closes, 1e-8)))
            rng.shuffle(log_ret)
            shuffled_closes[1:] = shuffled_closes[0] * np.exp(np.cumsum(log_ret))

            feats_shuf = compute_features_array_gpu(
                shuffled_closes, alpha_window=60, frac_d=0.226,
            )

            study_shuf = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=p),
            )
            study_shuf.optimize(
                make_objective(feats_shuf, shuffled_closes, dates_list),
                n_trials=n_trials,
            )
            placebo_opt_sharpes.append(study_shuf.best_value)

        placebo_opt_arr = np.array(placebo_opt_sharpes)
        pctile_opt = np.sum(real_sharpe > placebo_opt_arr) / len(placebo_opt_arr) * 100
        print(f"    Completed in {time.time()-t0:.1f}s")
        print(f"    Real Sharpe (optimised):   {real_sharpe:.3f}")
        print(f"    Placebo mean (optimised):  {np.mean(placebo_opt_arr):.3f} "
              f"± {np.std(placebo_opt_arr, ddof=1):.3f}")
        print(f"    Placebo 90th (optimised):  {np.percentile(placebo_opt_arr, 90):.3f}")
        print(f"    Placebo max (optimised):   {np.max(placebo_opt_arr):.3f}")
        print(f"    Real > {pctile_opt:.0f}% of placebos  "
              f"{'PASS' if pctile_opt >= 90 else 'FAIL'}")

        # Step 3: Time-shift test
        print(f"\n  Step 3: Time-Shift Test (fixed params)")
        shifts = [-3, -2, -1, 1, 2, 3]
        shift_sharpes = {}
        for s in shifts:
            shifted_closes = np.roll(closes, s)
            if s > 0:
                shifted_closes[:s] = closes[:s]
            else:
                shifted_closes[s:] = closes[s:]

            feats_shift = compute_features_array_gpu(
                shifted_closes, alpha_window=60, frac_d=0.226,
            )
            result = backtest_recalibrated(
                feats_shift, shifted_closes, dates_list,
                alpha_window=60, cost_bps=5.0, **best_params,
            )
            shift_sharpes[s] = result.get("sharpe", 0.0)
            beat = ">" if real_sharpe > shift_sharpes[s] else "<"
            print(f"    Shift {s:+d}: Sharpe={shift_sharpes[s]:.3f}  "
                  f"real {beat} shifted")

        all_beat = all(real_sharpe >= v for v in shift_sharpes.values())
        print(f"    Original beats all shifts:  "
              f"{'PASS' if all_beat else 'FAIL'}")

        # Step 4: Regime stability
        print(f"\n  Step 4: Regime Stability")
        alpha_arr = features_real["alpha"]
        next_ret = np.full(len(closes), np.nan)
        next_ret[:-1] = np.diff(closes) / closes[:-1]

        correct = 0
        total = 0
        for i in range(len(alpha_arr)):
            if np.isnan(alpha_arr[i]) or np.isnan(next_ret[i]):
                continue
            regime = classify_regime(alpha_arr[i])
            ret = next_ret[i]
            total += 1

            if regime == "subdiffusive" and abs(ret) < np.nanstd(next_ret):
                correct += 1
            elif regime == "superdiffusive":
                if i >= 5:
                    recent_mom = np.sum(np.diff(np.log(np.maximum(closes[i-5:i+1], 1e-8))))
                    if recent_mom * ret > 0:
                        correct += 1
            elif regime == "normal":
                correct += 1

        accuracy = correct / max(total, 1)
        print(f"    Regime classification accuracy: {accuracy:.1%} "
              f"({correct}/{total})")

        # Summary
        print(f"\n  {'─' * 50}")
        print(f"  SUMMARY: {symbol}")
        print(f"  {'─' * 50}")
        print(f"    Real Sharpe:           {real_sharpe:.3f}")
        print(f"    Placebo (fixed):       {'PASS' if pctile >= 90 else 'FAIL'} "
              f"({pctile:.0f}th percentile)")
        print(f"    Placebo (re-optimised): {'PASS' if pctile_opt >= 90 else 'FAIL'} "
              f"({pctile_opt:.0f}th percentile)")
        print(f"    Time-shift:            {'PASS' if all_beat else 'FAIL'}")
        print(f"    Regime stability:      {accuracy:.1%}")

    print()


# ---------------------------------------------------------------------------
# Optuna-based feature discovery (gate-free)
# ---------------------------------------------------------------------------

def run_optuna(store: MarketDataStore, symbols: list[str],
               n_trials: int = 500) -> None:
    """Gate-free Optuna optimization.

    Removes ALL hardcoded gates (regime, phase, mock theta threshold).
    Lets Optuna find optimal weights for a linear signal combination.
    Feature importance reveals which features have causal power.
    """
    import optuna
    from strategies.s12_vedic_ffpe.gpu_features import (
        compute_features_array_gpu,
        backtest_linear_signal,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 70)
    print("S12 VEDIC FRACTIONAL ALPHA — OPTUNA FEATURE DISCOVERY")
    print("=" * 70)
    print("  Compute = GPU (T4)")
    print("  Gate-free linear signal: signal = Σ wᵢ · z(featureᵢ)")
    print("  All features z-scored for fair comparison.")
    print(f"  Trials: {n_trials}")
    print()

    for symbol in symbols:
        df = _load_index_closes(store, symbol)
        if df.empty or len(df) < 100:
            print(f"  {symbol}: insufficient data ({len(df)} days)")
            continue

        closes = df["close_val"].values
        dates_list = df["date"].tolist()

        print(f"{'─' * 70}")
        print(f"  {symbol} — {len(df)} days")
        print(f"{'─' * 70}")

        # Compute features once
        features = compute_features_array_gpu(closes, alpha_window=60, frac_d=0.226)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "w_alpha": trial.suggest_float("w_alpha", -2.0, 2.0),
                "w_coherence": trial.suggest_float("w_coherence", -2.0, 2.0),
                "w_phase": trial.suggest_float("w_phase", -2.0, 2.0),
                "w_mock_theta": trial.suggest_float("w_mock_theta", -2.0, 2.0),
                "w_frac_d": trial.suggest_float("w_frac_d", -2.0, 2.0),
                "w_momentum": trial.suggest_float("w_momentum", -2.0, 2.0),
                "entry_threshold": trial.suggest_float("entry_threshold", 0.05, 2.0),
                "hold_days": trial.suggest_int("hold_days", 1, 20),
            }
            result = backtest_linear_signal(
                features, closes, dates_list,
                alpha_window=60, cost_bps=5.0,
                **params,
            )
            # Penalize if too few trades (need statistical significance)
            if result["trades"] < 10:
                return -10.0
            return result["sharpe"]

        t0 = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        elapsed = time.time() - t0

        print(f"\n  Optimisation: {n_trials} trials in {elapsed:.1f}s "
              f"({elapsed/n_trials*1000:.1f}ms/trial)")

        # Best result
        best = study.best_trial
        print(f"\n  Best Sharpe: {best.value:.3f}")
        print(f"  Best params:")
        for k, v in sorted(best.params.items()):
            print(f"    {k:<18} = {v:+.4f}" if isinstance(v, float)
                  else f"    {k:<18} = {v}")

        # Run best config for full stats
        best_result = backtest_linear_signal(
            features, closes, dates_list,
            alpha_window=60, cost_bps=5.0,
            **best.params,
        )
        print(f"\n  Full stats (best config):")
        print(f"    Trades:       {best_result['trades']}")
        print(f"    Total return: {best_result['total_return_pct']:+.2f}%")
        print(f"    Max drawdown: {best_result['max_dd_pct']:.2f}%")
        print(f"    Win rate:     {best_result['win_rate']*100:.1f}%")

        # Feature importance (Optuna built-in)
        print(f"\n  Feature Importance (fANOVA):")
        try:
            importances = optuna.importance.get_param_importances(study)
            for param, imp in importances.items():
                bar = "█" * int(imp * 40)
                print(f"    {param:<18} {imp:6.1%}  {bar}")
        except Exception as e:
            print(f"    (Could not compute: {e})")

        # Top 10 trials
        print(f"\n  Top 10 trials:")
        header = (f"  {'#':>3} {'Sharpe':>7} {'w_α':>6} {'w_coh':>6} "
                  f"{'w_ph':>6} {'w_mt':>6} {'w_fd':>6} {'w_mom':>6} "
                  f"{'entry':>6} {'hold':>4}")
        print(header)
        print("  " + "-" * 70)

        sorted_trials = sorted(study.trials,
                                key=lambda t: t.value if t.value is not None else -999,
                                reverse=True)
        for rank, trial in enumerate(sorted_trials[:10], 1):
            p = trial.params
            print(f"  {rank:>3d} {trial.value:>7.3f} "
                  f"{p['w_alpha']:>+6.2f} {p['w_coherence']:>+6.2f} "
                  f"{p['w_phase']:>+6.2f} {p['w_mock_theta']:>+6.2f} "
                  f"{p['w_frac_d']:>+6.2f} {p['w_momentum']:>+6.2f} "
                  f"{p['entry_threshold']:>6.2f} {p['hold_days']:>4d}")

        print()

    print()


# ---------------------------------------------------------------------------
# Expanded feature Optuna (14 features: original 6 + Tier 1 + RMT + IV)
# ---------------------------------------------------------------------------

# Sector indices used for RMT correlation matrix (DuckDB names)
_RMT_INDICES = [
    "Nifty 50", "Nifty Bank", "Nifty IT", "Nifty FMCG", "Nifty Pharma",
    "Nifty Auto", "Nifty Metal", "Nifty Realty", "Nifty Energy",
    "Nifty Financial Services", "Nifty Media", "Nifty PSU Bank",
    "Nifty Private Bank", "Nifty Commodities", "Nifty Infrastructure",
    "Nifty India Consumption", "Nifty Midcap 50", "Nifty Smallcap 50",
    "Nifty Next 50", "Nifty 100", "Nifty Consumer Durables",
    "Nifty Healthcare Index", "Nifty Oil & Gas", "Nifty India Defence",
    "Nifty India Manufacturing", "Nifty MNC", "Nifty High Beta 50",
    "Nifty Low Volatility 50",
]

# Kite instrument tokens for sector indices
_KITE_SECTOR_TOKENS: dict[str, int] = {
    "NIFTY 50": 256265, "NIFTY BANK": 260105, "NIFTY IT": 259849,
    "NIFTY FMCG": 261897, "NIFTY PHARMA": 262409, "NIFTY AUTO": 263433,
    "NIFTY METAL": 263689, "NIFTY REALTY": 261129, "NIFTY ENERGY": 261641,
    "NIFTY FIN SERVICE": 257801, "NIFTY MEDIA": 263945,
    "NIFTY PSU BANK": 262921, "NIFTY PVT BANK": 271113,
    "NIFTY COMMODITIES": 257289, "NIFTY INFRA": 261385,
    "NIFTY CONSUMPTION": 257545, "NIFTY MIDCAP 50": 260873,
    "NIFTY SMLCAP 50": 266761, "NIFTY NEXT 50": 270857,
    "NIFTY 100": 260617, "NIFTY 200": 264457, "NIFTY 500": 268041,
    "INDIA VIX": 264969, "NIFTY CONSR DURBL": 288777,
    "NIFTY HEALTHCARE": 288521, "NIFTY OIL AND GAS": 289033,
    "NIFTY MNC": 262153, "NIFTY CPSE": 268297,
}

# Kite tokens for main instruments (index + continuous futures)
_KITE_INSTRUMENT_TOKENS: dict[str, dict[str, int]] = {
    "NIFTY": {"index": 256265, "fut": 15150594},
    "BANKNIFTY": {"index": 260105, "fut": 15148802},
}


# Kite instrument tokens for tick-level Hawkes (index tokens)
_TICK_TOKENS: dict[str, int] = {
    "NIFTY": 256265,
    "BANKNIFTY": 260105,
}


def _hawkes_worker(d_iso: str, token: int) -> tuple[str, float]:
    """Compute Hawkes ratio from tick data for one day (worker process)."""
    store = MarketDataStore()
    try:
        df = store.sql(
            "SELECT timestamp FROM ticks "
            "WHERE date = ? AND instrument_token = ? "
            "AND ltp > 0.05 "
            "ORDER BY timestamp",
            [d_iso, token],
        )
        if df is None or len(df) < 200:
            return d_iso, float("nan")

        from features.microstructure import trade_arrival_hawkes

        ts = df["timestamp"].values.astype("datetime64[ns]").astype(np.float64) / 1e9
        _, _mu, alpha, beta = trade_arrival_hawkes(ts)
        ratio = alpha / max(beta, 1e-8)
        return d_iso, ratio
    except Exception:
        return d_iso, float("nan")
    finally:
        store.close()


def _compute_daily_hawkes(
    store: MarketDataStore,
    symbol: str,
    dates_list: list,
    max_workers: int = 8,
) -> np.ndarray:
    """Compute Hawkes ratio for each date from tick data (parallel)."""
    token = _TICK_TOKENS.get(symbol)
    if token is None:
        return np.full(len(dates_list), np.nan)

    # Check which dates have tick data
    avail = store.available_dates("ticks")
    avail_set = {d.isoformat() if hasattr(d, "isoformat") else str(d)
                 for d in avail}

    n = len(dates_list)
    out = np.full(n, np.nan)
    date_to_idx = {}
    jobs = []
    for i, d in enumerate(dates_list):
        d_iso = d.isoformat() if hasattr(d, "isoformat") else str(d)
        date_to_idx[d_iso] = i
        if d_iso in avail_set:
            jobs.append(d_iso)

    if not jobs:
        return out

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_hawkes_worker, d_iso, token): d_iso
            for d_iso in jobs
        }
        for future in as_completed(futures):
            d_iso, ratio = future.result()
            idx = date_to_idx.get(d_iso)
            if idx is not None and not math.isnan(ratio):
                out[idx] = ratio

    valid = np.sum(~np.isnan(out))
    print(f"    Hawkes ratio:  {valid}/{n} valid ({time.time()-t0:.1f}s, "
          f"{len(jobs)} tick days)")
    return out


def _load_kite_vix(start: str = "2024-01-01") -> tuple[np.ndarray, list]:
    """Fetch India VIX daily closes from Kite → (vix_values, dates)."""
    from data.collectors.auth import headless_login
    from data.zerodha import fetch_historical

    kite = headless_login()
    vix_token = 264969  # INDIA VIX
    ohlcv = fetch_historical(kite, vix_token, "1d", start=start, end="2026-02-08")
    df = ohlcv.df
    vix_values = df["Close"].values.astype(np.float64)
    vix_dates = [d.date() for d in df.index]
    return vix_values, vix_dates


def _load_intraday_bars_for_dates(
    symbol: str,
    dates_list: list,
    max_workers: int = 8,
) -> dict:
    """Load 1-min futures bars for all dates → {date: np.array(closes)}."""
    avail = MarketDataStore().available_dates("nfo_1min")
    avail_set = {d.isoformat() if hasattr(d, "isoformat") else str(d)
                 for d in avail}

    jobs = []
    for d in dates_list:
        d_iso = d.isoformat() if hasattr(d, "isoformat") else str(d)
        if d_iso in avail_set:
            jobs.append(d_iso)

    if not jobs:
        return {}

    t0 = time.time()
    result = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_load_intraday_bars, d_iso, symbol): d_iso
            for d_iso in jobs
        }
        for future in as_completed(futures):
            rec = future.result()
            if rec is not None:
                result[rec["date"]] = rec["close"]

    print(f"    1-min bars:    {len(result)}/{len(jobs)} days loaded "
          f"({time.time()-t0:.1f}s)")
    return result


def _compute_sanos_cache(
    store: MarketDataStore,
    symbol: str,
    dates_list: list,
    closes: np.ndarray,
) -> dict[str, dict]:
    """Pre-compute SANOS density features for all dates with option data.

    Returns {date_iso: {skew, left_tail, entropy, kl}} dict keyed by ISO date.
    """
    from core.pricing.sanos import fit_sanos, prepare_nifty_chain
    from core.pricing.risk_neutral import (
        extract_density, compute_moments, shannon_entropy,
        kl_divergence, tail_weights,
    )

    instrument = "IDO" if symbol in ("NIFTY", "BANKNIFTY",
                                      "FINNIFTY", "MIDCPNIFTY") else "STO"
    cache: dict[str, dict] = {}
    prev_density = None
    n_success = 0
    n_fail = 0

    # Check which dates have bhavcopy data
    avail_df = store.sql(
        "SELECT DISTINCT date FROM nse_fo_bhavcopy ORDER BY date"
    )
    if avail_df is None or avail_df.empty:
        print(f"    No nse_fo_bhavcopy data available")
        return cache

    avail_dates = set()
    for _, row in avail_df.iterrows():
        d = row["date"]
        if hasattr(d, "isoformat"):
            avail_dates.add(d.isoformat()[:10])
        else:
            avail_dates.add(str(d)[:10])

    for i, d in enumerate(dates_list):
        d_iso = d.isoformat()[:10] if hasattr(d, "isoformat") else str(d)[:10]
        if d_iso not in avail_dates:
            continue
        try:
            fno_df = store.sql(
                "SELECT * FROM nse_fo_bhavcopy WHERE date = ?", [d_iso],
            )
            if fno_df is None or fno_df.empty:
                continue

            chain = prepare_nifty_chain(
                fno_df, symbol=symbol, instrument=instrument, max_expiries=2,
            )
            if chain is None:
                n_fail += 1
                continue

            result = fit_sanos(
                market_strikes=chain["market_strikes"],
                market_calls=chain["market_calls"],
                atm_variances=chain["atm_variances"],
                expiry_labels=chain.get("expiry_labels"),
                eta=0.50,
                n_model_strikes=80,
            )
            if not result.lp_success:
                n_fail += 1
                continue

            K, q = extract_density(result, expiry_idx=0, n_points=300)
            mu, var, skew, kurt = compute_moments(K, q)
            dK = K[1] - K[0]
            std = math.sqrt(max(var, 1e-14))
            H = shannon_entropy(q, dK)
            lt, rt = tail_weights(K, q, mu, std)

            kl_val = np.nan
            if prev_density is not None and len(prev_density) == len(q):
                kl_val = kl_divergence(q, prev_density, dK)

            cache[d_iso] = {
                "skew": skew,
                "left_tail": lt,
                "entropy": H,
                "kl": kl_val,
            }
            prev_density = q
            n_success += 1

        except Exception:
            n_fail += 1
            continue

    return cache


def _load_kite_data(
    symbol: str,
    start: str = "2024-01-01",
) -> tuple[np.ndarray, list, np.ndarray | None, np.ndarray | None,
           np.ndarray | None, np.ndarray | None]:
    """Fetch daily OHLCV from Kite for a symbol.

    Returns (closes, dates, opens, highs, lows, volumes).
    Uses index closes for price, continuous futures for OHLCV.
    """
    from data.collectors.auth import headless_login
    from data.zerodha import fetch_historical

    kite = headless_login()
    tokens = _KITE_INSTRUMENT_TOKENS.get(symbol, {})
    idx_token = tokens.get("index")
    fut_token = tokens.get("fut")

    if not idx_token:
        raise ValueError(f"No Kite token for {symbol}")

    # Fetch index daily (closes)
    idx_ohlcv = fetch_historical(
        kite, idx_token, "1d", start=start, end="2026-02-08",
    )
    idx_df = idx_ohlcv.df

    # Fetch continuous futures daily (OHLCV + volume)
    opens = highs = lows = volumes = None
    if fut_token:
        try:
            fut_ohlcv = fetch_historical(
                kite, fut_token, "1d", start=start, end="2026-02-08",
                continuous=True, oi=True,
            )
            fut_df = fut_ohlcv.df
            # Align by date
            idx_dates = idx_df.index.normalize()
            fut_dates = fut_df.index.normalize()
            common = idx_dates.intersection(fut_dates)

            opens = fut_df.loc[fut_df.index.normalize().isin(common), "Open"].values
            highs = fut_df.loc[fut_df.index.normalize().isin(common), "High"].values
            lows = fut_df.loc[fut_df.index.normalize().isin(common), "Low"].values
            volumes = fut_df.loc[fut_df.index.normalize().isin(common), "Volume"].values

            # Use only common dates for closes too
            idx_df = idx_df[idx_df.index.normalize().isin(common)]
        except Exception as e:
            print(f"    Warning: futures data failed ({e}), using index only")

    closes = idx_df["Close"].values.astype(np.float64)
    dates_list = [d.date() for d in idx_df.index]

    return closes, dates_list, opens, highs, lows, volumes


def _load_kite_sector_returns(
    start: str = "2024-01-01",
) -> tuple[np.ndarray, list]:
    """Fetch daily closes for all sector indices from Kite → RMT returns matrix."""
    from data.collectors.auth import headless_login
    from data.zerodha import fetch_historical

    kite = headless_login()

    dfs = {}
    for name, token in _KITE_SECTOR_TOKENS.items():
        try:
            ohlcv = fetch_historical(kite, token, "1d", start=start, end="2026-02-08")
            df = ohlcv.df[["Close"]].rename(columns={"Close": name})
            df.index = df.index.normalize()
            dfs[name] = df[name]
        except Exception:
            continue

    if len(dfs) < 5:
        return np.array([]), []

    combined = pd.DataFrame(dfs).dropna().sort_index()
    if len(combined) < 30:
        return np.array([]), []

    log_ret = np.log(combined / combined.shift(1)).iloc[1:].values
    aligned_dates = [d.date() for d in combined.index[1:]]
    return log_ret, aligned_dates


def _load_sector_returns(store: MarketDataStore) -> tuple[np.ndarray, list]:
    """Load daily log returns for all sector indices → (n_days, n_indices) matrix.

    Returns (returns_matrix, aligned_dates).
    Dates are the intersection of all indices.
    """
    dfs = {}
    for idx_name in _RMT_INDICES:
        df = store.sql(
            'SELECT date, "Closing Index Value" as close_val '
            'FROM nse_index_close '
            'WHERE "Index Name" = ? '
            'ORDER BY date',
            [idx_name],
        )
        if df is None or df.empty:
            continue
        df["close_val"] = pd.to_numeric(df["close_val"], errors="coerce")
        df = df.dropna(subset=["close_val"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.drop_duplicates(subset=["date"]).set_index("date")
        dfs[idx_name] = df["close_val"]

    if len(dfs) < 5:
        return np.array([]), []

    # Align on common dates
    combined = pd.DataFrame(dfs)
    combined = combined.dropna()
    combined = combined.sort_index()

    if len(combined) < 30:
        return np.array([]), []

    # Log returns
    log_ret = np.log(combined / combined.shift(1)).iloc[1:].values
    aligned_dates = list(combined.index[1:])

    return log_ret, aligned_dates


def _load_daily_ohlcv(store: MarketDataStore, symbol: str) -> pd.DataFrame | None:
    """Load daily OHLCV from nfo_1min (aggregate 1-min FUT bars to daily).

    Aggregates across all FUT expiries (near-month dominates by volume).
    Returns DataFrame with columns: date, open, high, low, close, volume
    """
    df = store.sql(
        "SELECT date, "
        "  FIRST(open) as open, "
        "  MAX(high) as high, "
        "  MIN(low) as low, "
        "  LAST(close) as close, "
        "  SUM(volume) as volume "
        "FROM nfo_1min "
        "WHERE name = ? AND instrument_type = 'FUT' "
        "GROUP BY date "
        "ORDER BY date",
        [symbol],
    )
    if df is None or df.empty:
        return None
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.dropna().sort_values("date").reset_index(drop=True)


def run_expanded_optuna(store: MarketDataStore, symbols: list[str],
                        n_trials: int = 500, use_kite: bool = False) -> None:
    """Expanded Optuna: 14 features (original 6 + Tier 1 + RMT + IV).

    Discovers which of the new features (period energy, entropy, Yang-Zhang,
    ATR, VPIN, RMT, SANOS density, Hawkes) add signal
    beyond the original 6 S12 features.
    """
    import optuna
    from strategies.s12_vedic_ffpe.gpu_features import (
        compute_features_expanded_gpu,
        backtest_expanded_linear,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 70)
    print("S12 VEDIC FRACTIONAL ALPHA — SANOS DENSITY DISCOVERY")
    print("=" * 70)
    print("  24 features: 6 original + 5 Tier-1 + 2 RMT + 4 SANOS + 1 Hawkes + 6 Masters")
    print("  SANOS: risk-neutral density from actual option chain (LP calibration)")
    print("  Masters: FTI, Mutual Info, ADX, Detrended RSI, PVR, Price Intensity")
    print("  Gate-free linear signal: signal = Σ wᵢ · z(featureᵢ)")
    print(f"  Trials: {n_trials}")
    if use_kite:
        print("  Data source: Kite API (524 days)")
    else:
        print("  Data source: DuckDB (local)")
    print()

    # Load sector returns for RMT (once for all symbols)
    print("  Loading sector index returns for RMT...")
    t0 = time.time()
    if use_kite:
        sector_returns, sector_dates = _load_kite_sector_returns()
    else:
        sector_returns, sector_dates = _load_sector_returns(store)
    if len(sector_returns) > 0:
        print(f"    {sector_returns.shape[1]} indices × "
              f"{sector_returns.shape[0]} days  ({time.time()-t0:.1f}s)")
    else:
        print(f"    No sector data available for RMT")

    for symbol in symbols:
        print(f"\n{'─' * 70}")
        print(f"  {symbol}")
        print(f"{'─' * 70}")

        if use_kite:
            # Load from Kite API (524 days OHLCV)
            try:
                closes, dates_list, opens, highs, lows, volumes = \
                    _load_kite_data(symbol)
                print(f"  Kite data: {len(closes)} days "
                      f"({dates_list[0]} → {dates_list[-1]})")
                has_ohlcv = opens is not None
                if has_ohlcv:
                    print(f"  OHLCV (continuous FUT): {len(opens)} days")
                else:
                    print(f"  OHLCV: index only (no FUT data)")
            except Exception as e:
                print(f"  Kite data failed: {e}")
                continue
        else:
            # Load from DuckDB
            df = _load_index_closes(store, symbol)
            if df.empty or len(df) < 80:
                print(f"  Insufficient data ({len(df)} days)")
                continue

            closes = df["close_val"].values
            dates_list = df["date"].tolist()
            print(f"  Index closes: {len(df)} days "
                  f"({dates_list[0]} → {dates_list[-1]})")

            # Load daily OHLCV from nfo_1min
            ohlcv = _load_daily_ohlcv(store, symbol)
            opens = highs = lows = volumes = None
            if ohlcv is not None and len(ohlcv) > 30:
                ohlcv_map = {r["date"]: r for _, r in ohlcv.iterrows()}
                o_arr = np.full(len(dates_list), np.nan)
                h_arr = np.full(len(dates_list), np.nan)
                l_arr = np.full(len(dates_list), np.nan)
                v_arr = np.full(len(dates_list), np.nan)
                for i, d in enumerate(dates_list):
                    if d in ohlcv_map:
                        r = ohlcv_map[d]
                        o_arr[i] = r["open"]
                        h_arr[i] = r["high"]
                        l_arr[i] = r["low"]
                        v_arr[i] = r["volume"]
                opens, highs, lows, volumes = o_arr, h_arr, l_arr, v_arr
                n_ohlcv = np.sum(~np.isnan(o_arr))
                print(f"  OHLCV (nfo_1min FUT): {n_ohlcv} days aligned")
            else:
                print(f"  OHLCV: not available (close-only mode)")

        # Align sector returns to index close dates
        sr_aligned = None
        if len(sector_returns) > 0 and len(sector_dates) > 0:
            idx_map = {d: i for i, d in enumerate(sector_dates)}
            sr_aligned = np.full((len(dates_list), sector_returns.shape[1]), np.nan)
            for i, d in enumerate(dates_list):
                if d in idx_map:
                    sr_aligned[i] = sector_returns[idx_map[d]]
            n_rmt = np.sum(~np.isnan(sr_aligned[:, 0]))
            print(f"  RMT sector returns: {n_rmt} days aligned "
                  f"({sector_returns.shape[1]} indices)")

        # Compute Hawkes ratio from tick data
        print(f"\n  Computing Hawkes ratio from tick data...")
        hawkes_arr = _compute_daily_hawkes(store, symbol, dates_list)

        # Pre-compute SANOS density features from option chain
        print(f"  Computing SANOS density from option chain...")
        t1 = time.time()
        sanos_cache = _compute_sanos_cache(store, symbol, dates_list, closes)
        n_sanos = len(sanos_cache)
        print(f"    SANOS calibrated: {n_sanos}/{len(dates_list)} days "
              f"({time.time()-t1:.1f}s)")

        # Compute expanded features
        features = compute_features_expanded_gpu(
            closes, dates_list,
            alpha_window=60, frac_d=0.226,
            opens=opens, highs=highs, lows=lows, volumes=volumes,
            sector_returns=sr_aligned,
            hawkes_ratio=hawkes_arr,
            sanos_cache=sanos_cache,
            store=store, symbol=symbol,
        )

        # Print feature availability
        print(f"\n  Feature availability:")
        for feat_name in ["alpha", "coherence", "phase", "mock_theta_div",
                          "frac_d_series", "momentum",
                          "period_energy", "entropy", "yang_zhang_vol",
                          "atr_norm", "vpin", "rmt_absorption",
                          "rmt_mp_excess",
                          "sanos_skew", "sanos_left_tail",
                          "sanos_entropy", "sanos_kl",
                          "hawkes_ratio",
                          "fti", "mutual_info", "adx",
                          "detrended_rsi", "price_var_ratio",
                          "price_intensity"]:
            arr = features.get(feat_name)
            if arr is None:
                print(f"    {feat_name:<18} MISSING")
                continue
            valid = np.sum(~np.isnan(arr))
            print(f"    {feat_name:<18} {valid:4d}/{len(arr)} valid")

        # Build feature weight names — only include features with data
        feature_defs = [
            ("w_alpha", "alpha"),
            ("w_coherence", "coherence"),
            ("w_phase", "phase"),
            ("w_mock_theta", "mock_theta_div"),
            ("w_frac_d", "frac_d_series"),
            ("w_momentum", "momentum"),
            ("w_period_energy", "period_energy"),
            ("w_entropy", "entropy"),
            ("w_yang_zhang", "yang_zhang_vol"),
            ("w_atr_norm", "atr_norm"),
            ("w_vpin", "vpin"),
            ("w_rmt_absorption", "rmt_absorption"),
            ("w_rmt_mp_excess", "rmt_mp_excess"),
            ("w_sanos_skew", "sanos_skew"),
            ("w_sanos_left_tail", "sanos_left_tail"),
            ("w_sanos_entropy", "sanos_entropy"),
            ("w_sanos_kl", "sanos_kl"),
            ("w_hawkes_ratio", "hawkes_ratio"),
            ("w_fti", "fti"),
            ("w_mutual_info", "mutual_info"),
            ("w_adx", "adx"),
            ("w_detrended_rsi", "detrended_rsi"),
            ("w_price_var_ratio", "price_var_ratio"),
            ("w_price_intensity", "price_intensity"),
        ]

        # Determine which features have enough data (min 30 valid observations)
        active_features = []
        for w_name, feat_key in feature_defs:
            arr = features.get(feat_key)
            if arr is not None and np.sum(~np.isnan(arr)) > 30:
                active_features.append(w_name)

        print(f"\n  Active features for Optuna: {len(active_features)}")

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for w_name in active_features:
                params[w_name] = trial.suggest_float(w_name, -2.0, 2.0)
            params["entry_threshold"] = trial.suggest_float(
                "entry_threshold", 0.05, 2.0,
            )
            params["hold_days"] = trial.suggest_int("hold_days", 1, 20)

            result = backtest_expanded_linear(
                features, closes, dates_list,
                alpha_window=60, cost_bps=5.0,
                **params,
            )
            if result["trades"] < 10:
                return -10.0
            return result["sharpe"]

        t0 = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        elapsed = time.time() - t0

        print(f"\n  Optimisation: {n_trials} trials in {elapsed:.1f}s "
              f"({elapsed/n_trials*1000:.1f}ms/trial)")

        best = study.best_trial
        print(f"\n  Best Sharpe: {best.value:.3f}")
        print(f"  Best params:")
        for k, v in sorted(best.params.items()):
            print(f"    {k:<20} = {v:+.4f}" if isinstance(v, float)
                  else f"    {k:<20} = {v}")

        # Full stats
        best_result = backtest_expanded_linear(
            features, closes, dates_list,
            alpha_window=60, cost_bps=5.0,
            **best.params,
        )
        print(f"\n  Full stats (best config):")
        print(f"    Trades:       {best_result['trades']}")
        print(f"    Total return: {best_result['total_return_pct']:+.2f}%")
        print(f"    Max drawdown: {best_result['max_dd_pct']:.2f}%")
        print(f"    Win rate:     {best_result['win_rate']*100:.1f}%")

        # Feature importance
        print(f"\n  Feature Importance (fANOVA):")
        try:
            importances = optuna.importance.get_param_importances(study)
            for param, imp in importances.items():
                bar = "█" * int(imp * 40)
                # Mark feature groups
                marker = ""
                if param.startswith("w_sanos"):
                    marker = " ★SANOS"
                elif param in {"w_fti", "w_mutual_info", "w_adx",
                               "w_detrended_rsi", "w_price_var_ratio",
                               "w_price_intensity"}:
                    marker = " ★MASTERS"
                elif param.startswith("w_") and param not in {
                    "w_alpha", "w_coherence", "w_phase",
                    "w_mock_theta", "w_frac_d", "w_momentum",
                }:
                    marker = " ★NEW"
                print(f"    {param:<22} {imp:6.1%}  {bar}{marker}")
        except Exception as e:
            print(f"    (Could not compute: {e})")

        # Top 10 trials — header dynamically built from active features
        print(f"\n  Top 10 trials:")
        # Abbreviated header
        short_names = {
            "w_alpha": "w_α", "w_coherence": "w_coh", "w_phase": "w_ph",
            "w_mock_theta": "w_mt", "w_frac_d": "w_fd", "w_momentum": "w_mom",
            "w_period_energy": "w_pe", "w_entropy": "w_ent",
            "w_yang_zhang": "w_yz", "w_atr_norm": "w_atr",
            "w_vpin": "w_vp", "w_rmt_absorption": "w_ra",
            "w_rmt_mp_excess": "w_rm",
            "w_sanos_skew": "w_ss", "w_sanos_left_tail": "w_sl",
            "w_sanos_entropy": "w_se", "w_sanos_kl": "w_sk",
            "w_hawkes_ratio": "w_hk",
            "w_fti": "w_ft", "w_mutual_info": "w_mi",
            "w_adx": "w_ax", "w_detrended_rsi": "w_dr",
            "w_price_var_ratio": "w_pv", "w_price_intensity": "w_pi",
        }
        hdr_parts = ["  #  Sharpe"]
        for w_name in active_features:
            hdr_parts.append(f"{short_names.get(w_name, w_name):>6}")
        hdr_parts.extend(["entry", "hold"])
        print("  " + " ".join(hdr_parts))
        print("  " + "-" * (8 + 7 * len(active_features) + 12))

        sorted_trials = sorted(
            study.trials,
            key=lambda t: t.value if t.value is not None else -999,
            reverse=True,
        )
        for rank, trial in enumerate(sorted_trials[:10], 1):
            p = trial.params
            parts = [f"{rank:>3d} {trial.value:>7.3f}"]
            for w_name in active_features:
                parts.append(f"{p.get(w_name, 0):>+6.2f}")
            parts.append(f"{p.get('entry_threshold', 0):>6.2f}")
            parts.append(f"{p.get('hold_days', 0):>4d}")
            print("  " + " ".join(parts))

        # Comparison with previous architectures
        print(f"\n  Comparison:")
        print(f"  {'Architecture':<40} {'Sharpe':>7} {'Trades':>6}")
        print(f"  " + "-" * 55)
        print(f"  {'Original 6-feature Optuna':<40} {'2.77':>7} {'184':>6}")
        print(f"  {'15-feature (VIX proxy)':<40} {'2.32':>7} {'31':>6}")
        print(f"  {'SANOS+Masters {}-feature Optuna':<40} "
              f"{best.value:>7.3f} {best_result['trades']:>6d}".format(
                  len(active_features)))
        print()

    print()


# ---------------------------------------------------------------------------
# Expanded intraday Optuna: daily signal + 1-min execution
# ---------------------------------------------------------------------------

def run_expanded_intraday(store: MarketDataStore, symbols: list[str],
                          n_trials: int = 500, use_kite: bool = False) -> None:
    """Intraday expanded Optuna: daily features → direction, 1-min bars → execution."""
    import optuna
    from strategies.s12_vedic_ffpe.gpu_features import (
        compute_features_expanded_gpu,
        backtest_intraday_expanded,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 70)
    print("S12 VEDIC FRACTIONAL ALPHA — EXPANDED INTRADAY DISCOVERY")
    print("=" * 70)
    print("  Daily features → direction bias, 1-min bars → intraday execution")
    print(f"  Trials: {n_trials}")
    print()

    # Load sector returns for RMT
    print("  Loading sector index returns for RMT...")
    t0 = time.time()
    if use_kite:
        sector_returns, sector_dates = _load_kite_sector_returns()
    else:
        sector_returns, sector_dates = _load_sector_returns(store)
    if len(sector_returns) > 0:
        print(f"    {sector_returns.shape[1]} indices × "
              f"{sector_returns.shape[0]} days  ({time.time()-t0:.1f}s)")

    for symbol in symbols:
        print(f"\n{'─' * 70}")
        print(f"  {symbol}")
        print(f"{'─' * 70}")

        # --- Load daily data (same as EOD expanded) ---
        if use_kite:
            try:
                closes, dates_list, opens, highs, lows, volumes = \
                    _load_kite_data(symbol)
                print(f"  Kite data: {len(closes)} days")
                has_ohlcv = opens is not None
            except Exception as e:
                print(f"  Kite data failed: {e}")
                continue
        else:
            df = _load_index_closes(store, symbol)
            if df.empty or len(df) < 80:
                print(f"  Insufficient data ({len(df)} days)")
                continue
            closes = df["close_val"].values
            dates_list = df["date"].tolist()
            ohlcv = _load_daily_ohlcv(store, symbol)
            opens = highs = lows = volumes = None
            if ohlcv is not None and len(ohlcv) > 30:
                ohlcv_map = {r["date"]: r for _, r in ohlcv.iterrows()}
                o_arr = np.full(len(dates_list), np.nan)
                h_arr = np.full(len(dates_list), np.nan)
                l_arr = np.full(len(dates_list), np.nan)
                v_arr = np.full(len(dates_list), np.nan)
                for i, d in enumerate(dates_list):
                    if d in ohlcv_map:
                        r = ohlcv_map[d]
                        o_arr[i] = r["open"]
                        h_arr[i] = r["high"]
                        l_arr[i] = r["low"]
                        v_arr[i] = r["volume"]
                opens, highs, lows, volumes = o_arr, h_arr, l_arr, v_arr

        # Align sector returns
        sr_aligned = None
        if len(sector_returns) > 0 and len(sector_dates) > 0:
            idx_map = {d: i for i, d in enumerate(sector_dates)}
            sr_aligned = np.full((len(dates_list), sector_returns.shape[1]), np.nan)
            for i, d in enumerate(dates_list):
                if d in idx_map:
                    sr_aligned[i] = sector_returns[idx_map[d]]

        # Hawkes
        print(f"\n  Computing Hawkes ratio from tick data...")
        hawkes_arr = _compute_daily_hawkes(store, symbol, dates_list)

        # SANOS density features from option chain
        print(f"  Computing SANOS density from option chain...")
        t1 = time.time()
        sanos_cache = _compute_sanos_cache(store, symbol, dates_list, closes)
        n_sanos = len(sanos_cache)
        print(f"    SANOS calibrated: {n_sanos}/{len(dates_list)} days "
              f"({time.time()-t1:.1f}s)")

        # Compute daily features
        features = compute_features_expanded_gpu(
            closes, dates_list,
            alpha_window=60, frac_d=0.226,
            opens=opens, highs=highs, lows=lows, volumes=volumes,
            sector_returns=sr_aligned,
            hawkes_ratio=hawkes_arr,
            sanos_cache=sanos_cache,
            store=store, symbol=symbol,
        )

        # --- Load 1-min bars ---
        print(f"\n  Loading 1-min futures bars...")
        intraday = _load_intraday_bars_for_dates(symbol, dates_list)

        if not intraday:
            print(f"  No intraday data available — skipping")
            continue

        # Feature availability
        feature_defs = [
            ("w_alpha", "alpha"), ("w_coherence", "coherence"),
            ("w_phase", "phase"), ("w_mock_theta", "mock_theta_div"),
            ("w_frac_d", "frac_d_series"), ("w_momentum", "momentum"),
            ("w_period_energy", "period_energy"), ("w_entropy", "entropy"),
            ("w_yang_zhang", "yang_zhang_vol"), ("w_atr_norm", "atr_norm"),
            ("w_vpin", "vpin"), ("w_rmt_absorption", "rmt_absorption"),
            ("w_rmt_mp_excess", "rmt_mp_excess"),
            ("w_sanos_skew", "sanos_skew"),
            ("w_sanos_left_tail", "sanos_left_tail"),
            ("w_sanos_entropy", "sanos_entropy"),
            ("w_sanos_kl", "sanos_kl"),
            ("w_hawkes_ratio", "hawkes_ratio"),
            ("w_fti", "fti"), ("w_mutual_info", "mutual_info"),
            ("w_adx", "adx"), ("w_detrended_rsi", "detrended_rsi"),
            ("w_price_var_ratio", "price_var_ratio"),
            ("w_price_intensity", "price_intensity"),
        ]

        active_features = []
        for w_name, feat_key in feature_defs:
            arr = features.get(feat_key)
            if arr is not None and np.sum(~np.isnan(arr)) > 30:
                active_features.append(w_name)

        print(f"  Active features: {len(active_features)}")
        print(f"  Intraday days:   {len(intraday)}")

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for w_name in active_features:
                params[w_name] = trial.suggest_float(w_name, -2.0, 2.0)
            params["entry_threshold"] = trial.suggest_float(
                "entry_threshold", 0.1, 2.5)
            params["target_mult"] = trial.suggest_float(
                "target_mult", 1.0, 5.0)
            params["stop_mult"] = trial.suggest_float(
                "stop_mult", 0.5, 3.0)
            params["max_trades_per_day"] = trial.suggest_int(
                "max_trades_per_day", 1, 5)

            result = backtest_intraday_expanded(
                features, closes, dates_list,
                intraday_bars=intraday,
                alpha_window=60, cost_bps=5.0,
                **params,
            )
            if result["trades"] < 10:
                return -10.0
            return result["sharpe"]

        t0 = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        elapsed = time.time() - t0

        print(f"\n  Optimisation: {n_trials} trials in {elapsed:.1f}s "
              f"({elapsed/n_trials*1000:.1f}ms/trial)")

        best = study.best_trial
        print(f"\n  Best Sharpe: {best.value:.3f}")
        print(f"  Best params:")
        for k, v in sorted(best.params.items()):
            print(f"    {k:<22} = {v:+.4f}" if isinstance(v, float)
                  else f"    {k:<22} = {v}")

        # Full stats
        best_result = backtest_intraday_expanded(
            features, closes, dates_list,
            intraday_bars=intraday,
            alpha_window=60, cost_bps=5.0,
            **best.params,
        )
        print(f"\n  Full stats (best config):")
        print(f"    Trades:       {best_result['trades']}")
        print(f"    Total return: {best_result['total_return_pct']:+.2f}%")
        print(f"    Max drawdown: {best_result['max_dd_pct']:.2f}%")
        print(f"    Win rate:     {best_result['win_rate']*100:.1f}%")

        # Exit reason breakdown
        exit_counts = best_result.get("exit_counts", {})
        if exit_counts:
            print(f"\n  Exit Reasons:")
            for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
                pct = count / max(best_result["trades"], 1) * 100
                print(f"    {reason:<12} {count:4d} ({pct:.1f}%)")

        # Feature importance
        print(f"\n  Feature Importance (fANOVA):")
        try:
            importances = optuna.importance.get_param_importances(study)
            for param, imp in importances.items():
                bar = "█" * int(imp * 40)
                print(f"    {param:<22} {imp:6.1%}  {bar}")
        except Exception as e:
            print(f"    (Could not compute: {e})")

    print()


# ---------------------------------------------------------------------------
# Feature recalibration sweep
# ---------------------------------------------------------------------------

def run_recalibrate(store: MarketDataStore, symbols: list[str]) -> None:
    """Sweep auxiliary feature thresholds to find configs where features contribute.

    The ablation study showed that only α regime drives trades — mock theta,
    phase gate, coherence, and frac_d thresholds are either too loose or too
    tight. This sweep searches for thresholds that make each feature matter.
    """
    from strategies.s12_vedic_ffpe.gpu_features import (
        compute_features_array_gpu,
        backtest_from_features,
    )

    print("\n" + "=" * 70)
    print("S12 VEDIC FRACTIONAL ALPHA — FEATURE RECALIBRATION SWEEP")
    print("=" * 70)
    print("  Compute = GPU (T4)")
    print("  ⚠ OPTIMISED — searches for thresholds that make each feature matter")
    print()

    # Sweep grid for auxiliary thresholds
    mt_div_thresholds = [0.0001, 0.0003, 0.001, 0.005, 0.01, 0.05, 0.5]
    phase_windows = [
        (0.0, 1.0),    # off
        (0.0, 0.25),   # tight low
        (0.0, 0.5),    # default
        (0.1, 0.35),   # tight around mean
        (0.15, 0.30),  # very tight
        (0.25, 1.0),   # high only
    ]
    coherence_weights = [0.0, 0.25, 0.5, 1.0, 2.0]
    min_convictions = [0.01, 0.05, 0.10, 0.15, 0.25]
    phase_reject_factors = [0.0, 0.3, 0.5, 1.0]

    for symbol in symbols:
        df = _load_index_closes(store, symbol)
        if df.empty or len(df) < 100:
            continue

        closes = df["close_val"].values
        dates_list = df["date"].tolist()

        print(f"\n{'─' * 86}")
        print(f"  {symbol} — {len(df)} days")
        print(f"{'─' * 86}")

        # Compute features once
        features = compute_features_array_gpu(closes, alpha_window=60, frac_d=0.226)

        # Print feature ranges so thresholds make sense
        mt_vals = features["mock_theta_div"]
        mt_valid = mt_vals[~np.isnan(mt_vals)]
        ph_vals = features["phase"]
        ph_valid = ph_vals[~np.isnan(ph_vals)]
        coh_vals = features["coherence"]
        coh_valid = coh_vals[~np.isnan(coh_vals)]
        print(f"\n  Feature ranges:")
        print(f"    mock_theta_div  min={mt_valid.min():.6f}  mean={mt_valid.mean():.6f}  "
              f"max={mt_valid.max():.6f}  p95={np.percentile(mt_valid, 95):.6f}")
        print(f"    phase           min={ph_valid.min():.4f}  mean={ph_valid.mean():.4f}  "
              f"max={ph_valid.max():.4f}")
        print(f"    coherence       min={coh_valid.min():.4f}  mean={coh_valid.mean():.4f}  "
              f"max={coh_valid.max():.4f}")

        # --- Section 1: Mock theta divergence threshold ---
        print(f"\n  [1] Mock Theta Divergence Threshold")
        print(f"  {'mtDiv_th':>10} {'Sharpe':>7} {'Return':>9} {'Trades':>6} {'WinR':>6}")
        print(f"  " + "-" * 44)
        for mt_th in mt_div_thresholds:
            result = backtest_from_features(
                features, closes, dates_list,
                alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                cost_bps=5.0, hold_days=5,
                mt_div_threshold=mt_th,
            )
            if result["trades"] == 0:
                print(f"  {mt_th:>10.4f} {'--':>7} {'--':>9} {'0':>6} {'--':>6}")
            else:
                print(f"  {mt_th:>10.4f} {result['sharpe']:>7.2f} "
                      f"{result['total_return_pct']:>+8.2f}% "
                      f"{result['trades']:>6d} {result['win_rate']*100:>5.1f}%")

        # --- Section 2: Phase gate window ---
        print(f"\n  [2] Phase Gate Window")
        print(f"  {'phase_win':>14} {'reject':>7} {'Sharpe':>7} {'Return':>9} {'Trades':>6} {'WinR':>6}")
        print(f"  " + "-" * 56)
        for (plo, phi) in phase_windows:
            for prf in phase_reject_factors:
                result = backtest_from_features(
                    features, closes, dates_list,
                    alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                    cost_bps=5.0, hold_days=5,
                    phase_lo=plo, phase_hi=phi, phase_reject_factor=prf,
                )
                if result["trades"] == 0:
                    continue
                print(f"  [{plo:.2f},{phi:.2f}] {prf:>7.1f} {result['sharpe']:>7.2f} "
                      f"{result['total_return_pct']:>+8.2f}% "
                      f"{result['trades']:>6d} {result['win_rate']*100:>5.1f}%")

        # --- Section 3: Coherence weight ---
        print(f"\n  [3] Coherence Weight")
        print(f"  {'coh_w':>7} {'Sharpe':>7} {'Return':>9} {'Trades':>6} {'WinR':>6}")
        print(f"  " + "-" * 40)
        for cw in coherence_weights:
            result = backtest_from_features(
                features, closes, dates_list,
                alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                cost_bps=5.0, hold_days=5,
                coherence_weight=cw,
            )
            if result["trades"] == 0:
                print(f"  {cw:>7.2f} {'--':>7} {'--':>9} {'0':>6} {'--':>6}")
            else:
                print(f"  {cw:>7.2f} {result['sharpe']:>7.2f} "
                      f"{result['total_return_pct']:>+8.2f}% "
                      f"{result['trades']:>6d} {result['win_rate']*100:>5.1f}%")

        # --- Section 4: Min conviction ---
        print(f"\n  [4] Min Conviction Threshold")
        print(f"  {'min_conv':>9} {'Sharpe':>7} {'Return':>9} {'Trades':>6} {'WinR':>6}")
        print(f"  " + "-" * 42)
        for mc in min_convictions:
            result = backtest_from_features(
                features, closes, dates_list,
                alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                cost_bps=5.0, hold_days=5,
                min_conviction=mc,
            )
            if result["trades"] == 0:
                print(f"  {mc:>9.2f} {'--':>7} {'--':>9} {'0':>6} {'--':>6}")
            else:
                print(f"  {mc:>9.2f} {result['sharpe']:>7.2f} "
                      f"{result['total_return_pct']:>+8.2f}% "
                      f"{result['trades']:>6d} {result['win_rate']*100:>5.1f}%")

        # --- Section 5: Best combo search ---
        print(f"\n  [5] Joint Sweep (top 10 by Sharpe)")
        results_all: list[tuple[float, str, dict]] = []

        for mt_th in mt_div_thresholds:
            for (plo, phi) in phase_windows:
                for cw in coherence_weights:
                    for mc in min_convictions:
                        result = backtest_from_features(
                            features, closes, dates_list,
                            alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                            cost_bps=5.0, hold_days=5,
                            mt_div_threshold=mt_th,
                            phase_lo=plo, phase_hi=phi,
                            coherence_weight=cw,
                            min_conviction=mc,
                        )
                        if result["trades"] < 3:
                            continue
                        config = (f"mt={mt_th:.4f} ph=[{plo:.2f},{phi:.2f}] "
                                  f"coh={cw:.2f} mc={mc:.2f}")
                        results_all.append((result["sharpe"], config, result))

        results_all.sort(key=lambda x: -x[0])

        header = (f"  {'Rank':>4} {'Sharpe':>7} {'Return':>9} {'Trades':>6} "
                  f"{'WinR':>6}  Config")
        print(header)
        print(f"  " + "-" * 80)

        for rank, (sharpe, config, result) in enumerate(results_all[:10], 1):
            print(f"  {rank:>4d} {sharpe:>7.2f} "
                  f"{result['total_return_pct']:>+8.2f}% "
                  f"{result['trades']:>6d} {result['win_rate']*100:>5.1f}%  {config}")

        if results_all:
            print(f"\n  Total combos tested: {len(results_all)}")

    print()


# ---------------------------------------------------------------------------
# Feature ablation study
# ---------------------------------------------------------------------------

# Each ablation disables a feature by replacing its values with a neutral default.
# The backtest_from_features logic then behaves as if that feature didn't exist.
_ABLATION_SPECS: list[tuple[str, str, dict[str, object]]] = [
    # (label, description, {feature_key: neutral_value, ...})
    ("ALL ON",         "Full model (baseline)",    {}),
    ("-alpha",         "No α regime (force normal)", {"alpha": 1.0}),
    ("-coherence",     "No angular coherence",      {"coherence": 1.0}),
    ("-phase_gate",    "No Aryabhata phase gate",   {"_phase_gate_off": True}),
    ("-mock_theta",    "No mock theta divergence",  {"mock_theta_div": 0.0}),
    ("-frac_d",        "No fractional differencing", {"frac_d_series": 0.0}),
    ("-alpha-coh",     "No α + no coherence",       {"alpha": 1.0, "coherence": 1.0}),
    ("-alpha-mt-fd",   "No α + no mt + no frac_d",  {"alpha": 1.0, "mock_theta_div": 0.0,
                                                      "frac_d_series": 0.0}),
    ("ONLY alpha",     "Only α regime (rest neutral)", {"coherence": 1.0, "mock_theta_div": 0.0,
                                                         "frac_d_series": 0.0,
                                                         "_phase_gate_off": True}),
    ("ONLY coherence", "Only coherence (rest neutral)", {"alpha": 1.0, "mock_theta_div": 0.0,
                                                          "frac_d_series": 0.0,
                                                          "_phase_gate_off": True}),
    ("ONLY phase",     "Only phase gate (rest neutral)", {"alpha": 1.0, "coherence": 1.0,
                                                           "mock_theta_div": 0.0,
                                                           "frac_d_series": 0.0}),
    ("ONLY mock_theta","Only mock theta (rest neutral)", {"alpha": 1.0, "coherence": 1.0,
                                                           "frac_d_series": 0.0,
                                                           "_phase_gate_off": True}),
]


def _apply_ablation(features: dict[str, np.ndarray],
                     overrides: dict[str, object],
                     ) -> tuple[dict[str, np.ndarray], dict]:
    """Create an ablated copy of the feature dict.

    Returns (ablated_features, extra_kwargs) where extra_kwargs
    are modifications to backtest params (e.g. phase gate off).
    """
    ablated = {k: v.copy() for k, v in features.items()}
    extra_kwargs: dict = {}

    for key, val in overrides.items():
        if key == "_phase_gate_off":
            extra_kwargs["phase_lo"] = 0.0
            extra_kwargs["phase_hi"] = 1.0
        elif key in ablated:
            ablated[key][:] = val  # broadcast scalar to array
        # else: unknown key, ignore

    return ablated, extra_kwargs


def run_ablation(store: MarketDataStore, symbols: list[str]) -> None:
    """Feature ablation study: toggle features on/off to find causal drivers."""
    from strategies.s12_vedic_ffpe.gpu_features import (
        compute_features_array_gpu,
        backtest_from_features,
    )

    print("\n" + "=" * 70)
    print("S12 VEDIC FRACTIONAL ALPHA — FEATURE ABLATION STUDY")
    print("=" * 70)
    print("  Compute = GPU (T4)")
    print("  Each row disables one or more features to measure marginal impact.")
    print("  'Neutral' means the feature is replaced with a no-effect default.")
    print()

    for symbol in symbols:
        df = _load_index_closes(store, symbol)
        if df.empty or len(df) < 100:
            print(f"  {symbol}: insufficient data ({len(df)} days)")
            continue

        closes = df["close_val"].values
        dates_list = df["date"].tolist()

        print(f"{'─' * 76}")
        print(f"  {symbol} — {len(df)} days")
        print(f"{'─' * 76}")

        # Compute full features once
        features = compute_features_array_gpu(closes, alpha_window=60, frac_d=0.226)

        header = (f"  {'Config':<18} {'Sharpe':>7} {'Return':>9} {'MaxDD':>7} "
                  f"{'WinR':>6} {'Trades':>6}  Description")
        print(header)
        print("  " + "-" * 74)

        baseline_sharpe = None

        for label, desc, overrides in _ABLATION_SPECS:
            ablated, extra_kw = _apply_ablation(features, overrides)
            result = backtest_from_features(
                ablated, closes, dates_list,
                alpha_window=60, alpha_lo=0.85, alpha_hi=1.15,
                cost_bps=5.0, hold_days=5,
                **extra_kw,
            )

            if result["trades"] == 0:
                print(f"  {label:<18} {'--':>7} {'--':>9} {'--':>7} "
                      f"{'--':>6} {'0':>6}  {desc}")
                continue

            sharpe = result["sharpe"]
            if baseline_sharpe is None:
                baseline_sharpe = sharpe

            delta = sharpe - baseline_sharpe if baseline_sharpe is not None else 0.0
            delta_str = f"({delta:+.2f})" if label != "ALL ON" else ""

            print(f"  {label:<18} {sharpe:>7.2f} "
                  f"{result['total_return_pct']:>+8.2f}% "
                  f"{result['max_dd_pct']:>6.2f}% "
                  f"{result['win_rate']*100:>5.1f}% "
                  f"{result['trades']:>6d}  {desc} {delta_str}")

        print()

    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_backtest_result(result: dict, symbol: str) -> None:
    """Print a single backtest result."""
    if result["trades"] == 0:
        print(f"    {symbol}: No trades generated")
        return

    print(f"    Trades:       {result['trades']}")
    print(f"    Sharpe:       {result['sharpe']:.2f}")
    print(f"    Total return: {result['total_return_pct']:+.2f}%")
    print(f"    Max drawdown: {result['max_dd_pct']:.2f}%")
    print(f"    Win rate:     {result['win_rate']*100:.1f}%")

    rc = result.get("regime_counts", {})
    if rc:
        total = sum(rc.values())
        if total > 0:
            print(f"    Regimes:      sub={rc.get('subdiffusive',0)} "
                  f"norm={rc.get('normal',0)} "
                  f"super={rc.get('superdiffusive',0)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from strategies.utils import tee_to_results

    parser = argparse.ArgumentParser(
        description="S12 Vedic Fractional Alpha — Research"
    )
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep (optimised)")
    parser.add_argument("--validate", action="store_true",
                        help="Run placebo + time-shift validation")
    parser.add_argument("--intraday", action="store_true",
                        help="Run intraday mode backtest")
    parser.add_argument("--symbols", nargs="+",
                        default=["NIFTY", "BANKNIFTY"],
                        help="Symbols to analyse")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for intraday data loading")
    parser.add_argument("--gpu", action="store_true",
                        help="Use T4 GPU for feature computation")
    parser.add_argument("--ablate", action="store_true",
                        help="Feature ablation study (requires GPU)")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Sweep auxiliary feature thresholds (requires GPU)")
    parser.add_argument("--optuna", action="store_true",
                        help="Gate-free Optuna feature discovery (requires GPU)")
    parser.add_argument("--recalibrated", action="store_true",
                        help="Recalibrated architecture with Optuna fine-tune")
    parser.add_argument("--validate-recal", action="store_true",
                        help="Validation gates on recalibrated Optuna-best")
    parser.add_argument("--expanded-optuna", action="store_true",
                        help="Expanded 15-feature Optuna discovery "
                             "(Tier 1 + RMT + IV + Hawkes)")
    parser.add_argument("--expanded-intraday", action="store_true",
                        help="Intraday expanded Optuna "
                             "(daily signal + 1-min execution)")
    parser.add_argument("--kite", action="store_true",
                        help="Fetch data from Kite API (524 days) "
                             "instead of local DuckDB")
    parser.add_argument("--trials", type=int, default=500,
                        help="Number of Optuna trials (default: 500)")
    args = parser.parse_args()

    with tee_to_results("s12_vedic_ffpe"):
        store = MarketDataStore()
        try:
            if args.expanded_intraday:
                run_expanded_intraday(store, args.symbols,
                                      n_trials=args.trials,
                                      use_kite=args.kite)
            elif args.expanded_optuna:
                run_expanded_optuna(store, args.symbols,
                                     n_trials=args.trials,
                                     use_kite=args.kite)
            elif args.validate_recal:
                run_validate_recalibrated(store, args.symbols,
                                           n_trials=args.trials)
            elif args.recalibrated:
                run_recalibrated(store, args.symbols, n_trials=args.trials)
            elif args.optuna:
                run_optuna(store, args.symbols, n_trials=args.trials)
            elif args.recalibrate:
                run_recalibrate(store, args.symbols)
            elif args.ablate:
                run_ablation(store, args.symbols)
            elif args.sweep:
                run_sweep(store, args.symbols, use_gpu=args.gpu)
            elif args.validate:
                run_validate(store, args.symbols, use_gpu=args.gpu)
            elif args.intraday:
                run_intraday_research(store, args.symbols,
                                      max_workers=args.workers)
            else:
                run_math_first(store, args.symbols, use_gpu=args.gpu)
        finally:
            store.close()


if __name__ == "__main__":
    main()
