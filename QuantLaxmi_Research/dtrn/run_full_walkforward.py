#!/usr/bin/env python3
"""Full-scale DTRN walk-forward training + backtest over the entire dataset.

Walk-forward protocol:
- Train on 120 days, purge 5 days, test on ~160 days (non-overlapping)
- 4 folds covering ~847 available dates
- Both NIFTY and BANKNIFTY
- All audit fixes applied (masked EWMA, KL direction, TC scaling, purge_days=5, etc.)
"""
import gc
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from dtrn.config import DTRNConfig
from dtrn.data.loader import list_available_dates
from dtrn.train import train_full_pipeline, prepare_training_data
from dtrn.engine.backtest import run_backtest
from dtrn.model.dtrn import create_dtrn

RESULTS_DIR = Path("/home/ubuntu/Desktop/7hills/QuantLaxmi_Research/dtrn/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_walkforward(
    instrument: str = "NIFTY",
    train_window: int = 120,
    purge_gap: int = 5,
    n_folds: int = 4,
    s1_epochs: int = 8,
    s2_epochs: int = 8,
    s3_epochs: int = 15,
):
    """Run walk-forward validation for one instrument."""
    config = DTRNConfig()
    all_dates = list_available_dates()
    N = len(all_dates)
    print(f"\n{'='*70}")
    print(f"  DTRN Walk-Forward: {instrument}")
    print(f"  Available dates: {N} ({all_dates[0]} to {all_dates[-1]})")
    print(f"  Train window: {train_window}d, Purge: {purge_gap}d, Folds: {n_folds}")
    print(f"  Epochs: {s1_epochs}+{s2_epochs}+{s3_epochs}")
    print(f"{'='*70}\n")

    # Compute test window to cover as much data as possible
    # Total test days = N - train_window - purge_gap
    # Split evenly across folds, stepping backwards from the end
    total_test = N - train_window - purge_gap
    test_window = total_test // n_folds

    fold_results = []
    all_oos_returns = []

    for fold_idx in range(n_folds):
        print(f"\n{'─'*60}")
        print(f"  Fold {fold_idx+1}/{n_folds}")
        print(f"{'─'*60}")

        # Compute fold boundaries
        # Train on: [train_start, train_end)
        # Purge:    [train_end, test_start)
        # Test on:  [test_start, test_end)
        test_end_idx = N - fold_idx * test_window
        test_start_idx = test_end_idx - test_window
        train_end_idx = test_start_idx - purge_gap
        train_start_idx = train_end_idx - train_window

        if train_start_idx < 0:
            print(f"  Skipping fold — not enough data")
            continue

        train_dates = all_dates[train_start_idx:train_end_idx]
        test_dates = all_dates[test_start_idx:test_end_idx]

        print(f"  Train: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
        print(f"  Test:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
        print(f"  Purge: {purge_gap} days")

        t0 = time.time()

        # Train
        checkpoint_dir = RESULTS_DIR / f"checkpoints/fold{fold_idx+1}_{instrument}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model, losses = train_full_pipeline(
            config=config,
            instrument=instrument,
            train_dates=train_dates,
            s1_epochs=s1_epochs,
            s2_epochs=s2_epochs,
            s3_epochs=s3_epochs,
            save_dir=checkpoint_dir,
        )

        train_time = time.time() - t0

        # Backtest on OOS test dates
        print(f"\n  Running OOS backtest ({len(test_dates)} days)...")
        t1 = time.time()

        bt_results = run_backtest(
            config=config,
            start_date=test_dates[0],
            end_date=test_dates[-1],
            instrument=instrument,
            model=model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False,
        )

        bt_time = time.time() - t1

        fold_info = {
            "fold": fold_idx + 1,
            "instrument": instrument,
            "train_start": str(train_dates[0]),
            "train_end": str(train_dates[-1]),
            "test_start": str(test_dates[0]),
            "test_end": str(test_dates[-1]),
            "train_days": len(train_dates),
            "test_days": bt_results.get("n_days", 0),
            "sharpe": bt_results.get("sharpe", 0),
            "sortino": bt_results.get("sortino", 0),
            "total_return_pct": bt_results.get("total_return_pct", 0),
            "max_drawdown_pct": bt_results.get("max_drawdown_pct", 0),
            "win_rate": bt_results.get("win_rate", 0),
            "total_trades": bt_results.get("total_trades", 0),
            "total_costs": bt_results.get("total_costs", 0),
            "avg_regime_probs": bt_results.get("avg_regime_probs", []),
            "regime_names": config.regime_names,
            "train_time_min": train_time / 60,
            "backtest_time_min": bt_time / 60,
            "stage1_final_loss": losses["stage1"][-1] if losses.get("stage1") else None,
            "stage2_final_loss": losses["stage2"][-1] if losses.get("stage2") else None,
            "stage3_final_loss": losses["stage3"][-1] if losses.get("stage3") else None,
        }

        fold_results.append(fold_info)

        # Collect OOS daily returns
        if "daily_results" in bt_results:
            oos_rets = [d["return"] for d in bt_results["daily_results"]]
            all_oos_returns.extend(oos_rets)

        print(f"\n  Fold {fold_idx+1} Results ({instrument}):")
        print(f"    OOS Sharpe:  {fold_info['sharpe']:.2f}")
        print(f"    OOS Return:  {fold_info['total_return_pct']:+.2f}%")
        print(f"    OOS Max DD:  {fold_info['max_drawdown_pct']:.2f}%")
        print(f"    Win Rate:    {fold_info['win_rate']:.1%}")
        print(f"    Trades:      {fold_info['total_trades']}")
        print(f"    Train time:  {train_time/60:.1f} min")
        print(f"    BT time:     {bt_time/60:.1f} min")

        # Free memory between folds
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate OOS results
    print(f"\n{'='*70}")
    print(f"  AGGREGATE OOS RESULTS: {instrument}")
    print(f"{'='*70}")

    if all_oos_returns:
        oos_arr = np.array(all_oos_returns)
        agg_sharpe = np.mean(oos_arr) / np.std(oos_arr, ddof=1) * np.sqrt(252) if np.std(oos_arr, ddof=1) > 1e-10 else 0
        agg_return = (1 + oos_arr).prod() - 1
        cum = np.cumprod(1 + oos_arr)
        peak = np.maximum.accumulate(cum)
        agg_max_dd = ((peak - cum) / peak).max()

        print(f"    Total OOS Days: {len(oos_arr)}")
        print(f"    OOS Sharpe:     {agg_sharpe:.2f}")
        print(f"    OOS Return:     {agg_return*100:+.2f}%")
        print(f"    OOS Max DD:     {agg_max_dd*100:.2f}%")

    sharpes = [f["sharpe"] for f in fold_results]
    if sharpes:
        print(f"\n    Per-fold Sharpes: {[f'{s:.2f}' for s in sharpes]}")
        print(f"    Mean: {np.mean(sharpes):.2f}  Std: {np.std(sharpes):.2f}  "
              f"Min: {np.min(sharpes):.2f}  Max: {np.max(sharpes):.2f}")

    return fold_results, all_oos_returns


def main():
    all_results = {}

    for instrument in ["NIFTY", "BANKNIFTY"]:
        fold_results, oos_returns = run_walkforward(
            instrument=instrument,
            train_window=120,
            purge_gap=5,
            n_folds=4,
            s1_epochs=8,
            s2_epochs=8,
            s3_epochs=15,
        )
        all_results[instrument] = {
            "folds": fold_results,
            "n_oos_days": len(oos_returns),
        }

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"walkforward_{ts}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL WALK-FORWARD SUMMARY")
    print(f"{'='*70}")
    for inst, res in all_results.items():
        sharpes = [f["sharpe"] for f in res["folds"]]
        returns = [f["total_return_pct"] for f in res["folds"]]
        print(f"  {inst:>12s}:  Mean Sharpe {np.mean(sharpes):.2f}  "
              f"Avg Return {np.mean(returns):+.2f}%  "
              f"Folds: {len(sharpes)}  OOS days: {res['n_oos_days']}")


if __name__ == "__main__":
    main()
