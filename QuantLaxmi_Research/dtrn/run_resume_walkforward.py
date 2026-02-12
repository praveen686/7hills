#!/usr/bin/env python3
"""Resume DTRN walk-forward from Stage 1 checkpoints.

Loads saved stage1 weights, runs Stage 2 (fixed KL) + Stage 3 (fixed TC),
then backtests on OOS dates. For folds without stage1 checkpoints, trains
from scratch.
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
from dtrn.train import (
    prepare_training_data,
    train_stage1,
    train_stage2,
    train_stage3,
)
from dtrn.engine.backtest import run_backtest
from dtrn.model.dtrn import create_dtrn

RESULTS_DIR = Path("/home/ubuntu/Desktop/7hills/QuantLaxmi_Research/dtrn/results")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_fold(
    fold_idx: int,
    instrument: str,
    train_dates: list,
    test_dates: list,
    config: DTRNConfig,
    s1_epochs: int = 8,
    s2_epochs: int = 10,
    s3_epochs: int = 25,
):
    """Run one fold, resuming from stage1 checkpoint if available."""
    fold_dir = CHECKPOINT_DIR / f"fold{fold_idx}_{instrument}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = fold_dir / "dtrn_stage1.pt"

    print(f"  Preparing data ({len(train_dates)} days)...", flush=True)
    train_data = prepare_training_data(train_dates, instrument, config)
    print(f"  Loaded {len(train_data)} valid days", flush=True)

    if not train_data:
        return None

    n_features = train_data[0]["features"].shape[1]
    _, model = create_dtrn(config, n_features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params, {n_features} features, device={DEVICE}", flush=True)

    all_losses = {}

    # Try to resume from stage1 checkpoint
    if stage1_path.exists():
        print(f"  Loading stage1 checkpoint from {stage1_path}", flush=True)
        model.load_state_dict(torch.load(stage1_path, map_location="cpu"))
        all_losses["stage1"] = [0.0]  # placeholder
    else:
        print(f"  No stage1 checkpoint — training from scratch", flush=True)
        print(f"\n  {'='*50}")
        print(f"  Stage 1: Self-supervised Prediction")
        print(f"  {'='*50}")
        all_losses["stage1"] = train_stage1(model, train_data, config, s1_epochs, DEVICE)
        torch.save(model.state_dict(), stage1_path)
        print(f"  Stage 1 checkpoint saved", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stage 2: Regime Discovery (with fixed KL direction)
    print(f"\n  {'='*50}")
    print(f"  Stage 2: Regime Discovery (fixed KL)")
    print(f"  {'='*50}")
    all_losses["stage2"] = train_stage2(model, train_data, config, s2_epochs, DEVICE)
    torch.save(model.state_dict(), fold_dir / "dtrn_stage2.pt")
    print(f"  Stage 2 checkpoint saved", flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage 3: Trading Objective (with fixed TC scaling)
    print(f"\n  {'='*50}")
    print(f"  Stage 3: Trading Objective (fixed TC)")
    print(f"  {'='*50}")
    all_losses["stage3"] = train_stage3(model, train_data, config, s3_epochs, DEVICE)

    # Save final checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "n_features": n_features,
            "d_embed": config.d_embed,
            "d_hidden": config.d_hidden,
            "n_message_passes": config.n_message_passes,
            "d_temporal": config.d_temporal,
            "n_regimes": config.n_regimes,
            "pred_horizon": config.pred_horizon,
        },
        "losses": {k: [float(v) for v in vals] for k, vals in all_losses.items()},
        "instrument": instrument,
        "train_dates": [str(d) for d in train_dates],
        "fixes": ["masked_ewma", "kl_direction", "tc_scaling_1e2", "activity_weight_1.0",
                   "purge_gap_5", "nan_guard", "backtest_t1_fill", "vwap_entry", "kill_switch"],
    }, fold_dir / "dtrn_final.pt")
    print(f"  Final checkpoint saved", flush=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Backtest on OOS test dates
    print(f"\n  Running OOS backtest ({len(test_dates)} days)...", flush=True)
    t1 = time.time()
    bt = run_backtest(
        config=config,
        start_date=test_dates[0],
        end_date=test_dates[-1],
        instrument=instrument,
        model=model,
        device=DEVICE,
        verbose=False,
    )
    bt_time = time.time() - t1
    print(f"  Backtest done in {bt_time/60:.1f} min", flush=True)

    return {
        "bt": bt,
        "losses": all_losses,
        "bt_time": bt_time,
    }


def main():
    config = DTRNConfig()
    all_dates = list_available_dates()
    N = len(all_dates)

    train_window = 120
    purge_gap = 5
    n_folds = 4

    total_test = N - train_window - purge_gap
    test_window = total_test // n_folds

    all_results = {}

    for instrument in ["NIFTY", "BANKNIFTY"]:
        print(f"\n{'='*70}")
        print(f"  DTRN Resume Walk-Forward: {instrument}")
        print(f"  Dates: {N} ({all_dates[0]} to {all_dates[-1]})")
        print(f"  Fixes: KL direction, TC×1e2, activity=1.0, purge=5, t+1 fill")
        print(f"{'='*70}")

        fold_results = []
        all_oos_returns = []

        for fold_idx in range(n_folds):
            # Compute fold boundaries (same as original script)
            test_end_idx = N - fold_idx * test_window
            test_start_idx = test_end_idx - test_window
            train_end_idx = test_start_idx - purge_gap
            train_start_idx = train_end_idx - train_window

            if train_start_idx < 0:
                continue

            train_dates = all_dates[train_start_idx:train_end_idx]
            test_dates = all_dates[test_start_idx:test_end_idx]

            print(f"\n{'─'*60}")
            print(f"  Fold {fold_idx+1}/{n_folds} — {instrument}")
            print(f"  Train: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)}d)")
            print(f"  Test:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)}d)")
            print(f"{'─'*60}")

            t0 = time.time()
            result = run_fold(
                fold_idx=fold_idx + 1,
                instrument=instrument,
                train_dates=train_dates,
                test_dates=test_dates,
                config=config,
                s1_epochs=8,
                s2_epochs=10,
                s3_epochs=25,
            )
            fold_time = time.time() - t0

            if result is None:
                print(f"  SKIPPED — no valid data")
                continue

            bt = result["bt"]
            info = {
                "fold": fold_idx + 1,
                "instrument": instrument,
                "train": f"{train_dates[0]} to {train_dates[-1]}",
                "test": f"{test_dates[0]} to {test_dates[-1]}",
                "sharpe": bt.get("sharpe", 0),
                "sortino": bt.get("sortino", 0),
                "return_pct": bt.get("total_return_pct", 0),
                "max_dd_pct": bt.get("max_drawdown_pct", 0),
                "win_rate": bt.get("win_rate", 0),
                "trades": bt.get("total_trades", 0),
                "costs": bt.get("total_costs", 0),
                "regimes": bt.get("avg_regime_probs", []),
                "fold_time_min": fold_time / 60,
            }
            fold_results.append(info)

            if "daily_results" in bt:
                all_oos_returns.extend([d["return"] for d in bt["daily_results"]])

            print(f"\n  Fold {fold_idx+1} OOS Results ({instrument}):")
            print(f"    Sharpe:  {info['sharpe']:.2f}")
            print(f"    Return:  {info['return_pct']:+.2f}%")
            print(f"    Max DD:  {info['max_dd_pct']:.2f}%")
            print(f"    Win:     {info['win_rate']:.1%}")
            print(f"    Trades:  {info['trades']}")
            print(f"    Time:    {fold_time/60:.1f} min")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate
        print(f"\n{'='*70}")
        print(f"  AGGREGATE: {instrument}")
        print(f"{'='*70}")
        if all_oos_returns:
            arr = np.array(all_oos_returns)
            sh = np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252) if np.std(arr, ddof=1) > 1e-10 else 0
            ret = (1 + arr).prod() - 1
            cum = np.cumprod(1 + arr)
            pk = np.maximum.accumulate(cum)
            mdd = ((pk - cum) / np.maximum(pk, 1e-10)).max()
            print(f"    OOS Days:   {len(arr)}")
            print(f"    Sharpe:     {sh:.2f}")
            print(f"    Return:     {ret*100:+.2f}%")
            print(f"    Max DD:     {mdd*100:.2f}%")

        sharpes = [f["sharpe"] for f in fold_results]
        if sharpes:
            print(f"    Per-fold:   {[f'{s:.2f}' for s in sharpes]}")
            print(f"    Mean={np.mean(sharpes):.2f}  Std={np.std(sharpes):.2f}")

        all_results[instrument] = {"folds": fold_results, "n_oos": len(all_oos_returns)}

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"resume_walkforward_{ts}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
