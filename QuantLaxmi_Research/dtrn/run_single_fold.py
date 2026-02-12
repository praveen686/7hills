#!/usr/bin/env python3
"""Run a single DTRN fold (for parallel execution).

Modes:
  --backtest-only    Load dtrn_final.pt, skip all training, run OOS only
  --resume-stage3    Load dtrn_stage2.pt, train stage3 only, then OOS
  (default)          Full pipeline: stage1 → stage2 → stage3 → OOS
"""
import argparse
import gc
import json
import os
import sys
import time
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
from dtrn.engine.backtest_fast import run_backtest_fast as run_backtest
from dtrn.model.dtrn import create_dtrn

RESULTS_DIR = Path("/home/ubuntu/Desktop/7hills/QuantLaxmi_Research/dtrn/results")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_from_checkpoint(ckpt_path: Path, config: DTRNConfig, n_features: int = None):
    """Load model from a checkpoint file (either state_dict or full checkpoint).

    For full checkpoints (dtrn_final.pt), n_features is read from the saved config.
    For raw state_dicts (stage1/stage2), n_features must be provided.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        # Full checkpoint (dtrn_final.pt format)
        n_features = ckpt["config"]["n_features"]
        _, model = create_dtrn(config, n_features)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, n_features
    else:
        # Raw state_dict (stage1/stage2 format)
        if n_features is None:
            from dtrn.data.features import FeatureEngine
            n_features = FeatureEngine(config).n_features
        _, model = create_dtrn(config, n_features)
        model.load_state_dict(ckpt)
        return model, n_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold index (1-based)")
    parser.add_argument("--instrument", type=str, required=True, choices=["NIFTY", "BANKNIFTY"])
    parser.add_argument("--s1-epochs", type=int, default=8)
    parser.add_argument("--s2-epochs", type=int, default=10)
    parser.add_argument("--s3-epochs", type=int, default=25)
    parser.add_argument("--backtest-only", action="store_true",
                        help="Skip training, load dtrn_final.pt and run OOS backtest only")
    parser.add_argument("--resume-stage3", action="store_true",
                        help="Load dtrn_stage2.pt, train stage3 only, then OOS backtest")
    args = parser.parse_args()

    config = DTRNConfig()
    all_dates = list_available_dates()
    N = len(all_dates)

    train_window = 120
    purge_gap = 5
    n_folds = 4
    total_test = N - train_window - purge_gap
    test_window = total_test // n_folds

    # Compute fold boundaries (0-indexed internally)
    fold_idx = args.fold - 1
    test_end_idx = N - fold_idx * test_window
    test_start_idx = test_end_idx - test_window
    train_end_idx = test_start_idx - purge_gap
    train_start_idx = train_end_idx - train_window

    if train_start_idx < 0:
        print(f"SKIPPED — train_start_idx={train_start_idx} < 0", flush=True)
        return

    train_dates = all_dates[train_start_idx:train_end_idx]
    test_dates = all_dates[test_start_idx:test_end_idx]

    fold_dir = CHECKPOINT_DIR / f"fold{args.fold}_{args.instrument}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    final_path = fold_dir / "dtrn_final.pt"
    stage2_path = fold_dir / "dtrn_stage2.pt"
    stage1_path = fold_dir / "dtrn_stage1.pt"

    mode = "full"
    if args.backtest_only:
        mode = "backtest-only"
    elif args.resume_stage3:
        mode = "resume-stage3"

    print(f"{'='*60}", flush=True)
    print(f"  Fold {args.fold}/4 — {args.instrument}  [mode: {mode}]", flush=True)
    print(f"  Train: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)}d)", flush=True)
    print(f"  Test:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)}d)", flush=True)
    print(f"  Fixes: KL direction, TC×1e2, activity=1.0, purge=5, t+1 fill, entry_price persist", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    all_losses = {}

    # ── Mode: backtest-only ──
    if mode == "backtest-only":
        if not final_path.exists():
            print(f"  ERROR: --backtest-only but {final_path} not found!", flush=True)
            return
        print(f"\n  Loading final checkpoint: {final_path.name}", flush=True)
        model, n_features = load_model_from_checkpoint(final_path, config)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {n_params:,} params, {n_features} features, device={DEVICE}", flush=True)

    # ── Mode: resume-stage3 ──
    elif mode == "resume-stage3":
        if not stage2_path.exists():
            print(f"  ERROR: --resume-stage3 but {stage2_path} not found!", flush=True)
            return
        print(f"\n  Loading stage2 checkpoint: {stage2_path.name}", flush=True)
        model, n_features = load_model_from_checkpoint(stage2_path, config)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {n_params:,} params, {n_features} features, device={DEVICE}", flush=True)

        # Prepare training data for stage3
        print(f"\n  Preparing data ({len(train_dates)} days)...", flush=True)
        train_data = prepare_training_data(train_dates, args.instrument, config)
        print(f"  Loaded {len(train_data)} valid days", flush=True)
        if not train_data:
            print("  SKIPPED — no valid data", flush=True)
            return

        # Train stage3
        print(f"\n  {'='*50}", flush=True)
        print(f"  Stage 3: Trading Objective (fixed TC)", flush=True)
        print(f"  {'='*50}", flush=True)
        all_losses["stage3"] = train_stage3(model, train_data, config, args.s3_epochs, DEVICE)

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
            "instrument": args.instrument,
            "train_dates": [str(d) for d in train_dates],
            "fixes": ["masked_ewma", "kl_direction", "tc_scaling_1e2", "activity_weight_1.0",
                       "purge_gap_5", "nan_guard", "backtest_t1_fill", "vwap_entry",
                       "kill_switch", "entry_price_persist"],
        }, final_path)
        print(f"  Final checkpoint saved", flush=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Mode: full pipeline ──
    else:
        # Prepare data
        print(f"\n  Preparing data ({len(train_dates)} days)...", flush=True)
        train_data = prepare_training_data(train_dates, args.instrument, config)
        print(f"  Loaded {len(train_data)} valid days", flush=True)

        if not train_data:
            print("  SKIPPED — no valid data", flush=True)
            return

        n_features = train_data[0]["features"].shape[1]
        _, model = create_dtrn(config, n_features)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {n_params:,} params, {n_features} features, device={DEVICE}", flush=True)

        # Stage 1
        if stage1_path.exists():
            print(f"\n  Loading stage1 checkpoint from {stage1_path.name}", flush=True)
            model.load_state_dict(torch.load(stage1_path, map_location="cpu", weights_only=False))
            all_losses["stage1"] = [0.0]
        else:
            print(f"\n  {'='*50}", flush=True)
            print(f"  Stage 1: Self-supervised Prediction", flush=True)
            print(f"  {'='*50}", flush=True)
            all_losses["stage1"] = train_stage1(model, train_data, config, args.s1_epochs, DEVICE)
            torch.save(model.state_dict(), stage1_path)
            print(f"  Stage 1 checkpoint saved", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Stage 2
        if stage2_path.exists() and not stage1_path.exists():
            # Only skip stage2 if we have stage2 but no stage1 (shouldn't happen, but safe)
            print(f"\n  Loading stage2 checkpoint from {stage2_path.name}", flush=True)
            model.load_state_dict(torch.load(stage2_path, map_location="cpu", weights_only=False))
            all_losses["stage2"] = [0.0]
        else:
            print(f"\n  {'='*50}", flush=True)
            print(f"  Stage 2: Regime Discovery (fixed KL)", flush=True)
            print(f"  {'='*50}", flush=True)
            all_losses["stage2"] = train_stage2(model, train_data, config, args.s2_epochs, DEVICE)
            torch.save(model.state_dict(), stage2_path)
            print(f"  Stage 2 checkpoint saved", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Stage 3
        print(f"\n  {'='*50}", flush=True)
        print(f"  Stage 3: Trading Objective (fixed TC)", flush=True)
        print(f"  {'='*50}", flush=True)
        all_losses["stage3"] = train_stage3(model, train_data, config, args.s3_epochs, DEVICE)

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
            "instrument": args.instrument,
            "train_dates": [str(d) for d in train_dates],
            "fixes": ["masked_ewma", "kl_direction", "tc_scaling_1e2", "activity_weight_1.0",
                       "purge_gap_5", "nan_guard", "backtest_t1_fill", "vwap_entry",
                       "kill_switch", "entry_price_persist"],
        }, final_path)
        print(f"  Final checkpoint saved", flush=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── OOS Backtest ──
    print(f"\n  Running OOS backtest ({len(test_dates)} days)...", flush=True)
    bt_t0 = time.time()
    bt = run_backtest(
        config=config,
        start_date=test_dates[0],
        end_date=test_dates[-1],
        instrument=args.instrument,
        model=model,
        device=DEVICE,
        verbose=False,
    )
    bt_time = time.time() - bt_t0
    print(f"  Backtest done in {bt_time/60:.1f} min", flush=True)

    fold_time = time.time() - t0

    # Save results
    info = {
        "fold": args.fold,
        "instrument": args.instrument,
        "mode": mode,
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
        "backtest_time_min": bt_time / 60,
        "daily_returns": [d["return"] for d in bt.get("daily_results", [])],
        "losses": {k: [float(v) for v in vals] for k, vals in all_losses.items()},
    }

    out = RESULTS_DIR / f"fold{args.fold}_{args.instrument}_result.json"
    with open(out, "w") as f:
        json.dump(info, f, indent=2, default=str)

    print(f"\n  {'='*50}", flush=True)
    print(f"  RESULTS: Fold {args.fold} {args.instrument}", flush=True)
    print(f"  {'='*50}", flush=True)
    print(f"    Mode:    {mode}", flush=True)
    print(f"    Sharpe:  {info['sharpe']:.2f}", flush=True)
    print(f"    Sortino: {info['sortino']:.2f}", flush=True)
    print(f"    Return:  {info['return_pct']:+.2f}%", flush=True)
    print(f"    Max DD:  {info['max_dd_pct']:.2f}%", flush=True)
    print(f"    Win:     {info['win_rate']:.1%}", flush=True)
    print(f"    Trades:  {info['trades']}", flush=True)
    print(f"    BT time: {bt_time/60:.1f} min", flush=True)
    print(f"    Total:   {fold_time/60:.1f} min", flush=True)
    print(f"    Saved:   {out}", flush=True)


if __name__ == "__main__":
    main()
