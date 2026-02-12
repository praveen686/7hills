#!/usr/bin/env python3
"""TFT 30-trial HP sweep with narrowed search + feature group selection.

Based on v6 findings:
- d_hidden=96, n_heads=4, lstm_layers=1 are consistently best
- n_context=8, batch=128 fixed (GPU/OOM constraints)

Adds Optuna-based feature GROUP selection:
- Each feature group is a boolean trial param (use_xxx = True/False)
- fANOVA then tells us which groups actually contribute to OOS Sharpe
- 13 groups total → manageable with 30 trials
"""
import sys, os, gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch

gc.collect()
torch.cuda.empty_cache()

# ── Feature group → prefix mapping ──────────────────────────────────
# Groups that appear in the pruned feature set (post Phase 3).
# Each maps to column name prefixes.
FEATURE_GROUPS = {
    "price":           ["px_"],
    "options":         ["opt_", "optx_"],
    "institutional":   ["inst_"],
    "breadth":         ["brd_"],
    "vix":             ["vix_"],
    "intraday":        ["intra_"],
    "futures":         ["fut_"],
    "fii":             ["fii_"],
    "nfo_1min":        ["nfo_"],
    "participant_vol": ["pvol_"],
    "divergence_flow": ["dff_"],
    "cross_asset":     ["ca_"],
    "news_sentiment":  ["ns_"],
}
# Groups NOT searched (always included): price is baseline
ALWAYS_ON = {"price"}  # keep price features as baseline


def _classify_feature(name: str) -> str | None:
    """Map a feature name to its group, or None if unmapped."""
    for group, prefixes in FEATURE_GROUPS.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                return group
    return None


def _filter_features_by_groups(
    features: np.ndarray,
    feature_names: list[str],
    enabled_groups: set[str],
) -> tuple[np.ndarray, list[str]]:
    """Filter feature array to only include enabled groups."""
    keep_idx = []
    keep_names = []
    for i, name in enumerate(feature_names):
        group = _classify_feature(name)
        if group is None or group in enabled_groups:
            keep_idx.append(i)
            keep_names.append(name)
    if not keep_idx:
        return features, feature_names  # safety: keep all
    return features[:, :, keep_idx], keep_names


# ── Monkey-patch HP tuner ───────────────────────────────────────────
import quantlaxmi.models.ml.tft.production.hp_tuner as hp_mod

_original_objective = hp_mod.HPTuner._objective


def _objective_with_groups(self, trial, features, targets, feature_names, n_assets):
    """Objective that adds feature group selection to the search space."""
    # Sample feature group booleans
    searchable_groups = sorted(FEATURE_GROUPS.keys() - ALWAYS_ON)
    enabled_groups = set(ALWAYS_ON)  # always keep baseline

    for group in searchable_groups:
        if trial.suggest_categorical(f"use_{group}", [True, False]):
            enabled_groups.add(group)

    # Filter features
    filtered_features, filtered_names = _filter_features_by_groups(
        features, feature_names, enabled_groups,
    )

    if len(filtered_names) < 3:
        return float("-inf")

    # Call original objective with filtered features
    return _original_objective(
        self, trial, filtered_features, targets, filtered_names, n_assets,
    )


def _narrowed_sample(self, trial):
    """Narrowed HP search: architecture params fixed from v6 results."""
    return {
        "d_hidden": 96,
        "n_heads": 4,
        "lstm_layers": 1,
        "dropout": trial.suggest_float("dropout", 0.1, 0.35),
        "seq_len": trial.suggest_categorical("seq_len", [21, 42, 63]),
        "n_context": 8,
        "lr": trial.suggest_float("lr", 6e-4, 6e-3, log=True),
        "batch_size": 128,
        "mle_weight": trial.suggest_float("mle_weight", 0.01, 0.2),
        "loss_mode": trial.suggest_categorical("loss_mode", ["sharpe", "joint_mle"]),
        "max_position": trial.suggest_float("max_position", 0.15, 0.3),
        "position_smooth": trial.suggest_float("position_smooth", 0.2, 0.5),
    }


hp_mod.HPTuner._sample_params = _narrowed_sample
hp_mod.HPTuner._objective = _objective_with_groups

# ── Pipeline config ─────────────────────────────────────────────────
from quantlaxmi.models.ml.tft.production.training_pipeline import (
    TrainingPipelineConfig,
    TrainingPipeline,
)

cfg = TrainingPipelineConfig(
    start_date="2024-01-01",
    end_date="2026-02-06",
    symbols=["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "BTCUSDT", "ETHUSDT"],
    skip_feature_selection=False,
    skip_tuning=False,
    tuning_n_trials=30,
    tuning_timeout_seconds=0,
    purge_gap=5,
    use_amp=True,
    use_cosine_lr=True,
    loss_mode="sharpe",
    d_hidden=96,
    batch_size=128,
    production_train_window=150,
    production_test_window=42,
    production_step_size=21,
    phase5_patience=10,
)

print("=" * 70)
print("  TFT 30-TRIAL SWEEP + FEATURE GROUP SELECTION")
print("  Fixed arch: d_hidden=96, n_heads=4, lstm=1, n_ctx=8, batch=128")
print("  HP search:  seq_len, dropout, lr, mle_weight, loss_mode, pos params")
print(f"  Feature groups: {len(FEATURE_GROUPS)} total, {len(ALWAYS_ON)} always-on,")
print(f"                  {len(FEATURE_GROUPS) - len(ALWAYS_ON)} searchable via Optuna")
print("  30 trials, purge_gap=5")
print("=" * 70)

pipeline = TrainingPipeline(cfg)
result = pipeline.run()

print()
print("=" * 70)
print("  TFT 30-TRIAL SWEEP COMPLETE")
print("=" * 70)
if hasattr(result, "sharpe_oos"):
    print(f"  OOS Sharpe: {result.sharpe_oos:.3f}")
if hasattr(result, "total_return_oos"):
    print(f"  OOS Return: {result.total_return_oos:.2%}")
print(result)
