#!/usr/bin/env python3
"""Run TFT Phase 5 (production walk-forward) only.

Loads cached state from Phases 1-4 disk checkpoints, reconstructs
Phase 3 pruning in-memory, and jumps directly to Phase 5.
"""
import sys, os, gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch

gc.collect()
torch.cuda.empty_cache()

from pathlib import Path
from quantlaxmi.models.ml.tft.production.training_pipeline import (
    TrainingPipelineConfig,
    TrainingPipeline,
)
from quantlaxmi.models.ml.tft.production.feature_selection import FeatureSelector

CHECKPOINT_DIR = Path("checkpoints")

# ── Load Phase 1 ─────────────────────────────────────────────────────
p1 = torch.load(CHECKPOINT_DIR / "phase1_features.pt", weights_only=False)
features_raw = p1["features_raw"]
feature_names_raw = p1["feature_names_raw"]
features_prefiltered = p1["features"]
feature_names_prefiltered = p1["feature_names"]
dates = p1["dates"]
targets = p1["targets"]
print(f"Phase 1: {len(feature_names_raw)} raw → {len(feature_names_prefiltered)} pre-filtered features, {len(dates)} days")

# ── Load Phase 2 ─────────────────────────────────────────────────────
selector = FeatureSelector.load(CHECKPOINT_DIR / "phase2_selector.pkl")
print(f"Phase 2: {selector.n_folds} folds loaded from disk")

# ── Run Phase 3 pruning ──────────────────────────────────────────────
report = selector.stability_report(feature_names_prefiltered)
recommended = report.recommended_features
if not recommended:
    ranked = sorted(report.mean_vsn_weights.items(), key=lambda x: -x[1])
    max_k = selector.config.final_max_features
    recommended = [name for name, _ in ranked[:max_k]]
    print(f"Phase 3: fallback top-{len(recommended)} by mean VSN weight")
else:
    print(f"Phase 3: {len(recommended)} stable features selected")

kept_idx = [feature_names_prefiltered.index(n) for n in recommended if n in feature_names_prefiltered]
kept_names = [feature_names_prefiltered[i] for i in kept_idx]
features_final = features_prefiltered[:, :, kept_idx]
print(f"  Features pruned: {len(feature_names_prefiltered)} → {len(kept_names)}")
for i, f in enumerate(kept_names):
    print(f"    {i+1}. {f}")

# ── Load Phase 4 ─────────────────────────────────────────────────────
p4 = torch.load(CHECKPOINT_DIR / "phase4_best_params.pt", weights_only=False)
best_params = p4["best_params"]
print(f"Phase 4: best_params loaded — seq_len={best_params.get('seq_len')}, d_hidden={best_params.get('d_hidden')}")

# ── Build pipeline config ─────────────────────────────────────────────
cfg = TrainingPipelineConfig(
    start_date="2024-01-01",
    end_date="2026-02-06",
    symbols=["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "BTCUSDT", "ETHUSDT"],
    skip_feature_selection=True,   # We already did it above
    skip_tuning=True,              # We already have best_params
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

# ── Create pipeline and inject state ─────────────────────────────────
pipeline = TrainingPipeline(cfg)

# We need to run the pipeline but with pre-injected state.
# Override the phase methods to skip 1-4 and inject state into Phase 5.
from quantlaxmi.models.ml.tft.production.training_pipeline import TrainingResult
import time, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

print()
print("=" * 70)
print("  TFT PHASE 5 ONLY — PRODUCTION WALK-FORWARD")
print(f"  {len(kept_names)} features, seq_len={best_params.get('seq_len')}, purge_gap=5")
print("=" * 70)

# Directly call _phase5_production with the reconstructed state
result = TrainingResult()
result.n_features_initial = len(feature_names_raw)
result.n_features_prefiltered = len(feature_names_prefiltered)
result.n_features_final = len(kept_names)

state = {
    "config": cfg,
    "result": result,
    "features": features_final,
    "features_raw": features_raw,
    "targets": targets,
    "feature_names": kept_names,
    "feature_names_raw": feature_names_raw,
    "dates": dates,
    "best_params": best_params,
    "selector": selector,
}

# Free memory before Phase 5
del p1, p4, features_raw, features_prefiltered, feature_names_raw, feature_names_prefiltered
gc.collect()
torch.cuda.empty_cache()

t0 = time.time()
pipeline._phase5_production(state)
elapsed = time.time() - t0

print()
print("=" * 70)
print("  TFT PHASE 5 COMPLETE")
print("=" * 70)
print(f"  Features: {result.n_features_initial} → {result.n_features_final}")
print(f"  OOS Sharpe (gross): {result.sharpe_oos:.3f}")
print(f"  OOS Sharpe (net):   {result.sharpe_oos_net:.3f}  ({cfg.oos_cost_bps:.0f} bps one-way)")
print(f"  OOS Return (gross): {result.total_return_oos:.4f}")
print(f"  OOS Return (net):   {result.total_return_oos_net:.4f}")
print(f"  Max DD (gross): {result.max_drawdown_oos:.4f}")
print(f"  Max DD (net):   {result.max_drawdown_oos_net:.4f}")
print(f"  Checkpoint: {result.checkpoint_path}")
print(f"  Elapsed: {elapsed:.0f}s")
