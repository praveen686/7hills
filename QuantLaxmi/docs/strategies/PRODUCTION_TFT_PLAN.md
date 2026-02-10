# Production TFT/X-Trend Inference System — Feature Selection, HP Tuning, Checkpoints

## Context

The TFT/X-Trend model has ~292 mega features but **no checkpoint save/load**, **no feature selection** (beyond VSN soft weighting), **no HP tuning**, and **no production inference path**. Every run trains from scratch. The goal is to build institutional-grade infrastructure to: discover the best feature subset, tune hyperparameters via Optuna, save/load versioned checkpoints, and serve predictions in production via the BaseStrategy protocol.

---

## Architecture — TFT-Native Feature Selection Strategy

**Key insight**: TFT's VSN (Variable Selection Network) learns per-feature softmax weights during training. This IS learned feature selection — the model tells us what matters, including nonlinear and cross-feature interactions. External methods (MI, permutation importance) measure what simpler models think is important, which may disagree with what TFT actually uses. **VSN is the primary selector; external methods are cheap pre-filters and validation only.**

```
MegaFeatureBuilder (292 features)
         ↓
Pre-filter: coverage (<30% NaN) + correlation (|r|>0.95) → ~210 features
         ↓
Train TFT on ~210 features (walk-forward, VSN learns soft selection)
         ↓
Extract VSN weights per fold → stability analysis → ~60-80 stable features
         ↓
Retrain TFT on reduced set (faster, less noise, often better Sharpe)
         ↓
HPTuner (Optuna on reduced features — tunes architecture, NOT features)
         ↓
Final Training (best features + best HPs) → CheckpointManager.save()
         ↓
TFTInferencePipeline.from_checkpoint() → predict(date, store)
         ↓
TFTStrategy (BaseStrategy wrapper) → Signal objects → Orchestrator
```

### Why TFT-native beats external feature selection
| Method | What it measures | Problem |
|--------|-----------------|---------|
| MI / Permutation (XGBoost) | What a *different* model thinks matters | XGBoost ≠ TFT; misses attention/LSTM dynamics |
| Correlation filter | Redundancy | Doesn't know which of two correlated features the model prefers |
| **VSN weights (TFT-native)** | **What the actual production model attends to** | **Requires initial training first (solved by 2-pass approach)** |

**2-pass approach**: Train once on ~210 pre-filtered features → extract VSN importance → prune to stable 60-80 → retrain. The first pass is the "exploration" pass; the second pass is the "production" pass with focused features.

---

## New Files (7) — all in `models/ml/tft/production/`

### 1. `feature_selection.py` (~600 lines)

**FeatureSelectionConfig** dataclass:
- `min_coverage=0.3`, `correlation_threshold=0.95`, `vsn_weight_threshold=0.005`
- `final_max_features=80`, `stability_min_folds=0.5`

**FeatureSelector** class — 2-tier approach:

**Tier 1: Cheap pre-filters (no model needed, run before any training):**
1. **Coverage filter** — drop features with <30% non-NaN in training fold
2. **Correlation filter** — remove features with |corr| > 0.95 (keep higher-variance of each pair)
→ Reduces ~292 → ~210 features in seconds

**Tier 2: VSN-native selection (after training, the primary method):**
3. **VSN weight extraction** — read softmax weights from trained model's `weight_grn` per fold
4. **VSN stability analysis** — across folds: features with mean VSN weight > threshold AND selected in >= 50% of folds are "stable"
5. **Feature importance report** — ranked list with per-fold weights, stability score, group membership

**Optional validation (not for selection, for confidence):**
- MI correlation between VSN ranking and `mutual_info_regression` ranking (sanity check)
- If VSN and MI strongly disagree on a feature, flag for manual review

**Stability analysis**: `StabilityReport` with `stable_features`, `stability_scores`, `vsn_weight_per_fold`, `recommended_features`, `feature_group_breakdown`.

### 2. `checkpoint_manager.py` (~450 lines)

**CheckpointMetadata** frozen dataclass:
- `version`, `model_type`, `created_at`, `git_commit`
- `feature_names`, `n_features`, `n_assets`, `asset_names`
- `config` (full XTrendConfig as dict), `normalization` (means + stds arrays)
- `training_info` (fold ranges, epochs, per-fold metrics)
- `feature_selection` (selected features, stability scores)
- `optuna_best_params`, `sharpe_oos`, `total_return_oos`, `max_drawdown_oos`

**CheckpointManager** class:
- `save(model, metadata, feature_selector?, optuna_study?)` → saves to `checkpoints/{model_type}/v{version}_{timestamp}/`
  - `model.pt` (state_dict only), `metadata.json`, `feature_selector.pkl`, `optuna_study.db`
- `load(checkpoint_dir)` → returns `(state_dict, metadata)` — caller constructs model
- `load_latest(model_type)` → most recent checkpoint
- `list_checkpoints(model_type?)` → summary list sorted by date
- Auto-incrementing version per model_type

### 3. `hp_tuner.py` (~500 lines)

**TunerConfig**: `n_trials=40`, `timeout_seconds=14400` (4h), `n_folds_per_trial=3`, `seed=42`

**HPTuner** class — Optuna with MedianPruner:
- Search space:
  - `d_hidden`: [32, 48, 64, 96, 128]
  - `n_heads`: [2, 4, 8]
  - `lstm_layers`: [1, 2, 3]
  - `dropout`: [0.05, 0.3]
  - `seq_len`: [21, 42, 63]
  - `n_context`: [8, 16, 32]
  - `lr`: [3e-4, 3e-3] log-uniform
  - `batch_size`: [16, 32, 64]
  - `mle_weight`: [0.01, 0.2]
  - `loss_mode`: ["sharpe", "joint_mle"]
  - `max_position`: [0.15, 0.3]
  - `position_smooth`: [0.2, 0.5]
- Objective: mean OOS Sharpe across 3 walk-forward folds
- Abbreviated training: 15 pretrain + 25 finetune epochs per fold (vs 30+50 full)
- Reports per-fold Sharpe for pruning (prune after fold 1 if below median)
- Returns `TuningResult` with `best_params`, `best_sharpe`, full `optuna.Study`

### 4. `inference.py` (~550 lines)

**InferenceResult** dataclass: `positions`, `confidences`, `feature_importance`, `raw_output`, `metadata`

**TFTInferencePipeline** class:
- `from_checkpoint(checkpoint_dir)` — reconstruct model from metadata, load state_dict, extract norm stats
- `from_latest(model_type)` — load most recent checkpoint
- `predict(date, store, lookback_days=120)`:
  1. Build features via MegaFeatureBuilder for `[d - lookback, d]`
  2. Select only checkpoint's feature columns
  3. Apply saved normalization (means, stds from training)
  4. Construct target sequences + context sets (causal)
  5. Forward pass (no_grad)
  6. Position sizing (clip, vol-scale, smooth)
  7. Extract confidence from output distribution
  8. Return `InferenceResult`
- `predict_batch(dates, store)` — batch mode for backtesting

### 5. `training_pipeline.py` (~700 lines)

**TrainingPipelineConfig**: toggles for each phase, sub-configs, date range, symbols, epoch budgets

**TrainingPipeline** — 5-phase orchestrator (2-pass training):

| Phase | What | Time Est. |
|-------|------|-----------|
| 1. Pre-filter | Build mega features, coverage + correlation filter (292 → ~210) | ~3 min |
| 2. Exploration pass | Train TFT on ~210 features (walk-forward), extract VSN weights per fold | ~30 min |
| 3. Feature pruning | VSN stability analysis → select ~60-80 stable features | ~10 sec |
| 4. HP Tuning | Optuna 40 trials x 3 folds with reduced features (abbreviated epochs) | ~2-3 hrs |
| 5. Production pass | Full walk-forward with best features + best HPs → save checkpoint + report | ~45 min |

- Phase 2 uses abbreviated epochs (15+25) — goal is VSN weights, not best model
- Phase 3 is pure analysis: rank features by VSN weight stability, prune low-weight ones
- Phase 4 tunes architecture/training HPs on the reduced feature set
- Phase 5 does full training (30+50 epochs), saves LAST fold model as production checkpoint
- Returns `TrainingResult` with checkpoint_path, feature_report, tuning_result, OOS metrics

**Progress Monitoring (tqdm):**
- Outer bar: 5 phases with name + ETA
- Inner bars: per-epoch, per-trial, per-fold progress
- Live postfix: current loss, Sharpe, LR, features remaining
- Optuna TQDMCallback for trial progress

**CLI entry**: `python -m models.ml.tft.production.training_pipeline [--skip-tuning] [--skip-feature-selection] [--start ...] [--end ...]`

### 6. `strategy_adapter.py` (~350 lines)

**TFTStrategy(BaseStrategy)**:
- Lazy-loads TFTInferencePipeline on first `scan()` call
- `_scan_impl(d, store)`: runs `pipeline.predict(d, store)`, converts to `Signal` objects
- Conviction threshold filter (default 0.3): only emit signals when |position| > threshold
- Signal metadata: raw_position, confidence, model_type, checkpoint_version
- `create_strategy()` factory for StrategyRegistry auto-discovery
- strategy_id: `s_tft_x_trend_v{version}`

### 7. `__init__.py` (~50 lines)

Exports: `FeatureSelector`, `CheckpointManager`, `HPTuner`, `TFTInferencePipeline`, `TrainingPipeline`, `TFTStrategy`

---

## New Test File

### `tests/test_production_tft.py` (~500 lines, ~25 tests)

| Class | Tests |
|-------|-------|
| `TestFeatureSelector` | coverage filter, correlation filter, MI ranking, permutation importance, consensus merge, stability analysis, select_fold validity |
| `TestCheckpointManager` | save directory structure, load roundtrip state_dict, load roundtrip metadata, list sorted, load_latest, auto-version |
| `TestHPTuner` | build_config_from_trial, objective returns finite, tune completes with small budget |
| `TestInferencePipeline` | from_checkpoint loads, predict returns valid positions, normalization uses saved stats, confidence bounds |
| `TestTFTStrategy` | implements protocol, scan returns valid signals, lazy loading, conviction threshold |
| `TestTrainingPipeline` | phase1 runs, phase3 saves checkpoint, full pipeline on synthetic data |

All tests use synthetic data (random features/targets) — no DuckDB dependency.

---

## Modified Files (2, minimal)

### `backbone.py` (+30 lines)
Add to `XTrendBackbone`:
- `save_production_checkpoint(checkpoint_dir, metadata_extras)` — convenience wrapper
- `load_production_checkpoint(checkpoint_dir)` — convenience loader

### `models/rl/integration/__init__.py` (+3 lines)
Add production package exports.

---

## Verification

```bash
cd QuantLaxmi && source venv/bin/activate

# Unit tests (~25 new tests, all pass)
python -m pytest tests/test_production_tft.py -v

# Full suite (1222 + ~25 = ~1247 tests, 0 failures)
python -m pytest tests/ -v

# Run full training pipeline
python -m models.ml.tft.production.training_pipeline \
  --start 2024-01-01 --end 2026-02-06

# Verify checkpoint
ls -la checkpoints/x_trend/

# Run inference
python -c "
from models.ml.tft.production import TFTInferencePipeline
from data.store import MarketDataStore
from datetime import date
pipe = TFTInferencePipeline.from_latest('x_trend')
result = pipe.predict(date(2026, 2, 6), MarketDataStore())
print(result.positions, result.confidences)
"
```

### Success Criteria
- Feature selection reduces 292 -> 60-80 features with stability scores
- Optuna finds HP config with OOS Sharpe > baseline (default config)
- Checkpoint contains everything for inference (model.pt + metadata.json)
- `TFTInferencePipeline.from_checkpoint()` loads and predicts in <1s
- `TFTStrategy.scan()` returns valid Signal objects
- All ~1247 tests pass (0 failures)
- Zero stubs, zero TODOs
- Full pipeline runs end-to-end in <4 hours on T4 GPU
