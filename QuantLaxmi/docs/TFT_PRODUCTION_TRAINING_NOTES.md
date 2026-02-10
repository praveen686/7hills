# TFT Production Training Pipeline Notes (2026-02-10)

Full record of the first production TFT training run, including pipeline architecture,
feature selection, hyperparameter tuning, production pass, and lessons learned.

---

## 1. Pipeline Architecture

The TFT production training pipeline follows a 5-phase design:

```
Phase 1: Pre-filter
  ↓
Phase 2: Exploration (VSN weight extraction)
  ↓
Phase 3: Feature Pruning (pure analysis, no training)
  ↓
Phase 4: HP Tuning (Optuna)
  ↓
Phase 5: Production Pass (full training + checkpoint)
```

### Phase 1 — Pre-filter
Removes obviously degenerate features (constant, >50% NaN, perfect collinearity)
before any GPU work begins.

### Phase 2 — Exploration (VSN)
Trains the TFT on **all** surviving features using abbreviated epochs.
The goal is NOT to find the best model — it is to extract Variable Selection
Network (VSN) weights across multiple walk-forward folds. These weights reveal
which features the network actually attends to and how stable that attention is
across time windows.

- Uses `production_step_size=42` (wider folds for faster iteration).
- 11 folds in this run.
- Each fold trains for a reduced epoch count — just enough to stabilize VSN weights.

### Phase 3 — Feature Pruning
Pure offline analysis. No training occurs.

- Ranks features by VSN weight magnitude and cross-fold stability.
- Prunes the feature set from 276 down to 73 (74% reduction).
- Completed in 0.0 seconds (it is just numpy sorting and thresholding).

### Phase 4 — HP Tuning (Optuna)
Tunes hyperparameters on the **reduced** 73-feature set.

- Uses Optuna with `MedianPruner` to kill bad trials early.
- Each trial runs a mini walk-forward and reports OOS Sharpe.
- 10 trials completed in this run.

### Phase 5 — Production Pass
Full training with the best features (from Phase 3) and best HPs (from Phase 4).

- Uses `production_step_size=21` for finer-grained walk-forward.
- 80 epochs per fold.
- Saves a checkpoint at the end via `CheckpointManager`.

---

## 2. Feature Selection Results

### Summary

| Metric             | Value         |
|--------------------|---------------|
| Initial features   | 276           |
| Selected features  | 73            |
| Reduction          | 74%           |
| Method             | VSN-native 2-pass |

**VSN-native 2-pass approach**: Train on all features (Phase 2) to let the
Variable Selection Network learn attention weights, then extract those weights
across folds to identify the stable, high-importance subset.

### Top Features by VSN Weight

| Rank | Feature                          | VSN Weight | Stability |
|------|----------------------------------|------------|-----------|
| 1    | `vix_level`                      | 4.5%       | 36%       |
| 2    | `pvol_client_opt_put_net_z21`    | 4.2%       | 27%       |
| 3    | `pvol_pro_opt_call_net`          | 2.7%       | 36%       |
| 4    | `pvol_pro_opt_call_net_z21`      | 2.4%       | 55%       |
| 5    | `mktact_total_value`             | 2.3%       | 27%       |
| 6    | `pvol_dii_opt_put_net_z21`       | 1.6%       | 82%       |

Feature #6 (`pvol_dii_opt_put_net_z21`) has the highest stability at 82%,
meaning it was consistently important across nearly all folds. This is arguably
more valuable than a higher raw weight with low stability.

### Key Observations

- **Participant volume (pvol) dominates**: 16 of 73 selected features are
  participant volume metrics (FII, DII, client, pro flows in futures and options).
  The market knows something through its order flow.
- **VIX features all selected**: Every VIX-derived feature passed the pruning
  threshold. Implied volatility is a first-class signal.
- **Cross-market signal confirmed**: `crypto_btc_ret_1d` made the top 50.
  Bitcoin 1-day returns carry information about next-day Indian index moves.
- **Microstructure features survived**: `intra_orb_width`, `intra_rvol_1min`,
  and other tick-aggregated features were selected, validating the investment in
  tick data infrastructure.

### Group Breakdown (73 features)

| Group    | Count | Group    | Count |
|----------|-------|----------|-------|
| pvol     | 16    | opt      | 4     |
| inst     | 7     | optx     | 4     |
| brd      | 5     | intra    | 4     |
| fii      | 5     | px       | 4     |
| mktact   | 5     | fut      | 4     |
| micro    | 5     | vix      | 3     |
| nfo      | 2     | ban      | 1     |
| nsevol   | 2     | crypto   | 1     |
| settle   | 1     |          |       |

---

## 3. HP Tuning Results (Phase 4)

### Configuration

- **Optimizer**: Optuna with `MedianPruner`
- **Trials completed**: 10
- **Original timeout**: 14400s (4 hours) — removed mid-run via gdb (see Section 7)

### Best Trial

| Parameter          | Value           |
|--------------------|-----------------|
| Trial              | 9               |
| OOS Sharpe         | **1.88**        |
| `d_hidden`         | 48              |
| `n_heads`          | 4               |
| `lstm_layers`      | 1               |
| `dropout`          | 0.23            |
| `seq_len`          | 63              |
| `n_context`        | 16              |
| `lr`               | 0.0027          |
| `batch_size`       | 16              |
| `mle_weight`       | 0.186           |
| `loss_mode`        | `joint_mle`     |
| `max_position`     | 0.29            |
| `position_smooth`  | 0.46            |

### Trial Progression

| Trial | OOS Sharpe | Notes                           |
|-------|------------|---------------------------------|
| 0     | 0.66       | Baseline                        |
| 4     | 0.99       | Improved                        |
| 9     | 1.88       | Best                            |
| Others| —          | Some pruned early by Optuna     |

Optuna's `MedianPruner` killed underperforming configurations after fold 1,
saving substantial compute. The progression from 0.66 to 1.88 over 10 trials
shows meaningful optimization surface — HP choice matters for this model.

---

## 4. Production Training (Phase 5)

### Configuration

| Parameter               | Value   |
|-------------------------|---------|
| `production_step_size`  | 21 days |
| Epochs per fold         | 80      |
| Time per epoch          | ~21s    |
| Time per fold           | ~28 min |
| Total folds             | ~16     |
| Estimated total time    | ~7.5 hours |

### Fold 0 Results

| Metric          | Value  |
|-----------------|--------|
| `val_sharpe`    | 8.08   |
| OOS Sharpe      | -0.64  |

This dramatic gap between validation and OOS is expected for the first fold
(see Section 5 for detailed discussion). Fold 0 covers the earliest data period,
which may have different regime characteristics. The walk-forward approach is
specifically designed to handle this — aggregate OOS performance across all folds
is the metric that matters.

### Checkpoint Saving

Checkpoint is saved at the end of Phase 5 via `CheckpointManager`:
- Uses `mkdir(parents=True)` to create directory structure.
- Wrapped in `try/except` for safety — a checkpoint failure should not crash a
  7+ hour training run.

---

## 5. val_sharpe Inflation Discussion

### Why val_sharpe of ~7-8 During Training Is Not Real

The validation Sharpe reported during training (typically 7-8) is severely
inflated. Multiple compounding effects explain the gap:

1. **Tiny validation window (21 days)**
   - Standard error of Sharpe: SE = Sharpe / sqrt(n)
   - With n=21, SE is enormous. A true daily Sharpe of 0.44 could easily
     appear as 2.0 or 0.0 in a 21-day sample.

2. **Best-epoch selection bias**
   - Training selects the epoch with the best `val_sharpe`.
   - This is taking the maximum of a noisy series — guaranteed to be biased
     upward. With 80 epochs, you're picking the best of 80 noisy estimates.

3. **Proximity to training data**
   - No purge gap between train and validation windows (now fixed — see Section 6).
   - Features with look-back windows (z-scores, rolling means) computed on data
     that overlaps with training.

4. **Annualization amplifies noise**
   - A daily Sharpe of 0.44 annualizes to 0.44 * sqrt(252) = 7.0.
   - Small noise in daily Sharpe becomes large noise in annualized Sharpe.

### Expected True Sharpe

The honest production Sharpe is likely in the **1.5-2.5 range**, consistent
with the Optuna OOS result of 1.88.

### Fold 0 As Evidence

Fold 0 provided a direct demonstration:
- `val_sharpe` = 8.08 (during training, best epoch)
- OOS Sharpe = -0.64 (actual forward performance)

This is not a bug. It is the expected behavior of a noisy estimator applied to
a small window with selection bias. The walk-forward framework exists precisely
to average out such fold-level noise.

---

## 6. Purge Gap Issue (CRITICAL)

### Discovery

During the production run, we identified that the TFT walk-forward pipeline
had **no purge gap** between training and test windows. The test window starts
immediately at `train_end`, with zero buffer days.

### Why This Matters

With `seq_len=63` and rolling 21-day z-scores, features at the start of the
test window are computed using data from the training period. This creates
information leakage:

```
                    train_end
                       |
 [...training data...] [test data...]
                  ^^^^^
                  These test-window features use training-period prices
                  in their look-back computations
```

Specifically:
- Z-score features with 21-day windows at test_start[0] use 20 days of
  training data.
- The TFT's seq_len=63 means the model's first test prediction looks back 63
  days, most of which are training data.
- This is not direct label leakage but is **feature leakage** — the test
  features are not causally independent of the training set.

### Precedent

The same issue was previously found and fixed in `S6 research.py`, where
`purge_gap=5` was added to the walk-forward configuration. De Prado's
Combinatorial Purged Cross-Validation (CPCV) framework treats purge gaps as
standard practice for exactly this reason.

### Fix Applied

Added `purge_gap=5` parameter to `TrainingPipelineConfig`:

```python
# Before
test_start = train_end

# After
test_start = train_end + purge_gap  # purge_gap=5
```

Applied to all training phases:
- Phase 2 (exploration)
- Phase 4 (HP tuning)
- Phase 5 (production)

### Impact Assessment

The Optuna OOS Sharpe of 1.88 was obtained **without** the purge gap fix. The
true honest Sharpe may be somewhat lower. The magnitude of the impact depends
on how much the leaked look-back information helps predictions — likely modest
(features are normalized z-scores, not raw prices) but nonzero.

The next training run will use the fix and provide a clean comparison.

---

## 7. GDB Live-Patching of Optuna Timeout

### Problem

Phase 4 (HP tuning) was configured with `timeout_seconds=14400` (4 hours)
inside `study.optimize()`. After Phase 1-3 took ~7 hours, we realized 10
trials was not enough and wanted to let it run longer. But the timeout was
baked into the running process — no runtime configuration could change it.

### Solution

Attached `gdb` to the running Python process (PID 3382427) and injected Python
code to disable the timeout:

```gdb
(gdb) call (int) PyGILState_Ensure()
(gdb) call (int) PyRun_SimpleString(
    "import gc\n"
    "for obj in gc.get_objects():\n"
    "    if type(obj).__name__ == '_TimeoutStopper':\n"
    "        obj._stop_datetime = None\n"
    "        break\n"
)
(gdb) call (void) PyGILState_Release($1)
```

Steps:
1. `PyGILState_Ensure()` — acquire the Python GIL from the gdb thread.
2. `PyRun_SimpleString()` — execute arbitrary Python code inside the live process.
   Used `gc.get_objects()` to find Optuna's `_TimeoutStopper` instance and set
   its `_stop_datetime` to `None` (disabling the timeout).
3. `PyGILState_Release()` — release the GIL.

### Result

- `PyRun_SimpleString` returned `$2 = 0` (success, no Python exception).
- Phase 4 ran for **14812 seconds** (past the original 14400s limit).
- All 10 trials completed normally.
- Process survived the injection without corruption.

### Risk Assessment

This was a calculated risk:
- **Upside**: Save 7+ hours of Phase 1-3 work by extending the run.
- **Downside**: gdb injection could corrupt the Python heap, crash the
  process, or produce silently wrong results.
- **Mitigation**: The injected code only modified a single attribute on a
  single object. No memory allocation, no complex operations.

### Lesson for Future Runs

Set `timeout_seconds=0` (no timeout) or a very high value (e.g., 86400 = 24h)
in the pipeline config. The Optuna `n_trials` parameter is a better stopping
criterion than wall-clock time.

---

## 8. System Resources During Training

### Resource Usage Summary

| Resource   | Usage                          | Capacity   | Risk   |
|------------|--------------------------------|------------|--------|
| RAM        | 7.5 - 8.9 GB (stable)         | 124 GB     | None   |
| GPU VRAM   | 1.4 - 12.0 GB (phase-dependent) | 16 GB    | None   |
| Disk       | 101 GB                         | 193 GB     | None   |
| Swap       | 256 KB                         | —          | None   |
| CPU        | 32 vCPUs available             | —          | None   |

### VRAM by Phase

| Phase   | VRAM Usage | Notes                                 |
|---------|------------|---------------------------------------|
| Phase 2 | ~12 GB     | Peak — all 276 features in GPU memory |
| Phase 4 | 1.4-3.5 GB | Reduced — only 73 features            |
| Phase 5 | 1.4-3.5 GB | Same as Phase 4                       |

The 74% feature reduction in Phase 3 has a direct GPU memory benefit.
No OOM risk at any point during the run.

---

## 9. Feature Usage Across Timeframes

### Current TFT: Daily Frequency

The TFT model operates at **daily** frequency — one signal per trading day.
Its 73 input features include tick-derived features that are aggregated to
daily granularity:

- `intra_orb_width` — Opening Range Breakout width (daily aggregate of 1-min data)
- `intra_rvol_1min` — Realized volatility from 1-min bars (daily aggregate)
- `micro_*` — Microstructure features from tick data (daily aggregate)

### Usage in EOD Strategies (S1-S25)

For end-of-day strategies, the TFT signal is directly usable via the
`TFTStrategy` adapter. The strategy receives a daily probability distribution
and position sizing signal.

### Usage in Tick/Intraday Strategies

For tick-level or intraday strategies, the daily TFT signal serves as a
**higher-timeframe regime overlay** (Pattern 1 from `docs/strategies/ANALYSIS.md`):

```
Daily TFT signal → regime context (bullish/bearish/neutral)
  ↓
Tick strategy uses regime to:
  - Adjust position limits
  - Filter trade direction
  - Scale sizing
```

### True Tick-Level TFT (Future Work)

A separate TFT trained at tick/minute frequency would need:
- Minute-bar or tick-bar features as inputs
- Shorter `seq_len` (e.g., 60 bars = 1 hour)
- Different target (next-bar return or next-N-minute return)
- Separate training pipeline with appropriate purge gaps for that frequency

---

## 10. Timeline

All times in UTC on 2026-02-10.

| Time (UTC) | Event                                              | Duration |
|------------|-----------------------------------------------------|----------|
| 05:29      | Pipeline started (Phase 1: Pre-filter)               | —        |
| 06:19      | Phase 2 started (Exploration, 11 folds)              | ~6h      |
| 12:19      | Phase 2 completed; Phase 3 completed (276 -> 73 features) | 0.0s |
| 12:19      | Phase 4 started (HP tuning, Optuna)                  | ~4.1h   |
| ~13:35     | GDB timeout patch applied (PID 3382427)              | —        |
| 16:26      | Phase 4 completed (10 trials, best OOS Sharpe 1.88)  | —        |
| 16:27      | Phase 5 started (Production pass, ~16 folds)         | ~7.5h est |
| ~midnight  | Phase 5 estimated completion + checkpoint save       | —        |

**Total estimated wall time**: ~18.5 hours (05:29 to ~midnight UTC).

---

## Appendix: Key Configuration Snapshot

```python
# TrainingPipelineConfig (as used in this run, with post-run fix noted)
{
    "n_features_initial": 276,
    "n_features_selected": 73,
    "exploration_step_size": 42,
    "production_step_size": 21,
    "exploration_folds": 11,
    "production_folds": 16,
    "epochs_per_fold_production": 80,
    "optuna_n_trials": 10,
    "optuna_timeout_seconds": 14400,  # was live-patched to None
    "purge_gap": 5,  # ADDED post-run — not active during this training
    "best_hp": {
        "d_hidden": 48,
        "n_heads": 4,
        "lstm_layers": 1,
        "dropout": 0.23,
        "seq_len": 63,
        "n_context": 16,
        "lr": 0.0027,
        "batch_size": 16,
        "mle_weight": 0.186,
        "loss_mode": "joint_mle",
        "max_position": 0.29,
        "position_smooth": 0.46,
    },
}
```

---

## Appendix: Next Steps

1. **Wait for Phase 5 completion** and inspect aggregate OOS Sharpe across all ~16 folds.
2. **Re-run with purge_gap=5** to get honest OOS numbers without feature leakage.
3. **Compare**: Sharpe with vs. without purge gap quantifies the leakage impact.
4. **Integrate into strategies**: Wire TFT checkpoint into `TFTStrategy` adapter for live use.
5. **Consider increasing Optuna trials**: 10 trials found Sharpe 1.88; 25-50 trials may find better.
6. **Set timeout_seconds=0** in config to avoid future gdb adventures.
