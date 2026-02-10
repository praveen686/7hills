"""5-Phase Training Pipeline for Production TFT.

Orchestrates the full 2-pass training approach:
  Phase 1: Pre-filter features (coverage + correlation)
  Phase 2: Exploration pass (train on ~210 features, extract VSN weights)
  Phase 3: Feature pruning (VSN stability analysis → ~60-80 features)
  Phase 4: HP Tuning (Optuna on reduced features)
  Phase 5: Production pass (full training + checkpoint save)

Progress monitoring via tqdm bars at phase, fold, and epoch levels.

Usage
-----
    pipeline = TrainingPipeline(TrainingPipelineConfig())
    result = pipeline.run()

CLI
---
    python -m models.ml.tft.production.training_pipeline \\
        --start 2024-01-01 --end 2026-02-06
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm, trange
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    tqdm = None  # type: ignore

try:
    import torch
    _HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _HAS_TORCH = False
    _DEVICE = None

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TrainingPipelineConfig:
    """Configuration for the 5-phase training pipeline."""

    # Date range
    start_date: str = "2024-01-01"
    end_date: str = "2026-02-06"

    # Symbols
    symbols: list[str] = field(
        default_factory=lambda: ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
    )

    # Phase toggles
    skip_feature_selection: bool = False
    skip_tuning: bool = False

    # Phase 2: Exploration pass
    exploration_pretrain_epochs: int = 15
    exploration_finetune_epochs: int = 25
    exploration_train_window: int = 150
    exploration_test_window: int = 42
    exploration_step_size: int = 42

    # Phase 4: HP tuning
    tuning_n_trials: int = 40
    tuning_timeout_seconds: int = 0  # 0 = no timeout (use n_trials as stopping criterion)

    # Phase 5: Production pass
    production_pretrain_epochs: int = 30
    production_finetune_epochs: int = 50
    production_train_window: int = 150
    production_test_window: int = 42
    production_step_size: int = 21

    # Purge gap (days between train end and test start to prevent look-ahead)
    purge_gap: int = 5

    # Target mode: "forward_return" (default) or "triple_barrier"
    target_mode: str = "forward_return"
    # Triple barrier params (only used when target_mode="triple_barrier")
    tb_pt_sl: tuple[float, float] = (1.0, 1.0)
    tb_num_days: int = 5
    tb_min_ret: float = 0.0

    # CV mode: "walk_forward" (default) or "cpcv"
    cv_mode: str = "walk_forward"
    # CPCV params (only used when cv_mode="cpcv")
    cpcv_n_splits: int = 10
    cpcv_n_test_groups: int = 2

    # Checkpoint
    checkpoint_base_dir: str = "checkpoints"
    model_type: str = "x_trend"

    # Base architecture (overridden by tuner in Phase 4)
    d_hidden: int = 64
    n_heads: int = 4
    lstm_layers: int = 2
    dropout: float = 0.1
    seq_len: int = 42
    n_context: int = 16
    lr: float = 1e-3
    batch_size: int = 32
    loss_mode: str = "sharpe"
    mle_weight: float = 0.1
    max_position: float = 0.25
    position_smooth: float = 0.3
    patience: int = 15


# ============================================================================
# Result
# ============================================================================


@dataclass
class TrainingResult:
    """Result of the full training pipeline."""

    checkpoint_path: Optional[Path] = None
    feature_report: Optional[str] = None
    tuning_result: Optional[Any] = None
    n_features_initial: int = 0
    n_features_prefiltered: int = 0
    n_features_final: int = 0
    sharpe_oos: float = 0.0
    total_return_oos: float = 0.0
    max_drawdown_oos: float = 0.0
    elapsed_seconds: float = 0.0
    phase_times: dict[str, float] = field(default_factory=dict)


# ============================================================================
# Training Pipeline
# ============================================================================


class TrainingPipeline:
    """5-phase production training pipeline.

    Parameters
    ----------
    config : TrainingPipelineConfig
    """

    def __init__(self, config: Optional[TrainingPipelineConfig] = None) -> None:
        self.config = config or TrainingPipelineConfig()
        self._progress_bar = None
        self._wandb_run = None

    def run(self) -> TrainingResult:
        """Execute the full 5-phase pipeline.

        Returns
        -------
        TrainingResult
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch required for training pipeline")

        cfg = self.config
        result = TrainingResult()
        t0 = time.time()

        # Initialize W&B run for experiment tracking
        if _HAS_WANDB:
            try:
                self._wandb_run = wandb.init(
                    project="quantlaxmi-tft",
                    job_type="training",
                    name=f"pipeline_{cfg.model_type}_{time.strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "start_date": cfg.start_date,
                        "end_date": cfg.end_date,
                        "symbols": cfg.symbols,
                        "d_hidden": cfg.d_hidden,
                        "n_heads": cfg.n_heads,
                        "lstm_layers": cfg.lstm_layers,
                        "dropout": cfg.dropout,
                        "seq_len": cfg.seq_len,
                        "lr": cfg.lr,
                        "batch_size": cfg.batch_size,
                        "loss_mode": cfg.loss_mode,
                        "purge_gap": cfg.purge_gap,
                        "production_pretrain_epochs": cfg.production_pretrain_epochs,
                        "production_finetune_epochs": cfg.production_finetune_epochs,
                        "tuning_n_trials": cfg.tuning_n_trials,
                        "skip_feature_selection": cfg.skip_feature_selection,
                        "skip_tuning": cfg.skip_tuning,
                    },
                )
                logger.info("W&B run initialized: %s", self._wandb_run.url)
            except Exception as e:
                logger.warning("W&B init failed (continuing without): %s", e)
                self._wandb_run = None

        phases = [
            ("Phase 1: Pre-filter features", self._phase1_prefilter),
            ("Phase 2: Exploration pass", self._phase2_exploration),
            ("Phase 3: Feature pruning", self._phase3_pruning),
            ("Phase 4: HP Tuning", self._phase4_tuning),
            ("Phase 5: Production pass", self._phase5_production),
        ]

        # State passed between phases
        state: dict[str, Any] = {
            "config": cfg,
            "result": result,
        }

        if _HAS_TQDM:
            phase_bar = tqdm(phases, desc="Pipeline", unit="phase", position=0)
        else:
            phase_bar = phases

        for phase_name, phase_fn in phase_bar:
            if _HAS_TQDM and hasattr(phase_bar, 'set_description'):
                phase_bar.set_description(phase_name)

            phase_start = time.time()
            logger.info("=" * 60)
            logger.info("Starting %s", phase_name)
            logger.info("=" * 60)

            try:
                phase_fn(state)
            except Exception as e:
                logger.error("%s failed: %s", phase_name, e, exc_info=True)
                raise

            elapsed = time.time() - phase_start
            result.phase_times[phase_name] = elapsed
            logger.info("%s completed in %.1fs", phase_name, elapsed)

            # Log phase timing to W&B
            if self._wandb_run is not None:
                try:
                    self._wandb_run.log({
                        f"phase_time/{phase_name}": elapsed,
                    })
                except Exception:
                    pass

        result.elapsed_seconds = time.time() - t0
        logger.info(
            "Pipeline complete in %.1fs — Sharpe=%.3f, features=%d→%d",
            result.elapsed_seconds, result.sharpe_oos,
            result.n_features_initial, result.n_features_final,
        )

        # Log final summary to W&B
        if self._wandb_run is not None:
            try:
                self._wandb_run.summary.update({
                    "sharpe_oos": result.sharpe_oos,
                    "total_return_oos": result.total_return_oos,
                    "max_drawdown_oos": result.max_drawdown_oos,
                    "n_features_initial": result.n_features_initial,
                    "n_features_prefiltered": result.n_features_prefiltered,
                    "n_features_final": result.n_features_final,
                    "elapsed_seconds": result.elapsed_seconds,
                })
                self._wandb_run.finish()
                logger.info("W&B run finished successfully")
            except Exception as e:
                logger.warning("W&B finish failed: %s", e)

        return result

    # ------------------------------------------------------------------
    # Phase 1: Pre-filter features
    # ------------------------------------------------------------------

    def _phase1_prefilter(self, state: dict) -> None:
        """Build mega features and apply coverage + correlation filters."""
        from .feature_selection import FeatureSelector, FeatureSelectionConfig
        from quantlaxmi.models.rl.integration.backbone import MegaFeatureAdapter

        cfg = state["config"]
        result = state["result"]

        # Build multi-asset features
        adapter = MegaFeatureAdapter(cfg.symbols)
        features, feature_names, dates = adapter.build_multi_asset(
            cfg.start_date, cfg.end_date,
        )
        # features: (n_days, n_assets, n_features)
        result.n_features_initial = len(feature_names)
        logger.info(
            "Built %d features × %d days × %d assets",
            len(feature_names), len(dates), len(cfg.symbols),
        )

        # Compute targets (vol-scaled next-day returns)
        targets = self._compute_targets(features, feature_names, dates, cfg)

        state["features_raw"] = features
        state["feature_names_raw"] = feature_names
        state["dates"] = dates
        state["targets"] = targets

        if cfg.skip_feature_selection:
            state["feature_names"] = feature_names
            state["features"] = features
            result.n_features_prefiltered = len(feature_names)
            logger.info("Skipping feature selection (using all %d features)", len(feature_names))
            return

        # Pre-filter: coverage + correlation on flattened data
        selector = FeatureSelector(FeatureSelectionConfig())

        # Flatten (n_days * n_assets, n_features) for pre-filter
        n_days, n_assets, n_feat = features.shape
        flat_df = pd.DataFrame(
            features.reshape(-1, n_feat),
            columns=feature_names,
        )

        kept_names = selector.prefilter(flat_df)
        result.n_features_prefiltered = len(kept_names)

        # Filter the 3D array
        kept_idx = [feature_names.index(n) for n in kept_names]
        features_filtered = features[:, :, kept_idx]

        state["features"] = features_filtered
        state["feature_names"] = kept_names
        state["selector"] = selector

    # ------------------------------------------------------------------
    # Phase 2: Exploration pass
    # ------------------------------------------------------------------

    def _phase2_exploration(self, state: dict) -> None:
        """Train TFT on pre-filtered features, extract VSN weights per fold."""
        from quantlaxmi.models.ml.tft.x_trend import XTrendConfig, XTrendModel, build_context_set
        from quantlaxmi.models.ml.tft.x_trend import sharpe_loss, joint_loss

        cfg = state["config"]
        features = state["features"]
        targets = state["targets"]
        feature_names = state["feature_names"]

        if cfg.skip_feature_selection:
            logger.info("Skipping exploration pass (feature selection disabled)")
            return

        n_days, n_assets, n_features = features.shape
        rng = np.random.default_rng(42)

        # Get or create selector
        selector = state.get("selector")
        if selector is None:
            from .feature_selection import FeatureSelector, FeatureSelectionConfig
            selector = FeatureSelector(FeatureSelectionConfig())
            state["selector"] = selector

        x_cfg = XTrendConfig(
            d_hidden=cfg.d_hidden,
            n_heads=cfg.n_heads,
            lstm_layers=cfg.lstm_layers,
            dropout=cfg.dropout,
            n_features=n_features,
            seq_len=cfg.seq_len,
            n_context=cfg.n_context,
            n_assets=n_assets,
            train_window=cfg.exploration_train_window,
            test_window=cfg.exploration_test_window,
            step_size=cfg.exploration_step_size,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            loss_mode=cfg.loss_mode,
            mle_weight=cfg.mle_weight,
            patience=cfg.patience,
        )

        # Walk-forward folds
        fold_idx = 0
        start = 0

        while start + x_cfg.train_window + cfg.purge_gap + x_cfg.test_window <= n_days:
            train_end = start + x_cfg.train_window
            test_start = train_end + cfg.purge_gap
            test_end = min(test_start + x_cfg.test_window, n_days)

            logger.info(
                "Exploration fold %d: train=[%d:%d], purge=%d, test=[%d:%d]",
                fold_idx, start, train_end, cfg.purge_gap, test_start, test_end,
            )

            # Normalize
            train_feats = features[start:train_end]
            flat = train_feats.reshape(-1, n_features)
            valid = ~np.any(np.isnan(flat), axis=1)
            if valid.sum() < 30:
                start += x_cfg.step_size
                continue

            feat_mean = np.nanmean(flat[valid], axis=0)
            feat_std = np.nanstd(flat[valid], axis=0, ddof=1)
            feat_std = np.where(feat_std > 1e-10, feat_std, 1.0)
            # Only normalize the relevant slice (train+test window + seq lookback)
            norm_lo = max(0, start - x_cfg.seq_len)
            norm_hi = test_end
            norm_features = np.full_like(features, np.nan)
            norm_features[norm_lo:norm_hi] = (features[norm_lo:norm_hi] - feat_mean) / feat_std

            # Build episodes
            X_tgt, X_ctx, X_tid, X_cid, Y = [], [], [], [], []
            for a in range(n_assets):
                for t in range(start + x_cfg.seq_len, train_end):
                    tw = norm_features[t - x_cfg.seq_len: t, a, :]
                    if np.any(np.isnan(tw)):
                        continue
                    tgt = targets[t, a]
                    if np.isnan(tgt):
                        continue
                    ctx_s, ctx_i = build_context_set(
                        norm_features, t - x_cfg.seq_len,
                        x_cfg.n_context, x_cfg.ctx_len, rng,
                    )
                    X_tgt.append(tw)
                    X_ctx.append(ctx_s)
                    X_tid.append(a)
                    X_cid.append(ctx_i)
                    Y.append(tgt)

            if len(X_tgt) < 10:
                start += x_cfg.step_size
                fold_idx += 1
                continue

            X_tgt_arr = np.array(X_tgt, dtype=np.float32)
            X_ctx_arr = np.array(X_ctx, dtype=np.float32)
            X_tid_arr = np.array(X_tid, dtype=np.int64)
            X_cid_arr = np.array(X_cid, dtype=np.int64)
            Y_arr = np.array(Y, dtype=np.float32)

            # Train abbreviated model
            model = XTrendModel(x_cfg).to(_DEVICE)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=x_cfg.lr, weight_decay=x_cfg.weight_decay,
            )

            n_total = len(Y_arr)
            n_val = max(1, int(n_total * 0.2))
            n_train = n_total - n_val

            total_epochs = cfg.exploration_pretrain_epochs + cfg.exploration_finetune_epochs

            epoch_iter = range(total_epochs)
            if _HAS_TQDM:
                epoch_iter = tqdm(
                    epoch_iter, desc=f"Fold {fold_idx}", unit="ep",
                    position=1, leave=False,
                )

            for epoch in epoch_iter:
                model.train()
                perm = rng.permutation(n_train)

                for bs in range(0, n_train, x_cfg.batch_size):
                    bi = perm[bs: bs + x_cfg.batch_size]
                    tgt_seq = torch.tensor(X_tgt_arr[bi], device=_DEVICE)
                    ctx_set = torch.tensor(X_ctx_arr[bi], device=_DEVICE)
                    tgt_id = torch.tensor(X_tid_arr[bi], device=_DEVICE)
                    ctx_id = torch.tensor(X_cid_arr[bi], device=_DEVICE)
                    y_batch = torch.tensor(Y_arr[bi], device=_DEVICE)

                    if x_cfg.loss_mode == "joint_mle":
                        mu, log_sigma = model(tgt_seq, ctx_set, tgt_id, ctx_id)
                        loss = joint_loss(mu, log_sigma, y_batch.unsqueeze(-1), x_cfg.mle_weight)
                    else:
                        pos = model(tgt_seq, ctx_set, tgt_id, ctx_id)
                        loss = sharpe_loss(pos.squeeze(-1), y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                if _HAS_TQDM and hasattr(epoch_iter, 'set_postfix'):
                    epoch_iter.set_postfix(loss=f"{loss.item():.4f}")

                # Abort fold early if loss diverged to NaN
                if not math.isfinite(loss.item()):
                    logger.warning("Fold %d diverged (NaN loss) at epoch %d — skipping VSN extraction", fold_idx, epoch)
                    break
            else:
                # Only extract VSN weights if training completed without NaN
                selector.add_fold_weights(model, feature_names)
                # Jump past the skip block below
                del model
                torch.cuda.empty_cache()
                start += x_cfg.step_size
                fold_idx += 1
                continue

            # NaN fold: skip VSN extraction, clean up
            del model
            torch.cuda.empty_cache()
            start += x_cfg.step_size
            fold_idx += 1
            continue

        state["selector"] = selector

    # ------------------------------------------------------------------
    # Phase 3: Feature pruning
    # ------------------------------------------------------------------

    def _phase3_pruning(self, state: dict) -> None:
        """Analyze VSN stability and select final feature subset."""
        cfg = state["config"]
        result = state["result"]

        if cfg.skip_feature_selection:
            result.n_features_final = len(state["feature_names"])
            logger.info("Skipping feature pruning (feature selection disabled)")
            return

        selector = state.get("selector")
        if selector is None or selector.n_folds == 0:
            logger.warning("No fold weights available, keeping all features")
            result.n_features_final = len(state["feature_names"])
            return

        report = selector.stability_report(state["feature_names"])
        result.feature_report = selector.feature_importance_report(report)
        logger.info("\n%s", result.feature_report)

        # Apply pruning
        recommended = report.recommended_features
        if not recommended:
            # Fallback: use top-K by mean VSN weight (sorted desc)
            ranked = sorted(
                report.mean_vsn_weights.items(), key=lambda x: -x[1],
            )
            max_k = selector.config.final_max_features
            recommended = [name for name, _ in ranked[:max_k]]
            logger.warning(
                "No stable features found — falling back to top-%d by mean VSN weight",
                len(recommended),
            )

        # Re-filter the feature array
        current_names = state["feature_names"]
        kept_idx = [current_names.index(n) for n in recommended if n in current_names]
        kept_names = [current_names[i] for i in kept_idx]

        state["features"] = state["features"][:, :, kept_idx]
        state["feature_names"] = kept_names
        result.n_features_final = len(kept_names)

        logger.info(
            "Feature pruning: %d → %d features (%.0f%% reduction)",
            len(current_names), len(kept_names),
            (1.0 - len(kept_names) / max(len(current_names), 1)) * 100,
        )

        # Log feature selection to W&B
        if self._wandb_run is not None:
            try:
                self._wandb_run.log({
                    "features/n_initial": result.n_features_initial,
                    "features/n_prefiltered": len(current_names),
                    "features/n_final": len(kept_names),
                    "features/reduction_pct": (1.0 - len(kept_names) / max(len(current_names), 1)) * 100,
                })
                # Log top-20 feature weights as a table
                if report.mean_vsn_weights:
                    ranked = sorted(report.mean_vsn_weights.items(), key=lambda x: -x[1])[:20]
                    table = wandb.Table(columns=["rank", "feature", "vsn_weight", "stability"])
                    for i, (feat, weight) in enumerate(ranked, 1):
                        stab = report.stability_scores.get(feat, 0.0)
                        table.add_data(i, feat, weight, stab)
                    self._wandb_run.log({"features/top_20_vsn": table})
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Phase 4: HP Tuning
    # ------------------------------------------------------------------

    def _phase4_tuning(self, state: dict) -> None:
        """Run Optuna HP tuning on reduced features."""
        import gc

        cfg = state["config"]
        result = state["result"]

        if cfg.skip_tuning:
            logger.info("Skipping HP tuning (using default config)")
            return

        # Clear GPU memory from Phase 2 exploration models
        if _HAS_TORCH:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e9
                logger.info("GPU memory before HP tuning: %.2f GB allocated", alloc)

        from .hp_tuner import HPTuner, TunerConfig

        tuner_cfg = TunerConfig(
            n_trials=cfg.tuning_n_trials,
            timeout_seconds=cfg.tuning_timeout_seconds,
            train_window=cfg.production_train_window,
            test_window=cfg.production_test_window,
            step_size=cfg.production_step_size,
            purge_gap=cfg.purge_gap,
            n_assets=len(cfg.symbols),
        )

        tuner = HPTuner(tuner_cfg)
        tuning_result = tuner.tune(
            features=state["features"],
            targets=state["targets"],
            feature_names=state["feature_names"],
            dates=state["dates"],
            asset_names=cfg.symbols,
        )

        result.tuning_result = tuning_result
        state["best_params"] = tuning_result.best_params
        state["optuna_study"] = tuning_result.study

        logger.info(
            "HP tuning: best Sharpe=%.3f after %d trials",
            tuning_result.best_sharpe, tuning_result.n_trials_completed,
        )

        # Log fANOVA HP importances
        if tuning_result.hp_importances:
            state["hp_importances"] = tuning_result.hp_importances

        # Log HP tuning results to W&B
        if self._wandb_run is not None:
            try:
                self._wandb_run.log({
                    "tuning/best_sharpe": tuning_result.best_sharpe,
                    "tuning/n_trials": tuning_result.n_trials_completed,
                })
                # Log best params
                for k, v in tuning_result.best_params.items():
                    self._wandb_run.config.update({f"best_{k}": v})
                # Log fANOVA importances
                if tuning_result.hp_importances:
                    table = wandb.Table(columns=["param", "importance"])
                    for param, imp in sorted(
                        tuning_result.hp_importances.items(), key=lambda x: -x[1]
                    ):
                        table.add_data(param, imp)
                    self._wandb_run.log({"tuning/fanova_importances": table})
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Phase 5: Production pass
    # ------------------------------------------------------------------

    def _phase5_production(self, state: dict) -> None:
        """Full walk-forward with best features + best HPs, save checkpoint."""
        from quantlaxmi.models.ml.tft.x_trend import XTrendConfig, XTrendModel, build_context_set
        from quantlaxmi.models.ml.tft.x_trend import sharpe_loss, joint_loss
        from .checkpoint_manager import CheckpointManager, CheckpointMetadata
        from dataclasses import asdict

        cfg = state["config"]
        result = state["result"]
        features = state["features"]
        targets = state["targets"]
        feature_names = state["feature_names"]
        dates = state["dates"]

        n_days, n_assets, n_features = features.shape
        rng = np.random.default_rng(42)

        # Build config from best params or defaults
        best_params = state.get("best_params", {})
        if best_params:
            from .hp_tuner import HPTuner
            x_cfg = HPTuner.build_config_from_params(
                best_params, n_features, n_assets,
                train_window=cfg.production_train_window,
                test_window=cfg.production_test_window,
                step_size=cfg.production_step_size,
                patience=cfg.patience,
            )
        else:
            x_cfg = XTrendConfig(
                d_hidden=cfg.d_hidden,
                n_heads=cfg.n_heads,
                lstm_layers=cfg.lstm_layers,
                dropout=cfg.dropout,
                n_features=n_features,
                seq_len=cfg.seq_len,
                n_context=cfg.n_context,
                n_assets=n_assets,
                train_window=cfg.production_train_window,
                test_window=cfg.production_test_window,
                step_size=cfg.production_step_size,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                loss_mode=cfg.loss_mode,
                mle_weight=cfg.mle_weight,
                max_position=cfg.max_position,
                position_smooth=cfg.position_smooth,
                patience=cfg.patience,
            )

        # Walk-forward with full epoch budget
        all_oos_returns = []
        last_model_state = None
        last_norm_means = None
        last_norm_stds = None
        fold_metrics = []

        fold_idx = 0
        start = 0

        while start + x_cfg.train_window + cfg.purge_gap + x_cfg.test_window <= n_days:
            train_end = start + x_cfg.train_window
            test_start = train_end + cfg.purge_gap
            test_end = min(test_start + x_cfg.test_window, n_days)

            logger.info(
                "Production fold %d: train=[%d:%d], purge=%d, test=[%d:%d]",
                fold_idx, start, train_end, cfg.purge_gap, test_start, test_end,
            )

            # Normalize
            train_feats = features[start:train_end]
            flat = train_feats.reshape(-1, n_features)
            valid = ~np.any(np.isnan(flat), axis=1)
            if valid.sum() < 30:
                start += x_cfg.step_size
                continue

            feat_mean = np.nanmean(flat[valid], axis=0)
            feat_std = np.nanstd(flat[valid], axis=0, ddof=1)
            feat_std = np.where(feat_std > 1e-10, feat_std, 1.0)
            # Only normalize the relevant slice (train+test window + seq lookback)
            norm_lo = max(0, start - x_cfg.seq_len)
            norm_hi = test_end
            norm_features = np.full_like(features, np.nan)
            norm_features[norm_lo:norm_hi] = (features[norm_lo:norm_hi] - feat_mean) / feat_std

            last_norm_means = feat_mean
            last_norm_stds = feat_std

            # Build episodes
            X_tgt, X_ctx, X_tid, X_cid, Y = [], [], [], [], []
            for a in range(n_assets):
                for t in range(start + x_cfg.seq_len, train_end):
                    tw = norm_features[t - x_cfg.seq_len: t, a, :]
                    if np.any(np.isnan(tw)):
                        continue
                    tgt = targets[t, a]
                    if np.isnan(tgt):
                        continue
                    ctx_s, ctx_i = build_context_set(
                        norm_features, t - x_cfg.seq_len,
                        x_cfg.n_context, x_cfg.ctx_len, rng,
                    )
                    X_tgt.append(tw)
                    X_ctx.append(ctx_s)
                    X_tid.append(a)
                    X_cid.append(ctx_i)
                    Y.append(tgt)

            if len(X_tgt) < 10:
                start += x_cfg.step_size
                fold_idx += 1
                continue

            X_tgt_arr = np.array(X_tgt, dtype=np.float32)
            X_ctx_arr = np.array(X_ctx, dtype=np.float32)
            X_tid_arr = np.array(X_tid, dtype=np.int64)
            X_cid_arr = np.array(X_cid, dtype=np.int64)
            Y_arr = np.array(Y, dtype=np.float32)

            # Train full model
            model = XTrendModel(x_cfg).to(_DEVICE)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=x_cfg.lr, weight_decay=x_cfg.weight_decay,
            )

            n_total = len(Y_arr)
            n_val = max(1, int(n_total * 0.2))
            n_train = n_total - n_val

            total_epochs = cfg.production_pretrain_epochs + cfg.production_finetune_epochs
            best_val_sharpe = -np.inf
            best_state = None

            epoch_iter = range(total_epochs)
            if _HAS_TQDM:
                epoch_iter = tqdm(
                    epoch_iter, desc=f"Prod fold {fold_idx}",
                    unit="ep", position=1, leave=False,
                )

            for epoch in epoch_iter:
                model.train()
                perm = rng.permutation(n_train)
                epoch_loss = 0.0
                n_batches = 0

                for bs in range(0, n_train, x_cfg.batch_size):
                    bi = perm[bs: bs + x_cfg.batch_size]
                    tgt_seq = torch.tensor(X_tgt_arr[bi], device=_DEVICE)
                    ctx_set = torch.tensor(X_ctx_arr[bi], device=_DEVICE)
                    tgt_id = torch.tensor(X_tid_arr[bi], device=_DEVICE)
                    ctx_id = torch.tensor(X_cid_arr[bi], device=_DEVICE)
                    y_batch = torch.tensor(Y_arr[bi], device=_DEVICE)

                    if x_cfg.loss_mode == "joint_mle":
                        mu, log_sigma = model(tgt_seq, ctx_set, tgt_id, ctx_id)
                        loss = joint_loss(mu, log_sigma, y_batch.unsqueeze(-1), x_cfg.mle_weight)
                    else:
                        pos = model(tgt_seq, ctx_set, tgt_id, ctx_id)
                        loss = sharpe_loss(pos.squeeze(-1), y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                # Abort if loss diverged
                if not math.isfinite(epoch_loss):
                    logger.warning("Prod fold %d diverged at epoch %d", fold_idx, epoch)
                    break

                # Validation
                if n_val > 0:
                    model.eval()
                    with torch.no_grad():
                        vi = np.arange(n_train, n_total)
                        v_tgt = torch.tensor(X_tgt_arr[vi], device=_DEVICE)
                        v_ctx = torch.tensor(X_ctx_arr[vi], device=_DEVICE)
                        v_tid = torch.tensor(X_tid_arr[vi], device=_DEVICE)
                        v_cid = torch.tensor(X_cid_arr[vi], device=_DEVICE)
                        v_y = torch.tensor(Y_arr[vi], device=_DEVICE)

                        v_pos = model.predict_position(v_tgt, v_ctx, v_tid, v_cid).squeeze(-1)
                        strat_ret = v_pos * v_y
                        mean_r = strat_ret.mean().item()
                        std_r = strat_ret.std(correction=1).item()
                        val_sharpe = (mean_r / max(std_r, 1e-8)) * math.sqrt(252)

                    if val_sharpe > best_val_sharpe:
                        best_val_sharpe = val_sharpe
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                if _HAS_TQDM and hasattr(epoch_iter, 'set_postfix'):
                    avg_loss = epoch_loss / max(n_batches, 1)
                    epoch_iter.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        val_sharpe=f"{best_val_sharpe:.3f}",
                    )

            if best_state is not None:
                model.load_state_dict(best_state)
                model.to(_DEVICE)

            # OOS evaluation
            fold_returns = []
            model.eval()
            with torch.no_grad():
                for t in range(test_start, test_end):
                    if t < x_cfg.seq_len:
                        continue
                    for a in range(n_assets):
                        tw = norm_features[t - x_cfg.seq_len: t, a, :]
                        if np.any(np.isnan(tw)):
                            continue
                        tgt = targets[t, a]
                        if np.isnan(tgt):
                            continue

                        ctx_s, ctx_i = build_context_set(
                            norm_features, t - x_cfg.seq_len,
                            x_cfg.n_context, x_cfg.ctx_len, rng,
                        )
                        tgt_t = torch.tensor(tw[np.newaxis], dtype=torch.float32, device=_DEVICE)
                        ctx_t = torch.tensor(ctx_s[np.newaxis], dtype=torch.float32, device=_DEVICE)
                        tid_t = torch.tensor([a], dtype=torch.long, device=_DEVICE)
                        cid_t = torch.tensor(ctx_i[np.newaxis], dtype=torch.long, device=_DEVICE)

                        pos_val = model.predict_position(tgt_t, ctx_t, tid_t, cid_t).item()
                        pos_val = max(-x_cfg.max_position, min(x_cfg.max_position, pos_val))
                        fold_returns.append(pos_val * tgt)

            if fold_returns:
                fold_arr = np.array(fold_returns)
                fold_sharpe = (fold_arr.mean() / max(fold_arr.std(ddof=1), 1e-8)) * math.sqrt(252)
                fold_metrics.append({
                    "fold": fold_idx,
                    "sharpe": float(fold_sharpe),
                    "n_predictions": len(fold_returns),
                    "mean_return": float(fold_arr.mean()),
                })
                all_oos_returns.extend(fold_returns)
                logger.info(
                    "Fold %d OOS: Sharpe=%.3f (%d predictions)",
                    fold_idx, fold_sharpe, len(fold_returns),
                )

                # Log per-fold metrics to W&B
                if self._wandb_run is not None:
                    try:
                        running_sharpe = 0.0
                        if all_oos_returns:
                            oos_so_far = np.array(all_oos_returns)
                            running_sharpe = float(
                                (oos_so_far.mean() / max(oos_so_far.std(ddof=1), 1e-8))
                                * math.sqrt(252)
                            )
                        self._wandb_run.log({
                            "fold/idx": fold_idx,
                            "fold/sharpe_oos": float(fold_sharpe),
                            "fold/n_predictions": len(fold_returns),
                            "fold/mean_return": float(fold_arr.mean()),
                            "fold/best_val_sharpe": float(best_val_sharpe),
                            "running/sharpe_oos": running_sharpe,
                            "running/n_returns": len(all_oos_returns),
                        })
                    except Exception:
                        pass

            last_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            del model
            torch.cuda.empty_cache()

            start += x_cfg.step_size
            fold_idx += 1

        # Compute overall OOS metrics
        if all_oos_returns:
            oos_arr = np.array(all_oos_returns)
            result.sharpe_oos = float(
                (oos_arr.mean() / max(oos_arr.std(ddof=1), 1e-8)) * math.sqrt(252)
            )
            result.total_return_oos = float(np.sum(oos_arr))

            # Max drawdown
            cum = np.cumsum(oos_arr)
            running_max = np.maximum.accumulate(cum)
            dd = running_max - cum
            result.max_drawdown_oos = float(np.max(dd)) if len(dd) > 0 else 0.0

        # Save checkpoint (wrapped to never lose 8+ hours of training)
        if last_model_state is not None:
            try:
                mgr = CheckpointManager(cfg.checkpoint_base_dir)

                metadata = CheckpointMetadata(
                    model_type=cfg.model_type,
                    feature_names=feature_names,
                    n_features=n_features,
                    n_assets=n_assets,
                    asset_names=cfg.symbols,
                    config=asdict(x_cfg) if hasattr(x_cfg, '__dataclass_fields__') else {},
                    normalization={
                        "means": last_norm_means.tolist() if last_norm_means is not None else [],
                        "stds": last_norm_stds.tolist() if last_norm_stds is not None else [],
                    },
                    training_info={
                        "fold_metrics": fold_metrics,
                        "total_folds": fold_idx,
                        "pretrain_epochs": cfg.production_pretrain_epochs,
                        "finetune_epochs": cfg.production_finetune_epochs,
                        "date_range": [cfg.start_date, cfg.end_date],
                    },
                    feature_selection={
                        "n_initial": result.n_features_initial,
                        "n_prefiltered": result.n_features_prefiltered,
                        "n_final": result.n_features_final,
                    },
                    optuna_best_params=state.get("best_params", {}),
                    sharpe_oos=result.sharpe_oos,
                    total_return_oos=result.total_return_oos,
                    max_drawdown_oos=result.max_drawdown_oos,
                )

                selector = state.get("selector")
                optuna_study = state.get("optuna_study")

                ckpt_dir = mgr.save(
                    last_model_state, metadata, selector, optuna_study,
                )
                result.checkpoint_path = ckpt_dir
            except Exception as e:
                logger.error(
                    "Checkpoint save failed: %s — training results are still "
                    "available in the returned TrainingResult object", e,
                )
                result.checkpoint_path = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_targets(
        self,
        features: np.ndarray,
        feature_names: list[str],
        dates: pd.DatetimeIndex,
        cfg: TrainingPipelineConfig,
    ) -> np.ndarray:
        """Compute vol-scaled next-day returns from price features.

        Looks for 'price_close' or 'ret_1d' in feature names to derive
        returns. Falls back to loading from MegaFeatureBuilder.
        """
        n_days, n_assets, n_features = features.shape
        targets = np.full((n_days, n_assets), np.nan)

        # Try to find return features
        ret_1d_idx = None
        for i, name in enumerate(feature_names):
            if name in ("ret_1d", "price_ret_1d"):
                ret_1d_idx = i
                break

        if ret_1d_idx is not None:
            # Use existing 1d return feature, shifted forward by 1 day
            for a in range(n_assets):
                for t in range(n_days - 1):
                    targets[t, a] = features[t + 1, a, ret_1d_idx]
        else:
            # Try to find close price and compute returns
            close_idx = None
            for i, name in enumerate(feature_names):
                if "close" in name.lower() and "log" not in name.lower():
                    close_idx = i
                    break

            if close_idx is not None:
                for a in range(n_assets):
                    close = features[:, a, close_idx]
                    log_close = np.log(np.where(close > 0, close, np.nan))
                    for t in range(n_days - 1):
                        if not np.isnan(log_close[t]) and not np.isnan(log_close[t + 1]):
                            targets[t, a] = log_close[t + 1] - log_close[t]
            else:
                # Fallback: use the first return-like feature
                for i, name in enumerate(feature_names):
                    if "ret" in name.lower():
                        ret_1d_idx = i
                        break
                if ret_1d_idx is not None:
                    for a in range(n_assets):
                        for t in range(n_days - 1):
                            targets[t, a] = features[t + 1, a, ret_1d_idx]
                else:
                    logger.warning(
                        "Could not find return/close features for target computation. "
                        "Using zeros."
                    )

        # Vol-scale targets
        for a in range(n_assets):
            col = targets[:, a]
            vol = pd.Series(col).rolling(20, min_periods=10).std(ddof=1).values
            safe_vol = np.where((vol > 0) & ~np.isnan(vol), vol, 1.0)
            targets[:, a] = col / safe_vol

        return targets


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """CLI entry point for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Production TFT Training Pipeline"
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default="2026-02-06", help="End date")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--skip-feature-selection", action="store_true")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--purge-gap", type=int, default=5, help="Days between train/test to prevent look-ahead bias")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = TrainingPipelineConfig(
        start_date=args.start,
        end_date=args.end,
        skip_tuning=args.skip_tuning,
        skip_feature_selection=args.skip_feature_selection,
        tuning_n_trials=args.n_trials,
        checkpoint_base_dir=args.checkpoint_dir,
        purge_gap=args.purge_gap,
    )

    pipeline = TrainingPipeline(config)
    result = pipeline.run()

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {result.checkpoint_path}")
    print(f"  Features: {result.n_features_initial} → {result.n_features_final}")
    print(f"  OOS Sharpe: {result.sharpe_oos:.3f}")
    print(f"  OOS Return: {result.total_return_oos:.4f}")
    print(f"  Max DD: {result.max_drawdown_oos:.4f}")
    print(f"  Elapsed: {result.elapsed_seconds:.0f}s")
    for phase, t in result.phase_times.items():
        print(f"    {phase}: {t:.0f}s")
    if result.feature_report:
        print(f"\n{result.feature_report}")


if __name__ == "__main__":
    main()
