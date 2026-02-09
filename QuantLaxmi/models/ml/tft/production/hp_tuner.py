"""Hyperparameter Tuner for Production TFT — Optuna with MedianPruner.

Tunes architecture and training HPs on a reduced feature set (post feature
selection). Uses abbreviated walk-forward (fewer epochs) per trial for speed.

Search space:
- Architecture: d_hidden, n_heads, lstm_layers, dropout
- Sequence: seq_len, n_context
- Training: lr, batch_size, mle_weight, loss_mode
- Position sizing: max_position, position_smooth

Objective: mean OOS Sharpe across 3 walk-forward folds.

Usage
-----
    tuner = HPTuner(TunerConfig(n_trials=40))
    result = tuner.tune(features, targets, feature_names, dates)
    print(result.best_params, result.best_sharpe)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _HAS_TORCH = False
    _DEVICE = None

try:
    import optuna
    from optuna.pruners import MedianPruner
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TunerConfig:
    """Configuration for HP tuning."""

    n_trials: int = 40
    timeout_seconds: int = 14400  # 4 hours
    n_folds_per_trial: int = 3
    seed: int = 42

    # Abbreviated training budget per fold (faster than full training)
    pretrain_epochs: int = 15
    finetune_epochs: int = 25

    # Walk-forward settings
    train_window: int = 150
    test_window: int = 42
    step_size: int = 42

    # Fixed settings (not tuned)
    n_assets: int = 4
    cost_bps: float = 5.0
    max_daily_turnover: float = 0.5
    patience: int = 8

    # Optuna
    study_name: str = "tft_hp_tuning"
    storage: Optional[str] = None  # e.g. "sqlite:///optuna.db"


@dataclass
class TuningResult:
    """Result of HP tuning."""

    best_params: dict[str, Any]
    best_sharpe: float
    n_trials_completed: int
    elapsed_seconds: float
    study: Optional[Any] = None  # optuna.Study
    all_trials: list[dict[str, Any]] = field(default_factory=list)


# ============================================================================
# HP Tuner
# ============================================================================


class HPTuner:
    """Optuna-based hyperparameter tuner for X-Trend/TFT.

    Parameters
    ----------
    config : TunerConfig
        Tuning budget and settings.
    """

    def __init__(self, config: Optional[TunerConfig] = None) -> None:
        if not _HAS_OPTUNA:
            raise ImportError("optuna is required for HP tuning: pip install optuna")
        self.config = config or TunerConfig()

    def tune(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: list[str],
        dates: pd.DatetimeIndex,
        asset_names: Optional[list[str]] = None,
        progress_callback: Optional[Any] = None,
    ) -> TuningResult:
        """Run HP tuning.

        Parameters
        ----------
        features : ndarray of shape (n_days, n_assets, n_features)
            Normalized, pre-filtered features.
        targets : ndarray of shape (n_days, n_assets)
            Vol-scaled next-day returns.
        feature_names : list[str]
        dates : DatetimeIndex
        asset_names : list[str], optional
        progress_callback : callable, optional
            Called with (trial_number, best_sharpe) after each trial.

        Returns
        -------
        TuningResult
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch required for HP tuning")

        cfg = self.config
        n_days, n_assets, n_features = features.shape

        logger.info(
            "Starting HP tuning: %d trials, %d folds/trial, %d features, %d days",
            cfg.n_trials, cfg.n_folds_per_trial, n_features, n_days,
        )

        # Create Optuna study
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        study = optuna.create_study(
            study_name=cfg.study_name,
            storage=cfg.storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        start_time = time.time()

        def objective(trial: optuna.Trial) -> float:
            return self._objective(
                trial, features, targets, feature_names, n_assets,
            )

        # Optuna callbacks
        callbacks = []
        if progress_callback is not None:
            def _cb(study, trial):
                progress_callback(trial.number, study.best_value)
            callbacks.append(_cb)

        try:
            study.optimize(
                objective,
                n_trials=cfg.n_trials,
                timeout=cfg.timeout_seconds,
                callbacks=callbacks,
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            logger.info("HP tuning interrupted by user")

        elapsed = time.time() - start_time

        # Collect results
        all_trials = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                all_trials.append({
                    "number": t.number,
                    "params": t.params,
                    "value": t.value,
                    "duration": t.duration.total_seconds() if t.duration else 0,
                })

        best_params = study.best_params if study.best_trial else {}
        best_sharpe = study.best_value if study.best_trial else float("-inf")

        logger.info(
            "HP tuning complete: %d trials in %.0fs, best Sharpe=%.3f",
            len(all_trials), elapsed, best_sharpe,
        )
        logger.info("Best params: %s", best_params)

        return TuningResult(
            best_params=best_params,
            best_sharpe=best_sharpe,
            n_trials_completed=len(all_trials),
            elapsed_seconds=elapsed,
            study=study,
            all_trials=all_trials,
        )

    def _objective(
        self,
        trial: "optuna.Trial",
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: list[str],
        n_assets: int,
    ) -> float:
        """Optuna objective: mean OOS Sharpe across walk-forward folds.

        Parameters
        ----------
        trial : optuna.Trial
        features : (n_days, n_assets, n_features)
        targets : (n_days, n_assets)
        feature_names : list[str]
        n_assets : int

        Returns
        -------
        float — mean OOS Sharpe
        """
        from models.ml.tft.x_trend import XTrendConfig, XTrendModel, build_context_set

        cfg = self.config

        # Sample hyperparameters
        params = self._sample_params(trial)
        x_cfg = self._build_config(params, len(feature_names), n_assets)

        n_days = features.shape[0]
        rng = np.random.default_rng(cfg.seed)

        # Walk-forward folds
        fold_sharpes = []
        fold_idx = 0
        start = 0

        while start + cfg.train_window + cfg.test_window <= n_days:
            if fold_idx >= cfg.n_folds_per_trial:
                break

            train_end = start + cfg.train_window
            test_end = min(train_end + cfg.test_window, n_days)

            # Normalize using train stats
            train_feats = features[start:train_end]
            flat = train_feats.reshape(-1, x_cfg.n_features)
            valid = ~np.any(np.isnan(flat), axis=1)
            if valid.sum() < 30:
                start += cfg.step_size
                continue

            feat_mean = np.nanmean(flat[valid], axis=0)
            feat_std = np.nanstd(flat[valid], axis=0, ddof=1)
            feat_std = np.where(feat_std > 1e-10, feat_std, 1.0)
            norm_features = (features - feat_mean) / feat_std

            # Build training episodes
            episodes = self._build_episodes(
                norm_features, targets, start, train_end,
                x_cfg, n_assets, rng,
            )

            if len(episodes["targets"]) < 10:
                start += cfg.step_size
                continue

            # Train abbreviated model
            sharpe = self._train_and_evaluate(
                x_cfg, episodes, norm_features, targets,
                train_end, test_end, n_assets, rng,
            )

            fold_sharpes.append(sharpe)

            # Report intermediate for pruning
            trial.report(np.mean(fold_sharpes), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

            fold_idx += 1
            start += cfg.step_size

        if not fold_sharpes:
            return float("-inf")

        return float(np.mean(fold_sharpes))

    def _sample_params(self, trial: "optuna.Trial") -> dict[str, Any]:
        """Sample hyperparameters from the search space."""
        return {
            "d_hidden": trial.suggest_categorical("d_hidden", [32, 48, 64, 96, 128]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
            "lstm_layers": trial.suggest_categorical("lstm_layers", [1, 2, 3]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.3),
            "seq_len": trial.suggest_categorical("seq_len", [21, 42, 63]),
            "n_context": trial.suggest_categorical("n_context", [8, 16, 32]),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "mle_weight": trial.suggest_float("mle_weight", 0.01, 0.2),
            "loss_mode": trial.suggest_categorical("loss_mode", ["sharpe", "joint_mle"]),
            "max_position": trial.suggest_float("max_position", 0.15, 0.3),
            "position_smooth": trial.suggest_float("position_smooth", 0.2, 0.5),
        }

    def _build_config(
        self,
        params: dict[str, Any],
        n_features: int,
        n_assets: int,
    ) -> Any:
        """Build XTrendConfig from sampled params."""
        from models.ml.tft.x_trend import XTrendConfig

        return XTrendConfig(
            d_hidden=params["d_hidden"],
            n_heads=params["n_heads"],
            lstm_layers=params["lstm_layers"],
            dropout=params["dropout"],
            n_features=n_features,
            seq_len=params["seq_len"],
            n_context=params["n_context"],
            n_assets=n_assets,
            train_window=self.config.train_window,
            test_window=self.config.test_window,
            step_size=self.config.step_size,
            lr=params["lr"],
            batch_size=params["batch_size"],
            mle_weight=params["mle_weight"],
            loss_mode=params["loss_mode"],
            max_position=params["max_position"],
            position_smooth=params["position_smooth"],
            patience=self.config.patience,
        )

    def _build_episodes(
        self,
        norm_features: np.ndarray,
        targets: np.ndarray,
        start: int,
        train_end: int,
        cfg: Any,
        n_assets: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Build training episodes for a fold."""
        from models.ml.tft.x_trend import build_context_set

        X_target, X_context, X_tid, X_cid, Y = [], [], [], [], []
        seq_len = cfg.seq_len

        for asset_idx in range(n_assets):
            for t in range(start + seq_len, train_end):
                tw = norm_features[t - seq_len: t, asset_idx, :]
                if np.any(np.isnan(tw)):
                    continue
                tgt = targets[t, asset_idx]
                if np.isnan(tgt):
                    continue

                ctx_seqs, ctx_ids = build_context_set(
                    norm_features,
                    target_start=t - seq_len,
                    n_context=cfg.n_context,
                    ctx_len=cfg.ctx_len if hasattr(cfg, 'ctx_len') else cfg.seq_len,
                    rng=rng,
                )
                X_target.append(tw)
                X_context.append(ctx_seqs)
                X_tid.append(asset_idx)
                X_cid.append(ctx_ids)
                Y.append(tgt)

        return {
            "target_seqs": np.array(X_target, dtype=np.float32) if X_target else np.empty((0,)),
            "context_sets": np.array(X_context, dtype=np.float32) if X_context else np.empty((0,)),
            "target_ids": np.array(X_tid, dtype=np.int64) if X_tid else np.empty((0,)),
            "context_ids": np.array(X_cid, dtype=np.int64) if X_cid else np.empty((0,)),
            "targets": np.array(Y, dtype=np.float32) if Y else np.empty((0,)),
        }

    def _train_and_evaluate(
        self,
        cfg: Any,
        episodes: dict[str, np.ndarray],
        norm_features: np.ndarray,
        targets: np.ndarray,
        train_end: int,
        test_end: int,
        n_assets: int,
        rng: np.random.Generator,
    ) -> float:
        """Train abbreviated model and evaluate OOS Sharpe."""
        from models.ml.tft.x_trend import (
            XTrendModel, build_context_set, sharpe_loss, joint_loss,
        )

        model = XTrendModel(cfg).to(_DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

        X_tgt = episodes["target_seqs"]
        X_ctx = episodes["context_sets"]
        X_tid = episodes["target_ids"]
        X_cid = episodes["context_ids"]
        Y = episodes["targets"]

        n_total = len(Y)
        n_val = max(1, int(n_total * 0.2))
        n_train = n_total - n_val

        # Combined pretrain + finetune in abbreviated form
        total_epochs = self.config.pretrain_epochs + self.config.finetune_epochs
        best_state = None
        best_val_sharpe = -np.inf

        for epoch in range(total_epochs):
            model.train()
            perm = rng.permutation(n_train)

            for batch_start in range(0, n_train, cfg.batch_size):
                batch_idx = perm[batch_start: batch_start + cfg.batch_size]

                tgt_seq = torch.tensor(X_tgt[batch_idx], device=_DEVICE)
                ctx_set = torch.tensor(X_ctx[batch_idx], device=_DEVICE)
                tgt_id = torch.tensor(X_tid[batch_idx], device=_DEVICE)
                ctx_id = torch.tensor(X_cid[batch_idx], device=_DEVICE)
                y_batch = torch.tensor(Y[batch_idx], device=_DEVICE)

                if cfg.loss_mode == "joint_mle":
                    mu, log_sigma = model(tgt_seq, ctx_set, tgt_id, ctx_id)
                    loss = joint_loss(mu, log_sigma, y_batch.unsqueeze(-1), cfg.mle_weight)
                else:
                    positions = model(tgt_seq, ctx_set, tgt_id, ctx_id)
                    loss = sharpe_loss(positions.squeeze(-1), y_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validation Sharpe
            if n_val > 0:
                model.eval()
                with torch.no_grad():
                    val_idx = np.arange(n_train, n_total)
                    v_tgt = torch.tensor(X_tgt[val_idx], device=_DEVICE)
                    v_ctx = torch.tensor(X_ctx[val_idx], device=_DEVICE)
                    v_tid = torch.tensor(X_tid[val_idx], device=_DEVICE)
                    v_cid = torch.tensor(X_cid[val_idx], device=_DEVICE)
                    v_y = torch.tensor(Y[val_idx], device=_DEVICE)

                    v_pos = model.predict_position(v_tgt, v_ctx, v_tid, v_cid).squeeze(-1)
                    strat_ret = v_pos * v_y
                    mean_r = strat_ret.mean().item()
                    std_r = strat_ret.std(correction=1).item()
                    val_sharpe = (mean_r / max(std_r, 1e-8)) * math.sqrt(252)

                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Restore best
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(_DEVICE)

        # OOS evaluation
        oos_sharpe = self._evaluate_oos(
            model, cfg, norm_features, targets, train_end, test_end, n_assets, rng,
        )

        del model
        torch.cuda.empty_cache()
        return oos_sharpe

    def _evaluate_oos(
        self,
        model: "nn.Module",
        cfg: Any,
        norm_features: np.ndarray,
        targets: np.ndarray,
        test_start: int,
        test_end: int,
        n_assets: int,
        rng: np.random.Generator,
    ) -> float:
        """Evaluate OOS Sharpe on test fold."""
        from models.ml.tft.x_trend import build_context_set

        model.eval()
        all_returns = []

        with torch.no_grad():
            for t in range(test_start, test_end):
                seq_len = cfg.seq_len
                if t < seq_len:
                    continue

                for asset_idx in range(n_assets):
                    tw = norm_features[t - seq_len: t, asset_idx, :]
                    if np.any(np.isnan(tw)):
                        continue
                    tgt = targets[t, asset_idx]
                    if np.isnan(tgt):
                        continue

                    ctx_seqs, ctx_ids = build_context_set(
                        norm_features,
                        target_start=t - seq_len,
                        n_context=cfg.n_context,
                        ctx_len=cfg.ctx_len if hasattr(cfg, 'ctx_len') else cfg.seq_len,
                        rng=rng,
                    )

                    tgt_t = torch.tensor(tw[np.newaxis], dtype=torch.float32, device=_DEVICE)
                    ctx_t = torch.tensor(ctx_seqs[np.newaxis], dtype=torch.float32, device=_DEVICE)
                    tid_t = torch.tensor([asset_idx], dtype=torch.long, device=_DEVICE)
                    cid_t = torch.tensor(ctx_ids[np.newaxis], dtype=torch.long, device=_DEVICE)

                    pos = model.predict_position(tgt_t, ctx_t, tid_t, cid_t).item()
                    pos = max(-cfg.max_position, min(cfg.max_position, pos))
                    strat_return = pos * tgt
                    all_returns.append(strat_return)

        if len(all_returns) < 5:
            return float("-inf")

        returns_arr = np.array(all_returns)
        mean_r = returns_arr.mean()
        std_r = returns_arr.std(ddof=1)
        if std_r < 1e-8:
            return 0.0
        return float((mean_r / std_r) * math.sqrt(252))

    # ------------------------------------------------------------------
    # Utility: build config from best params
    # ------------------------------------------------------------------

    @staticmethod
    def build_config_from_params(
        params: dict[str, Any],
        n_features: int,
        n_assets: int = 4,
        **overrides: Any,
    ) -> Any:
        """Build a full XTrendConfig from tuning result params.

        Parameters
        ----------
        params : dict
            Best params from TuningResult.
        n_features : int
        n_assets : int
        **overrides : Any
            Additional config overrides (e.g. epochs, train_window).

        Returns
        -------
        XTrendConfig
        """
        from models.ml.tft.x_trend import XTrendConfig

        kwargs = {
            "n_features": n_features,
            "n_assets": n_assets,
        }

        # Map tuning params to config fields
        param_map = {
            "d_hidden": "d_hidden",
            "n_heads": "n_heads",
            "lstm_layers": "lstm_layers",
            "dropout": "dropout",
            "seq_len": "seq_len",
            "n_context": "n_context",
            "lr": "lr",
            "batch_size": "batch_size",
            "mle_weight": "mle_weight",
            "loss_mode": "loss_mode",
            "max_position": "max_position",
            "position_smooth": "position_smooth",
        }

        for param_key, cfg_key in param_map.items():
            if param_key in params:
                kwargs[cfg_key] = params[param_key]

        kwargs.update(overrides)
        return XTrendConfig(**kwargs)
