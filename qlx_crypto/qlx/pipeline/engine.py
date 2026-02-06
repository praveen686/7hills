"""ResearchEngine — the orchestrator.

This is the top-level entry point for running a complete research pipeline:

    1. Validate all features (lookforward == 0)
    2. Validate CV gap >= target horizon
    3. Build feature matrix, build target
    4. Align X and y, validate alignment
    5. Walk-forward or expanding CV with fresh model per fold
    6. Collect predictions, evaluate metrics
    7. Run backtest with mandatory cost model
    8. Return structured results

The engine is stateless: every ``run()`` call starts fresh.  Configuration
is frozen.  Results are returned, never stored as side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from qlx.backtest.costs import CostModel
from qlx.backtest.portfolio import BacktestResult, run_backtest
from qlx.core.timeguard import TimeGuard
from qlx.core.types import OHLCV
from qlx.features.base import Feature
from qlx.features.matrix import FeatureMatrix
from qlx.models.factory import ModelFactory
from qlx.pipeline.config import PipelineConfig
from qlx.pipeline.split import ExpandingSplit, SplitResult, WalkForwardSplit


@dataclass(frozen=True)
class FoldResult:
    """Metrics and predictions from a single CV fold."""

    fold: int
    train_size: int
    test_size: int
    mse: float
    mae: float
    r2: float
    predictions: pd.Series
    actuals: pd.Series


@dataclass(frozen=True)
class EngineResult:
    """Complete output of a research run."""

    config: PipelineConfig
    fold_results: tuple[FoldResult, ...]
    all_predictions: pd.Series
    all_actuals: pd.Series
    aggregate_mse: float
    aggregate_mae: float
    aggregate_r2: float
    backtest: BacktestResult | None
    feature_names: list[str]

    def summary(self) -> str:
        lines = [
            "=== QLX Research Engine Results ===",
            f"Folds:          {len(self.fold_results)}",
            f"Total test obs: {len(self.all_predictions)}",
            f"MSE:            {self.aggregate_mse:.6f}",
            f"MAE:            {self.aggregate_mae:.6f}",
            f"R^2:            {self.aggregate_r2:.6f}",
        ]
        if self.backtest:
            lines.append("")
            lines.append(self.backtest.summary())
        return "\n".join(lines)


class ResearchEngine:
    """Stateless research pipeline orchestrator."""

    def __init__(
        self,
        ohlcv: OHLCV,
        features: list[Feature],
        target_transform,  # TargetTransform protocol
        config: PipelineConfig,
    ):
        self._ohlcv = ohlcv
        self._features = features
        self._target_transform = target_transform
        self._config = config

        # Validate at construction time
        TimeGuard.validate_features(features)
        TimeGuard.validate_cv_gap(config.cv.gap, config.target.horizon)

    def run(self, verbose: bool = True) -> EngineResult:
        """Execute the full pipeline.  Returns structured results."""

        # 1. Build feature matrix
        builder = FeatureMatrix(self._ohlcv)
        for feat in self._features:
            builder = builder.add(feat)
        X = builder.build(dropna=True)

        if verbose:
            print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")

        # 2. Build target
        y = self._target_transform.transform(self._ohlcv)

        # 3. Align X and y (intersect indices)
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # 4. Validate alignment
        TimeGuard.validate_alignment(X, y, self._config.target.horizon)

        if verbose:
            print(f"Aligned dataset: {len(X)} samples")

        # 5. Create splitter
        splitter = self._make_splitter()
        n_folds = splitter.n_splits(len(X))
        if verbose:
            print(f"CV folds: {n_folds}")

        # 6. Walk-forward training
        model_factory = ModelFactory(
            model_name=self._config.model.name,
            task=self._config.model.task,
            model_params=self._config.model.params,
        )

        fold_results = []
        all_preds = []
        all_actuals = []

        X_values = X.values
        y_values = y.values
        X_index = X.index
        y_index = y.index

        for split in splitter.split(len(X)):
            # Fresh model each fold
            pipeline = model_factory.build()

            X_train = X_values[split.train_idx]
            y_train = y_values[split.train_idx]
            X_test = X_values[split.test_idx]
            y_test = y_values[split.test_idx]
            test_index = X_index[split.test_idx]

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            pred_series = pd.Series(preds, index=test_index, name="prediction")
            actual_series = pd.Series(y_test, index=test_index, name="actual")

            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds) if len(y_test) > 1 else float("nan")

            fold_result = FoldResult(
                fold=split.fold,
                train_size=len(split.train_idx),
                test_size=len(split.test_idx),
                mse=mse,
                mae=mae,
                r2=r2,
                predictions=pred_series,
                actuals=actual_series,
            )
            fold_results.append(fold_result)
            all_preds.append(pred_series)
            all_actuals.append(actual_series)

            if verbose and split.fold % max(1, n_folds // 10) == 0:
                print(
                    f"  Fold {split.fold:4d} | "
                    f"train={len(split.train_idx):5d} | "
                    f"test={len(split.test_idx):4d} | "
                    f"MSE={mse:.6f} | R2={r2:.4f}"
                )

        # 7. Aggregate predictions (deduplicate overlapping folds)
        all_predictions = pd.concat(all_preds)
        all_actuals_series = pd.concat(all_actuals)

        # Keep first occurrence for any duplicate index
        mask = ~all_predictions.index.duplicated(keep="first")
        all_predictions = all_predictions[mask]
        all_actuals_series = all_actuals_series[mask]

        agg_mse = mean_squared_error(all_actuals_series, all_predictions)
        agg_mae = mean_absolute_error(all_actuals_series, all_predictions)
        agg_r2 = r2_score(all_actuals_series, all_predictions)

        # 8. Backtest with costs
        cost_model = CostModel(
            commission_bps=self._config.costs.commission_bps,
            slippage_bps=self._config.costs.slippage_bps,
            funding_annual_pct=self._config.costs.funding_annual_pct,
        )

        prices = self._ohlcv.close.loc[all_predictions.index]
        backtest_result = run_backtest(
            prices=prices,
            predictions=all_predictions,
            cost_model=cost_model,
            long_entry_th=self._config.long_entry_threshold,
            long_exit_th=self._config.long_exit_threshold,
            short_entry_th=self._config.short_entry_threshold,
            short_exit_th=self._config.short_exit_threshold,
        )

        return EngineResult(
            config=self._config,
            fold_results=tuple(fold_results),
            all_predictions=all_predictions,
            all_actuals=all_actuals_series,
            aggregate_mse=agg_mse,
            aggregate_mae=agg_mae,
            aggregate_r2=agg_r2,
            backtest=backtest_result,
            feature_names=list(X.columns),
        )

    def _make_splitter(self) -> WalkForwardSplit | ExpandingSplit:
        cfg = self._config.cv
        horizon = self._config.target.horizon

        if cfg.method == "walk_forward":
            return WalkForwardSplit(
                window=cfg.window,
                train_frac=cfg.train_frac,
                gap=cfg.gap,
                horizon=horizon,
                step=cfg.step,
            )
        elif cfg.method == "expanding":
            train_size = int(cfg.window * cfg.train_frac)
            test_size = cfg.window - train_size - cfg.gap
            return ExpandingSplit(
                min_train=train_size,
                test_size=test_size,
                gap=cfg.gap,
                horizon=horizon,
                step=cfg.step,
            )
        else:
            raise ValueError(f"Unknown CV method: {cfg.method}")
