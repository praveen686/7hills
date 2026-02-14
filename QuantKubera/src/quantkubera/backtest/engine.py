"""Core backtesting engine for QuantKubera."""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
from .metrics import calculate_all_metrics
from .cpcv import CombPurgedKFoldCV
from ..training.dataset import TimeSeriesWindowDataset
from ..training.trainer import MomentumTrainer
from ..models.tft import MomentumTransformer


class BacktestEngine:
    """Handles training and evaluation across multiple temporal splits."""
    
    def __init__(
        self,
        model_config: dict,
        data_config: dict,
        train_config: dict
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config
        self.results = []

    def run_fold(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fold_id: str = "1",
        verbose: int = 0
    ) -> Dict:
        """Runs a single train/test fold."""
        # Create datasets
        train_ds = TimeSeriesWindowDataset(
            train_df,
            window_size=self.data_config['window_size'],
            target_col=self.data_config['target_col'],
            feature_cols=self.data_config['feature_cols']
        ).to_tf_dataset(batch_size=self.train_config['batch_size'], shuffle=True)
        
        test_ds = TimeSeriesWindowDataset(
            test_df,
            window_size=self.data_config['window_size'],
            target_col=self.data_config['target_col'],
            feature_cols=self.data_config['feature_cols']
        ).to_tf_dataset(batch_size=self.train_config['batch_size'], shuffle=False)
        
        # Instantiate model
        model = MomentumTransformer(
            time_steps=self.data_config['window_size'],
            input_size=len(self.data_config['feature_cols']),
            output_size=self.model_config['output_size'],
            hidden_size=self.model_config['hidden_size'],
            num_heads=self.model_config['num_heads'],
            dropout_rate=self.model_config['dropout_rate']
        )
        
        trainer = MomentumTrainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=test_ds,
            learning_rate=self.train_config['learning_rate'],
            model_dir=None
        )
        
        # Train
        trainer.train(epochs=6, verbose=verbose)
        
        # Predict on test set
        test_X, test_y = self._get_full_numpy(test_ds)
        predictions = model.predict(test_X, verbose=0)
        
        # Simple signal: sign of prediction
        # (In reality, TMT outputs weights, but for binary direction evaluation:)
        actual_returns = test_y[:, -1, 0]
        predicted_signals = predictions[:, -1, 0]
        
        # Strategy returns: direction * actual
        strategy_returns = np.sign(predicted_signals) * actual_returns
        
        # Calculate metrics
        metrics = calculate_all_metrics(strategy_returns)
        metrics['fold_id'] = fold_id
        metrics['train_samples'] = len(train_df)
        metrics['test_samples'] = len(test_df)
        
        return metrics

    def run_cpcv(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        n_test_groups: int = 2,
        purge_window: int = 21,
        embargo_pct: float = 0.01,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Runs CPCV backtest on full dataset."""
        cpcv = CombPurgedKFoldCV(
            n_splits=n_splits,
            n_test_groups=n_test_groups,
            purge_window=purge_window,
            embargo_pct=embargo_pct
        )
        
        all_metrics = []
        for i, (train_idx, test_idx) in enumerate(cpcv.split(df)):
            if verbose:
                print(f"Running Fold {i+1}...")
            
            metrics = self.run_fold(
                df.iloc[train_idx],
                df.iloc[test_idx],
                fold_id=str(i+1),
                verbose=0
            )
            all_metrics.append(metrics)
            
            if verbose:
                print(f"  Fold {i+1} Sharpe: {metrics['sharpe_ratio']:.3f}")
                
        self.results = all_metrics
        return pd.DataFrame(all_metrics)

    def _get_full_numpy(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Exhausts a dataset to get full numpy arrays for evaluation."""
        Xs, ys = [], []
        for X, y in dataset:
            Xs.append(X.numpy())
            ys.append(y.numpy())
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
