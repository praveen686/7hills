"""Time series dataset with sliding window for TMT training."""
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Optional


class TimeSeriesWindowDataset:
    """Creates sliding window batches for time series momentum training.
    
    This mirrors the TMT approach of creating overlapping windows from
    continuous time series data, where each window contains:
    - X: input features (normalized returns, MACD, etc.)
    - y: target forward returns
    
    Args:
        df: DataFrame with features (must have DatetimeIndex)
        window_size: Number of timesteps in each window (e.g., 21 for ~1 month)
        target_col: Name of target column (e.g., 'target_returns')
        feature_cols: List of feature column names
        stride: Step size for sliding window (default=1, overlapping windows)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        target_col: str = 'target_returns',
        feature_cols: Optional[list] = None,
        stride: int = 1
    ):
        self.df = df.copy()
        self.window_size = window_size
        self.target_col = target_col
        self.stride = stride
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            self.feature_cols = [c for c in df.columns if c != target_col]
        else:
            self.feature_cols = feature_cols
            
        self.num_features = len(self.feature_cols)
        
    def create_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows from the time series.
        
        Returns:
            X: (num_windows, window_size, num_features)
            y: (num_windows, window_size, 1) - target returns
        """
        data = self.df[self.feature_cols].values  # (T, F)
        targets = self.df[self.target_col].values.reshape(-1, 1)  # (T, 1)
        
        T = len(data)
        num_windows = (T - self.window_size) // self.stride + 1
        
        X = np.zeros((num_windows, self.window_size, self.num_features))
        y = np.zeros((num_windows, self.window_size, 1))
        
        for i in range(num_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            X[i] = data[start_idx:end_idx]
            y[i] = targets[start_idx:end_idx]
            
        return X, y
    
    def to_tf_dataset(self, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        """Convert to TensorFlow Dataset for training.
        
        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle windows
            
        Returns:
            tf.data.Dataset ready for model.fit()
        """
        X, y = self.create_windows()
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
            
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset


def create_train_val_test_datasets(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    window_size: int = 21,
    target_col: str = 'target_returns',
    feature_cols: Optional[list] = None,
    batch_size: int = 32
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create train/val/test datasets with temporal split.
    
    Args:
        df: Full DataFrame with DatetimeIndex
        train_end: End date for training (e.g., '2020-12-31')
        val_end: End date for validation (e.g., '2022-12-31')
        window_size: Window size for sequences
        target_col: Target column name
        feature_cols: Feature column names
        batch_size: Batch size
        
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Temporal split - NO shuffle to preserve time ordering
    train_df = df[df.index <= train_end]
    val_df = df[(df.index > train_end) & (df.index <= val_end)]
    test_df = df[df.index > val_end]
    
    # Create window datasets
    train_ds = TimeSeriesWindowDataset(
        train_df, window_size, target_col, feature_cols
    ).to_tf_dataset(batch_size=batch_size, shuffle=True)
    
    val_ds = TimeSeriesWindowDataset(
        val_df, window_size, target_col, feature_cols
    ).to_tf_dataset(batch_size=batch_size, shuffle=False)
    
    test_ds = TimeSeriesWindowDataset(
        test_df, window_size, target_col, feature_cols
    ).to_tf_dataset(batch_size=batch_size, shuffle=False)
    
    return train_ds, val_ds, test_ds
