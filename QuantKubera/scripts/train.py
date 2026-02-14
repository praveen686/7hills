#!/usr/bin/env python
"""End-to-end training script for Momentum Transformer."""
import os
import sys
import argparse
import pandas as pd
import tensorflow as tf

# Add src and config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantkubera.data.kite_fetcher import KiteFetcher
from quantkubera.features.build_features import FeatureEngineer
from quantkubera.models.tft import MomentumTransformer
from quantkubera.training.dataset import create_train_val_test_datasets
from quantkubera.training.trainer import MomentumTrainer

from config.train_config import MODEL_CONFIG, DATA_CONFIG, TRAIN_CONFIG, SMOKE_TEST_CONFIG


def main(smoke_test: bool = False):
    """Run training pipeline.
    
    Args:
        smoke_test: If True, use smoke test config (1 year NIFTY only)
    """
    # Load configuration
    if smoke_test:
        print("=" * 80)
        print("SMOKE TEST MODE - Training on 1 year NIFTY data")
        print("=" * 80)
        model_cfg = SMOKE_TEST_CONFIG['model_config']
        train_cfg = SMOKE_TEST_CONFIG['train_config']
        data_cfg = DATA_CONFIG.copy()
        data_cfg.update({
            'train_end': SMOKE_TEST_CONFIG['train_end'],
            'val_end': SMOKE_TEST_CONFIG['val_end'],
            'tickers': SMOKE_TEST_CONFIG['tickers'],
        })
    else:
        model_cfg = MODEL_CONFIG
        train_cfg = TRAIN_CONFIG
        data_cfg = DATA_CONFIG
    
    print(f"\nConfiguration:")
    print(f"  Tickers: {data_cfg['tickers']}")
    print(f"  Train end: {data_cfg['train_end']}")
    print(f"  Val end: {data_cfg['val_end']}")
    print(f"  Model: hidden={model_cfg['hidden_size']}, heads={model_cfg['num_heads']}")
    
    # ========================================================================
    # STEP 1: Data Fetching
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Fetching Data from Kite")
    print("=" * 80)
    
    fetcher = KiteFetcher()
    all_data = []
    
    for ticker in data_cfg['tickers']:
        print(f"\nFetching {ticker}...")
        df = fetcher.fetch_continuous_futures(
            ticker=ticker,
            start_date='2010-01-01',  # Fetch more than needed
            end_date='2024-12-31'
        )
        df['ticker'] = ticker
        all_data.append(df)
    
    # Combine all tickers
    combined_df = pd.concat(all_data, ignore_index=False)
    print(f"\nTotal data points: {len(combined_df)}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    # ========================================================================
    # STEP 2: Feature Engineering
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Building Features")
    print("=" * 80)
    
    engineer = FeatureEngineer()
    features_df = engineer.process_ticker(combined_df)
    
    # Add volatility features to match config
    features_df = engineer.add_volatility(features_df, window=20)
    features_df = engineer.add_volatility(features_df, window=60)
    
    # Add CPD features if available
    for ticker in data_cfg['tickers']:
        ticker_mask = features_df['ticker'] == ticker
        if ticker_mask.any():
            ticker_df = features_df[ticker_mask].copy()
            ticker_df = engineer.add_cpd_features(ticker_df, ticker=ticker, lookback=21)
            features_df.loc[ticker_mask, :] = ticker_df
    
    print(f"Features created: {list(features_df.columns)}")
    print(f"Feature matrix shape: {features_df.shape}")
    
    # Drop NaN rows from feature calculation
    features_df = features_df.dropna()
    print(f"After dropping NaN: {features_df.shape}")
    
    # ========================================================================
    # STEP 3: Create Train/Val/Test Datasets
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Creating Train/Val/Test Datasets")
    print("=" * 80)
    
    train_ds, val_ds, test_ds = create_train_val_test_datasets(
        df=features_df,
        train_end=data_cfg['train_end'],
        val_end=data_cfg['val_end'],
        window_size=data_cfg['window_size'],
        target_col=data_cfg['target_col'],
        feature_cols=data_cfg['feature_cols'],
        batch_size=train_cfg['batch_size']
    )
    
    print(f"✅ Datasets created")
    print(f"   Window size: {data_cfg['window_size']}")
    print(f"   Features: {len(data_cfg['feature_cols'])}")
    
    # ========================================================================
    # STEP 4: Instantiate Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Creating Momentum Transformer Model")
    print("=" * 80)
    
    model = MomentumTransformer(
        time_steps=data_cfg['window_size'],
        input_size=len(data_cfg['feature_cols']),
        output_size=1,  # Predicting position weights
        hidden_size=model_cfg['hidden_size'],
        num_heads=model_cfg['num_heads'],
        dropout_rate=model_cfg['dropout_rate']
    )
    
    # Build model to enable param counting
    dummy_input = tf.zeros((1, data_cfg['window_size'], len(data_cfg['feature_cols'])))
    _ = model(dummy_input)
    
    print(f"✅ Model created")
    print(f"   Parameters: ~{model.count_params():,} trainable params")
    
    # ========================================================================
    # STEP 5: Train Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Training Model")
    print("=" * 80)
    
    trainer = MomentumTrainer(
        model=model,
        learning_rate=train_cfg['learning_rate'],
        model_dir=train_cfg['model_dir']
    )
    
    history = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=train_cfg['epochs'],
        patience=train_cfg['early_stop_patience']
    )
    
    # ========================================================================
    # STEP 6: Evaluate on Test Set
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Evaluating on Test Set")
    print("=" * 80)
    
    test_results = trainer.evaluate(test_ds)
    print(f"\nTest Results:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # ========================================================================
    # STEP 7: Save Final Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Saving Model")
    print("=" * 80)
    
    trainer.save_model()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model saved to: {train_cfg['model_dir']}")
    print(f"TensorBoard logs: {os.path.join(train_cfg['model_dir'], 'logs')}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir={os.path.join(train_cfg['model_dir'], 'logs')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Momentum Transformer')
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Run smoke test (1 year NIFTY, small model)'
    )
    
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
