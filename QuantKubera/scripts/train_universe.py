#!/usr/bin/env python
"""
Robust Master Training Script for Universe-Wide Momentum Transformer.
Features:
- Batched local data loading (212 tickers)
- Memory management (GC, Keras session cleanup)
- Checkpointing & Fail-over recovery (BackupAndRestore)
- Multi-ticker dataset creation
- Numerical stability guardrails
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gc
from pathlib import Path
import glob
import time

# Add src and config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantkubera.features.build_features import FeatureEngineer
from quantkubera.models.tft import MomentumTransformer
from quantkubera.training.dataset import TimeSeriesWindowDataset
from quantkubera.training.trainer import MomentumTrainer
from config.train_config import MODEL_CONFIG, DATA_CONFIG, TRAIN_CONFIG

def load_universe_data(data_cfg):
    """Load and process all available tickers from data/raw."""
    raw_files = glob.glob('data/raw/**/*.csv', recursive=True)
    if not raw_files:
        raise FileNotFoundError("No raw data found. Run data acquisition first.")
    
    engineer = FeatureEngineer()
    all_X = []
    all_y = []
    
    train_end = data_cfg['train_end']
    val_end = data_cfg['val_end']
    
    print(f"Processing universe: {len(raw_files)} tickers...")
    
    processed_count = 0
    for f in raw_files:
        ticker = Path(f).stem
        try:
            # Load raw data
            df = pd.read_csv(f, parse_dates=['date'])
            df.set_index('date', inplace=True)
            
            # Feature processing
            df = engineer.process_ticker(df)
            df = engineer.add_volatility(df, window=20)
            df = engineer.add_volatility(df, window=60)
            df = engineer.add_cpd_features(df, ticker=ticker, lookback=21)
            
            # Drop NaNs
            df = df.dropna(subset=data_cfg['feature_cols'] + [data_cfg['target_col']])
            
            if len(df) < data_cfg['window_size'] + 10:
                continue
                
            # Create windows for this ticker
            window_gen = TimeSeriesWindowDataset(
                df=df,
                window_size=data_cfg['window_size'],
                target_col=data_cfg['target_col'],
                feature_cols=data_cfg['feature_cols']
            )
            X, y = window_gen.create_windows()
            
            all_X.append(X)
            all_y.append(y)
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(raw_files)} tickers... (Memory cleanup)")
                gc.collect()
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    # Combine all
    print("Stacking universe datasets...")
    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    
    print(f"Total Universe Samples: {X_final.shape[0]}")
    return X_final, y_final

def main():
    parser = argparse.ArgumentParser(description='Universe-Wide Training with Safeguards')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--model-dir', default='models/master_model')
    args = parser.parse_args()

    # 1. GPU Check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 Found GPUs: {gpus}")
        # Enable memory growth to prevent pre-allocation crash
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️ No GPU found. Training on CPU will be very slow.")

    # 2. Data Preparation
    print("\n" + "="*50)
    print("STEP 1/3: UNIVERSE DATA PREPARATION")
    print("="*50)
    
    X, y = load_universe_data(DATA_CONFIG)
    
    # Shuffle universe once before split or use temporal split?
    # For a cross-sectional model, we often shuffle across tickers but keep time order for testing.
    # Here we perform a simple random shuffle for the ML baseline across the mixed universe.
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    # Simple split
    num_val = int(len(X) * 0.15)
    num_test = int(len(X) * 0.1)
    num_train = len(X) - num_val - num_test
    
    X_train, y_train = X[:num_train], y[:num_train]
    X_val, y_val = X[num_train:num_train+num_val], y[num_train:num_train+num_val]
    X_test, y_test = X[num_train+num_val:], y[num_train+num_val:]
    
    # Convert to TF datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(args.batch_size).shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # 3. Model & Trainer
    print("\n" + "="*50)
    print("STEP 2/3: MODEL INITIALIZATION")
    print("="*50)
    
    model = MomentumTransformer(
        time_steps=DATA_CONFIG['window_size'],
        input_size=len(DATA_CONFIG['feature_cols']),
        output_size=1,
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_heads=MODEL_CONFIG['num_heads'],
        dropout_rate=MODEL_CONFIG['dropout_rate']
    )

    # Build model (Required for BackupAndRestore callback)
    dummy_input = tf.zeros((1, DATA_CONFIG['window_size'], len(DATA_CONFIG['feature_cols'])))
    _ = model(dummy_input)
    
    # Custom Optimizer with Gradient Clipping (Guardrail)
    optimizer = keras.optimizers.Adam(learning_rate=TRAIN_CONFIG['learning_rate'], clipnorm=1.0)
    
    trainer = MomentumTrainer(
        model=model,
        learning_rate=TRAIN_CONFIG['learning_rate'],
        model_dir=args.model_dir
    )
    # Overwrite default compile with our clipped optimizer
    trainer.model.compile(optimizer=optimizer, loss=trainer.model.loss)

    # 4. Enhanced Callbacks (Fail-safe)
    os.makedirs(os.path.join(args.model_dir, 'backup'), exist_ok=True)
    
    extra_callbacks = [
        # Resumes training from last saved epoch if script restarts
        keras.callbacks.BackupAndRestore(backup_dir=os.path.join(args.model_dir, 'backup')),
        # Stops training if we hit NaNs
        keras.callbacks.TerminateOnNaN(),
        # Save checkpoints frequently
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_dir, 'ckpt_epoch_{epoch:02d}.keras'),
            save_freq='epoch',
            save_weights_only=False
        )
    ]

    # 5. Training
    print("\n" + "="*50)
    print("STEP 3/3: MASTER UNIVERSE TRAINING")
    print("="*50)
    
    trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        patience=args.patience,
        extra_callbacks=extra_callbacks
    )

    # 6. Final Evaluation
    print("\nTraining Complete. Evaluating on Universe Test Set...")
    results = trainer.evaluate(test_ds)
    print(f"Test Set Sharpe Performance: {results}")
    
    trainer.save_model(os.path.join(args.model_dir, 'universe_master_model.keras'))

if __name__ == '__main__':
    main()
