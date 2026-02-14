#!/usr/bin/env python
"""
AFML-Native Training Script.
Trains the Momentum Transformer on CUSUM-sampled events with Triple Barrier Labels.
Uses temporal windows leading up to each event.
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
import traceback

# Add src and config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantkubera.features.build_features import FeatureEngineer
from quantkubera.models.tft import MomentumTransformer
from quantkubera.training.trainer import MomentumTrainer
from config.train_config import MODEL_CONFIG, DATA_CONFIG, TRAIN_CONFIG
from quantkubera.models.losses import SharpeLoss

MARKET_TZ = "Asia/Kolkata"

def _as_dt_index(x):
    # Always returns a DatetimeIndex or raises.
    di = pd.to_datetime(x, errors="raise")
    return pd.DatetimeIndex(di)

def normalize_dt_index(idx, source_tz=MARKET_TZ, target_tz=MARKET_TZ):
    """
    Canonicalize timestamps to target_tz wall-time, then strip tz -> tz-naive.
    Handles tz-aware and tz-naive inputs deterministically.
    """
    di = _as_dt_index(idx)

    if di.tz is None:
        di = di.tz_localize(source_tz)      # assume naive means source_tz
    else:
        di = di.tz_convert(target_tz)

    di = di.tz_convert(target_tz).tz_localize(None)

    # hard guard
    if di.tz is not None:
        raise ValueError("normalize_dt_index failed: index still tz-aware")

    return di

def normalize_df_time(df, time_col=None, source_tz=MARKET_TZ, target_tz=MARKET_TZ):
    """
    Normalize dataframe time axis for safe merging.
    - If time_col is None: normalize df.index
    - Else: normalize df[time_col] and (optionally) set index
    """
    df = df.copy()

    if time_col is None:
        df.index = normalize_dt_index(df.index, source_tz, target_tz)
        assert df.index.tz is None
        return df

    # normalize merge key column
    di = _as_dt_index(df[time_col].values)
    if di.tz is None:
        di = di.tz_localize(source_tz)
    else:
        di = di.tz_convert(target_tz)
    df[time_col] = di.tz_convert(target_tz).tz_localize(None)
    return df

def debug_time(df, name):
    idx = df.index
    print(f"{name}: tz={idx.tz}, dtype={idx.dtype}, "
          f"min={idx.min()}, max={idx.max()}, n={len(idx)}", flush=True)

def load_afml_sequence_data(data_cfg):
    """
    Loads raw history but only samples windows ending at AFML event timestamps.
    """
    afml_files = glob.glob('data/afml_events/*.csv')
    if not afml_files:
        raise FileNotFoundError("No AFML events found. Run prepare_afml_dataset.py first.")
    
    engineer = FeatureEngineer()
    all_X = []
    all_y = []
    
    window_size = data_cfg['window_size']
    feature_cols = data_cfg['feature_cols']
    
    ticker_files = {}
    for f in afml_files:
        ticker = Path(f).stem.replace('_afml', '')
        ticker_files[ticker] = f

    print(f"Loading AFML sequences for {len(ticker_files)} tickers...", flush=True)
    
    processed_count = 0
    for ticker, f in ticker_files.items():
        # Find raw file
        raw_f = f'data/raw/{ticker}.csv'
        if not os.path.exists(raw_f):
            found_files = glob.glob(f'data/raw/**/{ticker}.csv', recursive=True)
            if found_files:
                raw_f = found_files[0]
            else:
                continue 
            
        try:
            # 1. Load AFML event metadata
            event_meta = pd.read_csv(f, index_col=0, parse_dates=True)
            event_meta = normalize_df_time(event_meta)
            
            # 2. Load full raw history (Cast to naive immediately)
            df = pd.read_csv(raw_f, parse_dates=['date']).set_index('date')
            df = normalize_df_time(df)
            
            if ticker == '360ONE': # Debug probe for failing ticker
                 print(f"--- DEBUG: {ticker} ---")
                 debug_time(event_meta, "event_meta")
                 debug_time(df, "raw_df_norm")

            # Process features (Returns, MACD)
            df = engineer.process_ticker(df)
            df = normalize_df_time(df)
            
            # Volatility
            df = engineer.add_volatility(df, window=20)
            df = engineer.add_volatility(df, window=60)
            df = normalize_df_time(df)
            
            # CPD Features (Inlined for safety and normalization control)
            cpd_path = f'data/cpd/{ticker}_cpd_21.csv'
            if os.path.exists(cpd_path):
                cpd_df = pd.read_csv(cpd_path, index_col=0, parse_dates=True)
                cpd_df = normalize_df_time(cpd_df)
                
                if ticker == '360ONE':
                    debug_time(cpd_df, "cpd_df_norm")
                    print("overlap:", len(df.index.intersection(cpd_df.index)))
                
                # Reindex to match main df (both are robustly naive now)
                cpd_aligned = cpd_df.reindex(df.index)
                df['cp_rl_21'] = cpd_aligned['cp_location_norm']
                df['cp_score_21'] = cpd_aligned['cp_score']
            else:
                 # Fill with defaults if missing, to prevent dimension errors
                 df['cp_rl_21'] = 0.0
                 df['cp_score_21'] = 0.0
            
            # Final safety norm (though should be redundant)
            df = normalize_df_time(df)

            # Align features with events
            for t_event in event_meta.index:
                try:
                    if t_event not in df.index:
                        continue
                        
                    idx_obj = df.index.get_loc(t_event)
                    if isinstance(idx_obj, slice):
                         idx = idx_obj.start
                    elif isinstance(idx_obj, np.ndarray):
                         if idx_obj.dtype == 'bool':
                             idx = np.where(idx_obj)[0][0]
                         else:
                             idx = idx_obj[0]
                    else:
                        idx = idx_obj
                    
                    if idx < window_size - 1:
                        continue
                    
                    window_x = df.iloc[idx - window_size + 1 : idx + 1][feature_cols].values
                    target_y = event_meta.loc[t_event, 'label_ret']
                    
                    if np.isnan(window_x).any() or np.isnan(target_y):
                        continue
                        
                    all_X.append(window_x.astype(np.float32))
                    all_y.append(np.full((window_size, 1), target_y, dtype=np.float32))
                except Exception as e:
                    continue
            
            if not all_X and ticker == '360ONE':
                raise ValueError(f"{ticker}: df empty after merges or no overlap; likely time alignment issue")

            processed_count += 1
            if processed_count % 20 == 0:
                print(f"Processed {processed_count}/{len(ticker_files)} tickers... (Events: {len(all_X)})", flush=True)
                gc.collect()
                
        except Exception as e:
            print(f"Error loading {ticker}: {e}", flush=True)
            if ticker == '360ONE': # Fail fast for debugging
                 raise e
            continue

    if not all_X:
        raise ValueError("No valid sequences found. Check data alignment.")

    X = np.stack(all_X)
    y = np.stack(all_y)
    print(f"AFML Dataset Ready. Total Windows: {len(X)}", flush=True)
    return X, y

def main():
    parser = argparse.ArgumentParser(description='AFML-Native Training (Phase 3)')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--model-dir', default='models/afml_primary_v1')
    args = parser.parse_args()

    # Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    
    print("\n" + "="*50, flush=True)
    print("STEP 1/3: AFML EVENT DATA LOADING", flush=True)
    print("="*50, flush=True)
    X, y = load_afml_sequence_data(DATA_CONFIG)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    num_val = int(len(X) * 0.15)
    train_ds = tf.data.Dataset.from_tensor_slices((X[:-num_val], y[:-num_val])).batch(args.batch_size).shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X[-num_val:], y[-num_val:])).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    print("\n" + "="*50, flush=True)
    print("STEP 2/3: AFML MODEL INITIALIZATION", flush=True)
    print("="*50, flush=True)
    model = MomentumTransformer(
        time_steps=DATA_CONFIG['window_size'],
        input_size=len(DATA_CONFIG['feature_cols']),
        output_size=1,
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_heads=MODEL_CONFIG['num_heads'],
        dropout_rate=0.2
    )
    
    # Build explicitly
    dummy_x = tf.zeros((1, DATA_CONFIG['window_size'], len(DATA_CONFIG['feature_cols'])))
    _ = model(dummy_x)
    
    trainer = MomentumTrainer(model=model, learning_rate=5e-4, model_dir=args.model_dir)
    trainer.model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4, clipnorm=0.5), loss=SharpeLoss())

    extra_callbacks = [
        keras.callbacks.BackupAndRestore(backup_dir=os.path.join(args.model_dir, 'backup')),
        keras.callbacks.TerminateOnNaN()
    ]

    print("\n" + "="*50, flush=True)
    print("STEP 3/3: AFML PRIMARY TRAINING", flush=True)
    print("="*50, flush=True)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds, epochs=args.epochs, patience=args.patience, extra_callbacks=extra_callbacks)
    
    trainer.save_model(os.path.join(args.model_dir, 'afml_primary_model.keras'))
    print("AFML Primary Training Complete!", flush=True)

if __name__ == '__main__':
    main()
