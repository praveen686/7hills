#!/usr/bin/env python
"""
AFML Meta-Labeling Training Script.
Trains a secondary 'Confidence' model to predict whether the primary model's signal will be profitable.
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
from config.train_config import MODEL_CONFIG, DATA_CONFIG

# --- Memory Safeguards ---
import gc
# Enable memory growth to prevent TF from hogging all VRAM immediately
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Memory growth config error: {e}")

# --- Timezone Normalization (Same robust logic as train_afml.py) ---
MARKET_TZ = "Asia/Kolkata"
def _as_dt_index(x):
    di = pd.to_datetime(x, errors="raise")
    return pd.DatetimeIndex(di)

def normalize_dt_index(idx, source_tz=MARKET_TZ, target_tz=MARKET_TZ):
    di = _as_dt_index(idx)
    if di.tz is None: di = di.tz_localize(source_tz)
    else: di = di.tz_convert(target_tz)
    di = di.tz_convert(target_tz).tz_localize(None)
    return di

def normalize_df_time(df, time_col=None, source_tz=MARKET_TZ, target_tz=MARKET_TZ):
    df = df.copy()
    if time_col is None:
        df.index = normalize_dt_index(df.index, source_tz, target_tz)
        return df
    di = _as_dt_index(df[time_col].values)
    if di.tz is None: di = di.tz_localize(source_tz)
    else: di = di.tz_convert(target_tz)
    df[time_col] = di.tz_convert(target_tz).tz_localize(None)
    return df
# -------------------------------------------------------------------

def load_data_and_predictions(data_cfg, primary_model_path):
    """
    Loads data, runs primary model inference, and generates meta-labels.
    Meta-Label: 1 if Primary Signal matches True Direction (and is non-zero), 0 otherwise.
    For Regression Primary Model: 
      - Signal = sign(pred_ret)
      - Meta-Label = 1 if sign(pred_ret) == sign(true_ret) AND |true_ret| > threshold?
      - Or strictly based on Triple Barrier 'label_bin' (-1, 0, 1).
      - If pred > 0 and label_bin == 1 -> Correct (1)
      - If pred < 0 and label_bin == -1 -> Correct (1)
      - Else -> Incorrect (0)
    """
    if not os.path.exists(primary_model_path):
        raise FileNotFoundError(f"Primary model not found at {primary_model_path}")
        
    print(f"Loading Primary Model: {primary_model_path}...", flush=True)
    
    # Explicitly register custom objects to avoid serialization issues
    from quantkubera.models.layers import (
        GluLayer, GatedResidualNetwork, VariableSelectionNetwork, 
        InterpretableMultiHeadAttention, ScaledDotProductAttention
    )
    from quantkubera.models.losses import SharpeLoss
    
    custom_objects = {
        'MomentumTransformer': MomentumTransformer,
        'GluLayer': GluLayer,
        'GatedResidualNetwork': GatedResidualNetwork,
        'VariableSelectionNetwork': VariableSelectionNetwork,
        'InterpretableMultiHeadAttention': InterpretableMultiHeadAttention,
        'ScaledDotProductAttention': ScaledDotProductAttention,
        'SharpeLoss': SharpeLoss
    }
    
    primary_model = keras.models.load_model(primary_model_path, custom_objects=custom_objects, compile=False)
    
    afml_files = glob.glob('data/afml_events/*.csv')
    engineer = FeatureEngineer()
    
    all_X = []
    all_meta_y = []
    
    window_size = data_cfg['window_size']
    feature_cols = data_cfg['feature_cols']
    
    ticker_files = {}
    for f in afml_files:
        ticker = Path(f).stem.replace('_afml', '')
        ticker_files[ticker] = f
        
    print(f"Generating Meta-Labels for {len(ticker_files)} tickers...", flush=True)
    
    processed_count = 0
    for ticker, f in ticker_files.items():
        raw_f = f'data/raw/{ticker}.csv'
        if not os.path.exists(raw_f):
            found = glob.glob(f'data/raw/**/{ticker}.csv', recursive=True)
            if found: raw_f = found[0]
            else: continue
            
        try:
            # Load & Normalize
            event_meta = normalize_df_time(pd.read_csv(f, index_col=0, parse_dates=True))
            df = normalize_df_time(pd.read_csv(raw_f, parse_dates=['date']).set_index('date'))
            
            # Features
            df = engineer.process_ticker(df)
            df = normalize_df_time(df)
            df = engineer.add_volatility(df, window=20)
            df = engineer.add_volatility(df, window=60)
            df = normalize_df_time(df)
            
            # CPD (Inlined)
            cpd_path = f'data/cpd/{ticker}_cpd_21.csv'
            if os.path.exists(cpd_path):
                cpd_df = normalize_df_time(pd.read_csv(cpd_path, index_col=0, parse_dates=True))
                cpd_aligned = cpd_df.reindex(df.index)
                df['cp_rl_21'] = cpd_aligned['cp_location_norm']
                df['cp_score_21'] = cpd_aligned['cp_score']
            else:
                df['cp_rl_21'] = 0.0
                df['cp_score_21'] = 0.0
            df = normalize_df_time(df)
            
            # Extract Windows for this ticker
            ticker_X = []
            ticker_labels = [] # label_bin (-1, 0, 1)
            
            for t_event in event_meta.index:
                if t_event not in df.index: continue
                idx_obj = df.index.get_loc(t_event)
                idx = idx_obj.start if isinstance(idx_obj, slice) else (idx_obj[0] if isinstance(idx_obj, np.ndarray) else idx_obj)
                
                if idx < window_size - 1: continue
                
                window_x = df.iloc[idx - window_size + 1 : idx + 1][feature_cols].values
                target_bin = event_meta.loc[t_event, 'label_bin'] # -1, 0, 1
                
                if np.isnan(window_x).any() or np.isnan(target_bin): continue
                
                ticker_X.append(window_x.astype(np.float32))
                ticker_labels.append(target_bin)
                
            if not ticker_X: continue
            
            # Batch Predict with Primary Model
            ticker_X = np.stack(ticker_X)
            ticker_labels = np.array(ticker_labels)
            
            # Get Primary Predictions (Returns)
            # Use a smaller batch size for inference to avoid OOM
            preds = primary_model.predict(ticker_X, batch_size=512, verbose=0)
            
            # preds shape is (N, time_steps, 1). We need the last step prediction.
            last_step_preds = preds[:, -1, :]
            
            # Generate Meta-Labels
            # Logic: If Primary says Long (>0) and Label is Long (1) -> 1
            #        If Primary says Short (<0) and Label is Short (-1) -> 1
            #        Else -> 0 (Incorrect side or flat)
            
            primary_side = np.sign(last_step_preds).flatten() # 1.0, -1.0, 0.0
            # Meta Label = 1 if sides match AND label is not 0
            # We treat 0 label as "No Trade" or "Loss" if we took a position.
            
            meta_y = np.zeros_like(primary_side)
            
            # Correct Long: Pred > 0 and Label == 1
            meta_y[(primary_side > 0) & (ticker_labels == 1)] = 1
            
            # Correct Short: Pred < 0 and Label == -1
            meta_y[(primary_side < 0) & (ticker_labels == -1)] = 1
            
            # Append features + primary_pred for Secondary Model?
            # Standard stacking: Secondary Model takes X + [Primary_Pred]
            # Primary pred has shape (N, 1). We need to broadcast or concat?
            # TMT takes sequence input [N, T, F].
            # A simple approach for secondary model:
            # 1. Use same TMT architecture.
            # 2. Add 'primary_pred' as a new feature to every timestep? Or just use raw features?
            # Lopez de Prado suggests trained on the same features X.
            # "The secondary model is trained on (X, y_meta)".
            
            all_X.append(ticker_X)
            all_meta_y.append(meta_y)
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(ticker_files)}...", flush=True)
                # Safeguards
                gc.collect()
                keras.backend.clear_session() 
                
        except Exception as e:
            print(f"Error {ticker}: {e}")
            continue

    X = np.concatenate(all_X)
    y = np.concatenate(all_meta_y).astype(np.float32)
    
    # Reshape y to (N, 1) so timeseries_dataset yields (batch, time, 1)
    # matching the model output (batch, time, 1)
    y = np.expand_dims(y, axis=-1)
    
    print(f"Total Training Data: X={X.shape}, y={y.shape}")
    # Check class balance
    print(f"Meta-Label Balance: {np.mean(y):.4f} (Positive Rate)")
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--primary-model', required=True, help='Path to saved primary model')
    parser.add_argument('--output-dir', default='models/afml_meta_v1')
    args = parser.parse_args()
    
    # Load Data & Generate Labels
    X, y = load_data_and_predictions(DATA_CONFIG, args.primary_model)
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    # Split
    num_val = int(len(X) * 0.15)
    train_ds = tf.data.Dataset.from_tensor_slices((X[:-num_val], y[:-num_val])).batch(args.batch_size).shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X[-num_val:], y[-num_val:])).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Model: Same TMT backbone
    backbone = MomentumTransformer(
        time_steps=DATA_CONFIG['window_size'],
        input_size=len(DATA_CONFIG['feature_cols']),
        output_size=1, # Probability
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_heads=MODEL_CONFIG['num_heads'],
        dropout_rate=0.4
    )
    
    # Functional Wrapper to match target shape (Batch, 1)
    inputs = keras.Input(shape=(DATA_CONFIG['window_size'], len(DATA_CONFIG['feature_cols'])))
    x = backbone(inputs)
    # Select last timestep: (Batch, Time, 1) -> (Batch, 1)
    outputs = keras.layers.Lambda(lambda z: z[:, -1, :])(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', keras.metrics.AUC(name='auc')])
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.output_dir, 'best_meta_model.keras'), save_best_only=True, monitor='val_auc', mode='max'),
        keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max')
    ]
    
    print("\nStarting Meta-Model Training...", flush=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    
    model.save(os.path.join(args.output_dir, 'final_meta_model.keras'))
    print("Meta-Model Training Complete!")

if __name__ == '__main__':
    main()
