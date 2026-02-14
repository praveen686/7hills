#!/usr/bin/env python
"""
AFML Evaluation Script.
Compares:
1. Baseline (Primary Model Only)
2. Meta-Strategy (Primary * Meta-Confidence)
3. Buy & Hold
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
from pathlib import Path
import matplotlib.pyplot as plt

# Add src and config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantkubera.features.build_features import FeatureEngineer
from config.train_config import MODEL_CONFIG, DATA_CONFIG

# Custom Object Loading
from quantkubera.models.tft import MomentumTransformer
from quantkubera.models.layers import (
    GluLayer, GatedResidualNetwork, VariableSelectionNetwork, 
    InterpretableMultiHeadAttention, ScaledDotProductAttention
)
from quantkubera.models.losses import SharpeLoss

def load_model_robust(path):
    print(f"Loading model: {path}...", flush=True)
    custom_objects = {
        'MomentumTransformer': MomentumTransformer,
        'GluLayer': GluLayer,
        'GatedResidualNetwork': GatedResidualNetwork,
        'VariableSelectionNetwork': VariableSelectionNetwork,
        'InterpretableMultiHeadAttention': InterpretableMultiHeadAttention,
        'ScaledDotProductAttention': ScaledDotProductAttention,
        'SharpeLoss': SharpeLoss
    }
    return keras.models.load_model(path, custom_objects=custom_objects, compile=False)

# Robust Timezone Logic
MARKET_TZ = "Asia/Kolkata"
def normalize_df_time(df, time_col=None, source_tz=MARKET_TZ, target_tz=MARKET_TZ):
    df = df.copy()
    if time_col is None:
        di = pd.DatetimeIndex(pd.to_datetime(df.index, errors='raise'))
    else:
        di = pd.DatetimeIndex(pd.to_datetime(df[time_col].values, errors='raise'))
        
    if di.tz is None: di = di.tz_localize(source_tz)
    else: di = di.tz_convert(target_tz)
    
    if time_col is None: df.index = di.tz_convert(target_tz).tz_localize(None)
    else: df[time_col] = di.tz_convert(target_tz).tz_localize(None)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', required=True, help='Path to Primary Model')
    parser.add_argument('--meta', required=True, help='Path to Meta Model')
    parser.add_argument('--ticker', default='NIFTY', help='Ticker to evaluate (or "all")')
    args = parser.parse_args()
    
    # Load Models
    primary_model = load_model_robust(args.primary)
    meta_model = load_model_robust(args.meta)
    
    engineer = FeatureEngineer()
    tickers = [args.ticker] if args.ticker != 'all' else [Path(f).stem.replace('_afml','') for f in glob.glob('data/afml_events/*.csv')]
    
    results = []
    
    for ticker in tickers[:10]: # Limit for quick check
        print(f"Evaluating {ticker}...", flush=True)
        try:
            raw_f = f'data/raw/{ticker}.csv'
            if not os.path.exists(raw_f): 
                # Try recursive
                found = glob.glob(f'data/raw/**/{ticker}.csv', recursive=True)
                if found: raw_f = found[0]
                else: continue
                
            df = normalize_df_time(pd.read_csv(raw_f, parse_dates=['date']).set_index('date'))
            df['Close_Raw'] = df['close']
            
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
            
            # Prepare Sequences
            window_size = DATA_CONFIG['window_size']
            feature_cols = DATA_CONFIG['feature_cols']
            
            X_arr = []
            dates = []
            
            # Evaluate on last 20% ? Or full?
            # Let's do full timeline to see cumsum
            # Step size 1 for full curve
            
            # Optimize: Vectorized prep?
            # Rolling window view
            vals = df[feature_cols].values
            # Stride tricks for rolling
            # (N, window, F)
            # This is memory heavy. Let's do batching or just accept it's slow.
            # Or use `timeseries_dataset_from_array`
            
            ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                vals, None, sequence_length=window_size, batch_size=4096, shuffle=False
            )
            
            all_preds_primary = []
            all_preds_meta = []
            
            for batch in ds:
                # Primary: (batch, time, 1) -> take last [-1]
                p = primary_model.predict(batch, verbose=0)[:, -1, :]
                # Meta: (batch, time, 1) -> take last [-1]
                # Wait, meta model output size is 1. TMT returns (batch, time, 1).
                # We need probability. Did I add sigmoid?
                # In train_meta.py: output_size=1, from_logits=True in loss.
                # So output is Logits. We need sigmoid.
                m = meta_model.predict(batch, verbose=0)[:, -1, :]
                
                all_preds_primary.append(p)
                all_preds_meta.append(m)
                
            y_primary = np.concatenate(all_preds_primary).flatten()
            logits_meta = np.concatenate(all_preds_meta).flatten()
            y_meta = tf.sigmoid(logits_meta).numpy()
            
            # Re-align dates
            # data starts at index `window_size-1`
            valid_dates = df.index[window_size-1:]
            
            # Trim if dataset dropped last batch? usually strict.
            valid_len = len(y_primary)
            valid_dates = valid_dates[:valid_len]
            close_prices = df['Close_Raw'].iloc[window_size-1 : window_size-1+valid_len]
            returns = close_prices.pct_change().fillna(0)
            
            # Strategy Returns
            # Signal: sign(y_primary)
            signal = np.sign(y_primary)
            
            # Primary Strategy
            strat_primary = signal * returns.shift(-1) # Forward return
            
            # Meta Strategy: Size by Confidence
            # If Confidence > 0.6 -> Size 1, else 0? Or continuous?
            # Let's try simple threshold
            size = (y_meta > 0.55).astype(float)
            strat_meta = signal * size * returns.shift(-1)
            
            # Metrics
            def get_sharpe(r):
                if r.std() == 0: return 0
                return r.mean() / r.std() * np.sqrt(252)
            
            sharpe_base = get_sharpe(strat_primary)
            sharpe_meta = get_sharpe(strat_meta)
            
            print(f"{ticker}: Base Sharpe={sharpe_base:.2f}, Meta Sharpe={sharpe_meta:.2f}")
            results.append({
                'Ticker': ticker,
                'Base_Sharpe': sharpe_base,
                'Meta_Sharpe': sharpe_meta
            })
            
        except Exception as e:
            print(f"Error {ticker}: {e}")
            
    if results:
        res_df = pd.DataFrame(results)
        print("\n--- Summary ---")
        print(res_df.describe())
        print(f"Mean Base: {res_df['Base_Sharpe'].mean():.2f}")
        print(f"Mean Meta: {res_df['Meta_Sharpe'].mean():.2f}")

if __name__ == '__main__':
    main()
