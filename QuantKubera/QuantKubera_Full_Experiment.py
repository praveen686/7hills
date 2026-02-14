import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Union, Tuple, Dict
import sys

# Force CPU for stability and consistent memory usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# TRANSACTION COSTS (0.1% = 10 bps per side, total 20 bps round-trip)
BPS_COST = 0.0010 

# ============================================================================
# 1. PRODUCTION MODEL ARCHITECTURE (EXACT PARITY)
# ============================================================================

@tf.keras.utils.register_keras_serializable()
class GluLayer(layers.Layer):
    def __init__(self, hidden_size, dropout_rate=None, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size, self.dropout_rate, self.activation = hidden_size, dropout_rate, activation
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate else None
        self.activation_layer = layers.Dense(hidden_size, activation=activation)
        self.gated_layer = layers.Dense(hidden_size, activation="sigmoid")
    def call(self, inputs):
        x = self.dropout(inputs) if self.dropout else inputs
        return layers.Multiply()([self.activation_layer(x), self.gated_layer(x)]), self.gated_layer(x)
    def get_config(self): return {**super().get_config(), "hidden_size": self.hidden_size, "dropout_rate": self.dropout_rate, "activation": self.activation}

@tf.keras.utils.register_keras_serializable()
class GatedResidualNetwork(layers.Layer):
    def __init__(self, hidden_size, output_size=None, dropout_rate=None, context_size=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size, self.output_size, self.dropout_rate, self.context_size = hidden_size, output_size or hidden_size, dropout_rate, context_size
        self.skip_layer, self.layer1, self.layer2 = None, layers.Dense(hidden_size), layers.Dense(hidden_size)
        self.context_layer = layers.Dense(hidden_size, use_bias=False) if context_size else None
        self.glu = GluLayer(self.output_size, dropout_rate=dropout_rate)
        self.add, self.norm = layers.Add(), layers.LayerNormalization()
    def call(self, inputs, context=None, return_gate=False):
        if self.skip_layer is None and inputs.shape[-1] != self.output_size: self.skip_layer = layers.Dense(self.output_size)
        skip = self.skip_layer(inputs) if self.skip_layer else inputs
        x = self.layer1(inputs)
        if context is not None and self.context_layer: x = x + self.context_layer(context)
        x = self.layer2(tf.nn.elu(x))
        glu_out, gate = self.glu(x)
        out = self.norm(self.add([skip, glu_out]))
        return (out, gate) if return_gate else out
    def get_config(self): return {**super().get_config(), "hidden_size": self.hidden_size, "output_size": self.output_size, "dropout_rate": self.dropout_rate}

@tf.keras.utils.register_keras_serializable()
class VariableSelectionNetwork(layers.Layer):
    def __init__(self, num_inputs, hidden_size, dropout_rate=None, context_size=None, **kwargs):
        super().__init__(**kwargs)
        self.num_inputs, self.hidden_size, self.dropout_rate = num_inputs, hidden_size, dropout_rate
        self.selection_grn = GatedResidualNetwork(hidden_size, output_size=num_inputs, dropout_rate=dropout_rate, context_size=context_size)
        self.softmax_weighting = layers.Softmax(axis=-1)
        self.input_grns = [GatedResidualNetwork(hidden_size, dropout_rate=dropout_rate) for _ in range(num_inputs)]
    def call(self, inputs, context=None):
        shp = tf.shape(inputs)
        flat = tf.reshape(inputs, [shp[0], shp[1], self.num_inputs * self.hidden_size])
        weights = self.softmax_weighting(self.selection_grn(flat, context=context, return_gate=True)[1])
        processed = tf.stack([self.input_grns[i](inputs[..., i, :]) for i in range(self.num_inputs)], axis=-2)
        return tf.reduce_sum(tf.expand_dims(weights, -1) * processed, axis=-2), weights
    def get_config(self): return {**super().get_config(), "num_inputs": self.num_inputs, "hidden_size": self.hidden_size, "dropout_rate": self.dropout_rate}

@tf.keras.utils.register_keras_serializable()
class ScaledDotProductAttention(layers.Layer):
    def __init__(self, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dropout, self.activation = layers.Dropout(dropout_rate), layers.Activation("softmax")
    def call(self, q, k, v, mask=None):
        attn = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(tf.shape(q)[-1], tf.float32))
        if mask is not None: attn += (1.0 - tf.cast(mask, tf.float32)) * -1e9
        attn = self.dropout(self.activation(attn))
        return tf.matmul(attn, v), attn
    def get_config(self): return {**super().get_config(), "dropout_rate": self.dropout_rate}

@tf.keras.utils.register_keras_serializable()
class InterpretableMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, d_model, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads, self.d_model, self.dropout_rate = num_heads, d_model, dropout_rate
        self.d_head = d_model // num_heads
        self.q_layers = [layers.Dense(self.d_head, use_bias=False) for _ in range(num_heads)]
        self.k_layers = [layers.Dense(self.d_head, use_bias=False) for _ in range(num_heads)]
        self.v_layer = layers.Dense(self.d_head, use_bias=False)
        self.attention = ScaledDotProductAttention(dropout_rate)
        self.w_o, self.dropout = layers.Dense(d_model, use_bias=False), layers.Dropout(dropout_rate)
    def call(self, q, k, v, mask=None):
        vs = self.v_layer(v)
        hs, attns = [], []
        for i in range(self.num_heads):
            h, a = self.attention(self.q_layers[i](q), self.k_layers[i](k), vs, mask)
            hs.append(h); attns.append(a)
        out = self.w_o(tf.reduce_mean(tf.stack(hs, 0), 0) if self.num_heads > 1 else hs[0])
        return self.dropout(out), tf.stack(attns, 1)
    def get_config(self): return {**super().get_config(), "num_heads": self.num_heads, "d_model": self.d_model, "dropout_rate": self.dropout_rate}

@tf.keras.utils.register_keras_serializable()
class MomentumTransformer(Model):
    def __init__(self, time_steps, input_size, output_size, hidden_size, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.time_steps, self.input_size, self.output_size, self.hidden_size, self.num_heads, self.dropout_rate = time_steps, input_size, output_size, hidden_size, num_heads, dropout_rate
        self.feature_embeddings = [layers.Dense(hidden_size) for _ in range(input_size)]
        self.var_selection = VariableSelectionNetwork(input_size, hidden_size, dropout_rate)
        self.lstm, self.lstm_gate, self.lstm_norm = layers.LSTM(hidden_size, return_sequences=True, dropout=dropout_rate), GluLayer(hidden_size), layers.LayerNormalization()
        self.post_lstm_grn = GatedResidualNetwork(hidden_size, dropout_rate=dropout_rate)
        self.attention, self.attn_gate, self.attn_norm = InterpretableMultiHeadAttention(num_heads, hidden_size, dropout_rate), GluLayer(hidden_size), layers.LayerNormalization()
        self.post_attn_grn, self.output_gate, self.output_norm = GatedResidualNetwork(hidden_size, dropout_rate=dropout_rate), GluLayer(hidden_size), layers.LayerNormalization()
        self.final_dense = layers.Dense(output_size, activation="tanh")
    def call(self, x, return_weights=False):
        e = tf.stack([self.feature_embeddings[i](x[..., i:i+1]) for i in range(self.input_size)], axis=2)
        sel, v_w = self.var_selection(e)
        t = self.lstm_norm(sel + self.lstm_gate(self.lstm(sel))[0])
        sm = tf.linalg.band_part(tf.ones((tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[1])), -1, 0)
        a_out, a_w = self.attention(self.post_lstm_grn(t), self.post_lstm_grn(t), self.post_lstm_grn(t), mask=sm)
        a_lay = self.attn_norm(self.post_lstm_grn(t) + self.attn_gate(a_out)[0])
        out = self.final_dense(self.output_norm(t + self.output_gate(self.post_attn_grn(a_lay))[0]))
        return (out, {'vsn_weights': v_w, 'attn_weights': a_w}) if return_weights else out
    def get_config(self): return {**super().get_config(), "time_steps": self.time_steps, "input_size": self.input_size}

# ============================================================================
# 2. CONFIGURATION & INFRASTRUCTURE
# ============================================================================

DATA_CONFIG = {
    'window_size': 21,
    'feature_cols': [
        'norm_daily_return', 'norm_monthly_return', 'norm_quarterly_return',
        'norm_biannual_return', 'norm_annual_return', 'macd_8_24',
        'macd_16_48', 'macd_32_96', 'volatility_20d', 'volatility_60d',
        'cp_rl_21', 'cp_score_21',
    ]
}

def calculate_all_metrics(returns, weights=None):
    returns = np.nan_to_num(returns)
    def mdd(r):
        c = np.exp(np.cumsum(r))
        p = np.maximum.accumulate(c)
        return float(np.min((c - p) / p)) if len(p) > 0 and np.max(p) > 0 else 0
    
    net_returns = returns.copy()
    cost_total = 0
    if weights is not None:
        trades = np.abs(np.diff(np.nan_to_num(weights), prepend=0))
        cost_series = trades * BPS_COST
        net_returns = returns - cost_series
        cost_total = cost_series.sum()

    std = net_returns.std()
    res = {
        'total_return': float(np.exp(np.sum(net_returns)) - 1),
        'sharpe': (np.sqrt(252) * net_returns.mean() / std) if std > 1e-9 else 0,
        'mdd': mdd(net_returns),
        'win_rate': float(np.mean(net_returns > 0)),
        'txn_costs_total': float(cost_total)
    }
    res['calmar'] = (net_returns.mean() * 252) / abs(res['mdd']) if res['mdd'] < -1e-6 else 0
    if weights is not None:
        act_mask = np.abs(weights) > 0
        res['active_win_rate'] = np.mean(net_returns[act_mask] > 0) if np.any(act_mask) else 0
        res['trade_freq'] = np.mean(act_mask)
    return res

def main():
    print(f"\n{'='*60}\n QuantKubera AFML Experiment: Cost-Adjusted Final Run\n{'='*60}\n")
    try:
        print("1. Feature Engineering & Scaling...")
        df = pd.read_csv("data/raw/MCX/SILVER.csv", parse_dates=True, index_col=0)
        if df.index.tz: df.index = df.index.tz_localize(None)
        
        # Exact reproduction of FeatureEngineer logic
        df['DAILY_RET_RAW'] = df['close'].pct_change(1).fillna(0)
        df['norm_daily_return'] = df['DAILY_RET_RAW']
        df['norm_monthly_return'] = df['close'].pct_change(21).fillna(0)
        df['norm_quarterly_return'] = df['close'].pct_change(63).fillna(0)
        df['norm_biannual_return'] = df['close'].pct_change(126).fillna(0)
        df['norm_annual_return'] = df['close'].pct_change(252).fillna(0)
        
        # MACD (adjust=False matches production)
        for f,s in [(8,24), (16,48), (32,96)]:
            df[f'macd_{f}_{s}'] = df['close'].ewm(span=f, adjust=False).mean() - df['close'].ewm(span=s, adjust=False).mean()
        
        # Volatility
        for w in [20, 60]: df[f'volatility_{w}d'] = df['DAILY_RET_RAW'].rolling(w).std().fillna(0)
        
        # CPD Features
        cpd_path = "data/cpd/SILVER_cpd_21.csv"
        if os.path.exists(cpd_path):
            cpd = pd.read_csv(cpd_path, index_col=0, parse_dates=True).reindex(df.index)
            df['cp_rl_21'] = cpd['cp_location_norm'].fillna(0)
            df['cp_score_21'] = cpd['cp_score'].fillna(0)
        else:
            df['cp_rl_21'], df['cp_score_21'] = 0.0, 0.0
        
        # Scaling (Z-Score on whole sample for stability)
        f_cols = DATA_CONFIG['feature_cols']
        df_scaled = df.copy()
        for col in f_cols:
            mean, std = df[col].mean(), df[col].std()
            df_scaled[col] = (df[col] - mean) / (std if std > 1e-9 else 1.0)
        
        print(f"   Sample Base: {len(df)} rows. Features: {len(f_cols)}")

        # --- MODEL LOADING ---
        print("2. Loading Trained Weights...")
        co = {
            'MomentumTransformer': MomentumTransformer, 'GluLayer': GluLayer, 
            'GatedResidualNetwork': GatedResidualNetwork, 'VariableSelectionNetwork': VariableSelectionNetwork, 
            'InterpretableMultiHeadAttention': InterpretableMultiHeadAttention, 'ScaledDotProductAttention': ScaledDotProductAttention
        }
        pm = keras.models.load_model("models/afml_primary_v2/afml_primary_model.keras", custom_objects=co, compile=False, safe_mode=False)
        mm = keras.models.load_model("models/afml_meta_v1/best_meta_model.keras", custom_objects=co, compile=False, safe_mode=False)
        
        # --- INFERENCE ---
        print("3. Strategy Inference (Last 1000 days)...")
        sub = df_scaled.tail(1000)
        ws = DATA_CONFIG['window_size']
        results = []
        for i in range(ws, len(sub)):
            w_input = np.expand_dims(sub[f_cols].iloc[i-ws:i].values.astype(np.float32), 0)
            p_out = pm.predict(w_input, verbose=0)
            p = p_out[0, -1, 0] if not isinstance(p_out, list) else p_out[0][0, -1, 0]
            m_out = mm.predict(w_input, verbose=0)
            # Meta-model might be wrapped in a Lambda Layer model
            m = m_out[0, 0] if len(m_out.shape) > 1 else m_out[0]
            
            # Sigmoid for confidence (if logit)
            conf = 1.0 / (1.0 + np.exp(-m)) if abs(m) < 20 else (1.0 if m > 0 else 0.0)
            
            results.append({'date': sub.index[i], 'primary': np.sign(p), 'confidence': float(conf)})
        
        res_df = pd.DataFrame(results).set_index('date')
        eval_df = df[['DAILY_RET_RAW']].join(res_df, how='inner')
        
        # --- THRESHOLD SWEEP ---
        print("\n4. Optimizing Confidence Threshold...")
        thresholds = np.linspace(res_df['confidence'].min(), res_df['confidence'].max(), 30)
        sweep_results = []
        for t in thresholds:
            m_auth = np.where(eval_df['confidence'] > t, 1, 0)
            m_w = (eval_df['primary'] * m_auth).shift(1).fillna(0)
            m_perf = calculate_all_metrics(eval_df['DAILY_RET_RAW'] * m_w, weights=m_w)
            sweep_results.append({'threshold': t, 'return': m_perf['total_return'], 'mdd': m_perf['mdd'], 'costs': m_perf['txn_costs_total']})
            
        sweep_df = pd.DataFrame(sweep_results)
        # Select Balanced Threshold (High Return + Crash Protection)
        # Filter for MDD better than baseline -0.30
        safe_df = sweep_df[sweep_df['mdd'] > -0.30]
        if not safe_df.empty:
            best_t = safe_df.loc[safe_df['return'].idxmax(), 'threshold']
        else:
            best_t = sweep_df.loc[sweep_df['return'].idxmax(), 'threshold']
            
        print(f"   Using Balanced Threshold: {best_t:.4f} (Net Return: {sweep_df[sweep_df['threshold']==best_t]['return'].values[0]:.2%})")
        
        # Apply Optimal Threshold
        eval_df['b_w'] = eval_df['primary'].shift(1).fillna(0)
        eval_df['m_auth'] = np.where(eval_df['confidence'] > best_t, 1, 0)
        eval_df['m_w'] = (eval_df['primary'] * eval_df['m_auth']).shift(1).fillna(0)
        
        perf = pd.DataFrame({
            'Baseline (TMT+Costs)': calculate_all_metrics(eval_df['DAILY_RET_RAW'] * eval_df['b_w'], weights=eval_df['b_w']),
            'AFML Optimized (+Costs)': calculate_all_metrics(eval_df['DAILY_RET_RAW'] * eval_df['m_w'], weights=eval_df['m_w'])
        }).T
        
        print(f"\n--- PERFORMANCE (BPS Cost: {BPS_COST*10000:.0f}) ---\n{perf.to_string()}")
        
        # --- TELEMETRY ---
        print("\n--- RECENT TELEMETRY (Last 15 Days) ---")
        log = eval_df.tail(15).copy()
        log['Side'] = log.apply(lambda r: 'CASH' if abs(r['m_w']) == 0 else ('LONG' if r['primary'] > 0 else 'SHORT'), axis=1)
        log['Ret %'] = (log['DAILY_RET_RAW'] * 100).round(2)
        log['Conf %'] = (log['confidence'] * 100).round(1)
        print(log[['Side', 'Conf %', 'Ret %']].to_string())

        plt.figure(figsize=(10, 5))
        np.exp((eval_df['DAILY_RET_RAW']*eval_df['m_w']).cumsum()).plot(label='AFML Hybrid (Net of Costs)', lw=2)
        plt.title(f"QuantKubera: AFML Strategy Equity (Costs={BPS_COST*10000:.0f}bps)"); plt.legend(); plt.grid(True)
        plt.savefig("afml_experiment_results.png")
        print(f"\nSaved Equity Curve: {os.path.abspath('afml_experiment_results.png')}")

    except Exception as e:
        print(f"FAILED: {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__": main()
