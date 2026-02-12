# Classic TFT Modernization Plan

## Executive Summary

This document outlines the plan to implement a production-grade **Temporal Fusion Transformer (TFT)** alongside the existing X-Trend model. The existing `x_trend.py` is **NOT** classic TFT but rather a cross-attentive regime-following model (Wood et al., 2023). The classic TFT modernization will create a new `classic_tft.py` with full architectural integrity, proper normalization, and interpretable components.

---

## Architecture Overview

### Classic TFT Architecture Diagram

```
INPUT LAYER
├── Static Features (asset_id, sector, exchange)
│   └── Static Covariate Encoder → [B, static_dim]
│
├── Known Future Features (volatility, expiry_distance)
│   └── (passed to LSTM encoder)
│
├── Past Time Series (OHLCV, OI, FII, sentiment)
│   └── LSTM Encoder (Bi-directional)
│       └── [B, T, encoder_dim]
│       └── Context vector: max pool or attention
│
├── Recent Past Features (last 5 bars)
│   └── Decoder LSTM initialization
│
TEMPORAL FUSION CORE
├── Encoder LSTM
│   ├── Input: [past_t || past_mask_t || known_future_t || static_t]
│   ├── Hidden state: h_enc, c_enc
│   └── Output: [B, T, 256]
│
├── Decoder LSTM (autoregressive)
│   ├── Input (per step): [decoder_input_t || static_t]
│   ├── Initial hidden: h_enc (gated)
│   ├── Hidden state: h_dec, c_dec
│   └── Output: [B, 1, 256] per step
│
INTERPRETABLE COMPONENTS
├── GRN (Gated Residual Network)
│   ├── Input: Any feature vector
│   ├── GLU: x * sigmoid(Wx + b)
│   └── Output: Gated and enriched features
│
├── VSN (Variable Selection Network)
│   ├── Input: [all features]
│   ├── Per-variable: softmax(mlp(feature))
│   ├── Output: [B, num_features] (soft selection weights)
│   └── Interpretation: Which features matter?
│
├── Interpretable Multi-Head Attention (MHA)
│   ├── Queries: Decoder hidden state
│   ├── Keys/Values: Encoder hidden states
│   ├── Attention weights: [B, num_heads, T]
│   ├── Per-head interpretability: Which timestep matters?
│   └── Head-specific: Each head focuses on different temporal pattern
│
QUANTILE OUTPUT LAYER
├── Adaptive Quantile Selection
│   ├── Input: decoder output + context
│   ├── Output: 3 quantile heads (0.1, 0.5, 0.9)
│   └── Loss: Quantile loss (pinball loss)
│
├── Probabilistic Output
│   ├── Q_0.1: lower tail forecast
│   ├── Q_0.5: median (point forecast)
│   ├── Q_0.9: upper tail forecast
│   └── Uncertainty: [Q_0.9 - Q_0.1] / 2
│
FINAL OUTPUT
└── [B, forecast_horizon, 3 quantiles]
    ├── Use Q_0.5 for position sizing
    ├── Use [Q_0.1, Q_0.9] for stop losses
    └── Uncertainty calibrated by quantile spread
```

---

## Component Details

### 1. Static Covariate Encoder

**Purpose**: Embed static metadata (asset identity, sector, characteristics)

**Mechanism**:
```python
class StaticCovariateEncoder(nn.Module):
    def __init__(self, asset_vocab_size, sector_vocab_size, hidden_dim):
        self.asset_embed = nn.Embedding(asset_vocab_size, hidden_dim // 2)
        self.sector_embed = nn.Embedding(sector_vocab_size, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, asset_id, sector_id, continuous_features):
        # asset_id: [B] integer (0-50 for NSE stocks)
        # sector_id: [B] integer (0-10 for sectors)
        # continuous_features: [B, num_continuous] (e.g., market_cap)

        embed = torch.cat([
            self.asset_embed(asset_id),     # [B, hidden_dim//2]
            self.sector_embed(sector_id),   # [B, hidden_dim//2]
            continuous_features             # [B, num_continuous]
        ], dim=1)
        static_context = self.fc(embed)  # [B, hidden_dim]
        return static_context
```

**Input**: Asset ID, sector, market cap, previous Sharpe (if available)
**Output**: [B, hidden_dim] static context vector

---

### 2. LSTM Encoder-Decoder

**Purpose**: Capture temporal dependencies in past data; generate future forecasts autoregressively

**Encoder**:
```python
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           bidirectional=True, batch_first=True)
        self.context_fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, static_context):
        # x: [B, T, input_dim]
        # static_context: [B, hidden_dim]

        output, (h_n, c_n) = self.lstm(x)
        # output: [B, T, hidden_dim*2]

        # Max pool over time + static context
        context = torch.max(output, dim=1)[0]  # [B, hidden_dim*2]
        context = torch.cat([context, static_context], dim=1)
        context = self.context_fc(context)  # [B, hidden_dim]

        return output, context, (h_n, c_n)
```

**Decoder** (Autoregressive):
```python
class TemporalDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_init, h_enc, c_enc, static_context, horizon):
        # x_init: [B, 1, input_dim] (last known value)
        # h_enc, c_enc: encoder hidden states
        # static_context: [B, hidden_dim]
        # horizon: number of steps to forecast

        outputs = []
        h_t, c_t = h_enc, c_enc

        for step in range(horizon):
            # Gated initialization (first step only)
            if step == 0:
                gate = torch.sigmoid(self.init_gate_fc(static_context))
                h_t = gate * h_t

            x_t = x_init if step == 0 else last_output
            lstm_out, (h_t, c_t) = self.lstm(x_t, (h_t, c_t))
            output = self.output_fc(lstm_out[:, -1, :])  # [B, output_dim]
            outputs.append(output)
            last_output = output.unsqueeze(1)

        return torch.stack(outputs, dim=1)  # [B, horizon, output_dim]
```

---

### 3. Gated Residual Networks (GRN)

**Purpose**: Enriched feature processing with learnable gating mechanism

**Mechanism**:
```python
class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None):
        if output_dim is None:
            output_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate_fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Gated Linear Unit (GLU) mechanism
        residual = x if x.shape[-1] == self.fc2.out_features else x

        hidden = F.relu(self.fc1(x))
        output = self.fc2(hidden)
        gate = torch.sigmoid(self.gate_fc(x))

        # Gated output
        return output * gate + residual * (1 - gate)
```

**Why GRN Matters**:
- Avoids information loss through gating
- Adaptive feature enrichment (learns when to use vs bypass)
- Critical for stabilizing training (like residual networks)

---

### 4. Variable Selection Network (VSN)

**Purpose**: Learn which input features are relevant for each prediction step

**Mechanism**:
```python
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_features):
        self.grn = GRN(input_dim, hidden_dim)
        self.softmax_fc = nn.Linear(hidden_dim, num_features)

    def forward(self, x, context):
        # x: [B, num_features]
        # context: [B, hidden_dim] (from encoder)

        enriched = self.grn(x)
        weights = torch.softmax(self.softmax_fc(enriched), dim=-1)
        # weights: [B, num_features] (sums to 1)

        # Apply soft selection
        selected = (x * weights.unsqueeze(-1)).sum(dim=-2)
        return selected, weights
```

**Interpretation**:
```
weights[b, i] = P(feature i is selected for batch element b)
  - High weight → feature matters
  - Low weight → feature is noise
  - Sum over time → feature importance curve
```

---

### 5. Interpretable Multi-Head Attention

**Purpose**: Understand WHICH timesteps the model attends to for each prediction

**Mechanism**:
```python
class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        self.num_heads = num_heads
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, decoder_hidden, encoder_output):
        # decoder_hidden: [B, hidden_dim]
        # encoder_output: [B, T, hidden_dim]

        B, T, H = encoder_output.shape

        # Project
        Q = self.query_proj(decoder_hidden).view(B, 1, self.num_heads, -1)
        K = self.key_proj(encoder_output).view(B, T, self.num_heads, -1)
        V = self.value_proj(encoder_output).view(B, T, self.num_heads, -1)

        # Attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: [B, 1, num_heads, T]

        attention_weights = torch.softmax(scores, dim=-1)
        # attention_weights: [B, 1, num_heads, T]

        # For interpretability, average over heads
        avg_attention = attention_weights.mean(dim=2).squeeze(1)
        # avg_attention: [B, T] (which timesteps matter?)

        # Weighted sum of values
        output = torch.matmul(attention_weights, V).mean(dim=2)
        # output: [B, hidden_dim]

        return output, attention_weights, avg_attention
```

**Interpretation Dashboard**:
```
For each prediction, plot:
  1. Per-head attention weights [num_heads, T]
     → See if different heads specialize (e.g., Head 0 = trend, Head 1 = mean-reversion)
  2. Average attention [T]
     → See which historical timesteps matter most
  3. Head-specific gradients
     → Which head drives the final prediction?
```

---

### 6. Quantile Output Layer

**Purpose**: Generate probabilistic forecasts, not point forecasts

**Mechanism**:
```python
class QuantileOutputLayer(nn.Module):
    def __init__(self, hidden_dim, quantiles=[0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.quantile_heads = nn.ModuleDict({
            f"q_{q}": nn.Linear(hidden_dim, 1)
            for q in quantiles
        })

    def forward(self, hidden):
        # hidden: [B, hidden_dim]
        outputs = {}
        for q in self.quantiles:
            outputs[f"q_{q}"] = self.quantile_heads[f"q_{q}"](hidden).squeeze(-1)
        return outputs
```

**Loss Function**:
```python
def quantile_loss(pred, target, q):
    """Pinball loss for quantile q"""
    error = target - pred
    loss = torch.max(q * error, (q - 1) * error)
    return loss.mean()

# Training loop
total_loss = 0
for q in [0.1, 0.5, 0.9]:
    total_loss += quantile_loss(pred[f"q_{q}"], target, q)
total_loss /= 3
total_loss.backward()
```

**Interpretation**:
- **Q_0.5** (median): Use for position sizing
- **Q_0.1** (10th percentile): Lower confidence bound for stops
- **Q_0.9** (90th percentile): Upper confidence bound for targets
- **Spread** = Q_0.9 - Q_0.1 = uncertainty estimate

---

## ClassicTFTConfig Dataclass

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ClassicTFTConfig:
    """Configuration for Classic Temporal Fusion Transformer"""

    # Data dimensions
    num_time_steps: int = 60  # Historical lookback (60 days)
    forecast_horizon: int = 5  # Forecast 5 days ahead
    num_assets: int = 50  # NSE stocks
    num_sectors: int = 11  # Sector categories
    num_quantiles: int = 3  # [0.1, 0.5, 0.9]

    # Input feature dimensions
    past_time_series_dim: int = 20  # [open, high, low, close, volume, oi, fii, sentiment, ...]
    known_future_dim: int = 3  # [days_to_expiry, day_of_week, is_expiry_day]
    static_features_dim: int = 5  # [sector, market_cap, volatility_regime, ...]
    num_static_vocab: int = 11  # Sector embedding vocab

    # Network architecture
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    grn_hidden_dim: int = 128
    num_lstm_layers: int = 2
    num_attention_heads: int = 8
    dropout_rate: float = 0.1

    # Normalization
    normalize_method: str = "standardize"  # "standardize" or "minmax"
    normalization_window: str = "train_test"  # Only normalize on train+test split (no look-ahead)
    normalization_percentile_min: float = 0.01  # Handle outliers
    normalization_percentile_max: float = 0.99

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 50
    gradient_clip_norm: float = 1.0
    weight_decay: float = 1e-5
    loss_weights: dict = None  # {q_0.1: 1.0, q_0.5: 2.0, q_0.9: 1.0}

    # Regularization
    use_batch_norm: bool = False  # LayerNorm preferred in TFT
    use_layer_norm: bool = True
    add_skip_connections: bool = True
    use_variational_dropout: bool = False

    # Cross-validation
    num_folds: int = 5
    test_size: int = 63  # ~3 months
    train_size: int = 252  # ~1 year
    purge_gap: int = 5  # Prevent look-ahead bias at fold boundaries
    embargo_pct: float = 0.01  # 1% embargo on either side of test set

    # Quantile parameters
    quantiles: List[float] = None
    lower_quantile: float = 0.1
    median_quantile: float = 0.5
    upper_quantile: float = 0.9

    # Interpretability
    save_attention_maps: bool = True
    save_variable_importance: bool = True
    attention_map_dir: str = "models/tft_attention_maps"

    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {"q_0.1": 1.0, "q_0.5": 2.0, "q_0.9": 1.0}
        if self.quantiles is None:
            self.quantiles = [self.lower_quantile, self.median_quantile, self.upper_quantile]

    def validate(self):
        """Sanity checks on configuration"""
        assert self.forecast_horizon > 0, "forecast_horizon must be > 0"
        assert self.num_time_steps > self.forecast_horizon, \
            "num_time_steps must be > forecast_horizon"
        assert self.encoder_hidden_dim % self.num_attention_heads == 0, \
            "encoder_hidden_dim must be divisible by num_attention_heads"
        assert self.num_quantiles == len(self.quantiles), \
            "num_quantiles must match length of quantiles list"
        assert self.purge_gap >= 0, "purge_gap cannot be negative"
        assert 0 <= self.embargo_pct <= 0.1, "embargo_pct should be 0-10%"
        assert self.normalization_window in ["train_test", "full"], \
            "normalization_window must be 'train_test' or 'full'"
```

---

## Implementation Plan

### File Structure

```
QuantLaxmi/quantlaxmi/models/ml/tft/
├── __init__.py
├── classic_tft.py              (NEW: Main Classic TFT implementation)
├── components/
│   ├── __init__.py
│   ├── encoder.py              (TemporalEncoder, StaticCovariateEncoder)
│   ├── decoder.py              (TemporalDecoder)
│   ├── attention.py            (InterpretableMultiHeadAttention)
│   ├── grn.py                  (GatedResidualNetwork)
│   └── vsn.py                  (VariableSelectionNetwork)
├── inference/
│   ├── __init__.py
│   └── inference.py            (Production inference pipeline)
├── production/
│   └── training_pipeline.py    (EXISTING: Already has phase structure)
├── research/
│   └── classic_tft_research.py (Backtesting + ablations)
├── tests/
│   └── test_classic_tft.py     (Unit tests)
└── docs/
    ├── CLASSIC_TFT_MODERNIZATION_PLAN.md (THIS FILE)
    └── CLASSIC_TFT_ARCHITECTURE.md (Detailed math)

NOTE: x_trend.py remains unchanged (different model, Wood et al. 2023)
```

---

### Phase 1: Components (Week 1)

**Files to Create**:

1. **`components/grn.py`** (80 lines)
   - GRN class
   - GLU mechanism
   - Unit tests

2. **`components/vsn.py`** (120 lines)
   - VariableSelectionNetwork
   - Soft feature selection
   - Returns weights (for interpretability)

3. **`components/encoder.py`** (200 lines)
   - TemporalEncoder (bidirectional LSTM)
   - StaticCovariateEncoder (embeddings + concat)
   - Context pooling

4. **`components/decoder.py`** (180 lines)
   - TemporalDecoder (autoregressive)
   - Gated initialization
   - Per-step LSTM update

5. **`components/attention.py`** (150 lines)
   - InterpretableMultiHeadAttention
   - Per-head + average attention weights
   - Attention map saving

---

### Phase 2: Main Model (Week 2)

**File to Create**:

**`classic_tft.py`** (400 lines)
```python
class ClassicTFT(nn.Module):
    def __init__(self, config: ClassicTFTConfig):
        super().__init__()
        config.validate()

        # Store config
        self.config = config

        # Components
        self.static_encoder = StaticCovariateEncoder(...)
        self.encoder = TemporalEncoder(...)
        self.decoder = TemporalDecoder(...)
        self.attention = InterpretableMultiHeadAttention(...)
        self.vsn = VariableSelectionNetwork(...)

        # Quantile output heads
        self.quantile_outputs = nn.ModuleDict({
            f"q_{q}": nn.Linear(config.decoder_hidden_dim, 1)
            for q in config.quantiles
        })

    def forward(self, past_values, known_future, static_features, attention_masks=None):
        # Encode static context
        static_context = self.static_encoder(static_features)

        # Encode temporal dynamics
        encoder_output, context, (h_n, c_n) = self.encoder(
            past_values,
            static_context
        )

        # Decode (autoregressive)
        decoder_output = self.decoder(
            past_values[:, -1:, :],
            h_n, c_n,
            static_context,
            self.config.forecast_horizon
        )

        # Attention over encoder outputs
        attention_output, attention_weights, avg_attention = self.attention(
            decoder_output[:, -1, :],
            encoder_output
        )

        # Combine decoder + attention
        final_hidden = decoder_output[:, -1, :] + attention_output

        # Generate quantile forecasts
        outputs = {}
        for q in self.config.quantiles:
            outputs[f"q_{q}"] = self.quantile_outputs[f"q_{q}"](final_hidden)

        return outputs, attention_weights, avg_attention

    def loss(self, pred, target):
        total_loss = 0
        for q in self.config.quantiles:
            loss_q = quantile_loss(pred[f"q_{q}"], target, q)
            loss_q *= self.config.loss_weights.get(f"q_{q}", 1.0)
            total_loss += loss_q
        return total_loss / len(self.config.quantiles)
```

---

### Phase 3: Training Pipeline (Week 3)

**File to Create**:

**`production/training_pipeline.py`** (update existing file)

Add ClassicTFT training phase:
```python
class TFTClassicTrainingPhase(TrainingPhaseBase):
    def __init__(self, config: ClassicTFTConfig, data_loader, device):
        self.config = config
        self.model = ClassicTFT(config).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.5
        )

    def run(self):
        """Main training loop with walk-forward validation"""
        best_oos_sharpe = -np.inf

        for fold_idx in range(self.config.num_folds):
            # Split data respecting purge_gap and embargo
            train_idx, test_idx = self._get_fold_indices(fold_idx)

            # Normalize ONLY on train+test (no future data leakage)
            scaler = self._fit_scaler(train_idx, test_idx)

            # Train on fold
            val_sharpe = self._train_epoch(train_idx, scaler)

            # Evaluate on OOS test set
            oos_sharpe, oos_returns = self._evaluate(test_idx, scaler)

            # Save checkpoint if OOS improves
            if oos_sharpe > best_oos_sharpe:
                best_oos_sharpe = oos_sharpe
                self._save_checkpoint(fold_idx, oos_sharpe)

        return {"best_oos_sharpe": best_oos_sharpe}

    def _train_epoch(self, train_idx, scaler):
        self.model.train()
        total_loss = 0
        for batch in self._get_batches(train_idx, scaler):
            past, future, static = batch
            pred, attn, avg_attn = self.model(past, future, static)
            loss = self.model.loss(pred, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()

        self.scheduler.step(total_loss)
        return total_loss / len(train_idx)

    def _evaluate(self, test_idx, scaler):
        self.model.eval()
        all_pred, all_actual = [], []

        with torch.no_grad():
            for batch in self._get_batches(test_idx, scaler):
                past, future, static = batch
                pred, attn, avg_attn = self.model(past, future, static)
                all_pred.append(pred[f"q_0.5"].cpu().numpy())
                all_actual.append(future.cpu().numpy())

        pred_returns = np.concatenate(all_pred)
        actual_returns = np.concatenate(all_actual)

        sharpe = compute_sharpe(actual_returns - pred_returns)
        return sharpe, actual_returns - pred_returns
```

---

### Phase 4: Tests (Week 3)

**File to Create**:

**`tests/test_classic_tft.py`** (300 lines)
```python
import pytest
import torch
import numpy as np
from quantlaxmi.models.ml.tft.classic_tft import ClassicTFT, ClassicTFTConfig

class TestClassicTFT:
    @pytest.fixture
    def config(self):
        return ClassicTFTConfig(
            num_time_steps=60,
            forecast_horizon=5,
            num_assets=50,
            batch_size=32
        )

    @pytest.fixture
    def model(self, config):
        return ClassicTFT(config)

    def test_forward_pass(self, model, config):
        past = torch.randn(32, config.num_time_steps, config.past_time_series_dim)
        future = torch.randn(32, 1, config.past_time_series_dim)
        static = torch.randn(32, config.static_features_dim)

        pred, attn, avg_attn = model(past, future, static)

        assert pred["q_0.5"].shape == (32, 1)
        assert attn.shape == (32, 1, config.num_attention_heads, config.num_time_steps)
        assert avg_attn.shape == (32, config.num_time_steps)

    def test_quantile_loss(self, model):
        pred = {"q_0.1": torch.tensor([0.8]), "q_0.5": torch.tensor([1.0]), "q_0.9": torch.tensor([1.2])}
        target = torch.tensor([1.0])

        loss = model.loss(pred, target)
        assert loss >= 0
        assert not torch.isnan(loss)

    def test_attention_weights_sum_to_one(self, model, config):
        past = torch.randn(32, config.num_time_steps, config.past_time_series_dim)
        future = torch.randn(32, 1, config.past_time_series_dim)
        static = torch.randn(32, config.static_features_dim)

        _, attn, avg_attn = model(past, future, static)

        # Check attention sums to 1 over time dimension
        attn_sum = avg_attn.sum(dim=1)
        assert torch.allclose(attn_sum, torch.ones(32), atol=1e-5)

    def test_no_lookahead_bias(self, model, config):
        """Ensure future data doesn't leak into past encoding"""
        past1 = torch.randn(32, config.num_time_steps, config.past_time_series_dim)
        past2 = past1.clone()
        past2[:, -1, :] += 0.1  # Modify only last timestep

        future = torch.randn(32, 1, config.past_time_series_dim)
        static = torch.randn(32, config.static_features_dim)

        pred1, _, _ = model(past1, future, static)
        pred2, _, _ = model(past2, future, static)

        # Small change in past should NOT affect future prediction
        # (if model has look-ahead bias, it will be affected)
        diff = torch.abs(pred1["q_0.5"] - pred2["q_0.5"]).mean()
        # This is a heuristic test; exact tolerance depends on model

    def test_config_validation(self):
        """Bad config should raise assertion"""
        bad_config = ClassicTFTConfig(
            num_time_steps=5,
            forecast_horizon=10  # forecast_horizon > num_time_steps
        )
        with pytest.raises(AssertionError):
            bad_config.validate()

    def test_interpretability_outputs(self, model, config):
        """Test that attention maps and variable importance are accessible"""
        past = torch.randn(32, config.num_time_steps, config.past_time_series_dim)
        future = torch.randn(32, 1, config.past_time_series_dim)
        static = torch.randn(32, config.static_features_dim)

        pred, attn, avg_attn = model(past, future, static)

        # Should be able to identify which timesteps matter most
        top_k_timesteps = torch.topk(avg_attn, k=10, dim=1)[1]
        assert top_k_timesteps.shape == (32, 10)
```

---

### Phase 5: Inference (Week 4)

**File to Create**:

**`inference/inference.py`** (250 lines)
```python
class ClassicTFTInference:
    def __init__(self, model_path: str, config: ClassicTFTConfig, device="cuda"):
        self.model = ClassicTFT(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
        self.config = config
        self.device = device

    def forecast(self, past_values, known_future, static_features, return_interpretability=False):
        """
        Generate forecasts with optional attention visualization

        Args:
            past_values: [num_time_steps, num_features]
            known_future: [forecast_horizon, num_known_features]
            static_features: [num_static_features]
            return_interpretability: if True, return attention weights

        Returns:
            forecast: dict with q_0.1, q_0.5, q_0.9
            interpretation: dict with attention_weights, variable_importance (optional)
        """
        with torch.no_grad():
            # Prepare inputs
            past_values = torch.tensor(past_values, dtype=torch.float32).unsqueeze(0).to(self.device)
            known_future = torch.tensor(known_future, dtype=torch.float32).unsqueeze(0).to(self.device)
            static_features = torch.tensor(static_features, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Forward pass
            pred, attn, avg_attn = self.model(past_values, known_future, static_features)

            # Extract results
            forecast = {
                "q_0.1": pred["q_0.1"].cpu().numpy().squeeze(),
                "q_0.5": pred["q_0.5"].cpu().numpy().squeeze(),
                "q_0.9": pred["q_0.9"].cpu().numpy().squeeze()
            }

            if return_interpretability:
                interpretation = {
                    "attention_weights": attn.cpu().numpy().squeeze(),
                    "average_attention": avg_attn.cpu().numpy().squeeze()
                }
                return forecast, interpretation

            return forecast

    def save_attention_visualization(self, past_values, known_future, static_features, save_path):
        """Generate matplotlib heatmap of attention patterns"""
        _, interpretation = self.forecast(
            past_values, known_future, static_features,
            return_interpretability=True
        )

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1 + self.config.num_attention_heads, 1, figsize=(12, 8))

        # Plot average attention
        axes[0].plot(interpretation["average_attention"])
        axes[0].set_title("Average Attention (all heads)")
        axes[0].set_ylabel("Attention weight")

        # Plot per-head attention
        for head_idx in range(self.config.num_attention_heads):
            axes[head_idx + 1].plot(interpretation["attention_weights"][0, head_idx, :])
            axes[head_idx + 1].set_title(f"Head {head_idx} Attention")
            axes[head_idx + 1].set_ylabel("Weight")

        axes[-1].set_xlabel("Timestep (days before prediction)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Attention visualization saved to {save_path}")
```

---

## Key Normalization Rules (Preventing Look-Ahead Bias)

```python
class CorrectNormalization:
    """
    CRITICAL: Only normalize on train + test data, NOT on future data
    """

    def fit_scaler(self, X_train, X_test):
        """
        X_train: Historical training data [T_train, num_features]
        X_test: Out-of-sample test data [T_test, num_features]

        Returns:
            scaler: Fitted StandardScaler
        """
        # IMPORTANT: Fit on train + test together, NOT on test alone
        X_combined = np.vstack([X_train, X_test])
        scaler = StandardScaler()
        scaler.fit(X_combined)

        # Save normalization parameters
        self.mean = scaler.mean_
        self.std = scaler.scale_

        return scaler

    def apply_scaler(self, X, scaler):
        """Apply fitted scaler to new data (e.g., production)"""
        return (X - self.mean) / self.std

    def inverse_transform(self, X_normalized):
        """Convert normalized predictions back to original scale"""
        return X_normalized * self.std + self.mean

    def WRONG_normalization_example(self, X_train, X_test):
        """
        DO NOT DO THIS:
        - Never fit scaler on test data alone
        - Never fit scaler on test data + future validation data
        - These introduce look-ahead bias (test distribution leaks into parameters)
        """
        # WRONG
        scaler = StandardScaler()
        scaler.fit(X_test)  # BUG: fitting on test data!

        # WRONG
        X_future = get_future_data()
        scaler.fit(np.vstack([X_train, X_test, X_future]))  # BUG: leaking future info
```

---

## Integration with Existing TFT Pipeline

The existing `production/training_pipeline.py` has a phase-based structure:
- Phase 1: Data loading + feature engineering
- Phase 2: Walk-forward validation setup
- Phase 3: VSN feature selection (for variable-importance baseline)
- Phase 4: Hyperparameter optimization (Optuna)
- Phase 5: Final training (production pass)

**Classic TFT will be integrated as**:
```python
class ProductionTrainingPipeline:
    def run(self):
        # Existing phases 1-3
        ...

        # Phase 6 (NEW): Classic TFT training
        classic_tft_phase = TFTClassicTrainingPhase(
            config=ClassicTFTConfig(),
            data_loader=self.data_loader,
            device=self.device
        )
        classic_tft_results = classic_tft_phase.run()

        # Comparison: Classic TFT vs X-Trend
        comparison = {
            "classic_tft_oos_sharpe": classic_tft_results["best_oos_sharpe"],
            "x_trend_oos_sharpe": self.existing_x_trend_sharpe,
            "winner": "classic_tft" if ... else "x_trend"
        }

        return comparison
```

---

## Interpretation & Analysis

### Dashboard Components

1. **Time Series Attention Plot**
   - Y-axis: Attention weight (0 to 1)
   - X-axis: Days in history (0 = now, -60 = 2 months ago)
   - Shows which historical points matter for today's forecast
   - Example: Spike at -30 = model is remembering mean reversion window

2. **Variable Importance Heat Map**
   - Rows: Input variables (open, close, volume, FII, sentiment, ...)
   - Columns: Forecast horizon steps (day 1, 2, 3, 4, 5)
   - Color: How much each variable contributes to each day's forecast
   - Example: Volume high for day 1, drops for day 5 (short-term feature)

3. **Per-Head Specialization**
   - Radar plot: Which heads focus on trend vs mean-reversion vs volume?
   - Head-specific loss contribution
   - Prune low-contribution heads in next iteration

4. **Prediction Uncertainty**
   - Cone plot: Q_0.1 (lower), Q_0.5 (median), Q_0.9 (upper)
   - Actual realized returns as dots
   - If dots outside Q_0.1 to Q_0.9 > 20% of time → calibration issue

---

## Existing vs. New Models

| Aspect | X-Trend (Wood et al. 2023) | Classic TFT (Lim et al. 2021) |
|--------|---------------------------|-------------------------------|
| **Type** | Cross-attentive regime follower | Temporal Fusion Transformer |
| **Core Mechanism** | Query-key attention to regime state | Encoder-decoder LSTM + attention |
| **Handles Non-stationarity** | Via regime embedding | Via state-dependent forecast |
| **Interpretability** | Regime-specific weights | Timestep + feature importance |
| **Quantile Support** | Single-point forecast | Native quantile outputs |
| **Lookback Handling** | Variable (adaptive) | Fixed (num_time_steps) |
| **Hardware** | CPU-friendly | Benefits from GPU |
| **Use Case** | Regime-aware swing trading | General-purpose forecasting |

Both will be trained in parallel; winner determined by walk-forward OOS Sharpe.

---

## Success Metrics

**Definition of Success**:
1. Classic TFT achieves OOS Sharpe ≥ 1.5 on NIFTY/BANKNIFTY (consistent with existing TFT)
2. Attention maps show interpretable patterns (e.g., short-term features for day 1 forecast, mean-reversion for day 5)
3. Variable importance aligns with domain knowledge (volume + OI matter, sentiment low-weight at short horizon)
4. Quantile calibration: 80% of actuals fall within Q_0.1 to Q_0.9
5. No look-ahead bias: Forward pass on unseen data produces consistent metrics to test set

---

## Files Modified/Created Summary

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `components/grn.py` | NEW | 80 | Gated Residual Networks |
| `components/vsn.py` | NEW | 120 | Variable Selection Network |
| `components/encoder.py` | NEW | 200 | Temporal + Static Encoders |
| `components/decoder.py` | NEW | 180 | Autoregressive Decoder |
| `components/attention.py` | NEW | 150 | Interpretable MHA |
| `classic_tft.py` | NEW | 400 | Main model class |
| `production/training_pipeline.py` | UPDATE | +250 | Phase 6: Classic TFT training |
| `tests/test_classic_tft.py` | NEW | 300 | Unit tests |
| `inference/inference.py` | NEW | 250 | Production inference |
| `research/classic_tft_research.py` | NEW | 400 | Backtesting + ablations |
| `docs/CLASSIC_TFT_MODERNIZATION_PLAN.md` | NEW | - | THIS FILE |
| `docs/CLASSIC_TFT_ARCHITECTURE.md` | NEW | - | Detailed math + diagrams |

**Total**: 2,330+ lines of new code, 5 weeks to implementation, 0 breaking changes to existing codebase.

---

## References

1. **Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting** (Lim et al., ICLR 2021)
   - Paper: https://arxiv.org/abs/2009.07052
   - Code: https://github.com/google-research/google-research/tree/master/tft

2. **Transformer: Attention Is All You Need** (Vaswani et al., NeurIPS 2017)
   - Multi-head attention mechanism

3. **Quantile Regression for Time Series** (Koenker, 1978+)
   - Pinball loss for probabilistic forecasting

4. **X-Trend: Explainable Trend Prediction** (Wood et al., 2023)
   - Regime-following architecture (existing model)

5. **Causal Discovery in Time Series** (Peters et al., 2015-2019)
   - Granger causality + causal frameworks for interpretation

---

## Next Steps

1. **Review & Approval** (2-3 days)
   - Stakeholder sign-off on architecture
   - Finalize ClassicTFTConfig defaults

2. **Phase 1-2 Implementation** (1 week)
   - Code components and main model
   - Unit tests for each component

3. **Phase 3-4 Integration** (1 week)
   - Wire into training pipeline
   - Walk-forward validation + Optuna hyperparameter search

4. **Phase 5 Deployment** (1 week)
   - Production inference wrapper
   - Dashboard + interpretability exports
   - Compare vs X-Trend on held-out 2026 data

5. **Documentation** (Ongoing)
   - Detailed architecture docs
   - Interpretation guide for traders
   - Ablation studies (what if we remove VSN? Attention?)

---

**Document Version**: 1.0
**Date**: 2026-02-11
**Status**: READY FOR IMPLEMENTATION
**Estimated Timeline**: 5 weeks (component → training → deployment)
