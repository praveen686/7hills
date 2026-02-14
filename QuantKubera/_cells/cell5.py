# ============================================================================
# CELL 5: MOMENTUM TRANSFORMER (Temporal Fusion Transformer Architecture)
# ============================================================================
# Implements the full TFT architecture from Lim et al. (2021) adapted for
# momentum signal generation with Sharpe ratio loss.
# All layers are serializable with proper get_config() methods.
# ============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


@tf.keras.utils.register_keras_serializable(package="MomentumTFT")
class GluLayer(layers.Layer):
    """Gated Linear Unit: splits transformation into value and gate streams.
    
    GLU(x) = Dense_value(x) * sigmoid(Dense_gate(x))
    
    The gate learns which components of the transformation to pass through,
    providing a learnable skip-like mechanism at the feature level.
    """
    
    def __init__(self, hidden_size, dropout_rate=None, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        
        # Value stream: optional activation (e.g., ELU for GRN intermediate)
        self.dense_value = layers.Dense(hidden_size, activation=activation)
        # Gate stream: always sigmoid for gating
        self.dense_gate = layers.Dense(hidden_size, activation='sigmoid')
        
        self.dropout_layer = None
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout_layer = layers.Dropout(dropout_rate)
    
    def call(self, inputs):
        value = self.dense_value(inputs)
        gate = self.dense_gate(inputs)
        
        if self.dropout_layer is not None:
            value = self.dropout_layer(value)
        
        glu_output = value * gate
        return glu_output, gate
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_name,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MomentumTFT")
class GatedResidualNetwork(layers.Layer):
    """Gated Residual Network: the core building block of TFT.
    
    Architecture:
        eta_1 = Dense(hidden_size)(input)             [+ Dense(hidden_size)(context) if context]
        eta_2 = ELU(eta_1)
        eta_1_prime = Dense(hidden_size)(eta_2)
        glu_output = GLU(eta_1_prime)
        output = LayerNorm(input_skip + glu_output)
    
    When output_size != input_size, a skip projection is applied to the input.
    """
    
    def __init__(self, hidden_size, output_size=None, dropout_rate=None,
                 context_size=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.dropout_rate = dropout_rate
        self.context_size = context_size
        
        # Primary pathway
        self.dense1 = layers.Dense(hidden_size)
        self.dense2 = layers.Dense(self.output_size)
        self.glu = GluLayer(self.output_size, dropout_rate=dropout_rate)
        self.layer_norm = layers.LayerNormalization()
        
        # Optional context injection
        self.context_dense = None
        if context_size is not None:
            self.context_dense = layers.Dense(hidden_size, use_bias=False)
        
        # Skip projection (created dynamically if needed)
        self._skip_layer = None
        self._skip_built = False
    
    def call(self, inputs, context=None, return_gate=False):
        # Build skip projection on first call if input dim != output_size
        if not self._skip_built:
            input_dim = inputs.shape[-1]
            if input_dim is not None and input_dim != self.output_size:
                self._skip_layer = layers.Dense(self.output_size)
            self._skip_built = True
        
        # Skip connection
        if self._skip_layer is not None:
            skip = self._skip_layer(inputs)
        else:
            skip = inputs
        
        # Primary pathway
        eta_1 = self.dense1(inputs)
        
        # Context injection (additive)
        if context is not None and self.context_dense is not None:
            eta_1 = eta_1 + self.context_dense(context)
        
        # ELU activation (using tf.nn.elu as specified)
        eta_2 = tf.nn.elu(eta_1)
        
        # Second dense
        eta_1_prime = self.dense2(eta_2)
        
        # GLU gating
        glu_output, gate = self.glu(eta_1_prime)
        
        # Residual connection + layer norm
        output = self.layer_norm(skip + glu_output)
        
        if return_gate:
            return output, gate
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'context_size': self.context_size,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MomentumTFT")
class VariableSelectionNetwork(layers.Layer):
    """Variable Selection Network: soft feature importance via learned weights.
    
    Each input feature is processed by its own GRN, then a selection GRN
    produces softmax weights over features. The output is the weighted
    combination of per-feature GRN outputs.
    
    This is the TFT's primary interpretability mechanism -- the softmax
    weights directly indicate feature importance at each timestep.
    
    Input shape:  (batch, time, num_inputs, hidden_size) -- after embedding
    Output shape: (batch, time, hidden_size)
    Weights shape: (batch, time, num_inputs, 1)
    """
    
    def __init__(self, num_inputs, hidden_size, dropout_rate=None,
                 context_size=None, **kwargs):
        super().__init__(**kwargs)
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.context_size = context_size
        
        # Per-feature GRNs: each processes one feature independently
        self.feature_grns = [
            GatedResidualNetwork(
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout_rate=dropout_rate,
                name=f"feature_grn_{i}"
            )
            for i in range(num_inputs)
        ]
        
        # Selection GRN: produces weights over features
        # Input is flattened features: (batch, time, num_inputs * hidden_size)
        # Output: (batch, time, num_inputs) -> softmax -> weights
        self.selection_grn = GatedResidualNetwork(
            hidden_size=hidden_size,
            output_size=num_inputs,
            dropout_rate=dropout_rate,
            context_size=context_size,
            name="selection_grn"
        )
    
    def call(self, inputs, context=None):
        # inputs: (batch, time, num_inputs, hidden_size)
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        # Flatten for selection GRN: (batch, time, num_inputs * hidden_size)
        flattened = tf.reshape(inputs, [batch_size, time_steps,
                                        self.num_inputs * self.hidden_size])
        
        # Selection weights via GRN + softmax
        selection_output = self.selection_grn(flattened, context=context)
        # selection_output: (batch, time, num_inputs)
        weights = tf.nn.softmax(selection_output, axis=-1)
        # weights: (batch, time, num_inputs)
        weights_expanded = tf.expand_dims(weights, axis=-1)
        # weights_expanded: (batch, time, num_inputs, 1)
        
        # Process each feature through its own GRN
        processed_features = []
        for i in range(self.num_inputs):
            # Extract feature i: (batch, time, hidden_size)
            feat_i = inputs[:, :, i, :]
            # Apply per-feature GRN
            grn_out = self.feature_grns[i](feat_i)
            processed_features.append(grn_out)
        
        # Stack: (batch, time, num_inputs, hidden_size)
        stacked = tf.stack(processed_features, axis=2)
        
        # Weighted combination: (batch, time, hidden_size)
        selected = tf.reduce_sum(stacked * weights_expanded, axis=2)
        
        return selected, weights_expanded
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_inputs': self.num_inputs,
            'hidden_size': self.hidden_size,
            'dropout_rate': self.dropout_rate,
            'context_size': self.context_size,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MomentumTFT")
class ScaledDotProductAttention(layers.Layer):
    """Standard scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Supports optional causal masking (lower-triangular) to prevent
    attending to future positions.
    """
    
    def __init__(self, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dropout_layer = layers.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def call(self, q, k, v, mask=None):
        # q, k, v: (batch, ..., seq_len, d_k)
        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        
        # Scaled dot product: (batch, ..., seq_q, seq_k)
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(d_k)
        
        # Apply mask (e.g., causal mask): masked positions get -1e9
        if mask is not None:
            scores = scores + (1.0 - mask) * (-1e9)
        
        # Softmax over keys
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        if self.dropout_layer is not None:
            attention_weights = self.dropout_layer(attention_weights)
        
        # Weighted sum of values
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dropout_rate': self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MomentumTFT")
class InterpretableMultiHeadAttention(layers.Layer):
    """Interpretable Multi-Head Attention from TFT paper.
    
    Key differences from standard Transformer MHA:
    1. All heads SHARE the same value projection (W_v)
    2. Head outputs are AVERAGED, not concatenated
    
    This design enables direct interpretation of attention patterns
    because each head attends to the same value representation.
    The averaged attention weights can be examined per-head to understand
    what temporal patterns the model has learned.
    """
    
    def __init__(self, num_heads, d_model, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.d_head = d_model // num_heads
        
        # Per-head Q and K projections
        self.W_q = [layers.Dense(self.d_head, use_bias=False, name=f"W_q_{i}")
                     for i in range(num_heads)]
        self.W_k = [layers.Dense(self.d_head, use_bias=False, name=f"W_k_{i}")
                     for i in range(num_heads)]
        
        # SHARED value projection across all heads
        self.W_v = layers.Dense(self.d_head, use_bias=False, name="W_v_shared")
        
        # Output projection
        self.W_o = layers.Dense(d_model, name="W_o")
        
        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout_rate=dropout_rate)
    
    def call(self, q, k, v, mask=None):
        # Shared value projection: (batch, seq, d_head)
        v_shared = self.W_v(v)
        
        head_outputs = []
        head_attentions = []
        
        for i in range(self.num_heads):
            # Per-head query and key projections
            q_i = self.W_q[i](q)  # (batch, seq_q, d_head)
            k_i = self.W_k[i](k)  # (batch, seq_k, d_head)
            
            # Attention with shared values
            attn_output, attn_weights = self.attention(q_i, k_i, v_shared, mask=mask)
            head_outputs.append(attn_output)
            head_attentions.append(attn_weights)
        
        # AVERAGE head outputs (not concatenate) -- key TFT design choice
        # Stack: (num_heads, batch, seq, d_head)
        stacked_outputs = tf.stack(head_outputs, axis=0)
        averaged = tf.reduce_mean(stacked_outputs, axis=0)  # (batch, seq, d_head)
        
        # Output projection back to d_model
        output = self.W_o(averaged)  # (batch, seq, d_model)
        
        # Stack attention weights for interpretability: (batch, num_heads, seq_q, seq_k)
        stacked_attentions = tf.stack(head_attentions, axis=1)
        
        return output, stacked_attentions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MomentumTFT")
class MomentumTransformer(Model):
    """Complete Temporal Fusion Transformer for momentum signal generation.
    
    Architecture flow:
    1. Feature Embedding: per-feature Dense projections to hidden_size
    2. Variable Selection: learned soft feature importance via VSN
    3. LSTM Encoder: capture temporal dependencies
    4. Post-LSTM: GRN + GLU gate + skip + LayerNorm
    5. Interpretable Multi-Head Self-Attention with causal mask
    6. Post-Attention: GRN + GLU gate + skip + LayerNorm
    7. Output: Dense(tanh) -> signal in [-1, 1]
    
    The model is fully interpretable: VSN weights show feature importance,
    attention weights show temporal dependencies.
    """
    
    def __init__(self, time_steps, input_size, output_size, hidden_size,
                 num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.time_steps = time_steps
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # 1. Per-feature embedding layers
        self.feature_embeddings = [
            layers.Dense(hidden_size, name=f"feat_embed_{i}")
            for i in range(input_size)
        ]
        
        # 2. Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            num_inputs=input_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            name="vsn"
        )
        
        # 3. LSTM Encoder
        self.lstm = layers.LSTM(
            hidden_size,
            return_sequences=True,
            dropout=dropout_rate if dropout_rate else 0.0,
            name="lstm_encoder"
        )
        
        # 4. Post-LSTM processing
        self.post_lstm_grn = GatedResidualNetwork(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            name="post_lstm_grn"
        )
        self.post_lstm_glu = GluLayer(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            name="post_lstm_glu"
        )
        self.post_lstm_norm = layers.LayerNormalization(name="post_lstm_norm")
        
        # 5. Interpretable Multi-Head Self-Attention
        self.mha = InterpretableMultiHeadAttention(
            num_heads=num_heads,
            d_model=hidden_size,
            dropout_rate=dropout_rate,
            name="interpretable_mha"
        )
        
        # 6. Post-attention processing
        self.post_attn_grn = GatedResidualNetwork(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            name="post_attn_grn"
        )
        self.post_attn_glu = GluLayer(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            name="post_attn_glu"
        )
        self.post_attn_norm = layers.LayerNormalization(name="post_attn_norm")
        
        # 7. Output layer
        self.output_dense = layers.Dense(
            output_size,
            activation='tanh',
            name="output_signal"
        )
    
    def call(self, x, return_weights=False):
        # x shape: (batch, time_steps, input_size)
        
        # --- 1. Feature Embedding ---
        # Project each feature to hidden_size independently
        embedded = [self.feature_embeddings[i](x[:, :, i:i+1])
                     for i in range(self.input_size)]
        # Stack: (batch, time, num_features, hidden_size)
        embedded = tf.stack(embedded, axis=2)
        
        # --- 2. Variable Selection ---
        vsn_output, vsn_weights = self.vsn(embedded)
        # vsn_output: (batch, time, hidden_size)
        
        # --- 3. LSTM Encoder ---
        lstm_output = self.lstm(vsn_output)
        # lstm_output: (batch, time, hidden_size)
        
        # --- 4. Post-LSTM: GRN + GLU gate + residual + LayerNorm ---
        post_lstm = self.post_lstm_grn(lstm_output)
        post_lstm_gated, _ = self.post_lstm_glu(post_lstm)
        post_lstm_out = self.post_lstm_norm(vsn_output + post_lstm_gated)
        
        # --- 5. Causal Self-Attention ---
        seq_len = tf.shape(post_lstm_out)[1]
        causal_mask = tf.linalg.band_part(
            tf.ones([seq_len, seq_len]), -1, 0
        )  # Lower triangular: position i can attend to positions <= i
        causal_mask = tf.expand_dims(tf.expand_dims(causal_mask, 0), 0)
        # (1, 1, seq, seq) -- broadcasts over batch and heads
        
        attn_output, attn_weights = self.mha(
            post_lstm_out, post_lstm_out, post_lstm_out, mask=causal_mask
        )
        
        # --- 6. Post-Attention: GRN + GLU gate + residual + LayerNorm ---
        post_attn = self.post_attn_grn(attn_output)
        post_attn_gated, _ = self.post_attn_glu(post_attn)
        post_attn_out = self.post_attn_norm(post_lstm_out + post_attn_gated)
        
        # --- 7. Output signal ---
        signal = self.output_dense(post_attn_out)
        # signal: (batch, time, output_size) with values in [-1, 1]
        
        if return_weights:
            return signal, {
                'vsn_weights': vsn_weights,   # (batch, time, num_inputs, 1)
                'attn_weights': attn_weights,  # (batch, num_heads, seq, seq)
            }
        return signal
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'time_steps': self.time_steps,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MomentumTFT")
class SharpeLoss(tf.keras.losses.Loss):
    """Negative annualized Sharpe ratio as differentiable loss function.
    
    loss = -(mean(signal * return) / std(signal * return)) * sqrt(252)
    
    Uses ddof=1 (Bessel's correction) for unbiased sample standard deviation.
    The model learns to produce signals that maximize risk-adjusted returns
    directly, rather than optimizing a proxy like MSE.
    """
    
    def __init__(self, output_size=1, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
    
    def call(self, y_true, y_pred):
        # y_true: actual returns, y_pred: predicted signals
        # Both: (batch, time, 1) or (batch, time)
        strategy_returns = y_pred * y_true
        
        # Flatten to compute portfolio-level Sharpe
        strategy_returns = tf.reshape(strategy_returns, [-1])
        
        mean_ret = tf.reduce_mean(strategy_returns)
        n = tf.cast(tf.size(strategy_returns), tf.float32)
        
        # Unbiased variance with ddof=1: sum((x - mean)^2) / (n - 1)
        var = tf.reduce_sum(tf.square(strategy_returns - mean_ret)) / (n - 1.0)
        std = tf.sqrt(var + 1e-9)  # Small epsilon for numerical stability
        
        # Negative annualized Sharpe (negative because we minimize loss)
        sharpe = (mean_ret / std) * tf.sqrt(252.0)
        return -sharpe
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'output_size': self.output_size,
        })
        return config


# --- Build and verify the model ---
print("=" * 70)
print("MOMENTUM TRANSFORMER (TFT Architecture)")
print("=" * 70)

# Configuration
TIME_STEPS = 20
INPUT_SIZE = 8    # Number of features
OUTPUT_SIZE = 1   # Single signal output
HIDDEN_SIZE = 32  # Hidden dimension
NUM_HEADS = 4     # Attention heads
DROPOUT_RATE = 0.1

# Build model
model = MomentumTransformer(
    time_steps=TIME_STEPS,
    input_size=INPUT_SIZE,
    output_size=OUTPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    dropout_rate=DROPOUT_RATE,
)

# Test forward pass with dummy data
dummy_x = tf.random.normal([16, TIME_STEPS, INPUT_SIZE])
dummy_y = tf.random.normal([16, TIME_STEPS, OUTPUT_SIZE])

# Forward pass with weights
signal, weights = model(dummy_x, return_weights=True)
print(f"\nModel architecture verified:")
print(f"  Input shape:       {dummy_x.shape}")
print(f"  Signal shape:      {signal.shape}")
print(f"  VSN weights shape: {weights['vsn_weights'].shape}")
print(f"  Attn weights shape:{weights['attn_weights'].shape}")
print(f"  Signal range:      [{tf.reduce_min(signal):.4f}, {tf.reduce_max(signal):.4f}]")

# Verify causal masking: attention at position i should have zero weight for j > i
attn = weights['attn_weights'][0, 0].numpy()  # First sample, first head
upper_triangle_sum = np.triu(attn, k=1).sum()
print(f"\n  Causal mask check (upper triangle sum): {upper_triangle_sum:.10f}")
assert upper_triangle_sum < 1e-6, "Causal masking is broken!"
print(f"  Causal masking: VERIFIED")

# Test Sharpe loss
loss_fn = SharpeLoss(output_size=OUTPUT_SIZE)
loss_val = loss_fn(dummy_y, signal)
print(f"\n  Sharpe loss value: {loss_val:.4f}")
print(f"  (Negative = model has positive Sharpe)")

# Compile and do one training step to verify gradients flow
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=SharpeLoss(output_size=OUTPUT_SIZE)
)

# Single training step
history = model.fit(dummy_x, dummy_y, epochs=1, batch_size=16, verbose=0)
print(f"  Training step loss: {history.history['loss'][0]:.4f}")

# Count parameters
total_params = model.count_params()
print(f"\n  Total parameters: {total_params:,}")

# VSN feature importance (interpretability demo)
vsn_w = weights['vsn_weights'].numpy()
mean_importance = vsn_w.mean(axis=(0, 1)).flatten()
feature_names = [f"feat_{i}" for i in range(INPUT_SIZE)]
importance_order = np.argsort(-mean_importance)
print(f"\n  Feature importance (VSN weights, descending):")
for rank, idx in enumerate(importance_order[:5]):
    print(f"    {rank+1}. {feature_names[idx]}: {mean_importance[idx]:.4f}")

# Serialization verification
config = model.get_config()
print(f"\n  Serialization config keys: {sorted(config.keys())}")
print(f"  Config: time_steps={config.get('time_steps')}, "
      f"input_size={config.get('input_size')}, "
      f"hidden_size={config.get('hidden_size')}, "
      f"num_heads={config.get('num_heads')}")

print("\n" + "=" * 70)
print("Momentum Transformer built, tested, and verified.")
print("=" * 70)