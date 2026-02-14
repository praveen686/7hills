import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

@tf.keras.utils.register_keras_serializable()
class GluLayer(layers.Layer):
    """Gated Linear Unit (GLU) Layer."""
    def __init__(self, hidden_size, dropout_rate=None, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate else None
        self.activation_layer = layers.Dense(hidden_size, activation=activation)
        self.gated_layer = layers.Dense(hidden_size, activation="sigmoid")

    def call(self, inputs):
        x = inputs
        if self.dropout:
            x = self.dropout(x)
            
        act = self.activation_layer(x)
        gate = self.gated_layer(x)
        return layers.Multiply()([act, gate]), gate

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class GatedResidualNetwork(layers.Layer):
    """Gated Residual Network (GRN)."""
    def __init__(self, hidden_size, output_size=None, dropout_rate=None, context_size=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.dropout_rate = dropout_rate
        self.context_size = context_size
        
        # Layers
        self.skip_layer = None  # Will be created in call if needed
        
        self.layer1 = layers.Dense(hidden_size)
        self.layer2 = layers.Dense(hidden_size)
        
        self.context_layer = layers.Dense(hidden_size, use_bias=False) if context_size else None
        
        self.glu = GluLayer(self.output_size, dropout_rate=dropout_rate)
        self.add = layers.Add()
        self.norm = layers.LayerNormalization()

    def call(self, inputs, context=None, return_gate=False):
        # Dynamically create skip layer if needed
        if self.skip_layer is None and inputs.shape[-1] != self.output_size:
            self.skip_layer = layers.Dense(self.output_size)
        
        # skip connection
        if self.skip_layer:
            skip = self.skip_layer(inputs)
        else:
            skip = inputs
            
        # Feed Forward
        x = self.layer1(inputs)
        if context is not None and self.context_layer:
            x = x + self.context_layer(context)
            
        x = tf.nn.elu(x)  # Direct function call instead of Activation layer
        x = self.layer2(x)
        
        # GLU
        glu_out, gate = self.glu(x)
        
        # Add & Norm
        out = self.add([skip, glu_out])
        out = self.norm(out)
        
        if return_gate:
            return out, gate
        return out
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "dropout_rate": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class VariableSelectionNetwork(layers.Layer):
    """Variable Selection Network."""
    def __init__(self, num_inputs, hidden_size, dropout_rate=None, context_size=None, **kwargs):
        super().__init__(**kwargs)
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # GRN for variable selection (input is flattened context + flattened inputs potentially)
        # Actually in TFT, the selection GRN takes the flattened inputs/state and produces weights
        self.selection_grn = GatedResidualNetwork(hidden_size, output_size=num_inputs, dropout_rate=dropout_rate, context_size=context_size)
        self.softmax_weighting = layers.Softmax(axis=-1)
        
        # Individual GRNs for each input
        self.input_grns = [
            GatedResidualNetwork(hidden_size, dropout_rate=dropout_rate) 
            for _ in range(num_inputs)
        ]

    def call(self, inputs, context=None):
        # inputs: (batch, time, num_inputs, input_dim) or (batch, num_inputs, input_dim)
        # We assume inputs are ALREADY embedded to (..., num_inputs, hidden_size)
        
        # 1. Flatten for selection (batch, ..., num_inputs * hidden_size)
        # But wait, TFT paper says selection weights are based on state.
        # Implementation in mom_trans uses:
        # flatten = tf.reshape(embedding, ...) -> GRN -> softmax weights
        
        input_shape = tf.shape(inputs)
        # If time distributed: (batch, time, num_inputs, hidden_size)
        # If static: (batch, num_inputs, hidden_size)
        
        # Flatten the last two dims for the selection network
        # (batch, time, num_inputs * hidden_size)
        flat_last_dim = self.num_inputs * self.hidden_size
        
        if len(inputs.shape) == 4: # Time distributed
             flat = tf.reshape(inputs, tf.concat([input_shape[:-2], [flat_last_dim]], axis=0))
        else: # Static
             flat = tf.reshape(inputs, tf.concat([input_shape[:-2], [flat_last_dim]], axis=0))

        # Calculate weights
        # selection_grn returns (batch, ..., num_inputs)
        weights_logits, _ = self.selection_grn(flat, context=context, return_gate=True)
        weights = self.softmax_weighting(weights_logits)
        
        # Expand weights for broadcasting: (batch, ..., num_inputs, 1)
        weights_expanded = tf.expand_dims(weights, axis=-1)
        
        # Process each input feature independently
        processed_features = []
        for i in range(self.num_inputs):
            # slice: (batch, ..., hidden_size)
            # define slice per input 
            feat = inputs[..., i, :] 
            processed = self.input_grns[i](feat)
            processed_features.append(processed)
            
        # Stack back: (batch, ..., num_inputs, hidden_size)
        processed_stack = tf.stack(processed_features, axis=-2)
        
        # Weighted sum: sum( weights * processed ) -> (batch, ..., hidden_size)
        weighted_out = tf.reduce_sum(weights_expanded * processed_stack, axis=-2)
        
        return weighted_out, weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_inputs": self.num_inputs,
            "hidden_size": self.hidden_size,
            "dropout_rate": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class ScaledDotProductAttention(layers.Layer):
    """Scaled Dot Product Attention."""
    def __init__(self, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dropout = layers.Dropout(dropout_rate)
        self.activation = layers.Activation("softmax")

    def call(self, q, k, v, mask=None):
        # q, k, v: (batch, time, heads, head_dim) (if transposed before? typically q is (batch, time, dim))
        # Logic from mom_trans: 
        # attn = batch_dot(q, k) / sqrt(dim)
        
        d = tf.shape(q)[-1]
        temper = tf.sqrt(tf.cast(d, tf.float32))
        
        # (batch, q_time, dim) x (batch, k_time, dim).T -> (batch, q_time, k_time)
        attn = tf.matmul(q, k, transpose_b=True) / temper
        
        if mask is not None:
             # Mask is (batch, time, time) or broadcastable
             # 0 for keep, 1 for mask? mom_trans uses -1e9 for mask
             # Let's assume mask passed is 1 for valid, 0 for invalid, so (1-mask)*-1e9
             # Re-checking mom_trans: mask=cumsum(eye). It seems it masks FUTURE.
             # mom_trans: lambda x: (-1e9) * (1.0 - tf.cast(x, "float32"))
             mmask = (1.0 - tf.cast(mask, tf.float32)) * -1e9
             attn = attn + mmask
             
        attn = self.activation(attn)
        attn = self.dropout(attn)
        
        # (batch, q_time, k_time) x (batch, k_time, dim) -> (batch, q_time, dim)
        output = tf.matmul(attn, v)
        return output, attn


@tf.keras.utils.register_keras_serializable()
class InterpretableMultiHeadAttention(layers.Layer):
    """Interpretable Multi-Head Attention (heads share value matrix)."""
    def __init__(self, num_heads, d_model, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        
        self.d_head = d_model // num_heads
        
        self.q_layers = [layers.Dense(self.d_head, use_bias=False) for _ in range(num_heads)]
        self.k_layers = [layers.Dense(self.d_head, use_bias=False) for _ in range(num_heads)]
        self.v_layer = layers.Dense(self.d_head, use_bias=False) # Shared V
        
        self.attention = ScaledDotProductAttention(dropout_rate)
        self.w_o = layers.Dense(d_model, use_bias=False)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, q, k, v, mask=None):
        heads = []
        attns = []
        
        # Shared Values
        vs = self.v_layer(v) # (batch, time, d_head)
        
        for i in range(self.num_heads):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            
            head, attn = self.attention(qs, ks, vs, mask)
            heads.append(head) # (batch, time, d_head)
            attns.append(attn) # (batch, time, time)
            
        # Stack heads
        # If num_heads > 1, mom_trans does mean?
        # mom_trans: outputs = K.mean(head, axis=0) if n_head > 1 ??? 
        # Wait, usually MultiHead concat. 
        # TMT Interpretable MHA: "Ensemble of heads". 
        # Paper: H_tilde = (1/H) * sum(Attention(Q,K,V, head_h) * W_H) ??
        # Code line 311: Lambda(K.mean, arguments={"axis": 0})(head) if n_head > 1 else head
        # It stacks in axis=0? (heads, batch, ...) then mean?
        # My implementation of heads.append creates a list.
        # If I stack them: (heads, batch, time, d_head) -> mean -> (batch, time, d_head)
        # Then w_o projects back to d_model. 
        # d_model is usually d_head * num_heads.
        # If d_head = d_model // num_heads, then we have dimensionality mismatch if we just average?
        # Let's re-read the code carefully.
        # d_v = d_model // n_head.
        # outputs = self.w_o(outputs). w_o is Dense(d_model).
        # So outputs of loop must be size d_head?
        # Yes, vs is d_head.
        # So yes, they AVERAGE the heads, they don't CONCATENATE them. This is unique to proper Interpretation.
        
        if self.num_heads > 1:
            stacked_heads = tf.stack(heads, axis=0) # (num_heads, batch, time, d_head)
            averaged_head = tf.reduce_mean(stacked_heads, axis=0) # (batch, time, d_head)
        else:
            averaged_head = heads[0]
            
        outputs = self.w_o(averaged_head) # (batch, time, d_model)
        outputs = self.dropout(outputs)
        
        return outputs, tf.stack(attns, axis=1) # (batch, heads, time, time)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "d_model": self.d_model,
            "dropout_rate": self.dropout_rate,
        })
        return config
