import tensorflow as tf
from tensorflow.keras import layers, Model
from quantkubera.models.layers import (
    GatedResidualNetwork, 
    VariableSelectionNetwork, 
    InterpretableMultiHeadAttention, 
    GluLayer
)
from quantkubera.models.losses import SharpeLoss

@tf.keras.utils.register_keras_serializable()
class MomentumTransformer(Model):
    def __init__(self, 
                 time_steps,
                 input_size, 
                 output_size, 
                 hidden_size, 
                 num_heads, 
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.time_steps = time_steps
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # 1. Embeddings (Linear projection for continuous inputs)
        # We assume all inputs are continuous for now as per our feature engineering
        # If we had categorical, we'd need embedding layers.
        # Input shape: (batch, time, input_size)
        # We process each feature independently first? 
        # TFT typically embeds each feature to hidden_size.
        
        self.feature_embeddings = [
            layers.Dense(hidden_size) for _ in range(input_size)
        ]
        
        # 2. Variable Selection
        self.var_selection = VariableSelectionNetwork(
            num_inputs=input_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        # 3. LSTM Encoder/Decoder (Seq2Seq)
        # In TMT, they use a single LSTM layer for "temporal features"?
        # "lstm_layer = get_lstm(...)(input_embeddings)"
        self.lstm = layers.LSTM(
            hidden_size, 
            return_sequences=True, 
            dropout=dropout_rate
        )
        
        self.lstm_gate = GluLayer(hidden_size, dropout_rate=dropout_rate)
        self.lstm_norm = layers.LayerNormalization()
        
        # 4. Gated Residual Network (Post LSTM)
        # "enriched, _ = gated_residual_network(temporal_feature_layer...)"
        self.post_lstm_grn = GatedResidualNetwork(
            hidden_size, 
            dropout_rate=dropout_rate
        )
        
        # 5. Multi-Head Attention
        self.attention = InterpretableMultiHeadAttention(
            num_heads=num_heads, 
            d_model=hidden_size, 
            dropout_rate=dropout_rate
        )
        
        self.attn_gate = GluLayer(hidden_size, dropout_rate=dropout_rate)
        self.attn_norm = layers.LayerNormalization()
        
        # 6. Post Attention GRN
        self.post_attn_grn = GatedResidualNetwork(
            hidden_size, 
            dropout_rate=dropout_rate
        )
        
        self.output_gate = GluLayer(hidden_size)
        self.output_norm = layers.LayerNormalization()
        
        # 7. Final Output
        self.final_dense = layers.Dense(output_size, activation="tanh")

    def call(self, inputs, return_weights=False):
        # inputs: (batch, time, input_size)
        
        # A. Feature Embedding
        embeddings = []
        for i in range(self.input_size):
            feat = inputs[..., i:i+1]
            emb = self.feature_embeddings[i](feat)
            embeddings.append(emb)
            
        embedded_features = tf.stack(embeddings, axis=2)
        
        # B. Variable Selection
        selected_features, vsn_weights = self.var_selection(embedded_features)
        
        # C. LSTM
        lstm_out = self.lstm(selected_features)
        
        # D. GRN + Skip
        lstm_gated, _ = self.lstm_gate(lstm_out)
        temporal_features = self.lstm_norm(selected_features + lstm_gated)
        
        # E. Enrichment
        enriched = self.post_lstm_grn(temporal_features)
        
        # F. Attention
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        mask = tf.linalg.band_part(tf.ones((batch_size, time_steps, time_steps)), -1, 0)
        
        attn_out, attn_weights = self.attention(enriched, enriched, enriched, mask=mask)
        
        # G. Residual + Gate
        attn_gated, _ = self.attn_gate(attn_out)
        attn_layer = self.attn_norm(enriched + attn_gated)
        
        # H. Non-Linear + Final Skip
        non_linear = self.post_attn_grn(attn_layer)
        non_linear_gated, _ = self.output_gate(non_linear)
        transformer_out = self.output_norm(temporal_features + non_linear_gated)
        
        # I. Output
        output = self.final_dense(transformer_out)
        
        if return_weights:
            return output, {
                'vsn_weights': vsn_weights,
                'attn_weights': attn_weights
            }
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "time_steps": self.time_steps,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config
