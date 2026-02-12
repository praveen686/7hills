"""Classic Temporal Fusion Transformer (Lim et al. 2020).

Clean-room PyTorch implementation of the original TFT architecture for
multi-horizon time series forecasting. Designed for volatility prediction
with quantile outputs.

Key components:
1. Variable Selection Networks (VSN) -- per-variable GRNs + softmax weights
2. Static Covariate Encoder -- 4 context vectors from static inputs
3. Interpretable Multi-Head Attention -- shared V weights for interpretability
4. Gated skip connections + LayerNorm throughout
5. LSTM encoder-decoder for temporal processing
6. Quantile output head for prediction intervals

Reference: "Temporal Fusion Transformers for Interpretable Multi-horizon
Time Series Forecasting", Lim et al. 2020, arXiv:1912.09363
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports (same pattern as x_trend.py)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    _DEVICE = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ClassicTFTConfig:
    """Hyperparameters for the Classic TFT model.

    Attributes
    ----------
    d_hidden : int
        Hidden dimension throughout the model (paper default: 160).
    n_heads : int
        Number of attention heads.
    lstm_layers : int
        Number of LSTM layers in encoder and decoder.
    stack_size : int
        Number of self-attention stacks.
    dropout : float
        Dropout rate applied throughout.
    n_static_real : int
        Number of real-valued static inputs.
    n_static_cat : int
        Number of categorical static inputs (e.g. symbol ID).
    cat_cardinalities : list[int]
        Vocabulary size for each categorical static input.
    n_observed : int
        Number of past-only observed inputs (e.g. log_vol, returns).
    n_known : int
        Number of known future inputs (e.g. day_of_week, month).
    encoder_steps : int
        Length of the lookback / encoder window (paper: 252 for vol).
    decoder_steps : int
        Length of the forecast horizon.
    quantiles : list[float]
        Quantile levels for the output (e.g. [0.1, 0.5, 0.9]).
    output_size : int
        Number of target dimensions (usually 1).
    lr : float
        Learning rate.
    max_grad_norm : float
        Gradient clipping norm (paper default: 0.01).
    batch_size : int
        Training batch size.
    epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience (epochs without improvement).
    purge_gap : int
        Days between train end and test start to avoid look-ahead bias.
    """

    # Architecture
    d_hidden: int = 160
    n_heads: int = 4
    lstm_layers: int = 1
    stack_size: int = 1
    dropout: float = 0.3

    # Input structure (set by data formatter)
    n_static_real: int = 0
    n_static_cat: int = 0
    cat_cardinalities: list = field(default_factory=list)
    n_observed: int = 0
    n_known: int = 0

    # Temporal structure
    encoder_steps: int = 252
    decoder_steps: int = 5

    # Output
    quantiles: list = field(default_factory=lambda: [0.1, 0.5, 0.9])
    output_size: int = 1

    # Training
    lr: float = 0.001
    max_grad_norm: float = 0.01
    batch_size: int = 64
    epochs: int = 100
    patience: int = 5
    purge_gap: int = 5


# ============================================================================
# PyTorch modules (all behind HAS_TORCH guard)
# ============================================================================

if HAS_TORCH:

    class GatedResidualNetwork(nn.Module):
        """Gated Residual Network — GRN(a, c) = LN(a + GLU(W1·ELU(W2·a + W3·c)))."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.1,
            context_dim: Optional[int] = None,
        ):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim * 2)  # for GLU
            self.context_proj = (
                nn.Linear(context_dim, hidden_dim, bias=False)
                if context_dim is not None
                else None
            )
            self.skip = (
                nn.Linear(input_dim, output_dim)
                if input_dim != output_dim
                else nn.Identity()
            )
            self.layer_norm = nn.LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self, x: torch.Tensor, context: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            residual = self.skip(x)
            h = self.fc1(x)
            if self.context_proj is not None and context is not None:
                # Broadcast context to match h shape
                if context.dim() < h.dim():
                    context = context.unsqueeze(-2).expand_as(h)
                h = h + self.context_proj(context)
            h = F.elu(h)
            h = self.dropout(h)
            gate_input = self.fc2(h)
            value, gate = gate_input.chunk(2, dim=-1)
            h = value * torch.sigmoid(gate)
            return self.layer_norm(residual + h)

    # ------------------------------------------------------------------
    # GLU Gate + Add&Norm (used throughout TFT)
    # ------------------------------------------------------------------
    class GateAddNorm(nn.Module):
        """GLU gate -> dropout -> add residual -> LayerNorm.

        This is the standard gated skip connection used throughout TFT:
            output = LayerNorm(x + Dropout(GLU(W * input)))

        The GLU (Gated Linear Unit) splits its projection into value and gate
        halves, producing ``value * sigmoid(gate)``. This allows the network
        to learn which information to suppress entirely (gate near 0) versus
        pass through (gate near 1).

        Parameters
        ----------
        input_dim : int
            Dimension of the incoming tensor that gets projected.
        hidden_dim : int
            Target / residual dimension (also the LayerNorm dimension).
        dropout : float
            Dropout probability applied after the GLU gating.
        """

        def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
            super().__init__()
            self.fc = nn.Linear(input_dim, hidden_dim * 2)
            self.layer_norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
            """Apply gated skip connection.

            Parameters
            ----------
            x : torch.Tensor
                Input to be gated (any shape with last dim = input_dim).
            residual : torch.Tensor
                Skip connection tensor (same shape, last dim = hidden_dim).

            Returns
            -------
            torch.Tensor
                LayerNorm(residual + Dropout(GLU(Linear(x)))).
            """
            gate_input = self.fc(x)
            value, gate = gate_input.chunk(2, dim=-1)
            gated = self.dropout(value * torch.sigmoid(gate))
            return self.layer_norm(residual + gated)

    # ------------------------------------------------------------------
    # Interpretable Multi-Head Attention (paper Section 4.4)
    # ------------------------------------------------------------------
    class InterpretableMultiHeadAttention(nn.Module):
        """Multi-head attention with shared value weights across heads.

        Key innovation from the TFT paper: unlike standard multi-head attention
        where each head has its own V projection, here ALL heads share a SINGLE
        value projection. This makes the attention weights directly interpretable
        because each head attends to the same value representation.

        The per-head outputs are averaged (not concatenated) and then projected
        through a final linear layer, keeping the output dimension equal to
        ``d_model``.

        Parameters
        ----------
        n_heads : int
            Number of attention heads.
        d_model : int
            Model dimension (must be divisible by n_heads).
        dropout : float
            Attention dropout rate.
        """

        def __init__(self, n_heads: int, d_model: int, dropout: float = 0.0):
            super().__init__()
            assert d_model % n_heads == 0, (
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
            self.n_heads = n_heads
            self.d_model = d_model
            self.d_k = d_model // n_heads

            # Per-head Q and K projections
            self.q_projs = nn.ModuleList(
                [nn.Linear(d_model, self.d_k) for _ in range(n_heads)]
            )
            self.k_projs = nn.ModuleList(
                [nn.Linear(d_model, self.d_k) for _ in range(n_heads)]
            )
            # SHARED value projection across all heads (key paper innovation)
            self.v_proj = nn.Linear(d_model, self.d_k)

            # Output projection
            self.out_proj = nn.Linear(self.d_k, d_model)
            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.d_k)

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute interpretable multi-head attention.

            Parameters
            ----------
            query : torch.Tensor
                Shape ``(batch, T_q, d_model)``.
            key : torch.Tensor
                Shape ``(batch, T_k, d_model)``.
            value : torch.Tensor
                Shape ``(batch, T_k, d_model)``.
            mask : torch.Tensor, optional
                Causal mask of shape ``(T_q, T_k)``.  Positions with value 0
                are masked (filled with ``-inf`` before softmax).

            Returns
            -------
            output : torch.Tensor
                Shape ``(batch, T_q, d_model)``.
            attn_weights : torch.Tensor
                Shape ``(batch, n_heads, T_q, T_k)`` -- per-head attention
                weights, useful for interpretability analysis.
            """
            # Shared V across all heads: (batch, T_k, d_k)
            v = self.v_proj(value)

            head_outputs = []
            head_attns = []

            for h in range(self.n_heads):
                q_h = self.q_projs[h](query)   # (batch, T_q, d_k)
                k_h = self.k_projs[h](key)     # (batch, T_k, d_k)

                # Scaled dot-product attention
                scores = torch.bmm(q_h, k_h.transpose(1, 2)) / self.scale
                # scores: (batch, T_q, T_k)

                if mask is not None:
                    scores = scores.masked_fill(mask == 0, float("-inf"))

                attn = F.softmax(scores, dim=-1)
                attn = self.dropout(attn)
                head_attns.append(attn)

                # Apply attention to shared V
                out_h = torch.bmm(attn, v)  # (batch, T_q, d_k)
                head_outputs.append(out_h)

            # Average over heads (paper: mean aggregation for interpretability)
            stacked = torch.stack(head_outputs, dim=0)  # (n_heads, batch, T_q, d_k)
            mean_head = stacked.mean(dim=0)              # (batch, T_q, d_k)

            # Output projection
            output = self.out_proj(mean_head)  # (batch, T_q, d_model)

            # Stack attention weights for interpretability
            attn_weights = torch.stack(head_attns, dim=1)
            # attn_weights: (batch, n_heads, T_q, T_k)

            return output, attn_weights

    # ------------------------------------------------------------------
    # Static Covariate Encoder (paper Section 4.2)
    # ------------------------------------------------------------------
    class StaticCovariateEncoder(nn.Module):
        """Produces 4 context vectors from static features via separate GRNs.

        Each context vector serves a different conditioning role in the TFT:

        - ``c_s`` -- modulates temporal VSN feature selection
        - ``c_e`` -- enriches temporal features with static information
        - ``c_h`` -- initializes LSTM encoder hidden state
        - ``c_c`` -- initializes LSTM encoder cell state

        Parameters
        ----------
        d_hidden : int
            Hidden / model dimension.
        dropout : float
            Dropout rate for the GRNs.
        """

        def __init__(self, d_hidden: int, dropout: float = 0.1):
            super().__init__()
            self.grn_cs = GatedResidualNetwork(d_hidden, d_hidden, d_hidden, dropout)
            self.grn_ce = GatedResidualNetwork(d_hidden, d_hidden, d_hidden, dropout)
            self.grn_ch = GatedResidualNetwork(d_hidden, d_hidden, d_hidden, dropout)
            self.grn_cc = GatedResidualNetwork(d_hidden, d_hidden, d_hidden, dropout)

        def forward(
            self, static_rep: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Compute 4 context vectors from static representation.

            Parameters
            ----------
            static_rep : torch.Tensor
                Shape ``(batch, d_hidden)``.

            Returns
            -------
            c_s, c_e, c_h, c_c : tuple[torch.Tensor, ...]
                Each of shape ``(batch, d_hidden)``.
            """
            c_s = self.grn_cs(static_rep)
            c_e = self.grn_ce(static_rep)
            c_h = self.grn_ch(static_rep)
            c_c = self.grn_cc(static_rep)
            return c_s, c_e, c_h, c_c

    # ------------------------------------------------------------------
    # Temporal Variable Selection Network (paper Section 4.2)
    # ------------------------------------------------------------------
    class TemporalVSN(nn.Module):
        """Variable Selection Network for temporal (time-varying) inputs.

        Each input variable is individually projected to ``d_hidden`` and
        transformed through its own GRN.  A separate selection GRN (optionally
        conditioned on static context ``c_s``) produces softmax weights over
        variables, yielding an importance-weighted combination.

        This enables the model to *learn* which variables matter at each
        timestep, providing built-in feature importance.

        Parameters
        ----------
        n_variables : int
            Number of input variables.
        d_hidden : int
            Hidden / model dimension.
        dropout : float
            Dropout rate for the GRNs.
        """

        def __init__(self, n_variables: int, d_hidden: int, dropout: float = 0.1):
            super().__init__()
            self.n_variables = n_variables
            self.d_hidden = d_hidden

            # Per-variable input projection: scalar (1-dim) -> d_hidden
            self.var_transforms = nn.ModuleList(
                [nn.Linear(1, d_hidden) for _ in range(n_variables)]
            )

            # Per-variable GRNs
            self.var_grns = nn.ModuleList(
                [
                    GatedResidualNetwork(d_hidden, d_hidden, d_hidden, dropout)
                    for _ in range(n_variables)
                ]
            )

            # Selection weights GRN (flattened concatenated vars -> n_variables)
            # Conditioned on static context c_s via context_dim
            self.selection_grn = GatedResidualNetwork(
                n_variables * d_hidden,
                d_hidden,
                n_variables,
                dropout,
                context_dim=d_hidden,
            )
            self.softmax = nn.Softmax(dim=-1)

        def forward(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Apply variable selection.

            Parameters
            ----------
            x : torch.Tensor
                Shape ``(batch, T, n_variables)``.
            context : torch.Tensor, optional
                Static context ``c_s`` of shape ``(batch, d_hidden)``.

            Returns
            -------
            output : torch.Tensor
                Shape ``(batch, T, d_hidden)`` -- importance-weighted sum of
                per-variable GRN outputs.
            weights : torch.Tensor
                Shape ``(batch, T, n_variables)`` -- softmax selection weights
                (interpretable per-variable importance).
            """
            batch, T, n_var = x.shape

            # Transform each variable: (batch, T, 1) -> (batch, T, d_hidden)
            transformed = []
            for i in range(self.n_variables):
                xi = x[:, :, i : i + 1]                       # (batch, T, 1)
                proj_i = self.var_transforms[i](xi)            # (batch, T, d_hidden)
                grn_i = self.var_grns[i](proj_i)               # (batch, T, d_hidden)
                transformed.append(grn_i)

            # Stack: (batch, T, n_variables, d_hidden)
            stacked = torch.stack(transformed, dim=2)

            # Compute selection weights
            # Flatten variables: (batch, T, n_variables * d_hidden)
            flat = stacked.reshape(batch, T, -1)
            raw_weights = self.selection_grn(flat, context)    # (batch, T, n_variables)
            weights = self.softmax(raw_weights)                # (batch, T, n_variables)

            # Weighted combination over variables
            weights_expanded = weights.unsqueeze(-1)           # (batch, T, n_var, 1)
            output = (stacked * weights_expanded).sum(dim=2)   # (batch, T, d_hidden)

            return output, weights

    # ------------------------------------------------------------------
    # Classic TFT Model (paper Section 4)
    # ------------------------------------------------------------------
    class ClassicTFTModel(nn.Module):
        """Full Classic TFT architecture from Lim et al. 2020.

        Forward pass overview:

        1. Embed inputs (real -> Linear, categorical -> Embedding)
        2. Static VSN + Static Covariate Encoder -> 4 context vectors
        3. Temporal VSN (past inputs conditioned on c_s) -> encoder input
        4. Temporal VSN (future inputs conditioned on c_s) -> decoder input
        5. LSTM encoder (init from c_h, c_c) + LSTM decoder
        6. Gated skip connection on LSTM outputs
        7. Static enrichment layer (conditioned on c_e)
        8. Interpretable multi-head attention with causal mask
        9. Gated skip + Final GRN + Gated skip
        10. Quantile output from decoder timesteps

        Parameters
        ----------
        cfg : ClassicTFTConfig
            Model hyperparameters and input structure.
        """

        def __init__(self, cfg: ClassicTFTConfig):
            super().__init__()
            self.cfg = cfg
            d = cfg.d_hidden

            # ---- Input embeddings ----
            # Categorical static embeddings
            self.static_cat_embeddings = nn.ModuleList()
            for card in cfg.cat_cardinalities:
                self.static_cat_embeddings.append(nn.Embedding(card, d))

            # Real static input projections (per-variable -> d_hidden)
            self.static_real_projs = nn.ModuleList()
            if cfg.n_static_real > 0:
                for _ in range(cfg.n_static_real):
                    self.static_real_projs.append(nn.Linear(1, d))

            # Number of total static variable embeddings
            n_static_total = cfg.n_static_cat + cfg.n_static_real
            self._n_static_total = n_static_total

            if n_static_total > 0:
                # Static selection GRN: picks which static variables matter
                # Input: flattened (n_static_total * d_hidden) -> n_static_total
                self.static_selection_grn = GatedResidualNetwork(
                    n_static_total * d,
                    d,
                    n_static_total,
                    cfg.dropout,
                )
                self.static_selection_softmax = nn.Softmax(dim=-1)
            else:
                self.static_selection_grn = None
                # Learnable default static representation
                self.static_default = nn.Parameter(torch.randn(1, d) * 0.02)

            # Static covariate encoder (produces 4 context vectors)
            self.static_encoder = StaticCovariateEncoder(d, cfg.dropout)

            # ---- Temporal VSNs ----
            # Past VSN: observed + known + target (output_size)
            n_past_vars = cfg.n_observed + cfg.n_known + cfg.output_size
            self._n_past_vars = max(n_past_vars, 1)
            self.past_vsn = TemporalVSN(self._n_past_vars, d, cfg.dropout)

            # Future VSN: known-future variables only
            self._n_future_vars = max(cfg.n_known, 1)
            self.future_vsn = TemporalVSN(self._n_future_vars, d, cfg.dropout)

            # ---- LSTM Encoder-Decoder ----
            self.lstm_encoder = nn.LSTM(
                input_size=d,
                hidden_size=d,
                num_layers=cfg.lstm_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
            )
            self.lstm_decoder = nn.LSTM(
                input_size=d,
                hidden_size=d,
                num_layers=cfg.lstm_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
            )

            # Gate + LayerNorm after LSTM (gated skip over LSTM)
            self.post_lstm_gate = GateAddNorm(d, d, cfg.dropout)

            # ---- Static Enrichment (conditioned on c_e) ----
            self.static_enrichment = GatedResidualNetwork(
                d, d, d, cfg.dropout, context_dim=d
            )

            # ---- Self-Attention Stack ----
            self.attention_layers = nn.ModuleList()
            self.attention_gates = nn.ModuleList()
            for _ in range(cfg.stack_size):
                self.attention_layers.append(
                    InterpretableMultiHeadAttention(cfg.n_heads, d, cfg.dropout)
                )
                self.attention_gates.append(GateAddNorm(d, d, cfg.dropout))

            # ---- Final feedforward ----
            self.final_grn = GatedResidualNetwork(d, d, d, cfg.dropout)
            self.final_gate = GateAddNorm(d, d, cfg.dropout)

            # ---- Quantile output head ----
            n_quantiles = len(cfg.quantiles)
            self.output_proj = nn.Linear(d, cfg.output_size * n_quantiles)

            # Store temporal dims for slicing
            self.encoder_steps = cfg.encoder_steps
            self.decoder_steps = cfg.decoder_steps

        def _build_static_rep(
            self,
            batch: int,
            static_cat: Optional[torch.Tensor],
            static_real: Optional[torch.Tensor],
        ) -> torch.Tensor:
            """Embed and select static features into a single representation.

            Parameters
            ----------
            batch : int
                Batch size.
            static_cat : torch.Tensor or None
                Integer categories, shape ``(batch, n_static_cat)``.
            static_real : torch.Tensor or None
                Real-valued statics, shape ``(batch, n_static_real)``.

            Returns
            -------
            torch.Tensor
                Shape ``(batch, d_hidden)`` -- selected static representation.
            """
            if self.static_selection_grn is None:
                return self.static_default.expand(batch, -1)

            d = self.cfg.d_hidden
            static_embeds = []

            # Categorical embeddings
            if static_cat is not None:
                for i, emb_layer in enumerate(self.static_cat_embeddings):
                    static_embeds.append(emb_layer(static_cat[:, i]))

            # Real-valued projections (each scalar -> d_hidden)
            if static_real is not None and len(self.static_real_projs) > 0:
                for j, proj in enumerate(self.static_real_projs):
                    static_embeds.append(proj(static_real[:, j : j + 1]))

            if not static_embeds:
                return self.static_default.expand(batch, -1)

            # Stack: (batch, n_static_total, d_hidden)
            stacked = torch.stack(static_embeds, dim=1)

            # Flatten for selection GRN: (batch, n_static_total * d_hidden)
            flat = stacked.reshape(batch, -1)
            raw_weights = self.static_selection_grn(flat)
            # (batch, n_static_total)
            weights = self.static_selection_softmax(raw_weights)

            # Weighted combination: (batch, d_hidden)
            weights_exp = weights.unsqueeze(-1)           # (batch, n_static, 1)
            static_rep = (stacked * weights_exp).sum(dim=1)  # (batch, d_hidden)

            return static_rep

        def forward(
            self,
            past_inputs: torch.Tensor,
            future_inputs: torch.Tensor,
            static_cat: Optional[torch.Tensor] = None,
            static_real: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, dict]:
            """Full forward pass of the Classic TFT.

            Parameters
            ----------
            past_inputs : torch.Tensor
                Past observed + known + target values.
                Shape ``(batch, encoder_steps, n_observed + n_known + output_size)``.
            future_inputs : torch.Tensor
                Known future inputs (e.g. calendar features).
                Shape ``(batch, decoder_steps, n_known)``.
            static_cat : torch.Tensor, optional
                Categorical static features.  Shape ``(batch, n_static_cat)``,
                dtype ``int64``.
            static_real : torch.Tensor, optional
                Real-valued static features.  Shape ``(batch, n_static_real)``.

            Returns
            -------
            quantile_output : torch.Tensor
                Shape ``(batch, decoder_steps, output_size * n_quantiles)``.
            interpretability : dict
                Contains ``attention_weights``, ``past_vsn_weights``,
                ``future_vsn_weights``, and ``static_context`` for downstream
                analysis of learned variable importances and temporal patterns.
            """
            batch = past_inputs.size(0)
            d = self.cfg.d_hidden

            # ==== 1. Static processing ====
            static_rep = self._build_static_rep(batch, static_cat, static_real)
            # static_rep: (batch, d_hidden)

            # 4 context vectors from static representation
            c_s, c_e, c_h, c_c = self.static_encoder(static_rep)

            # ==== 2. Temporal VSN (Past) ====
            past_vsn_out, past_vsn_weights = self.past_vsn(past_inputs, c_s)
            # past_vsn_out:     (batch, encoder_steps, d_hidden)
            # past_vsn_weights: (batch, encoder_steps, n_past_vars)

            # ==== 3. Temporal VSN (Future) ====
            future_vsn_out, future_vsn_weights = self.future_vsn(
                future_inputs, c_s
            )
            # future_vsn_out:     (batch, decoder_steps, d_hidden)
            # future_vsn_weights: (batch, decoder_steps, n_future_vars)

            # ==== 4. LSTM Encoder ====
            # Initialize hidden state from static context
            h0 = c_h.unsqueeze(0).expand(
                self.cfg.lstm_layers, -1, -1
            ).contiguous()
            c0 = c_c.unsqueeze(0).expand(
                self.cfg.lstm_layers, -1, -1
            ).contiguous()

            encoder_out, (h_n, c_n) = self.lstm_encoder(past_vsn_out, (h0, c0))
            # encoder_out: (batch, encoder_steps, d)

            # ==== 5. LSTM Decoder (seeded from encoder final state) ====
            decoder_out, _ = self.lstm_decoder(future_vsn_out, (h_n, c_n))
            # decoder_out: (batch, decoder_steps, d)

            # ==== 6. Concat + Gated skip connection ====
            # Concatenate encoder and decoder outputs along time axis
            lstm_out = torch.cat([encoder_out, decoder_out], dim=1)
            # lstm_out: (batch, total_time, d)

            # Pre-LSTM temporal features (for skip connection)
            temporal_input = torch.cat([past_vsn_out, future_vsn_out], dim=1)
            # temporal_input: (batch, total_time, d)

            gated_lstm = self.post_lstm_gate(lstm_out, temporal_input)
            # gated_lstm: (batch, total_time, d)

            # ==== 7. Static Enrichment (conditioned on c_e) ====
            enriched = self.static_enrichment(gated_lstm, c_e)
            # enriched: (batch, total_time, d)

            # ==== 8. Self-Attention with causal mask ====
            total_time = self.encoder_steps + self.decoder_steps
            causal_mask = torch.tril(
                torch.ones(
                    total_time, total_time,
                    device=past_inputs.device, dtype=past_inputs.dtype,
                )
            )

            attn_input = enriched
            all_attn_weights = []
            for attn_layer, attn_gate in zip(
                self.attention_layers, self.attention_gates
            ):
                attn_out, attn_weights = attn_layer(
                    attn_input, attn_input, attn_input, mask=causal_mask
                )
                all_attn_weights.append(attn_weights)
                attn_input = attn_gate(attn_out, attn_input)

            # ==== 9. Final feedforward ====
            final_grn_out = self.final_grn(attn_input)
            output = self.final_gate(final_grn_out, attn_input)
            # output: (batch, total_time, d)

            # ==== 10. Quantile output (decoder timesteps only) ====
            decoder_output = output[:, self.encoder_steps :, :]
            # decoder_output: (batch, decoder_steps, d)

            quantile_out = self.output_proj(decoder_output)
            # quantile_out: (batch, decoder_steps, output_size * n_quantiles)

            interpretability = {
                "attention_weights": all_attn_weights,
                "past_vsn_weights": past_vsn_weights,
                "future_vsn_weights": future_vsn_weights,
                "static_context": {
                    "c_s": c_s,
                    "c_e": c_e,
                    "c_h": c_h,
                    "c_c": c_c,
                },
            }

            return quantile_out, interpretability

    # ------------------------------------------------------------------
    # Quantile Loss (paper Section 5.1)
    # ------------------------------------------------------------------
    class QuantileLoss(nn.Module):
        """Quantile (pinball) loss for multi-quantile predictions.

        For a given quantile level ``q``:

            L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))

        At ``q = 0.5`` this reduces to the MAE (symmetric penalty).
        At ``q < 0.5`` the loss penalizes over-prediction more heavily.
        At ``q > 0.5`` the loss penalizes under-prediction more heavily.

        Parameters
        ----------
        quantiles : list[float]
            Quantile levels (e.g. [0.1, 0.5, 0.9]).
        """

        def __init__(self, quantiles: list[float]):
            super().__init__()
            self.quantiles = quantiles
            self.register_buffer(
                "q_tensor", torch.tensor(quantiles, dtype=torch.float32)
            )

        def forward(
            self, predictions: torch.Tensor, targets: torch.Tensor
        ) -> torch.Tensor:
            """Compute mean quantile loss.

            Parameters
            ----------
            predictions : torch.Tensor
                Shape ``(batch, T, n_quantiles)``.
            targets : torch.Tensor
                Shape ``(batch, T, 1)`` or ``(batch, T)``.

            Returns
            -------
            torch.Tensor
                Scalar loss (mean over all samples, timesteps, and quantiles).
            """
            if targets.dim() == 2:
                targets = targets.unsqueeze(-1)

            # errors: (batch, T, n_quantiles) via broadcasting
            errors = targets - predictions

            # Quantile weights: (1, 1, n_quantiles)
            q = self.q_tensor.view(1, 1, -1)

            loss = torch.max(q * errors, (q - 1) * errors)
            return loss.mean()

    # ------------------------------------------------------------------
    # Normalized Quantile Loss (paper evaluation metric)
    # ------------------------------------------------------------------
    def normalized_quantile_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantile: float,
        quantile_idx: int,
    ) -> float:
        """Compute normalized quantile loss as defined in the TFT paper.

        NQL = 2 * sum(pinball_loss) / sum(|target|)

        This normalization makes the metric comparable across different
        time series with different scales.

        Parameters
        ----------
        predictions : torch.Tensor
            Shape ``(N, T, n_quantiles)``.
        targets : torch.Tensor
            Shape ``(N, T, 1)`` or ``(N, T)``.
        quantile : float
            Quantile level (e.g. 0.5, 0.9).
        quantile_idx : int
            Index into the quantile dimension of predictions.

        Returns
        -------
        float
            Normalized quantile loss.  Returns 0.0 if targets are all zero.
        """
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)

        pred_q = predictions[:, :, quantile_idx : quantile_idx + 1]
        errors = targets - pred_q

        pinball = torch.where(
            errors >= 0,
            quantile * errors,
            (quantile - 1) * errors,
        )

        denominator = torch.abs(targets).sum()
        if denominator < 1e-10:
            return 0.0

        return float((2.0 * pinball.sum() / denominator).detach())


else:
    # ---------------------------------------------------------------
    # No torch available -- provide stub classes so imports don't fail
    # ---------------------------------------------------------------
    class GateAddNorm:  # type: ignore[no-redef]
        """Stub (torch not available)."""
        pass

    class InterpretableMultiHeadAttention:  # type: ignore[no-redef]
        """Stub (torch not available)."""
        pass

    class StaticCovariateEncoder:  # type: ignore[no-redef]
        """Stub (torch not available)."""
        pass

    class TemporalVSN:  # type: ignore[no-redef]
        """Stub (torch not available)."""
        pass

    class ClassicTFTModel:  # type: ignore[no-redef]
        """Stub (torch not available)."""
        pass

    class QuantileLoss:  # type: ignore[no-redef]
        """Stub (torch not available)."""
        pass

    def normalized_quantile_loss(*args, **kwargs) -> float:  # type: ignore[no-redef]
        """Stub (torch not available)."""
        return 0.0
