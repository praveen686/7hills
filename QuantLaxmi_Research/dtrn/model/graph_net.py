"""Graph Neural Network with regime and policy heads.

Architecture:
1. Input embedding: phi([x_t, m_t]) -> h_t^(0) per node
2. L rounds of message passing on dynamic adjacency A_t
3. Graph pooling -> regime posterior pi_t
4. GRU temporal memory over pooled states
5. Policy head -> position target p_t in [-1, 1]
6. Regime-conditioned gating on position
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEmbedding(nn.Module):
    """Embed raw features + mask into node representations."""

    def __init__(self, n_features: int, d_embed: int):
        super().__init__()
        # Input: concatenate feature value + mask bit per node
        # So each "node" gets its feature value (1) + mask (1) = 2 inputs
        # But we treat ALL features as a single graph, so input per node = 2
        self.embed = nn.Sequential(
            nn.Linear(2, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d) feature values
        mask: (B, d) binary mask
        returns: (B, d, d_embed) per-node embeddings
        """
        B, d = x.shape
        # Stack feature value and mask for each node
        node_input = torch.stack([x, mask], dim=-1)  # (B, d, 2)
        return self.embed(node_input)  # (B, d, d_embed)


class GraphMessagePass(nn.Module):
    """One round of message passing on weighted adjacency."""

    def __init__(self, d_hidden: int):
        super().__init__()
        self.W_self = nn.Linear(d_hidden, d_hidden)
        self.W_msg = nn.Linear(d_hidden, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)

    def forward(
        self,
        h: torch.Tensor,       # (B, d, d_hidden) node states
        adj: torch.Tensor,     # (B, d, d) or (d, d) weighted adjacency
        weights: torch.Tensor,  # (B, d, d) or (d, d) edge weights
    ) -> torch.Tensor:
        """One round of message passing.

        h_i^(l+1) = LN(GELU(W1 * h_i^(l) + sum_j W2 * h_j^(l) * w_ji) + h_i^(l))
        """
        # Self update
        self_update = self.W_self(h)  # (B, d, d_hidden)

        # Message from neighbors
        # weights: (B, d, d) where weights[b, j, i] = edge weight from j to i
        # We want: for each node i, sum over j where adj[j,i]=1: W_msg(h_j) * w_ji
        msg = self.W_msg(h)  # (B, d, d_hidden)

        # Batched matrix multiply: (B, d, d) @ (B, d, d_hidden) = (B, d, d_hidden)
        # weights.transpose(-2,-1) to get incoming edges per node
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(h.size(0), -1, -1)
        if weights.dim() == 2:
            weights = weights.unsqueeze(0).expand(h.size(0), -1, -1)

        # adj_T[b, i, j] = adj[b, j, i] = "edge from j to i"
        # We want for node i: sum_j (w_ji * msg_j)
        weighted_adj = (adj * weights).transpose(-2, -1)  # (B, d, d) — incoming
        neighbor_msg = torch.bmm(weighted_adj, msg)  # (B, d, d_hidden)

        # Combine with residual
        h_new = self.norm(F.gelu(self_update + neighbor_msg) + h)

        return h_new


class GraphPool(nn.Module):
    """Pool node representations to graph-level representation."""

    def __init__(self, d_hidden: int):
        super().__init__()
        # Attention-weighted pooling
        self.attn = nn.Linear(d_hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, d, d_hidden) node states
        returns: (B, d_hidden) graph-level representation
        """
        weights = F.softmax(self.attn(h), dim=1)  # (B, d, 1)
        return (h * weights).sum(dim=1)  # (B, d_hidden)


class RegimeHead(nn.Module):
    """Regime classification head."""

    def __init__(self, d_input: int, n_regimes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.GELU(),
            nn.Linear(d_input, n_regimes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, d_input) graph representation
        returns: (B, n_regimes) log-probabilities
        """
        return F.log_softmax(self.head(h), dim=-1)


class PolicyHead(nn.Module):
    """Position target head with regime-conditioned gating."""

    def __init__(self, d_input: int, n_regimes: int):
        super().__init__()
        self.position_net = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.GELU(),
            nn.Linear(d_input, 1),
            nn.Tanh(),
        )
        # Regime gate: per-regime scaling factor
        self.regime_gate = nn.Linear(n_regimes, 1, bias=False)
        nn.init.ones_(self.regime_gate.weight)  # start with no gating

    def forward(
        self,
        h: torch.Tensor,            # (B, d_input)
        regime_probs: torch.Tensor,  # (B, n_regimes) probabilities (NOT log)
        gate_threshold: float = 0.6,
    ) -> torch.Tensor:
        """
        returns: (B, 1) position target in [-1, 1]
        """
        raw_position = self.position_net(h)  # (B, 1)

        # Regime gating: scale position by regime-conditioned factor
        gate = torch.sigmoid(self.regime_gate(regime_probs))  # (B, 1) in [0, 1]

        # Confidence gate: soft scaling based on regime certainty
        # With K=4 regimes, uniform=0.25. Use softer ramp so positions aren't zeroed
        # during early training when regimes haven't differentiated yet.
        max_prob = regime_probs.max(dim=-1, keepdim=True).values  # (B, 1)
        # Ramp from 0.2 at uniform (0.25) to 1.0 at confident (0.6+)
        confidence = torch.clamp((max_prob - 0.20) / 0.40, 0.1, 1.0)

        return raw_position * gate * confidence


class PredictorHead(nn.Module):
    """Multi-horizon prediction head for self-supervised training."""

    def __init__(self, d_input: int, horizon: int = 5):
        super().__init__()
        self.return_pred = nn.Linear(d_input, horizon)  # predict future returns
        self.vol_pred = nn.Sequential(
            nn.Linear(d_input, horizon),
            nn.Softplus(),  # vol must be positive
        )
        self.jump_pred = nn.Linear(d_input, horizon)  # jump probability (logits)

    def forward(self, h: torch.Tensor) -> dict:
        """
        h: (B, d_input)
        returns: dict with 'returns', 'volatility', 'jump_logits'
        """
        return {
            "returns": self.return_pred(h),      # (B, H)
            "volatility": self.vol_pred(h),      # (B, H)
            "jump_logits": self.jump_pred(h),    # (B, H)
        }


class DTRN(nn.Module):
    """Dynamic Topology Regime Network.

    Full model: embedding -> message passing -> pooling -> GRU -> heads
    """

    def __init__(
        self,
        n_features: int,
        d_embed: int = 32,
        d_hidden: int = 64,
        n_message_passes: int = 2,
        d_temporal: int = 64,
        n_regimes: int = 4,
        pred_horizon: int = 5,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_hidden = d_hidden
        self.d_temporal = d_temporal

        # Input embedding
        self.embed = NodeEmbedding(n_features, d_embed)

        # Project embedding to hidden dim if different
        self.input_proj = (
            nn.Linear(d_embed, d_hidden) if d_embed != d_hidden else nn.Identity()
        )

        # Message passing layers
        self.msg_layers = nn.ModuleList(
            [GraphMessagePass(d_hidden) for _ in range(n_message_passes)]
        )

        # Graph pooling
        self.pool = GraphPool(d_hidden)

        # Temporal memory (GRU over pooled graph states)
        self.gru = nn.GRU(d_hidden, d_temporal, batch_first=True)

        # Heads
        self.regime_head = RegimeHead(d_temporal, n_regimes)
        self.policy_head = PolicyHead(d_temporal, n_regimes)
        self.predictor_head = PredictorHead(d_temporal, pred_horizon)

    def forward(
        self,
        x: torch.Tensor,           # (B, T, d) features
        mask: torch.Tensor,         # (B, T, d) masks
        adj: torch.Tensor,          # (B, T, d, d) or (T, d, d) adjacency per step
        weights: torch.Tensor,      # (B, T, d, d) or (T, d, d) edge weights per step
        h0: Optional[torch.Tensor] = None,  # (1, B, d_temporal) GRU hidden
    ) -> dict:
        """Full forward pass.

        Returns dict with:
        - regime_logprobs: (B, T, K) log regime probabilities
        - regime_probs: (B, T, K) regime probabilities
        - position: (B, T, 1) position targets
        - predictions: dict of (B, T, H) prediction tensors
        - h_final: (1, B, d_temporal) final GRU hidden state
        """
        B, T, d = x.shape

        all_regime_logprobs = []
        all_regime_probs = []
        all_positions = []
        all_returns_pred = []
        all_vol_pred = []
        all_jump_pred = []

        h_state = h0  # GRU hidden state

        for t in range(T):
            x_t = x[:, t, :]       # (B, d)
            m_t = mask[:, t, :]     # (B, d)

            # Get adjacency for this step
            if adj.dim() == 4:
                a_t = adj[:, t, :, :]   # (B, d, d)
                w_t = weights[:, t, :, :]
            else:
                a_t = adj[t, :, :]      # (d, d)
                w_t = weights[t, :, :]

            # 1. Node embedding
            h_nodes = self.embed(x_t, m_t)       # (B, d, d_embed)
            h_nodes = self.input_proj(h_nodes)    # (B, d, d_hidden)

            # 2. Message passing
            for msg_layer in self.msg_layers:
                h_nodes = msg_layer(h_nodes, a_t, w_t)

            # 3. Pool to graph-level
            h_graph = self.pool(h_nodes)  # (B, d_hidden)

            # 4. GRU update
            h_graph_seq = h_graph.unsqueeze(1)  # (B, 1, d_hidden)
            gru_out, h_state = self.gru(h_graph_seq, h_state)
            h_t = gru_out.squeeze(1)  # (B, d_temporal)

            # 5. Heads
            regime_logp = self.regime_head(h_t)     # (B, K)
            regime_p = regime_logp.exp()
            position = self.policy_head(h_t, regime_p)  # (B, 1)
            preds = self.predictor_head(h_t)        # dict

            all_regime_logprobs.append(regime_logp)
            all_regime_probs.append(regime_p)
            all_positions.append(position)
            all_returns_pred.append(preds["returns"])
            all_vol_pred.append(preds["volatility"])
            all_jump_pred.append(preds["jump_logits"])

        return {
            "regime_logprobs": torch.stack(all_regime_logprobs, dim=1),  # (B, T, K)
            "regime_probs": torch.stack(all_regime_probs, dim=1),
            "position": torch.stack(all_positions, dim=1),              # (B, T, 1)
            "predictions": {
                "returns": torch.stack(all_returns_pred, dim=1),        # (B, T, H)
                "volatility": torch.stack(all_vol_pred, dim=1),
                "jump_logits": torch.stack(all_jump_pred, dim=1),
            },
            "h_final": h_state,
        }

    def forward_step(
        self,
        x_t: torch.Tensor,     # (B, d)
        m_t: torch.Tensor,     # (B, d)
        a_t: torch.Tensor,     # (B, d, d) or (d, d)
        w_t: torch.Tensor,     # (B, d, d) or (d, d)
        h_state: Optional[torch.Tensor] = None,
    ) -> Tuple[dict, torch.Tensor]:
        """Single-step forward (for online inference).

        Returns (outputs_dict, h_state).
        """
        # Node embedding
        h_nodes = self.embed(x_t, m_t)
        h_nodes = self.input_proj(h_nodes)

        # Message passing
        for msg_layer in self.msg_layers:
            h_nodes = msg_layer(h_nodes, a_t, w_t)

        # Pool
        h_graph = self.pool(h_nodes).unsqueeze(1)  # (B, 1, d_hidden)

        # GRU
        gru_out, h_state = self.gru(h_graph, h_state)
        h_t = gru_out.squeeze(1)

        # Heads
        regime_logp = self.regime_head(h_t)
        regime_p = regime_logp.exp()
        position = self.policy_head(h_t, regime_p)
        preds = self.predictor_head(h_t)

        outputs = {
            "regime_logprobs": regime_logp,
            "regime_probs": regime_p,
            "position": position,
            "predictions": preds,
        }

        return outputs, h_state
