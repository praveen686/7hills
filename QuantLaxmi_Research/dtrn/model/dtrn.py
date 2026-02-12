"""DTRN model wrapper — combines topology learner with graph neural network."""
from __future__ import annotations

from ..config import DTRNConfig
from .topology import DynamicTopology
from .graph_net import DTRN


def create_dtrn(config: DTRNConfig = None, n_features: int = 26) -> tuple:
    """Create DTRN components.

    Returns (topology, model) tuple.
    """
    if config is None:
        config = DTRNConfig()

    topology = DynamicTopology(
        d=n_features,
        ewma_span=config.ewma_cov_span,
        top_k=config.top_k_edges,
        tau_on=config.tau_on,
        tau_off=config.tau_off,
        max_flip_rate=config.max_edge_flip_rate,
        precision_reg=config.precision_reg,
    )

    model = DTRN(
        n_features=n_features,
        d_embed=config.d_embed,
        d_hidden=config.d_hidden,
        n_message_passes=config.n_message_passes,
        d_temporal=config.d_temporal,
        n_regimes=config.n_regimes,
        pred_horizon=config.pred_horizon,
    )

    return topology, model
