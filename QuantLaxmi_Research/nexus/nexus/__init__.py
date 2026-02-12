"""NEXUS — Neural Exchange Unified Simulator.

A JEPA world model with Mamba-2 backbone, hyperbolic latent geometry,
topological regime sensing, and model-predictive control for financial markets.

Architecture
------------
NEXUS
├── MultiScaleTokenizer     — tick/1min/daily → learned VQ tokens
├── Mamba2Backbone           — Selective State Space (O(n), not O(n²))
├── JEPAWorldModel           — predict future LATENT states (not prices)
│   ├── ContextEncoder       — Mamba2 on visible market data
│   ├── TargetEncoder        — EMA of context encoder (no gradient)
│   └── Predictor            — predicts target latent from context latent
├── LorentzManifold          — hyperbolic latent space (H^d, negative curvature)
├── TopologicalSensor        — persistent homology for regime detection
└── LatentPlanner            — TD-MPC2-style MPC in latent space

Key Innovation
--------------
Markets are NOT sequences to predict. Markets are WORLDS to understand.
NEXUS learns a world model of market dynamics, embeds states in hyperbolic
space (natural for hierarchical structure), detects regime changes via
topology, and plans optimal actions by simulating futures in latent space.

References
----------
- I-JEPA: Assran et al., CVPR 2023 (arXiv:2301.08243)
- Mamba-2: Dao & Gu, 2024 (arXiv:2405.21060)
- TD-MPC2: Hansen et al., ICLR 2024 (arXiv:2310.16828)
- DreamerV3: Hafner et al., Nature 2025 (arXiv:2301.04104)
- Lorentz model: Nickel & Kiela, NeurIPS 2018
- Persistent homology for finance: Indian markets study, ScienceDirect 2024
"""

__version__ = "0.1.0"
__codename__ = "NEXUS"
