
# NEXUS: Neural Exchange Unified Simulator

### A World-Model Approach to Financial Markets via Joint Embedding, Hyperbolic Geometry, and Topological Sensing

---

**Authors:** Seven Hills Research Team
**Date:** February 2026
**Version:** 0.2.0
**Contact:** research@sevenhills.capital

---

> *"The market is not a sequence to predict. It is a world to understand."*

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction & Motivation](#2-introduction--motivation)
   - 2.1 [Formal Problem Statement](#21-formal-problem-statement)
   - 2.2 [The Fundamental Problem: Prediction vs. Understanding](#22-the-fundamental-problem-prediction-vs-understanding)
   - 2.3 [Why Observation-Space Prediction Is Doomed](#23-why-observation-space-prediction-is-doomed)
   - 2.4 [The World Model Paradigm](#24-the-world-model-paradigm)
   - 2.5 [Why Financial Markets Need Different Geometry](#25-why-financial-markets-need-different-geometry)
   - 2.6 [Why Topology Detects What Statistics Cannot](#26-why-topology-detects-what-statistics-cannot)
3. [Architecture Overview](#3-architecture-overview)
4. [Component Deep Dives](#4-component-deep-dives)
   - 4.1 [Mamba-2 Backbone](#41-mamba-2-backbone-selective-state-space-model)
   - 4.2 [JEPA World Model](#42-jepa-world-model)
   - 4.3 [Hyperbolic Latent Space](#43-hyperbolic-latent-space)
   - 4.4 [Topological Regime Sensor](#44-topological-regime-sensor)
   - 4.5 [TD-MPC2 Planner](#45-td-mpc2-planner)
   - 4.6 [Multi-Scale VQ Tokenizer](#46-multi-scale-vq-tokenizer)
5. [Training Pipeline](#5-training-pipeline)
6. [Why This Combination is Novel](#6-why-this-combination-is-novel)
7. [Implementation Details](#7-implementation-details)
8. [Code Statistics](#8-code-statistics)
9. [Limitations and Future Work](#9-limitations-and-future-work)
10. [References](#10-references)

---

## 1. Abstract

Contemporary quantitative trading models overwhelmingly operate in observation space -- predicting future prices, returns, or signals directly from past observations. This paradigm suffers from three fundamental limitations: (i) O(n^2) attention complexity in transformer-based architectures prohibits ingestion of tick-level sequences; (ii) prediction in observation space conflates learnable market dynamics with irreducible microstructural noise; and (iii) Euclidean latent representations distort the inherently hierarchical structure of financial markets (macro regimes, sector flows, individual equities, derivatives, microstructure).

We introduce **NEXUS** (Neural Exchange Unified Simulator), a system that unifies five state-of-the-art techniques -- each previously confined to separate domains -- into a coherent framework for financial market modeling. To our knowledge, no published work has combined all five components into a single architecture for financial markets. NEXUS combines: **(1)** a Mamba-2 selective state space backbone for O(n) temporal encoding; **(2)** a Joint Embedding Predictive Architecture (JEPA) that learns market dynamics in latent space rather than predicting prices; **(3)** a Lorentz hyperboloid latent geometry that naturally captures market hierarchy; **(4)** topological sensing for regime detection via persistent homology (H0) and spectral cycle approximation (H1); and **(5)** TD-MPC2-style model-predictive control that plans optimal positions by simulating thousands of futures in learned latent space.

NEXUS treats financial markets as *worlds to simulate*, not sequences to extrapolate. We formulate the problem as a Partially Observable Markov Decision Process (POMDP) under the exogenous regime assumption, where the participant's actions do not causally affect market state transitions. The system comprises approximately 4,800 lines of pure PyTorch, ships in three model sizes (1.2M to 23M parameters), and includes a complete 3-phase training pipeline with 40 unit tests verifying component correctness, causality, and gradient flow.

---

## 2. Introduction & Motivation

### 2.1 Formal Problem Statement

We formulate financial trading as a Partially Observable Markov Decision Process (POMDP) (Kaelbling, Littman & Cassandra, 1998):

- **State** s_t in S: the true market state (unobserved). This includes all participants' beliefs, order books, pending institutional flows, and macroeconomic conditions -- the vast majority of which is never directly observable.
- **Observation** o_t in O: OHLCV bars, derived features, option chains, order flow imbalances, and news sentiment -- a noisy, partial projection of the true state.
- **Action** a_t in A, where A is a subset of [-max_pos, max_pos]^{n_assets}: the position vector across tradeable assets.
- **Transition**: s_{t+1} ~ P(s_{t+1} | s_t). **Exogenous assumption**: actions do NOT affect transitions.
- **Observation emission**: o_t ~ Q(o_t | s_t).
- **Reward**:

$$r_{t+1} = \underbrace{a_t^\top \Delta p_{t+1}}_{\text{P\&L (index points)}} - \underbrace{c(|\Delta a_t|)}_{\text{transaction costs}} - \underbrace{\lambda \cdot \text{risk}(a_t, \Sigma_t)}_{\text{risk penalty}}$$

**Key assumption -- the exogenous regime**: We operate under the assumption that the participant's actions do not causally affect market state transitions. This is appropriate for retail and prop-sized participants in liquid markets (NIFTY 50 futures trade ~$3B notional daily; a $10M position is <0.3% of daily volume). The exogenous assumption simplifies the POMDP significantly: the dynamics model need not condition on actions, and we avoid the intractable problem of modeling market impact on state transitions. For participants whose order flow materially impacts prices, this assumption would need to be relaxed.

The NEXUS encoder maps observation sequences to latent states:

$$z_t = \text{enc}(o_{t-L:t})$$

The dynamics model learns transitions in latent space:

$$z_{t+1} = f(z_t) \quad \text{(no action dependence under exogenous assumption)}$$

The reward model combines learned and explicit components:

$$\hat{r}_{t+1} = r_{\text{learned}}(z_t, a_t) + a_t^\top \Delta p_{t+1} - c(|\Delta a_t|)$$

The planner searches over action sequences to maximize cumulative discounted reward under the learned dynamics:

$$a_{0:H}^* = \arg\max_{a_{0:H}} \sum_{h=0}^{H} \gamma^h \hat{r}_{t+h+1} + \gamma^{H+1} V(z_{t+H+1})$$

This formulation makes explicit what most ML trading systems leave implicit: we are solving a sequential decision problem under partial observability, not a supervised prediction problem.

### 2.2 The Fundamental Problem: Prediction vs. Understanding

The dominant paradigm in quantitative finance reduces market modeling to a supervised learning problem: given historical features x_{1:T}, predict some target y_{T+1} (a return, a direction, a volatility). This framing -- regardless of the model class (linear, tree-based, deep neural network) -- makes an implicit assumption that the optimal action is a deterministic function of predicted y.

This assumption is wrong. Markets are dynamic systems with regime-dependent optimal policies. A position that is optimal in a mean-reverting regime is catastrophic in a trending one. The correct framing is not "what will the price be?" but rather "what state is the market in, and what action is optimal in that state?"

### 2.3 Why Observation-Space Prediction Is Doomed

Prediction in observation space (raw prices, returns) faces irreducible limitations:

- **Signal-to-noise ratio**: At daily frequency, the expected return of a typical equity is approximately 0.04% per day (10% annualized), while the standard deviation is approximately 1.5% per day. The SNR is roughly 0.03 -- the signal is 30x smaller than the noise.
- **Non-stationarity**: The data-generating process changes. Volatility regimes, correlation structures, and market microstructure evolve continuously. A model trained on 2023 data may face a fundamentally different market in 2025.
- **Sensitivity to noise**: Small perturbations in observation space (a single large trade, a flash crash, a data error) produce large prediction errors, while the *underlying market state* may be essentially unchanged.

### 2.4 The World Model Paradigm

NEXUS adopts a fundamentally different approach inspired by world models in robotics and reinforcement learning (Ha & Schmidhuber, 2018; Hafner et al., 2025). Rather than predicting observations, NEXUS learns a compressed, denoised *latent model* of market dynamics:

1. **Encode**: Map raw market data to a learned latent representation that captures market *state*, not market *observations*.
2. **Predict in latent space**: Learn dynamics z_{t+1} = f(z_t) in latent space, where the model is free to ignore noise and focus on structure.
3. **Plan**: Given the learned dynamics, simulate thousands of possible futures and select the action sequence that maximizes risk-adjusted return.

This is precisely how AlphaGo and MuZero defeated human champions at Go and Atari: not by predicting the opponent's next move, but by *simulating* millions of possible game trajectories and choosing the path with highest expected value.

### 2.5 Why Financial Markets Need Different Geometry

Financial markets have natural hierarchical structure:

```
Global macro regime
  |-- Regional dynamics (US, EU, Asia)
       |-- Sector rotations (Tech, Finance, Energy)
            |-- Individual equities (RELIANCE, TCS, HDFC)
                 |-- Derivatives (options, futures)
                      |-- Microstructure (order flow, ticks)
```

Euclidean space provides a poor inductive bias for hierarchical data. Embedding an n-node tree into Euclidean R^d with bounded distortion requires d = Omega(log n) dimensions (Linial, London & Rabinovich, 1995; based on Bourgain's 1985 embedding theorem which gives O(log n) distortion for finite metrics into Hilbert space). For a binary tree with n = 1024 leaves, this means d = O(log_2 1024) = O(10) Euclidean dimensions -- not prohibitive in itself, but the distortion grows as O(log n) with the tree size.

Hyperbolic space of constant negative curvature provides a qualitatively better match. Sarkar (2011) showed that *any* weighted tree can be embedded into the hyperbolic plane H^2 with arbitrarily small distortion (1 + epsilon for any epsilon > 0), using only 2 dimensions regardless of tree size. The key insight is that the area of a disk in hyperbolic space grows exponentially with its radius, naturally accommodating the exponential branching of trees without distortion.

For NEXUS, this means the market hierarchy (macro -> sector -> equity -> derivative -> microstructure) can be faithfully represented in a low-dimensional hyperbolic latent space with O(1) distortion, whereas Euclidean space would incur O(log n) distortion that grows with the depth and breadth of the hierarchy. The advantage is genuine -- constant versus logarithmic distortion -- though less dramatic than an exponential gap in dimensionality.

### 2.6 Why Topology Detects What Statistics Cannot

Traditional regime detection methods (HMMs, change-point detection, volatility clustering) operate on statistical summaries of the data. They are blind to *structural* changes -- qualitative shifts in the shape of the data manifold that precede statistical shifts.

Persistent homology, the central tool of topological data analysis (TDA), detects these structural changes. When a market transitions from a bull regime to a crisis, the topology of the return distribution changes: connected components fragment, cycles appear and vanish. These topological phase transitions are often visible in persistent homology *before* they manifest in volatility or correlation statistics.

---

## 3. Architecture Overview

```
                 NEXUS ARCHITECTURE -- END-TO-END DATA FLOW
  ======================================================================

  MARKET DATA (tick / 1-min / daily OHLCV)
       |
       |    +-------------------------------------------------+
       +--->|          MULTI-SCALE VQ TOKENIZER                |
            |                                                 |
            |  Micro (1-min) --+                              |
            |  Meso (hourly) --+--> Delta Enc -> Proj -> VQ   |
            |  Macro (daily) --+         -> Cross-Scale Fuse  |
            +-----------------------+-------------------------+
                                    |
                                    v
            +-------------------------------------------------+
            |           MAMBA-2 BACKBONE (O(n))               |
            |                                                 |
            |  x -> Linear -> Conv1d -> SiLU -> SSM -> Gate   |
            |        |                                  ^     |
            |        +------ Linear -> SiLU ----------->+     |
            |                                                 |
            |  6 layers x (d_model=256, d_state=64, 8 heads)  |
            +-----------------------+-------------------------+
                                    |
                           (B, L, d_model)
                                    |
            +-------------------------------------------------+
            |         JEPA WORLD MODEL                         |
            |                                                 |
            |  Context Enc -----> z_ctx -----> Predictor ---+  |
            |  (Mamba-2)          |            (MLP x 4)   |  |
            |                     |                        v  |
            |  Target Enc ------> z_tgt        z_hat_tgt   |  |
            |  (EMA, no grad)     |               |        |  |
            |                     +--> L_jepa = ||z_hat - z||  |
            +-----------------------+-------------------------+
                                    |
                          (B, L, d_latent=128)
                                    |
                   +----------------+----------------+
                   |                |                |
                   v                v                v
       +-----------+--+    +-------+------+   +-----+--------+
       | HYPERBOLIC   |    |  TOPOLOGICAL |   |  TD-MPC2     |
       | LATENT SPACE |    |    SENSOR    |   |  PLANNER     |
       |              |    |              |   |              |
       | Lorentz H^d  |    | Takens embed |   | Dynamics     |
       | Exp/Log maps |    | H0: exact    |   | Reward model |
       | Centroid     |    | H1: spectral |   | Value model  |
       | LorentzLinear|    | beta_0, H_0  |   | CEM (512x6)  |
       +-----------+--+    +------+-------+   +------+-------+
                   |              |                   |
                   +--------------+-------------------+
                                  |
                                  v
                        +---------+---------+
                        |  POSITION VECTOR  |
                        |  (B, n_assets)    |
                        |  [-0.25, +0.25]   |
                        +-------------------+

  Assets: NIFTY | BANKNIFTY | FINNIFTY | MIDCPNIFTY | BTC | ETH
```

**Data Flow Summary:**

1. Raw multi-scale market data (tick, 1-min, daily) enters the **Multi-Scale VQ Tokenizer**, which applies delta encoding for stationarity, learns per-scale projections, optionally discretizes through a 512-entry VQ codebook, and fuses scales via a learned cross-scale MLP.

2. The fused token sequence passes through 6 layers of the **Mamba-2 backbone**, which applies selective state space processing at O(n) complexity -- learning data-dependent state transitions that selectively attend to regime shifts while ignoring noise.

3. The **JEPA World Model** splits the encoded sequence into context (visible past) and target (masked future). The context encoder processes visible data; the EMA target encoder (no gradient) encodes the masked future. A predictor network learns to bridge context to target *in latent space*, learning market dynamics without ever predicting raw prices.

4. Three parallel modules consume the latent representations:
   - The **Hyperbolic Latent Space** projects embeddings onto the Lorentz hyperboloid H^d for natural hierarchical geometry.
   - The **Topological Sensor** applies Takens embedding and computes topological features: exact H0 persistence via single-linkage clustering (Kruskal's MST), and spectral H1 approximation via Laplacian eigenvalue gaps.
   - The **TD-MPC2 Planner** simulates 512 possible 5-day futures via Cross-Entropy Method (CEM), selects elite trajectories, and outputs the optimal position vector.

5. Output: a per-asset position vector in [-0.25, +0.25] for 6 tradeable assets.

---

## 4. Component Deep Dives

### 4.1 Mamba-2 Backbone: Selective State Space Model

#### Why Not Transformers?

The transformer architecture (Vaswani et al., 2017) has dominated sequence modeling for the past eight years. Its self-attention mechanism computes all pairwise interactions between tokens, yielding O(n^2) time and memory complexity in sequence length n. For financial applications processing 100,000+ tick-level tokens per trading day, this means O(10^{10}) operations per forward pass -- computationally prohibitive.

Mamba-2 (Dao & Gu, 2024) achieves transformer-quality modeling at O(n) complexity through **Structured State Space Duality (SSD)**. The key insight is that self-attention can be reformulated as a special case of a state space model (SSM) with data-dependent parameters.

#### Mathematical Formulation

The classical linear time-invariant SSM is:

$$h_t = \bar{A} \cdot h_{t-1} + \bar{B} \cdot x_t$$
$$y_t = C \cdot h_t + D \cdot x_t$$

where A, B, C, D are fixed matrices. Mamba's critical innovation is making B and C **data-dependent** (selective):

$$B_t = \text{Linear}(x_t), \quad C_t = \text{Linear}(x_t)$$

and introducing a learned, input-dependent discretization step:

$$\Delta_t = \text{softplus}(\text{Linear}(x_t))$$
$$\bar{A}_t = \exp(\Delta_t \cdot A), \quad \bar{B}_t = \Delta_t \cdot B_t$$

This selective mechanism allows the model to learn *what to remember and what to forget* at each timestep -- analogous to gating in LSTMs, but operating through the lens of continuous-time state space theory.

#### Architecture Per Layer

```
x --> Linear(d, 2*d_inner) --> [x_main, z]
                                  |      |
                           Conv1d(d_conv) |
                                  |      |
                                SiLU     |
                                  |      |
                          Selective SSM   |
                                  |      |
                                  x   SiLU(z)
                                  |      |
                                  +--gate-+
                                     |
                              Linear(d_inner, d)
                                     |
                                  + residual
```

Each Mamba-2 block combines:
- **Depthwise convolution** (kernel size 4): local context mixing, analogous to positional encoding
- **Selective scan**: the core SSM with data-dependent B, C, and Delta
- **Gating**: element-wise multiplication with a parallel SiLU-activated branch
- **Residual connection**: gradient highway for deep stacks

#### Why This Matters for Financial Markets

1. **O(n) complexity**: Process full trading day of tick data (100K+ tokens) without quadratic memory blow-up
2. **Selective attention to regime shifts**: The data-dependent discretization step Delta learns to "open the gate" when the input signals a regime change, and "close the gate" during stationary periods. This is a learned, continuous analogue of structural break detection.
3. **Kalman filter connection**: The SSM formulation has a natural connection to Kalman filtering -- the workhorse of quantitative finance. The state h_t can be interpreted as a belief state over latent market factors, updated at each timestep with new observations.
4. **Stability**: The A matrix is initialized as negative (exp(Delta * A) < 1), ensuring exponential decay of old information -- a desirable inductive bias for financial data where stale signals should be forgotten.

**NEXUS configuration**: d_model=256, d_state=64, d_conv=4, expand=2, n_layers=6, n_heads=8. Total backbone parameters: ~2M (base model).

---

### 4.2 JEPA World Model

#### The Core Idea: Predict Representations, Not Prices

The Joint Embedding Predictive Architecture (JEPA), introduced by LeCun (2022) and instantiated by Assran et al. (2023) for vision, represents a fundamental departure from generative (VAE, diffusion) and contrastive (SimCLR, CLIP) approaches to self-supervised learning.

The key insight: **do not predict raw observations**. Instead, predict *latent representations* of future states. This is superior for three reasons:

1. **Noise invariance**: The latent encoder abstracts away irreducible noise. Individual tick fluctuations, order book jitter, and data feed latency are compressed out of the representation. The predictor learns market *dynamics*, not market *noise*.

2. **No mode collapse without negatives**: Unlike contrastive learning, JEPA does not need carefully constructed negative pairs. The EMA target encoder provides a slowly-moving, stable training signal that prevents representation collapse.

3. **World model semantics**: The predictor network that maps context representations to target representations *is* a world model -- it learns the transition dynamics z_{t+1} = f(z_t) in a self-supervised manner, without any reward signal.

#### Architecture

```
                    JEPA World Model
    ================================================

    Context (visible past)          Target (masked future)
         |                                |
         v                                v
    +----+--------+               +-------+--------+
    | Context Enc |               | Target Encoder |
    | (Mamba-2)   |               | (EMA copy)     |
    | (gradient)  |               | (NO gradient)  |
    +----+--------+               +-------+--------+
         |                                |
    z_ctx (B, L_ctx, d_latent)     z_tgt (B, L_tgt, d_latent)
         |                                |
         v                                |
    +----+--------+                       |
    |  Predictor  |                       |
    |  (MLP x 4)  |                       |
    |  + pos emb  |                       |
    +----+--------+                       |
         |                                |
    z_hat (B, L_tgt, d_latent)            |
         |                                |
         +---- L_jepa = SmoothL1(z_hat, z_tgt.detach()) ----+
```

**Context Encoder**: The full Mamba-2 backbone followed by a LayerNorm + Linear projection to d_latent=128. Processes visible past market data with full gradient flow.

**Target Encoder**: A deep copy of the context encoder with **all gradients disabled**. Updated via exponential moving average (EMA):

$$\theta_{\text{target}} \leftarrow \tau \cdot \theta_{\text{target}} + (1 - \tau) \cdot \theta_{\text{context}}$$

The EMA decay tau is warmed up linearly from 0.996 to 0.9999 over training, following V-JEPA 2 (Meta AI, 2025). Early in training, the target moves quickly to track the rapidly-improving context encoder. Late in training, the target moves slowly, providing a stable, high-quality training signal.

**Predictor**: A lightweight residual MLP (4 blocks, hidden dim = 2 * d_model = 512) with learned positional embeddings. Takes the mean-pooled context encoding and target position information, and predicts what the target encoder would produce for the masked future. Each predictor block is:

$$h \leftarrow h + \text{MLP}(\text{LayerNorm}(h))$$

where MLP = Linear(d, 4d) -> GELU -> Dropout -> Linear(4d, d) -> Dropout.

**Loss**: Smooth L1 loss (Huber loss) in latent space:

$$\mathcal{L}_{\text{JEPA}} = \text{SmoothL1}(\hat{z}_{\text{target}}, z_{\text{target}}^{\text{detach}})$$

Smooth L1 is preferred over MSE because it is robust to outlier latent states (market shocks produce outlier representations; we want the predictor to learn general dynamics, not overfit to rare extremes).

#### Collapse Risk and Mitigations

JEPA's EMA target encoder is an *empirical* defense against representation collapse, not a theoretical guarantee. In financial time series where SNR is approximately 0.03, collapse to predicting the unconditional mean is a real and persistent risk. The low signal-to-noise ratio means that a trivially constant representation (z_t = z_bar for all t) can achieve low loss simply by averaging over the noise -- the predictor learns to output the mean and is rarely penalized enough to break out of this basin.

We implement three monitoring diagnostics that are evaluated every 100 training steps:

1. **Per-dimension variance**: var(z_i) should remain > 0.01 for all latent dimensions i in {1, ..., d_latent}. A single dimension collapsing to a constant indicates early-stage collapse.

2. **Effective rank**: The effective rank of the latent representation matrix Z in R^{B x d_latent}, computed as ||Z||_* / ||Z||_op (nuclear norm divided by operator norm), should stay above d_latent / 4. This measures how many dimensions the representation is actually using. Full collapse yields effective rank 1; a healthy representation uses most available dimensions.

3. **Mean pairwise cosine similarity**: avg_{i != j} cos(z_i, z_j) over a batch should stay below 0.5. High cosine similarity indicates that all representations are converging to the same direction, a hallmark of collapse.

If any diagnostic trips its threshold, we apply **variance regularization** as an additional loss term:

$$\mathcal{L}_{\text{var}} = \sum_{i=1}^{d_{\text{latent}}} \max(0, \gamma - \text{Var}(z_i))$$

where gamma = 1.0 is the variance floor. This loss penalizes any dimension whose variance falls below gamma, pushing the encoder to maintain informative representations. The regularization weight is set to 0.04, following VICReg (Bardes et al., 2022). We note that this is an engineering defense, not a fundamental solution -- the theoretical conditions under which JEPA provably avoids collapse in low-SNR regimes remain an open problem.

#### Imagining Futures

The trained predictor enables autoregressive imagination -- generating hypothetical future market states by iteratively applying the predictor:

$$z_{t+1} = \text{Predictor}(z_t, \text{pos}=t)$$
$$r_{t+1} = \text{RewardHead}(z_{t+1})$$

This is the foundation for model-based planning: the planner uses these imagined trajectories to evaluate candidate action sequences without ever requiring real market data for the future.

#### Comparison to Alternatives

| Method | Predicts | Needs Negatives | Mode Collapse Risk | Learns Dynamics |
|--------|----------|-----------------|-------------------|-----------------|
| Autoregressive (GPT) | Next token (observation space) | No | Low | Implicitly |
| MAE | Masked pixels (observation space) | No | Low | No |
| Contrastive (SimCLR) | Similarity scores | **Yes** | Medium | No |
| VAE | Reconstructed observations | No | Medium | Decoder only |
| **JEPA (NEXUS)** | **Latent representations** | **No** | **Medium (EMA + monitoring)** | **Yes (predictor)** |

---

### 4.3 Hyperbolic Latent Space

#### Why Euclidean Space Is a Poor Fit

Consider embedding a balanced binary tree with n = 1024 leaves into metric space with bounded distortion. The relevant results from metric embedding theory are:

**Euclidean lower bound** (Linial, London & Rabinovich, 1995; building on Bourgain, 1985): Embedding any n-point metric into R^d with distortion D requires d = Omega(n^{1/D^2}). For a tree with n = 1024 leaves embedded with O(1) distortion, this gives d = Omega(log n) = Omega(10). More precisely, Bourgain's 1985 theorem guarantees that any finite metric space on n points can be embedded into Hilbert space with O(log n) distortion -- so for n = 1024, the best achievable distortion in Euclidean space is O(log 1024) = O(10), which requires d = O(log^2 n) dimensions for the algorithmic construction. The distortion is logarithmic, not catastrophic -- but it grows with tree size.

**Hyperbolic guarantee** (Sarkar, 2011): Any weighted tree can be embedded into the hyperbolic plane H^2 -- just 2 dimensions -- with distortion 1 + epsilon for any epsilon > 0. This is because the hyperbolic plane has area that grows exponentially with radius, perfectly matching the exponential branching of trees.

The practical implication: for NEXUS with d_hyperbolic = 64, the hyperbolic space can represent vastly more complex hierarchies at O(1) distortion than 64 Euclidean dimensions, where distortion grows as O(log n) with the number of embedded entities. This is a genuine advantage for representing the multi-level market hierarchy, though it should be noted that the real-world market hierarchy is only approximately tree-like -- cross-sector correlations and regime-dependent structure introduce non-tree edges.

#### The Lorentz Model H^d_K

NEXUS uses the **Lorentz (hyperboloid) model** of hyperbolic space, defined as:

$$\mathbb{H}^d_K = \{x \in \mathbb{R}^{d+1} : \langle x, x \rangle_L = 1/K, \; x_0 > 0\}$$

where the **Minkowski inner product** is:

$$\langle x, y \rangle_L = -x_0 y_0 + \sum_{i=1}^{d} x_i y_i$$

and K < 0 is the (negative) sectional curvature.

We chose the Lorentz model over the Poincare ball model for three reasons:

1. **Numerical stability**: The Poincare ball confines points to the unit ball ||x|| < 1, creating severe numerical issues near the boundary where gradients explode. The Lorentz model has no boundary.
2. **GPU efficiency**: Distance computation uses inner products (highly optimized on GPUs), not the complex arctanh-based formula of the Poincare model.
3. **Closed-form geodesics**: All geometric operations (distance, exponential map, logarithmic map, centroid) have clean closed-form expressions.

#### Key Operations

**Geodesic distance:**

$$d_K(x, y) = \frac{1}{\sqrt{|K|}} \cdot \text{arccosh}(-K \cdot \langle x, y \rangle_L)$$

**Exponential map at origin** (tangent vector -> hyperboloid point):

$$\exp_o(v) = \cosh(\sqrt{|K|} \cdot \|v\|) \cdot o + \frac{\sinh(\sqrt{|K|} \cdot \|v\|)}{\sqrt{|K|} \cdot \|v\|} \cdot \bar{v}$$

where o = (1/sqrt{|K|}, 0, ..., 0) is the origin on H^d_K.

**Logarithmic map at origin** (hyperboloid point -> tangent vector):

$$\log_o(x) = \frac{d(o, x)}{\|x_{\text{spatial}}\|} \cdot x_{\text{spatial}}$$

**Lorentz centroid** (Einstein midpoint / Frechet mean):

$$c = \frac{\sum_i w_i x_i}{\sqrt{|\langle \sum_i w_i x_i, \sum_i w_i x_i \rangle_L \cdot |K||}}$$

This is the closed-form weighted mean on the hyperboloid, computed by taking the weighted sum in ambient Minkowski space and projecting back onto the hyperboloid.

#### LorentzLinear Layer

Neural network layers on the hyperboloid operate via the LResNet approach (2024): transform only the spatial coordinates x_{1:d} through a standard linear layer, then recompute the time coordinate x_0 from the hyperboloid constraint:

$$x_0 = \sqrt{1/|K| + \|x_{\text{spatial}}\|^2}$$

This avoids the expensive log-map -> linear -> exp-map cycle while remaining on the manifold.

#### Market Hierarchy Embedding

In the hyperbolic latent space:
- **Near the origin**: Broad macro states (risk-on / risk-off, high/low volatility)
- **Increasing radius**: Increasingly specific market conditions (sector rotation, single-stock dynamics)
- **Distant from origin**: Microstructural states (order book imbalance, tick patterns)

The negative curvature ensures that the "volume" of representable states grows exponentially with radius -- precisely matching the exponential branching of market hierarchy.

**NEXUS configuration**: d_hyperbolic=64, curvature K=-1.0 (unit hyperboloid).

---

### 4.4 Topological Regime Sensor

#### The Insight: Crises Are Topological Phase Transitions

Traditional regime detection methods (hidden Markov models, GARCH, change-point statistics) operate on *distributional* properties of the data -- means, variances, correlations. They detect regime changes *after* the statistical signature has shifted.

Persistent homology detects something deeper: **structural changes in the shape of the data manifold**. Before a crisis manifests as increased volatility or correlation breakdown, the *topology* of the market changes:

- **Bull market**: Return vectors cluster in a concentrated region of state space -> high beta_0 (many distinct micro-clusters), low beta_1 (no cycles)
- **Pre-crisis**: Return distribution begins to fragment and develop cyclic instability -> beta_1 increases (loops appear in the point cloud)
- **Crisis**: Returns collapse to a tight, chaotic cluster -> beta_0 drops dramatically, persistence entropy spikes (maximum structural complexity)

These topological transitions often precede statistical transitions by days or weeks, providing early warning signals invisible to conventional indicators.

#### Pipeline: Time Series -> Topology

**Step 1: Takens Embedding**

By Takens' theorem (1981), a scalar time series generated by a deterministic dynamical system can be embedded into a point cloud in R^d that preserves the topology of the underlying attractor, provided d >= 2 * dim(attractor) + 1 and the delay tau is chosen appropriately.

Given a scalar time series x(t), the delay embedding produces:

$$\mathbf{p}_t = [x(t), x(t-\tau), x(t-2\tau), \ldots, x(t-(d-1)\tau)]$$

NEXUS applies Takens embedding to the first principal component of the latent representation z (from the JEPA encoder), with d=3 and tau=1.

**Takens embedding limitations**: We note that d=3 and tau=1 are heuristic choices. Takens' theorem strictly applies to deterministic dynamical systems with smooth attractors. Financial markets are noisy, nonstationary, and likely have time-varying attractor dimension -- violating the theorem's assumptions. Our use of Takens embedding is therefore best understood as a heuristic delay-coordinate construction that produces point clouds amenable to topological analysis, rather than a rigorous reconstruction of a deterministic attractor. In practice, we find that the topological features extracted from this construction are empirically informative for regime detection, even though the theoretical guarantees do not formally hold. A principled approach would use false nearest neighbors to estimate the embedding dimension d and mutual information to select tau, which we defer to future work.

**Step 2: Persistent Homology (H0) and Spectral Cycle Approximation (H1)**

Given the point cloud P = {p_1, ..., p_N}, we construct topological features at two homological dimensions:

**H0 (connected components) -- exact computation**: H0 persistence is computed exactly via single-linkage clustering, which is equivalent to Kruskal's minimum spanning tree algorithm. As the filtration radius epsilon increases from 0, isolated points merge into connected components. Each merge event is recorded as a (birth, death) pair. This is computationally efficient (O(N^2 log N)) and exact -- no approximation is involved.

**H1 (loops/cycles) -- spectral approximation**: Computing exact H1 persistent homology requires constructing the full Vietoris-Rips complex and computing boundary operators, which is O(2^N) in the worst case. Instead, we use a spectral proxy: at sampled filtration radii, we construct the graph Laplacian L = D - A of the Rips graph and count the number of eigenvalues below a threshold. The eigenvalue gaps of the Laplacian provide information about cycle structure (by the Cheeger inequality and the relationship between the first non-trivial eigenvalue and graph connectivity), but this is an approximation to the true H1 Betti numbers, not an exact computation.

**Important caveat**: For production deployment, we recommend using dedicated persistent homology libraries (Ripser, GUDHI, or giotto-tda) for exact H1 persistence computation. These implementations achieve near-optimal complexity via cohomology and implicit matrix reduction. Our spectral approximation captures structural information about cycle formation but does not enjoy the stability guarantees of true persistent homology -- in particular, the bottleneck distance stability theorem (Cohen-Steiner, Edelsbrunner & Harer, 2007) does not apply to our spectral proxy, meaning small perturbations in the input can potentially cause large changes in the spectral H1 features.

**Step 3: Topological Feature Extraction**

From the persistence diagrams (exact for H0, approximate for H1), NEXUS extracts 8 features per sliding window:

| Feature | Description | Financial Interpretation |
|---------|-------------|------------------------|
| beta_0 | H0 Betti number at median radius | Market fragmentation |
| beta_1 | H1 Betti number at median radius (spectral approx.) | Cyclical instability |
| H0 entropy | H = -sum p_i log(p_i) for H0 | Component complexity |
| H1 entropy | H = -sum p_i log(p_i) for H1 | Cycle complexity |
| Max H0 persistence | Longest-lived component | Dominant market structure duration |
| Max H1 persistence | Longest-lived cycle (approx.) | Duration of cyclical regime |
| H0 landscape integral | Integral of lambda_1 for H0 | Total H0 topological significance |
| H1 landscape integral | Integral of lambda_1 for H1 (approx.) | Total H1 topological significance |

**Persistence entropy** is defined as:

$$H = -\sum_{i} p_i \log(p_i), \quad p_i = \frac{\ell_i}{\sum_j \ell_j}, \quad \ell_i = d_i - b_i$$

where (b_i, d_i) are birth-death pairs. High entropy indicates complex, multi-scale topological structure (many features with similar lifetimes); low entropy indicates dominance by a few long-lived features (simple structure).

**Persistence landscape** provides a functional summary amenable to statistical analysis:

$$\lambda_k(t) = k\text{-th largest value of } \min(t - b_i, d_i - t) \text{ over all pairs } (b_i, d_i)$$

The integral of lambda_1 measures the total topological "weight" of the persistence diagram.

#### Regime Classification

The 8 topological features are concatenated with the current latent state z and passed through a 2-layer MLP to classify the market into one of 4 regimes: **bull**, **bear**, **range-bound**, or **crisis**.

**NEXUS configuration**: window=50, takens_dim=3, takens_tau=1, max_homology_dim=1, n_landscape_bins=20.

---

### 4.5 TD-MPC2 Planner

#### Model-Based Planning: Simulate, Don't Predict

The TD-MPC2 planner (Hansen et al., 2024) represents the culmination of the NEXUS architecture. Given a learned world model, the planner does not predict a single future -- it **simulates thousands of possible futures** and selects the action sequence with the highest expected risk-adjusted return.

This is the same principle that powered AlphaGo's defeat of Lee Sedol and MuZero's superhuman Atari play: model-based planning with learned dynamics.

#### Components

**Latent Dynamics Model**: Predicts the next latent state given current state and action:

$$z_{t+1} = f_\theta(z_t, a_t) = \text{SimNorm}(\text{MLP}([z_t; a_t]))$$

The MLP has 3 layers with Mish activation and LayerNorm. The output passes through **SimNorm** (from TD-MPC2): features are partitioned into groups of 8, and softmax is applied within each group. This enforces sparse, stable representations and prevents latent state collapse.

**Reward Model**: Predicts scalar reward (risk-adjusted return) from state-action pair. The reward function has explicit units:

$$r_{t+1} = \underbrace{a_t^\top \Delta p_{t+1}}_{\text{P\&L in index points}} - \underbrace{\frac{\text{commission\_bps} \times |\Delta a_t|}{10{,}000}}_{\text{round-trip commission}} - \underbrace{\frac{\text{slippage\_bps} \times |\Delta a_t|}{10{,}000}}_{\text{estimated slippage}} - \underbrace{\lambda \cdot a_t^\top \Sigma_t a_t}_{\text{risk penalty (annualized var)}}$$

where:
- Delta p_{t+1} is the vector of asset price changes (index points for derivatives, percentage for crypto)
- Delta a_t = a_t - a_{t-1} is the trade vector (change in position)
- commission_bps = 3 (NIFTY futures) or 5 (BANKNIFTY futures) per leg
- slippage_bps = 2 (estimated market impact for retail-sized orders in liquid contracts)
- Sigma_t is the rolling covariance matrix (60-day exponentially weighted)
- lambda = 0.1 (risk aversion parameter, controlling the variance-return tradeoff)

The reward model R_phi learns to approximate this function from state-action pairs:

$$\hat{r}_t = R_\phi(z_t, a_t) = \text{MLP}([z_t; a_t]) \rightarrow \mathbb{R}$$

During planning, R_phi provides fast reward estimates without requiring actual price data for future steps. The explicit cost structure above is used during training to generate ground-truth reward labels.

**Value Model**: Twin value functions (ensemble of 2) for conservative value estimation:

$$V^{(1)}_\psi(z), V^{(2)}_\psi(z) \rightarrow \mathbb{R}$$

Conservative (pessimistic) value: V_min(z) = min(V^{(1)}(z), V^{(2)}(z)). The twin architecture prevents value overestimation (Fujimoto et al., 2018).

#### Cross-Entropy Method (CEM) Planning Algorithm

```
ALGORITHM: CEM Planning in Learned Latent Space
================================================
Input:  z_0 (current latent state from encoder)
Output: a_0* (optimal first action = position vector)

1. Initialize action prior:
     mu = 0^{H x d_action}
     sigma = 0.5 * 1^{H x d_action}

2. FOR iteration = 1 to M (M=6):

   a. SAMPLE N=512 action sequences from prior:
        a^{(i)}_{1:H} ~ N(mu, sigma^2),  clipped to [-0.25, +0.25]

   b. ROLLOUT each trajectory through dynamics:
        FOR t = 1 to H:
          r^{(i)}_t = R(z^{(i)}_t, a^{(i)}_t)
          z^{(i)}_{t+1} = f(z^{(i)}_t, a^{(i)}_t)

   c. EVALUATE: cumulative discounted reward + terminal value:
        G^{(i)} = sum_{t=1}^{H} gamma^{t-1} * r^{(i)}_t
                  + gamma^H * V_min(z^{(i)}_{H+1})

   d. SELECT top-K=64 elites by G^{(i)}

   e. REFIT prior to elites (softmax-weighted):
        w_k = softmax(G^{elite_k} / temperature)
        mu_new = sum_k w_k * a^{elite_k}
        sigma_new = sqrt(sum_k w_k * (a^{elite_k} - mu_new)^2)

3. RETURN: a_0* = a^{best}_{t=0}
     (first action of trajectory with highest total return)
```

**Planning horizon**: H=5 trading days
**Samples per iteration**: N=512
**Elite fraction**: K=64 (top 12.5%)
**CEM iterations**: M=6
**Temperature**: 0.5 (controls sharpness of elite weighting)
**Discount factor**: gamma=0.99
**Position clamp**: [-0.25, +0.25] per asset

#### TD Learning for World Model Training

The dynamics, reward, and value models are trained via temporal difference (TD) learning from a replay buffer of transitions (z_t, a_t, r_t, z_{t+1}, done_t):

**Dynamics loss**: $\mathcal{L}_{\text{dyn}} = \|f_\theta(z_t, a_t) - z_{t+1}\|^2$

**Reward loss**: $\mathcal{L}_{\text{rew}} = \|R_\phi(z_t, a_t) - r_t\|^2$

**Value loss** (TD(0) with target network):

$$\mathcal{L}_{\text{val}} = \frac{1}{2} \sum_{i=1}^{2} \|V^{(i)}_\psi(z_t) - (r_t + \gamma (1-d_t) V_{\text{min}}(z_{t+1}))\|^2$$

---

### 4.6 Multi-Scale VQ Tokenizer

The tokenizer bridges raw market data and the Mamba-2 backbone, operating at three temporal scales that capture the multi-scale nature of financial markets (Muller et al., 1997):

#### Three Scales

| Scale | Resolution | Features | Financial Content |
|-------|-----------|----------|-------------------|
| **Micro** | 1-minute bars | 8 per bar | Local price dynamics, microstructure |
| **Meso** | Hourly aggregates | 8 per bar | Intraday patterns, session structure |
| **Macro** | Daily bars | 8 per bar | Trend structure, regime |

#### Delta Encoding

Raw price levels are non-stationary (random walk). The DeltaEncoder converts to stationary representations:

- **Price columns (OHLC)**: Log returns = log(p_t / p_{t-1})
- **Volume columns**: Raw deltas = v_t - v_{t-1}
- **Other features**: Passthrough (pre-computed features already stationary)

All computations use shift(1) for strict causality: the delta at time t uses only data available at time t.

#### Vector Quantization

The optional VQ layer discretizes continuous embeddings into a finite codebook of 512 entries, analogous to how language models tokenize text into subwords:

- **Codebook**: 512 entries of dimension 64
- **EMA updates**: Codebook vectors updated via exponential moving average (no codebook gradients needed)
- **Dead code restart**: Entries unused for 2+ batches are re-initialized from random encoder outputs
- **Straight-through estimator**: Gradients pass through the quantization step unchanged
- **Commitment loss**: Pulls encoder outputs toward their nearest codebook entry

The VQ codebook learns a finite vocabulary of "market micro-states" -- each codebook entry corresponds to a recognizable market pattern at the corresponding scale.

#### VQ Discretization and Alpha Signal Preservation

A known limitation of vector quantization is that discretization can quantize away fine-grained alpha signals. When continuous micro-price deltas or order flow imbalances are snapped to the nearest of 512 codebook entries, subtle variations within a codebook region -- which may carry exploitable signal -- are destroyed. This is a fundamental tension: VQ provides useful regime-level abstraction but sacrifices within-regime nuance.

Our recommended approach is a **hybrid architecture**: use VQ for macro and meso scales (where regime identification and structural pattern recognition benefit from discretization), but pass micro-scale features through a continuous linear projection that preserves fine-grained price dynamics. In NEXUS, this is configurable via the `use_vq` flag per scale. For strategies that depend on microstructural alpha (tick-level mean reversion, order flow imbalance), we recommend disabling VQ at the micro scale entirely.

#### Cross-Scale Fusion

The three scales have different temporal resolutions. Cross-scale fusion aligns them to the macro (daily) resolution via adaptive average pooling, then fuses via a learned MLP:

$$\text{fused} = \text{MLP}([\text{pool}(\text{micro}); \text{pool}(\text{meso}); \text{macro}])$$

**Configuration**: d_model=256, num_embeddings=512, d_embedding=64, commitment_cost=0.25.

---

## 5. Training Pipeline

NEXUS employs a 3-phase training pipeline inspired by DreamerV3 (Hafner et al., 2025) and TD-MPC2 (Hansen et al., 2024):

```
  Phase 1                Phase 2                Phase 3
  JEPA Pre-training      World Model Training   Policy Distillation
  (Self-Supervised)      (TD Learning)          (Supervised)
  ==================     ====================   ====================

  Input: market data     Input: replay buffer   Input: market data
  Target: latent pred    Target: TD targets     Target: CEM actions

  Learns: encoder        Learns: dynamics,      Learns: policy head
          predictor              reward,         (fast inference)
          hyperbolic             value
          projections

  Loss: L_jepa + L_hyp   Loss: L_dyn +          Loss: MSE(policy,
        + L_var (if              L_rew + L_val          CEM_action)
         collapse)

  Epochs: 50             Epochs: 30             Epochs: 20
  LR: 1e-4               LR: 3e-4              LR: 1e-4
  EMA: 0.996->0.9999     Discount: 0.99         Frozen: encoder

  Output: pretrained     Output: world model    Output: distilled
          representations         for planning           fast policy
```

### Phase 1: JEPA Pre-training (Self-Supervised)

No labels or reward signals required. The model learns to predict future latent representations from past context, using only raw market data. Key design choices:

- **EMA schedule**: Linear warmup from tau=0.996 to tau=0.9999 over training (following V-JEPA 2)
- **LR schedule**: Linear warmup (1000 steps) + cosine decay to 1e-6
- **Mask ratio**: 40% of future timesteps masked
- **Hyperbolic regularization**: Lambda=0.1 weight on Lorentz distance between predicted and target hyperbolic embeddings
- **Collapse monitoring**: Per-dimension variance, effective rank, and mean cosine similarity evaluated every 100 steps; variance regularization (L_var) activated if any threshold trips

### Phase 2: World Model Training (TD Learning)

Collects transitions into a replay buffer (capacity 100K) by running the current policy on market data, then trains the dynamics/reward/value models via TD learning:

1. Encode market data to latent sequences using the frozen Phase 1 encoder
2. Generate actions via current policy head at each timestep
3. Store (z_t, a_t, r_t, z_{t+1}, done) transitions
4. Train dynamics, reward, and twin value functions from replay

### Phase 3: Policy Distillation

The CEM planner produces high-quality actions but requires ~3000 forward passes per decision (512 samples * 6 iterations). For real-time inference, we distill the planner into the lightweight policy head:

1. Freeze all model parameters except policy_head
2. For each market state: teacher_action = CEM_plan(z), student_action = policy_head(z)
3. Minimize MSE(student, teacher)

After distillation, the policy head produces near-planner-quality actions in a single forward pass.

### Infrastructure

- **Mixed precision**: Automatic mixed precision (AMP) with GradScaler on CUDA
- **Gradient clipping**: Max norm 1.0 across all phases
- **Checkpointing**: Model + optimizer + scaler + replay buffer size saved at configurable intervals
- **GPU memory management**: gc.collect() + torch.cuda.empty_cache() between phases (critical for avoiding false OOM)

---

## 6. Why This Combination is Novel

To our knowledge, no published work has combined all five components -- Mamba-2, JEPA, hyperbolic geometry, persistent homology, and TD-MPC2 planning -- into a single architecture for financial market modeling. The contribution is not merely the combination, but the specific architectural choices that make them mutually reinforcing: JEPA's latent predictions gain from hyperbolic geometry; the topological sensor benefits from operating in a geometrically appropriate latent space; and the planner leverages all three to simulate realistic market trajectories.

We acknowledge that unpublished industry work may have explored similar combinations. The quantitative finance industry is characterized by strong publication disincentives -- profitable approaches are closely guarded. Our claims of novelty are based on a thorough survey of the academic literature and publicly available industry research as of February 2026.

### Novelty Matrix

| Component | Domain of Origin | Used in Finance? | Used in ML? | Combined Here? |
|-----------|-----------------|------------------|-------------|----------------|
| **JEPA** | Computer Vision (Meta AI) | Not in published work | Yes (I-JEPA, V-JEPA, TS-JEPA) | **Yes** |
| **Mamba-2** | NLP / Sequence Modeling | Rare (a few papers) | Yes (language, genomics) | **Yes -- with JEPA + planning** |
| **Hyperbolic Geometry** | Graph Learning | Rare (word embeddings) | Yes (Poincare, Lorentz) | **Yes -- for trading latent space** |
| **Persistent Homology** | Topological Data Analysis | Academic papers only | Yes (sensor data, materials) | **Yes -- integrated end-to-end** |
| **TD-MPC2** | Robotics / RL | Not in published work | Yes (robot control, Atari) | **Yes -- for financial markets** |

### Synergies Between Components

The five components are not merely stacked -- they create synergies that amplify each other:

1. **Mamba-2 + JEPA**: The O(n) backbone enables JEPA to process year-long sequences (252 trading days) that would be infeasible with O(n^2) transformers. Long context is critical for learning multi-regime market dynamics.

2. **JEPA + Hyperbolic**: JEPA's latent predictions are naturally enhanced by hyperbolic geometry. The predictor learns that "market state moved from bull to correction" corresponds to a specific geodesic path on the hyperboloid, not an arbitrary vector in Euclidean space.

3. **Hyperbolic + Topology**: Topological features computed in hyperbolic latent space are more meaningful than in Euclidean space, because the geometry already respects market hierarchy. Topological changes in hyperbolic space correspond to genuine structural regime shifts, not geometric artifacts.

4. **Topology + Planner**: The topological regime sensor provides the planner with structural context. In a "crisis" regime (low beta_0, high entropy), the planner can automatically tighten position limits and shorten planning horizons.

5. **JEPA + Planner**: The JEPA predictor provides the *imagination* capability that the planner requires. Without a learned world model, the planner would need expensive Monte Carlo simulation from a hand-crafted market model. With JEPA, imagination is a single forward pass.

### What NEXUS Enables

- **Plan in latent space**: Simulate 512 hypothetical 5-day futures in ~50ms on GPU, evaluate each for risk-adjusted return, and select the optimal position -- all without ever predicting raw prices.
- **Detect topological regime changes**: Identify structural market transitions using exact H0 persistence and spectral H1 approximation, providing early warning signals that complement statistical indicators.
- **Represent market hierarchy naturally**: Embed macro-to-micro market structure with O(1) distortion in hyperbolic space, compared to O(log n) distortion in Euclidean space.
- **Process year-long tick-level sequences**: O(n) Mamba-2 backbone handles 100K+ tokens per forward pass.
- **Train without labels**: Phase 1 (JEPA pre-training) requires only raw market data -- no returns, no signals, no supervised targets.

---

## 7. Implementation Details

### Design Philosophy

NEXUS is implemented in **pure PyTorch** with minimal external dependencies (numpy for topology, einops for tensor manipulation). Every component is self-contained and independently testable. There are no calls to proprietary libraries, no CUDA kernels, and no dependencies that would complicate deployment.

### Model Sizes

| Size | d_model | d_latent | d_state | n_layers | CEM Samples | Parameters |
|------|---------|----------|---------|----------|-------------|------------|
| **Small** | 128 | 64 | 32 | 3 | 128 | ~1.2M |
| **Base** | 256 | 128 | 64 | 6 | 512 | ~5.8M |
| **Large** | 512 | 256 | 128 | 12 | 1024 | ~23M |

All sizes are instantiable via a single factory function:

```python
from nexus.model import create_nexus

model = create_nexus(n_features=192, n_assets=6, size="base")
```

### Hardware Requirements

- **Training**: Tesla T4 (16 GB VRAM) or better. Mixed precision (AMP) enabled by default.
- **Inference (with planning)**: ~50ms per decision on T4 (512 samples x 6 CEM iterations)
- **Inference (direct policy)**: ~2ms per decision on T4 (single forward pass after Phase 3 distillation)
- **CPU**: Full functionality on CPU (AMP auto-disabled), suitable for development and testing

### Testing

40 unit tests organized by component, verifying:

- **Mamba-2**: Output shapes, causality (future inputs do not affect past outputs), residual connections, gradient flow
- **Hyperbolic**: Minkowski dot product signature, hyperboloid constraint satisfaction, exp/log map inversion, distance positivity, centroid on-manifold
- **Topology**: Takens embedding shapes, H0 persistence count (N-1 for N points), entropy non-negativity, Betti number monotonicity
- **JEPA**: Forward pass shapes, loss decrease during training, target encoder gradient isolation, EMA update direction, imagination shapes
- **Planner**: Dynamics/reward/value model shapes, CEM action bounds, TD loss computation and gradient flow
- **NEXUS (end-to-end)**: Full forward pass, direct policy inference, planning inference, imagined futures, hyperbolic embedding constraint, gradient flow through entire model, 10-step training loop stability

---

## 8. Code Statistics

| File | Lines | Description |
|------|------:|-------------|
| `mamba2.py` | 282 | Selective State Space backbone with selective scan |
| `hyperbolic.py` | 341 | Lorentz manifold operations, projections, attention |
| `jepa.py` | 386 | JEPA World Model with EMA target encoder |
| `topology.py` | 490 | Persistent homology sensor (pure NumPy) |
| `planner.py` | 387 | TD-MPC2 CEM planner with dynamics/reward/value |
| `model.py` | 403 | Full NEXUS model -- ties all components together |
| `config.py` | 100 | Configuration dataclass with all hyperparameters |
| `tokenizer.py` | 773 | Multi-scale VQ-VAE tokenizer with delta encoding |
| `trainer.py` | 1,140 | 3-phase training pipeline with AMP + checkpointing |
| `__init__.py` | 37 | Package metadata and architecture docstring |
| `test_nexus.py` | 474 | 40 comprehensive unit tests |
| **Total** | **4,813** | **Complete system** |

---

## 9. Limitations and Future Work

### 9.1 Known Limitations

We believe it is important to be explicit about the current limitations of NEXUS. Several are fundamental to the approach; others are engineering constraints that future work can address.

**Exogenous assumption and market impact**: NEXUS assumes the participant's actions do not affect market state transitions (Section 2.1). This is reasonable for retail-sized positions in liquid index derivatives but breaks down for larger participants, illiquid instruments, or during market stress when liquidity evaporates. Relaxing this assumption requires modeling market impact within the transition dynamics, which introduces a challenging feedback loop between planning and state prediction.

**JEPA collapse in low-SNR regimes**: As discussed in Section 4.2, the EMA target encoder is an empirical defense against representation collapse, not a theoretical guarantee. Financial time series have SNR of approximately 0.03, placing them in a regime where collapse to the unconditional mean is energetically favorable. Our monitoring diagnostics (variance, effective rank, cosine similarity) detect collapse but do not prevent it with certainty. The theoretical conditions under which JEPA provably avoids collapse remain an open problem.

**Topological approximations**: Our H1 computation uses a spectral proxy rather than exact persistent homology (Section 4.4). This means we lack the stability guarantees (bottleneck distance theorem) that make persistent homology theoretically robust. Additionally, the Takens embedding parameters (d=3, tau=1) are heuristic choices that may be suboptimal for nonstationary, noisy financial data.

**VQ information loss**: Vector quantization at the tokenizer stage (Section 4.6) can destroy fine-grained alpha signals. While we recommend a hybrid approach (VQ for macro/meso, continuous for micro), the current default configuration applies VQ uniformly across scales.

**Limited empirical validation**: As of this writing, NEXUS has been tested on synthetic data and unit tests but has not yet undergone rigorous walk-forward backtest validation on historical market data with realistic transaction costs. The architecture is theoretically motivated, but trading systems are ultimately judged by out-of-sample performance after costs.

**Hyperbolic geometry assumptions**: The benefit of hyperbolic embedding relies on market structure being approximately tree-like. In reality, cross-sector correlations, contagion dynamics, and regime-dependent factor structures introduce non-tree edges that hyperbolic space does not naturally represent. The fixed curvature K=-1 may also be suboptimal for different market regimes.

**Computational cost of planning**: While the distilled policy head reduces inference to ~2ms, the full CEM planner requires ~50ms per decision and ~3000 forward passes. For HFT applications requiring sub-millisecond latency, this is prohibitive even after distillation.

**Stationarity of learned dynamics**: The dynamics model f(z_t) is trained on historical data and may fail to generalize to genuinely novel market regimes (e.g., a new type of crisis, regulatory changes, or structural market microstructure shifts). The model has no mechanism for detecting that it is operating outside its training distribution, beyond the topological sensor's regime classification.

### 9.2 Near-Term Future Work (Q1 2026)

- **Real-time inference pipeline**: WebSocket market data feed -> tokenizer -> encoder -> planner -> order management. Target latency: <100ms from tick to order.
- **Walk-forward validation**: Expanding-window backtest on 2+ years of India FnO data (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY) plus crypto (BTC, ETH) with realistic transaction costs.
- **Production TFT integration**: Feed NEXUS latent states as additional features into the existing Temporal Fusion Transformer pipeline for ensemble signal generation.
- **Exact H1 computation**: Replace spectral proxy with Ripser or GUDHI for exact H1 persistent homology, enabling proper bottleneck distance stability guarantees.
- **Adaptive Takens parameters**: Use false nearest neighbors and mutual information to select embedding dimension d and delay tau, rather than fixed heuristics.

### 9.3 Medium-Term Future Work (Q2-Q3 2026)

- **Multi-agent competitive dynamics**: Model the market as a multi-agent system where NEXUS is one player among institutional, retail, and algorithmic participants. Train adversarial agents to stress-test the planner.
- **Conformal prediction intervals in hyperbolic space**: Extend conformal prediction (Vovk et al., 2005) to the Lorentz manifold to produce distribution-free prediction sets with guaranteed coverage -- providing calibrated uncertainty quantification on the hyperboloid.
- **Adaptive curvature learning**: Replace fixed curvature K=-1 with a learnable, potentially time-varying curvature that adapts to the current market regime. Crisis periods may require more negative curvature (faster exponential growth of representable states).
- **Out-of-distribution detection**: Add explicit OOD detection to the encoder, flagging when current market conditions are sufficiently far from the training distribution that model predictions should be distrusted.

### 9.4 Long-Term Research Directions (2027+)

- **Causal discovery in latent space**: Apply causal inference methods (do-calculus, structural causal models) to the learned latent dynamics to discover genuine causal relationships between market factors, as opposed to mere statistical associations.
- **Federated learning across trading desks**: Train NEXUS across multiple proprietary data sources without sharing raw data, using federated averaging on the hyperboloid (Frechet mean aggregation of model updates in hyperbolic space).
- **Market impact modeling**: Relax the exogenous assumption by incorporating a learned market impact function into the transition dynamics, enabling deployment at institutional scale.

---

## 10. References

### Core Architecture Papers

1. **I-JEPA**: Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR 2023*. arXiv:2301.08243.

2. **V-JEPA 2**: Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., & Ballas, N. (2025). Revisiting Feature Prediction for Learning Visual Representations from Video. *Meta AI, June 2025*.

3. **TS-JEPA**: Ekambaram, V., Jati, A., Phan, N.H., Nguyen, N.A., Dayama, P., Kalagnanam, J., & Muller, M. (2024). Joint Embeddings Go Temporal. *NeurIPS 2024*.

### State Space Models

4. **Mamba**: Gu, A. & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.

5. **Mamba-2**: Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. *ICML 2024*. arXiv:2405.21060.

6. **S4**: Gu, A., Goel, K., & Re, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. *ICLR 2022*.

### Model-Based Planning

7. **TD-MPC2**: Hansen, N., Su, H., & Wang, X. (2024). TD-MPC2: Scalable, Robust World Models for Continuous Control. *ICLR 2024*. arXiv:2310.16828.

8. **DreamerV3**: Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2025). Mastering Diverse Domains through World Models. *Nature, 2025*. arXiv:2301.04104.

9. **MuZero**: Schrittwieser, J., Antonoglou, I., Hubert, T., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. *Nature, 588*(7839), 604-609.

### Hyperbolic Geometry

10. **Poincare Embeddings**: Nickel, M. & Kiela, D. (2017). Poincare Embeddings for Learning Hierarchical Representations. *NeurIPS 2017*.

11. **Hyperbolic Neural Networks**: Ganea, O., Becigneul, G., & Hofmann, T. (2018). Hyperbolic Neural Networks. *NeurIPS 2018*.

12. **Lorentz Model**: Nickel, M. & Kiela, D. (2018). Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry. *ICML 2018*.

13. **LResNet**: Schwethelm, K., Kamp, M., Wilke, D.N., & Hammer, B. (2024). Fully Hyperbolic CNNs. arXiv:2412.14695.

### Topological Data Analysis

14. **TDA in Indian Stock Markets**: (2024). Investigation of Indian Stock Markets Using Topological Data Analysis. *ScienceDirect*.

15. **TDA Change Point Detection**: (2024). Change Point Detection Using Topological Data Analysis. *MDPI Mathematics*.

16. **TDA Crisis Prediction**: (2024). Topological Features for Financial Crisis Prediction. *Neural Computing and Applications*.

17. **Persistent Homology**: Edelsbrunner, H. & Harer, J. (2010). *Computational Topology: An Introduction*. American Mathematical Society.

18. **Takens' Theorem**: Takens, F. (1981). Detecting Strange Attractors in Turbulence. *Lecture Notes in Mathematics*, 898, 366-381.

19. **Stability Theorem**: Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of Persistence Diagrams. *Discrete & Computational Geometry*, 37(1), 103-120.

### Metric Embeddings

20. **Bourgain's Theorem**: Bourgain, J. (1985). On Lipschitz Embedding of Finite Metric Spaces in Hilbert Space. *Israel Journal of Mathematics*, 52(1-2), 46-52.

21. **LLR Lower Bound**: Linial, N., London, E., & Rabinovich, Y. (1995). The Geometry of Graphs and Some of Its Algorithmic Applications. *Combinatorica*, 15(2), 215-245.

22. **Sarkar's Construction**: Sarkar, R. (2011). Low Distortion Delaunay Embedding of Trees in Hyperbolic Plane. *Graph Drawing 2011*.

### Additional

23. **VQ-VAE**: van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. *NeurIPS 2017*.

24. **SimNorm**: Hansen, N. et al. (2024). SimNorm in TD-MPC2 (see [7]).

25. **VICReg**: Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR 2022*.

26. **POMDP**: Kaelbling, L.P., Littman, M.L., & Cassandra, A.R. (1998). Planning and Acting in Partially Observable Stochastic Domains. *Artificial Intelligence*, 101(1-2), 99-134.

---

*This document describes the NEXUS architecture as of February 2026. The system is under active development. All claims about novelty are made to the best of our knowledge based on a thorough survey of the published academic literature. We acknowledge that unpublished industry work may have explored similar approaches.*

---

```
 ___   _   _____  __  __  _   _  ___
|   \ | | | ____| \ \/ / | | | |/ __|
| |) || | | __|    >  <  | |_| |\__ \
|___/ |_| |_____|_/_/\_\_|\___/ |___/
                    v0.2.0
```
