# Architecture

## System Overview

QuantLaxmi is structured as a six-layer pipeline. Each layer is decoupled and testable in isolation.

```
+-------------------+     +---------------------+     +--------------------+
|  Data Ingestion   | --> | Feature Engineering | --> | Signal Generation  |
|  (Kite, NSE,      |     | (MegaFeatureBuilder |     | (25 Strategies,    |
|   Binance, Tgram) |     |  200+ features)     |     |  ML/RL models)     |
+-------------------+     +---------------------+     +--------------------+
                                                              |
                                                              v
+-------------------+     +---------------------+     +--------------------+
|    Execution      | <-- |  Risk Management    | <-- | Portfolio          |
|  (Rust engine,    |     | (Risk gates, limits,|     |  Allocation        |
|   Kite API)       |     |  drawdown circuit)  |     | (HRP, inv-vol)    |
+-------------------+     +---------------------+     +--------------------+
```

## Python vs Rust Boundary

**Python** handles everything that benefits from rapid iteration and rich library support:
- Research notebooks and backtesting
- Feature engineering (MegaFeatureBuilder, 200+ features)
- ML model training (TFT, XGBoost, RL agents)
- Strategy signal generation
- FastAPI backend (19 routes)

**Rust** handles everything that requires low-latency determinism:
- Live order execution and management
- Real-time WebSocket feed handling
- Risk gates (position limits, drawdown circuits, notional caps)
- Event sourcing (WAL + hash chain)
- Broker connector adapters (Kite, Binance)

The boundary is the **signal interface**: Python produces a target position vector (instrument, direction, size) which Rust consumes and executes.

## Data Flow

```
Sources                    Storage                     Consumption
--------                   -------                     -----------
Zerodha Kite  --+
  tick data     |
  1-min OHLCV   |
  daily OHLCV   +--->  Hive-Partitioned Parquet  --->  DuckDB queries
                |       (11 GB, 27 categories,         |
NSE website   --+        316 days)                     v
  index close   |                                MegaFeatureBuilder
  FII/DII data  |                                (200+ features)
                |                                      |
Binance       --+                                      v
  BTC/ETH OHLCV                                  Strategy Signals
                                                       |
Telegram      --+                                      v
  news feed     +--->  FinBERT (T4 GPU)  ------> Sentiment Features
```

### Storage Schema

All market data is stored in Hive-partitioned Parquet under `common/data/`:

```
common/data/
  category=nse_index_close/
    date=2025-01-02/part-0.parquet
    date=2025-01-03/part-0.parquet
    ...
  category=kite_1min/
    date=2025-01-02/part-0.parquet
    ...
  category=tick_data/
    ...
```

DuckDB reads these via `read_parquet('common/data/**/*.parquet', hive_partitioning=true)`.

## Key Design Decisions

### 1. TimeGuard -- No Look-Ahead Bias

Every feature computation and strategy signal is wrapped in a `TimeGuard` context that enforces causality. A feature computed at time `t` can only use data from `t` and earlier. This is validated in tests and audits.

```python
# TimeGuard prevents access to future data
with TimeGuard(current_time=t):
    features = builder.compute(data_up_to_t)
```

### 2. Mandatory Cost Model

Every backtest must specify a `CostModel` with explicit per-leg costs:

```python
cost_model = CostModel(
    commission_bps=0,       # Commission in bps (usually 0 for index)
    slippage_bps=0,         # Slippage in bps
    fixed_cost_per_leg=3.0, # 3 index points per leg (NIFTY)
)
```

Results without costs are never reported. The cost model is separate from the strategy to prevent gaming.

### 3. Event Sourcing with Hash Chain

All trading events (signals, orders, fills, risk checks) are recorded in a write-ahead log with a cryptographic hash chain:

```
Event_n.hash = SHA256(Event_n.payload || Event_{n-1}.hash)
```

This provides a tamper-evident audit trail. The hash chain resets to a GENESIS hash on daily rotation.

### 4. MegaFeatureBuilder -- 10 Feature Groups

The `MegaFeatureBuilder` computes 200+ features organized into 10 groups:

| Group | Features | Source |
|---|---|---|
| 1. Returns | Log returns, momentum, mean-reversion | OHLCV |
| 2. Technical | RSI, MACD, Bollinger, ADX | OHLCV |
| 3. Volatility | Realized vol, Parkinson, GK, IV spreads | OHLCV + options |
| 4. Microstructure | Tick imbalance, VPIN, Kyle lambda | Tick data |
| 5. Information | Entropy, mutual information, transfer entropy | Multi-asset |
| 6. Fractional | Fractional differentiation (FFD) | OHLCV |
| 7. Ramanujan | Ramanujan sums, spectral features | OHLCV |
| 8. FTI | Flow-toxicity indicators | Order flow |
| 9. RMT | Random Matrix Theory eigenvalue features | Correlation matrix |
| 10. DFF | Divergence Flow Field (Helmholtz decomposition of OI) | NSE 4-party OI |

### 5. Walk-Forward Validation Protocol

All strategy performance is validated using walk-forward analysis:

```
Fold 1: [Train: days 1-180]   [Test: days 181-243]
Fold 2: [Train: days 64-243]  [Test: days 244-306]
Fold 3: [Train: days 127-306] [Test: days 307-369]
Fold 4: [Train: days 190-369] [Test: days 370-432]
```

OOS (out-of-sample) Sharpe ratios are the primary metric. In-sample results are reported but never used for strategy selection.

### 6. Sharpe Ratio Protocol

All Sharpe ratios follow a strict protocol to prevent inflation:

- `ddof=1` for standard deviation (unbiased estimator)
- `sqrt(252)` annualization (trading days)
- All calendar days in the period included (flat days = 0 return)
- T+1 signal lag (signal generated at close of day `t`, position entered at close of day `t+1`)

## Component Map

```
quantlaxmi/
  data/
    store.py              DataStore -- unified Parquet read/write
    loaders.py            Asset-specific loaders (index, FII, options)
    connectors/           Binance WebSocket + REST
    collectors/           NSE daily, Kite 1-min, Telegram news
  features/
    mega.py               MegaFeatureBuilder -- orchestrates all groups
    base.py               BaseFeatureBuilder interface
    technical.py          RSI, MACD, Bollinger, etc.
    volatility.py         Realized vol, Parkinson, GK
    divergence_flow.py    DFF features (Helmholtz decomposition)
    ...                   (12 more feature modules)
  models/
    ml/tft/               Temporal Fusion Transformer
    rl/                   RL for Finance (MDP, DQN, DDPG, etc.)
    afml/                 Advances in Financial ML (meta-labeling, HRP)
  strategies/
    base.py               BaseStrategy interface
    protocol.py           Signal protocol definition
    registry.py           Strategy registry (S1-S25)
    s1_vrp/ ... s25_divergence_flow/
  core/
    base/timeguard.py     Look-ahead bias prevention
    pricing/              Option pricing (Black-Scholes, binomial)
    risk/                 Risk metrics, VaR, drawdown
    execution/            Order management, fill simulation
    events/               Event sourcing, WAL, hash chain
    backtest/             Vectorized backtester
  engine/
    api/                  FastAPI application (19 routes)
    paper.py              Paper trading runtime
```

## Deployment

```
EC2 Instance (32 vCPUs, 124 GB RAM, T4 GPU)
  |
  +-- Python venv (quantlaxmi/)
  |     +-- FastAPI (port 8000)
  |     +-- Data collectors (cron)
  |     +-- ML training (GPU)
  |
  +-- Rust binary (rust/)
  |     +-- Execution engine
  |     +-- WebSocket feeds
  |     +-- Risk gates
  |
  +-- Next.js (ui/)
        +-- Dev server (port 3000)
```
