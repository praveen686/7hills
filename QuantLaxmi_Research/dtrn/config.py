"""DTRN configuration — all hyperparameters and paths."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# Data paths
DATA_ROOT = Path("/home/ubuntu/Desktop/7hills/QuantLaxmi/data/telegram_source_files/india_tick_data")
TELEGRAM_ROOT = Path("/home/ubuntu/Desktop/7hills/QuantLaxmi_Telegram/files")
KITE_1MIN_ROOT = Path("/home/ubuntu/Desktop/7hills/QuantLaxmi/common/data/kite_1min")
TICK_ROOT = Path("/home/ubuntu/Desktop/7hills/QuantLaxmi/common/data/market/ticks")

# Instruments (name field in feather files)
INSTRUMENTS = ["NIFTY", "BANKNIFTY"]
INSTRUMENT_TYPES = ["FUT"]  # futures only for now


@dataclass
class DTRNConfig:
    """Full DTRN configuration."""

    # ── Data ──
    bar_interval: str = "1min"  # micro-bar interval
    instruments: list = field(default_factory=lambda: ["NIFTY", "BANKNIFTY"])

    # ── Feature Engine ──
    n_return_lags: int = 5          # lag-1 to lag-5 returns
    ewma_spans: list = field(default_factory=lambda: [10, 30, 60, 120])  # EWMA vol windows (in bars)
    zscore_window: int = 60         # z-score lookback
    rsi_period: int = 14            # RSI period

    # ── Dynamic Topology ──
    ewma_cov_span: int = 120        # EWMA span for covariance estimation
    top_k_edges: int = 6            # max incoming edges per node
    tau_on: float = 0.15            # edge activation threshold
    tau_off: float = 0.08           # edge deactivation threshold (hysteresis)
    max_edge_flip_rate: float = 0.02  # max fraction of edges that can change per step
    precision_reg: float = 1e-4     # regularization for precision matrix inversion

    # ── Graph Network ──
    d_embed: int = 32               # node embedding dimension
    d_hidden: int = 64              # GNN hidden dimension
    n_message_passes: int = 2       # L rounds of message passing
    d_temporal: int = 64            # GRU hidden dimension
    temporal_window: int = 60       # steps of temporal context for GRU

    # ── Regime Head ──
    n_regimes: int = 4              # K regimes
    regime_names: list = field(default_factory=lambda: [
        "calm_mr",       # R0: calm mean-reversion
        "trend",         # R1: trending
        "high_vol",      # R2: high volatility / jumps
        "liq_stress",    # R3: liquidity stress
    ])

    # ── Policy Head ──
    max_position: float = 1.0       # max |position target|
    regime_gate_threshold: float = 0.6  # min regime confidence to trade

    # ── Training ──
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256           # number of time steps per batch
    pred_horizon: int = 5           # predict 5 bars ahead
    lambda_vol: float = 1.0         # vol prediction weight
    lambda_jump: float = 0.5        # jump prediction weight
    lambda_sparse: float = 0.01     # topology sparsity weight
    lambda_stable: float = 0.01     # topology stability weight
    lambda_tc: float = 0.001        # transaction cost penalty
    lambda_dd: float = 0.01         # drawdown penalty

    # ── Risk ──
    # max_lots: maximum number of LOTS (not contracts). 1 NIFTY lot = 75 contracts.
    max_lots: int = 10              # hard limit in lots
    max_daily_loss: float = 0.02    # 2% of capital
    max_drawdown: float = 0.05      # 5% rolling drawdown
    max_position_change_per_step: float = 0.2  # max 20% of max_lots change per bar

    # ── Execution (India FnO) ──
    # Rates effective as of Feb 2026. Sources: NSE circulars, Zerodha charges page.
    # STT on equity futures raised to 0.02% (sell side) per Budget 2025-26.
    # Exchange txn charge ~0.00173% for futures (NSE, as of Oct 2024 revision).
    # Stamp duty 0.002% on buy side (equity futures).
    # Brokerage: ₹20 flat or 0.03% whichever lower per executed order (Zerodha).
    brokerage_per_order: float = 20.0       # INR flat (Zerodha)
    exchange_txn_pct: float = 0.0000173     # 0.00173% NSE futures
    stt_pct: float = 0.0002                 # 0.02% STT on sell side (equity futures)
    gst_pct: float = 0.18                   # 18% GST on (brokerage + exchange charge)
    stamp_duty_pct: float = 0.00002         # 0.002% stamp duty (buy side)
    slippage_bps: float = 1.0               # 1 bps slippage estimate
    lot_sizes: dict = field(default_factory=lambda: {
        "NIFTY": 75,
        "BANKNIFTY": 30,
    })

    # ── Backtest ──
    initial_capital: float = 10_000_000.0  # 1 Crore INR
    train_days: int = 120           # rolling train window
    test_days: int = 20             # OOS test window
    purge_days: int = 5             # purge gap between train/test to avoid leakage
