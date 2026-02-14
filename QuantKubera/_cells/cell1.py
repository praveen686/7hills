# ============================================================================
# CELL 1: Setup & Configuration
# ============================================================================

import os
import sys
import warnings
import gc
import time
import math
import logging
import json
import hashlib

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gammaln, gamma as gamma_fn
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from kiteconnect import KiteConnect
from dotenv import load_dotenv
import pyotp
import requests
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Suppress warnings
# ---------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
# Suppress Keras masking warnings — we use fixed-length sequences (window_size=21),
# no padding, no variable-length inputs. Masking is irrelevant.
warnings.filterwarnings('ignore', message='.*does not support masking.*')
tf.get_logger().setLevel('ERROR')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('QuantKubera')

# ---------------------------------------------------------------------------
# GPU detection & memory growth
# ---------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"GPU memory growth setting failed: {e}")
    GPU_AVAILABLE = True
    GPU_NAME = tf.test.gpu_device_name() or gpus[0].name
else:
    GPU_AVAILABLE = False
    GPU_NAME = "None"

# ---------------------------------------------------------------------------
# Seed everything
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class MonolithConfig:
    # Data
    tickers: list = field(default_factory=lambda: [
        # NSE Index Futures
        'NIFTY', 'BANKNIFTY', 'FINNIFTY',
        # MCX Commodity Futures
        'GOLD', 'SILVER', 'CRUDEOIL', 'NATURALGAS', 'COPPER',
        # NSE Stock Futures (top FnO by volume)
        'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'SBIN',
    ])
    exchanges: dict = field(default_factory=lambda: {
        'NIFTY': 'NSE', 'BANKNIFTY': 'NSE', 'FINNIFTY': 'NSE',
        'GOLD': 'MCX', 'SILVER': 'MCX', 'CRUDEOIL': 'MCX',
        'NATURALGAS': 'MCX', 'COPPER': 'MCX',
        'RELIANCE': 'NFO', 'HDFCBANK': 'NFO', 'ICICIBANK': 'NFO',
        'INFY': 'NFO', 'TCS': 'NFO', 'SBIN': 'NFO',
    })
    lookback_days: int = 2500
    window_size: int = 21
    # Model
    hidden_size: int = 128
    num_heads: int = 4
    dropout_rate: float = 0.2
    # Training
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    early_stop_patience: int = 10
    lr_reduce_patience: int = 5
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6
    clipnorm: float = 1.0
    # Walk-forward
    min_train_days: int = 504   # 2 years minimum
    test_days: int = 63         # 3 months
    purge_gap: int = 5
    # Costs
    bps_cost: float = 0.0010   # 10 bps per side
    # Quick mode
    quick_mode: bool = False    # True: 1 ticker, fewer epochs; False: full universe

CFG = MonolithConfig()

# ---------------------------------------------------------------------------
# Version tag — verify after Kernel Restart that you see this version
# ---------------------------------------------------------------------------
NOTEBOOK_VERSION = "v2.1-2026-02-14"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 70)
print(f"QuantKubera Monolith {NOTEBOOK_VERSION}")
print("=" * 70)
print(f"  GPU Available : {GPU_AVAILABLE} ({GPU_NAME})")
print(f"  TensorFlow    : {tf.__version__}")
print(f"  NumPy         : {np.__version__}")
print(f"  Pandas        : {pd.__version__}")
print(f"  Seed          : {SEED}")
print(f"  Quick Mode    : {CFG.quick_mode}")
print(f"  Tickers       : {CFG.tickers}")
print(f"  Hidden Size   : {CFG.hidden_size}")
print(f"  Num Heads     : {CFG.num_heads}")
print(f"  Epochs        : {CFG.epochs}")
print(f"  Batch Size    : {CFG.batch_size}")
print(f"  Walk-forward  : train>={CFG.min_train_days}d, test={CFG.test_days}d, purge={CFG.purge_gap}d")
print(f"  Cost Model    : {CFG.bps_cost * 10000:.0f} bps per side")
print("=" * 70)