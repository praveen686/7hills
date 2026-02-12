"""NEXUS Data Bridge -- connects NEXUS to real market data.

Loads market data from QuantLaxmi's DuckDB/Parquet store and prepares
it for JEPA self-supervised pre-training.

Data sources:
    - NSE daily index data (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)
      via Hive-partitioned Parquet at common/data/market/nse_index_close/
    - Crypto daily (BTCUSDT, ETHUSDT) via Binance 1h klines aggregated
      to daily OHLCV at common/data/binance/{symbol}/1h/
    - NSE 1-min data (NIFTY, BANKNIFTY spot) via Kite at
      common/data/kite_1min/{symbol}_SPOT/
    - NSE raw daily CSV (fallback) at common/data/nse/daily/{date}/index_close.csv

All features are strictly causal (no look-ahead bias). Data paths are
configurable via environment variables with sensible defaults.

Usage
-----
    from nexus.data import create_dataloaders

    train_loader, val_loader = create_dataloaders(
        data_root="/path/to/common/data",
        context_len=126,
        target_len=126,
        batch_size=32,
    )

    for batch in train_loader:
        context = batch["context"]          # (B, ctx_len, n_features)
        target = batch["target"]            # (B, tgt_len, n_features)
        target_pos = batch["target_positions"]  # (B, tgt_len) int
        asset_id = batch["asset_id"]        # (B,) int
        # ... feed to NEXUS model
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .tokenizer import MarketFeatureExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (configurable via environment variables)
# ---------------------------------------------------------------------------

_DEFAULT_DATA_ROOT = os.environ.get(
    "NEXUS_DATA_ROOT",
    "/home/ubuntu/Desktop/7hills/QuantLaxmi/common/data",
)

# Asset registries
NSE_SYMBOLS = ["NIFTY 50", "NIFTY BANK", "NIFTY FINANCIAL SERVICES", "NIFTY MIDCAP SELECT"]
NSE_SYMBOL_SHORT = {
    "NIFTY 50": "NIFTY",
    "NIFTY BANK": "BANKNIFTY",
    "NIFTY FINANCIAL SERVICES": "FINNIFTY",
    "NIFTY MIDCAP SELECT": "MIDCPNIFTY",
}
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
KITE_1MIN_SYMBOLS = ["NIFTY_SPOT", "BANKNIFTY_SPOT", "FINNIFTY_SPOT", "MIDCPNIFTY_SPOT"]

# All asset IDs (consistent ordering for multi-asset model)
ALL_ASSETS = (
    list(NSE_SYMBOL_SHORT.values()) + CRYPTO_SYMBOLS
)  # NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, BTCUSDT, ETHUSDT


# ---------------------------------------------------------------------------
# Data Loader -- load raw data from disk
# ---------------------------------------------------------------------------


class NexusDataLoader:
    """Loads market data from QuantLaxmi's Parquet/CSV store.

    Supports three data sources:
        1. NSE daily index close (Hive-partitioned Parquet)
        2. Binance crypto 1h klines (aggregated to daily)
        3. Kite 1-min bars (if available)

    All data is returned as pandas DataFrames with standardized columns:
    date, open, high, low, close, volume.

    Parameters
    ----------
    data_root : str
        Root directory of the market data store.
    """

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = Path(data_root or _DEFAULT_DATA_ROOT)

    # -- NSE Daily -----------------------------------------------------------

    def _load_nse_daily(self, symbol: str) -> "pd.DataFrame":
        """Load NSE daily index data from Hive-partitioned Parquet.

        Tries the Parquet store first (market/nse_index_close/), falls back
        to raw CSV files (nse/daily/{date}/index_close.csv).

        Parameters
        ----------
        symbol : str
            NSE index name as it appears in the "Index Name" column,
            e.g. "NIFTY 50", "NIFTY BANK".

        Returns
        -------
        pd.DataFrame
            Columns: date, open, high, low, close, volume.
            Sorted by date ascending. Empty DataFrame on failure.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        # Strategy 1: Hive-partitioned Parquet via DuckDB
        parquet_dir = self.data_root / "market" / "nse_index_close"
        if parquet_dir.exists():
            try:
                return self._load_nse_parquet(parquet_dir, symbol)
            except Exception as e:
                logger.warning(
                    "Parquet load failed for %s: %s. Trying CSV fallback.",
                    symbol, e,
                )

        # Strategy 2: Raw CSV files per date
        csv_root = self.data_root / "nse" / "daily"
        if csv_root.exists():
            try:
                return self._load_nse_csv(csv_root, symbol)
            except Exception as e:
                logger.warning("CSV load failed for %s: %s", symbol, e)

        logger.error("No NSE data found for %s", symbol)
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    def _load_nse_parquet(
        self, parquet_dir: Path, symbol: str
    ) -> "pd.DataFrame":
        """Load from Hive-partitioned Parquet using DuckDB.

        DuckDB column names: "Index Name", "Open Index Value",
        "High Index Value", "Low Index Value", "Closing Index Value",
        "Volume", "date".

        Uses LOWER("Index Name") = LOWER(?) for exact match (not ILIKE).
        """
        try:
            import duckdb
        except ImportError:
            raise ImportError("duckdb is required to read Parquet data")

        query = f"""
            SELECT
                "date"::DATE AS date,
                TRY_CAST("Open Index Value" AS DOUBLE) AS open,
                TRY_CAST("High Index Value" AS DOUBLE) AS high,
                TRY_CAST("Low Index Value" AS DOUBLE) AS low,
                TRY_CAST("Closing Index Value" AS DOUBLE) AS close,
                COALESCE(TRY_CAST("Volume" AS DOUBLE), 0.0) AS volume
            FROM read_parquet('{parquet_dir}/**/*.parquet', hive_partitioning=true)
            WHERE LOWER("Index Name") = LOWER(?)
            ORDER BY date
        """

        con = duckdb.connect(":memory:")
        try:
            df = con.execute(query, [symbol]).fetchdf()
        finally:
            con.close()

        if df.empty:
            logger.warning("No Parquet data for symbol %s", symbol)
        else:
            df["date"] = pd.to_datetime(df["date"])
            # Drop rows with NaN close
            df = df.dropna(subset=["close"])
            logger.info(
                "Loaded %d days for %s from Parquet", len(df), symbol
            )

        return df

    def _load_nse_csv(
        self, csv_root: Path, symbol: str
    ) -> "pd.DataFrame":
        """Fallback: load from per-date CSV files.

        Each date directory has an index_close.csv with columns:
        "Index Name", "Open Index Value", etc.
        """
        all_rows = []
        for date_dir in sorted(csv_root.iterdir()):
            if not date_dir.is_dir():
                continue
            csv_file = date_dir / "index_close.csv"
            if not csv_file.exists():
                continue
            try:
                df_day = pd.read_csv(csv_file)
                # Exact match on Index Name (case-insensitive)
                mask = df_day["Index Name"].str.strip().str.lower() == symbol.lower()
                row = df_day[mask]
                if row.empty:
                    continue
                row = row.iloc[0]
                all_rows.append({
                    "date": pd.to_datetime(date_dir.name),
                    "open": float(str(row["Open Index Value"]).replace(",", "")),
                    "high": float(str(row["High Index Value"]).replace(",", "")),
                    "low": float(str(row["Low Index Value"]).replace(",", "")),
                    "close": float(str(row["Closing Index Value"]).replace(",", "")),
                    "volume": float(str(row.get("Volume", 0)).replace(",", "")),
                })
            except Exception as e:
                logger.debug("Error reading %s: %s", csv_file, e)
                continue

        if not all_rows:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_rows).sort_values("date").reset_index(drop=True)
        logger.info("Loaded %d days for %s from CSV", len(df), symbol)
        return df

    # -- Crypto Daily --------------------------------------------------------

    def _load_crypto_daily(self, symbol: str) -> "pd.DataFrame":
        """Load crypto daily data by aggregating Binance 1h klines.

        Reads Hive-partitioned Parquet from
        common/data/binance/{symbol}/1h/ and aggregates to daily OHLCV.

        Binance 1h columns: timestamp, open, high, low, close, volume,
        quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, date.

        Parameters
        ----------
        symbol : str
            Binance trading pair, e.g. "BTCUSDT".

        Returns
        -------
        pd.DataFrame
            Columns: date, open, high, low, close, volume.
            Sorted by date ascending.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        kline_dir = self.data_root / "binance" / symbol / "1h"
        if not kline_dir.exists():
            logger.warning("No crypto data directory: %s", kline_dir)
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )

        try:
            import duckdb
        except ImportError:
            # Fallback: read with pyarrow
            return self._load_crypto_pyarrow(kline_dir, symbol)

        query = f"""
            SELECT
                "date"::DATE AS date,
                FIRST(open ORDER BY timestamp) AS open,
                MAX(high) AS high,
                MIN(low) AS low,
                LAST(close ORDER BY timestamp) AS close,
                SUM(volume) AS volume
            FROM read_parquet('{kline_dir}/**/*.parquet', hive_partitioning=true)
            GROUP BY "date"::DATE
            ORDER BY date
        """

        con = duckdb.connect(":memory:")
        try:
            df = con.execute(query).fetchdf()
        finally:
            con.close()

        if df.empty:
            logger.warning("No crypto data for %s", symbol)
        else:
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["close"])
            logger.info(
                "Loaded %d days for %s from Binance 1h", len(df), symbol
            )

        return df

    def _load_crypto_pyarrow(
        self, kline_dir: Path, symbol: str
    ) -> "pd.DataFrame":
        """Fallback: load crypto data via PyArrow (no DuckDB)."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("Neither duckdb nor pyarrow available for crypto data")
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )

        try:
            table = pq.read_table(str(kline_dir))
            df = table.to_pandas()
        except Exception as e:
            logger.warning("PyArrow read failed for %s: %s", symbol, e)
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )

        # Aggregate 1h -> daily
        df["date"] = pd.to_datetime(df["date"])
        daily = df.groupby("date").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).reset_index().sort_values("date")

        logger.info(
            "Loaded %d days for %s via PyArrow", len(daily), symbol
        )
        return daily

    # -- Kite 1-min ----------------------------------------------------------

    def _load_1min(self, symbol: str) -> "pd.DataFrame":
        """Load 1-min bar data from Kite collector.

        Reads Hive-partitioned Parquet from common/data/kite_1min/{symbol}/.
        Columns: timestamp, open, high, low, close, volume, oi, date.

        Parameters
        ----------
        symbol : str
            Kite instrument symbol, e.g. "NIFTY_SPOT".

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, open, high, low, close, volume.
            Sorted by timestamp ascending. Empty DataFrame if unavailable.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        kite_dir = self.data_root / "kite_1min" / symbol
        if not kite_dir.exists():
            logger.debug("No 1-min data for %s at %s", symbol, kite_dir)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        try:
            import duckdb
        except ImportError:
            return self._load_1min_pyarrow(kite_dir, symbol)

        query = f"""
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM read_parquet('{kite_dir}/**/*.parquet', hive_partitioning=true)
            ORDER BY timestamp
        """

        con = duckdb.connect(":memory:")
        try:
            df = con.execute(query).fetchdf()
        finally:
            con.close()

        if df.empty:
            logger.debug("No 1-min data for %s", symbol)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            logger.info(
                "Loaded %d 1-min bars for %s", len(df), symbol
            )

        return df

    def _load_1min_pyarrow(
        self, kite_dir: Path, symbol: str
    ) -> "pd.DataFrame":
        """Fallback: load 1-min data via PyArrow."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        try:
            table = pq.read_table(str(kite_dir))
            df = table.to_pandas()
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(
                "Loaded %d 1-min bars for %s via PyArrow", len(df), symbol
            )
            return df
        except Exception as e:
            logger.warning("PyArrow 1-min read failed for %s: %s", symbol, e)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

    # -- Load All ------------------------------------------------------------

    def load_all(self) -> Dict[str, "pd.DataFrame"]:
        """Load all available symbols and return as a dict.

        Returns
        -------
        dict of str -> pd.DataFrame
            Keys are short asset names (NIFTY, BANKNIFTY, ..., BTCUSDT, ETHUSDT).
            Values are daily OHLCV DataFrames (columns: date, open, high, low,
            close, volume).
        """
        result = {}

        # NSE indices
        for full_name, short_name in NSE_SYMBOL_SHORT.items():
            try:
                df = self._load_nse_daily(full_name)
                if not df.empty:
                    result[short_name] = df
            except Exception as e:
                logger.warning("Failed to load %s: %s", full_name, e)

        # Crypto
        for sym in CRYPTO_SYMBOLS:
            try:
                df = self._load_crypto_daily(sym)
                if not df.empty:
                    result[sym] = df
            except Exception as e:
                logger.warning("Failed to load %s: %s", sym, e)

        logger.info(
            "Loaded %d assets: %s",
            len(result),
            ", ".join(f"{k}({len(v)}d)" for k, v in result.items()),
        )
        return result


# ---------------------------------------------------------------------------
# JEPA Masking Strategy
# ---------------------------------------------------------------------------


class JEPAMaskingStrategy:
    """Generates target position masks for JEPA pre-training.

    Two masking modes:
        - block_masking: Mask a contiguous future block (default for
          time series -- predict the next N steps).
        - random_masking: Randomly mask positions (for ablation /
          robustness studies).

    All masking is applied to the target window, which is already
    a separate future block in the sliding window dataset.
    """

    @staticmethod
    def block_masking(
        L: int,
        mask_ratio: float = 0.4,
        min_block: int = 4,
    ) -> "np.ndarray":
        """Create a contiguous block mask.

        Parameters
        ----------
        L : int
            Target sequence length.
        mask_ratio : float
            Fraction of target to mask (0.0 to 1.0).
        min_block : int
            Minimum block size.

        Returns
        -------
        np.ndarray of int, shape (n_masked,)
            Indices of masked (target) positions.
        """
        block_size = max(min_block, int(L * mask_ratio))
        block_size = min(block_size, L)
        start = np.random.randint(0, max(1, L - block_size + 1))
        return np.arange(start, start + block_size)

    @staticmethod
    def random_masking(
        L: int,
        mask_ratio: float = 0.3,
    ) -> "np.ndarray":
        """Randomly mask positions in the target.

        Parameters
        ----------
        L : int
            Target sequence length.
        mask_ratio : float
            Fraction of positions to mask.

        Returns
        -------
        np.ndarray of int, shape (n_masked,)
            Indices of masked positions, sorted.
        """
        n_mask = max(1, int(L * mask_ratio))
        indices = np.sort(np.random.choice(L, size=n_mask, replace=False))
        return indices


# ---------------------------------------------------------------------------
# Dataset for JEPA Pre-training
# ---------------------------------------------------------------------------


class NexusDataset(Dataset):
    """PyTorch Dataset for NEXUS JEPA training.

    Creates context/target pairs for self-supervised JEPA pre-training.
    Context = visible past window. Target = future window to predict.

    Sliding windows are generated from all asset time series, interleaved.
    Each sample contains:
        - context : (context_len, n_features) -- visible past
        - target : (target_len, n_features) -- masked future
        - target_positions : (target_len,) -- integer positions of targets
        - asset_id : int -- index into ALL_ASSETS list

    Parameters
    ----------
    data_dict : dict of str -> pd.DataFrame
        Keys are asset names, values are daily OHLCV DataFrames.
    context_len : int
        Number of timesteps in the visible context window.
    target_len : int
        Number of timesteps in the target (masked future) window.
    feature_extractor : MarketFeatureExtractor, optional
        Computes technical features from OHLCV. Defaults to standard
        7-feature extractor.
    stride : int
        Step size between consecutive windows.
    masking : str
        Masking strategy: "block" or "random".
    mask_ratio : float
        Fraction of target to mask.
    """

    def __init__(
        self,
        data_dict: Dict[str, "pd.DataFrame"],
        context_len: int = 126,
        target_len: int = 126,
        feature_extractor: Optional[MarketFeatureExtractor] = None,
        stride: int = 1,
        masking: str = "block",
        mask_ratio: float = 0.4,
    ):
        if not HAS_TORCH:
            raise ImportError("torch is required for NexusDataset")
        if not HAS_NUMPY:
            raise ImportError("numpy is required for NexusDataset")

        super().__init__()
        self.context_len = context_len
        self.target_len = target_len
        self.stride = stride
        self.masking = masking
        self.mask_ratio = mask_ratio
        self.feature_extractor = feature_extractor or MarketFeatureExtractor()

        # Build sample index: list of (asset_name, asset_id, start_idx)
        self.samples: List[Tuple[str, int, int]] = []
        self.feature_tensors: Dict[str, "torch.Tensor"] = {}

        total_window = context_len + target_len

        for asset_name, df in data_dict.items():
            if len(df) < total_window:
                logger.warning(
                    "Skipping %s: only %d bars (need %d)",
                    asset_name, len(df), total_window,
                )
                continue

            # Get asset ID
            if asset_name in ALL_ASSETS:
                asset_id = ALL_ASSETS.index(asset_name)
            else:
                asset_id = len(ALL_ASSETS)  # Unknown asset fallback

            # Extract features
            features = self.feature_extractor.extract_tensor(df)  # (N, n_feat)
            self.feature_tensors[asset_name] = features

            # Generate sliding window start indices
            N = len(features)
            for start in range(0, N - total_window + 1, stride):
                self.samples.append((asset_name, asset_id, start))

        logger.info(
            "NexusDataset: %d samples from %d assets "
            "(ctx=%d, tgt=%d, stride=%d)",
            len(self.samples),
            len(self.feature_tensors),
            context_len,
            target_len,
            stride,
        )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def n_features(self) -> int:
        """Number of features per timestep."""
        if self.feature_tensors:
            return next(iter(self.feature_tensors.values())).shape[1]
        return 7  # Default from MarketFeatureExtractor

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        """Get a single training sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict with:
            context : (context_len, n_features) -- visible past
            target : (target_len, n_features) -- future to predict
            target_positions : (target_len,) -- int positions
            asset_id : scalar int tensor
        """
        asset_name, asset_id, start = self.samples[idx]
        features = self.feature_tensors[asset_name]

        # Context: [start, start + context_len)
        context = features[start : start + self.context_len]

        # Target: [start + context_len, start + context_len + target_len)
        target_start = start + self.context_len
        target = features[target_start : target_start + self.target_len]

        # Target positions (relative indices within the full sequence)
        target_positions = torch.arange(
            self.target_len, dtype=torch.long
        )

        return {
            "context": context,
            "target": target,
            "target_positions": target_positions,
            "asset_id": torch.tensor(asset_id, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------


def create_dataloaders(
    data_root: Optional[str] = None,
    context_len: int = 126,
    target_len: int = 126,
    train_ratio: float = 0.8,
    batch_size: int = 64,
    stride: int = 1,
    purge_gap: int = 5,
    masking: str = "block",
    mask_ratio: float = 0.4,
    num_workers: int = 0,
    feature_extractor: Optional[MarketFeatureExtractor] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for NEXUS.

    Walk-forward split: train on earlier data, validate on later data.
    A purge gap of ``purge_gap`` days separates train/val to prevent
    look-ahead bias at the boundary.

    Parameters
    ----------
    data_root : str, optional
        Root data directory. Defaults to NEXUS_DATA_ROOT env var or
        /home/ubuntu/Desktop/7hills/QuantLaxmi/common/data.
    context_len : int
        Context (visible past) window length.
    target_len : int
        Target (masked future) window length.
    train_ratio : float
        Fraction of data for training (0.8 = first 80%).
    batch_size : int
        Batch size for DataLoader.
    stride : int
        Step size between consecutive windows.
    purge_gap : int
        Number of days between train and validation sets (no-man's-land).
    masking : str
        Masking strategy for JEPA targets.
    mask_ratio : float
        Fraction of target positions to mask.
    num_workers : int
        Number of DataLoader workers.
    feature_extractor : MarketFeatureExtractor, optional
        Custom feature extractor.

    Returns
    -------
    (train_loader, val_loader) : tuple of DataLoader
        Standard PyTorch DataLoaders ready for NEXUS training.
    """
    if not HAS_TORCH:
        raise ImportError("torch is required for create_dataloaders")
    if not HAS_PANDAS:
        raise ImportError("pandas is required for create_dataloaders")

    # Load all data
    loader = NexusDataLoader(data_root=data_root)
    data_dict = loader.load_all()

    if not data_dict:
        raise RuntimeError(
            f"No market data found at {data_root or _DEFAULT_DATA_ROOT}. "
            "Set NEXUS_DATA_ROOT environment variable or pass data_root."
        )

    # Walk-forward split per asset: first train_ratio fraction -> train,
    # rest -> val, with purge_gap removed in between.
    train_dict: Dict[str, pd.DataFrame] = {}
    val_dict: Dict[str, pd.DataFrame] = {}

    for name, df in data_dict.items():
        N = len(df)
        train_end = int(N * train_ratio)
        val_start = train_end + purge_gap

        if train_end < context_len + target_len:
            logger.warning(
                "Skipping %s for train: only %d bars in train split",
                name, train_end,
            )
            continue

        train_dict[name] = df.iloc[:train_end].reset_index(drop=True)

        if val_start < N and (N - val_start) >= context_len + target_len:
            val_dict[name] = df.iloc[val_start:].reset_index(drop=True)

    extractor = feature_extractor or MarketFeatureExtractor()

    # Create datasets
    train_dataset = NexusDataset(
        data_dict=train_dict,
        context_len=context_len,
        target_len=target_len,
        feature_extractor=extractor,
        stride=stride,
        masking=masking,
        mask_ratio=mask_ratio,
    )

    val_dataset = NexusDataset(
        data_dict=val_dict,
        context_len=context_len,
        target_len=target_len,
        feature_extractor=extractor,
        stride=stride,
        masking=masking,
        mask_ratio=mask_ratio,
    ) if val_dict else None

    # Create DataLoaders
    # IMPORTANT: no shuffle for time series -- walk-forward only
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    else:
        # Return a minimal loader that yields one dummy batch
        # so downstream code doesn't need to special-case None
        logger.warning("No validation data available; creating empty val_loader")
        val_loader = DataLoader(
            train_dataset,
            batch_size=min(batch_size, len(train_dataset)),
            shuffle=False,
            num_workers=0,
        )

    logger.info(
        "Created DataLoaders: train=%d samples, val=%d samples",
        len(train_dataset),
        len(val_dataset) if val_dataset else 0,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Synthetic Data (for testing / development without real data)
# ---------------------------------------------------------------------------


def create_synthetic_data(
    n_assets: int = 6,
    n_days: int = 500,
    seed: int = 42,
) -> Dict[str, "pd.DataFrame"]:
    """Create synthetic OHLCV data for testing.

    Generates geometric Brownian motion price series with realistic
    characteristics (vol clustering, mean-reverting volume).

    Parameters
    ----------
    n_assets : int
        Number of synthetic assets.
    n_days : int
        Number of trading days per asset.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict of str -> pd.DataFrame
        Keys are asset names, values are daily OHLCV DataFrames.
    """
    if not HAS_NUMPY or not HAS_PANDAS:
        raise ImportError("numpy and pandas required for synthetic data")

    rng = np.random.RandomState(seed)
    asset_names = ALL_ASSETS[:n_assets]
    result = {}

    for i, name in enumerate(asset_names):
        # Base price depends on asset type
        if "NIFTY" in name or "BANK" in name or "FIN" in name or "MID" in name:
            base_price = 20000.0 + rng.randn() * 2000
            daily_vol = 0.012
        else:
            # Crypto
            base_price = 50000.0 + rng.randn() * 10000
            daily_vol = 0.035

        # Generate GBM returns with vol clustering (GARCH-like)
        returns = np.zeros(n_days)
        vol = daily_vol
        for t in range(n_days):
            vol = 0.95 * vol + 0.05 * daily_vol + 0.1 * abs(rng.randn()) * daily_vol
            returns[t] = rng.randn() * vol

        # Build OHLCV
        close = base_price * np.exp(np.cumsum(returns))
        open_ = np.roll(close, 1)
        open_[0] = base_price
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n_days))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n_days))
        volume = np.abs(rng.randn(n_days) * 1e6 + 5e6)

        dates = pd.bdate_range(start="2024-01-02", periods=n_days)

        df = pd.DataFrame({
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        result[name] = df

    return result


def create_synthetic_dataloaders(
    n_assets: int = 6,
    n_days: int = 500,
    context_len: int = 126,
    target_len: int = 126,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple["DataLoader", "DataLoader"]:
    """Create DataLoaders from synthetic data (for testing/dev).

    Parameters
    ----------
    n_assets : int
        Number of synthetic assets.
    n_days : int
        Days of synthetic data per asset.
    context_len : int
        Context window length.
    target_len : int
        Target window length.
    batch_size : int
        Batch size.
    seed : int
        Random seed.

    Returns
    -------
    (train_loader, val_loader) : tuple of DataLoader
    """
    data = create_synthetic_data(n_assets=n_assets, n_days=n_days, seed=seed)
    extractor = MarketFeatureExtractor()

    # Manual split
    train_dict = {}
    val_dict = {}
    purge_gap = 5

    for name, df in data.items():
        N = len(df)
        train_end = int(N * 0.8)
        val_start = train_end + purge_gap
        train_dict[name] = df.iloc[:train_end].reset_index(drop=True)
        if val_start < N:
            val_dict[name] = df.iloc[val_start:].reset_index(drop=True)

    train_ds = NexusDataset(
        data_dict=train_dict,
        context_len=context_len,
        target_len=target_len,
        feature_extractor=extractor,
        stride=1,
    )

    val_ds = NexusDataset(
        data_dict=val_dict,
        context_len=context_len,
        target_len=target_len,
        feature_extractor=extractor,
        stride=1,
    ) if val_dict else train_ds

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader
