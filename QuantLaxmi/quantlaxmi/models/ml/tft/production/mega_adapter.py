"""MegaFeatureAdapter — multi-asset feature alignment for TFT pipeline.

Extracted from the RL integration backbone. This is a pure data adapter
with zero RL dependencies: wraps MegaFeatureBuilder for multi-asset
outer-join alignment into (n_days, n_assets, n_features) tensors.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol mapping: env name → MegaFeatureBuilder symbol
# ---------------------------------------------------------------------------

_SYMBOL_MAP = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "Nifty Bank",
    "FINNIFTY": "NIFTY FINANCIAL SERVICES",
    "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
    "BTCUSDT": "BTCUSDT",
    "ETHUSDT": "ETHUSDT",
    "SOLUSDT": "SOLUSDT",
}


class MegaFeatureAdapter:
    """Convert per-asset MegaFeatureBuilder output to multi-asset tensor.

    Calls ``MegaFeatureBuilder.build(symbol, start, end)`` per asset,
    outer-joins on date, forward-fills, zeros remaining NaN, and returns
    an aligned (n_days, n_assets, n_features) ndarray plus the feature
    name registry.

    Parameters
    ----------
    symbols : list[str]
        Trading symbols (e.g. ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]).
    """

    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols

    def build_multi_asset(
        self, start: str, end: str
    ) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
        """Build aligned multi-asset feature tensor.

        Parameters
        ----------
        start, end : str
            Date range "YYYY-MM-DD".

        Returns
        -------
        features : ndarray of shape (n_days, n_assets, n_features)
        names : list[str] — feature column names
        dates : pd.DatetimeIndex
        """
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        per_asset_dfs: list[pd.DataFrame] = []
        per_asset_names: list[list[str]] = []

        for asset_idx, symbol in enumerate(self.symbols):
            mega_sym = _SYMBOL_MAP.get(symbol.upper(), symbol)
            df, names = builder.build(mega_sym, start, end)
            per_asset_dfs.append(df)
            per_asset_names.append(names)
            logger.info(
                "MegaFeatureAdapter: %s → %d features, %d rows",
                mega_sym, len(names), len(df),
            )

        if not per_asset_dfs:
            raise ValueError("No features built — check symbols and data availability")

        # Union of all feature names (preserves order: first asset's names first,
        # then any additional names from subsequent assets in order)
        seen: set[str] = set()
        feature_names: list[str] = []
        for names in per_asset_names:
            for n in names:
                if n not in seen:
                    seen.add(n)
                    feature_names.append(n)

        # Outer-join all assets on date
        all_dates = sorted(
            set().union(*(df.index for df in per_asset_dfs))
        )
        dates = pd.DatetimeIndex(all_dates)
        n_days = len(dates)
        n_assets = len(self.symbols)
        n_features = len(feature_names)

        features = np.full((n_days, n_assets, n_features), np.nan, dtype=np.float64)

        for asset_idx, df in enumerate(per_asset_dfs):
            aligned = df.reindex(dates)
            # Forward-fill within each asset (causal — only uses past values)
            aligned = aligned.ffill()
            # Reindex columns to the union feature set (missing cols become NaN)
            aligned = aligned.reindex(columns=feature_names)
            vals = aligned.values
            features[:, asset_idx, :] = vals

        # Zero remaining NaN (leading NaN before first valid date, or
        # asset-specific features missing for other assets)
        features = np.nan_to_num(features, nan=0.0)

        logger.info(
            "MegaFeatureAdapter: %d days × %d assets × %d features (union)",
            n_days, n_assets, n_features,
        )
        return features, feature_names, dates
