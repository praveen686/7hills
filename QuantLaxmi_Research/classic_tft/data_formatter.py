"""Data Formatter for Classic TFT.

Bridges raw time series data to the input format expected by ClassicTFTModel.
Handles:
- Column type classification (static, known, observed, target)
- Scaler fitting on training data only (StandardScaler)
- Sliding window creation for encoder-decoder structure
- Per-entity (symbol) data separation
- No look-ahead guarantee: scalers fit only on training portion

Usage
-----
    formatter = TFTDataFormatter.for_volatility()
    formatter.fit(train_df)
    train_windows = formatter.transform(train_df)
    test_windows = formatter.transform(test_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnDefinition:
    """Defines column types for TFT data formatting.

    Each column is classified as one of:
    - 'target': The variable to forecast
    - 'observed': Past-only real-valued inputs (only available in encoder)
    - 'known': Known real-valued inputs (available in both encoder and decoder)
    - 'static_real': Time-invariant real features
    - 'static_cat': Time-invariant categorical features
    - 'id': Entity identifier column (not used as feature)
    - 'date': Date/time column (not used as feature)
    """

    name: str
    col_type: str  # 'target', 'observed', 'known', 'static_cat', 'static_real', 'id', 'date'


@dataclass
class FormatterConfig:
    """Configuration for TFTDataFormatter."""

    encoder_steps: int = 252
    decoder_steps: int = 5
    target_col: str = "log_vol"
    id_col: str = "Symbol"
    date_col: str = "date"
    columns: list[ColumnDefinition] = field(default_factory=list)


class TFTDataFormatter:
    """Formats raw data into Classic TFT input tensors.

    Steps:
    1. fit(): Learn scalers from training data
    2. transform(): Create sliding windows from any data slice
    3. get_config(): Return ClassicTFTConfig with correct input sizes
    """

    def __init__(self, config: FormatterConfig):
        self.config = config
        self._scalers: dict[str, tuple[float, float]] = {}  # col -> (mean, std)
        self._cat_maps: dict[str, dict] = {}  # col -> {value: int}
        self._is_fitted = False

        # Separate columns by type
        self._target_cols = [c.name for c in config.columns if c.col_type == "target"]
        self._observed_cols = [c.name for c in config.columns if c.col_type == "observed"]
        self._known_cols = [c.name for c in config.columns if c.col_type == "known"]
        self._static_real_cols = [c.name for c in config.columns if c.col_type == "static_real"]
        self._static_cat_cols = [c.name for c in config.columns if c.col_type == "static_cat"]
        self._id_col = config.id_col
        self._date_col = config.date_col

    @classmethod
    def for_volatility(
        cls,
        encoder_steps: int = 252,
        decoder_steps: int = 5,
    ) -> "TFTDataFormatter":
        """Create formatter for Oxford-Man volatility forecasting.

        Column assignments:
        - Target: log_vol (log of rv5_ss)
        - Observed (past only): open_to_close, log_bv, log_medrv, log_rk_parzen,
          log_rv10, bv_ratio
        - Known (past + future): day_of_week (0-4), days_from_start
        - Static categorical: Symbol
        - ID: Symbol
        """
        columns = [
            ColumnDefinition("log_vol", "target"),
            # Observed (past only) - these are realized vol measures
            ColumnDefinition("open_to_close", "observed"),
            ColumnDefinition("log_bv", "observed"),
            ColumnDefinition("log_medrv", "observed"),
            ColumnDefinition("log_rk_parzen", "observed"),
            ColumnDefinition("log_rv10", "observed"),
            ColumnDefinition("bv_ratio", "observed"),
            # Known (past + future) - calendar features
            ColumnDefinition("dow_0", "known"),
            ColumnDefinition("dow_1", "known"),
            ColumnDefinition("dow_2", "known"),
            ColumnDefinition("dow_3", "known"),
            ColumnDefinition("dow_4", "known"),
            ColumnDefinition("days_from_start", "known"),
            # Static
            ColumnDefinition("Symbol", "static_cat"),
        ]

        config = FormatterConfig(
            encoder_steps=encoder_steps,
            decoder_steps=decoder_steps,
            target_col="log_vol",
            id_col="Symbol",
            date_col="date",
            columns=columns,
        )
        return cls(config)

    def prepare_oxford_man(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw Oxford-Man CSV into features.

        Adds derived columns: log_vol, log_bv, log_medrv, log_rk_parzen,
        log_rv10, bv_ratio, day_of_week dummies, days_from_start.
        """
        df = df.copy()

        # Parse date index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Create date column
        df["date"] = df.index

        # Log-transform volatility measures (add small epsilon for stability)
        eps = 1e-10
        df["log_vol"] = np.log(df["rv5_ss"].clip(lower=eps))
        df["log_bv"] = np.log(df["bv"].clip(lower=eps))
        df["log_medrv"] = np.log(df["medrv"].clip(lower=eps))
        df["log_rk_parzen"] = np.log(df["rk_parzen"].clip(lower=eps))
        df["log_rv10"] = np.log(df["rv10"].clip(lower=eps))

        # BV ratio (bipower variation / realized variance)
        df["bv_ratio"] = df["bv"] / df["rv5_ss"].clip(lower=eps)

        # Calendar features (known)
        dow = pd.get_dummies(df.index.dayofweek, prefix="dow").astype(float)
        dow.index = df.index
        # Keep only dow_0 through dow_4
        for i in range(5):
            col = f"dow_{i}"
            if col in dow.columns:
                df[col] = dow[col].values
            else:
                df[col] = 0.0

        # Days from start (per symbol)
        df["days_from_start"] = 0.0
        for sym in df["Symbol"].unique():
            mask = df["Symbol"] == sym
            sym_dates = df.loc[mask].index
            if len(sym_dates) > 0:
                start_date = sym_dates.min()
                df.loc[mask, "days_from_start"] = (
                    (sym_dates - start_date).days
                ).astype(float)

        # Drop rows with NaN in target
        df = df.dropna(subset=["log_vol"])

        return df

    def fit(self, df: pd.DataFrame) -> "TFTDataFormatter":
        """Fit scalers on training data only.

        Parameters
        ----------
        df : DataFrame with all required columns

        Returns
        -------
        self
        """
        # Fit scalers for real-valued columns
        real_cols = self._observed_cols + self._known_cols + self._target_cols
        for col in real_cols:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    self._scalers[col] = (
                        float(vals.mean()),
                        float(vals.std(ddof=1).clip(min=1e-10)),
                    )
                else:
                    self._scalers[col] = (0.0, 1.0)

        # Fit categorical encoders
        for col in self._static_cat_cols:
            if col in df.columns:
                unique_vals = sorted(df[col].dropna().unique())
                self._cat_maps[col] = {v: i for i, v in enumerate(unique_vals)}

        self._is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        return_raw_targets: bool = False,
    ) -> dict[str, np.ndarray]:
        """Create sliding windows from data.

        Parameters
        ----------
        df : DataFrame with all required columns
        return_raw_targets : if True, also return unscaled target values

        Returns
        -------
        dict with keys:
            'past_inputs': (N, encoder_steps, n_observed + n_known + output_size)
            'future_inputs': (N, decoder_steps, n_known)
            'static_cat': (N, n_static_cat) or None
            'targets': (N, decoder_steps, 1)
            'entity_ids': (N,) -- original symbol strings
            'dates': (N,) -- prediction dates
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        enc = self.config.encoder_steps
        dec = self.config.decoder_steps
        total_steps = enc + dec

        all_past = []
        all_future = []
        all_static_cat = []
        all_targets = []
        all_raw_targets = []
        all_entity_ids = []
        all_dates = []

        for sym, sym_df in df.groupby(self._id_col):
            sym_df = sym_df.sort_index()
            n = len(sym_df)

            if n < total_steps:
                continue

            # Scale features
            scaled = sym_df.copy()
            for col, (mean, std) in self._scalers.items():
                if col in scaled.columns:
                    scaled[col] = (scaled[col] - mean) / std

            # Encode categoricals
            static_cat_vals = []
            for col in self._static_cat_cols:
                if col in sym_df.columns:
                    val = sym_df[col].iloc[0]
                    cat_map = self._cat_maps.get(col, {})
                    static_cat_vals.append(cat_map.get(val, 0))

            # Build column arrays
            observed_data = (
                scaled[self._observed_cols].values
                if self._observed_cols
                else np.zeros((n, 0))
            )
            known_data = (
                scaled[self._known_cols].values
                if self._known_cols
                else np.zeros((n, 0))
            )
            target_data = (
                scaled[self._target_cols].values
                if self._target_cols
                else np.zeros((n, 0))
            )

            if return_raw_targets:
                raw_target_data = sym_df[self._target_cols].values

            dates_arr = (
                sym_df.index
                if isinstance(sym_df.index, pd.DatetimeIndex)
                else pd.to_datetime(sym_df.index)
            )

            # Create sliding windows
            for i in range(n - total_steps + 1):
                # Past: observed + known + target for encoder steps
                past_obs = observed_data[i : i + enc]
                past_known = known_data[i : i + enc]
                past_target = target_data[i : i + enc]
                past = np.concatenate([past_obs, past_known, past_target], axis=1)

                # Future: known only for decoder steps
                future = known_data[i + enc : i + total_steps]

                # Target: target values for decoder steps
                target = target_data[i + enc : i + total_steps]

                all_past.append(past)
                all_future.append(future)
                all_static_cat.append(static_cat_vals)
                all_targets.append(target)
                all_entity_ids.append(sym)
                all_dates.append(dates_arr[i + enc])

                if return_raw_targets:
                    all_raw_targets.append(raw_target_data[i + enc : i + total_steps])

        result = {
            "past_inputs": np.array(all_past, dtype=np.float32),
            "future_inputs": np.array(all_future, dtype=np.float32),
            "static_cat": (
                np.array(all_static_cat, dtype=np.int64)
                if all_static_cat and all_static_cat[0]
                else None
            ),
            "targets": np.array(all_targets, dtype=np.float32),
            "entity_ids": np.array(all_entity_ids),
            "dates": np.array(all_dates),
        }

        if return_raw_targets:
            result["raw_targets"] = np.array(all_raw_targets, dtype=np.float32)

        return result

    def get_tft_config(self) -> dict:
        """Return config values to pass to ClassicTFTConfig.

        Returns
        -------
        dict with n_observed, n_known, n_static_cat, n_static_real,
        cat_cardinalities, encoder_steps, decoder_steps
        """
        cat_cards = []
        for col in self._static_cat_cols:
            cat_map = self._cat_maps.get(col, {})
            cat_cards.append(max(len(cat_map), 1))

        return {
            "n_observed": len(self._observed_cols),
            "n_known": len(self._known_cols),
            "n_static_cat": len(self._static_cat_cols),
            "n_static_real": len(self._static_real_cols),
            "cat_cardinalities": cat_cards,
            "encoder_steps": self.config.encoder_steps,
            "decoder_steps": self.config.decoder_steps,
            "output_size": len(self._target_cols),
        }

    @property
    def n_past_variables(self) -> int:
        """Number of past input variables (observed + known + target)."""
        return len(self._observed_cols) + len(self._known_cols) + len(self._target_cols)

    @property
    def n_future_variables(self) -> int:
        """Number of future input variables (known only)."""
        return len(self._known_cols)

    @property
    def target_scaler(self) -> tuple[float, float]:
        """Return (mean, std) for the target column for inverse transform."""
        target_col = self._target_cols[0] if self._target_cols else None
        if target_col and target_col in self._scalers:
            return self._scalers[target_col]
        return (0.0, 1.0)

    def inverse_transform_target(self, scaled_values: np.ndarray) -> np.ndarray:
        """Inverse-transform target values back to original scale."""
        mean, std = self.target_scaler
        return scaled_values * std + mean
