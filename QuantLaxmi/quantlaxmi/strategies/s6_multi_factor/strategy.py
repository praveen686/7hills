"""S6: Multi-Factor — Walk-forward XGBoost on NSE index returns.

Loads nse_index_close for 120+ NSE indices, computes multi-window
return features, and predicts the target index's 5-day forward
direction using XGBoost retrained on a rolling 60-day window with
a 5-day purge gap to prevent look-ahead bias.

Instruments: NIFTY index futures (or configurable target).
Holding period: 5 days.
"""

from __future__ import annotations

import logging
import math
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.strategies.base import BaseStrategy
from quantlaxmi.strategies.protocol import Signal

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_TARGET_INDEX = "Nifty 50"
DEFAULT_SYMBOL = "NIFTY"
DEFAULT_TRAIN_WINDOW = 60
DEFAULT_PURGE_GAP = 5
DEFAULT_FORWARD_HORIZON = 5
FEATURE_WINDOWS = (5, 10, 20)
RETRAIN_INTERVAL_DAYS = 5  # retrain every 5 trading days


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


class S6MultiFactorStrategy(BaseStrategy):
    """S6: Multi-Factor — XGBoost walk-forward strategy.

    Parameters
    ----------
    target_index : str
        NSE index name to predict (e.g. "Nifty 50"). Default: "Nifty 50".
    symbol : str
        Trading symbol for signal output. Default: "NIFTY".
    train_window : int
        Rolling training window in trading days. Default: 60.
    purge_gap : int
        Rows to purge between train and test to avoid target leakage.
        Must be >= forward_horizon. Default: 5.
    forward_horizon : int
        Forward return horizon in days. Default: 5.
    feature_windows : tuple[int, ...]
        Return windows for feature construction. Default: (5, 10, 20).
    """

    def __init__(
        self,
        target_index: str = DEFAULT_TARGET_INDEX,
        symbol: str = DEFAULT_SYMBOL,
        train_window: int = DEFAULT_TRAIN_WINDOW,
        purge_gap: int = DEFAULT_PURGE_GAP,
        forward_horizon: int = DEFAULT_FORWARD_HORIZON,
        feature_windows: tuple[int, ...] = FEATURE_WINDOWS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._target_index = target_index
        self._symbol = symbol
        self._train_window = train_window
        self._purge_gap = purge_gap
        self._forward_horizon = forward_horizon
        self._feature_windows = feature_windows
        self._model = None
        self._feature_names: list[str] = []

    @property
    def strategy_id(self) -> str:
        return "s6_multi_factor"

    def warmup_days(self) -> int:
        return self._train_window + self._purge_gap  # 65 days

    def _build_feature_matrix(
        self, store: MarketDataStore, start: date, end: date
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Build feature matrix from NSE index close data.

        Features: {window}-day return for each index.
        Target: forward_horizon-day return for target_index.
        Returns (X, y, feature_names). X/y may be empty on failure.
        """
        df = store.sql(
            "SELECT * FROM nse_index_close WHERE date >= ? AND date <= ? "
            "ORDER BY date",
            [start.isoformat(), end.isoformat()],
        )
        if df is None or df.empty:
            return pd.DataFrame(), pd.Series(dtype=float), []

        # Identify columns
        date_col = name_col = close_col = None
        for col in df.columns:
            if col in ("Index Date", "date"):
                date_col = col
            elif col == "Index Name":
                name_col = col
            elif col == "Closing Index Value":
                close_col = col
            elif "index" in col.lower() and "name" in col.lower() and name_col is None:
                name_col = col
            elif "clos" in col.lower() and "index" in col.lower() and close_col is None:
                close_col = col

        if date_col is None or name_col is None or close_col is None:
            logger.debug("Cannot identify nse_index_close columns: %s", list(df.columns))
            return pd.DataFrame(), pd.Series(dtype=float), []

        # Pivot to wide format: dates x indices
        df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
        pivot = df.pivot_table(index=date_col, columns=name_col, values=close_col)
        pivot = pivot.sort_index()

        # Drop indices with too many NaNs
        min_obs = len(pivot) * 0.5
        pivot = pivot.dropna(axis=1, thresh=int(min_obs))

        # Resolve target index
        target = self._target_index
        if target not in pivot.columns:
            matches = [c for c in pivot.columns if target.lower() in c.lower()]
            if matches:
                target = matches[0]
            else:
                logger.debug("Target '%s' not found in index data", self._target_index)
                return pd.DataFrame(), pd.Series(dtype=float), []

        # Compute return features over multiple windows
        feature_dfs = []
        feature_names: list[str] = []
        for w in self._feature_windows:
            rets = pivot.pct_change(w)
            rets.columns = [f"{c}_ret{w}d" for c in rets.columns]
            feature_dfs.append(rets)
            feature_names.extend(rets.columns.tolist())

        features = pd.concat(feature_dfs, axis=1)

        # Target: forward return (shifted backward — uses future prices)
        # At row i, target = (price[i+h] - price[i]) / price[i]
        y = pivot[target].pct_change(self._forward_horizon).shift(
            -self._forward_horizon
        )
        y.name = "target"

        combined = pd.concat([features, y], axis=1).dropna()
        X = combined[feature_names]
        y_out = combined["target"]

        return X, y_out, feature_names

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """Train an XGBoost regressor on the given data."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

        # Replace inf/-inf with NaN, then forward-fill + zero-fill residuals
        X_clean = X_train.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        y_clean = y_train.copy()
        y_clean = y_clean.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.5,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_clean, y_clean)
        return model

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        """Build feature matrix, retrain if needed, predict direction.

        Model is retrained every RETRAIN_INTERVAL_DAYS trading days.
        State persists the last retrain date to avoid redundant work.
        """
        # Determine date range for feature matrix
        lookback_days = self._train_window + max(self._feature_windows) + 30
        start = d - timedelta(days=int(lookback_days * 1.5))

        # Build feature matrix up to d
        X, y, feature_names = self._build_feature_matrix(store, start, d)
        if X.empty or len(X) < self._train_window + self._purge_gap:
            return []

        self._feature_names = feature_names

        # The last row with a valid target is at index -(forward_horizon+1)
        # because target uses shift(-forward_horizon). Rows at the end
        # will have NaN targets. We need the FEATURE row for date d
        # (which has NaN target) to make our prediction.

        # Re-build without dropna to get the latest feature row
        full_features = X  # X already had dropna applied
        # We need the feature row closest to d. Since X is aligned with
        # the pivot index (dates), the last row is the most recent.

        # Check if we need to retrain
        last_retrain = self.get_state("last_retrain_date")
        needs_retrain = True
        if last_retrain is not None and self._model is not None:
            last_dt = date.fromisoformat(last_retrain)
            if (d - last_dt).days < RETRAIN_INTERVAL_DAYS:
                needs_retrain = False

        if needs_retrain:
            # Train on all available data EXCEPT the last purge_gap rows
            # to avoid target leakage at the boundary
            train_end = len(X) - self._purge_gap
            if train_end < self._train_window:
                return []

            train_start = max(0, train_end - self._train_window)
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]

            if len(X_train) < 20:
                return []

            try:
                self._model = self._train_model(X_train, y_train)
                self.set_state("last_retrain_date", d.isoformat())
                logger.debug(
                    "S6 retrained on %d rows (%s to row %d)",
                    len(X_train), d, train_end,
                )
            except Exception as e:
                logger.debug("S6 training failed: %s", e)
                return []

        if self._model is None:
            return []

        # Predict on the most recent feature row (clean inf values)
        latest_features = X.iloc[[-1]].replace([np.inf, -np.inf], np.nan).ffill(axis=1).fillna(0.0)
        try:
            prediction = float(self._model.predict(latest_features)[0])
        except Exception as e:
            logger.debug("S6 prediction failed: %s", e)
            return []

        # Map prediction to signal
        if prediction > 0:
            direction = "long"
        elif prediction < 0:
            direction = "short"
        else:
            direction = "flat"

        # Conviction from sigmoid of raw prediction (scaled)
        # prediction is a return in [-0.1, 0.1] range typically
        conviction = abs(_sigmoid(prediction * 100) - 0.5) * 2.0
        conviction = min(1.0, max(0.0, conviction))

        if direction == "flat":
            conviction = 0.0

        # Check for existing position state
        pos_key = f"pos_{self._symbol}"
        pos = self.get_state(pos_key)

        if direction == "flat" or (
            direction in ("long", "short") and prediction == 0
        ):
            # Exit if we have a position
            if pos is not None:
                self.set_state(pos_key, None)
                return [Signal(
                    strategy_id=self.strategy_id,
                    symbol=self._symbol,
                    direction="flat",
                    conviction=0.0,
                    instrument_type="FUT",
                    ttl_bars=0,
                    metadata={"prediction": round(prediction, 6)},
                )]
            return []

        # Check for hold period expiry
        if pos is not None:
            entry_date = date.fromisoformat(pos["entry_date"])
            days_held = (d - entry_date).days
            if days_held >= self._forward_horizon:
                # Exit and potentially re-enter
                self.set_state(pos_key, None)
                pos = None

        # Already in same direction — hold
        if pos is not None and pos["direction"] == direction:
            return []

        # Direction flip — exit first
        if pos is not None and pos["direction"] != direction:
            self.set_state(pos_key, None)
            return [Signal(
                strategy_id=self.strategy_id,
                symbol=self._symbol,
                direction="flat",
                conviction=0.0,
                instrument_type="FUT",
                ttl_bars=0,
                metadata={"exit_reason": "direction_flip", "prediction": round(prediction, 6)},
            )]

        # New entry
        self.set_state(pos_key, {
            "entry_date": d.isoformat(),
            "direction": direction,
            "prediction": round(prediction, 6),
        })

        metadata = {
            "prediction": round(prediction, 6),
            "conviction_raw": round(conviction, 4),
            "n_features": len(self._feature_names),
        }

        return [Signal(
            strategy_id=self.strategy_id,
            symbol=self._symbol,
            direction=direction,
            conviction=conviction,
            instrument_type="FUT",
            ttl_bars=self._forward_horizon,
            metadata=metadata,
        )]


def create_strategy() -> S6MultiFactorStrategy:
    """Factory for registry auto-discovery."""
    return S6MultiFactorStrategy()
