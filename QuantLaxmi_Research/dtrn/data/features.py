"""Online feature engine for DTRN."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DTRNConfig

logger = logging.getLogger(__name__)


class StreamingEWMA:
    """Exponentially weighted moving average with online updates."""

    def __init__(self, span: int):
        self.alpha = 2.0 / (span + 1)
        self.value = None
        self.n = 0

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        self.n += 1
        return self.value

    def reset(self):
        self.value = None
        self.n = 0


class StreamingZScore:
    """Online z-score using rolling window."""

    def __init__(self, window: int):
        self.window = window
        self.buffer = []

    def update(self, x: float) -> float:
        self.buffer.append(x)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        if len(self.buffer) < 2:
            return 0.0
        mean = np.mean(self.buffer)
        std = np.std(self.buffer, ddof=1)
        if std < 1e-10:
            return 0.0
        return (x - mean) / std

    def reset(self):
        self.buffer = []


class FeatureEngine:
    """Compute features from 1-min OHLCV bars.

    All computation is causal (online-compatible, no look-ahead).
    Features are standardized using streaming robust stats.
    """

    FEATURE_NAMES = [
        # Returns (5)
        "log_return", "lag_return_1", "lag_return_2", "lag_return_3", "lag_return_4",
        # Realized vol (4) - one per EWMA span
        "rvol_10", "rvol_30", "rvol_60", "rvol_120",
        # Volume (4)
        "log_volume", "volume_zscore", "volume_ratio", "oi_change",
        # Price (3)
        "bar_range", "intra_bar_return", "hl_ratio",
        # Trend/MR (4)
        "return_zscore", "ema_slope", "rsi", "momentum",
        # Time (5)
        "time_sin", "time_cos", "minutes_since_open", "session_half", "near_close",
        # Jump (1)
        "jump_flag",
    ]

    def __init__(self, config: DTRNConfig = None):
        if config is None:
            config = DTRNConfig()
        self.config = config
        self.n_features = len(self.FEATURE_NAMES)
        self._init_state()

    def _init_state(self):
        """Initialize all streaming state."""
        self.prev_close = None
        self.return_buffer = []  # for lagged returns

        # EWMA volatility estimators
        self.vol_ewmas = {span: StreamingEWMA(span) for span in self.config.ewma_spans}

        # Volume EWMA for ratio
        self.vol_ewma_60 = StreamingEWMA(60)

        # Z-score trackers
        self.return_zscore = StreamingZScore(self.config.zscore_window)
        self.volume_zscore = StreamingZScore(self.config.zscore_window)

        # EMA for slope (fast/slow)
        self.ema_fast = StreamingEWMA(10)
        self.ema_slow = StreamingEWMA(30)

        # RSI state
        self.rsi_gain_ema = StreamingEWMA(self.config.rsi_period)
        self.rsi_loss_ema = StreamingEWMA(self.config.rsi_period)

        # Momentum buffer
        self.close_buffer = []

        # OI tracking
        self.prev_oi = None

        # Bar counter
        self.bar_count = 0

    def reset(self):
        """Reset for new trading day."""
        self._init_state()

    @property
    def feature_names(self) -> list[str]:
        return self.FEATURE_NAMES.copy()

    def update(self, bar: dict) -> tuple[np.ndarray, np.ndarray]:
        """Process one bar, return (features, mask).

        bar: dict with keys: open, high, low, close, volume, oi, datetime
        returns: (d,) feature vector, (d,) binary mask (1=valid, 0=missing/warmup)
        """
        features = np.zeros(self.n_features, dtype=np.float32)
        mask = np.zeros(self.n_features, dtype=np.float32)

        close = float(bar["close"])
        open_ = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        volume = float(bar.get("volume", 0))
        oi = float(bar.get("oi", 0))
        dt = bar.get("datetime", None)

        # ── Returns ──
        if self.prev_close is not None and self.prev_close > 0:
            log_ret = np.log(close / self.prev_close)
            features[0] = log_ret
            mask[0] = 1.0

            self.return_buffer.append(log_ret)
            if len(self.return_buffer) > 10:
                self.return_buffer.pop(0)

            # Lagged returns
            for lag in range(1, 5):
                idx = 1 + (lag - 1)
                if len(self.return_buffer) > lag:
                    features[idx] = self.return_buffer[-(lag + 1)]
                    mask[idx] = 1.0
        else:
            log_ret = 0.0

        # ── Realized volatility (EWMA of r^2) ──
        r_sq = log_ret ** 2
        for i, span in enumerate(self.config.ewma_spans):
            val = self.vol_ewmas[span].update(r_sq)
            features[5 + i] = np.sqrt(val) if val > 0 else 0.0
            mask[5 + i] = 1.0 if self.vol_ewmas[span].n >= span // 2 else 0.0

        # ── Volume features ──
        if volume > 0:
            features[9] = np.log(volume + 1)
            mask[9] = 1.0

            features[10] = self.volume_zscore.update(volume)
            mask[10] = 1.0 if len(self.volume_zscore.buffer) >= 10 else 0.0

            ewma_vol = self.vol_ewma_60.update(volume)
            features[11] = volume / ewma_vol if ewma_vol > 0 else 1.0
            mask[11] = 1.0 if self.vol_ewma_60.n >= 10 else 0.0

        # OI change
        if self.prev_oi is not None and self.prev_oi > 0:
            features[12] = (oi - self.prev_oi) / self.prev_oi
            mask[12] = 1.0
        self.prev_oi = oi

        # ── Price features ──
        if close > 0:
            features[13] = (high - low) / close  # bar range
            mask[13] = 1.0

            features[14] = np.log(close / open_) if open_ > 0 else 0.0  # intra-bar
            mask[14] = 1.0

            features[15] = high / low if low > 0 else 1.0  # HL ratio
            mask[15] = 1.0

        # ── Trend / MR signals ──
        features[16] = self.return_zscore.update(log_ret)
        mask[16] = 1.0 if len(self.return_zscore.buffer) >= 10 else 0.0

        ema_f = self.ema_fast.update(close)
        ema_s = self.ema_slow.update(close)
        if ema_s > 0:
            features[17] = (ema_f - ema_s) / ema_s  # EMA slope
            mask[17] = 1.0 if self.ema_slow.n >= 10 else 0.0

        # RSI
        gain = max(log_ret, 0.0)
        loss = max(-log_ret, 0.0)
        avg_gain = self.rsi_gain_ema.update(gain)
        avg_loss = self.rsi_loss_ema.update(loss)
        if avg_loss > 1e-10:
            rs = avg_gain / avg_loss
            features[18] = 1.0 - 2.0 / (1.0 + rs)  # RSI scaled to [-1, 1]
        else:
            features[18] = 1.0 if avg_gain > 0 else 0.0
        mask[18] = 1.0 if self.rsi_gain_ema.n >= self.config.rsi_period else 0.0

        # Momentum
        self.close_buffer.append(close)
        if len(self.close_buffer) > 60:
            self.close_buffer.pop(0)
        if len(self.close_buffer) >= 20 and self.close_buffer[-20] > 0:
            features[19] = np.log(close / self.close_buffer[-20])
            mask[19] = 1.0

        # ── Time features ──
        if dt is not None:
            if hasattr(dt, 'hour'):
                minutes_in_day = dt.hour * 60 + dt.minute
                market_open = 9 * 60 + 15   # 9:15
                market_close = 15 * 60 + 30  # 15:30
                total_minutes = market_close - market_open
                progress = (minutes_in_day - market_open) / total_minutes
                progress = np.clip(progress, 0, 1)

                features[20] = np.sin(2 * np.pi * progress)
                features[21] = np.cos(2 * np.pi * progress)
                features[22] = minutes_in_day - market_open
                features[23] = 1.0 if progress > 0.5 else 0.0  # PM session
                features[24] = 1.0 if progress > 0.96 else 0.0  # last ~15 min
                mask[20:25] = 1.0

        # ── Jump flag ──
        if mask[5]:  # need rvol_10
            threshold = 3.0 * features[5]  # 3 * rvol_10
            features[25] = 1.0 if abs(log_ret) > max(threshold, 1e-6) else 0.0
            mask[25] = 1.0

        self.prev_close = close
        self.bar_count += 1

        # Safety: replace any NaN/Inf with 0 and zero the mask for those entries
        bad = ~np.isfinite(features)
        if bad.any():
            features[bad] = 0.0
            mask[bad] = 0.0

        return features, mask

    def compute_batch(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Compute features for entire day (batch mode, but still causal).

        df: DataFrame with datetime index, columns: open, high, low, close, volume, oi
        returns: (T, d) feature array, (T, d) mask array
        """
        self.reset()
        T = len(df)
        features = np.zeros((T, self.n_features), dtype=np.float32)
        masks = np.zeros((T, self.n_features), dtype=np.float32)

        for i, (dt, row) in enumerate(df.iterrows()):
            bar = {
                "datetime": dt,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volume", 0),
                "oi": row.get("oi", 0),
            }
            features[i], masks[i] = self.update(bar)

        return features, masks

    def compute_multi_day(
        self,
        daily_dfs: dict[str, pd.DataFrame],
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """Compute features across multiple days with daily reset.

        daily_dfs: {date_str: DataFrame} sorted by date
        returns: (total_T, d) features, (total_T, d) masks, list of day boundaries
        """
        all_features = []
        all_masks = []
        boundaries = []  # (start_idx, end_idx, date_str)
        offset = 0

        for date_str in sorted(daily_dfs.keys()):
            df = daily_dfs[date_str]
            feat, mask = self.compute_batch(df)
            all_features.append(feat)
            all_masks.append(mask)
            boundaries.append((offset, offset + len(df), date_str))
            offset += len(df)

        if not all_features:
            return np.empty((0, self.n_features)), np.empty((0, self.n_features)), []

        return np.concatenate(all_features), np.concatenate(all_masks), boundaries
