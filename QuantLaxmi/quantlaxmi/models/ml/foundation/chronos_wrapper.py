"""Amazon Chronos integration for financial time-series forecasting.

Wraps the Chronos pretrained probabilistic time-series model for use in the
QuantLaxmi trading system.  Supports both the original T5-based ``ChronosPipeline``
and the newer Chronos-2 (Bolt) architecture via ``Chronos2Pipeline``.

Key capabilities
-----------------
- Zero-shot probabilistic forecasting (median, quantiles, samples)
- Fine-tuning on domain-specific data (LoRA or full, Chronos-2 only)
- Walk-forward evaluation with purge gaps
- Trading signal generation from forecast distributions
- Multi-asset batch prediction
- Embedding extraction for downstream models

Usage
-----
    from quantlaxmi.models.ml.foundation import ChronosForecaster

    fc = ChronosForecaster()                      # loads chronos-bolt-small
    out = fc.predict(price_series[-512:])          # probabilistic forecast
    sig = fc.generate_trading_signal(price_series) # → float in [-1, 1]

References
----------
- Ansari et al. (2024)  "Chronos: Learning the Language of Time Series"
- Abdul Fatir et al. (2024) "Chronos-Bolt: Efficient Pretrained Models for
  Probabilistic Time Series Forecasting"
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------

try:
    import torch

    _HAS_TORCH = True
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _HAS_TORCH = False
    _DEVICE = "cpu"

try:
    from chronos import (
        BaseChronosPipeline,
        Chronos2Pipeline,
        ChronosPipeline,
    )

    HAS_CHRONOS = True
    logger.info("chronos-forecasting available")
except ImportError:
    HAS_CHRONOS = False
    BaseChronosPipeline = None  # type: ignore[assignment,misc]
    Chronos2Pipeline = None  # type: ignore[assignment,misc]
    ChronosPipeline = None  # type: ignore[assignment,misc]
    logger.warning(
        "chronos-forecasting not installed; ChronosForecaster will be non-functional"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ChronosConfig:
    """Configuration for :class:`ChronosForecaster`.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Chronos-2 (Bolt) models are
        recommended for speed: ``amazon/chronos-bolt-{tiny,mini,small,base}``.
        Original T5 models: ``amazon/chronos-t5-{tiny,mini,small,base,large}``.
    device : str
        ``"cuda"`` or ``"cpu"``.  Auto-detected when omitted.
    prediction_length : int
        Default forecast horizon in time-steps (trading days).
    num_samples : int
        Number of forecast samples (T5 models only; Bolt returns quantiles
        directly).
    quantile_levels : list[float]
        Quantile levels to extract from the forecast distribution.
    context_length : int | None
        Maximum number of historical steps fed to the model.  ``None`` uses
        the model default (512 for T5, 2048 for Bolt).
    torch_dtype : str
        ``"float32"`` or ``"bfloat16"``.  bfloat16 halves memory on T4 GPU.
    """

    model_name: str = "amazon/chronos-t5-small"
    device: str = field(default_factory=lambda: _DEVICE)
    prediction_length: int = 5
    num_samples: int = 100
    quantile_levels: list[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9]
    )
    context_length: Optional[int] = None
    torch_dtype: str = "float32"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ChronosForecaster:
    """Wrapper around Amazon Chronos for financial time-series forecasting.

    Chronos is a pretrained probabilistic time-series model trained on a
    large corpus of open time-series data.  This wrapper adapts it for
    trading-signal generation in the QuantLaxmi system.

    The class transparently handles both Chronos-1 (T5, sample-based) and
    Chronos-2 (Bolt, quantile-based) model variants.

    Parameters
    ----------
    model_name : str
        HuggingFace model id, e.g. ``"amazon/chronos-t5-small"``.
    device : str
        ``"cuda"`` or ``"cpu"``.
    prediction_length : int
        Default forecast horizon (trading days).
    num_samples : int
        Number of samples for T5 models.
    config : ChronosConfig, optional
        Full configuration object.  Overrides individual kwargs.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        device: str | None = None,
        prediction_length: int = 5,
        num_samples: int = 100,
        config: ChronosConfig | None = None,
    ) -> None:
        if config is not None:
            self.cfg = config
        else:
            self.cfg = ChronosConfig(
                model_name=model_name,
                device=device or _DEVICE,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )

        self._pipeline: Any = None  # lazy-loaded
        self._is_bolt: bool = "bolt" in self.cfg.model_name or "chronos-2" in self.cfg.model_name
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the model on first use (avoids heavy import at module level)."""
        if self._loaded:
            return

        if not HAS_CHRONOS:
            raise ImportError(
                "chronos-forecasting is not installed. "
                "Install with: pip install chronos-forecasting"
            )
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for Chronos models.")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.cfg.torch_dtype, torch.float32)

        logger.info(
            "Loading Chronos model=%s device=%s dtype=%s",
            self.cfg.model_name,
            self.cfg.device,
            self.cfg.torch_dtype,
        )

        if self._is_bolt:
            try:
                self._pipeline = Chronos2Pipeline.from_pretrained(
                    self.cfg.model_name,
                    device_map=self.cfg.device,
                    dtype=torch_dtype,
                )
            except AttributeError as exc:
                # chronos-bolt models may require a newer version of
                # chronos-forecasting than what is installed.  Fall back
                # to equivalent T5 model if available.
                fallback = self.cfg.model_name.replace("-bolt-", "-t5-")
                logger.warning(
                    "Bolt model %s failed (%s). Falling back to %s",
                    self.cfg.model_name, exc, fallback,
                )
                self._is_bolt = False
                self._pipeline = ChronosPipeline.from_pretrained(
                    fallback,
                    device_map=self.cfg.device,
                    torch_dtype=torch_dtype,
                )
        else:
            self._pipeline = ChronosPipeline.from_pretrained(
                self.cfg.model_name,
                device_map=self.cfg.device,
                torch_dtype=torch_dtype,
            )

        self._loaded = True
        logger.info("Chronos model loaded successfully")

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        context: np.ndarray,
        prediction_length: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate probabilistic forecasts from a univariate context series.

        Parameters
        ----------
        context : np.ndarray
            1-D array of historical values (e.g. daily close prices).  The
            model uses the most recent ``context_length`` values.
        prediction_length : int, optional
            Override the default forecast horizon.

        Returns
        -------
        dict with keys:
            ``median``        – shape ``(H,)``
            ``mean``          – shape ``(H,)``
            ``quantile_10``   – shape ``(H,)``
            ``quantile_25``   – shape ``(H,)``
            ``quantile_75``   – shape ``(H,)``
            ``quantile_90``   – shape ``(H,)``
            ``samples``       – shape ``(num_samples, H)`` (T5) or
                                ``(1, H)`` (Bolt, single point estimate as fallback)
        where *H* is the prediction length.
        """
        self._ensure_loaded()

        pred_len = prediction_length or self.cfg.prediction_length
        context_clean = np.asarray(context, dtype=np.float64).ravel()

        # Trim to context window if specified
        if self.cfg.context_length is not None:
            context_clean = context_clean[-self.cfg.context_length:]

        # Build input tensor
        ctx_tensor = torch.tensor(context_clean, dtype=torch.float32).unsqueeze(0)

        if self._is_bolt:
            # Chronos-2 (Bolt) returns quantiles directly
            quantiles, mean_pred = self._pipeline.predict_quantiles(
                ctx_tensor,
                prediction_length=pred_len,
                quantile_levels=self.cfg.quantile_levels,
            )
            # quantiles: (batch, H, num_quantiles), mean_pred: (batch, H)
            q = quantiles[0].cpu().numpy()  # (H, num_quantiles)
            m = mean_pred[0].cpu().numpy()  # (H,)

            # Map quantile levels to named outputs
            ql = self.cfg.quantile_levels
            result: dict[str, np.ndarray] = {
                "mean": m,
                "median": q[:, ql.index(0.5)] if 0.5 in ql else m,
            }
            # Fill standard quantile keys
            _q_map = {0.1: "quantile_10", 0.25: "quantile_25",
                      0.75: "quantile_75", 0.9: "quantile_90"}
            for level, key in _q_map.items():
                if level in ql:
                    result[key] = q[:, ql.index(level)]
                else:
                    result[key] = m  # fallback
            result["samples"] = m[np.newaxis, :]  # (1, H) for API compat

        else:
            # Chronos-1 (T5) returns samples
            samples = self._pipeline.predict(
                ctx_tensor,
                prediction_length=pred_len,
                num_samples=self.cfg.num_samples,
            )
            # samples: (batch, num_samples, H)
            s = samples[0].cpu().numpy()  # (num_samples, H)

            result = {
                "median": np.median(s, axis=0),
                "mean": np.mean(s, axis=0),
                "quantile_10": np.quantile(s, 0.10, axis=0),
                "quantile_25": np.quantile(s, 0.25, axis=0),
                "quantile_75": np.quantile(s, 0.75, axis=0),
                "quantile_90": np.quantile(s, 0.90, axis=0),
                "samples": s,
            }

        return result

    # ------------------------------------------------------------------
    # Multi-asset prediction
    # ------------------------------------------------------------------

    def predict_multi_asset(
        self,
        asset_series: dict[str, np.ndarray],
        prediction_length: int | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Predict for multiple assets (NIFTY, BANKNIFTY, etc.).

        Parameters
        ----------
        asset_series : dict[str, np.ndarray]
            Mapping of asset name to 1-D price array.
        prediction_length : int, optional
            Override forecast horizon.

        Returns
        -------
        dict mapping asset name to the same dict structure as :meth:`predict`.
        """
        self._ensure_loaded()

        pred_len = prediction_length or self.cfg.prediction_length
        results: dict[str, dict[str, np.ndarray]] = {}

        # Build batch of tensors
        tensors = []
        names = []
        for name, series in asset_series.items():
            s = np.asarray(series, dtype=np.float64).ravel()
            if self.cfg.context_length is not None:
                s = s[-self.cfg.context_length:]
            tensors.append(torch.tensor(s, dtype=torch.float32))
            names.append(name)

        if self._is_bolt:
            quantiles, mean_pred = self._pipeline.predict_quantiles(
                tensors,
                prediction_length=pred_len,
                quantile_levels=self.cfg.quantile_levels,
            )
            # quantiles: (batch, H, nq), mean_pred: (batch, H)
            ql = self.cfg.quantile_levels
            _q_map = {0.1: "quantile_10", 0.25: "quantile_25",
                      0.75: "quantile_75", 0.9: "quantile_90"}
            for i, name in enumerate(names):
                q = quantiles[i].cpu().numpy()
                m = mean_pred[i].cpu().numpy()
                r: dict[str, np.ndarray] = {
                    "mean": m,
                    "median": q[:, ql.index(0.5)] if 0.5 in ql else m,
                }
                for level, key in _q_map.items():
                    if level in ql:
                        r[key] = q[:, ql.index(level)]
                    else:
                        r[key] = m
                r["samples"] = m[np.newaxis, :]
                results[name] = r
        else:
            # T5: predict individually (list-of-tensors input)
            samples = self._pipeline.predict(
                tensors,
                prediction_length=pred_len,
                num_samples=self.cfg.num_samples,
            )
            # samples: (batch, num_samples, H)
            for i, name in enumerate(names):
                s = samples[i].cpu().numpy()
                results[name] = {
                    "median": np.median(s, axis=0),
                    "mean": np.mean(s, axis=0),
                    "quantile_10": np.quantile(s, 0.10, axis=0),
                    "quantile_25": np.quantile(s, 0.25, axis=0),
                    "quantile_75": np.quantile(s, 0.75, axis=0),
                    "quantile_90": np.quantile(s, 0.90, axis=0),
                    "samples": s,
                }

        return results

    # ------------------------------------------------------------------
    # Trading signal generation
    # ------------------------------------------------------------------

    def generate_trading_signal(
        self,
        context: np.ndarray,
        threshold: float = 0.6,
        prediction_length: int | None = None,
    ) -> float:
        """Convert a probabilistic forecast to a trading signal in [-1, 1].

        The signal is computed as::

            expected_return = (median_forecast - last_price) / last_price
            raw_signal = 2 * P(return > 0) - 1    # in [-1, 1]

        The signal is then thresholded: if ``|raw_signal| < threshold`` the
        output is ``0.0`` (flat / no conviction).

        Additionally, a quantile safety check is applied:
        - For long signals:  require ``quantile_10 > -safety_pct``
        - For short signals: require ``quantile_90 < +safety_pct``

        Parameters
        ----------
        context : np.ndarray
            Historical price series.
        threshold : float
            Minimum |signal| to emit a position.  Default 0.6.
        prediction_length : int, optional
            Override forecast horizon.

        Returns
        -------
        float
            Signal in [-1, 1].  Positive → long, negative → short, 0 → flat.
        """
        self._ensure_loaded()

        pred = self.predict(context, prediction_length=prediction_length)
        last_price = float(context[-1])

        if last_price == 0.0:
            return 0.0

        # Compute probability of positive return using quantile crossing
        median_return = (pred["median"][0] - last_price) / last_price
        q10_return = (pred["quantile_10"][0] - last_price) / last_price
        q25_return = (pred["quantile_25"][0] - last_price) / last_price
        q75_return = (pred["quantile_75"][0] - last_price) / last_price
        q90_return = (pred["quantile_90"][0] - last_price) / last_price

        # Estimate P(return > 0) from quantile interpolation
        # Linear interpolation across quantile levels
        quantile_returns = np.array([q10_return, q25_return, median_return,
                                     q75_return, q90_return])
        quantile_levels = np.array([0.10, 0.25, 0.50, 0.75, 0.90])

        # P(return > 0) ≈ 1 - interp(0, quantile_returns, quantile_levels)
        if quantile_returns[-1] <= 0:
            p_positive = 0.0
        elif quantile_returns[0] >= 0:
            p_positive = 1.0
        else:
            # Interpolate: find the quantile level where return = 0
            p_negative = float(np.interp(0.0, quantile_returns, quantile_levels))
            p_positive = 1.0 - p_negative

        raw_signal = 2.0 * p_positive - 1.0  # in [-1, 1]

        # Threshold: insufficient conviction → flat
        if abs(raw_signal) < threshold:
            return 0.0

        # Safety check via tail quantiles
        safety_pct = 0.03  # 3% worst-case tolerance
        if raw_signal > 0 and q10_return < -safety_pct:
            # Long signal but left tail too heavy — scale down
            raw_signal *= max(0.0, 1.0 + q10_return / safety_pct)
        elif raw_signal < 0 and q90_return > safety_pct:
            # Short signal but right tail too heavy — scale down
            raw_signal *= max(0.0, 1.0 - q90_return / safety_pct)

        # Clip to [-1, 1]
        return float(np.clip(raw_signal, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Fine-tuning (Chronos-2 / Bolt only)
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        train_data: list[np.ndarray],
        epochs: int = 5,
        lr: float = 1e-4,
        finetune_mode: str = "lora",
        output_dir: str | Path | None = None,
        prediction_length: int | None = None,
    ) -> dict[str, Any]:
        """Fine-tune the model on domain-specific financial data.

        Only Chronos-2 (Bolt) models support ``.fit()``.  For T5 models,
        this method raises ``NotImplementedError`` (use the HuggingFace
        training scripts directly).

        Parameters
        ----------
        train_data : list[np.ndarray]
            List of 1-D time-series arrays (e.g. daily close prices for
            different assets or rolling windows).
        epochs : int
            Number of training steps expressed as approximate epochs.
            Internally converted to ``num_steps``.
        lr : float
            Learning rate.
        finetune_mode : str
            ``"lora"`` (parameter-efficient) or ``"full"``.
        output_dir : str or Path, optional
            Where to save the fine-tuned checkpoint.
        prediction_length : int, optional
            Forecast horizon used during training.

        Returns
        -------
        dict
            Summary with keys: ``num_series``, ``num_steps``, ``mode``,
            ``output_dir``.
        """
        self._ensure_loaded()

        if not self._is_bolt:
            raise NotImplementedError(
                "Fine-tuning via .fit() is only supported for Chronos-2 (Bolt) "
                "models.  For T5 models, use the HuggingFace training scripts "
                "or the chronos-forecasting CLI."
            )

        pred_len = prediction_length or self.cfg.prediction_length
        tensors = [
            torch.tensor(np.asarray(s, dtype=np.float64).ravel(), dtype=torch.float32)
            for s in train_data
        ]

        # Approximate epochs → steps
        avg_len = np.mean([len(t) for t in tensors])
        steps_per_epoch = max(1, int(len(tensors) * avg_len / 256))
        num_steps = max(1, epochs * steps_per_epoch)

        out = Path(output_dir) if output_dir else Path("checkpoints/chronos_finetuned")
        out.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Fine-tuning Chronos-2: %d series, %d steps, mode=%s, lr=%g",
            len(tensors), num_steps, finetune_mode, lr,
        )

        self._pipeline = self._pipeline.fit(
            inputs=tensors,
            prediction_length=pred_len,
            finetune_mode=finetune_mode,
            learning_rate=lr,
            num_steps=num_steps,
            output_dir=str(out),
        )

        return {
            "num_series": len(tensors),
            "num_steps": num_steps,
            "mode": finetune_mode,
            "output_dir": str(out),
        }

    # ------------------------------------------------------------------
    # Walk-forward evaluation
    # ------------------------------------------------------------------

    def walk_forward_evaluate(
        self,
        data: np.ndarray,
        train_size: int = 200,
        test_size: int = 42,
        step_size: int = 21,
        purge_gap: int = 5,
        prediction_length: int = 1,
    ) -> dict[str, Any]:
        """Walk-forward evaluation with proper purge gaps.

        For each fold:
        1. Use ``data[:train_end]`` as context (no actual training for
           zero-shot; fine-tune for each fold if desired).
        2. Skip ``purge_gap`` days.
        3. Predict the next ``test_size`` days one step at a time.
        4. Collect forecast-vs-actual errors and directional accuracy.

        Parameters
        ----------
        data : np.ndarray
            Full 1-D series (e.g. daily close prices).
        train_size : int
            Number of initial context days.
        test_size : int
            Number of OOS evaluation days per fold.
        step_size : int
            Step between fold starts.
        purge_gap : int
            Gap between context end and test start to prevent look-ahead.
        prediction_length : int
            Horizon for each point prediction (default 1 = next-day).

        Returns
        -------
        dict with keys:
            ``folds``           – list of per-fold dicts
            ``overall_mae``     – mean absolute error across all folds
            ``overall_rmse``    – root mean squared error
            ``directional_acc`` – fraction of correct direction calls
            ``mean_sharpe``     – average Sharpe of forecast-based returns
            ``num_folds``       – number of evaluation folds
        """
        self._ensure_loaded()

        data = np.asarray(data, dtype=np.float64).ravel()
        n = len(data)
        folds: list[dict[str, Any]] = []

        pos = train_size
        while pos + purge_gap + test_size <= n:
            ctx_end = pos
            test_start = pos + purge_gap
            test_end = test_start + test_size

            context = data[:ctx_end]
            actuals = data[test_start:test_end]

            # Rolling one-step predictions
            predictions = []
            running_ctx = context.copy()
            for t in range(test_size):
                if test_start + t >= n:
                    break
                pred = self.predict(running_ctx, prediction_length=prediction_length)
                predictions.append(float(pred["median"][0]))
                # Extend context with actual value (no look-ahead: we only
                # use the actual *after* we've made the prediction)
                running_ctx = np.append(running_ctx, actuals[t])

            predictions_arr = np.array(predictions)
            actuals_arr = actuals[: len(predictions)]

            # Metrics
            errors = predictions_arr - actuals_arr
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors ** 2)))

            # Directional accuracy (predict return direction)
            if len(actuals_arr) > 1:
                actual_returns = np.diff(actuals_arr)
                pred_returns = predictions_arr[1:] - actuals_arr[:-1]
                dir_correct = np.sum(np.sign(pred_returns) == np.sign(actual_returns))
                dir_acc = float(dir_correct / len(actual_returns))

                # Sharpe of forecast-aligned returns
                strategy_returns = np.sign(pred_returns) * actual_returns / actuals_arr[:-1]
                sr_mean = np.mean(strategy_returns)
                sr_std = np.std(strategy_returns, ddof=1) if len(strategy_returns) > 1 else 1e-8
                sharpe = float(sr_mean / max(sr_std, 1e-8) * math.sqrt(252))
            else:
                dir_acc = float("nan")
                sharpe = float("nan")

            folds.append({
                "train_end": ctx_end,
                "test_start": test_start,
                "test_end": test_end,
                "mae": mae,
                "rmse": rmse,
                "directional_acc": dir_acc,
                "sharpe": sharpe,
                "n_predictions": len(predictions),
            })

            pos += step_size

        if not folds:
            return {
                "folds": [],
                "overall_mae": float("nan"),
                "overall_rmse": float("nan"),
                "directional_acc": float("nan"),
                "mean_sharpe": float("nan"),
                "num_folds": 0,
            }

        return {
            "folds": folds,
            "overall_mae": float(np.mean([f["mae"] for f in folds])),
            "overall_rmse": float(np.mean([f["rmse"] for f in folds])),
            "directional_acc": float(np.nanmean([f["directional_acc"] for f in folds])),
            "mean_sharpe": float(np.nanmean([f["sharpe"] for f in folds])),
            "num_folds": len(folds),
        }

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def embed(self, context: np.ndarray) -> np.ndarray:
        """Extract Chronos embeddings for downstream use (e.g. clustering).

        Only available for models that expose an ``embed`` method (T5, Bolt).

        Parameters
        ----------
        context : np.ndarray
            1-D historical price series.

        Returns
        -------
        np.ndarray
            Embedding array of shape ``(seq_len, hidden_dim)``.
        """
        self._ensure_loaded()

        ctx = np.asarray(context, dtype=np.float64).ravel()
        if self.cfg.context_length is not None:
            ctx = ctx[-self.cfg.context_length:]
        ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)

        embedding, _ = self._pipeline.embed(ctx_tensor)
        return embedding[0].cpu().numpy()

    # ------------------------------------------------------------------
    # Feature-panel conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def dataframe_to_panel(
        df: pd.DataFrame,
        target_col: str = "close",
        feature_cols: list[str] | None = None,
        top_n_features: int = 10,
    ) -> dict[str, np.ndarray]:
        """Convert a MegaFeatureBuilder DataFrame to Chronos-compatible panel.

        Chronos expects univariate series.  This helper extracts the target
        column and optionally the top-N features (by variance) as separate
        "channels" that can be independently forecast.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``MegaFeatureBuilder.build()``.
        target_col : str
            Column to use as the primary series.
        feature_cols : list[str], optional
            Specific feature columns.  If ``None``, auto-selects by variance.
        top_n_features : int
            Number of features to include when auto-selecting.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of series name to 1-D array, including the target and
            selected feature channels.
        """
        panel: dict[str, np.ndarray] = {}

        if target_col in df.columns:
            panel["target"] = df[target_col].values.astype(np.float64)

        if feature_cols is not None:
            cols = [c for c in feature_cols if c in df.columns]
        else:
            # Auto-select by variance (most informative)
            numeric = df.select_dtypes(include=[np.number])
            variances = numeric.var().dropna().sort_values(ascending=False)
            cols = [c for c in variances.index[:top_n_features]
                    if c != target_col and c in df.columns]

        for col in cols:
            vals = df[col].values.astype(np.float64)
            # Skip columns with too many NaNs (>50%)
            if np.isnan(vals).sum() / len(vals) > 0.5:
                continue
            # Forward-fill NaNs for Chronos compatibility
            s = pd.Series(vals).ffill().bfill()
            panel[col] = s.values

        return panel

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"ChronosForecaster(model={self.cfg.model_name!r}, "
            f"device={self.cfg.device!r}, "
            f"prediction_length={self.cfg.prediction_length}, "
            f"status={status})"
        )
