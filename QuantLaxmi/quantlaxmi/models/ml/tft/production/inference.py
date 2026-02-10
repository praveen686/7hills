"""TFT Inference Pipeline — production prediction from checkpoint.

Loads a saved checkpoint, reconstructs the model, and serves predictions
without needing any training infrastructure. All normalization uses saved
stats from training (no data leakage).

Usage
-----
    pipe = TFTInferencePipeline.from_latest("x_trend")
    result = pipe.predict(date(2026, 2, 6), MarketDataStore())
    print(result.positions, result.confidences)

    # Batch mode for backtesting
    results = pipe.predict_batch(dates, store)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _HAS_TORCH = False
    _DEVICE = None


# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class InferenceResult:
    """Result of a single-date prediction.

    Attributes
    ----------
    date : date
        The prediction date.
    positions : dict[str, float]
        Asset name → position signal in [-max_pos, max_pos].
    confidences : dict[str, float]
        Asset name → confidence score in [0, 1].
    feature_importance : dict[str, float]
        Top feature weights from VSN.
    raw_output : dict[str, Any]
        Raw model outputs (mu, sigma for Gaussian mode).
    metadata : dict[str, Any]
        Checkpoint version, model type, inference time, etc.
    """

    date: date
    positions: dict[str, float]
    confidences: dict[str, float]
    feature_importance: dict[str, float] = field(default_factory=dict)
    raw_output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Inference Pipeline
# ============================================================================


_SYMBOL_MAP = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "Nifty Bank",
    "FINNIFTY": "NIFTY FINANCIAL SERVICES",
    "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
}

_SYMBOL_MAP_REV = {v.upper(): k for k, v in _SYMBOL_MAP.items()}


class TFTInferencePipeline:
    """Production inference from a saved checkpoint.

    Reconstructs the XTrendModel from checkpoint metadata, loads weights,
    and provides predict() for single-date and predict_batch() for
    multi-date inference.

    Parameters
    ----------
    model : nn.Module
        Loaded XTrendModel.
    metadata : CheckpointMetadata
        Full metadata from checkpoint.
    norm_means : ndarray
        Feature normalization means from training.
    norm_stds : ndarray
        Feature normalization stds from training.
    checkpoint_dir : Path
        Source checkpoint directory.
    """

    def __init__(
        self,
        model: "nn.Module",
        metadata: Any,
        norm_means: np.ndarray,
        norm_stds: np.ndarray,
        checkpoint_dir: Optional[Path] = None,
        inference_timeout: float = 30.0,
        drift_monitor: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.norm_means = norm_means
        self.norm_stds = norm_stds
        self.checkpoint_dir = checkpoint_dir
        self.inference_timeout = inference_timeout
        self._drift_monitor = drift_monitor

        self._feature_names = metadata.feature_names
        self._asset_names = metadata.asset_names or list(_SYMBOL_MAP.keys())
        self._n_assets = metadata.n_assets
        self._n_features = metadata.n_features

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: Optional[str] = None,
    ) -> "TFTInferencePipeline":
        """Load inference pipeline from a checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : str or Path
        device : str, optional — "cuda" or "cpu"

        Returns
        -------
        TFTInferencePipeline
        """
        from .checkpoint_manager import CheckpointManager, CheckpointMetadata
        from quantlaxmi.models.ml.tft.x_trend import XTrendConfig, XTrendModel

        mgr = CheckpointManager()
        state_dict, metadata = mgr.load(checkpoint_dir)

        # Reconstruct config
        cfg_dict = metadata.config
        x_cfg = XTrendConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in XTrendConfig.__dataclass_fields__
        })

        # Build model
        model = XTrendModel(x_cfg)
        model.load_state_dict(state_dict)

        dev = torch.device(device) if device else _DEVICE
        model = model.to(dev)
        model.eval()

        # Extract normalization stats
        norm_info = metadata.normalization
        norm_means = np.array(norm_info.get("means", [0.0] * x_cfg.n_features))
        norm_stds = np.array(norm_info.get("stds", [1.0] * x_cfg.n_features))

        logger.info(
            "Loaded TFTInferencePipeline from %s (v%d, %d features, %d assets)",
            checkpoint_dir, metadata.version, metadata.n_features, metadata.n_assets,
        )

        return cls(
            model=model,
            metadata=metadata,
            norm_means=norm_means,
            norm_stds=norm_stds,
            checkpoint_dir=Path(checkpoint_dir),
        )

    @classmethod
    def from_latest(
        cls,
        model_type: str = "x_trend",
        base_dir: str | Path = "checkpoints",
        device: Optional[str] = None,
    ) -> "TFTInferencePipeline":
        """Load the most recent checkpoint.

        Parameters
        ----------
        model_type : str
        base_dir : str or Path
        device : str, optional

        Returns
        -------
        TFTInferencePipeline
        """
        from .checkpoint_manager import CheckpointManager

        mgr = CheckpointManager(base_dir)
        checkpoints = mgr.list_checkpoints(model_type)
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found for {model_type!r} in {base_dir}"
            )
        latest = checkpoints[-1]
        return cls.from_checkpoint(latest["path"], device)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        d: date,
        store: Any,
        lookback_days: int = 120,
    ) -> InferenceResult:
        """Generate predictions for a single date.

        Parameters
        ----------
        d : date
            Target prediction date.
        store : MarketDataStore
            Data access.
        lookback_days : int
            Days of history to load for feature construction.

        Returns
        -------
        InferenceResult
        """
        t0 = time.time()

        # Build features
        features, dates_idx = self._build_features(d, store, lookback_days)

        if features is None or len(features) == 0:
            logger.warning("No features available for %s", d)
            return InferenceResult(
                date=d,
                positions={a: 0.0 for a in self._asset_names},
                confidences={a: 0.0 for a in self._asset_names},
                metadata={"error": "no_features"},
            )

        # Normalize using saved stats (guard against zero std — Bug #17)
        safe_stds = np.where(self.norm_stds > 1e-10, self.norm_stds, 1.0)
        norm_features = (features - self.norm_means) / safe_stds

        # Drift monitoring (Bug #24) — check before model forward
        if self._drift_monitor is not None:
            try:
                # Flatten features for drift check: (n_days * n_assets, n_features)
                flat_features = features.reshape(-1, features.shape[-1])
                # Use zero predictions placeholder (pre-forward)
                dummy_preds = np.zeros(flat_features.shape[0])
                drift_report = self._drift_monitor.check_drift(flat_features, dummy_preds)
                if drift_report.overall_status == "critical":
                    logger.warning(
                        "Feature drift detected: %d features drifted, status=%s",
                        len(drift_report.drifted_features),
                        drift_report.overall_status,
                    )
            except Exception as e:
                logger.error("Drift monitor check failed: %s", e)

        # Forward pass with timeout guard (Bug #7)
        # Use a daemon thread so a hung forward pass cannot block the process.
        # ThreadPoolExecutor creates non-daemon threads that prevent clean exit
        # when the forward pass hangs (e.g. a stuck GPU op or test mock with
        # time.sleep).  A daemon thread is abandoned on timeout and gets reaped
        # automatically when the process exits.
        _forward_result: list = []          # mutable container for thread result
        _forward_error: list = []           # mutable container for thread exception

        def _run_forward():
            try:
                _forward_result.append(self._forward(norm_features))
            except Exception as exc:
                _forward_error.append(exc)

        worker = threading.Thread(target=_run_forward, daemon=True)
        worker.start()
        worker.join(timeout=self.inference_timeout)

        if worker.is_alive():
            # Thread is still running — timed out
            logger.error(
                "Inference timed out after %.1fs for %s",
                self.inference_timeout,
                d,
            )
            return self._safe_default(d)

        if _forward_error:
            logger.error("Inference forward pass failed: %s", _forward_error[0])
            return self._safe_default(d)

        positions, confidences, raw = _forward_result[0]

        # Get feature importance
        feat_imp = self._get_feature_importance()

        elapsed = time.time() - t0

        result = InferenceResult(
            date=d,
            positions=positions,
            confidences=confidences,
            feature_importance=feat_imp,
            raw_output=raw,
            metadata={
                "checkpoint_version": self.metadata.version,
                "model_type": self.metadata.model_type,
                "inference_time_ms": elapsed * 1000,
                "n_features": self._n_features,
                "lookback_days": lookback_days,
            },
        )

        logger.info(
            "Predicted %s: %s (%.1fms)",
            d,
            {k: f"{v:.3f}" for k, v in positions.items()},
            elapsed * 1000,
        )
        return result

    def predict_batch(
        self,
        dates: Sequence[date],
        store: Any,
        lookback_days: int = 120,
    ) -> list[InferenceResult]:
        """Batch prediction for multiple dates.

        Parameters
        ----------
        dates : sequence of date
        store : MarketDataStore
        lookback_days : int

        Returns
        -------
        list[InferenceResult]
        """
        results = []
        for d in dates:
            results.append(self.predict(d, store, lookback_days))
        return results

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _safe_default(self, d: date) -> InferenceResult:
        """Return a zero-signal result when inference fails or times out."""
        return InferenceResult(
            date=d,
            positions={a: 0.0 for a in self._asset_names},
            confidences={a: 0.0 for a in self._asset_names},
            metadata={"error": "timeout", "timeout_s": self.inference_timeout},
        )

    def _build_features(
        self,
        d: date,
        store: Any,
        lookback_days: int,
    ) -> tuple[Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
        """Build multi-asset feature tensor for prediction date.

        Returns (n_days, n_assets, n_features) or (None, None).
        """
        from quantlaxmi.features.mega import MegaFeatureBuilder

        start_date = (d - timedelta(days=int(lookback_days * 1.5))).isoformat()
        end_date = d.isoformat()

        builder = MegaFeatureBuilder()
        per_asset: list[pd.DataFrame] = []

        for asset_name in self._asset_names:
            mega_sym = _SYMBOL_MAP.get(asset_name.upper(), asset_name)
            try:
                df, names = builder.build(mega_sym, start_date, end_date)
                # Reindex to checkpoint's feature set
                df = df.reindex(columns=self._feature_names)
                per_asset.append(df)
            except Exception as e:
                logger.warning("Feature build failed for %s: %s", mega_sym, e)
                per_asset.append(pd.DataFrame())

        if not per_asset or all(df.empty for df in per_asset):
            return None, None

        # Align on dates (outer join)
        all_dates = sorted(
            set().union(*(df.index for df in per_asset if not df.empty))
        )
        dates_idx = pd.DatetimeIndex(all_dates)
        n_days = len(dates_idx)
        n_features = len(self._feature_names)

        features = np.full(
            (n_days, self._n_assets, n_features), np.nan, dtype=np.float64,
        )

        for i, df in enumerate(per_asset):
            if df.empty:
                continue
            aligned = df.reindex(dates_idx).ffill()
            features[:, i, :] = aligned.values

        # Zero remaining NaN
        features = np.nan_to_num(features, nan=0.0)

        return features, dates_idx

    def _forward(
        self,
        norm_features: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
        """Run model forward pass on normalized features.

        Parameters
        ----------
        norm_features : (n_days, n_assets, n_features)

        Returns
        -------
        positions : dict[asset_name → position]
        confidences : dict[asset_name → confidence]
        raw : dict[asset_name → raw output]
        """
        from quantlaxmi.models.ml.tft.x_trend import build_context_set

        cfg_dict = self.metadata.config
        seq_len = cfg_dict.get("seq_len", 42)
        n_context = cfg_dict.get("n_context", 16)
        ctx_len = cfg_dict.get("ctx_len", seq_len)
        max_position = cfg_dict.get("max_position", 0.25)
        position_smooth = cfg_dict.get("position_smooth", 0.3)

        rng = np.random.default_rng(42)
        n_days = norm_features.shape[0]

        positions = {}
        confidences = {}
        raw = {}

        dev = next(self.model.parameters()).device

        self.model.eval()
        with torch.no_grad():
            for asset_idx, asset_name in enumerate(self._asset_names):
                # Use last seq_len days
                t = n_days  # end of available data
                if t < seq_len:
                    positions[asset_name] = 0.0
                    confidences[asset_name] = 0.0
                    continue

                tw = norm_features[t - seq_len: t, asset_idx, :]

                ctx_seqs, ctx_ids = build_context_set(
                    norm_features,
                    target_start=t - seq_len,
                    n_context=n_context,
                    ctx_len=ctx_len,
                    rng=rng,
                )

                tgt_t = torch.tensor(tw[np.newaxis], dtype=torch.float32, device=dev)
                ctx_t = torch.tensor(ctx_seqs[np.newaxis], dtype=torch.float32, device=dev)
                tid_t = torch.tensor([asset_idx], dtype=torch.long, device=dev)
                cid_t = torch.tensor(ctx_ids[np.newaxis], dtype=torch.long, device=dev)

                out = self.model(tgt_t, ctx_t, tid_t, cid_t)

                if isinstance(out, tuple):
                    # Gaussian mode (mu, log_sigma)
                    mu, log_sigma = out
                    mu_val = mu.item()
                    sigma_val = math.exp(log_sigma.item())

                    # PTP mapping: pos = 2*Phi(mu/sigma) - 1
                    z = mu_val / max(sigma_val, 1e-6)
                    from scipy.stats import norm as scipy_norm
                    prob_up = scipy_norm.cdf(z)
                    pos = 2.0 * prob_up - 1.0

                    # Confidence: 1 - 2*sigma (low sigma = high confidence)
                    conf = max(0.0, min(1.0, 1.0 - 2.0 * sigma_val))

                    raw[asset_name] = {"mu": mu_val, "sigma": sigma_val, "z": z}
                else:
                    pos = out.item()
                    conf = abs(pos)  # simple confidence proxy
                    raw[asset_name] = {"position_raw": pos}

                # Clip to max position
                pos = max(-max_position, min(max_position, pos))
                positions[asset_name] = float(pos)
                confidences[asset_name] = float(conf)

        return positions, confidences, raw

    def _get_feature_importance(self) -> dict[str, float]:
        """Get current VSN feature weights."""
        try:
            vsn = self.model.vsn
            n_feat = vsn.n_features
            dev = next(self.model.parameters()).device

            with torch.no_grad():
                dummy = torch.ones(1, 1, n_feat, device=dev)
                raw_w = vsn.weight_grn(dummy)
                weights = torch.softmax(raw_w, dim=-1)
                w = weights.squeeze().cpu().numpy()

            result = {}
            for i, name in enumerate(self._feature_names):
                if i < len(w):
                    result[name] = float(w[i])

            # Return top 20
            sorted_feats = sorted(result.items(), key=lambda x: -x[1])
            return dict(sorted_feats[:20])
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def asset_names(self) -> list[str]:
        return self._asset_names

    @property
    def version(self) -> int:
        return self.metadata.version
