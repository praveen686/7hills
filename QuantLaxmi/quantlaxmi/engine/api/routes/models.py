"""Model prediction endpoints — TFT and RL model status and predictions.

GET /api/models/status          — overall model loading status
GET /api/models/tft/predictions — per-asset TFT position predictions
GET /api/models/tft/features    — top feature importances from TFT
GET /api/models/rl/status       — RL agent status and training metrics
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from quantlaxmi.data._paths import _PROJECT_ROOT

logger = logging.getLogger(__name__)

# Model artifact directories (absolute paths via _paths.py)
_TFT_CHECKPOINT_DIR = _PROJECT_ROOT / "data" / "models" / "tft" / "production"
_VSN_WEIGHTS_PATH = _PROJECT_ROOT / "data" / "models" / "tft" / "vsn_weights.json"
_RL_METRICS_PATH = _PROJECT_ROOT / "data" / "models" / "rl" / "training_metrics.json"

router = APIRouter(prefix="/api/models", tags=["models"])

# Cache for TFT predictions (avoid GPU re-runs on every request)
_tft_cache: dict[str, Any] = {}
_tft_cache_ts: float = 0
_TFT_CACHE_TTL = 60.0  # seconds


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

class ModelStatus(BaseModel):
    tft_loaded: bool
    tft_checkpoint: str
    tft_features: int
    rl_loaded: bool
    rl_agent_type: str


class AssetPrediction(BaseModel):
    asset: str
    position: str  # "long", "short", "flat"
    confidence: float
    direction_score: float
    top_features: list[dict[str, Any]]


class TFTPredictions(BaseModel):
    timestamp: str
    assets: list[AssetPrediction]
    model_version: str


class FeatureWeight(BaseModel):
    name: str
    importance: float
    group: str


class RLStatus(BaseModel):
    loaded: bool
    agent_type: str
    config: dict[str, Any]
    training_metrics: dict[str, Any]


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.get("/status", response_model=ModelStatus)
async def get_model_status(request: Request) -> ModelStatus:
    """Return overall model loading status."""
    tft_loaded = False
    tft_checkpoint = ""
    tft_features = 0
    rl_loaded = False
    rl_agent_type = ""

    # Check TFT
    try:
        from quantlaxmi.models.ml.tft.production.inference import TFTInferencePipeline  # noqa: F401
        if _TFT_CHECKPOINT_DIR.exists():
            checkpoints = sorted(_TFT_CHECKPOINT_DIR.glob("*.pt"))
            if checkpoints:
                tft_loaded = True
                tft_checkpoint = checkpoints[-1].name
                tft_features = 73  # from Phase 3 feature selection
    except Exception:
        pass

    # Check RL
    try:
        from quantlaxmi.models.rl.integration.rl_trading_agent import RLTradingAgent
        rl_loaded = True
        rl_agent_type = "PPO"
    except Exception:
        pass

    return ModelStatus(
        tft_loaded=tft_loaded,
        tft_checkpoint=tft_checkpoint,
        tft_features=tft_features,
        rl_loaded=rl_loaded,
        rl_agent_type=rl_agent_type,
    )


@router.get("/tft/predictions", response_model=TFTPredictions)
async def get_tft_predictions(request: Request) -> TFTPredictions:
    """Return per-asset TFT position predictions.

    Results are cached for 60 seconds to avoid GPU re-runs.
    """
    global _tft_cache, _tft_cache_ts
    from datetime import datetime, timezone

    now = time.monotonic()

    # Return cached if fresh
    if _tft_cache and (now - _tft_cache_ts) < _TFT_CACHE_TTL:
        return TFTPredictions(**_tft_cache)

    # Try to generate predictions
    assets: list[dict[str, Any]] = []
    model_version = "unknown"

    try:
        from quantlaxmi.models.ml.tft.production.inference import TFTInferencePipeline
        from quantlaxmi.data.store import MarketDataStore
        from datetime import date as date_cls

        pipeline = TFTInferencePipeline.from_checkpoint(str(_TFT_CHECKPOINT_DIR))
        store = MarketDataStore()
        result = pipeline.predict(date_cls.today(), store)
        model_version = "v7"

        for asset_name, score in result.positions.items():
            confidence = result.confidences.get(asset_name, abs(score))
            position = "flat"
            if score > 0.3:
                position = "long"
            elif score < -0.3:
                position = "short"

            # Extract top features from VSN weights if available
            top_feats = [
                {"name": k, "importance": v}
                for k, v in sorted(
                    result.feature_importance.items(),
                    key=lambda x: x[1], reverse=True,
                )[:5]
            ] if result.feature_importance else []

            assets.append({
                "asset": asset_name,
                "position": position,
                "confidence": float(confidence),
                "direction_score": float(score),
                "top_features": top_feats,
            })
    except Exception as exc:
        logger.debug("TFT prediction failed: %s", exc)

        # Provide stub predictions for known assets
        for sym in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
            assets.append({
                "asset": sym,
                "position": "flat",
                "confidence": 0.0,
                "direction_score": 0.0,
                "top_features": [],
            })

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "assets": [AssetPrediction(**a) for a in assets],
        "model_version": model_version,
    }

    # Cache
    _tft_cache = {
        "timestamp": result["timestamp"],
        "assets": assets,
        "model_version": model_version,
    }
    _tft_cache_ts = now

    return TFTPredictions(**result)


@router.get("/tft/features", response_model=list[FeatureWeight])
async def get_tft_features(request: Request) -> list[FeatureWeight]:
    """Return TFT feature importances (VSN weights)."""
    features: list[dict[str, Any]] = []

    # Read from saved VSN weights (always available, no GPU needed)
    try:
        import json
        if _VSN_WEIGHTS_PATH.exists():
            vsn = json.loads(_VSN_WEIGHTS_PATH.read_text())
            for name, importance in vsn.items():
                features.append({
                    "name": name,
                    "importance": float(importance),
                    "group": _infer_feature_group(name),
                })
    except Exception as exc:
        logger.debug("VSN weights read failed: %s", exc)

    features.sort(key=lambda f: f["importance"], reverse=True)
    return [FeatureWeight(**f) for f in features[:50]]


@router.get("/rl/status", response_model=RLStatus)
async def get_rl_status(request: Request) -> RLStatus:
    """Return RL agent status and training metrics."""
    loaded = False
    agent_type = "none"
    config: dict[str, Any] = {}
    metrics: dict[str, Any] = {}

    try:
        from quantlaxmi.models.rl.integration.rl_trading_agent import RLTradingAgent
        loaded = True
        agent_type = "PPO"
        config = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        }
    except Exception:
        pass

    try:
        import json
        if _RL_METRICS_PATH.exists():
            metrics = json.loads(_RL_METRICS_PATH.read_text())
    except Exception:
        pass

    return RLStatus(
        loaded=loaded,
        agent_type=agent_type,
        config=config,
        training_metrics=metrics,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _infer_feature_group(name: str) -> str:
    """Infer feature group from feature name prefix."""
    prefixes = {
        "ret_": "technical",
        "vol_": "volatility",
        "rsi_": "momentum",
        "macd_": "momentum",
        "atr_": "volatility",
        "vpin_": "microstructure",
        "oi_": "structure",
        "dff_": "divergence",
        "ns_": "news",
        "fii_": "macro",
        "crypto_": "crypto",
        "iv_": "options",
        "skew_": "options",
        "gex_": "structure",
    }
    for prefix, group in prefixes.items():
        if name.startswith(prefix):
            return group
    return "other"
