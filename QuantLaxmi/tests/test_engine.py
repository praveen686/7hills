"""Integration test for the full ResearchEngine pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from core.base.timeguard import LookaheadError
from core.features import RSI, BollingerBands, CyclicalTime, HistoricalReturns
from core.pipeline.config import PipelineConfig
from core.pipeline.engine import EngineResult, ResearchEngine
from core.targets import FutureReturn


class TestResearchEngine:
    def test_full_pipeline(self, sample_ohlcv):
        """End-to-end: build features, train model, backtest with costs."""
        horizon = 10
        config = PipelineConfig.from_dict({
            "target": {"horizon": horizon},
            "model": {"name": "xgboost", "task": "regression"},
            "cv": {
                "method": "walk_forward",
                "window": 200,
                "train_frac": 0.7,
                "gap": horizon,
            },
            "costs": {"commission_bps": 10, "slippage_bps": 5},
            "long_entry_threshold": 0.001,
            "short_entry_threshold": -0.001,
        })

        features = [
            RSI(window=14),
            BollingerBands(window=20),
            HistoricalReturns(periods=(1, 5, 10)),
            CyclicalTime(),
        ]
        target = FutureReturn(horizon=horizon)

        engine = ResearchEngine(
            ohlcv=sample_ohlcv,
            features=features,
            target_transform=target,
            config=config,
        )

        result = engine.run(verbose=False)

        assert isinstance(result, EngineResult)
        assert len(result.fold_results) > 0
        assert len(result.all_predictions) > 0
        assert np.isfinite(result.aggregate_mse)
        assert result.backtest is not None
        assert result.backtest.cost_model.commission_bps == 10

    def test_rejects_bad_gap(self, sample_ohlcv):
        """Engine rejects config where gap < horizon."""
        with pytest.raises(ValueError):
            PipelineConfig.from_dict({
                "target": {"horizon": 50},
                "cv": {"gap": 10},
            })

    def test_summary_string(self, sample_ohlcv):
        config = PipelineConfig.from_dict({
            "target": {"horizon": 5},
            "cv": {"window": 100, "train_frac": 0.7, "gap": 5},
            "costs": {"commission_bps": 10, "slippage_bps": 5},
        })
        engine = ResearchEngine(
            ohlcv=sample_ohlcv,
            features=[RSI(window=14), HistoricalReturns(periods=(1, 5))],
            target_transform=FutureReturn(horizon=5),
            config=config,
        )
        result = engine.run(verbose=False)
        summary = result.summary()
        assert "MSE" in summary
        assert "Backtest" in summary
