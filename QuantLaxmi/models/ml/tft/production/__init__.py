"""Production TFT/X-Trend Inference System.

Provides feature selection, HP tuning, checkpoint management, and production
inference for the TFT/X-Trend trading model.

Components
----------
- FeatureSelector: 2-tier VSN-native feature selection
- CheckpointManager: Versioned model save/load
- HPTuner: Optuna-based hyperparameter tuning
- TFTInferencePipeline: Production prediction from checkpoint
- TrainingPipeline: 5-phase training orchestrator
- TFTStrategy: BaseStrategy wrapper for orchestrator integration
"""

from .feature_selection import (
    FeatureSelector,
    FeatureSelectionConfig,
    StabilityReport,
)
from .checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
)
from .hp_tuner import (
    HPTuner,
    TunerConfig,
    TuningResult,
)
from .inference import (
    TFTInferencePipeline,
    InferenceResult,
)
from .training_pipeline import (
    TrainingPipeline,
    TrainingPipelineConfig,
    TrainingResult,
)
from .strategy_adapter import (
    TFTStrategy,
    create_strategy,
)

__all__ = [
    "FeatureSelector",
    "FeatureSelectionConfig",
    "StabilityReport",
    "CheckpointManager",
    "CheckpointMetadata",
    "HPTuner",
    "TunerConfig",
    "TuningResult",
    "TFTInferencePipeline",
    "InferenceResult",
    "TrainingPipeline",
    "TrainingPipelineConfig",
    "TrainingResult",
    "TFTStrategy",
    "create_strategy",
]
