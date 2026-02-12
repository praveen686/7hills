"""Classic TFT research module (self-contained, no quantlaxmi dependency)."""

from .classic_tft import (
    ClassicTFTConfig,
    ClassicTFTModel,
    GateAddNorm,
    InterpretableMultiHeadAttention,
    StaticCovariateEncoder,
    TemporalVSN,
    QuantileLoss,
    normalized_quantile_loss,
)
from .data_formatter import (
    ColumnDefinition,
    FormatterConfig,
    TFTDataFormatter,
)

__all__ = [
    "ClassicTFTConfig",
    "ClassicTFTModel",
    "GateAddNorm",
    "InterpretableMultiHeadAttention",
    "StaticCovariateEncoder",
    "TemporalVSN",
    "QuantileLoss",
    "normalized_quantile_loss",
    "ColumnDefinition",
    "FormatterConfig",
    "TFTDataFormatter",
]
