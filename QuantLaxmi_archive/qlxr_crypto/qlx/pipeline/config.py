"""Pipeline configuration — frozen, serialisable, versionable.

A ``PipelineConfig`` captures every parameter needed to reproduce a
research run.  It can be created from a Python dict, loaded from YAML,
or constructed programmatically.  Because it is a frozen dataclass, it
is hashable and can serve as a cache key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    funding_annual_pct: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    name: str = "xgboost"
    task: str = "regression"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CVConfig:
    method: str = "walk_forward"  # or "expanding"
    window: int = 2000
    train_frac: float = 0.8
    gap: int = 150  # must >= target horizon
    step: int | None = None


@dataclass(frozen=True)
class TargetConfig:
    type: str = "future_return"
    horizon: int = 150
    smooth: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for a research run."""

    target: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    # Backtest signal thresholds
    long_entry_threshold: float = 0.01
    long_exit_threshold: float = 0.0
    short_entry_threshold: float = -0.01
    short_exit_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.cv.gap < self.target.horizon:
            raise ValueError(
                f"CV gap ({self.cv.gap}) must be >= target horizon "
                f"({self.target.horizon}).  This is non-negotiable."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> PipelineConfig:
        return PipelineConfig(
            target=TargetConfig(**d.get("target", {})),
            model=ModelConfig(**d.get("model", {})),
            cv=CVConfig(**d.get("cv", {})),
            costs=CostConfig(**d.get("costs", {})),
            long_entry_threshold=d.get("long_entry_threshold", 0.01),
            long_exit_threshold=d.get("long_exit_threshold", 0.0),
            short_entry_threshold=d.get("short_entry_threshold", -0.01),
            short_exit_threshold=d.get("short_exit_threshold", 0.0),
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
