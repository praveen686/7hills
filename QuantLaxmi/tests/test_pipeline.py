"""Tests for CV splitters and the pipeline engine."""

from __future__ import annotations

import numpy as np
import pytest

from core.core.timeguard import LookaheadError
from core.pipeline.split import ExpandingSplit, WalkForwardSplit
from core.pipeline.config import PipelineConfig


class TestWalkForwardSplit:
    def test_basic_split(self):
        splitter = WalkForwardSplit(
            window=100, train_frac=0.7, gap=10, horizon=10
        )
        folds = list(splitter.split(500))
        assert len(folds) > 0

        for fold in folds:
            # Train and test don't overlap
            assert fold.train_idx[-1] < fold.gap_idx[0]
            assert fold.gap_idx[-1] < fold.test_idx[0]

            # Sizes are correct
            assert len(fold.train_idx) == 70
            assert len(fold.gap_idx) == 10
            assert len(fold.test_idx) == 20

    def test_gap_enforced(self):
        """Cannot create splitter with gap < horizon."""
        with pytest.raises(LookaheadError):
            WalkForwardSplit(window=100, train_frac=0.7, gap=5, horizon=10)

    def test_no_temporal_overlap(self):
        splitter = WalkForwardSplit(
            window=100, train_frac=0.7, gap=10, horizon=10
        )
        folds = list(splitter.split(500))

        for fold in folds:
            train_set = set(fold.train_idx)
            gap_set = set(fold.gap_idx)
            test_set = set(fold.test_idx)

            assert train_set.isdisjoint(gap_set)
            assert train_set.isdisjoint(test_set)
            assert gap_set.isdisjoint(test_set)

    def test_non_overlapping_test_sets_default(self):
        """With default step (=test_size), test sets don't overlap."""
        splitter = WalkForwardSplit(
            window=100, train_frac=0.7, gap=10, horizon=10
        )
        folds = list(splitter.split(500))

        for i in range(1, len(folds)):
            prev_test_end = folds[i - 1].test_idx[-1]
            curr_test_start = folds[i].test_idx[0]
            assert curr_test_start > prev_test_end

    def test_n_splits_consistent(self):
        splitter = WalkForwardSplit(
            window=100, train_frac=0.7, gap=10, horizon=10
        )
        folds = list(splitter.split(500))
        assert len(folds) == splitter.n_splits(500)

    def test_too_small_window_raises(self):
        with pytest.raises(ValueError, match="too small"):
            WalkForwardSplit(window=20, train_frac=0.7, gap=15, horizon=10)


class TestExpandingSplit:
    def test_training_set_grows(self):
        splitter = ExpandingSplit(
            min_train=50, test_size=20, gap=10, horizon=10
        )
        folds = list(splitter.split(300))
        assert len(folds) > 1

        for i in range(1, len(folds)):
            assert len(folds[i].train_idx) > len(folds[i - 1].train_idx)

    def test_gap_enforced(self):
        with pytest.raises(LookaheadError):
            ExpandingSplit(min_train=50, test_size=20, gap=5, horizon=10)

    def test_train_always_starts_at_zero(self):
        splitter = ExpandingSplit(
            min_train=50, test_size=20, gap=10, horizon=10
        )
        folds = list(splitter.split(300))
        for fold in folds:
            assert fold.train_idx[0] == 0


class TestPipelineConfig:
    def test_gap_validation(self):
        """Config rejects gap < horizon at construction time."""
        with pytest.raises(ValueError, match="non-negotiable"):
            PipelineConfig.from_dict({
                "target": {"horizon": 150},
                "cv": {"gap": 50},
            })

    def test_valid_config(self):
        cfg = PipelineConfig.from_dict({
            "target": {"horizon": 10, "type": "future_return"},
            "cv": {"gap": 10, "window": 200, "train_frac": 0.8},
            "costs": {"commission_bps": 10, "slippage_bps": 5},
        })
        assert cfg.target.horizon == 10
        assert cfg.cv.gap == 10
        assert cfg.costs.commission_bps == 10
        assert cfg.costs.slippage_bps == 5

    def test_roundtrip_dict(self):
        cfg = PipelineConfig()
        d = cfg.to_dict()
        cfg2 = PipelineConfig.from_dict(d)
        assert cfg2.target.horizon == cfg.target.horizon
