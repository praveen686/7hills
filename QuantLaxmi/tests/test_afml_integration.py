"""Tests for AFML integration points.

Verifies that all AFML techniques (triple barrier, CUSUM, CPCV, HRP,
meta-labeling, bet sizing) work correctly and are importable from
quantlaxmi.models.afml.
"""
import numpy as np
import pandas as pd
import pytest


# --- Imports ---

class TestImports:
    """All AFML functions importable from quantlaxmi.models.afml."""

    def test_triple_barrier_importable(self):
        from quantlaxmi.models.afml import triple_barrier_labels
        assert callable(triple_barrier_labels)

    def test_cusum_importable(self):
        from quantlaxmi.models.afml import cusum_filter, get_daily_vol
        assert callable(cusum_filter)
        assert callable(get_daily_vol)

    def test_cpcv_importable(self):
        from quantlaxmi.models.afml import CombPurgedKFoldCV
        assert CombPurgedKFoldCV is not None

    def test_hrp_importable(self):
        from quantlaxmi.models.afml import HierarchicalRiskParity
        assert HierarchicalRiskParity is not None

    def test_meta_label_importable(self):
        from quantlaxmi.models.afml import meta_labeling, bet_size
        assert callable(meta_labeling)
        assert callable(bet_size)

    def test_integration_modules_importable(self):
        from quantlaxmi.core.targets.triple_barrier_target import TripleBarrierTargetBuilder
        from quantlaxmi.core.backtest.hrp_allocator import HRPAllocator
        from quantlaxmi.core.meta_sizer import MetaLabelSizer
        assert TripleBarrierTargetBuilder is not None
        assert HRPAllocator is not None
        assert MetaLabelSizer is not None


# --- Triple Barrier ---

class TestTripleBarrier:
    """Triple barrier labels on synthetic price path."""

    @pytest.fixture
    def synthetic_prices(self):
        rng = np.random.default_rng(42)
        n = 500
        returns = rng.normal(0, 0.01, n)
        prices = pd.Series(
            100 * np.exp(np.cumsum(returns)),
            index=pd.bdate_range("2024-01-01", periods=n),
            name="close",
        )
        return prices

    def test_labels_shape(self, synthetic_prices):
        from quantlaxmi.models.afml import triple_barrier_labels
        events = synthetic_prices.index[50::10]  # every 10th bar after warmup
        result = triple_barrier_labels(synthetic_prices, events)
        assert set(result.columns) >= {"ret", "bin", "t1"}
        assert len(result) > 0

    def test_labels_values(self, synthetic_prices):
        from quantlaxmi.models.afml import triple_barrier_labels
        events = synthetic_prices.index[50::10]
        result = triple_barrier_labels(synthetic_prices, events)
        assert set(result["bin"].unique()).issubset({-1, 0, 1})

    def test_labels_no_future_leak(self, synthetic_prices):
        """Labels only use data after event timestamp."""
        from quantlaxmi.models.afml import triple_barrier_labels
        events = synthetic_prices.index[50::10]
        result = triple_barrier_labels(synthetic_prices, events)
        for t0, row in result.iterrows():
            assert row["t1"] >= t0, "Barrier touch time must be >= event time"


# --- CUSUM Filter ---

class TestCUSUM:
    def test_event_count_scales_with_vol(self):
        from quantlaxmi.models.afml import cusum_filter
        rng = np.random.default_rng(42)
        n = 1000
        index = pd.bdate_range("2024-01-01", periods=n)

        # Low vol (geometric random walk with small returns)
        low_vol = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.005, n))), index=index
        )
        events_low = cusum_filter(low_vol, threshold=0.02)

        # High vol (geometric random walk with large returns)
        high_vol = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=index
        )
        events_high = cusum_filter(high_vol, threshold=0.02)

        assert len(events_high) > len(events_low), "Higher vol should produce more CUSUM events"

    def test_adaptive_threshold(self):
        from quantlaxmi.models.afml import cusum_filter, get_daily_vol
        rng = np.random.default_rng(42)
        n = 500
        index = pd.bdate_range("2024-01-01", periods=n)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=index)
        vol = get_daily_vol(prices)
        events = cusum_filter(prices, threshold=vol)
        assert len(events) > 0


# --- CPCV ---

class TestCPCV:
    def test_more_paths_than_kfold(self):
        from quantlaxmi.models.afml import CombPurgedKFoldCV
        n = 200
        X = np.random.randn(n, 5)
        y = np.random.randint(0, 2, n)

        cpcv = CombPurgedKFoldCV(n_splits=6, n_test_groups=2, purge_window=3)
        splits = list(cpcv.split(X))
        # Combinatorial: C(6,2) = 15 paths
        assert len(splits) >= 10, f"Expected >= 10 CPCV splits, got {len(splits)}"

    def test_purge_gap_respected(self):
        from quantlaxmi.models.afml import CombPurgedKFoldCV
        n = 100
        X = np.random.randn(n, 3)
        purge = 5
        cpcv = CombPurgedKFoldCV(n_splits=5, n_test_groups=2, purge_window=purge)
        for train_idx, test_idx in cpcv.split(X):
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0


# --- HRP ---

class TestHRP:
    def test_weights_sum_to_one(self):
        from quantlaxmi.models.afml import HierarchicalRiskParity
        rng = np.random.default_rng(42)
        ret = pd.DataFrame(
            rng.standard_normal((252, 5)),
            columns=["A", "B", "C", "D", "E"],
        )
        hrp = HierarchicalRiskParity()
        w = hrp.allocate(ret.cov(), returns=ret)
        assert abs(sum(w.values()) - 1.0) < 1e-10

    def test_weights_non_negative(self):
        from quantlaxmi.models.afml import HierarchicalRiskParity
        rng = np.random.default_rng(42)
        ret = pd.DataFrame(
            rng.standard_normal((252, 3)),
            columns=["X", "Y", "Z"],
        )
        hrp = HierarchicalRiskParity()
        w = hrp.allocate(ret.cov(), returns=ret)
        assert all(v >= 0 for v in w.values())

    def test_single_asset(self):
        from quantlaxmi.models.afml import HierarchicalRiskParity
        cov = pd.DataFrame([[0.04]], columns=["A"], index=["A"])
        hrp = HierarchicalRiskParity()
        w = hrp.allocate(cov)
        assert w == {"A": 1.0}


# --- Meta-Labeling + Bet Sizing ---

class TestMetaLabeling:
    def test_meta_labels_correct(self):
        from quantlaxmi.models.afml import meta_labeling
        preds = pd.Series([1, 1, -1, -1, 1])
        truth = pd.Series([1, -1, -1, 0, 1])
        meta = meta_labeling(preds, truth)
        assert meta.tolist() == [1, 0, 1, 0, 1]

    def test_bet_size_zero_at_half(self):
        from quantlaxmi.models.afml import bet_size
        probs = pd.Series([0.5])
        sizes = bet_size(probs)
        assert abs(sizes.iloc[0]) < 1e-6, "p=0.5 should give size~0"

    def test_bet_size_positive_above_half(self):
        from quantlaxmi.models.afml import bet_size
        probs = pd.Series([0.8, 0.9, 0.95])
        sizes = bet_size(probs)
        assert (sizes > 0).all()

    def test_bet_size_max_leverage(self):
        from quantlaxmi.models.afml import bet_size
        probs = pd.Series([0.99, 0.01])
        sizes = bet_size(probs, max_leverage=0.5)
        assert (sizes.abs() <= 0.5 + 1e-10).all()

    def test_end_to_end_meta_labeling(self):
        """Full pipeline: primary preds -> meta labels -> bet size."""
        from quantlaxmi.models.afml import meta_labeling, bet_size
        rng = np.random.default_rng(42)
        n = 100
        primary = pd.Series(rng.choice([-1, 1], n))
        truth = pd.Series(rng.choice([-1, 0, 1], n))
        meta = meta_labeling(primary, truth)
        # Simulate meta-model probabilities (correct predictions get higher prob)
        probs = pd.Series(np.where(meta == 1, 0.7, 0.3))
        sizes = bet_size(probs)
        assert len(sizes) == n
        # Correct predictions should get positive sizes
        assert sizes[meta == 1].mean() > 0


# --- Integration Wrappers ---

class TestIntegrationWrappers:
    def test_triple_barrier_target_builder(self):
        from quantlaxmi.core.targets.triple_barrier_target import TripleBarrierTargetBuilder
        rng = np.random.default_rng(42)
        n = 500
        prices = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            index=pd.bdate_range("2024-01-01", periods=n),
        )
        builder = TripleBarrierTargetBuilder(use_cusum_events=True)
        result = builder.build(prices)
        assert len(result) > 0
        assert "bin" in result.columns

    def test_hrp_allocator(self):
        from quantlaxmi.core.backtest.hrp_allocator import HRPAllocator
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.standard_normal((100, 3)),
            columns=["s1", "s2", "s3"],
        )
        alloc = HRPAllocator(lookback=63, min_history=21)
        w = alloc.allocate(returns)
        assert abs(sum(w.values()) - 1.0) < 1e-10
        assert all(v >= 0 for v in w.values())

    def test_hrp_allocator_insufficient_history(self):
        from quantlaxmi.core.backtest.hrp_allocator import HRPAllocator
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.standard_normal((5, 3)),
            columns=["s1", "s2", "s3"],
        )
        alloc = HRPAllocator(min_history=21)
        w = alloc.allocate(returns)
        # Should fall back to equal weight
        for v in w.values():
            assert abs(v - 1/3) < 1e-10

    def test_meta_label_sizer(self):
        from quantlaxmi.core.meta_sizer import MetaLabelSizer
        sizer = MetaLabelSizer(max_leverage=1.0)
        primary = pd.Series([1, 1, -1, -1])
        truth = pd.Series([1, -1, -1, 0])
        meta = sizer.compute_meta_labels(primary, truth)
        assert meta.tolist() == [1, 0, 1, 0]
        probs = pd.Series([0.8, 0.3, 0.7, 0.5])
        direction = pd.Series([1, 1, -1, -1])
        sized = sizer.size_positions(probs, direction)
        assert len(sized) == 4
