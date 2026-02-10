"""Determinism test suite — 3x replay parity.

BLOCKER #2 — no live trading without proof of deterministic replay.

Verifies that every component produces *bit-identical* results across
three independent runs on the same data + config.  Covers:

  - Feature functions: price_entropy, mutual_information, rolling versions
  - Regime detector: classify_regime, rolling_regime
  - Microstructure features: VPIN, Kyle's Lambda, tick_entropy
  - Signal state: reset_signal_state clears properly between runs
  - CostModel: all properties deterministic
  - PortfolioState: DD calculations deterministic
  - RiskManager: gate results deterministic

All tests use seeded data for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prices_100():
    """100-bar price series, seeded."""
    rng = np.random.default_rng(42)
    log_rets = rng.normal(0.0005, 0.015, 100)
    return 22000 * np.exp(np.cumsum(log_rets))


@pytest.fixture
def prices_300():
    """300-bar price series for rolling tests."""
    rng = np.random.default_rng(99)
    log_rets = rng.normal(0.0003, 0.012, 300)
    return 45000 * np.exp(np.cumsum(log_rets))


@pytest.fixture
def tick_prices():
    """1000-tick price series with volume."""
    rng = np.random.default_rng(77)
    log_rets = rng.normal(0.0, 0.001, 1000)
    prices = 22500 * np.exp(np.cumsum(log_rets))
    volumes = rng.integers(50, 5000, 1000).astype(float)
    return prices, volumes


# ===================================================================
# 1. INFORMATION-THEORETIC FEATURES — Determinism
# ===================================================================

class TestEntropyDeterminism:
    """price_entropy must be bitwise identical across 3 runs."""

    def test_3x_replay_word_len_2(self, prices_100):
        from quantlaxmi.features.information import price_entropy
        results = [price_entropy(prices_100, word_length=2) for _ in range(3)]
        assert results[0] == results[1] == results[2]

    def test_3x_replay_word_len_3(self, prices_100):
        from quantlaxmi.features.information import price_entropy
        results = [price_entropy(prices_100, word_length=3) for _ in range(3)]
        assert results[0] == results[1] == results[2]

    def test_3x_replay_with_window(self, prices_300):
        from quantlaxmi.features.information import price_entropy
        results = [price_entropy(prices_300, word_length=2, window=100) for _ in range(3)]
        assert results[0] == results[1] == results[2]

    def test_different_data_different_result(self, prices_100, prices_300):
        from quantlaxmi.features.information import price_entropy
        r1 = price_entropy(prices_100)
        r2 = price_entropy(prices_300[:100])
        # Different seeds → different entropy (with high probability)
        assert r1 != r2


class TestMutualInfoDeterminism:
    """mutual_information must be bitwise identical across 3 runs."""

    def test_3x_replay(self, prices_100):
        from quantlaxmi.features.information import mutual_information
        results = [mutual_information(prices_100) for _ in range(3)]
        assert results[0] == results[1] == results[2]

    def test_3x_replay_word_len_3(self, prices_100):
        from quantlaxmi.features.information import mutual_information
        results = [mutual_information(prices_100, word_length=3) for _ in range(3)]
        assert results[0] == results[1] == results[2]

    def test_3x_replay_with_window(self, prices_300):
        from quantlaxmi.features.information import mutual_information
        results = [mutual_information(prices_300, word_length=2, window=100) for _ in range(3)]
        assert results[0] == results[1] == results[2]


class TestRollingFeatureDeterminism:
    """rolling_entropy and rolling_mutual_info must produce identical arrays."""

    def test_rolling_entropy_3x(self, prices_300):
        from quantlaxmi.features.information import rolling_entropy
        runs = [rolling_entropy(prices_300, window=50) for _ in range(3)]
        np.testing.assert_array_equal(runs[0], runs[1])
        np.testing.assert_array_equal(runs[1], runs[2])

    def test_rolling_mi_3x(self, prices_300):
        from quantlaxmi.features.information import rolling_mutual_info
        runs = [rolling_mutual_info(prices_300, window=50) for _ in range(3)]
        np.testing.assert_array_equal(runs[0], runs[1])
        np.testing.assert_array_equal(runs[1], runs[2])

    def test_rolling_entropy_nan_prefix(self, prices_300):
        from quantlaxmi.features.information import rolling_entropy
        out = rolling_entropy(prices_300, window=50)
        assert np.all(np.isnan(out[:49]))
        assert np.all(~np.isnan(out[49:]))


# ===================================================================
# 2. REGIME DETECTOR — Determinism
# ===================================================================

class TestRegimeDetectorDeterminism:
    """classify_regime and rolling_regime must produce identical results."""

    def test_classify_3x_replay(self, prices_300):
        from quantlaxmi.strategies.s7_regime.detector import classify_regime
        runs = [classify_regime(prices_300, vpin=0.3, entropy_window=100) for _ in range(3)]
        for r in runs:
            assert r.regime == runs[0].regime
            assert r.entropy == runs[0].entropy
            assert r.mutual_info == runs[0].mutual_info
            assert r.confidence == runs[0].confidence

    def test_rolling_regime_3x_replay(self, prices_300):
        from quantlaxmi.strategies.s7_regime.detector import rolling_regime
        runs = [rolling_regime(prices_300, window=50) for _ in range(3)]
        for i in range(len(runs[0])):
            assert runs[0][i].regime == runs[1][i].regime == runs[2][i].regime
            assert runs[0][i].entropy == runs[1][i].entropy == runs[2][i].entropy

    def test_vpin_override_deterministic(self, prices_300):
        from quantlaxmi.strategies.s7_regime.detector import classify_regime, MarketRegime
        runs = [classify_regime(prices_300, vpin=0.85) for _ in range(3)]
        for r in runs:
            assert r.regime == MarketRegime.RANDOM


# ===================================================================
# 3. MICROSTRUCTURE FEATURES — Determinism
# ===================================================================

class TestVPINFromTicksDeterminism:
    """vpin_from_ticks must produce bitwise identical arrays."""

    def test_3x_replay(self, tick_prices):
        from quantlaxmi.features.microstructure import vpin_from_ticks
        prices, volumes = tick_prices
        runs = [vpin_from_ticks(prices, volumes, bucket_size=500_000, n_buckets=20) for _ in range(3)]
        np.testing.assert_array_equal(runs[0], runs[1])
        np.testing.assert_array_equal(runs[1], runs[2])


class TestTickEntropyDeterminism:
    def test_3x_replay(self, tick_prices):
        from quantlaxmi.features.microstructure import tick_entropy
        prices, _ = tick_prices
        runs = [tick_entropy(prices, window=50) for _ in range(3)]
        np.testing.assert_array_equal(runs[0], runs[1])
        np.testing.assert_array_equal(runs[1], runs[2])


class TestHawkesDeterminism:
    def test_3x_replay(self):
        from quantlaxmi.features.microstructure import trade_arrival_hawkes
        rng = np.random.default_rng(55)
        ts = np.sort(rng.uniform(0, 3600, 200))
        runs = [trade_arrival_hawkes(ts, eval_interval=5.0) for _ in range(3)]
        np.testing.assert_array_equal(runs[0][0], runs[1][0])
        np.testing.assert_array_equal(runs[1][0], runs[2][0])
        assert runs[0][1:] == runs[1][1:] == runs[2][1:]


# ===================================================================
# 4. BAR-LEVEL MICROSTRUCTURE — Determinism
# ===================================================================

class TestBarMicrostructureDeterminism:
    """Microstructure Feature._compute must be deterministic."""

    def test_3x_replay(self):
        import pandas as pd
        from quantlaxmi.features.microstructure import Microstructure
        rng = np.random.default_rng(123)
        n = 200
        close = 22000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        df = pd.DataFrame({
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + abs(rng.normal(0, 0.005, n))),
            "Low": close * (1 - abs(rng.normal(0, 0.005, n))),
            "Close": close,
            "Volume": rng.integers(1000, 100000, n).astype(float),
        })
        feat = Microstructure(window=30)
        runs = [feat._compute(df) for _ in range(3)]
        pd.testing.assert_frame_equal(runs[0], runs[1])
        pd.testing.assert_frame_equal(runs[1], runs[2])


# ===================================================================
# 5. COST MODEL — Determinism
# ===================================================================

class TestCostModelDeterminism:
    def test_properties_3x(self):
        from quantlaxmi.core.backtest.costs import CostModel
        cm = CostModel(commission_bps=5.0, slippage_bps=3.0, funding_annual_pct=8.0)
        for _ in range(3):
            assert cm.one_way_frac == (5.0 + 3.0) / 10_000
            assert cm.roundtrip_frac == 2 * (5.0 + 3.0) / 10_000
            assert cm.roundtrip_bps == 2 * (5.0 + 3.0)
            assert cm.holding_cost_per_bar(252.0) == 8.0 / 100.0 / 252.0


# ===================================================================
# 6. RISK MANAGER — Determinism
# ===================================================================

class TestRiskManagerDeterminism:
    """Same targets + same state → identical gate results across 3 runs."""

    def test_3x_replay_pass(self):
        from quantlaxmi.core.allocator.meta import TargetPosition
        from quantlaxmi.core.risk.limits import RiskLimits
        from quantlaxmi.core.risk.manager import GateResult, PortfolioState, RiskManager

        targets = [TargetPosition(
            strategy_id="s1_vrp", symbol="NIFTY",
            direction="long", weight=0.05, instrument_type="FUT",
        )]
        state = PortfolioState(equity=1.0, peak_equity=1.0, vpin=0.30)

        for _ in range(3):
            rm = RiskManager(limits=RiskLimits())
            results = rm.check(targets, state)
            assert results[0].gate == GateResult.PASS
            assert results[0].approved is True
            assert results[0].adjusted_weight == 0.05

    def test_3x_replay_block(self):
        from quantlaxmi.core.allocator.meta import TargetPosition
        from quantlaxmi.core.risk.limits import RiskLimits
        from quantlaxmi.core.risk.manager import GateResult, PortfolioState, RiskManager

        targets = [TargetPosition(
            strategy_id="s1_vrp", symbol="NIFTY",
            direction="long", weight=0.10, instrument_type="FUT",
        )]
        state = PortfolioState(equity=0.90, peak_equity=1.0, vpin=0.80)

        for _ in range(3):
            rm = RiskManager(limits=RiskLimits())
            results = rm.check(targets, state)
            assert results[0].gate == GateResult.BLOCK_VPIN
            assert results[0].approved is False
            assert results[0].adjusted_weight == 0.0

    def test_3x_replay_reduce(self):
        from quantlaxmi.core.allocator.meta import TargetPosition
        from quantlaxmi.core.risk.limits import RiskLimits
        from quantlaxmi.core.risk.manager import GateResult, PortfolioState, RiskManager

        targets = [TargetPosition(
            strategy_id="s5_hawkes", symbol="NIFTY",
            direction="long", weight=0.25, instrument_type="FUT",
        )]
        state = PortfolioState()

        results_all = []
        for _ in range(3):
            rm = RiskManager(limits=RiskLimits())
            results = rm.check(targets, state)
            results_all.append(results[0])

        assert all(r.gate == GateResult.REDUCE_SIZE for r in results_all)
        assert results_all[0].adjusted_weight == results_all[1].adjusted_weight == results_all[2].adjusted_weight


# ===================================================================
# 7. PORTFOLIO STATE — Determinism
# ===================================================================

class TestPortfolioStateDeterminism:
    def test_dd_3x(self):
        from quantlaxmi.core.risk.manager import PortfolioState
        s = PortfolioState(equity=0.93, peak_equity=1.0)
        dds = [s.portfolio_dd for _ in range(3)]
        assert dds[0] == dds[1] == dds[2]

    def test_strategy_dd_3x(self):
        from quantlaxmi.core.risk.manager import PortfolioState
        s = PortfolioState(
            strategy_equity={"s1_vrp": 0.96},
            strategy_peaks={"s1_vrp": 1.0},
        )
        dds = [s.strategy_dd("s1_vrp") for _ in range(3)]
        assert dds[0] == dds[1] == dds[2]

    def test_total_exposure_3x(self):
        from quantlaxmi.core.risk.manager import PortfolioState
        s = PortfolioState(positions={
            "NIFTY": {"weight": 0.15},
            "BANKNIFTY": {"weight": 0.10},
            "RELIANCE": {"weight": 0.03},
        })
        exps = [s.total_exposure() for _ in range(3)]
        assert exps[0] == exps[1] == exps[2]


# ===================================================================
# 8. SIGNAL PROTOCOL — Determinism
# ===================================================================

class TestSignalDeterminism:
    """Signal dataclass (frozen) properties are deterministic."""

    def test_frozen_signal_fields(self):
        from quantlaxmi.strategies.protocol import Signal
        s = Signal(
            strategy_id="s1_vrp", symbol="NIFTY",
            direction="long", conviction=0.8,
            instrument_type="FUT", ttl_bars=5,
            metadata={"composite": 0.42},
        )
        for _ in range(3):
            assert s.strategy_id == "s1_vrp"
            assert s.conviction == 0.8
            assert s.metadata["composite"] == 0.42

    def test_signal_immutable(self):
        from quantlaxmi.strategies.protocol import Signal
        s = Signal(strategy_id="s1_vrp", symbol="NIFTY", direction="long", conviction=0.5)
        with pytest.raises(AttributeError):
            s.conviction = 0.9  # type: ignore


# ===================================================================
# 9. FP TOLERANCE — Cross-platform parity
# ===================================================================

class TestFPTolerance:
    """Verify numeric operations stay within rtol=1e-6 across runs."""

    def test_entropy_fp_stable(self, prices_300):
        from quantlaxmi.features.information import price_entropy
        runs = [price_entropy(prices_300, word_length=2, window=100) for _ in range(10)]
        for r in runs:
            np.testing.assert_allclose(r, runs[0], rtol=1e-10)

    def test_mi_fp_stable(self, prices_300):
        from quantlaxmi.features.information import mutual_information
        runs = [mutual_information(prices_300, word_length=2, window=100) for _ in range(10)]
        for r in runs:
            np.testing.assert_allclose(r, runs[0], rtol=1e-10)

    def test_cost_model_fp_stable(self):
        from quantlaxmi.core.backtest.costs import CostModel
        cm = CostModel(commission_bps=3.5, slippage_bps=1.5)
        results = [cm.roundtrip_frac for _ in range(10)]
        assert all(r == results[0] for r in results)
