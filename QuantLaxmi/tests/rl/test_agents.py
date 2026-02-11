"""Tests for models.rl.agents package.

Tests the production RL agent wrappers:
  - KellySizer: Kelly-Merton position sizing
  - ThompsonStrategyAllocator: contextual Thompson Sampling
  - MarketMakingAgent: Avellaneda-Stoikov + RL market maker
  - DeepHedgingAgent: neural hedging for options portfolios
  - OptimalExecutionAgent: RL-based optimal order execution
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# KellySizer
# ---------------------------------------------------------------------------

class TestKellySizer:
    """Tests for Kelly-Merton position sizing."""

    def test_kelly_fraction(self):
        """f* = (mu - r) / sigma^2 = (0.15 - 0.05) / 0.04 = 2.5."""
        from quantlaxmi.models.rl.agents import KellySizer

        f = KellySizer.kelly_fraction(mu=0.15, sigma=0.2, r=0.05)
        assert abs(f - 2.5) < 1e-10, f"Expected 2.5, got {f}"

    def test_merton_fraction(self):
        """pi* = (mu - r) / (gamma * sigma^2) = (0.15 - 0.05) / (2 * 0.04) = 1.25."""
        from quantlaxmi.models.rl.agents import KellySizer

        f = KellySizer.merton_fraction(mu=0.15, sigma=0.2, r=0.05, gamma=2.0)
        assert abs(f - 1.25) < 1e-10, f"Expected 1.25, got {f}"

    def test_drawdown_adjustment_reduces_size(self):
        """Drawdown adjustment must reduce position size during drawdown."""
        from quantlaxmi.models.rl.agents import KellySizer

        sizer = KellySizer(mode="kelly", max_drawdown_pct=0.20)
        base_size = 1.0

        # No drawdown -> no change
        adj_none = sizer.drawdown_adjustment(base_size, drawdown=0.0)
        assert abs(adj_none - base_size) < 1e-10

        # 10% drawdown with 20% max -> 50% reduction
        adj_half = sizer.drawdown_adjustment(base_size, drawdown=0.10)
        assert abs(adj_half - 0.5) < 1e-10, f"Expected 0.5, got {adj_half}"

        # At max drawdown -> zero
        adj_zero = sizer.drawdown_adjustment(base_size, drawdown=0.20)
        assert abs(adj_zero) < 1e-10, f"Expected 0.0, got {adj_zero}"


# ---------------------------------------------------------------------------
# ThompsonStrategyAllocator
# ---------------------------------------------------------------------------

class TestThompsonStrategyAllocator:
    """Tests for contextual Thompson Sampling strategy allocator."""

    def _make_allocator(self):
        from quantlaxmi.models.rl.agents import ThompsonStrategyAllocator

        return ThompsonStrategyAllocator(
            strategy_names=["S1", "S2", "S3"],
            context_dim=3,
            seed=42,
        )

    def test_allocations_nonnegative_sum_leq_1(self):
        """Allocations must be non-negative and sum to <= 1."""
        alloc = self._make_allocator()
        context = np.array([0.1, 0.5, 0.3])
        weights = alloc.select_allocation(context)

        for name, w in weights.items():
            assert w >= 0.0, f"Negative allocation for {name}: {w}"

        total = sum(weights.values())
        assert total <= 1.0 + 1e-10, f"Allocations sum to {total} > 1"

    def test_update_shifts_posterior_mean(self):
        """Updating with positive returns should increase posterior mean."""
        alloc = self._make_allocator()
        context = np.array([0.1, 0.5, 0.3])

        # Get initial posterior mean for S1
        posteriors_before = alloc.get_posteriors()
        mu_before = posteriors_before["S1"]["mean"]

        # Feed several positive returns for S1
        for _ in range(10):
            alloc.update("S1", 0.05, context)

        posteriors_after = alloc.get_posteriors()
        mu_after = posteriors_after["S1"]["mean"]

        assert mu_after > mu_before, (
            f"Posterior mean should increase: before={mu_before}, after={mu_after}"
        )


# ---------------------------------------------------------------------------
# MarketMakingAgent
# ---------------------------------------------------------------------------

class TestMarketMakingAgent:
    """Tests for MarketMakingAgent."""

    def test_get_quotes_bid_lt_ask(self):
        """get_quotes must return bid < ask."""
        from quantlaxmi.models.rl.agents import MarketMakingAgent

        agent = MarketMakingAgent(
            instrument="BTCUSDT",
            max_inventory=10,
            sigma=0.02,
            gamma_risk=0.1,
            fill_rate_k=1.5,
            device="cpu",
            hidden_layers=(16, 8),
        )

        quotes = agent.get_quotes(
            mid_price=65000.0,
            inventory=0,
            market_state={"time_fraction": 0.5, "volatility": 0.02},
        )
        assert quotes["bid_price"] < quotes["ask_price"], (
            f"bid={quotes['bid_price']} >= ask={quotes['ask_price']}"
        )


# ---------------------------------------------------------------------------
# DeepHedgingAgent
# ---------------------------------------------------------------------------

class TestDeepHedgingAgent:
    """Tests for DeepHedgingAgent."""

    def test_construct_without_error(self):
        """DeepHedgingAgent can be constructed with device='cpu'."""
        from quantlaxmi.models.rl.agents import DeepHedgingAgent

        agent = DeepHedgingAgent(
            instrument="NIFTY",
            strategy="straddle",
            hedging_interval="daily",
            hidden_layers=(16, 8),
            device="cpu",
        )
        assert agent.instrument == "NIFTY"
        assert agent._trained is False


# ---------------------------------------------------------------------------
# OptimalExecutionAgent
# ---------------------------------------------------------------------------

class TestOptimalExecutionAgent:
    """Tests for OptimalExecutionAgent."""

    def test_construct_without_error(self):
        """OptimalExecutionAgent can be constructed with device='cpu'."""
        from quantlaxmi.models.rl.agents import OptimalExecutionAgent

        agent = OptimalExecutionAgent(
            instrument="NIFTY",
            algo="actor_critic",
            hidden_layers=(16, 8),
            device="cpu",
        )
        assert agent.instrument == "NIFTY"
        assert agent._trained is False
