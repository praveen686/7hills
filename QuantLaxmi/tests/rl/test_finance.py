"""Tests for models.rl.finance package.

Tests the analytical and RL-based financial models:
  - MertonSolution: optimal portfolio weights
  - BlackScholesHedger: option pricing and greeks
  - BertsimasLoSolution: TWAP execution schedule
  - AlmgrenChrissSolution: risk-averse optimal execution
  - AvellanedaStoikovSolution: market-making quotes
  - DeepHedger: neural network hedging
  - AssetAllocPG: policy gradient asset allocation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# MertonSolution
# ---------------------------------------------------------------------------

class TestMertonSolution:
    """Tests for the Merton analytical portfolio solution."""

    def test_optimal_weights_single_asset(self):
        """pi* = (mu - r) / (gamma * sigma^2).

        With mu=0.15, r=0.05, sigma=0.2, gamma=2:
            pi* = (0.15 - 0.05) / (2 * 0.04) = 0.10 / 0.08 = 1.25
        """
        from quantlaxmi.models.rl.finance import MertonSolution

        mu = np.array([0.15])
        sigma = np.array([[0.04]])  # sigma^2 = 0.2^2 = 0.04
        r = 0.05
        gamma = 2.0

        weights = MertonSolution.optimal_weights(mu, sigma, r, gamma)
        assert weights.shape == (1,)
        assert abs(weights[0] - 1.25) < 1e-10, f"Expected 1.25, got {weights[0]}"


# ---------------------------------------------------------------------------
# BlackScholesHedger
# ---------------------------------------------------------------------------

class TestBlackScholesHedger:
    """Tests for Black-Scholes option pricing and greeks."""

    def test_put_call_parity(self):
        """Put-call parity: C - P = S - K*exp(-rT)."""
        from quantlaxmi.models.rl.finance import BlackScholesHedger

        S, K, tau, sigma, r = 100.0, 100.0, 0.25, 0.20, 0.05
        C = BlackScholesHedger.price(S, K, tau, sigma, r, "call")
        P = BlackScholesHedger.price(S, K, tau, sigma, r, "put")
        parity_rhs = S - K * math.exp(-r * tau)

        assert abs((C - P) - parity_rhs) < 1e-10, (
            f"Put-call parity violated: C-P={C - P}, S-K*exp(-rT)={parity_rhs}"
        )

    def test_call_delta_in_range(self):
        """Call delta must be in [0, 1]."""
        from quantlaxmi.models.rl.finance import BlackScholesHedger

        S, K, tau, sigma, r = 100.0, 100.0, 0.25, 0.20, 0.05
        delta = BlackScholesHedger.delta(S, K, tau, sigma, r, "call")
        assert 0.0 <= delta <= 1.0, f"Call delta out of range: {delta}"

    def test_put_delta_in_range(self):
        """Put delta must be in [-1, 0]."""
        from quantlaxmi.models.rl.finance import BlackScholesHedger

        S, K, tau, sigma, r = 100.0, 100.0, 0.25, 0.20, 0.05
        delta = BlackScholesHedger.delta(S, K, tau, sigma, r, "put")
        assert -1.0 <= delta <= 0.0, f"Put delta out of range: {delta}"


# ---------------------------------------------------------------------------
# BertsimasLoSolution
# ---------------------------------------------------------------------------

class TestBertsimasLoSolution:
    """Tests for Bertsimas-Lo TWAP execution."""

    def test_twap_schedule_sums_to_total(self):
        """TWAP schedule must sum to total_shares."""
        from quantlaxmi.models.rl.finance import BertsimasLoSolution

        total = 1000
        steps = 20
        schedule = BertsimasLoSolution.twap_schedule(total, steps)
        assert schedule.shape == (steps,)
        assert abs(schedule.sum() - total) < 1e-10, (
            f"TWAP sum={schedule.sum()}, expected {total}"
        )


# ---------------------------------------------------------------------------
# AlmgrenChrissSolution
# ---------------------------------------------------------------------------

class TestAlmgrenChrissSolution:
    """Tests for Almgren-Chriss risk-averse optimal execution."""

    def test_trajectory_boundary_conditions(self):
        """trajectory[0] = total_shares, trajectory[-1] ~= 0."""
        from quantlaxmi.models.rl.finance import AlmgrenChrissSolution

        ac = AlmgrenChrissSolution(
            total_shares=1000,
            num_steps=20,
            sigma=0.02,
            eta=0.005,
            gamma_perm=0.001,
            risk_aversion=1e-6,
        )
        traj = ac.optimal_trajectory()
        assert abs(traj[0] - 1000.0) < 1e-6, f"traj[0]={traj[0]}, expected 1000"
        assert abs(traj[-1]) < 1e-6, f"traj[-1]={traj[-1]}, expected ~0"


# ---------------------------------------------------------------------------
# AvellanedaStoikovSolution
# ---------------------------------------------------------------------------

class TestAvellanedaStoikovSolution:
    """Tests for Avellaneda-Stoikov market-making."""

    def test_spread_positive(self):
        """Optimal spread must be positive."""
        from quantlaxmi.models.rl.finance import AvellanedaStoikovSolution

        av = AvellanedaStoikovSolution(
            sigma=0.02, gamma_risk=0.1, fill_rate_k=1.5, time_horizon=1.0
        )
        spread = av.optimal_spread(time_remaining=0.5)
        assert spread > 0, f"Spread must be positive, got {spread}"

    def test_bid_below_ask(self):
        """bid < ask for any inventory level."""
        from quantlaxmi.models.rl.finance import AvellanedaStoikovSolution

        av = AvellanedaStoikovSolution(
            sigma=0.02, gamma_risk=0.1, fill_rate_k=1.5
        )
        for inv in [-5, -1, 0, 1, 5]:
            bid, ask = av.optimal_quotes(100.0, inv, 0.5)
            assert bid < ask, (
                f"bid={bid} >= ask={ask} for inventory={inv}"
            )

    def test_positive_inventory_widens_bid(self):
        """Positive inventory -> higher bid offset (wider bid, lower bid price)."""
        from quantlaxmi.models.rl.finance import AvellanedaStoikovSolution

        av = AvellanedaStoikovSolution(
            sigma=0.02, gamma_risk=0.1, fill_rate_k=1.5
        )
        bid_zero, ask_zero = av.optimal_quotes(100.0, 0, 0.5)
        bid_long, ask_long = av.optimal_quotes(100.0, 3, 0.5)

        # When long: bid should be lower (wider bid offset, less keen to buy)
        assert bid_long < bid_zero, (
            f"Long inventory should lower bid: bid_long={bid_long}, bid_zero={bid_zero}"
        )

    def test_negative_inventory_widens_ask(self):
        """Negative inventory -> higher ask offset (wider ask, higher ask price)."""
        from quantlaxmi.models.rl.finance import AvellanedaStoikovSolution

        av = AvellanedaStoikovSolution(
            sigma=0.02, gamma_risk=0.1, fill_rate_k=1.5
        )
        bid_zero, ask_zero = av.optimal_quotes(100.0, 0, 0.5)
        bid_short, ask_short = av.optimal_quotes(100.0, -3, 0.5)

        # When short: ask should be higher (wider ask offset, less keen to sell)
        assert ask_short > ask_zero, (
            f"Short inventory should raise ask: ask_short={ask_short}, ask_zero={ask_zero}"
        )


# ---------------------------------------------------------------------------
# DeepHedger
# ---------------------------------------------------------------------------

class TestDeepHedger:
    """Tests for neural network hedging."""

    def test_construct_cpu(self):
        """DeepHedger can be constructed with device='cpu'."""
        from quantlaxmi.models.rl.finance import DeepHedger

        dh = DeepHedger(
            state_dim=4,
            hidden_layers=(16, 8),
            learning_rate=1e-3,
            device="cpu",
        )
        assert dh.device == "cpu"

    def test_train_on_small_gbm_paths(self):
        """DeepHedger.train runs on 50 GBM paths with 5 steps each."""
        from quantlaxmi.models.rl.finance import DeepHedger

        dh = DeepHedger(
            state_dim=4,
            hidden_layers=(16, 8),
            learning_rate=1e-3,
            device="cpu",
        )

        # Generate small GBM paths: shape (50, 6) = 50 paths, 5 steps + initial
        rng = np.random.default_rng(42)
        num_paths, T = 50, 5
        dt = 1.0 / 252
        sigma, r = 0.2, 0.05
        z = rng.standard_normal((num_paths, T))
        log_rets = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
        log_prices = np.log(100.0) + np.cumsum(log_rets, axis=1)
        prices = np.exp(log_prices)
        paths = np.column_stack([np.full(num_paths, 100.0), prices])

        history = dh.train(
            spot_paths=paths,
            strike=100.0,
            payoff_fn=lambda s: np.maximum(s - 100.0, 0.0),
            num_epochs=2,
            batch_size=50,
            verbose=False,
        )
        assert "epoch_loss" in history
        assert len(history["epoch_loss"]) == 2


# ---------------------------------------------------------------------------
# AssetAllocPG
# ---------------------------------------------------------------------------

class TestAssetAllocPG:
    """Tests for policy gradient asset allocation."""

    def test_construct_cpu(self):
        """AssetAllocPG can be constructed with device='cpu'."""
        from quantlaxmi.models.rl.finance import AssetAllocPG

        pg = AssetAllocPG(
            num_assets=1,
            state_dim=2,
            hidden_layers=(16, 8),
            learning_rate=1e-3,
            device="cpu",
        )
        assert pg.device == "cpu"
        assert pg.num_assets == 1
