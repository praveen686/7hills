"""Tests for models.rl.environments package.

Tests the trading MDP environments:
  - TradingState: universal state representation
  - SimulatedPriceEnv: parametric price dynamics
  - IndiaFnOEnv: India Futures & Options
  - CryptoEnv: Binance crypto
  - OptionsEnv: options trading with Greeks
  - ExecutionEnv: order execution with LOB
  - LOBSimulator: limit order book
"""
import sys
sys.path.insert(0, "/home/ubuntu/Desktop/7hills/QuantLaxmi")

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# TradingState
# ---------------------------------------------------------------------------

class TestTradingState:
    """Tests for TradingState.to_array()."""

    def test_to_array_returns_numpy(self):
        """to_array() must return a numpy array."""
        from models.rl.environments import TradingState

        state = TradingState(
            timestamp=0,
            prices=np.array([100.0]),
            position=np.array([0.0]),
            cash=1_000_000.0,
            pnl=0.0,
            features={"vol": 0.2},
        )
        arr = state.to_array()
        assert isinstance(arr, np.ndarray), f"Expected ndarray, got {type(arr)}"
        assert arr.dtype == np.float64


# ---------------------------------------------------------------------------
# SimulatedPriceEnv
# ---------------------------------------------------------------------------

class TestSimulatedPriceEnv:
    """Tests for SimulatedPriceEnv reset/step cycle."""

    def test_reset_returns_trading_state(self):
        """reset() must return a TradingState."""
        from models.rl.environments import SimulatedPriceEnv, TradingState

        env = SimulatedPriceEnv(
            dynamics="gbm",
            num_steps=10,
            num_assets=1,
            seed=42,
        )
        state = env.reset()
        assert isinstance(state, TradingState)

    def test_step_returns_step_result(self):
        """step() must return a StepResult."""
        from models.rl.environments import (
            SimulatedPriceEnv,
            TradingAction,
            StepResult,
        )

        env = SimulatedPriceEnv(
            dynamics="gbm",
            num_steps=10,
            num_assets=1,
            seed=42,
        )
        env.reset()
        action = TradingAction(trade_sizes=np.array([0.0]))
        result = env.step(action)
        assert isinstance(result, StepResult)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)


# ---------------------------------------------------------------------------
# IndiaFnOEnv
# ---------------------------------------------------------------------------

class TestIndiaFnOEnv:
    """Tests for India FnO environment."""

    def test_lot_size_nifty(self):
        """NIFTY lot size must be 25."""
        from models.rl.environments import IndiaFnOEnv

        assert IndiaFnOEnv.get_lot_size("NIFTY") == 25

    def test_lot_size_banknifty(self):
        """BANKNIFTY lot size must be 15."""
        from models.rl.environments import IndiaFnOEnv

        assert IndiaFnOEnv.get_lot_size("BANKNIFTY") == 15

    def test_cost_per_leg_nifty(self):
        """NIFTY cost per leg must be 3.0 index points."""
        from models.rl.environments import IndiaFnOEnv

        assert IndiaFnOEnv.get_cost_per_leg("NIFTY") == 3.0

    def test_cost_per_leg_banknifty(self):
        """BANKNIFTY cost per leg must be 5.0 index points."""
        from models.rl.environments import IndiaFnOEnv

        assert IndiaFnOEnv.get_cost_per_leg("BANKNIFTY") == 5.0


# ---------------------------------------------------------------------------
# CryptoEnv
# ---------------------------------------------------------------------------

class TestCryptoEnv:
    """Tests for CryptoEnv reset/step cycle."""

    def test_reset_step_cycle(self):
        """CryptoEnv reset + step cycle must work."""
        from models.rl.environments import CryptoEnv, TradingAction, StepResult

        env = CryptoEnv(
            symbol="BTCUSDT",
            num_steps=5,
            seed=42,
        )
        state = env.reset()
        assert state.prices.shape == (1,)

        action = TradingAction(trade_sizes=np.array([0.0]))
        result = env.step(action)
        assert isinstance(result, StepResult)
        assert isinstance(result.done, bool)


# ---------------------------------------------------------------------------
# OptionsEnv
# ---------------------------------------------------------------------------

class TestOptionsEnv:
    """Tests for OptionsEnv."""

    def test_reset_works(self):
        """OptionsEnv.reset() must return TradingState."""
        from models.rl.environments import OptionsEnv
        from models.rl.environments.trading_env import TradingState

        env = OptionsEnv(
            spot_init=100.0,
            expiry_days=5,
            sigma=0.20,
            num_steps_per_day=1,
            seed=42,
        )
        state = env.reset()
        assert isinstance(state, TradingState)
        assert state.prices[0] == 100.0

    def test_step_returns_step_result(self):
        """OptionsEnv.step() must return StepResult."""
        from models.rl.environments import OptionsEnv
        from models.rl.environments.trading_env import TradingAction, StepResult

        env = OptionsEnv(
            spot_init=100.0,
            expiry_days=5,
            sigma=0.20,
            num_steps_per_day=1,
            seed=42,
        )
        env.reset()
        action = TradingAction(trade_sizes=np.array([0.0, 0.0]))
        result = env.step(action)
        assert isinstance(result, StepResult)


# ---------------------------------------------------------------------------
# ExecutionEnv
# ---------------------------------------------------------------------------

class TestExecutionEnv:
    """Tests for ExecutionEnv reset/step cycle."""

    def test_reset_step_works(self):
        """ExecutionEnv reset + step must work."""
        from models.rl.environments import ExecutionEnv
        from models.rl.environments.trading_env import (
            TradingState,
            TradingAction,
            StepResult,
        )

        env = ExecutionEnv(
            total_shares=100,
            num_steps=10,
            price_init=100.0,
            seed=42,
        )
        state = env.reset()
        assert isinstance(state, TradingState)

        action = TradingAction(trade_sizes=np.array([0.5, 0.5]))
        result = env.step(action)
        assert isinstance(result, StepResult)


# ---------------------------------------------------------------------------
# LOBSimulator
# ---------------------------------------------------------------------------

class TestLOBSimulator:
    """Tests for LOBSimulator market orders."""

    def test_market_order_returns_fill(self):
        """submit_market_order must return (fill_price, filled_qty)."""
        from models.rl.environments import LOBSimulator

        lob = LOBSimulator(
            price=100.0,
            spread=0.01,
            depth_mean=100.0,
            seed=42,
        )
        fill_price, filled_qty = lob.submit_market_order(10)
        assert isinstance(fill_price, float)
        assert isinstance(filled_qty, float)
        assert filled_qty > 0, "Market order should fill at least partially"
        assert fill_price > 0, "Fill price should be positive"
