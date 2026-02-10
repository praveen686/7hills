"""Tests for cost model and backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlaxmi.core.backtest.costs import CostModel
from quantlaxmi.core.backtest.portfolio import BacktestResult, run_backtest


class TestCostModel:
    def test_roundtrip_calculation(self):
        cm = CostModel(commission_bps=10, slippage_bps=5)
        assert cm.roundtrip_bps == 30.0
        assert cm.roundtrip_frac == pytest.approx(0.003)

    def test_one_way_fraction(self):
        cm = CostModel(commission_bps=10, slippage_bps=5)
        assert cm.one_way_frac == pytest.approx(0.0015)

    def test_frozen(self):
        cm = CostModel(commission_bps=10, slippage_bps=5)
        with pytest.raises(AttributeError):
            cm.commission_bps = 20  # type: ignore

    def test_negative_commission_raises(self):
        with pytest.raises(ValueError, match="commission"):
            CostModel(commission_bps=-1, slippage_bps=5)

    def test_holding_cost(self):
        cm = CostModel(commission_bps=10, slippage_bps=5, funding_annual_pct=10.0)
        # 365 * 24 bars per year for hourly data
        cost = cm.holding_cost_per_bar(365 * 24)
        assert cost > 0
        assert cost == pytest.approx(0.10 / (365 * 24))


class TestRunBacktest:
    @pytest.fixture
    def simple_data(self):
        """100 bars of trending-up price data."""
        idx = pd.date_range("2023-01-01", periods=100, freq="h")
        prices = pd.Series(
            30000 + np.arange(100) * 10.0,  # steady uptrend
            index=idx,
        )
        return prices

    def test_always_long_profitable_in_uptrend(self, simple_data):
        """In a perfect uptrend with no costs, always-long should profit."""
        preds = pd.Series(0.5, index=simple_data.index)  # always above threshold
        costs = CostModel(commission_bps=0, slippage_bps=0)
        result = run_backtest(
            prices=simple_data,
            predictions=preds,
            cost_model=costs,
            long_entry_th=0.1,
        )
        assert result.total_return > 0

    def test_costs_reduce_returns(self, simple_data):
        preds = pd.Series(0.5, index=simple_data.index)
        result_free = run_backtest(
            simple_data, preds,
            CostModel(commission_bps=0, slippage_bps=0),
            long_entry_th=0.1,
        )
        result_costly = run_backtest(
            simple_data, preds,
            CostModel(commission_bps=50, slippage_bps=50),
            long_entry_th=0.1,
        )
        assert result_costly.total_return < result_free.total_return

    def test_no_trades_when_threshold_high(self, simple_data):
        preds = pd.Series(0.01, index=simple_data.index)
        costs = CostModel(commission_bps=10, slippage_bps=5)
        result = run_backtest(
            simple_data, preds, costs,
            long_entry_th=0.99,  # impossibly high
            short_entry_th=-0.99,
        )
        assert result.n_trades == 0

    def test_backtest_result_is_frozen(self, simple_data):
        preds = pd.Series(0.5, index=simple_data.index)
        costs = CostModel(commission_bps=10, slippage_bps=5)
        result = run_backtest(simple_data, preds, costs, long_entry_th=0.1)
        with pytest.raises(AttributeError):
            result.total_return = 999  # type: ignore

    def test_equity_curve_length(self, simple_data):
        preds = pd.Series(0.5, index=simple_data.index)
        costs = CostModel(commission_bps=10, slippage_bps=5)
        result = run_backtest(simple_data, preds, costs, long_entry_th=0.1)
        assert len(result.equity_curve) == len(simple_data)

    def test_trades_have_costs(self, simple_data):
        """Every trade should have non-zero cost when cost model is non-zero."""
        # Create predictions that trigger some trading
        preds = pd.Series(
            [0.5 if i % 20 < 10 else -0.5 for i in range(100)],
            index=simple_data.index,
        )
        costs = CostModel(commission_bps=10, slippage_bps=5)
        result = run_backtest(
            simple_data, preds, costs,
            long_entry_th=0.1, long_exit_th=0.0,
            short_entry_th=-0.1, short_exit_th=0.0,
        )
        if result.n_trades > 0:
            assert (result.trades["cost"] > 0).all()
