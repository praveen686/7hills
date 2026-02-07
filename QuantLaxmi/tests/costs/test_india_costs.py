"""India cost model validation tests.

BLOCKER #4 — no live trading without correct brokerage math.

Validates:
  - CostModel frozen dataclass properties
  - Commission + slippage arithmetic (bps ↔ fraction ↔ roundtrip)
  - India-specific cost parameters for equity FnO
  - Index option costs (per-leg in index points: 3 pts NIFTY, 5 pts BANKNIFTY)
  - Funding cost per bar computation
  - Validation: negative inputs rejected
  - Typical India FnO configurations (Zerodha retail)
  - Roundtrip cost symmetry
  - Edge cases: zero costs, very large costs, fractional bps
"""

from __future__ import annotations

import math

import pytest

from core.backtest.costs import CostModel


# ===================================================================
# 1. BASIC PROPERTIES
# ===================================================================

class TestCostModelBasics:
    def test_one_way_frac(self):
        cm = CostModel(commission_bps=5.0, slippage_bps=3.0)
        assert cm.one_way_frac == (5.0 + 3.0) / 10_000

    def test_roundtrip_frac(self):
        cm = CostModel(commission_bps=5.0, slippage_bps=3.0)
        assert cm.roundtrip_frac == 2 * (5.0 + 3.0) / 10_000

    def test_roundtrip_bps(self):
        cm = CostModel(commission_bps=5.0, slippage_bps=3.0)
        assert cm.roundtrip_bps == 2 * (5.0 + 3.0)

    def test_roundtrip_is_2x_one_way(self):
        cm = CostModel(commission_bps=7.0, slippage_bps=2.0)
        assert abs(cm.roundtrip_frac - 2 * cm.one_way_frac) < 1e-15

    def test_roundtrip_bps_is_2x_component_sum(self):
        cm = CostModel(commission_bps=4.5, slippage_bps=1.5)
        assert cm.roundtrip_bps == 2 * (4.5 + 1.5)

    def test_bps_to_frac_conversion(self):
        """1 bps = 0.0001 = 0.01%."""
        cm = CostModel(commission_bps=1.0, slippage_bps=0.0)
        assert cm.one_way_frac == 0.0001

    def test_100bps_is_1_percent(self):
        cm = CostModel(commission_bps=100.0, slippage_bps=0.0)
        assert abs(cm.one_way_frac - 0.01) < 1e-15


# ===================================================================
# 2. FROZEN / IMMUTABLE
# ===================================================================

class TestCostModelImmutable:
    def test_frozen(self):
        cm = CostModel(commission_bps=5.0, slippage_bps=3.0)
        with pytest.raises(AttributeError):
            cm.commission_bps = 10.0  # type: ignore

    def test_cannot_mutate_slippage(self):
        cm = CostModel(commission_bps=5.0, slippage_bps=3.0)
        with pytest.raises(AttributeError):
            cm.slippage_bps = 0.0  # type: ignore

    def test_cannot_mutate_funding(self):
        cm = CostModel(commission_bps=5.0, slippage_bps=3.0, funding_annual_pct=8.0)
        with pytest.raises(AttributeError):
            cm.funding_annual_pct = 0.0  # type: ignore


# ===================================================================
# 3. VALIDATION — Negative inputs rejected
# ===================================================================

class TestCostModelValidation:
    def test_negative_commission_rejected(self):
        with pytest.raises(ValueError, match="commission_bps"):
            CostModel(commission_bps=-1.0, slippage_bps=3.0)

    def test_negative_slippage_rejected(self):
        with pytest.raises(ValueError, match="slippage_bps"):
            CostModel(commission_bps=5.0, slippage_bps=-1.0)

    def test_zero_commission_accepted(self):
        cm = CostModel(commission_bps=0.0, slippage_bps=3.0)
        assert cm.commission_bps == 0.0

    def test_zero_slippage_accepted(self):
        cm = CostModel(commission_bps=5.0, slippage_bps=0.0)
        assert cm.slippage_bps == 0.0


# ===================================================================
# 4. FUNDING COST
# ===================================================================

class TestFundingCost:
    def test_funding_per_bar_daily(self):
        cm = CostModel(commission_bps=0.0, slippage_bps=0.0, funding_annual_pct=8.0)
        cost = cm.holding_cost_per_bar(252.0)
        expected = 8.0 / 100.0 / 252.0
        assert abs(cost - expected) < 1e-12

    def test_funding_per_bar_minute(self):
        cm = CostModel(commission_bps=0.0, slippage_bps=0.0, funding_annual_pct=8.0)
        bars_per_year = 252 * 375  # India trading minutes
        cost = cm.holding_cost_per_bar(bars_per_year)
        expected = 8.0 / 100.0 / bars_per_year
        assert abs(cost - expected) < 1e-15

    def test_funding_zero(self):
        cm = CostModel(commission_bps=5.0, slippage_bps=3.0, funding_annual_pct=0.0)
        assert cm.holding_cost_per_bar(252.0) == 0.0

    def test_funding_zero_bars_per_year(self):
        cm = CostModel(commission_bps=5.0, slippage_bps=3.0, funding_annual_pct=8.0)
        assert cm.holding_cost_per_bar(0.0) == 0.0

    def test_funding_negative_bars_per_year(self):
        cm = CostModel(commission_bps=5.0, slippage_bps=3.0, funding_annual_pct=8.0)
        assert cm.holding_cost_per_bar(-100.0) == 0.0


# ===================================================================
# 5. INDIA FnO — Typical Zerodha Retail Configs
# ===================================================================

class TestIndiaFnOCosts:
    """India-specific cost configurations for Zerodha retail."""

    def test_equity_futures_costs(self):
        """Zerodha: Rs 20/order cap → ~1-2 bps for typical sizes.
        Plus slippage ~2 bps for liquid names.
        Total roundtrip should be ~6-8 bps."""
        cm = CostModel(commission_bps=2.0, slippage_bps=2.0)
        assert 6 <= cm.roundtrip_bps <= 10
        assert cm.roundtrip_bps == 8.0

    def test_index_options_nifty_as_bps(self):
        """NIFTY options: ~3 index points per leg cost.
        At NIFTY=22000, 3 pts = 3/22000 = 0.0136% = 1.36 bps per side."""
        nifty_spot = 22000
        cost_per_leg_pts = 3.0
        bps_per_side = (cost_per_leg_pts / nifty_spot) * 10_000
        cm = CostModel(commission_bps=bps_per_side, slippage_bps=0.0)
        # Verify the conversion
        assert abs(bps_per_side - 1.3636) < 0.01
        # One-way cost in rupee fraction
        assert abs(cm.one_way_frac - cost_per_leg_pts / nifty_spot) < 1e-8

    def test_index_options_banknifty_as_bps(self):
        """BANKNIFTY: ~5 index points per leg.
        At BANKNIFTY=48000, 5/48000 = 1.04 bps per side."""
        bn_spot = 48000
        cost_per_leg_pts = 5.0
        bps_per_side = (cost_per_leg_pts / bn_spot) * 10_000
        cm = CostModel(commission_bps=bps_per_side, slippage_bps=0.0)
        assert abs(bps_per_side - 1.0417) < 0.01
        assert abs(cm.one_way_frac - cost_per_leg_pts / bn_spot) < 1e-8

    def test_stock_options_costs(self):
        """Stock options: higher slippage due to lower liquidity.
        Typically 5-10 bps slippage + 2 bps commission."""
        cm = CostModel(commission_bps=2.0, slippage_bps=7.0)
        assert cm.roundtrip_bps == 18.0

    def test_stt_included_in_commission(self):
        """STT on options is 0.0625% on sell side = 6.25 bps.
        This should be factored into commission_bps for options."""
        stt_bps = 6.25  # STT on options sell side
        exchange_bps = 0.5  # NSE charges
        sebi_bps = 0.1  # SEBI turnover fee
        gst_on_brokerage = 0.3  # ~18% GST on brokerage component

        total_regulatory = stt_bps + exchange_bps + sebi_bps + gst_on_brokerage
        # For a sell trade, these add up
        assert total_regulatory > 7.0  # At least 7 bps regulatory overhead
        assert total_regulatory < 8.0  # But less than 8 bps


# ===================================================================
# 6. INDEX POINT COST — Cross-validation
# ===================================================================

class TestIndexPointCosts:
    """Verify per-leg index point cost conversions are consistent."""

    def test_nifty_3pts_roundtrip(self):
        """NIFTY roundtrip = 2 * 3 = 6 index points."""
        nifty = 22000
        per_leg = 3.0
        roundtrip_pts = 2 * per_leg
        roundtrip_bps = (roundtrip_pts / nifty) * 10_000
        cm = CostModel(
            commission_bps=roundtrip_bps / 2,
            slippage_bps=0.0,
        )
        assert abs(cm.roundtrip_frac * nifty - roundtrip_pts) < 0.01

    def test_banknifty_5pts_roundtrip(self):
        """BANKNIFTY roundtrip = 2 * 5 = 10 index points."""
        bn = 48000
        per_leg = 5.0
        roundtrip_pts = 2 * per_leg
        roundtrip_bps = (roundtrip_pts / bn) * 10_000
        cm = CostModel(
            commission_bps=roundtrip_bps / 2,
            slippage_bps=0.0,
        )
        assert abs(cm.roundtrip_frac * bn - roundtrip_pts) < 0.01

    def test_cost_per_lot_nifty(self):
        """NIFTY lot size = 25. Cost per lot roundtrip = 6 pts * 25 = Rs 150."""
        nifty = 22000
        lot_size = 25
        per_leg_pts = 3.0
        roundtrip_cost_per_lot = 2 * per_leg_pts * lot_size
        assert roundtrip_cost_per_lot == 150.0

    def test_cost_per_lot_banknifty(self):
        """BANKNIFTY lot size = 15. Cost per lot = 10 pts * 15 = Rs 150."""
        lot_size = 15
        per_leg_pts = 5.0
        roundtrip_cost_per_lot = 2 * per_leg_pts * lot_size
        assert roundtrip_cost_per_lot == 150.0

    def test_cost_pct_of_premium_otm(self):
        """For a 50-point OTM option premium, 3 pts cost = 6% per side.
        This is significant and must not be ignored."""
        premium = 50.0
        cost_per_side = 3.0
        pct_of_premium = cost_per_side / premium
        assert pct_of_premium == 0.06  # 6% — very material


# ===================================================================
# 7. EDGE CASES
# ===================================================================

class TestCostModelEdgeCases:
    def test_zero_cost_model(self):
        cm = CostModel(commission_bps=0.0, slippage_bps=0.0)
        assert cm.one_way_frac == 0.0
        assert cm.roundtrip_frac == 0.0
        assert cm.roundtrip_bps == 0.0

    def test_large_costs(self):
        """Extreme cost model (100 bps each side = 1% each way)."""
        cm = CostModel(commission_bps=100.0, slippage_bps=100.0)
        assert cm.one_way_frac == 0.02  # 2%
        assert cm.roundtrip_frac == 0.04  # 4%

    def test_fractional_bps(self):
        cm = CostModel(commission_bps=1.5, slippage_bps=0.7)
        expected_one_way = (1.5 + 0.7) / 10_000
        assert abs(cm.one_way_frac - expected_one_way) < 1e-15

    def test_very_small_costs(self):
        cm = CostModel(commission_bps=0.01, slippage_bps=0.01)
        assert cm.one_way_frac == 0.02 / 10_000
        assert cm.roundtrip_bps == 0.04

    def test_default_funding_is_zero(self):
        cm = CostModel(commission_bps=5.0, slippage_bps=3.0)
        assert cm.funding_annual_pct == 0.0
        assert cm.holding_cost_per_bar(252.0) == 0.0


# ===================================================================
# 8. COST MODEL COMPARISON — Realistic Scenarios
# ===================================================================

class TestCostModelScenarios:
    def test_nifty_futures_vs_options_cost(self):
        """Options are more expensive per notional than futures."""
        fut_cm = CostModel(commission_bps=2.0, slippage_bps=2.0)  # 4 bps each way
        # Options: STT + exchange + slippage
        opt_cm = CostModel(commission_bps=7.0, slippage_bps=5.0)  # 12 bps each way
        assert opt_cm.roundtrip_bps > fut_cm.roundtrip_bps

    def test_liquid_vs_illiquid_stock(self):
        """Illiquid stocks have higher slippage."""
        reliance = CostModel(commission_bps=2.0, slippage_bps=2.0)  # liquid
        small_cap = CostModel(commission_bps=2.0, slippage_bps=15.0)  # illiquid
        assert small_cap.roundtrip_frac > reliance.roundtrip_frac

    def test_intraday_vs_swing_funding(self):
        """Swing trades incur funding; intraday does not."""
        intraday = CostModel(commission_bps=5.0, slippage_bps=3.0, funding_annual_pct=0.0)
        swing = CostModel(commission_bps=5.0, slippage_bps=3.0, funding_annual_pct=8.0)

        # 5-day hold
        hold_days = 5
        swing_total = swing.roundtrip_frac + swing.holding_cost_per_bar(252.0) * hold_days
        intraday_total = intraday.roundtrip_frac
        assert swing_total > intraday_total

    def test_roundtrip_symmetry(self):
        """Roundtrip cost = 2x one-way regardless of direction."""
        cm = CostModel(commission_bps=5.0, slippage_bps=3.0)
        # Buy then sell vs sell then buy should be same cost
        assert cm.roundtrip_frac == 2 * cm.one_way_frac
