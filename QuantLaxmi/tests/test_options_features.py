"""Tests for features/options_features.py — 7 comprehensive tests.

Covers:
    1. Newton-Raphson IV convergence for ATM call
    2. Newton-Raphson IV edge cases (intrinsic-only, zero T, deeply OTM)
    3. BS Greeks put-call parity
    4. BS delta bounds
    5. OptionsFeatureBuilder columns (mock store)
    6. Causality — features on date D use only data from <= D
    7. Max pain computation with analytically solvable chain
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm as _norm

from quantlaxmi.features.options_features import (
    OptionsFeatureBuilder,
    _bs_delta,
    _bs_gamma,
    _bs_price,
    _bs_theta,
    _bs_vega,
    _newton_iv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chain_df(date_str: str, spot: float, expiry_str: str, strikes=None):
    """Build a minimal FnO chain DataFrame for one date.

    Creates CE + PE rows for each strike, with synthetic BS prices at 20% vol.
    """
    if strikes is None:
        strikes = [spot * m for m in [0.92, 0.95, 0.98, 1.00, 1.02, 1.05, 1.08]]
    dt_date = pd.Timestamp(date_str)
    exp_date = pd.Timestamp(expiry_str)
    T = max(1, (exp_date - dt_date).days) / 365.0
    r = 0.065
    sigma = 0.20

    rows = []
    for K in strikes:
        for opt_type, is_call in [("CE", True), ("PE", False)]:
            price = _bs_price(spot, K, T, r, sigma, is_call)
            gamma = _bs_gamma(spot, K, T, r, sigma)
            delta = abs(_bs_delta(spot, K, T, r, sigma, is_call))
            # More OI near ATM, tapering away
            oi = max(1000.0, 50000.0 * math.exp(-((K - spot) / spot) ** 2 / 0.002))
            rows.append({
                "date": date_str,
                "symbol": "NIFTY",
                "instr_type": "IDO",
                "strike": K,
                "option_type": opt_type,
                "close": max(price, 0.5),  # floor to avoid 0
                "settle": max(price, 0.5),
                "volume": oi * 0.1,
                "oi": oi,
                "expiry": expiry_str,
                "lot_size": 50.0,
            })
    return pd.DataFrame(rows)


def _make_two_expiry_chain(date_str, spot, exp1, exp2, strikes=None):
    """Chain with 2 expiries (needed for term structure features)."""
    df1 = _make_chain_df(date_str, spot, exp1, strikes)
    df2 = _make_chain_df(date_str, spot, exp2, strikes)
    return pd.concat([df1, df2], ignore_index=True)


@pytest.fixture
def mock_store_single_date():
    """Mock store returning chain for a single date 2025-01-10, spot=20000."""
    store = MagicMock()

    spot = 20000.0
    date_str = "2025-01-10"
    exp1 = "2025-01-30"
    exp2 = "2025-02-27"
    strikes = [19200, 19500, 19800, 20000, 20200, 20500, 20800]
    chain = _make_two_expiry_chain(date_str, spot, exp1, exp2, strikes)

    # spot_series load (called first)
    spot_df = pd.DataFrame({
        "date": [f"2025-01-{d:02d}" for d in range(1, 11)],
        "close": [19800, 19850, 19900, 19920, 19950, 19980, 20000, 20010, 20020, 20000],
    })

    def sql_side_effect(query, params=None):
        if "nse_index_close" in query:
            return spot_df
        if "nse_fo_bhavcopy" in query:
            return chain
        return pd.DataFrame()

    store.sql = MagicMock(side_effect=sql_side_effect)
    return store


@pytest.fixture
def mock_store_two_dates():
    """Mock store with chains for 2025-01-10 and 2025-01-11 (for causality test)."""
    store = MagicMock()

    spot_d1 = 20000.0
    spot_d2 = 20100.0
    exp1 = "2025-01-30"
    exp2 = "2025-02-27"
    strikes = [19500, 19800, 20000, 20200, 20500]

    chain_d1 = _make_two_expiry_chain("2025-01-10", spot_d1, exp1, exp2, strikes)
    chain_d2 = _make_two_expiry_chain("2025-01-11", spot_d2, exp1, exp2, strikes)
    full_chain = pd.concat([chain_d1, chain_d2], ignore_index=True)

    spot_df = pd.DataFrame({
        "date": [f"2025-01-{d:02d}" for d in range(1, 12)],
        "close": [19800, 19850, 19900, 19920, 19950, 19980, 20000, 20010, 20020, 20000, 20100],
    })

    def sql_side_effect(query, params=None):
        if "nse_index_close" in query:
            return spot_df
        if "nse_fo_bhavcopy" in query:
            return full_chain
        return pd.DataFrame()

    store.sql = MagicMock(side_effect=sql_side_effect)
    return store


# ============================================================================
# Test 1: Newton-Raphson IV convergence
# ============================================================================


def test_newton_iv_convergence():
    """Newton-Raphson IV inverts a known ATM call price to recover sigma=0.20."""
    S, K, T, r, sigma_true = 100.0, 100.0, 0.25, 0.05, 0.20
    market_price = _bs_price(S, K, T, r, sigma_true, is_call=True)
    assert market_price > 0, "ATM call price must be positive"

    recovered_iv = _newton_iv(market_price, S, K, T, r, is_call=True)

    assert not np.isnan(recovered_iv), "IV should converge, not return NaN"
    assert abs(recovered_iv - sigma_true) < 1e-4, (
        f"Recovered IV {recovered_iv:.6f} should be within 1e-4 of {sigma_true}"
    )

    # Also verify the round-trip: BS price at recovered IV matches market price
    repriced = _bs_price(S, K, T, r, recovered_iv, is_call=True)
    assert abs(repriced - market_price) < 1e-4, (
        f"Repriced {repriced:.6f} should match market {market_price:.6f}"
    )


# ============================================================================
# Test 2: Newton-Raphson IV edge cases
# ============================================================================


def test_newton_iv_edge_cases():
    """Edge cases: intrinsic-only, zero T, deeply OTM with tiny price."""
    S, K, T, r = 100.0, 100.0, 0.25, 0.05

    # --- Intrinsic-only price ---
    # For a call with K=90, intrinsic is S-K = 10. If market_price == intrinsic,
    # there's no time value, so IV should be NaN (market_price < intrinsic + 1e-4).
    intrinsic_call = max(0.0, S - 90.0)  # = 10.0
    iv_intrinsic = _newton_iv(intrinsic_call, S, 90.0, T, r, is_call=True)
    assert np.isnan(iv_intrinsic), "Intrinsic-only price should yield NaN IV"

    # --- Zero time to expiry ---
    iv_zero_t = _newton_iv(5.0, S, K, 0.0, r, is_call=True)
    assert np.isnan(iv_zero_t), "Zero T should yield NaN IV"

    iv_neg_t = _newton_iv(5.0, S, K, -0.01, r, is_call=True)
    assert np.isnan(iv_neg_t), "Negative T should yield NaN IV"

    # --- Deeply OTM with small price ---
    # K=130 (30% OTM), very small price: should either converge to a high vol or NaN
    small_price = 0.05
    iv_otm = _newton_iv(small_price, S, 130.0, T, r, is_call=True)
    # If it converges, the IV should be large (>50%) because the option is far OTM
    if not np.isnan(iv_otm):
        assert iv_otm > 0.3, f"Deeply OTM IV should be high, got {iv_otm:.4f}"
        # Verify price round-trip
        repriced = _bs_price(S, 130.0, T, r, iv_otm, is_call=True)
        assert abs(repriced - small_price) < 0.01

    # --- Zero or negative spot ---
    iv_zero_s = _newton_iv(5.0, 0.0, K, T, r, is_call=True)
    assert np.isnan(iv_zero_s), "Zero spot should yield NaN IV"


# ============================================================================
# Test 3: Put-Call Parity
# ============================================================================


def test_bs_greeks_put_call_parity():
    """Verify put-call parity: C - P = S - K*exp(-rT)."""
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25

    C = _bs_price(S, K, T, r, sigma, is_call=True)
    P = _bs_price(S, K, T, r, sigma, is_call=False)

    lhs = C - P
    rhs = S - K * math.exp(-r * T)

    assert abs(lhs - rhs) < 1e-10, (
        f"Put-call parity violated: C-P={lhs:.10f}, S-Ke^(-rT)={rhs:.10f}, "
        f"diff={abs(lhs - rhs):.2e}"
    )

    # Also check with OTM parameters
    for K_test in [80.0, 90.0, 110.0, 120.0]:
        C2 = _bs_price(S, K_test, T, r, sigma, is_call=True)
        P2 = _bs_price(S, K_test, T, r, sigma, is_call=False)
        lhs2 = C2 - P2
        rhs2 = S - K_test * math.exp(-r * T)
        assert abs(lhs2 - rhs2) < 1e-10, (
            f"Put-call parity violated for K={K_test}: diff={abs(lhs2 - rhs2):.2e}"
        )


# ============================================================================
# Test 4: Delta bounds
# ============================================================================


def test_bs_delta_bounds():
    """Call delta in [0, 1], put delta in [-1, 0] for all strikes/vols/maturities."""
    S = 100.0
    r = 0.05

    test_params = [
        # (K, T, sigma) — various moneyness, maturity, vol
        (80.0, 0.5, 0.20),   # deep ITM call / deep OTM put
        (100.0, 0.5, 0.20),  # ATM
        (120.0, 0.5, 0.20),  # deep OTM call / deep ITM put
        (100.0, 0.01, 0.20), # very short dated
        (100.0, 2.0, 0.20),  # long dated
        (100.0, 0.5, 0.05),  # low vol
        (100.0, 0.5, 1.50),  # high vol
        (50.0, 0.25, 0.30),  # far ITM call
        (200.0, 0.25, 0.30), # far OTM call
    ]

    for K, T, sigma in test_params:
        delta_call = _bs_delta(S, K, T, r, sigma, is_call=True)
        delta_put = _bs_delta(S, K, T, r, sigma, is_call=False)

        assert 0.0 <= delta_call <= 1.0, (
            f"Call delta {delta_call:.6f} out of [0,1] for K={K}, T={T}, sigma={sigma}"
        )
        assert -1.0 <= delta_put <= 0.0, (
            f"Put delta {delta_put:.6f} out of [-1,0] for K={K}, T={T}, sigma={sigma}"
        )

        # Call delta - put delta = 1 (for European options with no dividends)
        # This is the delta form of put-call parity
        if T > 0.001 and sigma > 0.001:
            # For T>0 and sigma>0, N(d1) - (N(d1)-1) = 1
            assert abs((delta_call - delta_put) - 1.0) < 1e-10, (
                f"Delta_call - delta_put should be 1.0, got "
                f"{delta_call - delta_put:.10f} for K={K}"
            )


# ============================================================================
# Test 5: OptionsFeatureBuilder columns
# ============================================================================

_EXPECTED_OPTX_COLUMNS = [
    "optx_atm_iv",
    "optx_iv_skew_25d",
    "optx_pcr_vol",
    "optx_pcr_oi",
    "optx_term_slope",
    "optx_vrp",
    "optx_net_gamma",
    "optx_theta_rate",
    "optx_put_wall",
    "optx_call_wall",
    "optx_max_pain_dist",
    "optx_iv_rv_ratio",
    "optx_iv_term_contango",
    "optx_oi_pcr_zscore_21d",
    "optx_skew_zscore_21d",
    "optx_gamma_zscore_21d",
]


def test_options_feature_builder_columns(mock_store_single_date):
    """OptionsFeatureBuilder.build() returns DataFrame with all 16 optx_ columns."""
    builder = OptionsFeatureBuilder(risk_free_rate=0.065, rv_window=5, zscore_window=3)
    df = builder.build(
        ticker="NIFTY",
        start_date="2025-01-10",
        end_date="2025-01-10",
        store=mock_store_single_date,
    )

    assert not df.empty, "build() should return non-empty DataFrame"
    assert len(df) >= 1, "Should have at least 1 row"

    actual_cols = set(df.columns)
    for col in _EXPECTED_OPTX_COLUMNS:
        assert col in actual_cols, f"Missing expected column: {col}"

    # Verify no leftover _raw_ columns leaked through
    raw_cols = [c for c in df.columns if c.startswith("_raw_")]
    assert len(raw_cols) == 0, f"Raw columns should be dropped, but found: {raw_cols}"

    # All columns should be numeric
    for col in _EXPECTED_OPTX_COLUMNS:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"

    # ATM IV should be reasonable (close to the 20% we used in synthetic prices)
    atm_iv = df["optx_atm_iv"].iloc[0]
    if not np.isnan(atm_iv):
        assert 0.05 < atm_iv < 1.0, f"ATM IV {atm_iv} out of plausible range"


# ============================================================================
# Test 6: Causality — features on date D use only data from <= D
# ============================================================================


def test_options_feature_builder_causal(mock_store_two_dates):
    """Features on 2025-01-10 must not change when 2025-01-11 data is added."""
    builder = OptionsFeatureBuilder(risk_free_rate=0.065, rv_window=5, zscore_window=3)

    # Build for both dates
    df_both = builder.build(
        ticker="NIFTY",
        start_date="2025-01-10",
        end_date="2025-01-11",
        store=mock_store_two_dates,
    )

    assert not df_both.empty, "Should have data for both dates"

    # Now build for only the first date
    # Create a store that only has data for the first date
    store_d1_only = MagicMock()
    spot_d1 = 20000.0
    exp1 = "2025-01-30"
    exp2 = "2025-02-27"
    strikes = [19500, 19800, 20000, 20200, 20500]
    chain_d1 = _make_two_expiry_chain("2025-01-10", spot_d1, exp1, exp2, strikes)

    spot_df_d1 = pd.DataFrame({
        "date": [f"2025-01-{d:02d}" for d in range(1, 11)],
        "close": [19800, 19850, 19900, 19920, 19950, 19980, 20000, 20010, 20020, 20000],
    })

    def sql_d1_only(query, params=None):
        if "nse_index_close" in query:
            return spot_df_d1
        if "nse_fo_bhavcopy" in query:
            return chain_d1
        return pd.DataFrame()

    store_d1_only.sql = MagicMock(side_effect=sql_d1_only)

    df_d1_only = builder.build(
        ticker="NIFTY",
        start_date="2025-01-10",
        end_date="2025-01-10",
        store=store_d1_only,
    )

    assert not df_d1_only.empty, "Should have data for D1"

    # The features for 2025-01-10 should be identical whether or not D2 data exists.
    # This verifies causality: D1 features are not polluted by D2 data.
    d1_row_both = df_both.loc[df_both.index == "2025-01-10"]
    d1_row_single = df_d1_only

    if not d1_row_both.empty and not d1_row_single.empty:
        # Compare non-NaN columns that don't depend on z-score warmup
        non_zscore_cols = [
            c for c in _EXPECTED_OPTX_COLUMNS if "zscore" not in c
        ]
        for col in non_zscore_cols:
            val_both = d1_row_both[col].iloc[0]
            val_single = d1_row_single[col].iloc[0]
            if np.isnan(val_both) and np.isnan(val_single):
                continue  # Both NaN is fine
            assert abs(val_both - val_single) < 1e-8, (
                f"Causality violated for {col}: "
                f"with D2={val_both:.8f}, without D2={val_single:.8f}"
            )


# ============================================================================
# Test 7: Max pain computation
# ============================================================================


def test_max_pain_computation():
    """Test _compute_max_pain_distance with an analytically solvable chain.

    Setup:
        Strikes: 90, 100, 110
        Call OI:  100 at K=90, 200 at K=100, 300 at K=110
        Put OI:   300 at K=90, 200 at K=100, 100 at K=110
        Spot = 100

    At settle=90:
      call_pain = 100*max(90-90,0) + 200*max(90-100,0) + 300*max(90-110,0) = 0
      put_pain  = 300*max(90-90,0) + 200*max(100-90,0) + 100*max(110-90,0) = 0+2000+2000 = 4000
      total = 4000

    At settle=100:
      call_pain = 100*max(100-90,0) + 200*max(100-100,0) + 300*max(100-110,0) = 1000+0+0 = 1000
      put_pain  = 300*max(90-100,0) + 200*max(100-100,0) + 100*max(110-100,0) = 0+0+1000 = 1000
      total = 2000

    At settle=110:
      call_pain = 100*max(110-90,0) + 200*max(110-100,0) + 300*max(110-110,0) = 2000+2000+0 = 4000
      put_pain  = 300*max(90-110,0) + 200*max(100-110,0) + 100*max(110-110,0) = 0+0+0 = 0
      total = 4000

    Min pain at settle=100, so max_pain_strike=100, distance = (100-100)/100*100 = 0%.
    """
    chain = pd.DataFrame([
        {"strike": 90.0, "option_type": "CE", "oi": 100.0, "close": 12.0,
         "volume": 50, "settle": 12.0, "expiry": "2025-01-30", "lot_size": 50},
        {"strike": 100.0, "option_type": "CE", "oi": 200.0, "close": 5.0,
         "volume": 100, "settle": 5.0, "expiry": "2025-01-30", "lot_size": 50},
        {"strike": 110.0, "option_type": "CE", "oi": 300.0, "close": 1.0,
         "volume": 150, "settle": 1.0, "expiry": "2025-01-30", "lot_size": 50},
        {"strike": 90.0, "option_type": "PE", "oi": 300.0, "close": 1.0,
         "volume": 150, "settle": 1.0, "expiry": "2025-01-30", "lot_size": 50},
        {"strike": 100.0, "option_type": "PE", "oi": 200.0, "close": 5.0,
         "volume": 100, "settle": 5.0, "expiry": "2025-01-30", "lot_size": 50},
        {"strike": 110.0, "option_type": "PE", "oi": 100.0, "close": 12.0,
         "volume": 50, "settle": 12.0, "expiry": "2025-01-30", "lot_size": 50},
    ])

    spot = 100.0
    dist = OptionsFeatureBuilder._compute_max_pain_distance(chain, spot)

    # Max pain at strike=100, distance = 0%
    assert abs(dist - 0.0) < 1e-10, f"Max pain distance should be 0%, got {dist:.6f}%"

    # --- Asymmetric case: max pain should shift ---
    # Heavy put OI at K=110 pulls max pain upward
    chain_asym = pd.DataFrame([
        {"strike": 90.0, "option_type": "CE", "oi": 10.0, "close": 12.0,
         "volume": 5, "settle": 12.0, "expiry": "2025-01-30", "lot_size": 50},
        {"strike": 100.0, "option_type": "CE", "oi": 10.0, "close": 5.0,
         "volume": 5, "settle": 5.0, "expiry": "2025-01-30", "lot_size": 50},
        {"strike": 110.0, "option_type": "CE", "oi": 10.0, "close": 1.0,
         "volume": 5, "settle": 1.0, "expiry": "2025-01-30", "lot_size": 50},
        {"strike": 90.0, "option_type": "PE", "oi": 10.0, "close": 1.0,
         "volume": 5, "settle": 1.0, "expiry": "2025-01-30", "lot_size": 50},
        {"strike": 100.0, "option_type": "PE", "oi": 10.0, "close": 5.0,
         "volume": 5, "settle": 5.0, "expiry": "2025-01-30", "lot_size": 50},
        {"strike": 110.0, "option_type": "PE", "oi": 10000.0, "close": 12.0,
         "volume": 5000, "settle": 12.0, "expiry": "2025-01-30", "lot_size": 50},
    ])

    # At settle=90:
    #   call_pain = 10*0 + 10*0 + 10*0 = 0
    #   put_pain = 10*0 + 10*10 + 10000*20 = 0 + 100 + 200000 = 200100
    #   total = 200100
    # At settle=100:
    #   call_pain = 10*10 + 10*0 + 10*0 = 100
    #   put_pain = 10*0 + 10*0 + 10000*10 = 100000
    #   total = 100100
    # At settle=110:
    #   call_pain = 10*20 + 10*10 + 10*0 = 200 + 100 = 300
    #   put_pain = 10*0 + 10*0 + 10000*0 = 0
    #   total = 300
    # Min pain at settle=110
    dist_asym = OptionsFeatureBuilder._compute_max_pain_distance(chain_asym, spot)
    expected_asym = (110.0 - 100.0) / 100.0 * 100.0  # = 10%
    assert abs(dist_asym - expected_asym) < 1e-10, (
        f"Asymmetric max pain distance should be {expected_asym}%, got {dist_asym:.6f}%"
    )

    # --- Edge case: empty chain ---
    dist_empty = OptionsFeatureBuilder._compute_max_pain_distance(pd.DataFrame(), spot)
    assert np.isnan(dist_empty), "Empty chain should yield NaN"

    # --- Edge case: single strike (< 2 strikes returns NaN) ---
    chain_single = pd.DataFrame([
        {"strike": 100.0, "option_type": "CE", "oi": 100.0, "close": 5.0,
         "volume": 50, "settle": 5.0, "expiry": "2025-01-30", "lot_size": 50},
        {"strike": 100.0, "option_type": "PE", "oi": 100.0, "close": 5.0,
         "volume": 50, "settle": 5.0, "expiry": "2025-01-30", "lot_size": 50},
    ])
    dist_single = OptionsFeatureBuilder._compute_max_pain_distance(chain_single, spot)
    assert np.isnan(dist_single), "Single strike should yield NaN"
