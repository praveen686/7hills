"""SANOS: Smooth strictly Arbitrage-free Non-parametric Option Surfaces.

Implementation of the production model from Buehler, Horvath, Kratsios,
Limmer, Saqur (arXiv:2601.11209v2, Jan 2026).

Core idea: option prices are convex combinations of Black-Scholes call
payoffs anchored at model strikes, with martingale density weights found
via linear programming.

    Ĉ(Tj, K) = Σᵢ qⱼⁱ × Call(Kⁱ, K, η·Vⱼ)

where:
    - Kⁱ are model strikes (grid points)
    - qⱼ are martingale densities (non-negative, sum=1, unit mean)
    - Vⱼ are ATM implied variances per expiry
    - η ∈ [0,1) controls smoothness (η=0 → linear interp, η=0.25 recommended)

Calibration is a single global LP across all expiries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes call price (numpy, vectorized)
# ---------------------------------------------------------------------------

def bs_call(s: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Black-Scholes call price.

    Args:
        s: spot (or forward), shape (N,) or scalar
        k: strike, shape (M,) or scalar
        v: total variance (σ²T), shape broadcastable

    Returns:
        Call price, same shape as broadcast of inputs.
    """
    v_safe = np.maximum(v, 1e-12)
    sqrt_v = np.sqrt(v_safe)
    d_plus = (-np.log(k / s) + 0.5 * v_safe) / sqrt_v
    d_minus = d_plus - sqrt_v
    return s * norm.cdf(d_plus) - k * norm.cdf(d_minus)


def bs_call_vega(s: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """∂Call/∂σ (vega in vol terms, not variance)."""
    v_safe = np.maximum(v, 1e-12)
    sqrt_v = np.sqrt(v_safe)
    d_plus = (-np.log(k / s) + 0.5 * v_safe) / sqrt_v
    return s * norm.pdf(d_plus) * sqrt_v


# ---------------------------------------------------------------------------
# SANOS Surface
# ---------------------------------------------------------------------------

@dataclass
class SANOSResult:
    """Calibration result from SANOS surface fitting."""

    # Per-expiry marginal densities
    densities: list[np.ndarray]       # q_j for each expiry, shape (N_j,)
    model_strikes: np.ndarray         # K^1, ..., K^N (common model grid)
    variances: np.ndarray             # V_1, ..., V_M (ATM variances per expiry)
    eta: float                        # smoothness parameter

    # Market data used for calibration
    expiry_labels: list[str]          # expiry date strings
    market_strikes: list[np.ndarray]  # per-expiry market strikes
    market_mids: list[np.ndarray]     # per-expiry market mid prices

    # Diagnostics
    fit_errors: list[np.ndarray]      # model - market mid, per expiry
    max_fit_error: float              # worst absolute fit error
    lp_success: bool                  # did the LP converge?

    def price(self, expiry_idx: int, strikes: np.ndarray) -> np.ndarray:
        """Price call options at arbitrary strikes for a given expiry.

        This is the core SANOS formula:
            Ĉ(Tj, K) = Σᵢ qⱼⁱ × Call(Kⁱ, K, η·Vⱼ)
        """
        q = self.densities[expiry_idx]
        K_model = self.model_strikes
        v = self.eta * self.variances[expiry_idx]

        # Shape: (N_model,) x (N_strikes,) -> (N_model, N_strikes)
        # Call(K^i, K, η·V_j) for each model strike i and query strike K
        K_grid = strikes.reshape(1, -1)  # (1, N_strikes)
        S_grid = K_model.reshape(-1, 1)  # (N_model, 1)
        v_grid = np.full_like(S_grid, v)

        call_matrix = bs_call(S_grid, K_grid, v_grid)  # (N_model, N_strikes)

        # Weighted sum: Σᵢ qⱼⁱ × Call(Kⁱ, K, η·Vⱼ)
        prices = q @ call_matrix  # (N_strikes,)
        return prices

    def iv(self, expiry_idx: int, strikes: np.ndarray, T: float) -> np.ndarray:
        """Implied vol from the SANOS surface at given strikes.

        Inverts the SANOS price back to Black-Scholes IV.
        Uses vectorized bisection (guaranteed convergence).
        """
        prices = self.price(expiry_idx, strikes)
        forward = np.sum(self.densities[expiry_idx] * self.model_strikes)
        F = np.full_like(strikes, forward)

        # Bisection: BS is monotone increasing in σ
        lo = np.full_like(prices, 0.001)
        hi = np.full_like(prices, 5.0)

        for _ in range(80):  # 80 iterations → ~2^-80 ≈ 1e-24 precision
            mid = 0.5 * (lo + hi)
            v = mid ** 2 * T
            model_price = bs_call(F, strikes, v)
            too_high = model_price > prices
            lo = np.where(too_high, lo, mid)
            hi = np.where(too_high, mid, hi)

        return 0.5 * (lo + hi)

    def density(self, expiry_idx: int, strikes: np.ndarray) -> np.ndarray:
        """Risk-neutral density (∂²C/∂K²) at given strikes.

        Computed analytically from the SANOS formula.
        """
        q = self.densities[expiry_idx]
        K_model = self.model_strikes
        v = self.eta * self.variances[expiry_idx]
        v_safe = max(v, 1e-12)
        sqrt_v = math.sqrt(v_safe)

        density = np.zeros_like(strikes, dtype=float)
        for i, (qi, Ki) in enumerate(zip(q, K_model)):
            if qi < 1e-15:
                continue
            # ∂²Call(Ki, K, v)/∂K² = (1/(K·sqrt_v)) × φ(d_minus)
            K_safe = np.maximum(strikes, 1e-10)
            d_minus = (-np.log(K_safe / Ki) - 0.5 * v_safe) / sqrt_v
            density += qi * norm.pdf(d_minus) / (K_safe * sqrt_v)

        return density


def fit_sanos(
    market_strikes: list[np.ndarray],
    market_calls: list[np.ndarray],
    market_spreads: list[np.ndarray] | None = None,
    atm_variances: np.ndarray | None = None,
    expiry_labels: list[str] | None = None,
    eta: float = 0.50,
    n_model_strikes: int = 100,
    K_min: float = 0.7,
    K_max: float = 1.5,
) -> SANOSResult:
    """Fit a SANOS smooth arbitrage-free surface to market option data.

    This implements the production model from Section 4.2 of the paper.

    Args:
        market_strikes: List of arrays, one per expiry. Strikes normalized
            by forward (K/F) so that ATM ≈ 1.0.
        market_calls: List of arrays, call prices normalized by forward (C/F).
        market_spreads: Optional bid-ask spread per option for weighting.
            If None, equal weights.
        atm_variances: ATM total variance (σ²T) per expiry. If None, estimated
            from ATM option prices.
        expiry_labels: String labels for each expiry.
        eta: Smoothness parameter ∈ [0, 1). 0 = linear, 0.50 = recommended.
        n_model_strikes: Number of model grid strikes.
        K_min: Minimum model strike (as fraction of forward).
        K_max: Maximum model strike (as fraction of forward).

    Returns:
        SANOSResult with calibrated densities and pricing functions.
    """
    M = len(market_strikes)  # number of expiries
    if expiry_labels is None:
        expiry_labels = [f"T{j}" for j in range(M)]

    # Build model strike grid (common across expiries for production model)
    K_model = np.linspace(K_min, K_max, n_model_strikes)
    N = len(K_model)

    # Estimate ATM variances if not provided
    if atm_variances is None:
        atm_variances = np.zeros(M)
        for j in range(M):
            # Find ATM option and estimate variance from straddle
            atm_idx = np.argmin(np.abs(market_strikes[j] - 1.0))
            # Brenner-Subrahmanyam: C_atm ≈ F × σ × √T / √(2π)
            # So σ²T ≈ (C_atm / F × √(2π))²  (with F=1 after normalization)
            c_atm = market_calls[j][atm_idx]
            atm_variances[j] = max((c_atm * math.sqrt(2.0 * math.pi)) ** 2, 1e-6)

    # Ensure variances are increasing (required for no calendar arb)
    for j in range(1, M):
        atm_variances[j] = max(atm_variances[j], atm_variances[j - 1] + 1e-8)

    # --- Build LP ---
    # Decision variables: q_1, ..., q_M concatenated, each ∈ R^N
    # Total variables: M × N
    total_vars = M * N

    # We'll collect: A_ub (inequality), b_ub, A_eq (equality), b_eq, c (objective)
    A_ub_rows = []
    b_ub_rows = []
    A_eq_rows = []
    b_eq_rows = []

    # Objective: minimize Σ_j |wⱼ · (Cⱼ - cⱼ)|
    # We linearize |x| ≤ t by introducing slack variables t_j
    # But for simplicity, we minimize the L1 fitting error using the standard
    # LP trick: for each market observation, introduce slack s⁺, s⁻ ≥ 0
    # with model_price - market_mid = s⁺ - s⁻, minimize Σ w(s⁺ + s⁻)

    total_market = sum(len(mk) for mk in market_strikes)
    # Extended variables: [q_1, ..., q_M, s⁺_all, s⁻_all]
    n_extended = total_vars + 2 * total_market

    # Build objective: minimize weighted sum of slacks
    c_obj = np.zeros(n_extended)
    slack_offset = total_vars
    for j in range(M):
        n_j = len(market_strikes[j])
        offset = sum(len(market_strikes[jj]) for jj in range(j))
        for ℓ in range(n_j):
            w = 1.0
            if market_spreads is not None and market_spreads[j] is not None:
                spread = max(market_spreads[j][ℓ], 1e-8)
                w = 1.0 / spread
            c_obj[slack_offset + offset + ℓ] = w          # s⁺
            c_obj[slack_offset + total_market + offset + ℓ] = w  # s⁻

    # --- Equality constraints ---

    for j in range(M):
        q_start = j * N

        # 1. Sum to 1: 1' · qⱼ = 1
        row = np.zeros(n_extended)
        row[q_start:q_start + N] = 1.0
        A_eq_rows.append(row)
        b_eq_rows.append(1.0)

        # 2. Unit mean (martingale): K' · qⱼ = 1
        row = np.zeros(n_extended)
        row[q_start:q_start + N] = K_model
        A_eq_rows.append(row)
        b_eq_rows.append(1.0)

        # 3. Fitting: Cⱼ · qⱼ - s⁺ + s⁻ = market_mid
        #    i.e., Σᵢ Call(Kⁱ, kⱼˡ, η·Vⱼ) × qⱼⁱ - s⁺ₗ + s⁻ₗ = Cⱼˡ
        v_j = eta * atm_variances[j]
        n_j = len(market_strikes[j])
        offset = sum(len(market_strikes[jj]) for jj in range(j))

        for ℓ in range(n_j):
            row = np.zeros(n_extended)
            # Cⱼ matrix: Call(K_model^i, market_strike^ℓ, η·Vⱼ)
            k_mkt = market_strikes[j][ℓ]
            row[q_start:q_start + N] = bs_call(K_model, k_mkt, v_j)
            # Slack variables
            row[slack_offset + offset + ℓ] = -1.0          # -s⁺
            row[slack_offset + total_market + offset + ℓ] = 1.0  # +s⁻
            A_eq_rows.append(row)
            b_eq_rows.append(market_calls[j][ℓ])

    # --- Inequality constraints ---

    # Calendar arbitrage: uⱼ ≥ rⱼ
    # uⱼ = Uⱼ · qⱼ  (model prices at model strikes for expiry j)
    # rⱼ = Rⱼ · qⱼ₋₁ (previous expiry prices at current model strikes)
    # i.e., Rⱼ · qⱼ₋₁ - Uⱼ · qⱼ ≤ 0

    for j in range(1, M):
        v_j = eta * atm_variances[j]
        v_prev = eta * atm_variances[j - 1]
        q_start_j = j * N
        q_start_prev = (j - 1) * N

        for ℓ in range(N):
            row = np.zeros(n_extended)
            # Rⱼ: Call(K_model_prev^i, K_model^ℓ, η·V_{j-1})
            k_ℓ = K_model[ℓ]
            row[q_start_prev:q_start_prev + N] = bs_call(K_model, k_ℓ, v_prev)
            # -Uⱼ: -Call(K_model^i, K_model^ℓ, η·Vⱼ)
            row[q_start_j:q_start_j + N] = -bs_call(K_model, k_ℓ, v_j)
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)

    # --- Variable bounds ---
    # qⱼⁱ ≥ 0, s⁺ ≥ 0, s⁻ ≥ 0
    bounds = [(0, None)] * n_extended

    # --- Solve LP ---
    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_rows) if b_eq_rows else None
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None

    result = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"presolve": True, "time_limit": 60},
    )

    # --- Extract results ---
    densities = []
    fit_errors_list = []
    max_error = 0.0

    if result.success:
        x = result.x
        for j in range(M):
            q_j = x[j * N:(j + 1) * N]
            densities.append(q_j)
    else:
        # Fallback: use uniform density (won't be arbitrage-free)
        for j in range(M):
            q_j = np.zeros(N)
            q_j[N // 2] = 1.0  # delta at ATM
            densities.append(q_j)

    # Compute fit errors
    for j in range(M):
        v_j = eta * atm_variances[j]
        n_j = len(market_strikes[j])
        model_prices = np.zeros(n_j)
        for ℓ in range(n_j):
            k_mkt = market_strikes[j][ℓ]
            model_prices[ℓ] = np.dot(
                densities[j], bs_call(K_model, k_mkt, v_j)
            )
        errors = model_prices - market_calls[j]
        fit_errors_list.append(errors)
        max_error = max(max_error, np.max(np.abs(errors)))

    return SANOSResult(
        densities=densities,
        model_strikes=K_model,
        variances=atm_variances,
        eta=eta,
        expiry_labels=expiry_labels,
        market_strikes=market_strikes,
        market_mids=market_calls,
        fit_errors=fit_errors_list,
        max_fit_error=max_error,
        lp_success=result.success,
    )


# ---------------------------------------------------------------------------
# Helper: prepare NIFTY F&O bhavcopy data for SANOS
# ---------------------------------------------------------------------------


def prepare_nifty_chain(
    fno_df,
    symbol: str = "NIFTY",
    instrument: str = "IDO",
    max_expiries: int = 6,
):
    """Convert F&O bhavcopy DataFrame into SANOS-ready inputs.

    Uses raw NSE column names: TckrSymb, FinInstrmTp, XpryDt, StrkPric,
    OptnTp, ClsPric, UndrlygPric.  FinInstrmTp values: IDO (index option),
    IDF (index future), STO (stock option), STF (stock future).

    Normalizes strikes and prices by the implied forward (from put-call parity).

    Returns:
        dict with keys: market_strikes, market_calls, market_spreads,
        atm_variances, expiry_labels, forward, spot, trade_date
    """
    import pandas as pd

    df = fno_df[
        (fno_df["TckrSymb"] == symbol) & (fno_df["FinInstrmTp"] == instrument)
    ].copy()

    if df.empty:
        return None

    spot = float(df["UndrlygPric"].iloc[0])
    trade_date = df["TradDt"].iloc[0] if "TradDt" in df.columns else ""

    # Parse expiry dates
    df["_exp"] = pd.to_datetime(df["XpryDt"].astype(str).str.strip(), format="mixed")
    df["_dte"] = (df["_exp"] - pd.to_datetime(trade_date)).dt.days
    df = df[df["_dte"] > 0]

    # Select up to max_expiries
    expiries = sorted(df["_exp"].unique())[:max_expiries]

    market_strikes_list = []
    market_calls_list = []
    market_spreads_list = []
    atm_vars = []
    expiry_labels = []

    for exp in expiries:
        sub = df[df["_exp"] == exp].copy()
        dte = int((exp - pd.to_datetime(trade_date)).days)
        T = dte / 365.0

        # Compute implied forward from put-call parity
        calls = sub[sub["OptnTp"].str.strip() == "CE"]
        puts = sub[sub["OptnTp"].str.strip() == "PE"]
        common = set(calls["StrkPric"]) & set(puts["StrkPric"])
        if not common:
            continue

        best_F, best_diff = spot, float("inf")
        for K in common:
            c_row = calls[calls["StrkPric"] == K]
            p_row = puts[puts["StrkPric"] == K]
            if c_row.empty or p_row.empty:
                continue
            C = float(c_row["ClsPric"].iloc[0])
            P = float(p_row["ClsPric"].iloc[0])
            F = K + C - P
            diff = abs(C - P)
            if diff < best_diff:
                best_diff = diff
                best_F = F

        forward = best_F

        # Use OTM options only (better prices, more liquid)
        otm_calls = calls[calls["StrkPric"] >= forward].copy()
        otm_puts = puts[puts["StrkPric"] <= forward].copy()

        # Normalize: strike → K/F, price → C/F (for calls)
        # For puts, convert to calls via put-call parity: C = P + F - K
        rows = []
        for _, r in otm_calls.iterrows():
            k_norm = r["StrkPric"] / forward
            c_norm = r["ClsPric"] / forward
            if c_norm > 0.001:  # filter near-zero
                rows.append((k_norm, c_norm))

        for _, r in otm_puts.iterrows():
            k_norm = r["StrkPric"] / forward
            # Convert put to call: C = P + 1 - K/F (in normalized terms)
            c_norm = r["ClsPric"] / forward + 1.0 - k_norm
            if c_norm > 0.001:
                rows.append((k_norm, c_norm))

        if len(rows) < 5:
            continue

        rows.sort(key=lambda x: x[0])
        k_arr = np.array([r[0] for r in rows])
        c_arr = np.array([r[1] for r in rows])

        # ATM variance estimate from ATM straddle
        # Straddle = CE + PE at strike nearest forward
        # Brenner: straddle ≈ 2·F·σ·√T/√(2π) → σ²T = (straddle/(2F)·√(2π))²
        atm_K = min(common, key=lambda K: abs(K - forward))
        ce_row = calls[calls["StrkPric"] == atm_K]
        pe_row = puts[puts["StrkPric"] == atm_K]
        if not ce_row.empty and not pe_row.empty:
            straddle = float(ce_row["ClsPric"].iloc[0] + pe_row["ClsPric"].iloc[0])
            straddle_norm = straddle / forward
            atm_var = max((straddle_norm / 2.0 * math.sqrt(2.0 * math.pi)) ** 2, 1e-6)
        else:
            atm_idx = np.argmin(np.abs(k_arr - 1.0))
            c_atm = c_arr[atm_idx]
            atm_var = max((c_atm * math.sqrt(2.0 * math.pi)) ** 2, 1e-6)

        market_strikes_list.append(k_arr)
        market_calls_list.append(c_arr)
        market_spreads_list.append(None)  # no bid-ask from bhavcopy
        atm_vars.append(atm_var)
        expiry_labels.append(str(exp.date()))

    if not market_strikes_list:
        return None

    return {
        "market_strikes": market_strikes_list,
        "market_calls": market_calls_list,
        "market_spreads": market_spreads_list,
        "atm_variances": np.array(atm_vars),
        "expiry_labels": expiry_labels,
        "forward": forward,
        "spot": spot,
        "trade_date": trade_date,
    }
