"""GPU-accelerated implied volatility engine.

Black-Scholes IV inversion via Newton-Raphson on CUDA (PyTorch).
Processes entire option chains (1000+ contracts) in a single GPU kernel call.

Also includes:
  - HAR-RV realized volatility model
  - IV-RV spread computation
  - SVI smile parameterization (future)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

# Use GPU if available, else CPU (still vectorized via PyTorch)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.float32  # FP32 for Newton-Raphson stability; T4 FP32 = 8.1 TFLOPS


# ---------------------------------------------------------------------------
# Normal CDF approximation (GPU-friendly, no scipy dependency)
# ---------------------------------------------------------------------------

def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF."""
    return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Black-Scholes pricing (vectorized on GPU)
# ---------------------------------------------------------------------------


def bs_price(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Black-Scholes European option price.

    All inputs are 1-D tensors of the same length (one element per contract).
    is_call: boolean tensor (True = call, False = put).
    """
    # Clamp to avoid log(0) and div-by-zero
    T_safe = torch.clamp(T, min=1e-6)
    sigma_safe = torch.clamp(sigma, min=1e-6)

    sqrt_T = torch.sqrt(T_safe)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T

    discount = torch.exp(-r * T_safe)

    call_price = S * _norm_cdf(d1) - K * discount * _norm_cdf(d2)
    put_price = K * discount * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    return torch.where(is_call, call_price, put_price)


def bs_vega(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Black-Scholes vega (∂price/∂sigma)."""
    T_safe = torch.clamp(T, min=1e-6)
    sigma_safe = torch.clamp(sigma, min=1e-6)
    sqrt_T = torch.sqrt(T_safe)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * sqrt_T)
    return S * _norm_pdf(d1) * sqrt_T


def bs_delta(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Black-Scholes delta (∂price/∂S)."""
    T_safe = torch.clamp(T, min=1e-6)
    sigma_safe = torch.clamp(sigma, min=1e-6)
    sqrt_T = torch.sqrt(T_safe)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * sqrt_T)
    call_delta = _norm_cdf(d1)
    put_delta = call_delta - 1.0
    return torch.where(is_call, call_delta, put_delta)


def bs_gamma(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Black-Scholes gamma (∂²price/∂S²)."""
    T_safe = torch.clamp(T, min=1e-6)
    sigma_safe = torch.clamp(sigma, min=1e-6)
    sqrt_T = torch.sqrt(T_safe)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * sqrt_T)
    return _norm_pdf(d1) / (S * sigma_safe * sqrt_T)


# ---------------------------------------------------------------------------
# Newton-Raphson IV solver (GPU-vectorized)
# ---------------------------------------------------------------------------


def implied_vol(
    market_price: torch.Tensor,
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    is_call: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Compute implied volatility using Newton-Raphson.

    All inputs are 1-D tensors on the same device.
    Returns IV tensor; contracts that don't converge get NaN.
    """
    n = market_price.shape[0]

    # Initial guess: Brenner-Subrahmanyam approximation
    sigma = torch.full((n,), 0.3, device=market_price.device, dtype=market_price.dtype)

    # Identify contracts where IV is meaningful
    discount = torch.exp(-r * torch.clamp(T, min=1e-6))
    intrinsic = torch.where(
        is_call,
        torch.clamp(S - K * discount, min=0),
        torch.clamp(K * discount - S, min=0),
    )
    # Option price must be positive; for OTM options intrinsic ≈ 0 so any
    # positive price has time value. For ITM, price must exceed intrinsic.
    valid = (market_price > intrinsic * 0.99) & (T > 1e-6) & (market_price > 0.01)

    # Use relative tolerance: converge when price error < 0.01% of market price
    # or absolute error < 0.05 (5 paise — the minimum tick for NIFTY options)
    abs_tol = 0.05

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, is_call)
        vega = bs_vega(S, K, T, r, sigma)

        diff = price - market_price
        # Avoid division by near-zero vega
        vega_safe = torch.clamp(vega, min=1e-8)
        update = diff / vega_safe

        # Only update valid contracts that haven't converged
        rel_err = torch.abs(diff) / torch.clamp(market_price, min=0.01)
        converged_mask = (torch.abs(diff) < abs_tol) | (rel_err < tol)
        still_iterating = valid & ~converged_mask
        sigma = torch.where(still_iterating, sigma - update, sigma)

        # Clamp sigma to reasonable range
        sigma = torch.clamp(sigma, min=0.01, max=5.0)

        if not still_iterating.any():
            break

    # Mark non-converged / invalid as NaN
    final_price = bs_price(S, K, T, r, sigma, is_call)
    rel_err = torch.abs(final_price - market_price) / torch.clamp(market_price, min=0.01)
    converged = valid & ((torch.abs(final_price - market_price) < abs_tol * 10) | (rel_err < tol * 100))
    sigma = torch.where(converged, sigma, torch.tensor(float("nan"), device=sigma.device))

    return sigma


# ---------------------------------------------------------------------------
# High-level: compute IV for a full option chain DataFrame
# ---------------------------------------------------------------------------


@dataclass
class OptionChainIV:
    """IV computation result for an option chain."""
    df: pd.DataFrame          # Original data with IV column added
    atm_iv: float             # ATM implied vol (interpolated)
    spot: float               # Underlying spot price
    nearest_expiry: str       # Nearest expiry date string
    iv_skew_25d: float        # 25-delta put IV minus 25-delta call IV


def compute_chain_iv(
    chain_df: pd.DataFrame,
    risk_free_rate: float = 0.065,
    expiry_col: str = "XpryDt",
    strike_col: str = "StrkPric",
    type_col: str = "OptnTp",
    close_col: str = "ClsPric",
    spot_col: str = "UndrlygPric",
    oi_col: str = "OpnIntrst",
) -> OptionChainIV:
    """Compute IV for every contract in the chain using GPU.

    Args:
        chain_df: DataFrame with NIFTY option data from F&O bhavcopy.
        risk_free_rate: Annualized risk-free rate (default 6.5% for India).

    Returns:
        OptionChainIV with IV column added and summary stats.
    """
    df = chain_df.copy()

    if df.empty:
        return OptionChainIV(df=df, atm_iv=float("nan"), spot=0.0,
                             nearest_expiry="", iv_skew_25d=float("nan"))

    spot = float(df[spot_col].iloc[0])

    # Compute time to expiry in years
    trade_date = pd.to_datetime(df["TradDt"].iloc[0]) if "TradDt" in df.columns else pd.Timestamp.now()
    df["_expiry_dt"] = pd.to_datetime(df[expiry_col].astype(str).str.strip(), format="mixed")
    df["_tte_days"] = (df["_expiry_dt"] - trade_date).dt.days
    df["_tte_years"] = df["_tte_days"] / 365.0

    # Filter out expired or very short-dated options
    mask = (df["_tte_days"] > 0) & (df[close_col] > 0.01)
    df_valid = df[mask].copy()

    if df_valid.empty:
        return OptionChainIV(df=df, atm_iv=float("nan"), spot=spot,
                             nearest_expiry="", iv_skew_25d=float("nan"))

    # Compute implied forward per expiry using put-call parity:
    #   F = K + C - P  (at ATM strike where C ≈ P)
    # This avoids the spot-vs-forward bias that causes call/put IV asymmetry.
    forward_map: dict[str, float] = {}
    for exp in df_valid["_expiry_dt"].unique():
        exp_slice = df_valid[df_valid["_expiry_dt"] == exp]
        calls = exp_slice[exp_slice[type_col].str.strip() == "CE"]
        puts = exp_slice[exp_slice[type_col].str.strip() == "PE"]
        if calls.empty or puts.empty:
            forward_map[exp] = spot
            continue
        # Find strikes present in both calls and puts
        common_strikes = set(calls[strike_col]) & set(puts[strike_col])
        if not common_strikes:
            forward_map[exp] = spot
            continue
        best_F, best_diff = spot, float("inf")
        for K in common_strikes:
            c_row = calls[calls[strike_col] == K]
            p_row = puts[puts[strike_col] == K]
            if c_row.empty or p_row.empty:
                continue
            C = float(c_row[close_col].iloc[0])
            P = float(p_row[close_col].iloc[0])
            F = K + C - P
            diff = abs(C - P)
            if diff < best_diff:
                best_diff = diff
                best_F = F
        forward_map[exp] = best_F

    # Use implied forward instead of spot for each contract
    df_valid["_forward"] = df_valid["_expiry_dt"].map(forward_map).fillna(spot)

    # Move to GPU tensors
    # Use implied forward as S and r=0 (Black-76 equivalent).
    # The forward already incorporates cost-of-carry, so drift in d1
    # should be σ²/2 only.  Setting r=0 achieves this and also removes
    # the asymmetric discounting that causes call/put IV divergence.
    S = torch.tensor(df_valid["_forward"].values, dtype=_DTYPE, device=_DEVICE)
    K = torch.tensor(df_valid[strike_col].values, dtype=_DTYPE, device=_DEVICE)
    T = torch.tensor(df_valid["_tte_years"].values, dtype=_DTYPE, device=_DEVICE)
    r = torch.zeros_like(S)  # Black-76: r=0 when using forward
    mkt = torch.tensor(df_valid[close_col].values, dtype=_DTYPE, device=_DEVICE)
    is_call = torch.tensor(
        (df_valid[type_col].str.strip() == "CE").values,
        dtype=torch.bool,
        device=_DEVICE,
    )

    # GPU Newton-Raphson
    iv = implied_vol(mkt, S, K, T, r, is_call)

    # Back to CPU / DataFrame
    df_valid["IV"] = iv.cpu().numpy()

    # Also compute delta for each contract
    iv_for_greeks = torch.where(torch.isnan(iv), torch.tensor(0.3, device=_DEVICE), iv)
    delta = bs_delta(S, K, T, r, iv_for_greeks, is_call)
    gamma = bs_gamma(S, K, T, r, iv_for_greeks)
    df_valid["DELTA"] = delta.cpu().numpy()
    df_valid["GAMMA"] = gamma.cpu().numpy()

    # Merge IV back into original df
    df["IV"] = float("nan")
    df["DELTA"] = float("nan")
    df["GAMMA"] = float("nan")
    df.loc[df_valid.index, "IV"] = df_valid["IV"].values
    df.loc[df_valid.index, "DELTA"] = df_valid["DELTA"].values
    df.loc[df_valid.index, "GAMMA"] = df_valid["GAMMA"].values

    # Clean up temp columns
    df = df.drop(columns=["_expiry_dt", "_tte_days", "_tte_years"], errors="ignore")

    # --- Summary stats for nearest expiry ---
    nearest_expiry = df_valid["_expiry_dt"].min()
    nearest = df_valid[df_valid["_expiry_dt"] == nearest_expiry]
    nearest_valid = nearest.dropna(subset=["IV"])

    # ATM IV: weighted average of nearest CE and PE at closest strike to spot
    atm_iv = float("nan")
    if not nearest_valid.empty:
        nearest_valid = nearest_valid.copy()
        nearest_valid["_dist"] = abs(nearest_valid[strike_col] - spot)
        atm_strike = nearest_valid.loc[nearest_valid["_dist"].idxmin(), strike_col]
        atm_options = nearest_valid[nearest_valid[strike_col] == atm_strike]
        if not atm_options.empty:
            atm_iv = float(atm_options["IV"].mean())

    # 25-delta skew: IV of 25-delta put minus 25-delta call
    iv_skew_25d = float("nan")
    if not nearest_valid.empty:
        calls = nearest_valid[nearest_valid[type_col].str.strip() == "CE"].copy()
        puts = nearest_valid[nearest_valid[type_col].str.strip() == "PE"].copy()
        if not calls.empty and not puts.empty:
            calls["_d_dist"] = abs(calls["DELTA"] - 0.25)
            puts["_d_dist"] = abs(puts["DELTA"] + 0.25)  # put delta is negative
            if not calls["_d_dist"].isna().all() and not puts["_d_dist"].isna().all():
                call_25d_iv = calls.loc[calls["_d_dist"].idxmin(), "IV"]
                put_25d_iv = puts.loc[puts["_d_dist"].idxmin(), "IV"]
                if not (math.isnan(call_25d_iv) or math.isnan(put_25d_iv)):
                    iv_skew_25d = put_25d_iv - call_25d_iv

    return OptionChainIV(
        df=df,
        atm_iv=atm_iv,
        spot=spot,
        nearest_expiry=str(nearest_expiry.date()) if pd.notna(nearest_expiry) else "",
        iv_skew_25d=iv_skew_25d,
    )


# ---------------------------------------------------------------------------
# HAR-RV: Heterogeneous Autoregressive Realized Volatility
# ---------------------------------------------------------------------------


def realized_vol(
    close_prices: pd.Series,
    window: int = 20,
    annualize: float = 252.0,
) -> pd.Series:
    """Simple realized volatility (close-to-close)."""
    log_ret = np.log(close_prices / close_prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(annualize)


def har_rv(close_prices: pd.Series, annualize: float = 252.0) -> pd.Series:
    """HAR-RV: Heterogeneous Autoregressive Realized Volatility.

    Combines 1-day, 5-day, and 22-day realized vol components.
    This is a descriptive model (not predictive) — it gives a smooth
    estimate of current realized vol that reacts at multiple timescales.

    Returns annualized RV series.
    """
    log_ret = np.log(close_prices / close_prices.shift(1))
    rv_sq = log_ret ** 2  # daily squared return as RV proxy

    # HAR components: average of squared returns over different horizons
    rv_1d = rv_sq
    rv_5d = rv_sq.rolling(5).mean()
    rv_22d = rv_sq.rolling(22).mean()

    # HAR-RV estimate (equal-weighted combination, annualized)
    har = (rv_1d + rv_5d + rv_22d) / 3.0
    har_vol = np.sqrt(har * annualize)

    return har_vol


# ---------------------------------------------------------------------------
# Gamma Exposure (GEX) computation
# ---------------------------------------------------------------------------


def compute_gex(
    chain_df: pd.DataFrame,
    spot: float,
    strike_col: str = "StrkPric",
    type_col: str = "OptnTp",
    oi_col: str = "OpnIntrst",
    gamma_col: str = "GAMMA",
    lot_size: int = 75,
) -> pd.DataFrame:
    """Compute Gamma Exposure (GEX) per strike.

    GEX = OI × Gamma × Spot² × lot_size / 1e7 (in crores)

    Convention: dealers are short what retail is long.
    Retail is net long calls, net long puts →
      Call GEX is positive (dealers short calls → short gamma on calls)
      Put GEX is negative (dealers short puts → long gamma on puts)
    Net GEX > 0 → dealers long gamma → mean reversion
    Net GEX < 0 → dealers short gamma → momentum/amplification

    Returns DataFrame with per-strike GEX and summary.
    """
    df = chain_df.dropna(subset=[gamma_col]).copy()
    if df.empty:
        return pd.DataFrame()

    is_call = df[type_col].str.strip() == "CE"

    # Dealer gamma: +gamma for puts (dealer short put = long gamma),
    # -gamma for calls (dealer short call = short gamma on upside)
    # Actually: if retail buys calls, dealer sells calls → dealer has -gamma from calls
    # If retail buys puts, dealer sells puts → dealer has +gamma from puts
    sign = torch.where(
        torch.tensor(is_call.values), torch.tensor(-1.0), torch.tensor(1.0)
    )

    gex = (
        sign.numpy()
        * df[oi_col].values
        * df[gamma_col].values
        * spot ** 2
        * lot_size
        / 1e7  # scale to crores
    )
    df["GEX_CR"] = gex

    # Aggregate per strike
    per_strike = (
        df.groupby(strike_col)["GEX_CR"]
        .sum()
        .reset_index()
        .sort_values(strike_col)
    )
    per_strike.columns = ["Strike", "GEX_CR"]

    return per_strike
