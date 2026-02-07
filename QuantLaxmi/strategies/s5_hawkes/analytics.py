"""Microstructure analytics computed from option chain snapshots.

Signals:
  1. GEX (Gamma Exposure) — net dealer gamma, flip level
  2. OI Flow — delta-weighted OI change between snapshots
  3. Max Pain — strike minimizing total option buyer payoff
  4. Futures Basis — spot vs futures premium/discount
  5. PCR — put-call ratio (OI and volume weighted)
  6. IV Term Structure — near vs far IV ratio

All analytics operate on DataFrames produced by collector.snapshot_chain().
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Re-use GPU-accelerated BS functions from iv_engine
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Black-Scholes helpers (inlined to avoid circular import)
# ---------------------------------------------------------------------------

def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _bs_greeks(
    S: float, K: np.ndarray, T: np.ndarray, r: float,
    sigma: np.ndarray, is_call: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute delta and gamma for arrays of options (GPU-accelerated).

    Returns (delta, gamma) as numpy arrays.
    """
    S_t = torch.tensor(S, dtype=_DTYPE, device=_DEVICE)
    K_t = torch.tensor(K, dtype=_DTYPE, device=_DEVICE)
    T_t = torch.tensor(T, dtype=_DTYPE, device=_DEVICE).clamp(min=1e-6)
    sigma_t = torch.tensor(sigma, dtype=_DTYPE, device=_DEVICE).clamp(min=1e-4)
    is_call_t = torch.tensor(is_call, dtype=torch.bool, device=_DEVICE)
    r_t = torch.tensor(r, dtype=_DTYPE, device=_DEVICE)

    sqrt_T = torch.sqrt(T_t)
    d1 = (torch.log(S_t / K_t) + (r_t + 0.5 * sigma_t ** 2) * T_t) / (sigma_t * sqrt_T)
    d2 = d1 - sigma_t * sqrt_T

    nd1 = _norm_cdf(d1)
    # Delta
    delta = torch.where(is_call_t, nd1, nd1 - 1.0)

    # Gamma
    pdf_d1 = torch.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    gamma = pdf_d1 / (S_t * sigma_t * sqrt_T)

    return delta.cpu().numpy(), gamma.cpu().numpy()


def _bs_iv_bisection(
    price: np.ndarray, S: float, K: np.ndarray, T: np.ndarray,
    r: float, is_call: np.ndarray,
    lo: float = 0.01, hi: float = 3.0, n_iter: int = 50,
) -> np.ndarray:
    """Bisection IV solver (GPU-accelerated, robust for all options)."""
    n = len(price)
    S_t = torch.full((n,), S, dtype=_DTYPE, device=_DEVICE)
    K_t = torch.tensor(K, dtype=_DTYPE, device=_DEVICE)
    T_t = torch.tensor(T, dtype=_DTYPE, device=_DEVICE).clamp(min=1e-6)
    r_t = torch.tensor(r, dtype=_DTYPE, device=_DEVICE)
    is_call_t = torch.tensor(is_call, dtype=torch.bool, device=_DEVICE)
    mkt = torch.tensor(price, dtype=_DTYPE, device=_DEVICE)

    lo_t = torch.full((n,), lo, dtype=_DTYPE, device=_DEVICE)
    hi_t = torch.full((n,), hi, dtype=_DTYPE, device=_DEVICE)

    for _ in range(n_iter):
        mid = 0.5 * (lo_t + hi_t)
        sqrt_T = torch.sqrt(T_t)
        d1 = (torch.log(S_t / K_t) + (r_t + 0.5 * mid ** 2) * T_t) / (mid * sqrt_T)
        d2 = d1 - mid * sqrt_T

        nd1 = _norm_cdf(d1)
        nd2 = _norm_cdf(d2)
        nmd1 = _norm_cdf(-d1)
        nmd2 = _norm_cdf(-d2)

        call_price = S_t * nd1 - K_t * torch.exp(-r_t * T_t) * nd2
        put_price = K_t * torch.exp(-r_t * T_t) * nmd2 - S_t * nmd1
        model_price = torch.where(is_call_t, call_price, put_price)

        too_high = model_price > mkt
        lo_t = torch.where(too_high, lo_t, mid)
        hi_t = torch.where(too_high, mid, hi_t)

    iv = 0.5 * (lo_t + hi_t)
    # NaN out where price is too small (intrinsic only)
    iv[mkt < 0.5] = float("nan")
    return iv.cpu().numpy()


# ---------------------------------------------------------------------------
# Analytics dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GEXAnalysis:
    """Gamma Exposure analysis for a single snapshot."""
    net_gex_cr: float          # Net GEX in crores
    call_gex_cr: float         # Call-side GEX (dealer short gamma)
    put_gex_cr: float          # Put-side GEX (dealer long gamma)
    gex_flip_strike: float     # Strike where cumulative GEX changes sign
    regime: str                # "mean_revert" (GEX > 0) or "momentum" (GEX < 0)
    per_strike: pd.DataFrame   # Per-strike GEX breakdown
    spot: float


@dataclass
class OIFlowAnalysis:
    """OI change analysis between two snapshots."""
    net_delta_flow: float      # Net delta-weighted OI change
    call_oi_change: int        # Total call OI change
    put_oi_change: int         # Total put OI change
    direction: str             # "bullish" or "bearish"
    top_call_strikes: list     # Strikes with largest call OI additions
    top_put_strikes: list      # Strikes with largest put OI additions
    score: float               # Normalized signal [-1, +1]


@dataclass
class MaxPainAnalysis:
    """Max pain computation."""
    max_pain_strike: float
    distance_pct: float        # Distance from spot to max pain (%)
    direction: str             # "above" or "below" spot


@dataclass
class BasisAnalysis:
    """Futures basis analysis."""
    basis_points: float        # Futures - spot
    basis_pct: float           # Annualized basis %
    basis_zscore: float        # Z-score vs recent history
    signal: str                # "overleveraged_long", "overleveraged_short", "neutral"


@dataclass
class PCRAnalysis:
    """Put-Call Ratio analysis."""
    pcr_oi: float              # PCR by open interest
    pcr_volume: float          # PCR by volume
    signal: str                # "extreme_fear" (PCR > 1.3), "extreme_greed" (PCR < 0.7), "neutral"


@dataclass
class IVTermStructure:
    """IV term structure analysis."""
    near_iv: float             # Near-expiry ATM IV
    far_iv: float              # Far-expiry ATM IV
    slope: float               # near/far ratio
    signal: str                # "inverted" (panic), "steep" (complacent), "normal"


@dataclass
class MicrostructureSnapshot:
    """Combined analytics for a single point in time."""
    timestamp: str
    symbol: str
    spot: float
    futures: float
    gex: GEXAnalysis
    oi_flow: OIFlowAnalysis | None  # None if no previous snapshot
    max_pain: MaxPainAnalysis
    basis: BasisAnalysis
    pcr: PCRAnalysis
    iv_term: IVTermStructure


# ---------------------------------------------------------------------------
# GEX computation
# ---------------------------------------------------------------------------

MIN_GEX_OI = 100  # Minimum OI threshold per strike for GEX computation


def compute_gex(
    df: pd.DataFrame, spot: float, risk_free: float = 0.065,
    lot_size: int = 75, as_of_date: pd.Timestamp | None = None,
) -> GEXAnalysis:
    """Compute Gamma Exposure from a chain snapshot.

    Convention: dealers are net SHORT what retail is long.
    - Call GEX is negative (dealers short calls → short gamma on upside)
    - Put GEX is positive (dealers short puts → long gamma on downside)
    - Net GEX > 0 → dealers are long gamma → mean reversion
    - Net GEX < 0 → dealers are short gamma → momentum/amplification
    """
    # Use only nearest expiry for strongest signal
    nearest_expiry = sorted(df["expiry"].unique())[0]
    chain = df[df["expiry"] == nearest_expiry].copy()

    if chain.empty:
        return GEXAnalysis(0, 0, 0, spot, "neutral", pd.DataFrame(), spot)

    # Filter strikes with sufficient OI for reliable GEX computation
    valid_oi_strikes = chain[chain["oi"] >= MIN_GEX_OI]["strike"].nunique()
    if valid_oi_strikes < 5:
        # Insufficient valid-OI strikes — return neutral GEX
        return GEXAnalysis(0, 0, 0, spot, "neutral", chain[["strike", "option_type", "oi"]].head(0), spot)

    # Compute time to expiry (use as_of_date if provided, else wall-clock)
    from datetime import datetime, timezone, timedelta
    exp_date = pd.Timestamp(nearest_expiry).tz_localize(None)
    if as_of_date is not None:
        ref_date = pd.Timestamp(as_of_date).tz_localize(None).normalize()
    else:
        ref_date = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    T_days = max((exp_date - ref_date).days, 0) + 1  # at least 1 day
    T = T_days / 365.0

    # Compute IV and Greeks
    strikes = chain["strike"].values.astype(np.float64)
    prices = chain["ltp"].values.astype(np.float64)
    is_call = (chain["option_type"] == "CE").values
    oi = chain["oi"].values.astype(np.float64)

    T_arr = np.full(len(chain), T)

    # IV via bisection
    iv = _bs_iv_bisection(prices, spot, strikes, T_arr, risk_free, is_call)

    # Greeks
    valid = ~np.isnan(iv) & (iv > 0.01)
    delta = np.zeros(len(chain))
    gamma = np.zeros(len(chain))

    if valid.any():
        d, g = _bs_greeks(
            spot, strikes[valid], T_arr[valid], risk_free, iv[valid], is_call[valid]
        )
        delta[valid] = d
        gamma[valid] = g

    # GEX per contract: sign × OI × Gamma × S² × lot_size / 1e7 (crores)
    # Dealer perspective: short calls (negative gamma), short puts (positive gamma)
    sign = np.where(is_call, -1.0, 1.0)
    gex_per = sign * oi * gamma * spot ** 2 * lot_size / 1e7

    chain = chain.copy()
    chain["iv"] = iv
    chain["delta"] = delta
    chain["gamma"] = gamma
    chain["gex_cr"] = gex_per

    # Aggregate
    call_gex = gex_per[is_call].sum()
    put_gex = gex_per[~is_call].sum()
    net_gex = call_gex + put_gex

    # GEX flip level: strike where cumulative GEX changes sign
    strike_gex = chain.groupby("strike")["gex_cr"].sum().sort_index()
    cum_gex = strike_gex.cumsum().values
    sign_changes = (cum_gex[:-1] * cum_gex[1:]) < 0
    flip_candidates = strike_gex.index[1:][sign_changes] if len(cum_gex) > 1 else pd.Index([])

    flip_strike = float(flip_candidates[0]) if len(flip_candidates) > 0 else spot

    regime = "mean_revert" if net_gex > 0 else "momentum"

    return GEXAnalysis(
        net_gex_cr=float(net_gex),
        call_gex_cr=float(call_gex),
        put_gex_cr=float(put_gex),
        gex_flip_strike=flip_strike,
        regime=regime,
        per_strike=chain[["strike", "option_type", "oi", "iv", "delta", "gamma", "gex_cr"]],
        spot=spot,
    )


# ---------------------------------------------------------------------------
# OI Flow
# ---------------------------------------------------------------------------

def compute_oi_flow(
    current: pd.DataFrame, previous: pd.DataFrame, spot: float,
    risk_free: float = 0.065, as_of_date: pd.Timestamp | None = None,
) -> OIFlowAnalysis:
    """Compute delta-weighted OI change between two snapshots.

    Large put OI additions → institutional hedging → contrarian bullish.
    Large call OI additions → retail speculation → contrarian bearish.
    """
    # Merge on strike + option_type + expiry
    merged = current.merge(
        previous[["strike", "option_type", "expiry", "oi"]],
        on=["strike", "option_type", "expiry"],
        suffixes=("", "_prev"),
        how="left",
    )
    merged["oi_prev"] = merged["oi_prev"].fillna(0)
    merged["oi_change"] = merged["oi"] - merged["oi_prev"]

    is_call = (merged["option_type"] == "CE").values
    call_oi_chg = int(merged.loc[is_call, "oi_change"].sum())
    put_oi_chg = int(merged.loc[~is_call, "oi_change"].sum())

    # Delta-weight the OI changes
    nearest_expiry = sorted(merged["expiry"].unique())[0]
    near = merged[merged["expiry"] == nearest_expiry].copy()

    if not near.empty:
        exp_date = pd.Timestamp(nearest_expiry).tz_localize(None)
        if as_of_date is not None:
            ref_date = pd.Timestamp(as_of_date).tz_localize(None).normalize()
        else:
            ref_date = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
        T = max((exp_date - ref_date).days, 1) / 365.0
        strikes = near["strike"].values.astype(np.float64)
        prices = near["ltp"].values.astype(np.float64)
        is_call_arr = (near["option_type"] == "CE").values
        iv = _bs_iv_bisection(
            prices, spot, strikes,
            np.full(len(near), T), risk_free, is_call_arr,
        )
        valid = ~np.isnan(iv) & (iv > 0.01)
        delta = np.zeros(len(near))
        if valid.any():
            d, _ = _bs_greeks(
                spot, strikes[valid],
                np.full(valid.sum(), T), risk_free,
                iv[valid], is_call_arr[valid],
            )
            delta[valid] = d
        near["delta"] = delta
        net_delta_flow = float((near["oi_change"] * near["delta"]).sum())
    else:
        net_delta_flow = 0.0

    # Top strikes with OI changes
    top_calls = (
        merged[is_call].nlargest(5, "oi_change")[["strike", "oi_change"]]
        .to_dict("records")
    )
    top_puts = (
        merged[~is_call].nlargest(5, "oi_change")[["strike", "oi_change"]]
        .to_dict("records")
    )

    # Signal: put writing → bullish (contrarian), call writing → bearish
    # Normalize by total OI for comparability
    total_oi = max(merged["oi"].sum(), 1)
    score = -net_delta_flow / (total_oi * 0.001)  # flip sign: put delta is negative
    score = max(-1.0, min(1.0, score))

    direction = "bullish" if score > 0.1 else ("bearish" if score < -0.1 else "neutral")

    return OIFlowAnalysis(
        net_delta_flow=net_delta_flow,
        call_oi_change=call_oi_chg,
        put_oi_change=put_oi_chg,
        direction=direction,
        top_call_strikes=top_calls,
        top_put_strikes=top_puts,
        score=score,
    )


# ---------------------------------------------------------------------------
# Max Pain
# ---------------------------------------------------------------------------

def compute_max_pain(df: pd.DataFrame, spot: float) -> MaxPainAnalysis:
    """Compute max pain strike (where option buyers lose most).

    At expiry, for each candidate strike K:
      Call buyer loss = sum of call_OI × max(0, K_strike - K) for calls above K
      Put buyer loss = sum of put_OI × max(0, K - K_strike) for puts below K
    Total pain = call_loss + put_loss
    Max pain = K that maximizes total pain (= minimizes option buyer value).
    """
    nearest_expiry = sorted(df["expiry"].unique())[0]
    chain = df[df["expiry"] == nearest_expiry]

    calls = chain[chain["option_type"] == "CE"][["strike", "oi"]].copy()
    puts = chain[chain["option_type"] == "PE"][["strike", "oi"]].copy()

    strikes = sorted(chain["strike"].unique())
    if not strikes:
        return MaxPainAnalysis(spot, 0, "at")

    best_pain = -1
    best_strike = strikes[0]

    for K in strikes:
        # Call buyer intrinsic value at K
        call_value = calls.apply(
            lambda r: r["oi"] * max(0, r["strike"] - K) if r["strike"] > K else 0,
            axis=1,
        ).sum() if not calls.empty else 0

        # Put buyer intrinsic value at K
        put_value = puts.apply(
            lambda r: r["oi"] * max(0, K - r["strike"]) if r["strike"] < K else 0,
            axis=1,
        ).sum() if not puts.empty else 0

        # Total pain to buyers (we want to MAXIMIZE this, but actually max_pain
        # is the strike where total payoff to buyers is MINIMIZED)
        total_value = call_value + put_value

        # Actually, max pain is where option writer profits are maximized,
        # which is where option buyer VALUE is minimized.
        # Lower total_value → more pain for buyers
        if best_pain < 0 or total_value < best_pain:
            best_pain = total_value
            best_strike = K

    distance_pct = (best_strike - spot) / spot * 100
    direction = "above" if best_strike > spot else "below"

    return MaxPainAnalysis(
        max_pain_strike=best_strike,
        distance_pct=distance_pct,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# Futures Basis
# ---------------------------------------------------------------------------

_basis_history: dict[str, list[float]] = {}


def reset_basis_history() -> None:
    """Reset module-level basis history (call between CV folds / backtests)."""
    _basis_history.clear()


def compute_basis(
    spot: float, futures: float, days_to_expiry: int = 25,
    symbol: str = "",
) -> BasisAnalysis:
    """Compute futures basis (premium/discount) analysis."""
    basis = futures - spot
    basis_pct = (basis / spot) * (365 / max(days_to_expiry, 1)) * 100

    hist = _basis_history.setdefault(symbol, [])
    hist.append(basis)
    # Keep last 100 observations per symbol
    if len(hist) > 100:
        hist.pop(0)

    # Z-score vs recent history (require minimum 30 observations for statistical validity)
    if len(hist) >= 30:
        mean = np.mean(hist)
        std = max(np.std(hist, ddof=1), 0.01)
        zscore = (basis - mean) / std
    else:
        zscore = 0.0

    if zscore > 2.0:
        signal = "overleveraged_long"
    elif zscore < -2.0:
        signal = "overleveraged_short"
    else:
        signal = "neutral"

    return BasisAnalysis(
        basis_points=basis,
        basis_pct=basis_pct,
        basis_zscore=zscore,
        signal=signal,
    )


# ---------------------------------------------------------------------------
# Put-Call Ratio
# ---------------------------------------------------------------------------

def compute_pcr(df: pd.DataFrame) -> PCRAnalysis:
    """Compute put-call ratio (OI and volume weighted)."""
    nearest_expiry = sorted(df["expiry"].unique())[0]
    chain = df[df["expiry"] == nearest_expiry]

    calls = chain[chain["option_type"] == "CE"]
    puts = chain[chain["option_type"] == "PE"]

    call_oi = max(calls["oi"].sum(), 1)
    put_oi = max(puts["oi"].sum(), 1)
    pcr_oi = put_oi / call_oi

    call_vol = max(calls["volume"].sum(), 1)
    put_vol = max(puts["volume"].sum(), 1)
    pcr_vol = put_vol / call_vol

    if pcr_oi > 1.3:
        signal = "extreme_fear"  # contrarian bullish
    elif pcr_oi < 0.7:
        signal = "extreme_greed"  # contrarian bearish
    else:
        signal = "neutral"

    return PCRAnalysis(pcr_oi=pcr_oi, pcr_volume=pcr_vol, signal=signal)


# ---------------------------------------------------------------------------
# IV Term Structure
# ---------------------------------------------------------------------------

def compute_iv_term_structure(
    df: pd.DataFrame, spot: float, risk_free: float = 0.065,
    as_of_date: pd.Timestamp | None = None,
) -> IVTermStructure:
    """Compute ATM IV for near and far expiries."""
    expiries = sorted(df["expiry"].unique())
    if len(expiries) < 2:
        return IVTermStructure(0, 0, 1.0, "normal")

    # Use 1st and 2nd expiry (not last) — far expiries often illiquid
    near_exp, far_exp = expiries[0], expiries[min(1, len(expiries) - 1)]

    def _atm_iv(exp: str) -> float:
        chain = df[df["expiry"] == exp]
        # Find nearest ATM strike
        atm_strike = chain.iloc[
            (chain["strike"] - spot).abs().argsort()[:2]
        ]["strike"].iloc[0]
        atm = chain[chain["strike"] == atm_strike]
        if atm.empty:
            return 0
        # Use straddle mid
        ce = atm[atm["option_type"] == "CE"]
        pe = atm[atm["option_type"] == "PE"]
        straddle = 0
        if not ce.empty:
            straddle += ce.iloc[0]["ltp"]
        if not pe.empty:
            straddle += pe.iloc[0]["ltp"]

        exp_date = pd.Timestamp(exp).tz_localize(None)
        if as_of_date is not None:
            ref_date = pd.Timestamp(as_of_date).tz_localize(None).normalize()
        else:
            ref_date = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
        T = max((exp_date - ref_date).days, 1) / 365.0

        # Brenner-Subrahmanyam: sigma ≈ straddle / (S * sqrt(T)) * sqrt(2*pi)
        if spot > 0 and T > 0:
            return straddle / (spot * math.sqrt(T)) * math.sqrt(2 * math.pi)
        return 0

    near_iv = _atm_iv(near_exp)
    far_iv = _atm_iv(far_exp)

    slope = near_iv / max(far_iv, 0.001)

    if slope > 1.2:
        signal = "inverted"  # panic — near-term vol spiked
    elif slope < 0.8:
        signal = "steep"  # complacent — near vol cheap
    else:
        signal = "normal"

    return IVTermStructure(
        near_iv=near_iv, far_iv=far_iv, slope=slope, signal=signal,
    )


# ---------------------------------------------------------------------------
# Combined analysis
# ---------------------------------------------------------------------------

def analyze_snapshot(
    current: pd.DataFrame,
    previous: pd.DataFrame | None = None,
) -> MicrostructureSnapshot:
    """Run all analytics on a chain snapshot.

    Parameters
    ----------
    current : DataFrame from collector.snapshot_chain()
    previous : Previous snapshot (for OI flow computation), or None.
    """
    spot = current.iloc[0]["underlying_price"]
    futures = current.iloc[0]["futures_price"]
    symbol = current.iloc[0]["symbol"]
    timestamp = str(current.iloc[0]["timestamp"])

    gex = compute_gex(current, spot)
    max_pain = compute_max_pain(current, spot)
    basis = compute_basis(spot, futures, symbol=symbol)
    pcr = compute_pcr(current)
    iv_term = compute_iv_term_structure(current, spot)

    oi_flow = None
    if previous is not None:
        oi_flow = compute_oi_flow(current, previous, spot)

    return MicrostructureSnapshot(
        timestamp=timestamp,
        symbol=symbol,
        spot=spot,
        futures=futures,
        gex=gex,
        oi_flow=oi_flow,
        max_pain=max_pain,
        basis=basis,
        pcr=pcr,
        iv_term=iv_term,
    )
