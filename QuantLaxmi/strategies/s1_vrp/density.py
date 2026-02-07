"""Risk-Neutral Density Regime Strategy (RNDR) -- Futures Variant.

Trades index futures based on distributional features extracted from the
SANOS-calibrated volatility surface.  The core edge is the *skewness risk
premium*: retail option buyers systematically overpay for downside
protection, creating a gap between implied and realised skewness that
mean-reverts predictably.

See ``density_options.py`` for the options variant (bull put credit spreads).

Signal components
-----------------
1. **Skew Risk Premium**  (weight 0.40)
   physical_skewness(20d) − risk_neutral_skewness.
   Large positive gap → fear overpriced → contrarian LONG.

2. **Left Tail Weight**  (weight 0.25)
   P(K < μ − σ) from the RN density.
   Elevated vs history → crash premium too rich → LONG.

3. **Entropy Change**  (weight 0.20)
   ΔH = H_today − H_yesterday.
   Large positive ΔH → uncertainty spike → caution (negative).

4. **KL Divergence Direction** (weight 0.15)
   D_KL(q_today ‖ q_yesterday) × sign(Δskewness).
   Large KL with skew worsening → contrarian LONG (overreaction).

Entry : composite percentile ≥ entry_pctile  →  LONG index futures.
Exit  : hold_days reached  OR  composite drops below exit_pctile.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from core.pricing.risk_neutral import (
    DensitySnapshot,
    compute_snapshot,
    extract_density,
    kl_divergence,
    physical_skewness,
)
from core.pricing.sanos import SANOSResult, fit_sanos, prepare_nifty_chain
from strategies.s9_momentum.data import is_trading_day, get_fno
from core.data.store import MarketDataStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_LOOKBACK = 30
DEFAULT_ENTRY_PCTILE = 0.75
DEFAULT_EXIT_PCTILE = 0.40
DEFAULT_HOLD_DAYS = 5
DEFAULT_COST_BPS = 5.0
DEFAULT_PHYS_WINDOW = 20

# Composite signal weights
W_SKEW_PREMIUM = 0.40
W_LEFT_TAIL = 0.25
W_ENTROPY = 0.20
W_KL_DIRECTION = 0.15


# ---------------------------------------------------------------------------
# Daily observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DensityDayObs:
    """All features for one trading day / symbol."""

    date: date
    symbol: str
    spot: float
    atm_iv: float

    # RN density features (from SANOS)
    rn_skewness: float
    rn_kurtosis: float
    entropy: float
    left_tail: float
    right_tail: float

    # Physical (realised) features
    phys_skewness: float

    # Derived
    skew_premium: float       # phys - rn  (positive = fear overpriced)
    entropy_change: float     # ΔH vs yesterday
    kl_div: float             # D_KL(today ‖ yesterday)

    density_ok: bool


# ---------------------------------------------------------------------------
# Build daily series from F&O data
# ---------------------------------------------------------------------------

def _calibrate_density(
    store,
    d: date,
    symbol: str,
) -> tuple[DensitySnapshot | None, SANOSResult | None, float, float]:
    """Calibrate SANOS for a day/symbol and extract density snapshot.

    Returns (snapshot, sanos_result, spot, atm_iv).
    """
    try:
        fno = get_fno(store, d)
        if fno.empty:
            return None, None, 0.0, 0.0
    except Exception:
        return None, None, 0.0, 0.0

    chain_data = prepare_nifty_chain(fno, symbol=symbol, max_expiries=2)
    if chain_data is None:
        return None, None, 0.0, 0.0

    spot = chain_data["spot"]
    atm_vars = chain_data["atm_variances"]
    atm_iv = math.sqrt(max(atm_vars[0], 1e-12))

    try:
        result = fit_sanos(
            market_strikes=chain_data["market_strikes"],
            market_calls=chain_data["market_calls"],
            market_spreads=chain_data.get("market_spreads"),
            atm_variances=atm_vars,
            expiry_labels=chain_data["expiry_labels"],
            eta=0.50,
            n_model_strikes=100,
            K_min=0.7,
            K_max=1.5,
        )
        if not result.lp_success:
            return None, None, spot, atm_iv
    except Exception as e:
        logger.debug("SANOS failed for %s %s: %s", symbol, d, e)
        return None, None, spot, atm_iv

    # Extract density snapshot
    snap = compute_snapshot(result, d.isoformat(), symbol)
    return snap, result, spot, atm_iv


def build_density_series(
    store,
    start: date,
    end: date,
    symbol: str = "NIFTY",
    phys_window: int = DEFAULT_PHYS_WINDOW,
) -> list[DensityDayObs]:
    """Build daily density-feature series from F&O data.

    For each trading day:
    1. Calibrate SANOS → density snapshot
    2. Track spot prices → compute physical skewness
    3. Track yesterday's density → compute ΔH, KL divergence
    """
    series: list[DensityDayObs] = []
    spots: list[float] = []
    spot_dates: list[date] = []

    prev_q: np.ndarray | None = None
    prev_dK: float = 0.0
    prev_entropy: float = 0.0
    prev_rn_skew: float = 0.0

    d = start
    while d <= end:
        if not is_trading_day(d):
            d += timedelta(days=1)
            continue

        snap, result, spot, atm_iv = _calibrate_density(store, d, symbol)

        if snap is None or not snap.density_ok:
            d += timedelta(days=1)
            continue

        spots.append(spot)
        spot_dates.append(d)

        # Physical skewness from trailing consecutive-day log returns
        phys_skew = 0.0
        if len(spots) > phys_window:
            recent_spots = spots[-phys_window - 1:]
            recent_dates = spot_dates[-phys_window - 1:]
            # Only use returns between near-consecutive trading days
            # (gap <= 4 calendar days covers weekends and single holidays)
            log_rets = []
            for j in range(1, len(recent_spots)):
                gap = (recent_dates[j] - recent_dates[j - 1]).days
                if gap <= 4:
                    log_rets.append(math.log(recent_spots[j] / recent_spots[j - 1]))
            if len(log_rets) >= phys_window // 2:
                phys_skew = physical_skewness(np.array(log_rets))

        # KL divergence and entropy change vs yesterday
        K, q = extract_density(result, 0)
        dK = K[1] - K[0]
        kl = 0.0
        d_entropy = 0.0
        if prev_q is not None and len(prev_q) == len(q):
            kl = kl_divergence(q, prev_q, dK)
            d_entropy = snap.entropy - prev_entropy

        skew_premium = phys_skew - snap.rn_skewness

        obs = DensityDayObs(
            date=d,
            symbol=symbol,
            spot=spot,
            atm_iv=atm_iv,
            rn_skewness=snap.rn_skewness,
            rn_kurtosis=snap.rn_kurtosis,
            entropy=snap.entropy,
            left_tail=snap.left_tail,
            right_tail=snap.right_tail,
            phys_skewness=phys_skew,
            skew_premium=skew_premium,
            entropy_change=d_entropy,
            kl_div=kl,
            density_ok=snap.density_ok,
        )
        series.append(obs)

        prev_q = q.copy()
        prev_dK = dK
        prev_entropy = snap.entropy
        prev_rn_skew = snap.rn_skewness

        d += timedelta(days=1)

    return series


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def _rolling_percentile(values: list[float], idx: int, window: int) -> float:
    """Percentile rank of values[idx] within the trailing window."""
    start = max(0, idx - window + 1)
    w = values[start:idx + 1]
    if len(w) < 2:
        return 0.5
    current = w[-1]
    return sum(1 for v in w if v <= current) / len(w)


def _rolling_zscore(values: list[float], idx: int, window: int) -> float:
    """Z-score of values[idx] within the trailing window."""
    start = max(0, idx - window + 1)
    w = np.array(values[start:idx + 1])
    if len(w) < 3:
        return 0.0
    mu = np.mean(w)
    std = np.std(w, ddof=1)
    if std < 1e-12:
        return 0.0
    return float((w[-1] - mu) / std)


def compute_composite_signal(
    series: list[DensityDayObs],
    lookback: int = DEFAULT_LOOKBACK,
) -> list[float]:
    """Compute the composite signal for each day.

    Returns a list of floats (one per day in series).  Values > 0 are
    bullish, < 0 are bearish.  The signal is only meaningful after
    ``lookback`` days of history.
    """
    n = len(series)
    signals = [0.0] * n

    # Pre-extract feature vectors
    skew_premia = [o.skew_premium for o in series]
    left_tails = [o.left_tail for o in series]
    entropy_changes = [o.entropy_change for o in series]
    kl_divs = [o.kl_div for o in series]
    rn_skew_changes = [0.0] + [
        series[i].rn_skewness - series[i - 1].rn_skewness
        for i in range(1, n)
    ]

    for i in range(lookback, n):
        # 1. Skew Risk Premium — percentile rank (higher = more fear overpriced)
        srp_pctile = _rolling_percentile(skew_premia, i, lookback)

        # 2. Left Tail Weight — percentile rank (higher = more crash fear)
        lt_pctile = _rolling_percentile(left_tails, i, lookback)

        # 3. Entropy Change — z-score (positive = uncertainty spike = caution)
        ent_z = _rolling_zscore(entropy_changes, i, lookback)

        # 4. KL Direction — KL magnitude × sign of skew worsening
        #    Negative skew change (more fearful) + high KL = overreaction → bullish
        kl_z = _rolling_zscore(kl_divs, i, lookback)
        skew_dir = -1.0 if rn_skew_changes[i] < 0 else 1.0
        kl_directional = kl_z * skew_dir  # positive when fear + info shock

        # Composite: weighted sum, rescale to roughly [-1, 1]
        # Percentiles ∈ [0,1] → centre at 0.5 and scale to [-1,1]
        composite = (
            W_SKEW_PREMIUM * (2 * srp_pctile - 1)
            + W_LEFT_TAIL * (2 * lt_pctile - 1)
            + W_ENTROPY * (-ent_z / 3.0)        # dampen, negate (uncertainty = bad)
            + W_KL_DIRECTION * (kl_directional / 3.0)
        )
        signals[i] = composite

    return signals


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DensityTrade:
    """One completed round-trip trade."""

    symbol: str
    entry_date: date
    exit_date: date
    entry_spot: float
    exit_spot: float
    entry_signal: float
    pnl_pct: float
    hold_days: int
    exit_reason: str


@dataclass
class DensityBacktestResult:
    """Backtest output for one index."""

    symbol: str
    daily: list[DensityDayObs]
    signals: list[float]
    trades: list[DensityTrade]
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    win_rate: float = 0.0
    n_signals: int = 0


def run_density_backtest(
    series: list[DensityDayObs],
    lookback: int = DEFAULT_LOOKBACK,
    entry_pctile: float = DEFAULT_ENTRY_PCTILE,
    exit_pctile: float = DEFAULT_EXIT_PCTILE,
    hold_days: int = DEFAULT_HOLD_DAYS,
    cost_bps: float = DEFAULT_COST_BPS,
    symbol: str = "NIFTY",
) -> DensityBacktestResult:
    """Run the density regime strategy backtest.

    Entry: composite signal percentile ≥ entry_pctile (within lookback).
    Exit : max hold_days reached OR signal percentile < exit_pctile.
    """
    signals = compute_composite_signal(series, lookback)
    n = len(series)
    cost_frac = cost_bps / 10_000
    trades: list[DensityTrade] = []
    daily_pnl: list[float] = []  # ALL days from lookback onward

    in_trade = False
    entry_idx = 0
    n_signals = 0
    pending_entry = False  # T+1 execution: signal at close, enter next day

    for i in range(lookback, n):
        # Percentile-rank the composite signal itself for thresholding
        sig_pctile = _rolling_percentile(signals, i, lookback)

        # Execute pending entry from yesterday's signal
        if pending_entry and not in_trade:
            in_trade = True
            entry_idx = i
            pending_entry = False

        if not in_trade:
            if sig_pctile >= entry_pctile and signals[i] > 0:
                n_signals += 1
                pending_entry = True  # enter at next bar (T+1)
            daily_pnl.append(0.0)
        else:
            days_held = i - entry_idx
            # Daily mark-to-market return (relative to entry spot)
            spot_prev = series[i - 1].spot if i > entry_idx else series[entry_idx].spot
            day_ret = (series[i].spot - spot_prev) / series[entry_idx].spot

            should_exit = (
                days_held >= hold_days
                or sig_pctile < exit_pctile
            )
            if should_exit:
                entry_obs = series[entry_idx]
                exit_obs = series[i]
                raw_pnl = (exit_obs.spot - entry_obs.spot) / entry_obs.spot
                net_pnl = raw_pnl - cost_frac

                reason = "max_hold" if days_held >= hold_days else "signal_decay"
                trades.append(DensityTrade(
                    symbol=symbol,
                    entry_date=entry_obs.date,
                    exit_date=exit_obs.date,
                    entry_spot=entry_obs.spot,
                    exit_spot=exit_obs.spot,
                    entry_signal=signals[entry_idx],
                    pnl_pct=net_pnl,
                    hold_days=days_held,
                    exit_reason=reason,
                ))
                in_trade = False

            daily_pnl.append(day_ret)

    # Close open trade at end
    if in_trade:
        entry_obs = series[entry_idx]
        exit_obs = series[-1]
        raw_pnl = (exit_obs.spot - entry_obs.spot) / entry_obs.spot
        net_pnl = raw_pnl - cost_frac
        trades.append(DensityTrade(
            symbol=symbol,
            entry_date=entry_obs.date,
            exit_date=exit_obs.date,
            entry_spot=entry_obs.spot,
            exit_spot=exit_obs.spot,
            entry_signal=signals[entry_idx],
            pnl_pct=net_pnl,
            hold_days=len(series) - 1 - entry_idx,
            exit_reason="end_of_data",
        ))

    # --- Metrics ---
    if trades:
        equity = 1.0
        eq_curve = [1.0]
        for t in trades:
            equity *= (1 + t.pnl_pct)
            eq_curve.append(equity)

        total_ret = (equity - 1) * 100

        # Annualised
        if len(series) > 1:
            n_days = (series[-1].date - series[0].date).days
            if n_days > 0:
                ann_ret = ((equity) ** (365 / n_days) - 1) * 100
            else:
                ann_ret = 0.0
        else:
            ann_ret = 0.0

        # Sharpe from ALL daily returns (including flat days, ddof=1)
        sharpe = 0.0
        pnl_arr = np.array(daily_pnl)
        if len(pnl_arr) > 1:
            std = np.std(pnl_arr, ddof=1)
            if std > 0:
                sharpe = float(np.mean(pnl_arr) / std * math.sqrt(252))

        # Max drawdown
        peak = 1.0
        max_dd = 0.0
        for eq in eq_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        win_rate = sum(1 for t in trades if t.pnl_pct > 0) / len(trades)
    else:
        total_ret = ann_ret = sharpe = max_dd = win_rate = 0.0

    return DensityBacktestResult(
        symbol=symbol,
        daily=series,
        signals=signals,
        trades=trades,
        total_return_pct=total_ret,
        annual_return_pct=ann_ret,
        sharpe=sharpe,
        max_dd_pct=max_dd * 100,
        win_rate=win_rate,
        n_signals=n_signals,
    )


# ---------------------------------------------------------------------------
# Multi-index backtest
# ---------------------------------------------------------------------------

def run_multi_index_density_backtest(
    store,
    start: date,
    end: date,
    symbols: list[str] | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    entry_pctile: float = DEFAULT_ENTRY_PCTILE,
    exit_pctile: float = DEFAULT_EXIT_PCTILE,
    hold_days: int = DEFAULT_HOLD_DAYS,
    cost_bps: float = DEFAULT_COST_BPS,
    phys_window: int = DEFAULT_PHYS_WINDOW,
) -> dict[str, DensityBacktestResult]:
    """Run density backtest for multiple indices."""
    if symbols is None:
        symbols = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]

    results: dict[str, DensityBacktestResult] = {}

    for sym in symbols:
        print(f"\n  {sym}: building density series...", end="", flush=True)
        series = build_density_series(store, start, end, sym, phys_window)
        print(f" {len(series)} days", end="", flush=True)

        if len(series) < lookback + 10:
            print(f" (too few days, skipping)")
            continue

        result = run_density_backtest(
            series,
            lookback=lookback,
            entry_pctile=entry_pctile,
            exit_pctile=exit_pctile,
            hold_days=hold_days,
            cost_bps=cost_bps,
            symbol=sym,
        )
        results[sym] = result
        print(
            f" → {len(result.trades)} trades, "
            f"{result.total_return_pct:+.2f}%, "
            f"Sharpe {result.sharpe:.2f}, "
            f"WR {result.win_rate:.0%}"
        )

    return results
