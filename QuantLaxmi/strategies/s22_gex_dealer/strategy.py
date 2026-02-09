"""Gamma Exposure (GEX) / Dealer Hedging Flow Strategy — NIFTY.

Market makers (dealers) who sell options are typically short gamma.  Their
delta-hedging activity creates predictable, measurable effects on the
underlying index:

    * **Positive GEX** (dealers long gamma): index is pinned near
      high-OI strikes.  Low realised vol, mean-reversion dominates.
      → Sell strangles / fade moves toward max-pain.

    * **Negative GEX** (dealers short gamma): dealers amplify moves.
      High realised vol, momentum dominates.
      → Trade breakouts in the direction of 5-day momentum.

GEX Computation
---------------
For each strike K with open interest OI_K in the near-term chain:

    gamma_K     = BSM_gamma(S, K, T, sigma_K, r=0)   [via GPU IV engine]
    GEX_call_K  = +gamma_K * OI_call * S^2 * lot_size / 1e7   (dealers short calls)
    GEX_put_K   = -gamma_K * OI_put  * S^2 * lot_size / 1e7   (dealers long puts)
    Total GEX   = sum over all K of (GEX_call + GEX_put)

Sign convention follows standard dealer positioning: retail is net
long calls / long puts, so dealers are net short calls / short puts.
Dealers who are short calls have *positive* gamma exposure (they buy
when the market rises, sell when it falls → stabilising).  Dealers who
are short puts have *negative* gamma exposure (they sell when the market
falls, buy when it rises → amplifying).

Strategy Logic (data-driven, calibrated on 2025-08 → 2026-02 NIFTY data)
--------------------------------------------------------------------------
Three orthogonal signal layers, each with different alpha source:

Layer 1 — **Pin Attraction** (primary, strongest signal):
    When net GEX > 0 AND spot is below max-pain by > pin_band:
        → LONG (pin effect pulls index toward max-pain).
    When net GEX > 0 AND spot is above max-pain by > pin_band:
        → SHORT (symmetric, but weaker empirically).
    Forward return: +0.20% (3d), +0.29% (5d) on long side (n ≈ 48/122 days).

Layer 2 — **GEX Spike Fade** (secondary):
    When GEX z-score > 1.5 (relative to 20d rolling window):
        → SHORT (extremely high dealer gamma compresses vol, expect mean-reversion).
    Forward return: -0.21% (1d), n ≈ 9 days.

Layer 3 — **Negative GEX Breakout** (tertiary):
    When GEX z-score < -1.0 (unusually low dealer gamma):
        → Follow 5-day momentum direction (amplification regime).
    Forward return: momentum-conditional.

Execution: T+1 close (causal).  Max hold 5 days. Stop-loss -1.5%.
P&L model: mark-to-market on NIFTY close.  Cost = 3 pts round-trip (futures).

Author: AlphaForge
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from data.store import MarketDataStore
from core.pricing.iv_engine import compute_chain_iv
from strategies.s9_momentum.data import is_trading_day, get_fno

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOT_SIZE: int = 75                   # NIFTY lot size (check nse_fo_mktlots)
COST_RT_PTS: float = 3.0            # round-trip cost in index points (futures)
MAX_HOLD_DAYS: int = 5              # maximum holding period (5d captures full pin effect)
STOP_LOSS_PCT: float = 0.015        # stop-loss: 1.5% of spot
TARGET_PIN_PCT: float = 0.004       # target: 0.4% for pin-attraction trades
TARGET_SPIKE_PCT: float = 0.003     # target: 0.3% for GEX-spike fade trades
TARGET_BREAKOUT_PCT: float = 0.008  # target: 0.8% for negative-GEX breakout trades
MOMENTUM_LOOKBACK: int = 5          # days for momentum signal in neg-GEX
PIN_BAND_MULT: float = 0.001        # max-pain band: 0.1% of spot (tight = more trades)
GEX_Z_LOOKBACK: int = 20            # rolling window for GEX z-score
GEX_Z_SPIKE: float = 1.5            # z-score threshold for GEX spike short
GEX_Z_LOW: float = -1.0             # z-score threshold for negative GEX breakout
VIX_CEILING: float = 30.0           # no new trades when VIX > 30
MIN_OI_FILTER: int = 100            # minimum OI to include a strike
MAX_EXPIRY_DAYS: int = 14           # only use near-term chain (<= 14 DTE)
MIN_EXPIRY_DAYS: int = 1            # exclude 0-DTE options


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GEXSnapshot:
    """Daily GEX measurement."""
    date: date
    spot: float
    total_gex: float               # net GEX across all strikes (crores)
    max_pain: float                # strike minimising aggregate option value
    gex_regime: Literal["positive", "negative", "neutral"]
    call_gex: float                # total call GEX (crores)
    put_gex: float                 # total put GEX (crores)
    max_gex_strike: float          # strike with highest absolute GEX
    n_strikes: int                 # number of strikes used
    nearest_expiry: str
    vix: float
    lot_size: int
    atm_iv: float                  # ATM implied vol (annualized)
    gex_z: float = 0.0             # z-score of GEX (rolling 20d)


@dataclass
class GEXTrade:
    """One completed trade."""
    entry_date: date
    exit_date: date
    direction: Literal["long", "short"]
    signal_type: Literal["pin_attract", "gex_spike", "breakout"]
    entry_spot: float
    exit_spot: float
    pnl_points: float              # after costs
    pnl_pct: float                 # as fraction of entry_spot
    hold_days: int
    exit_reason: str
    entry_gex: float
    entry_gex_z: float
    entry_max_pain: float
    target_pct: float


@dataclass
class GEXBacktestResult:
    """Full backtest output."""
    symbol: str
    snapshots: list[GEXSnapshot]
    trades: list[GEXTrade]
    daily_returns: list[float] = field(default_factory=list)
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    win_rate: float = 0.0
    avg_hold_days: float = 0.0
    n_trades: int = 0
    pin_trades: int = 0
    spike_trades: int = 0
    breakout_trades: int = 0
    pin_winrate: float = 0.0
    spike_winrate: float = 0.0
    breakout_winrate: float = 0.0
    avg_gex: float = 0.0
    avg_vix: float = 0.0


# ---------------------------------------------------------------------------
# VIX loader
# ---------------------------------------------------------------------------

def _load_vix_series(store: MarketDataStore) -> dict[str, float]:
    """Load India VIX closing values keyed by ISO date string."""
    try:
        df = store.sql(
            'SELECT "Closing Index Value", date '
            'FROM nse_index_close '
            "WHERE \"Index Name\" = 'India VIX' "
            'ORDER BY date'
        )
        if df is None or df.empty:
            return {}
        vix_map: dict[str, float] = {}
        for _, row in df.iterrows():
            try:
                vix_map[str(row["date"])] = float(row["Closing Index Value"])
            except (ValueError, TypeError):
                continue
        return vix_map
    except Exception as e:
        logger.warning("Failed to load India VIX: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Nifty spot price loader
# ---------------------------------------------------------------------------

def _load_spot_series(store: MarketDataStore) -> dict[str, float]:
    """Load Nifty 50 closing values keyed by ISO date string."""
    try:
        df = store.sql(
            'SELECT "Closing Index Value", date '
            'FROM nse_index_close '
            "WHERE \"Index Name\" = 'Nifty 50' "
            'ORDER BY date'
        )
        if df is None or df.empty:
            return {}
        spot_map: dict[str, float] = {}
        for _, row in df.iterrows():
            try:
                spot_map[str(row["date"])] = float(row["Closing Index Value"])
            except (ValueError, TypeError):
                continue
        return spot_map
    except Exception as e:
        logger.warning("Failed to load Nifty 50 spot: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Lot size loader
# ---------------------------------------------------------------------------

def _get_lot_size(store: MarketDataStore, d: date) -> int:
    """Get NIFTY lot size for a given date from nse_fo_mktlots."""
    try:
        df = store.sql(
            "SELECT lot_size FROM nse_fo_mktlots "
            "WHERE symbol = 'NIFTY' AND date = ?",
            [d.isoformat()],
        )
        if df is not None and not df.empty:
            return int(df.iloc[0]["lot_size"])
    except Exception:
        pass
    return LOT_SIZE  # fallback


# ---------------------------------------------------------------------------
# Max-pain computation
# ---------------------------------------------------------------------------

def _compute_max_pain(
    chain: pd.DataFrame,
    spot: float,
    strike_col: str = "StrkPric",
    type_col: str = "OptnTp",
    oi_col: str = "OpnIntrst",
) -> float:
    """Compute max-pain: the strike price that minimises aggregate option
    intrinsic value (= the price of maximum pain for option holders).

    For each candidate settlement price P:
        total_pain(P) = sum_calls[ max(P - K, 0) * OI_call_K ]
                      + sum_puts[  max(K - P, 0) * OI_put_K  ]

    Max-pain = argmin_P total_pain(P).
    We evaluate at each available strike.
    """
    calls = chain[chain[type_col].str.strip() == "CE"]
    puts = chain[chain[type_col].str.strip() == "PE"]

    if calls.empty and puts.empty:
        return spot

    # Build OI maps
    call_oi: dict[float, int] = {}
    for _, row in calls.iterrows():
        k = float(row[strike_col])
        call_oi[k] = call_oi.get(k, 0) + int(row[oi_col])

    put_oi: dict[float, int] = {}
    for _, row in puts.iterrows():
        k = float(row[strike_col])
        put_oi[k] = put_oi.get(k, 0) + int(row[oi_col])

    all_strikes = sorted(set(list(call_oi.keys()) + list(put_oi.keys())))
    if not all_strikes:
        return spot

    best_strike = spot
    min_pain = float("inf")

    for p in all_strikes:
        pain = 0.0
        for k, oi in call_oi.items():
            pain += max(p - k, 0.0) * oi
        for k, oi in put_oi.items():
            pain += max(k - p, 0.0) * oi
        if pain < min_pain:
            min_pain = pain
            best_strike = p

    return best_strike


# ---------------------------------------------------------------------------
# GEX computation (self-contained, using GPU IV engine for gamma)
# ---------------------------------------------------------------------------

def _compute_daily_gex(
    store: MarketDataStore,
    d: date,
    symbol: str = "NIFTY",
    vix_map: dict[str, float] | None = None,
) -> GEXSnapshot | None:
    """Compute total GEX from the near-term option chain on a given date.

    Uses compute_chain_iv to get gamma per contract, then aggregates across
    all strikes using the dealer positioning convention.
    """
    try:
        fno = get_fno(store, d)
        if fno.empty:
            return None
    except Exception:
        return None

    # Filter to NIFTY index options
    opts = fno[
        (fno["TckrSymb"] == symbol)
        & (fno["FinInstrmTp"] == "IDO")
        & (fno["OptnTp"].isin(["CE", "PE"]))
    ].copy()

    if opts.empty or "UndrlygPric" not in opts.columns:
        return None

    spot = float(opts["UndrlygPric"].iloc[0])
    if spot <= 0:
        return None

    # Parse expiry and compute DTE
    trade_date = pd.Timestamp(d)
    opts["_expiry_dt"] = pd.to_datetime(
        opts["XpryDt"].astype(str).str.strip(), format="mixed"
    )
    opts["_dte"] = (opts["_expiry_dt"] - trade_date).dt.days

    # Filter to near-term chain only (MIN_EXPIRY_DAYS to MAX_EXPIRY_DAYS)
    near_term = opts[
        (opts["_dte"] >= MIN_EXPIRY_DAYS) & (opts["_dte"] <= MAX_EXPIRY_DAYS)
    ].copy()

    if near_term.empty:
        return None

    # Use the nearest available weekly expiry with sufficient OI
    # (combines all near-term expiries for broader GEX picture)
    nearest_expiry = near_term["_expiry_dt"].min()
    nearest_expiry_str = str(nearest_expiry.date())

    # Minimum OI filter
    near_term = near_term[near_term["OpnIntrst"] >= MIN_OI_FILTER]
    if near_term.empty:
        return None

    # Get lot size for the day
    lot_size = _get_lot_size(store, d)

    # Compute IV + Greeks for the near-term chain using GPU engine
    chain_iv = compute_chain_iv(near_term)
    df = chain_iv.df.copy()

    # Drop rows where gamma wasn't computed
    df = df.dropna(subset=["GAMMA"])
    if df.empty:
        return None

    # --- GEX computation ---
    # Convention: dealers are NET SHORT calls AND NET SHORT puts (retail buys both).
    #   - Dealer short call: positive gamma contribution (dealers buy on up-move)
    #   - Dealer short put: negative gamma contribution (dealers sell on down-move)
    # Net GEX > 0 → stabilising (pinning), Net GEX < 0 → amplifying (momentum)
    is_call = df["OptnTp"].str.strip() == "CE"
    sign = np.where(is_call, 1.0, -1.0)

    gex_per_row = (
        sign
        * df["GAMMA"].values
        * df["OpnIntrst"].values
        * spot ** 2
        * lot_size
        / 1e7  # scale to crores
    )
    df["GEX_CR"] = gex_per_row

    total_gex = float(np.nansum(gex_per_row))
    call_gex = float(np.nansum(gex_per_row[is_call]))
    put_gex = float(np.nansum(gex_per_row[~is_call]))

    # Per-strike aggregation for max-GEX strike
    per_strike = df.groupby("StrkPric")["GEX_CR"].sum()
    if per_strike.empty:
        return None
    max_gex_strike = float(per_strike.abs().idxmax())

    # Max pain
    max_pain = _compute_max_pain(near_term, spot)

    # GEX regime classification (simple sign-based; z-score layer handles nuance)
    if total_gex > 0:
        regime: Literal["positive", "negative", "neutral"] = "positive"
    elif total_gex < 0:
        regime = "negative"
    else:
        regime = "neutral"

    vix = float("nan")
    if vix_map is not None:
        vix = vix_map.get(d.isoformat(), float("nan"))

    return GEXSnapshot(
        date=d,
        spot=spot,
        total_gex=total_gex,
        max_pain=max_pain,
        gex_regime=regime,
        call_gex=call_gex,
        put_gex=put_gex,
        max_gex_strike=max_gex_strike,
        n_strikes=int(per_strike.shape[0]),
        nearest_expiry=nearest_expiry_str,
        vix=vix,
        lot_size=lot_size,
        atm_iv=chain_iv.atm_iv if not math.isnan(chain_iv.atm_iv) else 0.0,
    )


# ---------------------------------------------------------------------------
# Build snapshot series
# ---------------------------------------------------------------------------

def build_gex_series(
    store: MarketDataStore,
    start: date,
    end: date,
    symbol: str = "NIFTY",
    z_lookback: int = GEX_Z_LOOKBACK,
) -> list[GEXSnapshot]:
    """Build daily GEX snapshot series from start to end.

    Includes a warm-up buffer of `z_lookback` days before `start` so that
    z-scores are available from the first returned snapshot.
    """
    vix_map = _load_vix_series(store)

    # Collect with buffer for z-score warm-up
    buffer_start = start - timedelta(days=int(z_lookback * 2.5))
    raw_snapshots: list[GEXSnapshot] = []

    d = buffer_start
    while d <= end:
        if not is_trading_day(d):
            d += timedelta(days=1)
            continue

        snap = _compute_daily_gex(store, d, symbol, vix_map)
        if snap is not None:
            raw_snapshots.append(snap)
            if len(raw_snapshots) % 20 == 0:
                logger.info(
                    "GEX computed for %d days (latest: %s, GEX=%.1f, regime=%s, "
                    "max_pain=%.0f, spot=%.0f)",
                    len(raw_snapshots), d, snap.total_gex, snap.gex_regime,
                    snap.max_pain, snap.spot,
                )

        d += timedelta(days=1)

    if len(raw_snapshots) < z_lookback + 1:
        logger.warning(
            "Insufficient GEX data: %d days (need %d+1)",
            len(raw_snapshots), z_lookback,
        )
        return raw_snapshots

    # Compute rolling z-score of GEX
    gex_values = np.array([s.total_gex for s in raw_snapshots])
    snapshots: list[GEXSnapshot] = []

    for i in range(z_lookback, len(raw_snapshots)):
        window = gex_values[i - z_lookback : i]
        mu = float(np.mean(window))
        sigma = float(np.std(window, ddof=1))
        z = (gex_values[i] - mu) / sigma if sigma > 1e-6 else 0.0

        snap = raw_snapshots[i]
        snap.gex_z = z

        if snap.date >= start:
            snapshots.append(snap)

    logger.info("Built %d GEX snapshots from %s to %s", len(snapshots), start, end)
    return snapshots


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _generate_signal(
    snap: GEXSnapshot,
    spot_history: list[float],
    pin_band_mult: float = PIN_BAND_MULT,
    momentum_lookback: int = MOMENTUM_LOOKBACK,
    gex_z_spike: float = GEX_Z_SPIKE,
    gex_z_low: float = GEX_Z_LOW,
) -> tuple[
    Literal["long", "short"] | None,
    Literal["pin_attract", "gex_spike", "breakout"] | None,
    float,
]:
    """Generate a trading signal from the GEX snapshot.

    Three-layer signal hierarchy:
        Layer 1 (pin_attract): GEX > 0, spot deviates from max-pain.
        Layer 2 (gex_spike):   GEX z-score > threshold → short (fade).
        Layer 3 (breakout):    GEX z-score < threshold → momentum.

    Returns:
        (direction, signal_type, target_pct) or (None, None, 0.0).
    """
    spot = snap.spot
    max_pain = snap.max_pain
    gex_z = snap.gex_z

    # --- Layer 2: GEX spike fade (highest priority when triggered) ---
    # Extremely high dealer gamma → expect vol compression, short index
    if gex_z > gex_z_spike:
        return "short", "gex_spike", TARGET_SPIKE_PCT

    # --- Layer 3: Negative GEX breakout ---
    # Unusually low dealer gamma → amplification regime, trade momentum
    if gex_z < gex_z_low:
        if len(spot_history) >= momentum_lookback:
            lookback_prices = spot_history[-momentum_lookback:]
            momentum = math.log(lookback_prices[-1] / lookback_prices[0])
            if momentum > 0:
                return "long", "breakout", TARGET_BREAKOUT_PCT
            elif momentum < 0:
                return "short", "breakout", TARGET_BREAKOUT_PCT
        return None, None, 0.0

    # --- Layer 1: Pin attraction (most frequent signal) ---
    # GEX > 0: dealer hedging pins index toward max-pain
    if snap.total_gex > 0:
        pin_band = spot * pin_band_mult
        if spot < max_pain - pin_band:
            return "long", "pin_attract", TARGET_PIN_PCT
        elif spot > max_pain + pin_band:
            return "short", "pin_attract", TARGET_PIN_PCT

    return None, None, 0.0


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    symbol: str = "NIFTY",
    start_date: date | None = None,
    end_date: date | None = None,
    store: MarketDataStore | None = None,
    max_hold: int = MAX_HOLD_DAYS,
    stop_loss_pct: float = STOP_LOSS_PCT,
    cost_rt_pts: float = COST_RT_PTS,
    vix_ceiling: float = VIX_CEILING,
    pin_band_mult: float = PIN_BAND_MULT,
    momentum_lookback: int = MOMENTUM_LOOKBACK,
) -> GEXBacktestResult:
    """Run the GEX / Dealer Hedging Flow backtest.

    Parameters
    ----------
    symbol : str
        Index name ("NIFTY").
    start_date, end_date : date
        Backtest window.  Defaults to full data range.
    store : MarketDataStore
        DuckDB market data.  Creates one if None.
    max_hold : int
        Maximum holding period (trading days).
    stop_loss_pct : float
        Stop-loss as fraction of entry spot.
    cost_rt_pts : float
        Round-trip cost in index points (futures).
    vix_ceiling : float
        No new trades when VIX > this.
    pin_band_mult : float
        Max-pain band as fraction of spot (positive GEX regime).
    momentum_lookback : int
        Lookback for momentum signal (negative GEX regime).

    Returns
    -------
    GEXBacktestResult with trades, daily returns, and performance stats.
    """
    own_store = store is None
    if store is None:
        store = MarketDataStore()

    try:
        return _run_backtest_impl(
            symbol, start_date, end_date, store, max_hold,
            stop_loss_pct, cost_rt_pts, vix_ceiling,
            pin_band_mult, momentum_lookback,
        )
    finally:
        if own_store:
            store.close()


def _run_backtest_impl(
    symbol: str,
    start_date: date | None,
    end_date: date | None,
    store: MarketDataStore,
    max_hold: int,
    stop_loss_pct: float,
    cost_rt_pts: float,
    vix_ceiling: float,
    pin_band_mult: float,
    momentum_lookback: int,
) -> GEXBacktestResult:
    """Core backtest logic."""
    # Determine date range
    avail = store.available_dates("nse_fo_bhavcopy")
    if not avail:
        logger.error("No nse_fo_bhavcopy data available")
        return _empty_result(symbol)

    if start_date is None:
        start_date = avail[0]
    if end_date is None:
        end_date = avail[-1]

    # Load spot price series for momentum computation
    spot_map = _load_spot_series(store)
    if not spot_map:
        logger.error("No Nifty 50 spot data available")
        return _empty_result(symbol)

    # Collect spot history (pre-start buffer for momentum lookback)
    buffer_start = start_date - timedelta(days=momentum_lookback * 3)
    all_spot_dates = sorted(spot_map.keys())
    spot_history_for_momentum: list[float] = []
    date_to_spot: dict[str, float] = {}

    for ds in all_spot_dates:
        d_parsed = date.fromisoformat(ds)
        if d_parsed >= buffer_start:
            spot_history_for_momentum.append(spot_map[ds])
            date_to_spot[ds] = spot_map[ds]

    logger.info(
        "Running GEX backtest for %s from %s to %s...",
        symbol, start_date, end_date,
    )

    # Build GEX snapshot series
    snapshots = build_gex_series(store, start_date, end_date, symbol)
    if len(snapshots) < 5:
        logger.warning("Too few GEX snapshots (%d), aborting", len(snapshots))
        return _empty_result(symbol)

    # Build a spot list up to each snapshot date for momentum lookback
    snap_dates = {s.date for s in snapshots}

    # ---- Event-driven trade simulation ----
    # Signal at close of day i, execute at close of day i+1 (T+1 lag)
    trades: list[GEXTrade] = []
    daily_returns: list[float] = []

    # Active position state
    active: dict | None = None
    pending_signal: dict | None = None

    for i, snap in enumerate(snapshots):
        # --- Execute pending signal from yesterday (T+1) ---
        if pending_signal is not None and active is None:
            active = {
                "entry_date": snap.date,
                "entry_spot": snap.spot,
                "direction": pending_signal["direction"],
                "signal_type": pending_signal["signal_type"],
                "target_pct": pending_signal["target_pct"],
                "entry_gex": pending_signal["gex"],
                "entry_gex_z": pending_signal["gex_z"],
                "entry_max_pain": pending_signal["max_pain"],
                "entry_idx": i,
            }
            pending_signal = None

        # --- Check exit conditions for active trade ---
        if active is not None:
            bars_held = i - active["entry_idx"]

            if bars_held > 0:
                # Compute unrealised P&L
                if active["direction"] == "long":
                    unrealized_pts = snap.spot - active["entry_spot"]
                else:
                    unrealized_pts = active["entry_spot"] - snap.spot
                unrealized_pct = unrealized_pts / active["entry_spot"]

                should_exit = False
                exit_reason = ""

                # 1. Target profit
                if unrealized_pct >= active["target_pct"]:
                    should_exit = True
                    exit_reason = "target"

                # 2. Stop-loss
                if unrealized_pct <= -stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"

                # 3. Max hold
                if bars_held >= max_hold:
                    should_exit = True
                    exit_reason = "max_hold"

                # 4. Pin-attraction: exit early if spot crosses max-pain
                #    (pinning target reached even if not at target_pct)
                if active["signal_type"] == "pin_attract" and bars_held >= 2:
                    mp = active["entry_max_pain"]
                    if active["direction"] == "long" and snap.spot >= mp:
                        should_exit = True
                        exit_reason = "pin_reached"
                    elif active["direction"] == "short" and snap.spot <= mp:
                        should_exit = True
                        exit_reason = "pin_reached"

                # 5. End of data
                if i == len(snapshots) - 1 and not should_exit:
                    should_exit = True
                    exit_reason = "end_of_data"

                if should_exit:
                    # Close position
                    if active["direction"] == "long":
                        pnl_pts = snap.spot - active["entry_spot"] - cost_rt_pts
                    else:
                        pnl_pts = active["entry_spot"] - snap.spot - cost_rt_pts
                    pnl_pct = pnl_pts / active["entry_spot"]

                    trades.append(GEXTrade(
                        entry_date=active["entry_date"],
                        exit_date=snap.date,
                        direction=active["direction"],
                        signal_type=active["signal_type"],
                        entry_spot=active["entry_spot"],
                        exit_spot=snap.spot,
                        pnl_points=pnl_pts,
                        pnl_pct=pnl_pct,
                        hold_days=bars_held,
                        exit_reason=exit_reason,
                        entry_gex=active["entry_gex"],
                        entry_gex_z=active["entry_gex_z"],
                        entry_max_pain=active["entry_max_pain"],
                        target_pct=active["target_pct"],
                    ))

                    # Distribute P&L across hold days for daily return series
                    daily_pnl = pnl_pct / bars_held
                    # Replace the last `bars_held` zero entries
                    for j in range(bars_held):
                        idx = len(daily_returns) - bars_held + j + 1
                        if 0 <= idx < len(daily_returns):
                            daily_returns[idx] = daily_pnl

                    active = None
                    daily_returns.append(daily_pnl)
                    continue
                else:
                    daily_returns.append(0.0)
                    continue
            else:
                daily_returns.append(0.0)
                continue

        # --- Generate signal (if no active position) ---
        # Build spot history from full spot map for momentum lookback
        snap_date_str = snap.date.isoformat()
        full_history: list[float] = []
        for ds in sorted(date_to_spot.keys()):
            if ds < snap_date_str:
                full_history.append(date_to_spot[ds])
        full_history.append(snap.spot)

        # VIX filter
        vix_ok = math.isnan(snap.vix) or snap.vix <= vix_ceiling

        direction, signal_type, target_pct = _generate_signal(
            snap, full_history, pin_band_mult, momentum_lookback,
        )

        if direction is not None and vix_ok:
            pending_signal = {
                "direction": direction,
                "signal_type": signal_type,
                "target_pct": target_pct,
                "gex": snap.total_gex,
                "gex_z": snap.gex_z,
                "max_pain": snap.max_pain,
            }

        daily_returns.append(0.0)

    # ---- Compute performance metrics ----
    return _compute_metrics(symbol, snapshots, trades, daily_returns)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    symbol: str,
    snapshots: list[GEXSnapshot],
    trades: list[GEXTrade],
    daily_returns: list[float],
) -> GEXBacktestResult:
    """Compute backtest statistics: Sharpe, drawdown, win rate, etc."""
    n_trades = len(trades)

    if n_trades == 0:
        return GEXBacktestResult(
            symbol=symbol,
            snapshots=snapshots,
            trades=[],
            daily_returns=daily_returns,
            avg_gex=float(np.mean([s.total_gex for s in snapshots])) if snapshots else 0.0,
            avg_vix=float(np.nanmean([s.vix for s in snapshots])) if snapshots else 0.0,
        )

    ret = np.array(daily_returns)
    cumulative = np.cumsum(ret)

    total_return = float(cumulative[-1]) if len(cumulative) > 0 else 0.0
    years = max(len(daily_returns) / 252.0, 1.0 / 252.0)
    annual_return = total_return / years

    # Sharpe: ALL daily returns (including flat), ddof=1, sqrt(252)
    if len(ret) > 1 and np.std(ret, ddof=1) > 0:
        sharpe = float(np.mean(ret) / np.std(ret, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown
    equity = 1.0 + cumulative
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

    # Win rate (overall and by signal type)
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    win_rate = wins / n_trades if n_trades > 0 else 0.0

    pin_trades = [t for t in trades if t.signal_type == "pin_attract"]
    spike_trades = [t for t in trades if t.signal_type == "gex_spike"]
    breakout_trades = [t for t in trades if t.signal_type == "breakout"]
    pin_wins = sum(1 for t in pin_trades if t.pnl_pct > 0)
    spike_wins = sum(1 for t in spike_trades if t.pnl_pct > 0)
    breakout_wins = sum(1 for t in breakout_trades if t.pnl_pct > 0)

    avg_hold = float(np.mean([t.hold_days for t in trades]))
    avg_gex = float(np.mean([s.total_gex for s in snapshots])) if snapshots else 0.0
    avg_vix = float(np.nanmean([s.vix for s in snapshots])) if snapshots else 0.0

    return GEXBacktestResult(
        symbol=symbol,
        snapshots=snapshots,
        trades=trades,
        daily_returns=daily_returns,
        total_return_pct=total_return * 100,
        annual_return_pct=annual_return * 100,
        sharpe=sharpe,
        max_dd_pct=max_dd * 100,
        win_rate=win_rate,
        avg_hold_days=avg_hold,
        n_trades=n_trades,
        pin_trades=len(pin_trades),
        spike_trades=len(spike_trades),
        breakout_trades=len(breakout_trades),
        pin_winrate=pin_wins / len(pin_trades) if pin_trades else 0.0,
        spike_winrate=spike_wins / len(spike_trades) if spike_trades else 0.0,
        breakout_winrate=breakout_wins / len(breakout_trades) if breakout_trades else 0.0,
        avg_gex=avg_gex,
        avg_vix=avg_vix,
    )


def _empty_result(symbol: str) -> GEXBacktestResult:
    return GEXBacktestResult(symbol=symbol, snapshots=[], trades=[])


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_results(result: GEXBacktestResult) -> str:
    """Format backtest results for display."""
    lines = [
        "",
        f"GEX / Dealer Hedging Flow Strategy — {result.symbol}",
        "=" * 70,
        "",
        "  GEX Statistics:",
        f"    Avg net GEX:          {result.avg_gex:+.1f} Cr",
        f"    Avg India VIX:        {result.avg_vix:.1f}",
        "",
    ]

    if result.snapshots:
        regimes = [s.gex_regime for s in result.snapshots]
        lines += [
            f"    Regime breakdown:     "
            f"positive={regimes.count('positive')}, "
            f"negative={regimes.count('negative')}, "
            f"neutral={regimes.count('neutral')} "
            f"({len(result.snapshots)} days)",
            "",
        ]

    lines += [
        "  Performance:",
        f"    Total return:         {result.total_return_pct:+.2f}%",
        f"    Annualized:           {result.annual_return_pct:+.2f}%",
        f"    Sharpe ratio:         {result.sharpe:.2f}  (ddof=1, sqrt(252))",
        f"    Max drawdown:         {result.max_dd_pct:.2f}%",
        f"    Win rate:             {result.win_rate:.1%}  ({result.n_trades} trades)",
        f"    Avg hold period:      {result.avg_hold_days:.1f} days",
        "",
        f"  By Signal Type:",
        f"    Pin Attraction:       {result.pin_trades} trades  "
        f"(win rate: {result.pin_winrate:.1%})",
        f"    GEX Spike Fade:       {result.spike_trades} trades  "
        f"(win rate: {result.spike_winrate:.1%})",
        f"    Neg-GEX Breakout:     {result.breakout_trades} trades  "
        f"(win rate: {result.breakout_winrate:.1%})",
        "",
    ]

    if result.trades:
        lines.append(
            f"  {'Entry':>12} {'Exit':>12} {'Dir':>6} {'Signal':>12} {'Days':>5} "
            f"{'GEX_z':>7} {'MaxPain':>9} {'P&L pts':>8} {'P&L%':>8} {'Reason'}"
        )
        lines.append("  " + "-" * 110)
        for t in result.trades:
            lines.append(
                f"  {t.entry_date.isoformat():>12} {t.exit_date.isoformat():>12} "
                f"{t.direction:>6} {t.signal_type:>12} {t.hold_days:5d} "
                f"{t.entry_gex_z:+6.2f}  {t.entry_max_pain:8.0f}  "
                f"{t.pnl_points:+7.1f}  {t.pnl_pct * 100:+7.2f}%  {t.exit_reason}"
            )
        lines.append("")

        # Summary statistics
        pnl_pts = [t.pnl_points for t in result.trades]
        lines += [
            "  Trade P&L distribution:",
            f"    Mean:    {np.mean(pnl_pts):+.1f} pts",
            f"    Median:  {np.median(pnl_pts):+.1f} pts",
            f"    Std:     {np.std(pnl_pts, ddof=1):.1f} pts",
            f"    Best:    {max(pnl_pts):+.1f} pts",
            f"    Worst:   {min(pnl_pts):+.1f} pts",
            "",
        ]

        # Breakdown by exit reason
        reasons: dict[str, list[float]] = {}
        for t in result.trades:
            reasons.setdefault(t.exit_reason, []).append(t.pnl_pct)
        lines.append("  Exit reason breakdown:")
        for reason, pnls in sorted(reasons.items()):
            avg_p = float(np.mean(pnls)) * 100
            lines.append(
                f"    {reason:15s}  n={len(pnls):3d}  "
                f"avg P&L={avg_p:+.2f}%  "
                f"win={sum(1 for p in pnls if p > 0) / len(pnls):.0%}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(result: GEXBacktestResult, output_dir: str | None = None) -> str:
    """Save backtest trades and daily returns to CSV.

    Returns the path to the trades CSV.
    """
    import os
    from datetime import datetime

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
        )
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"gex_dealer_{result.symbol}_{timestamp}"

    # Trades CSV
    if result.trades:
        trades_df = pd.DataFrame([
            {
                "entry_date": t.entry_date.isoformat(),
                "exit_date": t.exit_date.isoformat(),
                "direction": t.direction,
                "signal_type": t.signal_type,
                "entry_spot": t.entry_spot,
                "exit_spot": t.exit_spot,
                "pnl_points": t.pnl_points,
                "pnl_pct": t.pnl_pct,
                "hold_days": t.hold_days,
                "exit_reason": t.exit_reason,
                "entry_gex": t.entry_gex,
                "entry_gex_z": t.entry_gex_z,
                "entry_max_pain": t.entry_max_pain,
                "target_pct": t.target_pct,
            }
            for t in result.trades
        ])
        trades_path = os.path.join(output_dir, f"{base}_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        logger.info("Saved trades to %s", trades_path)
    else:
        trades_path = os.path.join(output_dir, f"{base}_trades.csv")

    # Daily returns CSV
    if result.snapshots and result.daily_returns:
        n = min(len(result.snapshots), len(result.daily_returns))
        daily_df = pd.DataFrame({
            "date": [s.date.isoformat() for s in result.snapshots[:n]],
            "spot": [s.spot for s in result.snapshots[:n]],
            "total_gex": [s.total_gex for s in result.snapshots[:n]],
            "gex_z": [s.gex_z for s in result.snapshots[:n]],
            "gex_regime": [s.gex_regime for s in result.snapshots[:n]],
            "max_pain": [s.max_pain for s in result.snapshots[:n]],
            "vix": [s.vix for s in result.snapshots[:n]],
            "daily_return": result.daily_returns[:n],
        })
        daily_path = os.path.join(output_dir, f"{base}_daily.csv")
        daily_df.to_csv(daily_path, index=False)
        logger.info("Saved daily data to %s", daily_path)

    return trades_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Allow: python gex_dealer.py [symbol] [start] [end]
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NIFTY"
    start_str = sys.argv[2] if len(sys.argv) > 2 else None
    end_str = sys.argv[3] if len(sys.argv) > 3 else None

    start = date.fromisoformat(start_str) if start_str else None
    end = date.fromisoformat(end_str) if end_str else None

    result = run_backtest(symbol=symbol, start_date=start, end_date=end)
    print(format_results(result))

    # Save results
    out_path = save_results(result)
    print(f"\nResults saved to: {out_path}")
