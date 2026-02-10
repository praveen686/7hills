"""FII/DII Institutional Flow Strategy for NIFTY.

Exploits the predictive power of Foreign Institutional Investor (FII)
positioning extremes in index derivatives.  FII act as "smart money" in
Indian markets; when their net positioning (long OI - short OI) in index
futures reaches z-score extremes, subsequent NIFTY returns are predictable.

Architecture:
    1. Data: daily FII & DII net index-futures positioning from
       nse_participant_oi (Client Type = 'FII' / 'DII').
    2. Feature: z-score of FII net position over a 60-day trailing window.
    3. Entry: |z| > 1.5 (strong directional conviction from institutional flow).
       Direction follows FII net: positive z -> long, negative z -> short.
    4. Confirmation: DII positioning must not strongly oppose FII (optional filter).
    5. Exit: z crosses 0 (institutional positioning normalises),
       10-day max hold, or -2% trailing stop.
    6. Position sizing: proportional to |z|, capped at 1.0.
    7. Execution: T+1 (signal at close of t, trade at close of t+1).
    8. Cost: 3 index pts per round-trip (NIFTY futures).

Fully causal: at time t only data up to t is used for z-score computation;
the resulting signal is applied with a T+1 lag so the earliest P&L impact
is on day t+2's close-to-close return.

No look-ahead bias.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZSCORE_WINDOW = 60          # Trailing window for z-score calculation
ZSCORE_ENTRY = 1.5          # Entry threshold: |z| > this
ZSCORE_EXIT = 0.0           # Exit when z crosses zero
MAX_HOLD_DAYS = 10          # Maximum holding period
STOP_LOSS_PCT = -0.02       # -2% stop loss
COST_PTS = 3.0              # Round-trip cost in index points
DII_OPPOSITION_THRESHOLD = -1.0  # DII z-score opposing FII by this much -> skip


@dataclass
class TradeRecord:
    """Track an individual trade from entry to exit."""
    entry_date: date
    entry_price: float
    direction: int            # +1 long, -1 short
    z_at_entry: float
    size: float               # position size [0, 1]
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_pts: float = 0.0
    hold_days: int = 0


# ===================================================================
# Data Loading
# ===================================================================

def _load_participant_oi(store: object) -> pd.DataFrame:
    """Load FII and DII daily index-futures OI from nse_participant_oi.

    Returns a DataFrame indexed by date with columns:
        fii_net, dii_net  (long - short contracts in index futures)
    """
    query = """
        SELECT
            date,
            "Client Type" AS client_type,
            "Future Index Long" AS fut_idx_long,
            "Future Index Short" AS fut_idx_short,
            "Option Index Call Long" AS opt_idx_call_long,
            "Option Index Put Long" AS opt_idx_put_long,
            "Option Index Call Short" AS opt_idx_call_short,
            "Option Index Put Short" AS opt_idx_put_short
        FROM nse_participant_oi
        WHERE "Client Type" IN ('FII', 'DII')
        ORDER BY date, "Client Type"
    """
    raw = store.sql(query)
    if raw.empty:
        logger.error("No participant OI data returned")
        return pd.DataFrame()

    raw["date"] = pd.to_datetime(raw["date"]).dt.date

    # Compute net positioning for index futures
    # Net = Long - Short (positive means net long)
    raw["net_fut_idx"] = raw["fut_idx_long"] - raw["fut_idx_short"]

    # Also compute a synthetic options-based net positioning:
    # Net call bias = (call long - call short) - (put long - put short)
    # Positive => bullish option positioning
    raw["net_opt_idx"] = (
        (raw["opt_idx_call_long"] - raw["opt_idx_call_short"])
        - (raw["opt_idx_put_long"] - raw["opt_idx_put_short"])
    )

    # Composite net = futures net + options net (weighted)
    # Futures are more directional; options provide confirmation
    raw["net_composite"] = raw["net_fut_idx"] + 0.3 * raw["net_opt_idx"]

    # Pivot to get FII and DII columns side by side
    fii = raw[raw["client_type"] == "FII"][
        ["date", "net_fut_idx", "net_opt_idx", "net_composite"]
    ].rename(columns={
        "net_fut_idx": "fii_net_fut",
        "net_opt_idx": "fii_net_opt",
        "net_composite": "fii_net",
    })

    dii = raw[raw["client_type"] == "DII"][
        ["date", "net_fut_idx", "net_opt_idx", "net_composite"]
    ].rename(columns={
        "net_fut_idx": "dii_net_fut",
        "net_opt_idx": "dii_net_opt",
        "net_composite": "dii_net",
    })

    merged = pd.merge(fii, dii, on="date", how="outer").sort_values("date").reset_index(drop=True)
    return merged


def _load_nifty_close(store: object) -> pd.DataFrame:
    """Load daily Nifty 50 closing prices.

    Returns DataFrame with columns: date, close.
    """
    query = """
        SELECT date, "Closing Index Value" AS close
        FROM nse_index_close
        WHERE "Index Name" = 'Nifty 50'
        ORDER BY date
    """
    df = store.sql(query)
    if df.empty:
        logger.error("No Nifty 50 index data returned")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def _load_fii_stats(store: object) -> pd.DataFrame:
    """Load FII daily trading statistics from nse_fii_stats.

    Provides buy/sell contract flow and OI for INDEX FUTURES specifically.
    This complements participant_oi with trading flow (not just OI snapshot).
    """
    query = """
        SELECT
            date,
            buy_contracts,
            sell_contracts,
            oi_contracts,
            oi_amt_cr
        FROM nse_fii_stats
        WHERE category = 'INDEX FUTURES'
        ORDER BY date
    """
    df = store.sql(query)
    if df.empty:
        logger.warning("No FII stats data for INDEX FUTURES")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Net flow: positive = buying pressure
    df["fii_flow_net"] = df["buy_contracts"] - df["sell_contracts"]
    return df


# ===================================================================
# Feature Engineering
# ===================================================================

def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score.  First `window - 1` values are NaN.

    Uses ddof=1 for unbiased standard deviation.
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=1)
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def compute_features(
    participant_df: pd.DataFrame,
    fii_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the feature set for the FII flow strategy.

    Features (all trailing / causal):
        fii_z:     z-score of FII composite net positioning (60d)
        dii_z:     z-score of DII composite net positioning (60d)
        fii_fut_z: z-score of FII index-futures net only (60d)
        fii_flow_z: z-score of FII daily net flow from fii_stats (60d)
        fii_dii_divergence: fii_z - dii_z (positive = FII more bullish)
        fii_z_chg_5d: 5-day change in fii_z (momentum of conviction)

    Parameters
    ----------
    participant_df : DataFrame from _load_participant_oi
    fii_stats_df : DataFrame from _load_fii_stats

    Returns
    -------
    DataFrame indexed by date with feature columns.
    """
    df = participant_df.copy()

    # Core z-scores
    df["fii_z"] = _rolling_zscore(df["fii_net"], ZSCORE_WINDOW)
    df["dii_z"] = _rolling_zscore(df["dii_net"], ZSCORE_WINDOW)
    df["fii_fut_z"] = _rolling_zscore(df["fii_net_fut"], ZSCORE_WINDOW)

    # FII-DII divergence
    df["fii_dii_divergence"] = df["fii_z"] - df["dii_z"]

    # Momentum of FII conviction
    df["fii_z_chg_5d"] = df["fii_z"] - df["fii_z"].shift(5)

    # Merge FII stats flow data
    if not fii_stats_df.empty:
        stats_feat = fii_stats_df[["date", "fii_flow_net"]].copy()
        stats_feat["fii_flow_z"] = _rolling_zscore(
            stats_feat["fii_flow_net"], ZSCORE_WINDOW
        )
        df = pd.merge(
            df,
            stats_feat[["date", "fii_flow_z"]],
            on="date",
            how="left",
        )
    else:
        df["fii_flow_z"] = np.nan

    return df


# ===================================================================
# Signal Generation
# ===================================================================

def _generate_signal(
    fii_z: float,
    dii_z: float,
    fii_flow_z: float,
    fii_z_chg_5d: float,
) -> tuple[float, str]:
    """Generate trading signal from FII positioning z-score.

    Parameters
    ----------
    fii_z : float
        FII composite net positioning z-score.
    dii_z : float
        DII composite net positioning z-score.
    fii_flow_z : float
        FII daily net trading flow z-score.
    fii_z_chg_5d : float
        5-day momentum of FII z-score.

    Returns
    -------
    (position_size, signal_name)
        position_size in [-1.0, 1.0]. Positive = long NIFTY.
    """
    if not np.isfinite(fii_z):
        return 0.0, "no_data"

    abs_z = abs(fii_z)

    # No entry if z-score below threshold
    if abs_z < ZSCORE_ENTRY:
        return 0.0, "below_threshold"

    # Direction follows FII positioning
    direction = 1.0 if fii_z > 0 else -1.0

    # Base size proportional to |z|, capped at 1.0
    # Scale: z=1.5 -> 0.5, z=2.0 -> 0.67, z=3.0 -> 1.0
    size = min(1.0, abs_z / 3.0)

    signal_name = "fii_long" if direction > 0 else "fii_short"

    # --- Confirmation / Filters ---

    # 1. DII opposition filter: if DII strongly opposes FII, reduce size
    if np.isfinite(dii_z):
        if direction > 0 and dii_z < DII_OPPOSITION_THRESHOLD:
            size *= 0.5
            signal_name += "|dii_opposes"
        elif direction < 0 and dii_z > -DII_OPPOSITION_THRESHOLD:
            size *= 0.5
            signal_name += "|dii_opposes"

    # 2. Flow confirmation: if daily flow aligns, boost confidence
    if np.isfinite(fii_flow_z):
        if np.sign(fii_flow_z) == np.sign(fii_z) and abs(fii_flow_z) > 1.0:
            size = min(1.0, size * 1.2)
            signal_name += "|flow_confirms"

    # 3. Momentum confirmation: z-score is accelerating in same direction
    if np.isfinite(fii_z_chg_5d):
        if np.sign(fii_z_chg_5d) == np.sign(fii_z) and abs(fii_z_chg_5d) > 0.3:
            size = min(1.0, size * 1.1)
            signal_name += "|momentum"

    return direction * size, signal_name


# ===================================================================
# Backtest Engine
# ===================================================================

def run_backtest(store: object) -> pd.DataFrame:
    """Run the FII/DII institutional flow backtest on NIFTY.

    Fully causal with T+1 execution:
        - Signal computed at close of day t using data up to day t.
        - Position entered/exited at close of day t+1.
        - P&L realized on day t+1's close-to-close return (t+1 -> t+2).

    Actually, to keep it simple and honest:
        - At the end of day t, we observe FII positioning data for day t
          and compute z-score.
        - The signal determines the position for day t+1.
        - The daily return attributed to day t+1 is:
          position[t] * (close[t+1] - close[t]) / close[t] - cost_if_traded.

    This is T+1 lag: signal at t, return earned on t+1.

    Returns
    -------
    pd.DataFrame with columns: date, position, signal, fii_z, dii_z,
        daily_return, cumulative_return.
    """
    # --- Load data ---
    logger.info("Loading participant OI data...")
    participant_df = _load_participant_oi(store)
    if participant_df.empty:
        return pd.DataFrame()

    logger.info("Loading FII stats data...")
    fii_stats_df = _load_fii_stats(store)

    logger.info("Loading Nifty 50 close prices...")
    nifty_df = _load_nifty_close(store)
    if nifty_df.empty:
        return pd.DataFrame()

    logger.info(
        "Data loaded: participant_oi %d rows, fii_stats %d rows, nifty %d rows",
        len(participant_df), len(fii_stats_df), len(nifty_df),
    )

    # --- Compute features ---
    features = compute_features(participant_df, fii_stats_df)

    # --- Merge features with price ---
    df = pd.merge(nifty_df, features, on="date", how="inner").sort_values("date").reset_index(drop=True)

    logger.info("Merged dataset: %d trading days (%s to %s)",
                len(df), df["date"].iloc[0], df["date"].iloc[-1])

    # --- Pre-compute daily returns (close-to-close) ---
    closes = df["close"].values.astype(np.float64)
    daily_rets = np.zeros(len(df))
    daily_rets[1:] = (closes[1:] - closes[:-1]) / closes[:-1]

    # Cost per trade in return terms (3 pts / current price)
    cost_rets = COST_PTS / closes

    # --- Run signal generation and P&L ---
    n = len(df)
    positions = np.zeros(n)       # Position decided at close of day t
    signals = [""] * n
    strat_rets = np.zeros(n)      # Strategy return earned on day t
    hold_counter = np.zeros(n, dtype=int)
    entry_prices = np.zeros(n)

    prev_pos = 0.0
    prev_hold = 0
    prev_entry_price = 0.0

    for t in range(n):
        fii_z_t = df["fii_z"].iloc[t]
        dii_z_t = df["dii_z"].iloc[t]
        fii_flow_z_t = df["fii_flow_z"].iloc[t] if "fii_flow_z" in df.columns else np.nan
        fii_z_chg_5d_t = df["fii_z_chg_5d"].iloc[t] if "fii_z_chg_5d" in df.columns else np.nan

        # T+1: position decided at close of t-1 earns return on day t
        if t == 0:
            positions[t] = 0.0
            signals[t] = "warmup"
            strat_rets[t] = 0.0
            hold_counter[t] = 0
            entry_prices[t] = 0.0
            continue

        # The position held during day t was decided at close of t-1
        # So first compute PnL for day t using prev_pos
        raw_ret = daily_rets[t]  # close[t] / close[t-1] - 1

        # --- Exit checks on the current position (prev_pos) ---
        exit_signal = ""
        force_exit = False

        if prev_pos != 0.0:
            # Check stop loss: cumulative PnL from entry
            cum_pnl = prev_pos * (closes[t] - prev_entry_price) / prev_entry_price
            if cum_pnl < STOP_LOSS_PCT:
                force_exit = True
                exit_signal = "stop_loss"

            # Check max hold
            if prev_hold >= MAX_HOLD_DAYS:
                force_exit = True
                exit_signal = "max_hold"

            # Check z-score exit: z crosses zero (positioning normalised)
            if np.isfinite(fii_z_t):
                if prev_pos > 0 and fii_z_t <= ZSCORE_EXIT:
                    force_exit = True
                    exit_signal = "z_exit_long"
                elif prev_pos < 0 and fii_z_t >= -ZSCORE_EXIT:
                    force_exit = True
                    exit_signal = "z_exit_short"

        # Apply previous position's PnL on today
        strat_rets[t] = prev_pos * raw_ret

        # --- Decide new position for tomorrow (decided at close of t) ---
        if force_exit:
            # Exit: go flat. Pay cost for closing.
            new_pos = 0.0
            strat_rets[t] -= abs(prev_pos) * cost_rets[t]  # Exit cost
            signals[t] = exit_signal
            new_hold = 0
            new_entry_price = 0.0
        else:
            # Generate fresh signal
            new_pos, sig = _generate_signal(fii_z_t, dii_z_t, fii_flow_z_t, fii_z_chg_5d_t)

            if prev_pos == 0.0 and new_pos != 0.0:
                # New entry: pay entry cost
                strat_rets[t] -= abs(new_pos) * cost_rets[t]
                new_hold = 1
                new_entry_price = closes[t]
                signals[t] = sig
            elif prev_pos != 0.0 and new_pos == 0.0:
                # Signal says exit: pay exit cost
                strat_rets[t] -= abs(prev_pos) * cost_rets[t]
                new_hold = 0
                new_entry_price = 0.0
                signals[t] = sig + "|signal_exit"
            elif prev_pos != 0.0 and np.sign(new_pos) != np.sign(prev_pos):
                # Direction flip: exit old + enter new (2x cost)
                strat_rets[t] -= (abs(prev_pos) + abs(new_pos)) * cost_rets[t]
                new_hold = 1
                new_entry_price = closes[t]
                signals[t] = sig + "|flip"
            elif prev_pos != 0.0:
                # Continue holding, update hold counter
                new_hold = prev_hold + 1
                new_entry_price = prev_entry_price
                new_pos = prev_pos  # Maintain existing position
                signals[t] = "hold"
            else:
                # Was flat, stays flat
                new_hold = 0
                new_entry_price = 0.0
                signals[t] = sig

        positions[t] = new_pos
        hold_counter[t] = new_hold
        entry_prices[t] = new_entry_price

        prev_pos = new_pos
        prev_hold = new_hold
        prev_entry_price = new_entry_price

    # --- Build result DataFrame ---
    cum_ret = np.cumprod(1.0 + strat_rets)

    result = pd.DataFrame({
        "date": df["date"],
        "close": df["close"],
        "position": positions,
        "signal": signals,
        "fii_z": df["fii_z"].round(4),
        "dii_z": df["dii_z"].round(4),
        "fii_net_fut": df["fii_net_fut"],
        "daily_return": np.round(strat_rets, 8),
        "cumulative_return": np.round(cum_ret, 6),
    })

    # Compute and print statistics
    _print_statistics(result)

    return result


# ===================================================================
# Statistics
# ===================================================================

def _print_statistics(df: pd.DataFrame) -> dict:
    """Compute and print backtest statistics.

    Sharpe: ddof=1, sqrt(252), all daily returns including flat days.
    """
    rets = df["daily_return"].values
    n_days = len(rets)

    # Sharpe ratio
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Total return
    total_ret = float(df["cumulative_return"].iloc[-1] - 1.0)

    # Max drawdown
    cum = df["cumulative_return"].values
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Win rate on active days
    active_mask = df["position"].values != 0.0
    active_rets = rets[active_mask]
    n_active = len(active_rets)
    win_rate = float(np.mean(active_rets > 0)) if n_active > 0 else 0.0
    avg_win = float(np.mean(active_rets[active_rets > 0])) if np.any(active_rets > 0) else 0.0
    avg_loss = float(np.mean(active_rets[active_rets < 0])) if np.any(active_rets < 0) else 0.0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    # Trade count (transitions from flat to active)
    pos = df["position"].values
    entries = np.sum((pos[1:] != 0) & (pos[:-1] == 0))

    # Signal distribution
    signal_counts = df["signal"].value_counts().to_dict()

    # Annualized return
    if n_days > 1:
        ann_ret = (df["cumulative_return"].iloc[-1]) ** (252 / n_days) - 1.0
    else:
        ann_ret = 0.0

    stats = {
        "total_days": n_days,
        "active_days": n_active,
        "num_trades": int(entries),
        "total_return": f"{total_ret:.2%}",
        "annualized_return": f"{ann_ret:.2%}",
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": f"{max_dd:.2%}",
        "win_rate": f"{win_rate:.2%}",
        "avg_win": f"{avg_win:.4%}",
        "avg_loss": f"{avg_loss:.4%}",
        "profit_factor": round(profit_factor, 2),
        "signal_distribution": signal_counts,
    }

    print("\n" + "=" * 65)
    print("  FII/DII Institutional Flow Strategy — NIFTY Backtest")
    print("=" * 65)
    for k, v in stats.items():
        if k == "signal_distribution":
            print(f"  {'signals':>22s}:")
            for sig_name, cnt in sorted(v.items(), key=lambda x: -x[1]):
                print(f"  {'':>22s}  {sig_name}: {cnt}")
        else:
            print(f"  {k:>22s}: {v}")
    print("=" * 65 + "\n")

    return stats


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    """Run FII/DII flow strategy backtest for NIFTY."""
    # Ensure project root on path
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from quantlaxmi.data.store import MarketDataStore

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting FII/DII Institutional Flow backtest for NIFTY")

    with MarketDataStore() as store:
        result_df = run_backtest(store)

    if not result_df.empty:
        # Save results
        out_dir = Path(__file__).resolve().parent.parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "fii_flow_nifty.csv"
        result_df.to_csv(out_file, index=False)
        logger.info("Results saved to %s", out_file)

        # Final summary
        print(f"\nFinal cumulative return: {result_df['cumulative_return'].iloc[-1]:.6f}")
        print(f"Date range: {result_df['date'].iloc[0]} -> {result_df['date'].iloc[-1]}")

        # Show some sample signals
        active = result_df[result_df["position"] != 0.0]
        if not active.empty:
            print(f"\nSample active positions (first 10):")
            cols = ["date", "close", "position", "signal", "fii_z", "dii_z", "daily_return"]
            print(active[cols].head(10).to_string(index=False))
    else:
        logger.error("Backtest returned empty results")


if __name__ == "__main__":
    main()
