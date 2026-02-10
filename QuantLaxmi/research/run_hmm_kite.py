"""Run HMM Regime strategy using Kite historical data (longer history).

Fetches 2+ years of daily OHLCV for NIFTY 50 from Zerodha Kite API,
then runs the HMM Regime-Switching backtest with proper training data.

Usage:
    python -m research.run_hmm_kite [--years 2]
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from quantlaxmi.data.collectors.auth import headless_login
from quantlaxmi.data.zerodha import fetch_historical_chunked

logger = logging.getLogger(__name__)

NIFTY_TOKEN = 256265  # NIFTY 50 instrument token on Kite


def fetch_nifty_daily(years: int = 2) -> pd.DataFrame:
    """Fetch NIFTY 50 daily OHLCV from Kite for the given number of years."""
    logger.info("Authenticating with Zerodha Kite...")
    kite = headless_login()
    logger.info("Authenticated successfully")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    logger.info(
        "Fetching NIFTY 50 daily data: %s to %s (%d years)",
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        years,
    )

    ohlcv = fetch_historical_chunked(
        kite=kite,
        instrument_token=NIFTY_TOKEN,
        interval="1d",
        start=start_date,
        end=end_date,
        chunk_days=365,  # Daily data can use larger chunks
    )

    df = ohlcv.df.reset_index()
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date

    logger.info("Fetched %d daily bars for NIFTY 50", len(df))
    return df


def run_hmm_on_kite_data(years: int = 2) -> None:
    """Fetch Kite data and run HMM backtest."""
    from quantlaxmi.strategies.s13_hmm_regime.strategy import (
        extract_features,
        GaussianHMM,
        Regime,
        N_STATES,
        N_EM_ITER,
        EM_TOL,
        REFIT_INTERVAL,
        DAILY_STOP_LOSS,
        _standardize_features,
    )

    # Fetch data
    df = fetch_nifty_daily(years=years)

    if len(df) < 120:
        logger.error("Insufficient data: %d rows (need 120+)", len(df))
        return

    logger.info("Building features from %d daily bars...", len(df))

    # Build feature DataFrame
    feat_df = pd.DataFrame({
        "date": df["date"],
        "close": df["close"].values,
    })

    features = extract_features(feat_df)

    # Feature columns
    feat_cols = ["ret_1d", "ret_5d", "ret_21d", "vol_5d", "vol_21d", "rsi_14"]
    if features["norm_vol"].notna().any() and (features["norm_vol"] != 1.0).any():
        feat_cols.append("norm_vol")

    # Find first valid index
    first_valid = features[feat_cols].dropna().index[0]
    min_history = 60

    bt_start_idx = first_valid + min_history
    if bt_start_idx >= len(features):
        logger.error("Not enough data after feature computation")
        return

    logger.info(
        "Backtest range: row %d to %d (%d trading days), first_valid=%d",
        bt_start_idx, len(features) - 1,
        len(features) - bt_start_idx, first_valid,
    )

    # Run day-by-day backtest
    close_arr = df["close"].values.astype(np.float64)
    records = []
    hmm = None
    last_fit_idx = -REFIT_INTERVAL
    state_map: dict[int, Regime] = {}
    prev_regime = None
    cum_ret = 1.0
    position = 0.0
    fit_mean = None
    fit_std = None

    for t in range(bt_start_idx, len(features)):
        today_date = features["date"].iloc[t]
        today_close = close_arr[t]
        yesterday_close = close_arr[t - 1] if t > 0 else today_close
        day_return = np.log(today_close / yesterday_close) if yesterday_close > 0 else 0.0

        # Build feature matrix up to t-1 (causal)
        feat_slice = features.iloc[first_valid:t]
        X_raw = feat_slice[feat_cols].values.astype(np.float64)
        valid_rows = np.all(np.isfinite(X_raw), axis=1)
        X_clean = X_raw[valid_rows]

        if len(X_clean) < min_history:
            records.append({
                "date": today_date, "position": 0.0, "signal": "insufficient_data",
                "regime": "UNKNOWN", "confidence": 0.0, "daily_return": 0.0,
                "cumulative_return": cum_ret,
            })
            continue

        # Refit HMM if needed
        if hmm is None or (t - last_fit_idx) >= REFIT_INTERVAL:
            fit_mean = X_clean.mean(axis=0)
            fit_std = X_clean.std(axis=0, ddof=1)
            X_std = _standardize_features(X_clean, fit_mean, fit_std)

            hmm = GaussianHMM(
                n_states=N_STATES,
                n_iter=N_EM_ITER,
                tol=EM_TOL,
                covariance_type="diag",
                random_state=42,
            )
            try:
                hmm.fit(X_std)
                states = hmm.predict(X_std)
                # Label states by mean return
                state_means = {}
                for s in range(N_STATES):
                    mask = states == s
                    if mask.any():
                        state_means[s] = X_clean[mask, 0].mean()  # ret_1d
                    else:
                        state_means[s] = 0.0

                sorted_states = sorted(state_means.keys(), key=lambda s: state_means[s])
                state_map = {
                    sorted_states[0]: Regime.BEAR,
                    sorted_states[1]: Regime.NEUTRAL,
                    sorted_states[2]: Regime.BULL,
                }
                last_fit_idx = t
            except Exception as e:
                logger.warning("HMM fit failed at t=%d: %s", t, e)
                records.append({
                    "date": today_date, "position": 0.0, "signal": "fit_error",
                    "regime": "UNKNOWN", "confidence": 0.0, "daily_return": 0.0,
                    "cumulative_return": cum_ret,
                })
                continue

        # Predict today's regime using yesterday's features
        today_feat = features.iloc[t - 1:t][feat_cols].values.astype(np.float64)
        if not np.all(np.isfinite(today_feat)):
            records.append({
                "date": today_date, "position": 0.0, "signal": "nan_features",
                "regime": "UNKNOWN", "confidence": 0.0, "daily_return": 0.0,
                "cumulative_return": cum_ret,
            })
            continue

        today_std = _standardize_features(today_feat, fit_mean, fit_std)
        pred_state = hmm.predict(today_std)[0]
        regime = state_map.get(pred_state, Regime.NEUTRAL)
        proba = hmm.predict_proba(today_std)[0]
        confidence = proba[pred_state]

        # Generate signal
        if regime == Regime.BULL:
            new_position = 1.0  # Long
            signal = "bull_momentum"
        elif regime == Regime.BEAR:
            new_position = -1.0  # Short
            signal = "bear_short"
        else:
            # Neutral: RSI mean-reversion
            rsi = features["rsi_14"].iloc[t - 1]
            if rsi < 30:
                new_position = 0.5
                signal = "neutral_oversold"
            elif rsi > 70:
                new_position = -0.5
                signal = "neutral_overbought"
            else:
                new_position = 0.0
                signal = "neutral_flat"

        # Apply stop-loss
        strat_return = position * day_return
        if strat_return < DAILY_STOP_LOSS:
            new_position = 0.0
            signal = "stop_loss"

        # Update cumulative return
        cum_ret *= (1.0 + strat_return)
        position = new_position

        records.append({
            "date": today_date,
            "position": position,
            "signal": signal,
            "regime": regime.name if isinstance(regime, Regime) else str(regime),
            "confidence": round(confidence, 4),
            "daily_return": round(strat_return, 6),
            "cumulative_return": round(cum_ret, 6),
        })

    result_df = pd.DataFrame(records)

    if result_df.empty:
        logger.error("No backtest results")
        return

    # Calculate statistics
    active = result_df[result_df["position"] != 0.0]
    daily_rets = result_df["daily_return"]
    total_ret = (cum_ret - 1.0) * 100
    sharpe = (daily_rets.mean() / daily_rets.std(ddof=1)) * np.sqrt(252) if daily_rets.std(ddof=1) > 0 else 0

    # Max drawdown
    equity = result_df["cumulative_return"]
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100

    # Win rate
    trade_rets = active["daily_return"] if not active.empty else pd.Series(dtype=float)
    wins = (trade_rets > 0).sum()
    win_rate = wins / len(trade_rets) * 100 if len(trade_rets) > 0 else 0

    # Save results
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"hmm_regime_kite_NIFTY50_{date.today().isoformat()}.csv"
    result_df.to_csv(out_file, index=False)

    print(f"""
============================================================
  HMM Regime-Switching Backtest (Kite Data): NIFTY 50
============================================================
  Data source:      Zerodha Kite Historical API
  Total bars:       {len(df)}
  Backtest days:    {len(result_df)}
  Active days:      {len(active)}

  Total return:     {total_ret:+.2f}%
  Sharpe ratio:     {sharpe:.3f}
  Max drawdown:     {max_dd:.2f}%
  Win rate:         {win_rate:.1f}%

  Regime distribution:
{result_df['regime'].value_counts().to_string()}

  Signal distribution:
{result_df['signal'].value_counts().to_string()}
============================================================
  Results saved to: {out_file}
""")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=2, help="Years of history to fetch")
    args = parser.parse_args()

    run_hmm_on_kite_data(years=args.years)
