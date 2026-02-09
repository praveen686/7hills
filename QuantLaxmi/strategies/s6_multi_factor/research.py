"""S6: Multi-Factor Research — Walk-forward XGBoost on index returns.

Loads nse_index_close for 120+ indices, computes return features,
predicts NIFTY 50 5-day forward return using walk-forward cross-validation.

Usage:
    python -m apps.india_scanner.research.s6_multi_factor
    python -m apps.india_scanner.research.s6_multi_factor --start 2025-09-01
    python -m apps.india_scanner.research.s6_multi_factor --target "Nifty 50"
"""

from __future__ import annotations

import argparse
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd

from data.store import MarketDataStore

warnings.filterwarnings("ignore", category=UserWarning)


def _build_feature_matrix(
    store: MarketDataStore,
    start: date,
    end: date,
    target_index: str = "Nifty 50",
    feature_windows: tuple[int, ...] = (5, 10, 20),
    forward_horizon: int = 5,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build feature matrix from index close data.

    Features: {window}-day return for each index.
    Target: forward_horizon-day return for target_index.
    """
    # Load all index close data
    df = store.sql(
        "SELECT * FROM nse_index_close WHERE date >= ? AND date <= ? ORDER BY date",
        [start.isoformat(), end.isoformat()],
    )
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), []

    # Identify columns — NSE index_close uses specific names
    col_map = {
        "Index Date": "date_col",
        "Index Name": "name_col",
        "Closing Index Value": "close_col",
    }
    date_col = name_col = close_col = None
    for col in df.columns:
        if col == "Index Date" or col == "date":
            date_col = col
        elif col == "Index Name":
            name_col = col
        elif col == "Closing Index Value":
            close_col = col
        elif "index" in col.lower() and "name" in col.lower() and name_col is None:
            name_col = col
        elif "clos" in col.lower() and "index" in col.lower() and close_col is None:
            close_col = col

    if date_col is None or name_col is None or close_col is None:
        print(f"  Cannot identify columns: {list(df.columns)}")
        return pd.DataFrame(), pd.Series(dtype=float), []

    # Pivot to wide format: dates x indices
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    pivot = df.pivot_table(index=date_col, columns=name_col, values=close_col)
    pivot = pivot.sort_index()

    # Drop indices with too many NaNs
    min_obs = len(pivot) * 0.5
    pivot = pivot.dropna(axis=1, thresh=int(min_obs))

    if target_index not in pivot.columns:
        # Try fuzzy match
        matches = [c for c in pivot.columns if target_index.lower() in c.lower()]
        if matches:
            target_index = matches[0]
            print(f"  Matched target to: {target_index}")
        else:
            print(f"  Target '{target_index}' not found. Available: {list(pivot.columns)[:10]}...")
            return pd.DataFrame(), pd.Series(dtype=float), []

    print(f"  Pivot: {pivot.shape[0]} dates x {pivot.shape[1]} indices")

    # Compute features: returns over various windows
    feature_dfs = []
    feature_names = []

    for w in feature_windows:
        rets = pivot.pct_change(w)
        rets.columns = [f"{c}_ret{w}d" for c in rets.columns]
        feature_dfs.append(rets)
        feature_names.extend(rets.columns.tolist())

    features = pd.concat(feature_dfs, axis=1)

    # Target: forward return on target index
    # At row i, target = (price[i+h] - price[i]) / price[i]
    # Computed as: pct_change(h) shifted backward by h positions.
    # NOTE: target[i] uses price[i+h], so the last h training rows
    # must be PURGED in walk-forward to avoid leaking test data.
    target = pivot[target_index].pct_change(forward_horizon).shift(-forward_horizon)
    target.name = "target"

    # Align and drop NaN rows
    combined = pd.concat([features, target], axis=1).dropna()
    X = combined[feature_names]
    y = combined["target"]

    return X, y, feature_names


def _walk_forward_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = 60,
    test_window: int = 5,
    purge_gap: int = 5,
) -> dict:
    """Walk-forward XGBoost backtest.

    Parameters
    ----------
    purge_gap : int
        Number of rows to drop from the END of training data to prevent
        forward-return targets from leaking test-set prices.  Must be
        >= forward_horizon used in target construction.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  XGBoost not installed. Using sklearn GradientBoosting.")
        from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

    n = len(X)
    predictions = []
    actuals = []
    importances = np.zeros(X.shape[1])
    n_folds = 0

    i = train_window
    while i + test_window <= n:
        # Purge last `purge_gap` rows from training to prevent target
        # look-ahead: target[j] = return(j, j+h), so targets near the
        # train/test boundary use test-period prices.
        train_end = i - purge_gap
        X_train = X.iloc[i - train_window:train_end]
        y_train = y.iloc[i - train_window:train_end]
        X_test = X.iloc[i:i + test_window]
        y_test = y.iloc[i:i + test_window]

        try:
            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.5,
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            predictions.extend(pred.tolist())
            actuals.extend(y_test.values.tolist())

            # Feature importance
            if hasattr(model, "feature_importances_"):
                importances += model.feature_importances_
            n_folds += 1
        except Exception as e:
            print(f"  Fold at {i} failed: {e}")

        i += test_window

    if not predictions:
        return {"error": "No successful folds"}

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    importances = importances / max(n_folds, 1)

    # Metrics
    from scipy.stats import spearmanr
    ic, ic_pval = spearmanr(predictions, actuals)

    # Direction accuracy
    direction_correct = np.sign(predictions) == np.sign(actuals)
    hit_rate = direction_correct.mean()

    # R-squared
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Backtest: go long when prediction > 0, flat otherwise
    strategy_returns = np.where(predictions > 0, actuals, 0)
    cum_return = (1 + strategy_returns).prod() - 1
    sharpe = (strategy_returns.mean() / strategy_returns.std(ddof=1) * np.sqrt(252)
              if strategy_returns.std(ddof=1) > 0 else 0)

    # Max drawdown
    equity = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = float(drawdowns.min())

    return {
        "n_folds": n_folds,
        "n_predictions": len(predictions),
        "ic": float(ic),
        "ic_pval": float(ic_pval),
        "hit_rate": float(hit_rate),
        "r2": float(r2),
        "cum_return": float(cum_return),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "importances": importances,
        "feature_names": list(X.columns),
        "daily_returns": strategy_returns.tolist(),
    }


def run_research(
    store: MarketDataStore,
    start: date,
    end: date,
    target_index: str = "Nifty 50",
) -> None:
    """Main research loop."""
    print(f"\nS6 Multi-Factor Research: {start} to {end}")
    print(f"  Target: {target_index}")
    print()

    t0 = time.time()

    X, y, feature_names = _build_feature_matrix(
        store, start, end, target_index=target_index
    )
    if X.empty:
        print("  No data available for feature matrix.")
        return

    print(f"  Features: {X.shape[1]}, Observations: {X.shape[0]}")
    print(f"  Target range: [{y.min():.4f}, {y.max():.4f}]")

    # Walk-forward backtest
    print("\n  Running walk-forward XGBoost...")
    result = _walk_forward_backtest(X, y, train_window=60, test_window=5)
    elapsed = time.time() - t0

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    # Print results
    print(f"\n{'='*70}")
    print(f"S6 MULTI-FACTOR RESULTS ({elapsed:.1f}s)")
    print("=" * 70)

    print(f"\n  Walk-Forward CV:")
    print(f"    Folds:            {result['n_folds']}")
    print(f"    Predictions:      {result['n_predictions']}")
    print(f"    R-squared:        {result['r2']:.4f}")
    print(f"    IC (Spearman):    {result['ic']:+.4f} (p={result['ic_pval']:.4f})")
    print(f"    Hit rate:         {result['hit_rate']:.1%}")

    print(f"\n  Strategy (long when pred > 0):")
    print(f"    Cumulative return: {result['cum_return']*100:+.2f}%")
    print(f"    Sharpe (ann.):     {result['sharpe']:.2f}")
    print(f"    Max drawdown:      {result['max_dd']*100:.2f}%")

    # Top feature importances
    importances = result["importances"]
    names = result["feature_names"]
    top_idx = np.argsort(importances)[::-1][:20]

    print(f"\n  Top 20 Feature Importances:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {names[idx]:<35} {importances[idx]:.4f}")

    print()


def main() -> None:
    from strategies.utils import tee_to_results

    parser = argparse.ArgumentParser(description="S6 Multi-Factor Research")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--target", default="Nifty 50",
                        help="Target index name (default: 'Nifty 50')")
    args = parser.parse_args()

    with tee_to_results("s6_multi_factor"):
        store = MarketDataStore()

        if args.start:
            start = date.fromisoformat(args.start)
        else:
            dates = store.available_dates("nse_index_close")
            start = min(dates) if dates else date(2025, 8, 1)

        end = date.fromisoformat(args.end) if args.end else date.today()

        run_research(store, start, end, target_index=args.target)
        store.close()


if __name__ == "__main__":
    main()
