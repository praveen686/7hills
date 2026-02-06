"""Walk-forward research on real Binance crypto data.

Runs a full pipeline per symbol:
  1. Load cached 1h klines
  2. Build feature matrix (technical + crypto-specific)
  3. Generate triple-barrier labels
  4. Walk-forward XGBoost with mandatory CV gap
  5. Backtest with realistic costs (10bps roundtrip)
  6. Report OOS metrics

Usage:
    python3 research/crypto_walkforward.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlx.core.types import OHLCV
from qlx.data.cache import KlineCache
from qlx.features import (
    RSI,
    ATR,
    BollingerBands,
    CyclicalTime,
    FeatureMatrix,
    HistoricalReturns,
    MeanReversionZ,
    Momentum,
    MultiTimeframeMomentum,
    RangePosition,
    ReturnDistribution,
    Stochastic,
    SuperTrend,
    VolumeProfile,
    VolatilityRegime,
    VWAPDeviation,
)
from qlx.targets.returns import FutureReturn, TripleBarrier
from qlx.backtest.costs import CostModel
from qlx.backtest.portfolio import run_backtest
from qlx.metrics.performance import compute_metrics
from qlx.pipeline.split import WalkForwardSplit

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
]

HORIZON = 24           # predict 24h forward return direction
CV_GAP = 24            # gap between train and test (= horizon)
TRAIN_FRAC = 0.7
WALK_WINDOW = 4000     # ~167 days per fold
COST_BPS = 10          # 5bps commission + 5bps slippage each way

# XGBoost params tuned for noisy financial data
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=1.0,
    reg_lambda=5.0,
    min_child_weight=10,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------

def build_features(ohlcv: OHLCV) -> pd.DataFrame:
    """Build full feature matrix from OHLCV."""
    matrix = (
        FeatureMatrix(ohlcv)
        # Technical
        .add(RSI(window=14))
        .add(RSI(window=28))
        .add(BollingerBands(window=20))
        .add(SuperTrend(period=14, multiplier=3.0))
        .add(Stochastic(k_window=14, d_window=3))
        .add(ATR(window=14))
        # Returns & momentum
        .add(HistoricalReturns(periods=(1, 4, 12, 24, 72, 168)))
        .add(Momentum(fast=12, slow=72, run_window=12))
        # Crypto-specific
        .add(VolatilityRegime(fast=24, slow=168))
        .add(MeanReversionZ(windows=(24, 72, 168)))
        .add(VWAPDeviation(window=24))
        .add(VolumeProfile(window=48))
        .add(ReturnDistribution(window=72))
        .add(MultiTimeframeMomentum(fast=6, medium=24, slow=168))
        .add(RangePosition(window=48))
        # Temporal
        .add(CyclicalTime())
    )
    return matrix.build(dropna=True)


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def run_single_symbol(symbol: str, cache: KlineCache) -> dict | None:
    """Run full walk-forward pipeline for one symbol."""
    print(f"\n{'='*60}")
    print(f"  {symbol}")
    print(f"{'='*60}")

    # Load data
    ohlcv = cache.get(symbol, "1h", start="2024-01-01", market="spot")
    print(f"  Loaded {len(ohlcv)} bars")

    # Build features
    X = build_features(ohlcv)
    print(f"  Features: {X.shape[1]} columns, {len(X)} rows after warmup")

    # Build target: binary classification (up/down over HORIZON bars)
    target = FutureReturn(horizon=HORIZON)
    y_raw = target.transform(ohlcv)

    # Align X and y
    common_idx = X.index.intersection(y_raw.dropna().index)
    X = X.loc[common_idx]
    y_raw = y_raw.loc[common_idx]

    # Binary: 1 = positive return, 0 = negative
    y = (y_raw > 0).astype(int)
    prices = ohlcv.close.loc[common_idx]

    print(f"  Aligned: {len(X)} samples, class balance: {y.mean():.1%} positive")

    if len(X) < WALK_WINDOW + 500:
        print(f"  SKIP: not enough data ({len(X)} < {WALK_WINDOW + 500})")
        return None

    # Walk-forward split
    splitter = WalkForwardSplit(
        window=WALK_WINDOW,
        train_frac=TRAIN_FRAC,
        gap=CV_GAP,
        horizon=HORIZON,
    )
    splits = list(splitter.split(len(X)))
    print(f"  Folds: {len(splits)}")

    # Preprocessing
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    all_preds = pd.Series(dtype=float, name="pred")
    fold_metrics = []

    for fold in splits:
        X_train = X.iloc[fold.train_idx]
        y_train = y.iloc[fold.train_idx]
        X_test = X.iloc[fold.test_idx]
        y_test = y.iloc[fold.test_idx]

        # Fit preprocessing on train only
        X_train_imp = imputer.fit_transform(X_train)
        X_train_sc = scaler.fit_transform(X_train_imp)
        X_test_imp = imputer.transform(X_test)
        X_test_sc = scaler.transform(X_test_imp)

        # Train XGBoost
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train_sc, y_train, verbose=False)

        # Predict probabilities → signal
        proba = model.predict_proba(X_test_sc)[:, 1]

        # Convert probability to position: >0.55 long, <0.45 short, else flat
        signal = pd.Series(0.0, index=X_test.index)
        signal[proba > 0.55] = 1.0
        signal[proba < 0.45] = -1.0

        all_preds = pd.concat([all_preds, signal])

        # Fold accuracy
        pred_class = (proba > 0.5).astype(int)
        acc = (pred_class == y_test.values).mean()
        fold_metrics.append({"fold": fold.fold, "accuracy": acc, "n_test": len(y_test)})

        print(f"    Fold {fold.fold}: acc={acc:.3f}  long={int((signal==1).sum())}  short={int((signal==-1).sum())}  flat={int((signal==0).sum())}")

    # Backtest OOS predictions
    oos_prices = prices.loc[all_preds.index]
    cost_model = CostModel(commission_bps=COST_BPS / 2, slippage_bps=COST_BPS / 2)

    result = run_backtest(
        prices=oos_prices,
        predictions=all_preds,
        cost_model=cost_model,
        long_entry_th=0.0,    # already thresholded: signal is -1/0/+1
        short_entry_th=0.0,
    )

    # Compute metrics
    returns = result.equity_curve.pct_change().dropna()
    if len(returns) > 10:
        metrics = compute_metrics(returns, periods_per_year=8760)  # hourly
    else:
        print(f"  SKIP: not enough OOS returns")
        return None

    # Feature importance (from last fold)
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    print(f"\n  OOS Results ({len(all_preds)} bars):")
    print(f"    Sharpe:      {metrics.sharpe_ratio:.3f}")
    print(f"    Ann Return:  {metrics.annualised_return:.1%}")
    print(f"    Max DD:      {metrics.max_drawdown:.1%}")
    print(f"    Sortino:     {metrics.sortino_ratio:.3f}")
    print(f"    Calmar:      {metrics.calmar_ratio:.3f}")
    print(f"    Trades:      {len(result.trades)}")
    print(f"    Avg Acc:     {np.mean([f['accuracy'] for f in fold_metrics]):.3f}")
    print(f"\n  Top 10 features:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:40s} {imp:.4f}")

    return {
        "symbol": symbol,
        "sharpe": metrics.sharpe_ratio,
        "ann_return": metrics.annualised_return,
        "max_dd": metrics.max_drawdown,
        "sortino": metrics.sortino_ratio,
        "calmar": metrics.calmar_ratio,
        "trades": len(result.trades),
        "accuracy": np.mean([f["accuracy"] for f in fold_metrics]),
        "n_oos": len(all_preds),
        "top_features": list(importance.head(5).index),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  QLX Walk-Forward Research — Real Binance Data")
    print(f"  Horizon: {HORIZON}h | Cost: {COST_BPS}bps RT | Window: {WALK_WINDOW}")
    print("=" * 60)

    cache = KlineCache("data/klines")
    results = []

    for sym in SYMBOLS:
        try:
            r = run_single_symbol(sym, cache)
            if r:
                results.append(r)
        except Exception as e:
            print(f"\n  {sym}: ERROR — {e}")

    # Summary
    if results:
        print("\n\n" + "=" * 80)
        print("  UNIVERSE SUMMARY")
        print("=" * 80)
        print(f"{'Symbol':12s} {'Sharpe':>8s} {'AnnRet':>8s} {'MaxDD':>8s} {'Sortino':>8s} {'Trades':>7s} {'Acc':>6s}")
        print("-" * 80)

        for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
            print(
                f"{r['symbol']:12s} "
                f"{r['sharpe']:8.3f} "
                f"{r['ann_return']:7.1%} "
                f"{r['max_dd']:7.1%} "
                f"{r['sortino']:8.3f} "
                f"{r['trades']:7d} "
                f"{r['accuracy']:6.3f}"
            )

        avg_sharpe = np.mean([r["sharpe"] for r in results])
        print("-" * 80)
        print(f"{'AVERAGE':12s} {avg_sharpe:8.3f}")

        # Which features appear most across top symbols?
        from collections import Counter
        feat_counts = Counter()
        for r in results:
            feat_counts.update(r["top_features"])
        print(f"\n  Most important features across universe:")
        for feat, count in feat_counts.most_common(10):
            print(f"    {feat:40s}  appears in {count}/{len(results)} symbols")


if __name__ == "__main__":
    main()
