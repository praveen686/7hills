#!/usr/bin/env python3
"""
B1.2 Baseline LightGBM Model for Routing Decision Quality Prediction.

Target: y = edge_30s_bps - execution_tax_bps
  - Positive y => good decision (net value after execution)
  - Negative y => bad decision (no edge or execution too expensive)

Time-safe split: 70% train / 15% valid / 15% test (chronological)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import lightgbm as lgb

LABELS_PATH = "/tmp/labels_b12_v3.jsonl"

print("=" * 60)
print("B1.2 Baseline Model Training")
print("=" * 60)

# ----------------------------
# Load JSONL
# ----------------------------
print("\n[1] Loading labels...")
rows = []
with open(LABELS_PATH, "r") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

df = pd.json_normalize(rows)
print(f"    Loaded {len(df)} labels")

# ----------------------------
# Target
# ----------------------------
print("\n[2] Computing target: y = edge_30s_bps - execution_tax_bps")
df["edge_30s_bps"] = df["horizons.edge_30s_bps"].fillna(0.0)
df["execution_tax_bps"] = df["execution_tax_bps"].fillna(0.0)
df["y"] = df["edge_30s_bps"] - df["execution_tax_bps"]

print(f"    Target y: mean={df['y'].mean():.2f}, std={df['y'].std():.2f}")
print(f"    Range: [{df['y'].min():.2f}, {df['y'].max():.2f}]")

# ----------------------------
# Features
# ----------------------------
print("\n[3] Building features...")
df["vel_used"] = df["features.vel_used"].astype(int)
df["dt_ms"] = df["features.dt_ms"].fillna(0).astype(int)
df["spread_bps"] = df["features.spread_bps"].astype(float)
df["pressure"] = df["features.pressure"].astype(float)
df["signal_strength"] = df["features.signal_strength"].astype(float)
df["vel_abs_bps_sec"] = df["features.vel_abs_bps_sec"].astype(float)

# Derived features
df["log_spread"] = np.log1p(df["spread_bps"].clip(lower=0))
df["log_dt"] = np.log1p(df["dt_ms"].clip(lower=0))
df["abs_pressure"] = df["pressure"].abs()

# One-hot encode categoricals
cat = pd.get_dummies(df[["side", "order_type"]], drop_first=False)

feature_cols = [
    "spread_bps", "log_spread", "pressure", "abs_pressure", "signal_strength",
    "vel_abs_bps_sec", "vel_used", "dt_ms", "log_dt"
]
X = pd.concat([df[feature_cols], cat], axis=1)
y = df["y"]

print(f"    Feature matrix: {X.shape}")
print(f"    Features: {list(X.columns)}")

# ----------------------------
# Time-safe split (already sorted by ts_utc)
# ----------------------------
print("\n[4] Time-safe split (70/15/15)...")
n = len(X)
n_train = int(0.70 * n)
n_valid = int(0.85 * n)

X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
X_valid, y_valid = X.iloc[n_train:n_valid], y.iloc[n_train:n_valid]
X_test, y_test = X.iloc[n_valid:], y.iloc[n_valid:]

print(f"    Train: {len(X_train)} samples")
print(f"    Valid: {len(X_valid)} samples")
print(f"    Test:  {len(X_test)} samples")

# ----------------------------
# LightGBM Training
# ----------------------------
print("\n[5] Training LightGBM (Huber loss)...")
params = dict(
    objective="huber",
    alpha=0.9,
    metric=["l2", "mae"],
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    verbosity=-1,
    seed=42,
)

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

model = lgb.train(
    params,
    dtrain,
    num_boost_round=5000,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)]
)

print(f"    Best iteration: {model.best_iteration}")

# ----------------------------
# Evaluation
# ----------------------------
print("\n[6] Evaluation on TEST set...")
pred = model.predict(X_test, num_iteration=model.best_iteration)

mae = mean_absolute_error(y_test, pred)
rho, pval = spearmanr(y_test, pred)
print(f"    MAE (bps): {mae:.3f}")
print(f"    Spearman rho: {rho:.4f} (p={pval:.2e})")

# ----------------------------
# Top-K Uplift Analysis
# ----------------------------
print("\n[7] Top-K Uplift Analysis...")
test_df = X_test.copy()
test_df["y"] = y_test.values
test_df["pred"] = pred
test_df["filled"] = df.iloc[n_valid:]["filled"].values
test_df = test_df.sort_values("pred", ascending=False)

def bucket_stats(frac, name):
    k = int(len(test_df) * frac)
    bucket = test_df.head(k)
    mean_y = bucket["y"].mean()
    mean_pred = bucket["pred"].mean()
    fill_rate = bucket["filled"].mean() * 100
    return f"    {name}: mean_y={mean_y:+.2f} bps, mean_pred={mean_pred:+.2f}, fill_rate={fill_rate:.1f}%"

overall_mean = test_df["y"].mean()
print(f"    Overall: mean_y={overall_mean:+.2f} bps")
print(bucket_stats(0.10, "Top 10%"))
print(bucket_stats(0.20, "Top 20%"))
print(bucket_stats(0.50, "Top 50%"))

# Bottom bucket (worst predicted)
bottom_20 = test_df.tail(int(len(test_df) * 0.20))
print(f"    Bot 20%: mean_y={bottom_20['y'].mean():+.2f} bps (should be worse)")

# ----------------------------
# Feature Importance
# ----------------------------
print("\n[8] Feature Importance (gain)...")
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(importance.head(10).to_string(index=False))

# ----------------------------
# Counterfactual Market vs Limit Scoring
# ----------------------------
print("\n[9] Counterfactual Routing (Market vs Limit)...")

# Create copies of test features with forced order_type
X_test_market = X_test.copy()
X_test_limit = X_test.copy()

# Force order_type one-hot
X_test_market["order_type_Market"] = 1
X_test_market["order_type_Limit"] = 0
X_test_limit["order_type_Market"] = 0
X_test_limit["order_type_Limit"] = 1

# Score both variants
pred_market = model.predict(X_test_market, num_iteration=model.best_iteration)
pred_limit = model.predict(X_test_limit, num_iteration=model.best_iteration)

# Choose the better predicted option
test_df["pred_market"] = pred_market
test_df["pred_limit"] = pred_limit
test_df["cf_choice"] = np.where(pred_market > pred_limit, "Market", "Limit")
test_df["cf_pred"] = np.maximum(pred_market, pred_limit)

# What did the model prefer?
n_prefer_market = (test_df["cf_choice"] == "Market").sum()
n_prefer_limit = (test_df["cf_choice"] == "Limit").sum()
print(f"    Model prefers Market: {n_prefer_market} ({n_prefer_market/len(test_df)*100:.1f}%)")
print(f"    Model prefers Limit:  {n_prefer_limit} ({n_prefer_limit/len(test_df)*100:.1f}%)")

# Compare to actual order_type performance
actual_market = test_df[test_df["order_type_Market"] == 1]
actual_limit = test_df[test_df["order_type_Limit"] == 1]
print(f"    Actual Market orders: mean_y={actual_market['y'].mean():+.2f} bps (n={len(actual_market)})")
print(f"    Actual Limit orders:  mean_y={actual_limit['y'].mean():+.2f} bps (n={len(actual_limit)})")

# ----------------------------
# Quantile-Based Router (Top X%)
# ----------------------------
print("\n[10] Quantile-Based Router...")

def quantile_router(df, take_pct, pred_col="pred"):
    """Take top X% by predicted value"""
    threshold = df[pred_col].quantile(1 - take_pct)
    take_mask = df[pred_col] >= threshold
    take_df = df[take_mask]
    skip_df = df[~take_mask]

    n_take = len(take_df)
    mean_y = take_df["y"].mean() if n_take > 0 else 0
    total_y = take_df["y"].sum() if n_take > 0 else 0
    skip_mean = skip_df["y"].mean() if len(skip_df) > 0 else 0

    return {
        "take_pct": take_pct * 100,
        "n_take": n_take,
        "mean_y": mean_y,
        "total_y": total_y,
        "skip_mean_y": skip_mean,
    }

print("\n    Quantile Router Results:")
print("    " + "-" * 65)
print(f"    {'Take%':>6} | {'N':>4} | {'Mean Y (bps)':>12} | {'Total Y (bps)':>13} | {'Skip Mean':>10}")
print("    " + "-" * 65)

for pct in [0.20, 0.30, 0.40, 0.50, 0.70, 1.00]:
    r = quantile_router(test_df, pct)
    print(f"    {r['take_pct']:>5.0f}% | {r['n_take']:>4} | {r['mean_y']:>+12.2f} | {r['total_y']:>+13.2f} | {r['skip_mean_y']:>+10.2f}")

# ----------------------------
# B1 Rules Comparison
# ----------------------------
print("\n[11] B1 Rules vs ML Router Comparison...")

# B1 baseline: take all trades as-is (no ML filtering)
b1_total_y = test_df["y"].sum()
b1_mean_y = test_df["y"].mean()
b1_worst_10pct = test_df.nsmallest(int(len(test_df) * 0.10), "y")["y"].mean()

# ML Router: take top 30%
ml_30 = quantile_router(test_df, 0.30)
ml_worst_10pct_of_taken = test_df.nlargest(int(len(test_df) * 0.30), "pred").nsmallest(
    max(1, int(len(test_df) * 0.30 * 0.10)), "y"
)["y"].mean()

print("\n    Comparison Table:")
print("    " + "-" * 55)
print(f"    {'Metric':>20} | {'B1 (All)':>12} | {'ML Top 30%':>12}")
print("    " + "-" * 55)
print(f"    {'Trade Count':>20} | {len(test_df):>12} | {ml_30['n_take']:>12}")
print(f"    {'Mean Y (bps)':>20} | {b1_mean_y:>+12.2f} | {ml_30['mean_y']:>+12.2f}")
print(f"    {'Total Y (bps)':>20} | {b1_total_y:>+12.2f} | {ml_30['total_y']:>+12.2f}")
print(f"    {'Worst 10% Mean':>20} | {b1_worst_10pct:>+12.2f} | {ml_worst_10pct_of_taken:>+12.2f}")
print("    " + "-" * 55)

improvement = ml_30["mean_y"] - b1_mean_y
print(f"\n    ML Uplift: {improvement:+.2f} bps per trade")

# ----------------------------
# By Order Type Analysis
# ----------------------------
print("\n[10] Analysis by Order Type...")
for otype in ["Market", "Limit"]:
    col = f"order_type_{otype}"
    if col in test_df.columns:
        mask = test_df[col] == 1
        subset = test_df[mask]
        if len(subset) > 0:
            print(f"    {otype}: n={len(subset)}, mean_y={subset['y'].mean():+.2f}, mean_pred={subset['pred'].mean():+.2f}")

print("\n" + "=" * 60)
print("B1.2 Baseline Complete")
print("=" * 60)

# Save model
model.save_model("/tmp/b12_baseline_model.txt")
print(f"\nModel saved to: /tmp/b12_baseline_model.txt")
