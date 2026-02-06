#!/usr/bin/env python3
"""Triangle Arbitrage Analysis on 4-hour certified session with correct price exponents."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

SESSION_DIR = Path("/home/ubuntu/Desktop/7hills/QuantLaxmi/data/sessions/triangles/triangles_20260203_182151")
TAKER_FEE = 0.001  # 0.1% per leg
MAX_STALENESS_MS = 200

TRIANGLES = {
    "ETH-BTC": {"leg_a": "BTCUSDT", "leg_b": "ETHBTC", "leg_c": "ETHUSDT"},
    "BNB-BTC": {"leg_a": "BTCUSDT", "leg_b": "BNBBTC", "leg_c": "BNBUSDT"},
    "SOL-BTC": {"leg_a": "BTCUSDT", "leg_b": "SOLBTC", "leg_c": "SOLUSDT"},
}

# ── 0. Preconditions ─────────────────────────────────────────────────

print("=" * 70)
print("TRIANGLE ARBITRAGE ANALYSIS — 4hr Certified Session")
print("=" * 70)

with open(SESSION_DIR / "session_manifest.json") as f:
    manifest = json.load(f)

print(f"\nSession ID: {manifest['session_id']}")
print(f"Certified:  {manifest['determinism']['certified']}")
print(f"Duration:   {manifest['duration_secs']:.0f}s ({manifest['duration_secs']/3600:.1f}h)")
print(f"\nSymbols:")
for c in manifest["captures"]:
    print(f"  {c['symbol']:8s}  depth={c['events_written']:>8,}  trades={c['trades_written']:>8,}  gaps={c['gaps_detected']}")

assert manifest["determinism"]["certified"], "ABORT: session not certified"

# ── 1. Load depth quotes with per-record price exponents ─────────────

def load_depth(session_dir: Path, symbol: str) -> pd.DataFrame:
    depth_file = session_dir / symbol / "depth.jsonl"
    records = []
    with open(depth_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            bids = rec.get("bids", [])
            asks = rec.get("asks", [])
            if not bids or not asks:
                continue
            price_exp = rec.get("price_exponent", -2)
            scale = 10.0 ** price_exp
            best_bid = bids[0]["price"] * scale
            best_ask = asks[0]["price"] * scale
            if best_bid <= 0 or best_ask <= 0:
                continue
            records.append({
                "ts": pd.Timestamp(rec["ts"]),
                "bid": best_bid,
                "ask": best_ask,
            })
    df = pd.DataFrame(records).sort_values("ts").reset_index(drop=True)
    df["ts_ns"] = df["ts"].astype("int64")
    return df


print("\n── Loading depth data ──")
quotes = {}
symbols_needed = set()
for tri in TRIANGLES.values():
    symbols_needed.update([tri["leg_a"], tri["leg_b"], tri["leg_c"]])

for sym in sorted(symbols_needed):
    df = load_depth(SESSION_DIR, sym)
    quotes[sym] = df
    mid = (df["bid"].iloc[len(df)//2] + df["ask"].iloc[len(df)//2]) / 2
    spread_bps = ((df["ask"] - df["bid"]) / ((df["ask"] + df["bid"]) / 2) * 10000).median()
    print(f"  {sym:8s}  {len(df):>8,} quotes  mid~{mid:.8g}  spread_p50={spread_bps:.2f}bp")

# ── 2. Align quotes per triangle ─────────────────────────────────────

MAX_STALENESS_NS = MAX_STALENESS_MS * 1_000_000

def align_triangle(syms: list[str]) -> pd.DataFrame:
    all_ts = set()
    for s in syms:
        all_ts.update(quotes[s]["ts_ns"].tolist())
    result = pd.DataFrame({"ts_ns": sorted(all_ts)})
    for s in syms:
        sub = quotes[s][["ts_ns", "bid", "ask"]].copy()
        sub = sub.rename(columns={"bid": f"{s}_bid", "ask": f"{s}_ask"})
        sub[f"{s}_qts"] = sub["ts_ns"]
        result = pd.merge_asof(result, sub, on="ts_ns", direction="backward")
    # Staleness filter
    mask = pd.Series(True, index=result.index)
    for s in syms:
        result[f"{s}_stale"] = result["ts_ns"] - result[f"{s}_qts"]
        mask &= result[f"{s}_stale"] <= MAX_STALENESS_NS
    clean = result[mask].copy()
    return clean


# ── 3. Compute residuals for all triangles ───────────────────────────

print("\n" + "=" * 70)
print("TRIANGLE RESIDUALS")
print("=" * 70)

for tri_name, legs in TRIANGLES.items():
    a, b, c = legs["leg_a"], legs["leg_b"], legs["leg_c"]
    aligned = align_triangle([a, b, c])
    dropped_pct = 100 * (1 - len(aligned) / len(set().union(
        quotes[a]["ts_ns"], quotes[b]["ts_ns"], quotes[c]["ts_ns"])))

    # CW:  USDT → BTC → X → USDT
    #   buy A(ask), sell B(bid), sell C(bid)
    #   ε_cw = log(C_bid) - log(A_ask) - log(B_ask)
    aligned["eps_cw"] = (
        np.log(aligned[f"{c}_bid"])
        - np.log(aligned[f"{a}_ask"])
        - np.log(aligned[f"{b}_ask"])
    )

    # CCW: USDT → X → BTC → USDT
    #   buy C(ask), buy B(ask→bid for selling X for BTC...
    #   Actually: buy C(ask), sell B for BTC means sell X→BTC so we GET bid on B
    #   No: CCW = buy C(ask), then sell C-asset for BTC = buy B(ask)...
    #   Let me be precise:
    #   CCW: USDT→X→BTC→USDT = buy X with USDT (C_ask), sell X for BTC (B_bid), sell BTC for USDT (A_bid)
    #   ε_ccw = log(A_bid) + log(B_bid) - log(C_ask)
    aligned["eps_ccw"] = (
        np.log(aligned[f"{a}_bid"])
        + np.log(aligned[f"{b}_bid"])
        - np.log(aligned[f"{c}_ask"])
    )

    # Spreads
    for s in [a, b, c]:
        aligned[f"{s}_spread"] = (aligned[f"{s}_ask"] - aligned[f"{s}_bid"]) / (
            (aligned[f"{s}_ask"] + aligned[f"{s}_bid"]) / 2)
    aligned["spread_sum"] = aligned[f"{a}_spread"] + aligned[f"{b}_spread"] + aligned[f"{c}_spread"]

    # Fee-adjusted: 3 legs × taker fee each side
    fee_cost = 3 * TAKER_FEE  # 0.3% = 30 bps total
    aligned["eps_cw_net"] = aligned["eps_cw"] - fee_cost
    aligned["eps_ccw_net"] = aligned["eps_ccw"] - fee_cost

    # ── Stats ──
    print(f"\n{'─'*70}")
    print(f"  {tri_name}  ({a} / {b} / {c})")
    print(f"  Aligned quotes: {len(aligned):,}  (dropped {dropped_pct:.1f}% staleness)")
    print(f"{'─'*70}")

    for direction, col in [("CW ", "eps_cw"), ("CCW", "eps_ccw")]:
        eps = aligned[col]
        eps_net = aligned[f"{col}_net"]
        hr_gross = (eps > 0).mean() * 100
        hr_net = (eps_net > 0).mean() * 100

        print(f"\n  {direction} direction:")
        print(f"    Gross:  mean={eps.mean()*10000:+.2f}bp  std={eps.std()*10000:.2f}bp")
        print(f"            min={eps.min()*10000:+.1f}bp  max={eps.max()*10000:+.1f}bp")
        print(f"            p1={np.percentile(eps,1)*10000:+.2f}bp  p99={np.percentile(eps,99)*10000:+.2f}bp")
        print(f"    HR gross:     {hr_gross:.2f}%")
        print(f"    HR net(-30bp): {hr_net:.4f}%")

        # Spread conditioning
        spread_p50 = aligned["spread_sum"].median()
        tight = aligned["spread_sum"] <= spread_p50
        hr_tight = (eps[tight] > 0).mean() * 100
        hr_wide = (eps[~tight] > 0).mean() * 100
        print(f"    HR tight spread: {hr_tight:.2f}%  |  wide: {hr_wide:.2f}%")

        # Run durations (positive gross)
        mask_pos = eps > 0
        changes = mask_pos.astype(int).diff().fillna(0)
        starts = aligned.index[changes == 1]
        ends = aligned.index[changes == -1]
        if len(starts) > 0 and len(ends) > 0:
            if ends[0] < starts[0]:
                ends = ends[1:]
            n_runs = min(len(starts), len(ends))
            if n_runs > 0:
                durations_ms = []
                for i in range(n_runs):
                    dur = (aligned.loc[ends.values[i], "ts_ns"] - aligned.loc[starts.values[i], "ts_ns"]) / 1e6
                    durations_ms.append(dur)
                durations_ms = np.array(durations_ms)
                print(f"    Positive runs: {n_runs:,}  median={np.median(durations_ms):.0f}ms  p90={np.percentile(durations_ms,90):.0f}ms  max={np.max(durations_ms):.0f}ms")

    # Sanity: both CW and CCW positive simultaneously?
    both = ((aligned["eps_cw"] > 0) & (aligned["eps_ccw"] > 0)).mean() * 100
    print(f"\n  Sanity: both CW+CCW positive: {both:.2f}%")
    if both > 1:
        print("  ⚠ WARNING: high overlap — check formulas")

    # Best-of-two: max(eps_cw, eps_ccw)
    best = aligned[["eps_cw", "eps_ccw"]].max(axis=1)
    best_net = best - fee_cost
    hr_best_gross = (best > 0).mean() * 100
    hr_best_net = (best_net > 0).mean() * 100
    print(f"\n  Best-of-two (max CW/CCW):")
    print(f"    HR gross: {hr_best_gross:.2f}%")
    print(f"    HR net:   {hr_best_net:.4f}%")
    print(f"    p99:      {np.percentile(best,99)*10000:+.2f}bp")
    print(f"    max:      {best.max()*10000:+.2f}bp")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"\nFee hurdle: 3 × {TAKER_FEE*100:.1f}% = {fee_cost*10000:.0f} bps per round trip")
print("If HR_net > 0 with meaningful frequency → signal exists")
print("If HR_net ≈ 0 → gross residuals are spread artifacts, not real arb")
