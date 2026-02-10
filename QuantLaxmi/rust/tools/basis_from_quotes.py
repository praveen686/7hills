#!/usr/bin/env python3
"""
Compute basis series (spot vs perp mid) and basic sanity stats.

Usage:
    python3 tools/basis_from_quotes.py \
        --segment-dir data/perp_sessions/perp_20260125_051437/BTCUSDT \
        --out-dir data/perp_sessions/perp_20260125_family_stats/segA \
        --tolerance-ms 200
"""
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def parse_ts_to_ns(ts: str) -> int:
    """
    Handles:
    - "2026-01-25T05:14:38.042Z" (ms)
    - "2026-01-25T05:14:38.278215190Z" (ns)
    Use pandas for ISO parsing; keep ns precision.
    """
    return int(pd.Timestamp(ts).value)  # ns since epoch


def read_quotes_jsonl(path: Path, kind: str) -> pd.DataFrame:
    rows = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            ts_ns = parse_ts_to_ns(o["ts"])
            px_exp = int(o["price_exponent"])
            q_exp = int(o["qty_exponent"])
            bid = int(o["bid_price_mantissa"]) * (10.0 ** px_exp)
            ask = int(o["ask_price_mantissa"]) * (10.0 ** px_exp)
            mid = (bid + ask) / 2.0
            spread = ask - bid
            rows.append((ts_ns, o["symbol"], bid, ask, mid, spread, px_exp, q_exp))
    df = pd.DataFrame(
        rows,
        columns=[
            "ts_ns",
            "symbol",
            f"{kind}_bid",
            f"{kind}_ask",
            f"{kind}_mid",
            f"{kind}_spread",
            "price_exp",
            "qty_exp",
        ],
    )
    df.sort_values("ts_ns", inplace=True)
    return df


def read_funding_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            ts_ns = parse_ts_to_ns(o["ts"])
            px_exp = int(o["price_exponent"])
            rate_exp = int(o["rate_exponent"])
            mark = int(o["mark_price_mantissa"]) * (10.0 ** px_exp)
            indexp = int(o["index_price_mantissa"]) * (10.0 ** px_exp)
            fund = int(o["funding_rate_mantissa"]) * (10.0 ** rate_exp)
            nxt_ms = int(o["next_funding_time_ms"])
            rows.append((ts_ns, o["symbol"], mark, indexp, fund, nxt_ms))
    df = pd.DataFrame(
        rows,
        columns=[
            "ts_ns",
            "symbol",
            "mark_price",
            "index_price",
            "funding_rate",
            "next_funding_time_ms",
        ],
    )
    df.sort_values("ts_ns", inplace=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--segment-dir",
        required=True,
        help="e.g. data/perp_sessions/perp_20260125_051437/BTCUSDT",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="e.g. data/perp_sessions/perp_20260125_family_stats",
    )
    ap.add_argument(
        "--tolerance-ms",
        type=int,
        default=200,
        help="max time diff for spot<->perp pairing via nearest join",
    )
    args = ap.parse_args()

    seg = Path(args.segment_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading spot quotes from {seg / 'spot_quotes.jsonl'}...")
    spot = read_quotes_jsonl(seg / "spot_quotes.jsonl", "spot")
    print(f"  -> {len(spot)} rows")

    print(f"Reading perp quotes from {seg / 'perp_quotes.jsonl'}...")
    perp = read_quotes_jsonl(seg / "perp_quotes.jsonl", "perp")
    print(f"  -> {len(perp)} rows")

    print(f"Reading funding from {seg / 'funding.jsonl'}...")
    funding = read_funding_jsonl(seg / "funding.jsonl")
    print(f"  -> {len(funding)} rows")

    # nearest-time join: spot aligned to perp timestamps
    tol_ns = args.tolerance_ms * 1_000_000
    spot_idx = spot[["ts_ns", "spot_mid", "spot_spread"]].copy()
    perp_idx = perp[["ts_ns", "perp_mid", "perp_spread"]].copy()

    spot_idx.sort_values("ts_ns", inplace=True)
    perp_idx.sort_values("ts_ns", inplace=True)

    print(f"Merging spot<->perp with tolerance {args.tolerance_ms}ms...")
    merged = pd.merge_asof(
        perp_idx, spot_idx, on="ts_ns", direction="nearest", tolerance=tol_ns
    ).dropna()
    print(f"  -> {len(merged)} paired rows")

    merged["basis_abs"] = merged["perp_mid"] - merged["spot_mid"]
    merged["basis_rel"] = merged["basis_abs"] / merged["spot_mid"]

    merged.to_csv(out / "basis_series.csv", index=False)

    # funding alignment: nearest funding to each perp point (tolerance wider)
    merged_f = pd.merge_asof(
        merged.sort_values("ts_ns"),
        funding[
            ["ts_ns", "funding_rate", "mark_price", "index_price", "next_funding_time_ms"]
        ].sort_values("ts_ns"),
        on="ts_ns",
        direction="backward",
    )
    merged_f.to_csv(out / "basis_with_funding.csv", index=False)

    # summary stats
    def pct(x, p):
        return float(np.nanpercentile(x, p))

    stats = {
        "rows_basis": int(len(merged)),
        "spot_spread_p50": pct(merged["spot_spread"], 50),
        "spot_spread_p95": pct(merged["spot_spread"], 95),
        "perp_spread_p50": pct(merged["perp_spread"], 50),
        "perp_spread_p95": pct(merged["perp_spread"], 95),
        "basis_rel_p50": pct(merged["basis_rel"], 50),
        "basis_rel_p95": pct(merged["basis_rel"], 95),
        "basis_rel_p99": pct(merged["basis_rel"], 99),
        "basis_rel_min": float(merged["basis_rel"].min()),
        "basis_rel_max": float(merged["basis_rel"].max()),
        "basis_abs_p50": pct(merged["basis_abs"], 50),
        "basis_abs_p95": pct(merged["basis_abs"], 95),
    }
    (out / "basis_stats.json").write_text(json.dumps(stats, indent=2))

    print()
    print("=== Basis Stats ===")
    print(f"  Paired rows:      {stats['rows_basis']:,}")
    print(f"  Spot spread p50:  ${stats['spot_spread_p50']:.4f}")
    print(f"  Spot spread p95:  ${stats['spot_spread_p95']:.4f}")
    print(f"  Perp spread p50:  ${stats['perp_spread_p50']:.4f}")
    print(f"  Perp spread p95:  ${stats['perp_spread_p95']:.4f}")
    print(f"  Basis (rel) p50:  {stats['basis_rel_p50']*10000:.2f} bps")
    print(f"  Basis (rel) p95:  {stats['basis_rel_p95']*10000:.2f} bps")
    print(f"  Basis (rel) p99:  {stats['basis_rel_p99']*10000:.2f} bps")
    print(f"  Basis (rel) min:  {stats['basis_rel_min']*10000:.2f} bps")
    print(f"  Basis (rel) max:  {stats['basis_rel_max']*10000:.2f} bps")
    print()
    print("Wrote:", out / "basis_series.csv")
    print("Wrote:", out / "basis_with_funding.csv")
    print("Wrote:", out / "basis_stats.json")


if __name__ == "__main__":
    main()
