#!/usr/bin/env python3
"""
Extract a time window from JSONL capture files for CI fixture generation.

Usage:
    python3 tools/extract_window.py \
        --in-dir data/perp_sessions/perp_20260125_051437/BTCUSDT \
        --out-dir tests/fixtures/perp_session_fixture/BTCUSDT/window_20260125_052000_052100 \
        --start 2026-01-25T05:20:00Z \
        --end   2026-01-25T05:21:00Z
"""
import json
import argparse
from pathlib import Path
import pandas as pd


def ts_ns(ts: str) -> int:
    return int(pd.Timestamp(ts).value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", required=True, help="ISO, e.g. 2026-01-25T05:20:00Z")
    ap.add_argument("--end", required=True, help="ISO, e.g. 2026-01-25T05:21:00Z")
    args = ap.parse_args()

    start_ns = ts_ns(args.start)
    end_ns = ts_ns(args.end)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting window: {args.start} -> {args.end}")
    print(f"  From: {in_dir}")
    print(f"  To:   {out_dir}")
    print()

    total_written = 0
    for fname in ["perp_quotes.jsonl", "spot_quotes.jsonl", "funding.jsonl"]:
        src = in_dir / fname
        dst = out_dir / fname
        if not src.exists():
            print(f"  {fname}: skipped (not found)")
            continue
        w = 0
        with src.open("r") as fin, dst.open("w") as fout:
            for line in fin:
                if not line.strip():
                    continue
                o = json.loads(line)
                t = ts_ns(o["ts"])
                if t < start_ns:
                    continue
                if t >= end_ns:
                    break
                fout.write(line)
                w += 1
        print(f"  {fname}: {w} lines -> {dst}")
        total_written += w

    print()
    print(f"Total: {total_written} events extracted")

    # Write manifest for fixture
    manifest = {
        "source_dir": str(in_dir),
        "window_start": args.start,
        "window_end": args.end,
        "files": ["perp_quotes.jsonl", "spot_quotes.jsonl", "funding.jsonl"],
        "total_events": total_written,
    }
    manifest_path = out_dir / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
