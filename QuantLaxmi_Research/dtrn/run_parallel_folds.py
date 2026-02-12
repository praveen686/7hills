#!/usr/bin/env python3
"""Launch all DTRN walk-forward folds in parallel.

8 folds total: 4 NIFTY + 4 BANKNIFTY.
Each fold runs as a separate process with its own CUDA context.

Auto-detects checkpoint state per fold:
  - dtrn_final.pt exists → --backtest-only (OOS re-run, ~2 min)
  - dtrn_stage2.pt exists → --resume-stage3 (stage3 training + OOS, ~15 min)
  - neither → full pipeline (stage1 + stage2 + stage3 + OOS, ~45 min)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parent / "run_single_fold.py"
RESULTS_DIR = Path("/home/ubuntu/Desktop/7hills/QuantLaxmi_Research/dtrn/results")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def detect_mode(fold_idx: int, instrument: str) -> str:
    """Auto-detect the right mode based on available checkpoints."""
    fold_dir = CHECKPOINT_DIR / f"fold{fold_idx}_{instrument}"
    final_path = fold_dir / "dtrn_final.pt"
    stage2_path = fold_dir / "dtrn_stage2.pt"

    if final_path.exists():
        return "backtest-only"
    elif stage2_path.exists():
        return "resume-stage3"
    else:
        return "full"


def main():
    folds = []
    for instrument in ["NIFTY", "BANKNIFTY"]:
        for fold_idx in range(1, 5):
            mode = detect_mode(fold_idx, instrument)
            folds.append((fold_idx, instrument, mode))

    print(f"{'='*70}", flush=True)
    print(f"  DTRN Parallel Walk-Forward: {len(folds)} folds", flush=True)
    print(f"{'='*70}", flush=True)

    bt_only = sum(1 for _, _, m in folds if m == "backtest-only")
    resume_s3 = sum(1 for _, _, m in folds if m == "resume-stage3")
    full = sum(1 for _, _, m in folds if m == "full")
    print(f"  Modes: {bt_only} backtest-only, {resume_s3} resume-stage3, {full} full", flush=True)

    for fold_idx, instrument, mode in folds:
        print(f"    fold{fold_idx}_{instrument}: {mode}", flush=True)
    print(flush=True)

    # Launch all processes
    processes = {}
    for fold_idx, instrument, mode in folds:
        log_file = RESULTS_DIR / f"fold{fold_idx}_{instrument}.log"
        cmd = [
            sys.executable, "-u",
            str(SCRIPT),
            "--fold", str(fold_idx),
            "--instrument", instrument,
        ]
        if mode == "backtest-only":
            cmd.append("--backtest-only")
        elif mode == "resume-stage3":
            cmd.append("--resume-stage3")

        p = subprocess.Popen(
            cmd,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent.parent),
        )
        processes[(fold_idx, instrument)] = (p, mode)
        print(f"  Launched fold {fold_idx} {instrument} [{mode}] (PID {p.pid}) → {log_file.name}", flush=True)

    print(f"\n  All {len(processes)} processes launched. Waiting...\n", flush=True)

    # Poll for completion
    completed = set()
    t0 = time.time()
    last_progress = 0
    while len(completed) < len(processes):
        time.sleep(10)
        elapsed = (time.time() - t0) / 60
        for key, (p, mode) in processes.items():
            if key not in completed and p.poll() is not None:
                completed.add(key)
                fold_idx, instrument = key
                status = "OK" if p.returncode == 0 else f"FAILED (rc={p.returncode})"
                print(f"  [{elapsed:.1f}m] Fold {fold_idx} {instrument} [{mode}]: {status}", flush=True)

        # Progress update every 2 min
        if int(elapsed) // 2 > last_progress:
            last_progress = int(elapsed) // 2
            running = len(processes) - len(completed)
            if running > 0:
                print(f"  [{elapsed:.1f}m] {len(completed)}/{len(processes)} done, {running} running", flush=True)

    total_time = (time.time() - t0) / 60
    print(f"\n  All folds complete in {total_time:.1f} min\n", flush=True)

    # Aggregate results
    for instrument in ["NIFTY", "BANKNIFTY"]:
        print(f"\n{'='*70}", flush=True)
        print(f"  AGGREGATE: {instrument}", flush=True)
        print(f"{'='*70}", flush=True)

        all_oos_returns = []
        fold_results = []

        for fold_idx in range(1, 5):
            result_file = RESULTS_DIR / f"fold{fold_idx}_{instrument}_result.json"
            if not result_file.exists():
                print(f"  Fold {fold_idx}: NO RESULT FILE", flush=True)
                continue

            with open(result_file) as f:
                info = json.load(f)

            fold_results.append(info)
            all_oos_returns.extend(info.get("daily_returns", []))

            print(f"  Fold {fold_idx}: Sharpe={info['sharpe']:.2f}  "
                  f"Return={info['return_pct']:+.2f}%  "
                  f"MaxDD={info['max_dd_pct']:.2f}%  "
                  f"Trades={info['trades']}  "
                  f"Mode={info.get('mode', '?')}  "
                  f"Time={info.get('fold_time_min', 0):.1f}m", flush=True)

        if all_oos_returns:
            arr = np.array(all_oos_returns)
            std = np.std(arr, ddof=1)
            sh = np.mean(arr) / std * np.sqrt(252) if std > 1e-10 else 0
            ret = (1 + arr).prod() - 1
            cum = np.cumprod(1 + arr)
            pk = np.maximum.accumulate(cum)
            mdd = ((pk - cum) / np.maximum(pk, 1e-10)).max()

            print(f"\n  Combined OOS ({len(arr)} days):", flush=True)
            print(f"    Sharpe:     {sh:.2f}", flush=True)
            print(f"    Return:     {ret*100:+.2f}%", flush=True)
            print(f"    Max DD:     {mdd*100:.2f}%", flush=True)

        sharpes = [f["sharpe"] for f in fold_results]
        if sharpes:
            print(f"    Per-fold:   {[f'{s:.2f}' for s in sharpes]}", flush=True)
            print(f"    Mean={np.mean(sharpes):.2f}  Std={np.std(sharpes):.2f}", flush=True)

    # Save aggregate
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"parallel_walkforward_{ts}.json"
    all_results = {}
    for instrument in ["NIFTY", "BANKNIFTY"]:
        fold_data = []
        for fold_idx in range(1, 5):
            result_file = RESULTS_DIR / f"fold{fold_idx}_{instrument}_result.json"
            if result_file.exists():
                with open(result_file) as f:
                    fold_data.append(json.load(f))
        all_results[instrument] = fold_data

    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Aggregate saved to {out}", flush=True)


if __name__ == "__main__":
    main()
