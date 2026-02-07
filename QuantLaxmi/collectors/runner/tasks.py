"""Task registry for BRAHMASTRA qlx_runner.

Each task is a callable with metadata (phase, description, dependencies).
Tasks live in their proper modules (qlx/data/, qlx/features/) — this
registry just indexes them for the runner CLI.

To add a new task:
    1. Implement the logic in the appropriate qlx/ module
    2. Register it here with @register(phase=..., description=...)
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from .log import log_fail, log_info, log_ok, log_start

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A registered runner task."""

    name: str
    phase: str
    description: str
    fn: Callable
    dependencies: list[str] = field(default_factory=list)


# Global task registry
_TASKS: dict[str, Task] = {}


def register(
    phase: str,
    description: str,
    dependencies: list[str] | None = None,
) -> Callable:
    """Decorator to register a task function."""

    def wrapper(fn: Callable) -> Callable:
        name = fn.__name__
        _TASKS[name] = Task(
            name=name,
            phase=phase,
            description=description,
            fn=fn,
            dependencies=dependencies or [],
        )
        return fn

    return wrapper


def get_task(name: str) -> Task | None:
    return _TASKS.get(name)


def list_tasks(phase: str | None = None) -> list[Task]:
    tasks = list(_TASKS.values())
    if phase:
        tasks = [t for t in tasks if t.phase == phase]
    return sorted(tasks, key=lambda t: (t.phase, t.name))


def run_task(
    name: str,
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run a single registered task."""
    task = _TASKS.get(name)
    if not task:
        raise ValueError(f"Unknown task: {name}. Available: {list(_TASKS.keys())}")

    log_start(task.phase, task.name, task.description)
    t0 = time.time()

    try:
        result = task.fn(dates=dates, force=force, dry_run=dry_run)
        elapsed = time.time() - t0
        log_ok(task.phase, task.name, f"Completed in {elapsed:.1f}s", detail=str(result))
        return {"status": "ok", "elapsed": elapsed, "result": result}
    except Exception as e:
        elapsed = time.time() - t0
        log_fail(task.phase, task.name, str(e), detail=traceback.format_exc())
        return {"status": "fail", "elapsed": elapsed, "error": str(e)}


def run_phase(
    phase: str,
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Run all tasks in a phase, respecting dependencies."""
    tasks = list_tasks(phase)
    if not tasks:
        print(f"  No tasks registered for phase '{phase}'")
        return {}

    completed: set[str] = set()
    results: dict[str, dict] = {}

    for i, task in enumerate(tasks, 1):
        # Check dependencies
        missing_deps = [d for d in task.dependencies if d not in completed]
        if missing_deps:
            msg = f"Skipped (missing dependencies: {', '.join(missing_deps)})"
            log_info(task.phase, task.name, msg)
            results[task.name] = {"status": "skipped", "reason": msg}
            continue

        print(f"  [{i}/{len(tasks)}] {task.description}...")
        result = run_task(task.name, dates=dates, force=force, dry_run=dry_run)
        results[task.name] = result

        if result["status"] == "ok":
            completed.add(task.name)

    return results


# =========================================================================
# Phase 0: Foundation Hardening
# =========================================================================

@register(phase="phase0", description="Add explicit timestamps to 1-min bars")
def add_timestamps(
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Add explicit timestamp column to 1-min bar parquets.

    For each 1-min bar row within a (date, symbol) group, computes:
        timestamp = date + 09:15:00 IST + row_offset * 1 minute

    Handles two sources:
      1. Feather files (unconverted) — read feather, add timestamp, write parquet
      2. Existing parquets without timestamp — read parquet, add timestamp, rewrite
    """
    from core.data.convert import (
        MARKET_DIR,
        _feather_path,
        _parquet_out,
        discover_converted,
        discover_sources,
    )

    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    def _add_timestamps_to_df(df: pd.DataFrame, d: date) -> pd.DataFrame:
        """Add timestamp column to a 1-min bar DataFrame."""
        # Normalize timezone-aware date column
        if "date" in df.columns and hasattr(df["date"].dtype, "tz"):
            df["date"] = df["date"].dt.tz_localize(None)

        # Downcast float64 -> float32
        for col in df.select_dtypes("float64").columns:
            df[col] = df[col].astype(np.float32)

        # 09:15:00 IST + row_offset per symbol group
        base_time = pd.Timestamp(d.isoformat()) + pd.Timedelta(
            hours=9, minutes=15
        )
        timestamps = []
        group_col = "symbol" if "symbol" in df.columns else None
        if group_col:
            for _, grp in df.groupby(group_col, sort=False):
                n = len(grp)
                ts = pd.date_range(base_time, periods=n, freq="min")
                timestamps.append(pd.Series(ts, index=grp.index))
            df["timestamp"] = pd.concat(timestamps).sort_index()
        else:
            df["timestamp"] = pd.date_range(base_time, periods=len(df), freq="min")

        cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
        return df[cols]

    results = {}

    for category in ["nfo_1min", "bfo_1min"]:
        kind = "nfo" if category == "nfo_1min" else "bfo"
        cat_dir = MARKET_DIR / category

        if not cat_dir.exists():
            continue

        # Collect all date directories with parquet
        for date_dir in sorted(cat_dir.iterdir()):
            if not date_dir.is_dir() or not date_dir.name.startswith("date="):
                continue

            try:
                d = date.fromisoformat(date_dir.name[5:])
            except ValueError:
                continue

            if dates and d not in set(dates):
                continue

            pq_path = date_dir / "data.parquet"
            if not pq_path.exists():
                continue

            # Check if already has timestamp
            if not force:
                try:
                    schema = pq.read_schema(pq_path)
                    if "timestamp" in schema.names:
                        continue
                except Exception:
                    pass

            if dry_run:
                logger.info("[DRY-RUN] Would add timestamps: %s %s", category, d)
                continue

            # Try feather source first, fall back to existing parquet
            feather_src = _feather_path(d, kind)
            try:
                if feather_src.exists():
                    df = pd.read_feather(feather_src)
                else:
                    df = pd.read_parquet(pq_path)
            except Exception as e:
                logger.warning("Skipping %s/%s: %s", category, d, e)
                continue

            if df.empty:
                continue

            df = _add_timestamps_to_df(df, d)

            pq_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(pq_path, engine="pyarrow", compression="zstd", index=False)

            results[f"{category}/{d}"] = len(df)

    return results


@register(phase="phase0", description="Analyze and filter tick anomalies")
def filter_ticks(
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Create ticks_clean view excluding anomalous rows.

    Rather than rewriting parquet (expensive for 8+ GB), we create a SQL
    view that excludes epoch timestamps and zero-ltp rows.
    """
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent.parent)
    )
    from core.data.store import MarketDataStore

    store = MarketDataStore()

    query_dates = ""
    if dates:
        date_strs = ", ".join(f"'{d.isoformat()}'" for d in dates)
        query_dates = f" AND date IN ({date_strs})"

    df = store.sql(f"""
        SELECT
            date,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE timestamp < '2000-01-01') as epoch_ts,
            COUNT(*) FILTER (WHERE ltp <= 0.05) as zero_ltp,
            COUNT(*) FILTER (WHERE timestamp < '2000-01-01' OR ltp <= 0.05) as anomalous
        FROM ticks
        WHERE 1=1 {query_dates}
        GROUP BY date
        ORDER BY date
    """)

    total_rows = int(df["total"].sum())
    total_anomalous = int(df["anomalous"].sum())
    pct = (total_anomalous / total_rows * 100) if total_rows > 0 else 0

    report: dict[str, object] = {
        "total_rows": total_rows,
        "anomalous_rows": total_anomalous,
        "pct_anomalous": round(pct, 2),
        "dates_analyzed": len(df),
    }

    if not dry_run:
        view_sql = (
            "CREATE OR REPLACE VIEW ticks_clean AS "
            "SELECT * FROM ticks "
            "WHERE timestamp >= '2000-01-01' AND ltp > 0.05"
        )
        store._con.execute(view_sql)
        report["view_created"] = "ticks_clean"

    store.close()
    return report


@register(phase="phase0", description="Verify Phase 0 feature modules exist")
def verify_modules(
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, bool]:
    """Check which Phase 0 feature modules exist."""
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent.parent / "qlx" / "features"
    modules = {
        "iv.py": base / "iv.py",
        "ramanujan.py": base / "ramanujan.py",
        "microstructure.py": base / "microstructure.py",
    }
    return {name: path.exists() for name, path in modules.items()}


# =========================================================================
# Status
# =========================================================================

def status() -> dict:
    """Return status summary across all phases."""
    import pyarrow.parquet as pq
    from pathlib import Path

    from core.data.convert import MARKET_DIR

    info: dict[str, dict] = {"tasks": {}}

    # Phase 0: timestamps
    sample = MARKET_DIR / "nfo_1min"
    has_ts = False
    if sample.exists():
        for pf in sorted(sample.rglob("*.parquet"))[:1]:
            schema = pq.read_schema(pf)
            has_ts = "timestamp" in schema.names
    info["tasks"]["add_timestamps"] = "DONE" if has_ts else "PENDING"

    # Phase 0: tick cleaning (view is ephemeral)
    info["tasks"]["filter_ticks"] = "READY"

    # Phase 0: feature modules
    modules = verify_modules()
    for name, exists in modules.items():
        info["tasks"][name] = "DONE" if exists else "PENDING"

    # Phase 0: delta backfill
    delta_dir = MARKET_DIR / "nse_contract_delta"
    if delta_dir.exists():
        n_dates = len([d for d in delta_dir.iterdir() if d.is_dir()])
        info["tasks"]["backfill_deltas"] = f"{n_dates} dates"
    else:
        info["tasks"]["backfill_deltas"] = "PENDING"

    # Registered tasks summary
    info["registered"] = {}
    for phase_name in sorted({t.phase for t in _TASKS.values()}):
        phase_tasks = list_tasks(phase_name)
        info["registered"][phase_name] = [t.name for t in phase_tasks]

    return info


# =========================================================================
# Phase 0: Backfill BS Deltas for pre-Sep-2025 dates
# =========================================================================

@register(phase="phase0", description="Backfill BS deltas for pre-Sep-2025 dates")
def backfill_deltas(
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Compute Black-Scholes deltas for dates with fo_bhavcopy but no contract_delta.

    Uses NSE's published daily volatility (applicable_ann_vol) as sigma,
    and the GPU-vectorized bs_delta() from iv_engine.
    """
    import sys
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from core.pricing.iv_engine import _DEVICE, _DTYPE, bs_delta
    from core.data.convert import MARKET_DIR
    from core.data.store import MarketDataStore

    store = MarketDataStore()

    # Discover dates with fo_bhavcopy but no contract_delta
    bv_dir = MARKET_DIR / "nse_fo_bhavcopy"
    delta_dir = MARKET_DIR / "nse_contract_delta"

    bv_dates: set[str] = set()
    if bv_dir.exists():
        for d in bv_dir.iterdir():
            if d.is_dir() and d.name.startswith("date="):
                bv_dates.add(d.name[5:])

    delta_dates: set[str] = set()
    if delta_dir.exists():
        for d in delta_dir.iterdir():
            if d.is_dir() and d.name.startswith("date="):
                delta_dates.add(d.name[5:])

    missing = sorted(bv_dates - delta_dates)

    if dates:
        date_strs = {d.isoformat() for d in dates}
        missing = [m for m in missing if m in date_strs]

    if not force:
        # Only process dates that don't already have delta data
        pass  # missing already excludes delta_dates

    results: dict[str, int] = {}

    for d_str in missing:
        if dry_run:
            logger.info("[DRY-RUN] Would compute deltas for %s", d_str)
            results[d_str] = 0
            continue

        try:
            d = date.fromisoformat(d_str)
        except ValueError:
            continue

        # Load fo_bhavcopy for this date
        fno = store.sql(
            "SELECT * FROM nse_fo_bhavcopy WHERE date = ?", [d_str]
        )
        if fno is None or fno.empty:
            continue

        # Load volatility for this date
        vol = store.sql(
            "SELECT symbol, applicable_ann_vol FROM nse_volatility WHERE date = ?",
            [d_str],
        )
        if vol is None or vol.empty:
            logger.warning("No volatility data for %s, skipping", d_str)
            continue

        vol_map = dict(zip(vol["symbol"], vol["applicable_ann_vol"]))

        # Separate options and futures
        options = fno[fno["FinInstrmTp"].isin(["STO", "IDO"])].copy()
        futures = fno[fno["FinInstrmTp"].isin(["STF", "IDF"])].copy()

        delta_rows = []

        # Futures: delta = 1.0
        for _, row in futures.iterrows():
            delta_rows.append({
                "Date": row.get("TradDt", d_str),
                "Symbol": row.get("TckrSymb", ""),
                "Expiry day": row.get("XpryDt", ""),
                "Strike Price": 0.0,
                "Option Type": "FF",
                "Delta Factor": 1.0,
            })

        # Options: compute BS delta
        if not options.empty:
            # Map symbol → sigma from volatility data
            options["_sigma"] = options["TckrSymb"].map(vol_map)
            options = options.dropna(subset=["_sigma"])

            if not options.empty:
                # Parse expiry, compute T
                trade_dt = pd.to_datetime(options["TradDt"].iloc[0], format="mixed")
                options["_expiry_dt"] = pd.to_datetime(
                    options["XpryDt"].astype(str).str.strip(), format="mixed"
                )
                options["_T"] = (
                    (options["_expiry_dt"] - trade_dt).dt.days / 365.0
                ).clip(lower=1e-6)

                S = torch.tensor(
                    options["UndrlygPric"].values.astype(np.float64),
                    dtype=_DTYPE, device=_DEVICE,
                )
                K = torch.tensor(
                    options["StrkPric"].values.astype(np.float64),
                    dtype=_DTYPE, device=_DEVICE,
                )
                T = torch.tensor(
                    options["_T"].values.astype(np.float64),
                    dtype=_DTYPE, device=_DEVICE,
                )
                r = torch.full_like(S, 0.065)
                sigma = torch.tensor(
                    options["_sigma"].values.astype(np.float64),
                    dtype=_DTYPE, device=_DEVICE,
                )
                is_call = torch.tensor(
                    (options["OptnTp"].str.strip() == "CE").values,
                    dtype=torch.bool, device=_DEVICE,
                )

                deltas = bs_delta(S, K, T, r, sigma, is_call).cpu().numpy()

                for i, (_, row) in enumerate(options.iterrows()):
                    delta_rows.append({
                        "Date": row.get("TradDt", d_str),
                        "Symbol": row.get("TckrSymb", ""),
                        "Expiry day": row.get("XpryDt", ""),
                        "Strike Price": float(row.get("StrkPric", 0)),
                        "Option Type": row.get("OptnTp", "").strip(),
                        "Delta Factor": float(deltas[i]),
                    })

        if delta_rows:
            df_out = pd.DataFrame(delta_rows)
            out_dir = delta_dir / f"date={d_str}"
            out_dir.mkdir(parents=True, exist_ok=True)
            df_out.to_parquet(
                out_dir / "data.parquet",
                engine="pyarrow",
                compression="zstd",
                index=False,
            )
            results[d_str] = len(delta_rows)
            logger.info("Wrote %d delta rows for %s", len(delta_rows), d_str)

    store.close()
    return results


# =========================================================================
# Phase: Strategies — wrappers for unified orchestration
# =========================================================================

@register(phase="strategies", description="India FNO IV mean-reversion scan")
def india_fno_scan(
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run india_fno scan for a single day."""
    import argparse
    from _archive.india_fno_legacy.__main__ import cmd_scan
    from _archive.india_fno_legacy.paper_state import DEFAULT_STATE_FILE

    target = dates[0].isoformat() if dates else date.today().isoformat()

    if dry_run:
        return {"action": "scan", "date": target, "dry_run": True}

    args = argparse.Namespace(
        date=target,
        state_file=str(DEFAULT_STATE_FILE),
        verbose=False,
    )
    cmd_scan(args)
    return {"action": "scan", "date": target}


@register(phase="strategies", description="RNDR density regime scan")
def rndr_scan(
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run RNDR scan for a single day."""
    import argparse
    from _archive.india_fno_legacy.rndr.__main__ import cmd_scan
    from _archive.india_fno_legacy.rndr.state import DEFAULT_STATE_FILE

    target = dates[0].isoformat() if dates else date.today().isoformat()

    if dry_run:
        return {"action": "scan", "date": target, "dry_run": True}

    args = argparse.Namespace(
        date=target,
        state_file=str(DEFAULT_STATE_FILE),
        verbose=False,
    )
    cmd_scan(args)
    return {"action": "scan", "date": target}


@register(phase="strategies", description="Institutional footprint scanner scan")
def india_scanner_scan(
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run india_scanner daily scan."""
    import argparse
    from _archive.india_scanner_legacy.__main__ import cmd_scan

    target = dates[0].isoformat() if dates else date.today().isoformat()

    if dry_run:
        return {"action": "scan", "date": target, "dry_run": True}

    args = argparse.Namespace(
        date=target,
        top_n=10,
        verbose=False,
        state_file="data/india_scanner_state.json",
    )
    cmd_scan(args)
    return {"action": "scan", "date": target}


@register(phase="strategies", description="India news sentiment scan")
def india_news_scan(
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run india_news RSS scan and FinBERT scoring."""
    import argparse
    from collectors.news.__main__ import cmd_scan
    from collectors.news.strategy import (
        DEFAULT_CONFIDENCE_THRESHOLD,
        DEFAULT_MAX_POSITIONS,
        DEFAULT_SCORE_THRESHOLD,
    )

    if dry_run:
        return {"action": "scan", "dry_run": True}

    args = argparse.Namespace(
        max_age=60,
        confidence=DEFAULT_CONFIDENCE_THRESHOLD,
        score=DEFAULT_SCORE_THRESHOLD,
        max_positions=DEFAULT_MAX_POSITIONS,
        verbose=False,
        state_file="data/india_news_state.json",
    )
    cmd_scan(args)
    return {"action": "scan"}


@register(phase="strategies", description="Microstructure tick data collector")
def india_microstructure_collect(
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run microstructure tick data collection (requires Kite auth)."""
    if dry_run:
        return {"action": "collect", "dry_run": True}

    from collectors.auth import headless_login
    from _archive.india_micro_legacy.collector import run_collector

    kite = headless_login()
    run_collector(kite, interval_seconds=180)
    return {"action": "collect"}
