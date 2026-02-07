"""Replay Engine — deterministic re-execution for parity verification.

The replay engine:
  1. Reads the reference WAL (original run's event log)
  2. Re-runs the orchestrator on the same dates with same config
  3. Captures all emitted events into a fresh event log
  4. Compares reference vs replay event streams
  5. Produces a parity report

The key invariant: same code + same data + same config = same decisions.

Usage::

    engine = ReplayEngine(
        store=store,
        registry=registry,
        ref_events_dir=Path("data/events"),
    )
    result = engine.replay_range("2025-08-06", "2025-12-31")
    assert result.comparison.identical
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from core.market.store import MarketDataStore
from core.strategy.registry import StrategyRegistry
from core.allocator.meta import MetaAllocator
from core.risk.manager import RiskManager
from core.events.envelope import EventEnvelope

from engine.orchestrator import Orchestrator
from engine.live.event_log import EventLogWriter, read_event_log
from engine.replay.reader import WalReader
from engine.replay.comparator import ComparisonResult, compare_streams

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Complete result of a replay run."""

    # Reference events (from original run)
    ref_events: list[EventEnvelope] = field(default_factory=list)

    # Replay events (from re-execution)
    replay_events: list[EventEnvelope] = field(default_factory=list)

    # Comparison result
    comparison: ComparisonResult = field(default_factory=ComparisonResult)

    # Replay metadata
    dates_replayed: list[str] = field(default_factory=list)
    ref_event_count: int = 0
    replay_event_count: int = 0

    # Extracted decision traces
    ref_signals: list[dict] = field(default_factory=list)
    replay_signals: list[dict] = field(default_factory=list)
    ref_positions: list[dict] = field(default_factory=list)
    replay_positions: list[dict] = field(default_factory=list)

    # E2E determinism: final state and equity curve
    final_state: object = None       # PortfolioState from the replay run
    equity_curve: list[dict] = field(default_factory=list)  # equity_history snapshot

    def to_dict(self) -> dict:
        return {
            "identical": self.comparison.identical,
            "dates_replayed": self.dates_replayed,
            "ref_event_count": self.ref_event_count,
            "replay_event_count": self.replay_event_count,
            "comparison": self.comparison.to_dict(),
        }


class ReplayEngine:
    """Deterministic replay engine for parity verification.

    Parameters
    ----------
    store : MarketDataStore
        Same data store used in the original run.
    registry : StrategyRegistry
        Strategy registry (same strategies as original run).
    allocator : MetaAllocator, optional
        Same allocator config as original run.
    risk_manager : RiskManager, optional
        Same risk config as original run.
    ref_events_dir : Path
        Directory containing reference event logs (data/events/).
    verify_hashes : bool
        If True, verify hash chain on reference WAL.
    """

    def __init__(
        self,
        store: MarketDataStore,
        registry: StrategyRegistry,
        allocator: MetaAllocator | None = None,
        risk_manager: RiskManager | None = None,
        ref_events_dir: Path | str = Path("data/events"),
        verify_hashes: bool = False,
    ):
        self._store = store
        self._registry = registry
        self._allocator = allocator or MetaAllocator()
        self._risk_manager = risk_manager or RiskManager()
        self._ref_dir = Path(ref_events_dir)
        self._verify_hashes = verify_hashes

    def replay_date(self, day: str) -> ReplayResult:
        """Replay a single date and compare against reference.

        Parameters
        ----------
        day : str
            Date in YYYY-MM-DD format.

        Returns
        -------
        ReplayResult
            Comparison result with full event traces.
        """
        return self.replay_range(day, day)

    def replay_range(self, start: str, end: str) -> ReplayResult:
        """Replay a date range and compare against reference.

        Parameters
        ----------
        start : str
            Start date (YYYY-MM-DD).
        end : str
            End date (YYYY-MM-DD).

        Returns
        -------
        ReplayResult
            Comparison result with full event traces.
        """
        result = ReplayResult()

        # 1. Read reference events
        reader = WalReader(
            base_dir=self._ref_dir,
            verify_hashes=self._verify_hashes,
        )
        result.ref_events = reader.read_range(start, end)
        result.ref_event_count = len(result.ref_events)
        logger.info(
            "Read %d reference events for %s to %s",
            result.ref_event_count, start, end,
        )

        # 2. Re-run orchestrator with fresh state into temp event log
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_events_dir = Path(tmpdir) / "events"
            replay_state_file = Path(tmpdir) / "state.json"

            replay_log = EventLogWriter(
                base_dir=replay_events_dir,
                run_id=f"replay-{start}-{end}",
                fsync_policy="none",
            )

            orchestrator = Orchestrator(
                store=self._store,
                registry=self._registry,
                allocator=self._allocator,
                risk_manager=self._risk_manager,
                state_file=replay_state_file,
                event_log=replay_log,
            )

            # Run through date range
            orchestrator.start_session()
            d = date.fromisoformat(start)
            end_d = date.fromisoformat(end)
            while d <= end_d:
                try:
                    orchestrator.run_day(d)
                    result.dates_replayed.append(d.isoformat())
                except Exception as e:
                    logger.error("Replay failed on %s: %s", d.isoformat(), e)
                d += timedelta(days=1)
            orchestrator.finalize_session()

            # Read replayed events
            replay_reader = WalReader(base_dir=replay_events_dir)
            result.replay_events = replay_reader.read_all()
            result.replay_event_count = len(result.replay_events)

            # Capture final state and equity curve before temp dir cleanup
            result.final_state = orchestrator.state
            result.equity_curve = list(orchestrator.state.equity_history)

        logger.info(
            "Replay produced %d events for %d dates",
            result.replay_event_count, len(result.dates_replayed),
        )

        # 3. Extract decision traces
        result.ref_signals = self._extract_signals(result.ref_events)
        result.replay_signals = self._extract_signals(result.replay_events)
        result.ref_positions = self._extract_snapshots(result.ref_events)
        result.replay_positions = self._extract_snapshots(result.replay_events)

        # 4. Compare
        result.comparison = compare_streams(
            result.ref_events,
            result.replay_events,
        )

        logger.info(
            "Replay parity: %s (%d diffs)",
            "PASS" if result.comparison.identical else "FAIL",
            len(result.comparison.diffs),
        )

        return result

    def replay_n_times(
        self,
        start: str,
        end: str,
        n: int = 3,
    ) -> list[ReplayResult]:
        """Run replay N times and verify all produce identical results.

        This is the core parity test: N independent replays should
        produce bit-identical event streams.

        Parameters
        ----------
        start : str
            Start date.
        end : str
            End date.
        n : int
            Number of replay runs (default 3).

        Returns
        -------
        list[ReplayResult]
            Results for each run.
        """
        results: list[ReplayResult] = []
        for i in range(n):
            logger.info("Replay run %d/%d", i + 1, n)
            result = self._run_clean(start, end, run_label=f"run{i+1}")
            results.append(result)

        # Compare all runs pairwise against the first
        if len(results) >= 2:
            baseline = results[0]
            for i in range(1, len(results)):
                comparison = compare_streams(
                    baseline.replay_events,
                    results[i].replay_events,
                )
                results[i].comparison = comparison
                if not comparison.identical:
                    logger.error(
                        "Run %d differs from run 1: %d diffs",
                        i + 1, len(comparison.diffs),
                    )

        return results

    def _run_clean(
        self,
        start: str,
        end: str,
        run_label: str = "replay",
    ) -> ReplayResult:
        """Execute a clean replay run (fresh state, no reference comparison)."""
        result = ReplayResult()

        with tempfile.TemporaryDirectory() as tmpdir:
            events_dir = Path(tmpdir) / "events"
            state_file = Path(tmpdir) / "state.json"

            log = EventLogWriter(
                base_dir=events_dir,
                run_id=f"{run_label}-{start}-{end}",
                fsync_policy="none",
            )

            orchestrator = Orchestrator(
                store=self._store,
                registry=self._registry,
                allocator=self._allocator,
                risk_manager=self._risk_manager,
                state_file=state_file,
                event_log=log,
            )

            orchestrator.start_session()
            d = date.fromisoformat(start)
            end_d = date.fromisoformat(end)
            while d <= end_d:
                try:
                    orchestrator.run_day(d)
                    result.dates_replayed.append(d.isoformat())
                except Exception as e:
                    logger.error("%s failed on %s: %s", run_label, d.isoformat(), e)
                d += timedelta(days=1)
            orchestrator.finalize_session()

            reader = WalReader(base_dir=events_dir)
            result.replay_events = reader.read_all()
            result.replay_event_count = len(result.replay_events)

            # Capture final state and equity curve before temp dir cleanup
            result.final_state = orchestrator.state
            result.equity_curve = list(orchestrator.state.equity_history)

        result.replay_signals = self._extract_signals(result.replay_events)
        result.replay_positions = self._extract_snapshots(result.replay_events)

        return result

    # ------------------------------------------------------------------
    # Event extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_signals(events: list[EventEnvelope]) -> list[dict]:
        """Extract signal events as dicts for easy comparison."""
        return [
            {
                "seq": e.seq,
                "strategy_id": e.strategy_id,
                "symbol": e.symbol,
                "direction": e.payload.get("direction"),
                "conviction": e.payload.get("conviction"),
                "instrument_type": e.payload.get("instrument_type"),
                "regime": e.payload.get("regime"),
            }
            for e in events
            if e.event_type == "signal"
        ]

    @staticmethod
    def _extract_snapshots(events: list[EventEnvelope]) -> list[dict]:
        """Extract snapshot events as dicts."""
        return [
            {
                "seq": e.seq,
                "equity": e.payload.get("equity"),
                "peak_equity": e.payload.get("peak_equity"),
                "portfolio_dd": e.payload.get("portfolio_dd"),
                "position_count": e.payload.get("position_count"),
                "total_exposure": e.payload.get("total_exposure"),
            }
            for e in events
            if e.event_type == "snapshot"
        ]

    # ------------------------------------------------------------------
    # Artifact export
    # ------------------------------------------------------------------

    def save_artifacts(
        self,
        result: ReplayResult,
        output_dir: Path | str,
    ) -> None:
        """Save replay artifacts to disk.

        Creates:
          - replay_signals.jsonl
          - replay_positions.jsonl
          - replay_diff_report.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Signals
        with open(output_dir / "replay_signals.jsonl", "w") as f:
            for sig in result.replay_signals:
                f.write(json.dumps(sig, sort_keys=True, default=str) + "\n")

        # Positions (snapshots)
        with open(output_dir / "replay_positions.jsonl", "w") as f:
            for snap in result.replay_positions:
                f.write(json.dumps(snap, sort_keys=True, default=str) + "\n")

        # Diff report
        with open(output_dir / "replay_diff_report.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2, sort_keys=True, default=str)

        logger.info("Replay artifacts saved to %s", output_dir)
