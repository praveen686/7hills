"""ARS Surface Service — Activation Readiness Surface.

Builds a strategy × date surface of signal conviction levels,
tracking which signals were executed vs blocked by risk gates.

Single WAL read per day, grouping by strategy for efficiency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import date as date_type, timedelta
from pathlib import Path

from quantlaxmi.core.events.types import EventType
from quantlaxmi.engine.replay.reader import WalReader

logger = logging.getLogger(__name__)


@dataclass
class ARSPoint:
    """Single point on the ARS surface: one strategy on one day."""

    date: str
    strategy_id: str
    max_conviction: float = 0.0
    signal_count: int = 0
    executed_count: int = 0
    blocked_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ARSSurfaceService:
    """Build strategy activation readiness surfaces from WAL data."""

    def __init__(self, base_dir: Path | str):
        self._base_dir = Path(base_dir)

    def _reader(self) -> WalReader:
        return WalReader(base_dir=self._base_dir)

    # ------------------------------------------------------------------
    # Surface for a single strategy
    # ------------------------------------------------------------------

    def surface_for_strategy(
        self,
        strategy_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[ARSPoint]:
        """Build ARS surface for a single strategy across available dates."""
        reader = self._reader()
        dates = reader.available_dates()
        if not dates:
            return []

        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        points = []
        for day in dates:
            point = self._analyze_day_strategy(reader, day, strategy_id)
            points.append(point)
        return points

    # ------------------------------------------------------------------
    # Surface for all strategies
    # ------------------------------------------------------------------

    def surface_all_strategies(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, list[ARSPoint]]:
        """Build ARS surface for all strategies."""
        reader = self._reader()
        dates = reader.available_dates()
        if not dates:
            return {}

        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        # Collect all strategy IDs across all dates
        result: dict[str, list[ARSPoint]] = {}
        for day in dates:
            day_points = self._analyze_day_all(reader, day)
            for sid, point in day_points.items():
                result.setdefault(sid, []).append(point)

        return result

    # ------------------------------------------------------------------
    # Heatmap matrix
    # ------------------------------------------------------------------

    def heatmap_matrix(
        self,
        surface: dict[str, list[ARSPoint]],
    ) -> dict:
        """Convert surface dict to a heatmap matrix.

        Returns:
            {
                dates: list[str],
                strategies: list[str],
                matrix: list[list[float]],        # [strategy_idx][date_idx]
                status_matrix: list[list[str]],   # "executed"|"blocked"|"none"
            }
        """
        if not surface:
            return {
                "dates": [],
                "strategies": [],
                "matrix": [],
                "status_matrix": [],
            }

        # Collect all dates and strategies
        all_dates: set[str] = set()
        for points in surface.values():
            for p in points:
                all_dates.add(p.date)
        dates = sorted(all_dates)
        strategies = sorted(surface.keys())

        # Build lookup
        lookup: dict[tuple[str, str], ARSPoint] = {}
        for sid, points in surface.items():
            for p in points:
                lookup[(sid, p.date)] = p

        # Build matrices
        matrix: list[list[float]] = []
        status_matrix: list[list[str]] = []

        for sid in strategies:
            row_conv: list[float] = []
            row_status: list[str] = []
            for d in dates:
                point = lookup.get((sid, d))
                if point is None or point.signal_count == 0:
                    row_conv.append(0.0)
                    row_status.append("none")
                else:
                    row_conv.append(point.max_conviction)
                    if point.executed_count > 0:
                        row_status.append("executed")
                    elif point.blocked_count > 0:
                        row_status.append("blocked")
                    else:
                        row_status.append("none")
            matrix.append(row_conv)
            status_matrix.append(row_status)

        return {
            "dates": dates,
            "strategies": strategies,
            "matrix": matrix,
            "status_matrix": status_matrix,
        }

    # ------------------------------------------------------------------
    # Internal: analyze a single day
    # ------------------------------------------------------------------

    def _analyze_day_strategy(
        self, reader: WalReader, day: str, strategy_id: str,
    ) -> ARSPoint:
        """Analyze one day for one strategy."""
        events = reader.read_date(day)
        point = ARSPoint(date=day, strategy_id=strategy_id)

        signals = []
        gates = []

        for e in events:
            if e.strategy_id != strategy_id:
                continue
            if e.event_type == EventType.SIGNAL.value:
                signals.append(e)
            elif e.event_type == EventType.GATE_DECISION.value:
                gates.append(e)

        point.signal_count = len(signals)

        if signals:
            convictions = [
                s.payload.get("conviction", 0.0) for s in signals
            ]
            point.max_conviction = max(convictions) if convictions else 0.0

        # Classify signals as executed/blocked
        for sig in signals:
            gate_for_signal = None
            for g in gates:
                if g.seq > sig.seq and g.symbol == sig.symbol:
                    gate_for_signal = g
                    break

            if gate_for_signal is None:
                # No gate decision → assume executed (no gate blocking)
                point.executed_count += 1
            elif gate_for_signal.payload.get("approved", True):
                point.executed_count += 1
            else:
                point.blocked_count += 1

        return point

    def _analyze_day_all(
        self, reader: WalReader, day: str,
    ) -> dict[str, ARSPoint]:
        """Analyze one day for all strategies (single WAL read)."""
        events = reader.read_date(day)

        # Group by strategy
        strat_signals: dict[str, list] = {}
        strat_gates: dict[str, list] = {}

        for e in events:
            sid = e.strategy_id
            if not sid:
                continue
            if e.event_type == EventType.SIGNAL.value:
                strat_signals.setdefault(sid, []).append(e)
            elif e.event_type == EventType.GATE_DECISION.value:
                strat_gates.setdefault(sid, []).append(e)

        # Build points
        all_sids = set(strat_signals.keys()) | set(strat_gates.keys())
        result: dict[str, ARSPoint] = {}

        for sid in all_sids:
            signals = strat_signals.get(sid, [])
            gates = strat_gates.get(sid, [])

            point = ARSPoint(date=day, strategy_id=sid)
            point.signal_count = len(signals)

            if signals:
                convictions = [s.payload.get("conviction", 0.0) for s in signals]
                point.max_conviction = max(convictions) if convictions else 0.0

            for sig in signals:
                gate_for_signal = None
                for g in gates:
                    if g.seq > sig.seq and g.symbol == sig.symbol:
                        gate_for_signal = g
                        break

                if gate_for_signal is None:
                    point.executed_count += 1
                elif gate_for_signal.payload.get("approved", True):
                    point.executed_count += 1
                else:
                    point.blocked_count += 1

            result[sid] = point

        return result
