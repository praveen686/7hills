"""DataQualityGate — pre-signal validation for production hardening.

Checks before every strategy scan:
  1. Minimum 5 strikes in option chain
  2. Minimum OI ≥ 100 on at least one strike
  3. Tick staleness < 300 seconds
  4. Index close present for the date

On failure, emits a MISSINGNESS event to the WAL and returns a gate
result instructing the orchestrator to skip the strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from core.events.payloads import MissingnessPayload
from core.events.types import EventType

logger = logging.getLogger(__name__)

# Thresholds
MIN_STRIKES = 5
MIN_OI = 100
MAX_TICK_STALENESS_S = 300.0


@dataclass(frozen=True)
class DQCheckResult:
    """Result of a single data-quality check."""

    passed: bool
    check_type: str
    detail: str
    severity: str = "block"        # "block" or "warning"
    chain_strike_count: int = 0
    min_oi_found: int = 0


@dataclass(frozen=True)
class DQGateResult:
    """Aggregate gate result from all checks."""

    passed: bool
    checks: list[DQCheckResult]

    @property
    def failures(self) -> list[DQCheckResult]:
        return [c for c in self.checks if not c.passed]


class DataQualityGate:
    """Pre-signal data quality gate.

    Parameters
    ----------
    event_log : EventLogWriter or None
        If provided, MISSINGNESS events are emitted on failure.
    min_strikes : int
        Minimum number of distinct strikes required.
    min_oi : int
        Minimum OI on at least one strike.
    max_staleness_s : float
        Maximum tick staleness in seconds.
    """

    def __init__(
        self,
        event_log=None,
        min_strikes: int = MIN_STRIKES,
        min_oi: int = MIN_OI,
        max_staleness_s: float = MAX_TICK_STALENESS_S,
    ):
        self._event_log = event_log
        self._min_strikes = min_strikes
        self._min_oi = min_oi
        self._max_staleness_s = max_staleness_s

    def check(
        self,
        symbol: str,
        chain_df=None,
        index_close: float | None = None,
        last_tick_ts: datetime | None = None,
        now: datetime | None = None,
    ) -> DQGateResult:
        """Run all quality checks for a symbol before strategy scan.

        Parameters
        ----------
        symbol : str
            Instrument symbol (e.g., "NIFTY").
        chain_df : DataFrame or None
            Option chain snapshot. Must have 'strike' and 'oi' columns.
        index_close : float or None
            Index close price. None = missing.
        last_tick_ts : datetime or None
            Timestamp of the last received tick.
        now : datetime or None
            Current time (for staleness). Defaults to utcnow.

        Returns
        -------
        DQGateResult with passed=True if all checks pass.
        """
        checks: list[DQCheckResult] = []

        # 1. Index close present
        checks.append(self._check_index_close(index_close))

        # 2. Chain checks (only if chain provided)
        if chain_df is not None:
            checks.append(self._check_min_strikes(chain_df))
            checks.append(self._check_min_oi(chain_df))
        else:
            checks.append(DQCheckResult(
                passed=False,
                check_type="min_strikes",
                detail="No option chain data provided",
                severity="block",
                chain_strike_count=0,
            ))

        # 3. Tick staleness
        if last_tick_ts is not None:
            checks.append(self._check_tick_staleness(last_tick_ts, now))

        passed = all(c.passed for c in checks if c.severity == "block")
        result = DQGateResult(passed=passed, checks=checks)

        # Emit MISSINGNESS events for failures
        if not passed and self._event_log is not None:
            for c in result.failures:
                payload = MissingnessPayload(
                    check_type=c.check_type,
                    symbol=symbol,
                    detail=c.detail,
                    severity=c.severity,
                    chain_strike_count=c.chain_strike_count,
                    min_oi_found=c.min_oi_found,
                )
                self._event_log.emit(
                    event_type=EventType.MISSINGNESS.value,
                    source="data_quality_gate",
                    payload=payload.to_dict(),
                    strategy_id="",
                    symbol=symbol,
                )

        return result

    def _check_index_close(self, index_close: float | None) -> DQCheckResult:
        if index_close is not None and index_close > 0:
            return DQCheckResult(
                passed=True,
                check_type="index_close",
                detail=f"Index close present: {index_close:.2f}",
            )
        return DQCheckResult(
            passed=False,
            check_type="index_close",
            detail="Index close missing or zero",
            severity="block",
        )

    def _check_min_strikes(self, chain_df) -> DQCheckResult:
        n_strikes = chain_df["strike"].nunique() if "strike" in chain_df.columns else 0
        passed = n_strikes >= self._min_strikes
        return DQCheckResult(
            passed=passed,
            check_type="min_strikes",
            detail=f"{n_strikes} strikes (min {self._min_strikes})",
            severity="block",
            chain_strike_count=n_strikes,
        )

    def _check_min_oi(self, chain_df) -> DQCheckResult:
        if "oi" not in chain_df.columns or chain_df.empty:
            return DQCheckResult(
                passed=False,
                check_type="min_oi",
                detail="No OI data in chain",
                severity="block",
                min_oi_found=0,
            )
        import pandas as pd
        max_oi_raw = chain_df["oi"].max()
        if pd.isna(max_oi_raw):
            return DQCheckResult(
                passed=False,
                check_type="min_oi",
                detail="All OI values are NaN",
                severity="block",
                min_oi_found=0,
            )
        max_oi = int(max_oi_raw)
        passed = max_oi >= self._min_oi
        return DQCheckResult(
            passed=passed,
            check_type="min_oi",
            detail=f"Max OI {max_oi} (min {self._min_oi})",
            severity="block",
            min_oi_found=max_oi,
        )

    def _check_tick_staleness(
        self, last_tick_ts: datetime, now: datetime | None = None,
    ) -> DQCheckResult:
        if now is None:
            now = datetime.now(timezone.utc)
        staleness = (now - last_tick_ts).total_seconds()
        passed = staleness <= self._max_staleness_s
        return DQCheckResult(
            passed=passed,
            check_type="tick_staleness",
            detail=f"Staleness {staleness:.1f}s (max {self._max_staleness_s}s)",
            severity="block" if staleness > self._max_staleness_s else "warning",
        )
