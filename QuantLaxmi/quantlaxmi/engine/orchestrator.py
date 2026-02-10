"""Orchestrator — Main trading loop.

For each trading day:
  1. Each strategy.scan(date, store) → list[Signal]
  2. MetaAllocator.allocate(signals, regime) → list[TargetPosition]
  3. RiskManager.check(targets, state) → approved targets
  4. Execute: open/close positions based on approved targets
  5. Update and persist PortfolioState

Phase 2: Events emitted at every decision boundary for WAL persistence.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

from quantlaxmi.data._paths import EVENTS_DIR, SESSIONS_DIR
from quantlaxmi.core.allocator.meta import MetaAllocator, TargetPosition
from quantlaxmi.core.allocator.regime import VIXRegime, detect_regime
from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.core.events.envelope import EventEnvelope
from quantlaxmi.core.events.payloads import (
    SignalPayload,
    GateDecisionPayload,
    OrderPayload,
    FillPayload,
    RiskAlertPayload,
    SnapshotPayload,
)
from quantlaxmi.core.events.types import EventType
from quantlaxmi.core.risk.limits import RiskLimits
from quantlaxmi.core.risk.manager import PortfolioState as RiskPortfolioState, RiskManager
from quantlaxmi.strategies.protocol import Signal, StrategyProtocol
from quantlaxmi.strategies.registry import StrategyRegistry

from quantlaxmi.engine.live.event_log import EventLogWriter
from quantlaxmi.engine.live.journals import ExecutionJournal, SignalJournal
from quantlaxmi.engine.live.session_manifest import SessionManifest
from quantlaxmi.engine.state import PortfolioState, Position, DEFAULT_STATE_FILE

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main trading orchestrator.

    Coordinates strategies, allocation, risk management, and execution.
    Emits events to the EventLog at every decision boundary.
    """

    def __init__(
        self,
        store: MarketDataStore,
        registry: StrategyRegistry | None = None,
        allocator: MetaAllocator | None = None,
        risk_manager: RiskManager | None = None,
        state_file: Path = DEFAULT_STATE_FILE,
        event_log: EventLogWriter | None = None,
    ):
        self.store = store
        self.registry = registry or StrategyRegistry()
        self.allocator = allocator or MetaAllocator()
        self.risk_manager = risk_manager or RiskManager()
        self.state = PortfolioState.load(state_file)
        self._state_file = state_file

        # Event infrastructure (Phase 2)
        self._run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        self._event_log = event_log or EventLogWriter(
            base_dir=EVENTS_DIR,
            run_id=self._run_id,
        )
        self._signal_journal = SignalJournal(self._event_log)
        self._exec_journal = ExecutionJournal(self._event_log)
        self._session = SessionManifest(
            base_dir=SESSIONS_DIR,
            run_id=self._run_id,
        )

        # Track cumulative stats for session manifest
        self._total_signals = 0
        self._total_trades = 0
        self._total_blocks = 0

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def event_log(self) -> EventLogWriter:
        return self._event_log

    def start_session(self) -> None:
        """Initialize the session manifest. Call before first run_day()."""
        strategy_ids = [s.strategy_id for s in self.registry.all()]
        self._session.start(
            strategies=strategy_ids,
            risk_limits=asdict(self.risk_manager.limits),
            data_sources=["DuckDB"],
        )

    def finalize_session(self, error: str = "") -> None:
        """Finalize session manifest and close event log."""
        self._emit_snapshot()
        self._session.finalize(
            total_signals=self._total_signals,
            total_trades=self._total_trades,
            total_blocks=self._total_blocks,
            final_equity=self.state.equity,
            peak_equity=self.state.peak_equity,
            max_dd=self.state.peak_equity - self.state.equity if self.state.peak_equity > 0 else 0,
            total_pnl_pct=self.state.total_return_pct(),
            event_count=self._event_log.event_count,
            error=error,
        )
        self._event_log.close()

    def run_day(self, d: date) -> dict:
        """Execute the full trading loop for one day.

        Returns a summary dict with signals, targets, risk checks, and actions.
        """
        logger.info("=" * 70)
        logger.info("QuantLaxmi scan: %s", d.isoformat())
        logger.info("=" * 70)

        summary: dict = {
            "date": d.isoformat(),
            "signals": [],
            "targets": [],
            "risk_checks": [],
            "actions": [],
            "regime": None,
        }

        # 1. Detect VIX regime
        regime = detect_regime(self.store, d)
        summary["regime"] = {
            "vix": regime.vix,
            "type": regime.regime.value,
        }
        self.state.last_vix = regime.vix
        self.state.last_regime = regime.regime.value

        logger.info("VIX regime: %s (VIX=%.1f)", regime.regime.value, regime.vix)

        # 2. Collect signals from all strategies
        all_signals: list[Signal] = []
        strategies = self.registry.all()

        if not strategies:
            logger.warning("No strategies registered!")
            self._save()
            return summary

        for strategy in strategies:
            try:
                signals = strategy.scan(d, self.store)
                all_signals.extend(signals)
                for s in signals:
                    summary["signals"].append({
                        "strategy": s.strategy_id,
                        "symbol": s.symbol,
                        "direction": s.direction,
                        "conviction": s.conviction,
                        "instrument_type": s.instrument_type,
                    })
                    # Emit SignalEvent (pre-gate)
                    self._emit_signal(s, regime.regime.value)
            except Exception as e:
                logger.error(
                    "Strategy %s failed: %s", strategy.strategy_id, e, exc_info=True,
                )

        self._total_signals += len(all_signals)
        logger.info("Collected %d signals from %d strategies", len(all_signals), len(strategies))

        # 3. Allocate via MetaAllocator
        targets = self.allocator.allocate(all_signals, regime)
        for t in targets:
            summary["targets"].append({
                "strategy": t.strategy_id,
                "symbol": t.symbol,
                "direction": t.direction,
                "weight": t.weight,
            })

        logger.info("Allocator produced %d targets", len(targets))

        # 4. Risk check
        risk_state = self._build_risk_state()
        risk_results = self.risk_manager.check(targets, risk_state)

        for r in risk_results:
            summary["risk_checks"].append({
                "strategy": r.target.strategy_id,
                "symbol": r.target.symbol,
                "approved": r.approved,
                "gate": r.gate.value,
                "adjusted_weight": r.adjusted_weight,
                "reason": r.reason,
            })
            # Emit GateDecisionEvent
            self._emit_gate_decision(r, risk_state)

        approved = [r for r in risk_results if r.approved]
        blocked = [r for r in risk_results if r.blocked]
        self._total_blocks += len(blocked)
        logger.info(
            "Risk: %d approved, %d blocked",
            len(approved),
            len(blocked),
        )

        # Emit RiskAlertEvent for any blocks
        for r in blocked:
            self._emit_risk_alert(r)

        # 5. Execute approved targets
        for result in approved:
            action = self._execute_target(result.target, result.adjusted_weight, d)
            if action:
                summary["actions"].append(action)

        # 6. Emit portfolio snapshot
        self._emit_snapshot()

        # 7. Update equity history and state
        day_pnl = self.state.equity - self.state.equity_at_open
        drawdown = self.state.portfolio_dd
        self.state.equity_history.append({
            "date": d.isoformat(),
            "equity": self.state.equity,
            "drawdown": drawdown,
            "day_pnl": day_pnl,
        })
        self.state.equity_at_open = self.state.equity  # reset for next day

        self.state.last_scan_date = d.isoformat()
        self.state.last_scan_time = datetime.now(timezone.utc).isoformat()
        self.state.scan_count += 1
        self._save()

        # 8. Log summary
        self._log_summary(summary)

        return summary

    # ------------------------------------------------------------------
    # Event emission helpers
    # ------------------------------------------------------------------

    def _emit_signal(self, signal: Signal, regime: str) -> EventEnvelope:
        """Emit a SignalEvent (pre-gate, full context)."""
        payload = SignalPayload(
            direction=signal.direction,
            conviction=signal.conviction,
            instrument_type=signal.instrument_type,
            strike=signal.strike,
            expiry=signal.expiry,
            ttl_bars=signal.ttl_bars,
            regime=regime,
            components=signal.metadata,
        )
        return self._signal_journal.log_signal(
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            payload=payload,
        )

    def _emit_gate_decision(self, result, risk_state: RiskPortfolioState) -> EventEnvelope:
        """Emit a GateDecisionEvent (pass/fail with reasons + thresholds)."""
        payload = GateDecisionPayload(
            signal_seq=0,  # linked in replay, not at emit time
            gate=result.gate.value,
            approved=result.approved,
            adjusted_weight=result.adjusted_weight,
            reason=result.reason,
            vpin=risk_state.vpin,
            portfolio_dd=risk_state.portfolio_dd,
            strategy_dd=risk_state.strategy_dd(result.target.strategy_id),
            total_exposure=risk_state.total_exposure(),
        )
        return self._signal_journal.log_gate_decision(
            strategy_id=result.target.strategy_id,
            symbol=result.target.symbol,
            payload=payload,
        )

    def _emit_risk_alert(self, result) -> None:
        """Emit a RiskAlertEvent for a blocked position."""
        payload = RiskAlertPayload(
            alert_type=result.gate.value,
            new_state="blocked",
            threshold=0.0,
            current_value=0.0,
            detail=result.reason,
        )
        self._event_log.emit(
            event_type=EventType.RISK_ALERT.value,
            source="orchestrator",
            payload=payload.to_dict(),
            strategy_id=result.target.strategy_id,
            symbol=result.target.symbol,
        )

    def _emit_snapshot(self) -> None:
        """Emit a SnapshotEvent (portfolio + risk state)."""
        risk_state = self._build_risk_state()
        strat_dd = {}
        for sid in self.state.strategy_equity:
            strat_dd[sid] = risk_state.strategy_dd(sid)

        payload = SnapshotPayload(
            equity=self.state.equity,
            peak_equity=self.state.peak_equity,
            portfolio_dd=risk_state.portfolio_dd,
            total_exposure=risk_state.total_exposure(),
            vpin=risk_state.vpin,
            position_count=len(self.state.active_positions()),
            strategy_equity=dict(self.state.strategy_equity),
            strategy_dd=strat_dd,
            active_breakers=["circuit_breaker"] if self.risk_manager.circuit_breaker_active else [],
            regime=self.state.last_regime,
        )
        self._event_log.emit(
            event_type=EventType.SNAPSHOT.value,
            source="orchestrator",
            payload=payload.to_dict(),
        )

    def _emit_order(self, strategy_id: str, symbol: str, action: str, side: str) -> None:
        """Emit an OrderEvent."""
        payload = OrderPayload(
            order_id=str(uuid.uuid4())[:8],
            action=action,
            side=side,
            order_type="market",
        )
        self._exec_journal.log_order(
            strategy_id=strategy_id,
            symbol=symbol,
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_target(
        self,
        target: TargetPosition,
        adjusted_weight: float,
        d: date,
    ) -> dict | None:
        """Execute a single approved target position."""
        d_str = d.isoformat()

        if target.direction == "flat":
            # Close existing position
            pos = self.state.get_position(target.strategy_id, target.symbol)
            if pos is None:
                return None  # nothing to close

            # Emit order event
            self._emit_order(target.strategy_id, target.symbol, "submit", "sell")

            # Get real exit price from market data
            exit_price = self._get_spot(target.symbol, d)
            if exit_price <= 0:
                exit_price = pos.entry_price  # fallback if no market data

            trade = self.state.close_position(
                strategy_id=target.strategy_id,
                symbol=target.symbol,
                exit_date=d_str,
                exit_price=exit_price,
                exit_reason=target.metadata.get("exit_reason", "signal_flat"),
            )
            if trade:
                self._total_trades += 1
                return {
                    "action": "close",
                    "strategy": target.strategy_id,
                    "symbol": target.symbol,
                    "pnl_pct": trade.pnl_pct,
                    "reason": trade.exit_reason,
                }
            return None

        # Open or update position
        existing = self.state.get_position(target.strategy_id, target.symbol)
        if existing is not None:
            # Already in position — skip (no re-entry)
            return None

        if adjusted_weight <= 0:
            return None

        # Get spot price for entry
        spot = self._get_spot(target.symbol, d)

        # Emit order event
        self._emit_order(target.strategy_id, target.symbol, "submit", "buy")

        pos = Position(
            strategy_id=target.strategy_id,
            symbol=target.symbol,
            direction=target.direction,
            weight=adjusted_weight,
            instrument_type=target.instrument_type,
            entry_date=d_str,
            entry_price=spot,
            strike=target.strike,
            expiry=target.expiry,
            metadata=target.metadata,
        )
        self.state.open_position(pos)
        self._total_trades += 1

        return {
            "action": "open",
            "strategy": target.strategy_id,
            "symbol": target.symbol,
            "direction": target.direction,
            "weight": adjusted_weight,
            "entry_price": spot,
        }

    def _get_spot(self, symbol: str, d: date) -> float:
        """Get spot price for a symbol on a date."""
        d_str = d.isoformat()
        try:
            _idx_name = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank",
                         "FINNIFTY": "Nifty Financial Services",
                         "MIDCPNIFTY": "Nifty Midcap Select"}.get(
                symbol.upper(), f"Nifty {symbol}")
            df = self.store.sql(
                'SELECT "Closing Index Value" as close FROM nse_index_close '
                'WHERE date = ? AND "Index Name" = ? '
                "LIMIT 1",
                [d_str, _idx_name],
            )
            if not df.empty:
                return float(df["close"].iloc[0])
        except Exception as e:
            logger.debug("Index close lookup failed for %s: %s", symbol, e)

        # Fallback: try FnO close
        try:
            df = self.store.sql(
                "SELECT close FROM nfo_1min "
                "WHERE date = ? AND name = ? AND instrument_type = 'FUT' "
                "ORDER BY date DESC LIMIT 1",
                [d_str, symbol],
            )
            if not df.empty:
                return float(df["close"].iloc[0])
        except Exception as e:
            logger.debug("FnO close lookup failed for %s: %s", symbol, e)

        return 0.0

    def _build_risk_state(self) -> RiskPortfolioState:
        """Convert PortfolioState to RiskManager's PortfolioState."""
        positions = {}
        for key, pos in self.state.positions.items():
            positions[pos.symbol] = {
                "direction": pos.direction,
                "weight": pos.weight,
                "strategy_id": pos.strategy_id,
            }

        return RiskPortfolioState(
            equity=self.state.equity,
            peak_equity=self.state.peak_equity,
            positions=positions,
            strategy_equity=self.state.strategy_equity.copy(),
            strategy_peaks=self.state.strategy_peaks.copy(),
            vpin=self.state.last_vpin,
        )

    def _save(self) -> None:
        self.state.save(self._state_file)

    def _log_summary(self, summary: dict) -> None:
        n_signals = len(summary["signals"])
        n_actions = len(summary["actions"])
        opens = sum(1 for a in summary["actions"] if a["action"] == "open")
        closes = sum(1 for a in summary["actions"] if a["action"] == "close")
        n_pos = len(self.state.active_positions())

        logger.info("-" * 50)
        logger.info(
            "Summary: %d signals → %d actions (%d opens, %d closes)",
            n_signals, n_actions, opens, closes,
        )
        logger.info(
            "Portfolio: equity=%.4f (%.2f%%) positions=%d regime=%s",
            self.state.equity,
            self.state.total_return_pct(),
            n_pos,
            summary["regime"]["type"],
        )
        logger.info("-" * 50)
