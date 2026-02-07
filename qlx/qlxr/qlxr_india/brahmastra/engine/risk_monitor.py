"""Real-time risk monitor — mark-to-market P&L, Greeks, and circuit breakers.

Runs as a background async task that:
  - Marks positions to market on every tick (P&L update).
  - Recomputes portfolio Greeks every 30 seconds.
  - Checks circuit breaker conditions on every P&L update.
  - Publishes "risk_alert" events when limits are breached.

This is the live complement to the batch-mode RiskManager in
``qlx.risk.manager``.  The batch manager gates new entries; this monitor
watches for deterioration in real-time and triggers emergency actions.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from qlx.risk.limits import RiskLimits
from qlx.risk.greeks import PortfolioGreeks, compute_portfolio_greeks

from brahmastra.engine.event_bus import EventBus, EventType
from brahmastra.state import BrahmastraState

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Greeks recompute interval
_GREEKS_INTERVAL_SEC = 30.0

# P&L alert thresholds (these are *intraday* drawdown limits)
_INTRADAY_DD_WARN = 0.01     # 1% intraday DD → warning
_INTRADAY_DD_HALT = 0.02     # 2% intraday DD → circuit breaker


@dataclass
class RiskSnapshot:
    """Point-in-time risk summary."""

    timestamp: float                    # monotonic
    mark_to_market_pnl: float          # total unrealized P&L (INR fraction)
    intraday_pnl: float                # P&L since session start
    intraday_dd: float                 # max intraday drawdown
    total_exposure: float              # gross portfolio weight
    greeks: PortfolioGreeks | None     # latest Greeks (None if not computed yet)
    vpin_max: float                    # max VPIN across instruments
    circuit_breaker_active: bool       # True if tripped


@dataclass
class _PositionMtM:
    """Per-position mark-to-market state."""

    strategy_id: str
    symbol: str
    direction: str         # "long" or "short"
    weight: float
    entry_price: float
    instrument_type: str
    strike: float
    expiry: str
    current_price: float
    unrealized_pnl: float   # (current - entry) / entry, signed

    @property
    def pnl_contribution(self) -> float:
        """P&L contribution to portfolio (weight * pnl)."""
        return self.weight * self.unrealized_pnl


class RiskMonitor:
    """Real-time risk monitoring engine.

    Parameters
    ----------
    event_bus : EventBus
        For subscribing to ticks and publishing risk alerts.
    state : BrahmastraState
        Mutable portfolio state -- positions are read for MtM.
    limits : RiskLimits
        Risk thresholds (drawdown, VPIN, concentration, etc.).
    greeks_interval : float
        Seconds between Greeks recomputation (default 30).
    """

    def __init__(
        self,
        event_bus: EventBus,
        state: BrahmastraState,
        limits: RiskLimits | None = None,
        greeks_interval: float = _GREEKS_INTERVAL_SEC,
    ) -> None:
        self.event_bus = event_bus
        self.state = state
        self.limits = limits or RiskLimits()
        self._greeks_interval = greeks_interval

        # Subscribe to tick events
        self._tick_queue = event_bus.subscribe(EventType.TICK)

        # Internal state
        self._spot_prices: dict[str, float] = {}   # symbol -> latest LTP
        self._token_to_symbol: dict[int, str] = {}  # token -> symbol mapping
        self._latest_greeks: PortfolioGreeks | None = None
        self._last_greeks_time: float = 0.0

        # Session tracking (intraday)
        self._session_start_equity: float = state.equity
        self._session_peak_equity: float = state.equity
        self._intraday_dd: float = 0.0
        self._circuit_breaker_active: bool = False

        # VPIN tracking
        self._latest_vpin: dict[int, float] = {}

        # Stats
        self._mtm_updates = 0
        self._greeks_updates = 0
        self._alerts_published = 0

        # Control
        self._running = False
        self._tick_task: asyncio.Task | None = None
        self._greeks_task: asyncio.Task | None = None

    def register_token_mapping(self, instrument_token: int, symbol: str) -> None:
        """Map a Kite instrument token to a trading symbol.

        Must be called for each subscribed token before the monitor can
        correctly update position P&L from tick events.
        """
        self._token_to_symbol[instrument_token] = symbol

    async def start(self) -> None:
        """Start the risk monitoring loops."""
        self._running = True
        self._session_start_equity = self.state.equity
        self._session_peak_equity = self.state.equity
        self._intraday_dd = 0.0
        self._circuit_breaker_active = False

        self._tick_task = asyncio.create_task(
            self._tick_loop(), name="risk_monitor_tick",
        )
        self._greeks_task = asyncio.create_task(
            self._greeks_loop(), name="risk_monitor_greeks",
        )

        logger.info(
            "RiskMonitor started: %d positions, equity=%.4f, limits=DD%.0f%%/VPIN%.2f",
            len(self.state.active_positions()),
            self.state.equity,
            self.limits.max_portfolio_dd * 100,
            self.limits.vpin_block_threshold,
        )

    async def stop(self) -> None:
        """Stop all monitoring loops."""
        self._running = False
        for task in (self._tick_task, self._greeks_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info(
            "RiskMonitor stopped: %d MtM updates, %d Greeks updates, %d alerts",
            self._mtm_updates,
            self._greeks_updates,
            self._alerts_published,
        )

    # ------------------------------------------------------------------
    # Tick loop: mark-to-market on every tick
    # ------------------------------------------------------------------

    async def _tick_loop(self) -> None:
        """Main tick processing loop."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._tick_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            tick_data = event.data

            # Update spot price
            token = tick_data.get("instrument_token", 0)
            ltp = tick_data.get("ltp", 0.0)
            symbol = self._token_to_symbol.get(token)

            if symbol and ltp > 0:
                self._spot_prices[symbol] = ltp

            # Track VPIN
            vpin = tick_data.get("vpin")
            if vpin is not None and not math.isnan(vpin):
                self._latest_vpin[token] = vpin

            # Mark-to-market all positions
            await self._update_mtm()

            # Circuit breaker checks
            await self._check_circuit_breakers()

    async def _update_mtm(self) -> None:
        """Recompute unrealized P&L for all positions."""
        if not self.state.positions:
            return

        total_unrealized = 0.0

        for key, pos in self.state.positions.items():
            current = self._spot_prices.get(pos.symbol, 0.0)
            if current <= 0:
                continue

            pos.current_price = current

            if pos.direction == "long":
                pnl = (current - pos.entry_price) / pos.entry_price
            else:
                pnl = (pos.entry_price - current) / pos.entry_price

            pos.unrealized_pnl = pnl
            total_unrealized += pos.weight * pnl

        # Update equity estimate (mark-to-market)
        current_equity = self._session_start_equity + total_unrealized

        if current_equity > self._session_peak_equity:
            self._session_peak_equity = current_equity

        # Intraday drawdown from peak
        if self._session_peak_equity > 0:
            self._intraday_dd = (
                (self._session_peak_equity - current_equity) / self._session_peak_equity
            )

        self._mtm_updates += 1

    async def _check_circuit_breakers(self) -> None:
        """Check all circuit breaker conditions and publish alerts."""

        # 1. Intraday drawdown check
        if self._intraday_dd >= _INTRADAY_DD_HALT and not self._circuit_breaker_active:
            self._circuit_breaker_active = True
            self.state.circuit_breaker_active = True
            await self._publish_alert(
                alert_type="circuit_breaker",
                severity="critical",
                message=(
                    f"CIRCUIT BREAKER: intraday DD {self._intraday_dd:.2%} "
                    f">= {_INTRADAY_DD_HALT:.2%} halt threshold"
                ),
                data={
                    "intraday_dd": self._intraday_dd,
                    "threshold": _INTRADAY_DD_HALT,
                },
            )
            logger.critical(
                "CIRCUIT BREAKER TRIPPED: intraday DD %.2f%%",
                self._intraday_dd * 100,
            )

        elif self._intraday_dd >= _INTRADAY_DD_WARN:
            await self._publish_alert(
                alert_type="dd_warning",
                severity="warning",
                message=(
                    f"Drawdown warning: intraday DD {self._intraday_dd:.2%} "
                    f">= {_INTRADAY_DD_WARN:.2%}"
                ),
                data={
                    "intraday_dd": self._intraday_dd,
                    "threshold": _INTRADAY_DD_WARN,
                },
            )

        # 2. Portfolio-level drawdown (persistent, not just intraday)
        if self.state.portfolio_dd > self.limits.max_portfolio_dd:
            if not self._circuit_breaker_active:
                self._circuit_breaker_active = True
                self.state.circuit_breaker_active = True
                await self._publish_alert(
                    alert_type="circuit_breaker",
                    severity="critical",
                    message=(
                        f"CIRCUIT BREAKER: portfolio DD {self.state.portfolio_dd:.2%} "
                        f"> {self.limits.max_portfolio_dd:.2%}"
                    ),
                    data={
                        "portfolio_dd": self.state.portfolio_dd,
                        "limit": self.limits.max_portfolio_dd,
                    },
                )

        # 3. VPIN toxicity check
        if self._latest_vpin:
            max_vpin = max(self._latest_vpin.values())
            if max_vpin > self.limits.vpin_block_threshold:
                self.state.last_vpin = max_vpin
                await self._publish_alert(
                    alert_type="vpin_toxic",
                    severity="warning",
                    message=(
                        f"VPIN toxicity: max VPIN {max_vpin:.3f} "
                        f"> {self.limits.vpin_block_threshold:.2f} (new entries blocked)"
                    ),
                    data={
                        "max_vpin": max_vpin,
                        "threshold": self.limits.vpin_block_threshold,
                        "per_instrument": dict(self._latest_vpin),
                    },
                )

        # 4. Exposure check
        total_exp = self.state.total_exposure
        if total_exp > self.limits.max_total_exposure:
            await self._publish_alert(
                alert_type="exposure_limit",
                severity="warning",
                message=(
                    f"Exposure limit: {total_exp:.2%} > {self.limits.max_total_exposure:.2%}"
                ),
                data={
                    "total_exposure": total_exp,
                    "limit": self.limits.max_total_exposure,
                },
            )

    # ------------------------------------------------------------------
    # Greeks loop: periodic recomputation
    # ------------------------------------------------------------------

    async def _greeks_loop(self) -> None:
        """Recompute portfolio Greeks at regular intervals."""
        while self._running:
            try:
                await asyncio.sleep(self._greeks_interval)
            except asyncio.CancelledError:
                break

            if not self.state.positions:
                continue

            await self._recompute_greeks()

    async def _recompute_greeks(self) -> None:
        """Compute Greeks for all positions using qlx.risk.greeks."""
        positions_list: list[dict] = []

        for key, pos in self.state.positions.items():
            positions_list.append({
                "symbol": pos.symbol,
                "instrument_type": pos.instrument_type,
                "strike": pos.strike,
                "expiry": pos.expiry,
                "direction": pos.direction,
                "quantity": pos.weight,
                "iv": pos.metadata.get("iv", 0.20) if pos.metadata else 0.20,
                "dte": self._compute_dte(pos.expiry),
            })

        loop = asyncio.get_running_loop()

        def _compute() -> PortfolioGreeks:
            return compute_portfolio_greeks(positions_list, self._spot_prices)

        try:
            greeks = await loop.run_in_executor(None, _compute)
            self._latest_greeks = greeks
            self._greeks_updates += 1
            self._last_greeks_time = time.monotonic()

            # Check Greeks-based alerts
            if greeks.gross_delta > 2.0:
                await self._publish_alert(
                    alert_type="delta_exposure",
                    severity="warning",
                    message=(
                        f"High delta exposure: gross={greeks.gross_delta:.2f}, "
                        f"net={greeks.net_delta:+.2f}"
                    ),
                    data={
                        "net_delta": greeks.net_delta,
                        "gross_delta": greeks.gross_delta,
                        "net_gamma": greeks.net_gamma,
                        "delta_by_symbol": greeks.delta_by_symbol,
                    },
                )

            logger.debug(
                "Greeks updated: delta=%+.2f gamma=%+.4f vega=%+.2f theta=%+.2f",
                greeks.net_delta,
                greeks.net_gamma,
                greeks.net_vega,
                greeks.net_theta,
            )

        except Exception as e:
            logger.error("Greeks computation failed: %s", e, exc_info=True)

    def _compute_dte(self, expiry: str) -> int:
        """Compute days to expiry from an ISO date string."""
        if not expiry:
            return 7  # default
        try:
            from datetime import date as date_cls
            exp_date = date_cls.fromisoformat(expiry)
            today = datetime.now(IST).date()
            dte = (exp_date - today).days
            return max(dte, 1)
        except (ValueError, TypeError):
            return 7

    # ------------------------------------------------------------------
    # Alert publishing
    # ------------------------------------------------------------------

    async def _publish_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        data: dict | None = None,
    ) -> None:
        """Publish a risk alert event."""
        alert = {
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now(IST).isoformat(),
            "data": data or {},
            "circuit_breaker_active": self._circuit_breaker_active,
        }
        await self.event_bus.publish(EventType.RISK_ALERT, alert, source="risk_monitor")
        self._alerts_published += 1

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def snapshot(self) -> RiskSnapshot:
        """Return a point-in-time risk snapshot."""
        total_pnl = sum(
            p.weight * p.unrealized_pnl
            for p in self.state.positions.values()
            if p.unrealized_pnl != 0
        )

        return RiskSnapshot(
            timestamp=time.monotonic(),
            mark_to_market_pnl=total_pnl,
            intraday_pnl=total_pnl,
            intraday_dd=self._intraday_dd,
            total_exposure=self.state.total_exposure,
            greeks=self._latest_greeks,
            vpin_max=max(self._latest_vpin.values()) if self._latest_vpin else 0.0,
            circuit_breaker_active=self._circuit_breaker_active,
        )

    @property
    def circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_active

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (admin action)."""
        self._circuit_breaker_active = False
        self.state.circuit_breaker_active = False
        self._session_peak_equity = self.state.equity
        self._intraday_dd = 0.0
        logger.info("Circuit breaker manually reset")

    def stats(self) -> dict:
        """Return monitor-level diagnostics."""
        return {
            "running": self._running,
            "mtm_updates": self._mtm_updates,
            "greeks_updates": self._greeks_updates,
            "alerts_published": self._alerts_published,
            "intraday_dd": round(self._intraday_dd, 4),
            "circuit_breaker_active": self._circuit_breaker_active,
            "spot_prices": {k: round(v, 2) for k, v in self._spot_prices.items()},
            "positions_tracked": len(self.state.positions),
            "latest_greeks": {
                "net_delta": self._latest_greeks.net_delta if self._latest_greeks else None,
                "net_gamma": self._latest_greeks.net_gamma if self._latest_greeks else None,
                "gross_delta": self._latest_greeks.gross_delta if self._latest_greeks else None,
            },
        }
