"""WebSocket handlers for real-time streaming.

/ws/portfolio      — portfolio snapshot every 2 seconds
/ws/ticks          — market tick summary (indices + crypto LTPs)
/ws/ticks/{token}  — live tick data for a specific instrument token
/ws/signals        — strategy signal broadcasts
/ws/risk           — risk metrics stream
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# ------------------------------------------------------------------
# Connection manager — tracks active WS clients per channel
# ------------------------------------------------------------------

class ConnectionManager:
    """Manage WebSocket connections grouped by channel."""

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, channel: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            if channel not in self._connections:
                self._connections[channel] = set()
            self._connections[channel].add(ws)
        logger.debug("WS connect: channel=%s, total=%d", channel, len(self._connections[channel]))

    async def disconnect(self, channel: str, ws: WebSocket) -> None:
        async with self._lock:
            conns = self._connections.get(channel)
            if conns:
                conns.discard(ws)
                if not conns:
                    del self._connections[channel]
        logger.debug("WS disconnect: channel=%s", channel)

    async def broadcast(self, channel: str, data: dict[str, Any]) -> None:
        """Send JSON payload to all clients on a channel."""
        async with self._lock:
            conns = self._connections.get(channel)
            if not conns:
                return
            dead: list[WebSocket] = []
            payload = json.dumps(data, default=str)
            for ws in conns:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                conns.discard(ws)

    @property
    def active_channels(self) -> list[str]:
        return list(self._connections.keys())

    def channel_count(self, channel: str) -> int:
        conns = self._connections.get(channel)
        return len(conns) if conns else 0


# Module-level manager instance (attached to app.state in app.py)
manager = ConnectionManager()

# Per-token tick staleness tracking
_last_tick_ts: dict[int, datetime] = {}
_STALENESS_THRESHOLD_S = 300.0


def update_tick_ts(token: int, ts: datetime | None = None) -> None:
    """Update the last-seen tick timestamp for a token."""
    _last_tick_ts[token] = ts or datetime.now(timezone.utc)


def check_tick_staleness(token: int) -> tuple[bool, float]:
    """Check if a token's tick data is stale.

    Returns (is_stale, staleness_seconds).
    """
    ts = _last_tick_ts.get(token)
    if ts is None:
        return True, float("inf")
    staleness = (datetime.now(timezone.utc) - ts).total_seconds()
    return staleness > _STALENESS_THRESHOLD_S, staleness


# ------------------------------------------------------------------
# /ws/portfolio — push portfolio state every 2s
# ------------------------------------------------------------------

@router.websocket("/ws/portfolio")
async def ws_portfolio(websocket: WebSocket) -> None:
    """Stream portfolio snapshots at 2-second intervals.

    The client receives a JSON object with equity, positions, drawdown,
    exposure, and regime on each tick.
    """
    channel = "portfolio"
    await manager.connect(channel, websocket)

    try:
        while True:
            state = websocket.app.state.engine

            positions = [p.to_dict() for p in state.active_positions()]
            payload = {
                "type": "portfolio",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "equity": round(state.equity, 6),
                "peak_equity": round(state.peak_equity, 6),
                "cash": round(state.cash, 6),
                "drawdown_pct": round(state.portfolio_dd * 100, 4),
                "total_exposure": round(state.total_exposure, 6),
                "total_return_pct": round(state.total_return_pct(), 4),
                "n_positions": len(state.positions),
                "circuit_breaker_active": state.circuit_breaker_active,
                "last_vix": round(state.last_vix, 2),
                "last_regime": state.last_regime,
                "positions": positions,
            }

            await websocket.send_json(payload)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WS portfolio error: %s", exc)
    finally:
        await manager.disconnect(channel, websocket)


# ------------------------------------------------------------------
# /ws/ticks — generic market tick summary (indices + crypto)
# ------------------------------------------------------------------

# Map of index names for DuckDB queries
_INDEX_NAMES = {
    "NIFTY": "Nifty 50",
    "BANKNIFTY": "Nifty Bank",
    "FINNIFTY": "Nifty Financial Services",
    "MIDCPNIFTY": "NIFTY MidSmall Financial Services",
}


@router.websocket("/ws/ticks")
async def ws_ticks_generic(websocket: WebSocket) -> None:
    """Stream market summary ticks for all tracked instruments.

    In live mode: subscribes to EventBus TICK events for real-time data.
    In paper mode: pushes latest close prices from DuckDB every 2s.
    """
    channel = "ticks"
    await manager.connect(channel, websocket)

    try:
        # Check if live engine is running
        live_engine = getattr(websocket.app.state, "live_engine", None)
        event_bus = getattr(websocket.app.state, "event_bus", None)

        if event_bus and live_engine and live_engine.running:
            # LIVE MODE: Subscribe to EventBus TICK events
            from quantlaxmi.engine.live.event_bus import EventType
            tick_queue = event_bus.subscribe(EventType.TICK, max_size=256)

            # Also keep a latest-tick cache per symbol for periodic pushes
            latest_ticks: dict[str, dict] = {}

            try:
                while True:
                    # Drain tick events (non-blocking batch)
                    batch_count = 0
                    while batch_count < 50:
                        try:
                            event = await asyncio.wait_for(tick_queue.get(), timeout=0.5)
                            data = event.data
                            sym = data.get("symbol", "")
                            if sym:
                                latest_ticks[sym] = {
                                    "symbol": sym,
                                    "ltp": data.get("ltp", data.get("last_price", 0)),
                                    "open": data.get("open", 0),
                                    "high": data.get("high", 0),
                                    "low": data.get("low", 0),
                                    "close": data.get("ltp", data.get("last_price", 0)),
                                    "change": data.get("change", 0),
                                    "change_pct": data.get("change_pct", 0),
                                    "volume": data.get("volume", 0),
                                    "oi": data.get("oi", 0),
                                    "vpin": data.get("vpin", 0),
                                    "entropy": data.get("entropy", 0),
                                }
                            batch_count += 1
                        except asyncio.TimeoutError:
                            break

                    # Push latest ticks to client
                    if latest_ticks:
                        payload = {
                            "type": "tick_update",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "ticks": list(latest_ticks.values()),
                            "live": True,
                        }
                        await websocket.send_json(payload)

                    await asyncio.sleep(1)
            finally:
                event_bus.unsubscribe(EventType.TICK, tick_queue)
        else:
            # PAPER MODE: Read from DuckDB
            while True:
                ticks = []
                try:
                    svc = websocket.app.state.market_data_service
                    store = svc.store
                    for sym, index_name in _INDEX_NAMES.items():
                        try:
                            df = store.sql(
                                'SELECT "Closing Index Value", "Open Index Value", '
                                '"High Index Value", "Low Index Value", "Index Date" '
                                'FROM nse_index_close '
                                'WHERE LOWER("Index Name") = LOWER(?) '
                                'ORDER BY "Index Date" DESC LIMIT 2',
                                [index_name],
                            )
                            if not df.empty:
                                row = df.iloc[0]
                                close = float(row.iloc[0])
                                open_ = float(row.iloc[1])
                                high = float(row.iloc[2])
                                low = float(row.iloc[3])
                                prev_close = float(df.iloc[1, 0]) if len(df) > 1 else close
                                change = close - prev_close
                                change_pct = (change / prev_close * 100) if prev_close else 0
                                ticks.append({
                                    "symbol": sym,
                                    "ltp": close,
                                    "open": open_,
                                    "high": high,
                                    "low": low,
                                    "close": close,
                                    "change": round(change, 2),
                                    "change_pct": round(change_pct, 2),
                                    "volume": 0,
                                })
                        except Exception as exc:
                            logger.debug("Tick fetch failed for %s: %s", sym, exc)
                except Exception as exc:
                    logger.debug("WS ticks generic error: %s", exc)

                payload = {
                    "type": "tick_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ticks": ticks,
                    "live": False,
                }
                await websocket.send_json(payload)
                await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WS ticks generic error: %s", exc)
    finally:
        await manager.disconnect(channel, websocket)


# ------------------------------------------------------------------
# /ws/ticks/{token} — stream tick data for a specific instrument
# ------------------------------------------------------------------

@router.websocket("/ws/ticks/{token}")
async def ws_ticks(websocket: WebSocket, token: int) -> None:
    """Stream tick-level data for a given instrument token.

    In live mode, this would connect to the Zerodha WebSocket feed.
    In paper/backtest mode, it serves the latest available tick from
    the MarketDataStore.

    The client can send ``{"action": "subscribe"}`` to begin and
    ``{"action": "unsubscribe"}`` to stop.
    """
    channel = f"ticks:{token}"
    await manager.connect(channel, websocket)

    try:
        # Wait for the client to send a subscribe message (or proceed immediately)
        subscribed = True

        while True:
            # Check for incoming control messages (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                action = msg.get("action", "")
                if action == "unsubscribe":
                    subscribed = False
                    await websocket.send_json({
                        "type": "control",
                        "message": f"Unsubscribed from token {token}",
                    })
                elif action == "subscribe":
                    subscribed = True
                    await websocket.send_json({
                        "type": "control",
                        "message": f"Subscribed to token {token}",
                    })
            except asyncio.TimeoutError:
                pass

            if not subscribed:
                await asyncio.sleep(1)
                continue

            # Fetch latest tick from store (paper mode)
            try:
                svc = websocket.app.state.market_data_service
                store = svc.store
                dates = store.available_dates("ticks")
                if dates:
                    from datetime import date as date_type
                    latest = dates[-1]
                    df = store.get_ticks(token, latest)
                    if not df.empty:
                        last_row = df.iloc[-1]
                        tick_ts_str = str(last_row.get("timestamp", ""))
                        update_tick_ts(token)
                        payload = {
                            "type": "tick",
                            "token": token,
                            "timestamp": tick_ts_str,
                            "ltp": float(last_row.get("ltp", 0)),
                            "volume": int(last_row.get("volume", 0)),
                            "oi": int(last_row.get("oi", 0)),
                        }
                        # Check staleness
                        is_stale, staleness = check_tick_staleness(token)
                        if is_stale:
                            payload["warning"] = f"tick_stale_{staleness:.0f}s"
                            logger.warning(
                                "Tick stale for token %d: %.0fs", token, staleness,
                            )
                        await websocket.send_json(payload)
                    else:
                        await websocket.send_json({
                            "type": "tick",
                            "token": token,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "ltp": 0,
                            "volume": 0,
                            "oi": 0,
                            "message": "no data",
                        })
                else:
                    await websocket.send_json({
                        "type": "tick",
                        "token": token,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "ltp": 0,
                        "volume": 0,
                        "oi": 0,
                        "message": "no tick data available",
                    })
            except Exception as exc:
                logger.debug("WS tick error for token %d: %s", token, exc)
                await websocket.send_json({
                    "type": "error",
                    "token": token,
                    "message": str(exc),
                })

            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WS ticks/%d error: %s", token, exc)
    finally:
        await manager.disconnect(channel, websocket)


# ------------------------------------------------------------------
# /ws/signals — broadcast strategy signals as they arrive
# ------------------------------------------------------------------

# Signal queue: strategies push signals here, WS clients consume them.
_signal_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)


async def push_signal(signal: dict[str, Any]) -> None:
    """Push a signal to the broadcast queue (called by the orchestrator)."""
    try:
        _signal_queue.put_nowait(signal)
    except asyncio.QueueFull:
        logger.warning("Signal queue full, dropping oldest signal")
        try:
            _signal_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        _signal_queue.put_nowait(signal)


@router.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket) -> None:
    """Stream strategy signals in real time.

    In live mode: subscribes to EventBus SIGNAL events.
    In paper mode: reads from the signal queue (orchestrator pushes).
    """
    channel = "signals"
    await manager.connect(channel, websocket)

    try:
        # Check for live EventBus
        event_bus = getattr(websocket.app.state, "event_bus", None)
        live_engine = getattr(websocket.app.state, "live_engine", None)

        if event_bus and live_engine and live_engine.running:
            # LIVE MODE: Subscribe to EventBus SIGNAL events
            from quantlaxmi.engine.live.event_bus import EventType
            signal_queue = event_bus.subscribe(EventType.SIGNAL, max_size=128)

            try:
                while True:
                    try:
                        event = await asyncio.wait_for(signal_queue.get(), timeout=5.0)
                        payload = {
                            "type": "signal",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "live": True,
                            **event.data,
                        }
                        await manager.broadcast(channel, payload)
                    except asyncio.TimeoutError:
                        await websocket.send_json({
                            "type": "heartbeat",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
            finally:
                event_bus.unsubscribe(EventType.SIGNAL, signal_queue)
        else:
            # PAPER MODE: Read from signal queue
            while True:
                try:
                    signal = await asyncio.wait_for(_signal_queue.get(), timeout=5.0)
                    payload = {
                        "type": "signal",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "live": False,
                        **signal,
                    }
                    await manager.broadcast(channel, payload)
                except asyncio.TimeoutError:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WS signals error: %s", exc)
    finally:
        await manager.disconnect(channel, websocket)


# ------------------------------------------------------------------
# /ws/risk — stream risk metrics
# ------------------------------------------------------------------

@router.websocket("/ws/risk")
async def ws_risk(websocket: WebSocket) -> None:
    """Stream risk metrics (drawdown, VaR, exposure, VPIN) every 2s.

    In live mode: reads real VPIN from TickHandler and risk from RiskMonitor.
    """
    channel = "risk"
    await manager.connect(channel, websocket)

    try:
        while True:
            state = websocket.app.state.engine
            live_engine = getattr(websocket.app.state, "live_engine", None)

            # Get VPIN from live tick handler if available
            vpin_val = 0.0
            if live_engine and live_engine.tick_handler:
                # Get max VPIN across all tracked tokens
                for token in live_engine.token_map.values():
                    v = live_engine.tick_handler.get_vpin(token)
                    if v > vpin_val and not (v != v):  # skip NaN
                        vpin_val = v

            # Get risk snapshot from live monitor
            risk_data: dict[str, Any] = {}
            if live_engine and live_engine.risk_monitor:
                try:
                    snap = live_engine.risk_monitor.snapshot()
                    risk_data = {
                        "mtm_pnl": round(snap.mark_to_market_pnl, 6),
                        "intraday_pnl": round(snap.intraday_pnl, 6),
                        "intraday_dd": round(snap.intraday_dd, 6),
                    }
                except Exception:
                    pass

            payload = {
                "type": "vpin_update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "drawdown_pct": round(state.portfolio_dd * 100, 4),
                "total_exposure": round(state.total_exposure, 6),
                "circuit_breaker_active": state.circuit_breaker_active,
                "last_vix": round(state.last_vix, 2),
                "last_regime": state.last_regime,
                "n_positions": len(state.positions),
                "equity": round(state.equity, 6),
                "vpin": round(vpin_val, 4),
                "live": live_engine is not None and live_engine.running,
                **risk_data,
            }

            await websocket.send_json(payload)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WS risk error: %s", exc)
    finally:
        await manager.disconnect(channel, websocket)


# ------------------------------------------------------------------
# /ws/engine — engine diagnostics stream
# ------------------------------------------------------------------

@router.websocket("/ws/engine")
async def ws_engine(websocket: WebSocket) -> None:
    """Stream live engine diagnostics every 5 seconds.

    Includes EventBus stats, TickHandler stats, SignalGenerator stats,
    KiteTickFeed connection status.
    """
    channel = "engine"
    await manager.connect(channel, websocket)

    try:
        while True:
            live_engine = getattr(websocket.app.state, "live_engine", None)

            if live_engine and live_engine.running:
                payload = {
                    "type": "engine_status",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **live_engine.stats(),
                }
            else:
                payload = {
                    "type": "engine_status",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "running": False,
                    "mode": "offline",
                }

            await websocket.send_json(payload)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WS engine error: %s", exc)
    finally:
        await manager.disconnect(channel, websocket)
