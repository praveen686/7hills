"""Seed WAL event data for testing the Why Panel.

Writes a realistic day of events into data/events/ so the backend
can serve them via the Why Panel API endpoints.

Usage:
    python scripts/seed_why_panel.py
"""

from pathlib import Path
from engine.live.event_log import EventLogWriter
from core.events.types import EventType

WAL_DIR = Path("data/events")
RUN_ID = "seed-why-panel-001"
TODAY = "2025-10-15"


def main():
    WAL_DIR.mkdir(parents=True, exist_ok=True)

    writer = EventLogWriter(
        base_dir=WAL_DIR,
        run_id=RUN_ID,
        fsync_policy="none",
    )

    ts_base = f"{TODAY}T09:30:00.000000Z"

    # ── S5 Hawkes: signal → gate (pass) → order → fill ──
    s5_sig = writer.emit(
        event_type=EventType.SIGNAL.value,
        source="s5_hawkes",
        payload={
            "direction": "long",
            "conviction": 0.87,
            "instrument_type": "FUT",
            "strike": 0.0,
            "expiry": "",
            "ttl_bars": 5,
            "regime": "low_vol",
            "components": {
                "gex_regime": "positive",
                "raw_score": 0.72,
                "smoothed_score": 0.68,
                "hawkes_intensity": 1.45,
                "tick_imbalance": 0.31,
            },
            "reasoning": "Hawkes intensity spike + positive GEX regime → long bias",
        },
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=ts_base,
    )

    writer.emit(
        event_type=EventType.GATE_DECISION.value,
        source="risk_engine",
        payload={
            "signal_seq": s5_sig.seq,
            "gate": "risk_check",
            "approved": True,
            "adjusted_weight": 0.12,
            "reason": "",
            "vpin": 0.35,
            "portfolio_dd": 0.02,
            "strategy_dd": 0.01,
            "total_exposure": 0.45,
        },
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=f"{TODAY}T09:30:00.100000Z",
    )

    writer.emit(
        event_type=EventType.ORDER.value,
        source="executor",
        payload={
            "order_id": "ord-s5-001",
            "action": "submit",
            "side": "buy",
            "order_type": "market",
            "quantity": 50,
            "price": 24150.0,
        },
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=f"{TODAY}T09:30:00.200000Z",
    )

    writer.emit(
        event_type=EventType.FILL.value,
        source="executor",
        payload={
            "order_id": "ord-s5-001",
            "fill_price": 24152.5,
            "quantity": 50,
            "slippage_pts": 2.5,
            "commission": 20.0,
        },
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=f"{TODAY}T09:30:00.300000Z",
    )

    # ── S1 VRP Options: signal → gate (pass) → order ──
    s1_sig = writer.emit(
        event_type=EventType.SIGNAL.value,
        source="s1_vrp",
        payload={
            "direction": "short_vol",
            "conviction": 0.79,
            "instrument_type": "OPT",
            "strike": 24200.0,
            "expiry": "2025-10-30",
            "ttl_bars": 10,
            "regime": "contango",
            "components": {
                "composite": 0.65,
                "sig_pctile": 0.82,
                "skew_premium": 0.015,
                "left_tail": 0.03,
                "atm_iv": 14.2,
                "rv_21": 11.8,
                "vrp": 2.4,
            },
            "reasoning": "VRP > 2.0 in contango regime → sell vol via put spread",
        },
        strategy_id="s1_vrp",
        symbol="NIFTY",
        ts=f"{TODAY}T09:35:00.000000Z",
    )

    writer.emit(
        event_type=EventType.GATE_DECISION.value,
        source="risk_engine",
        payload={
            "signal_seq": s1_sig.seq,
            "gate": "risk_check",
            "approved": True,
            "adjusted_weight": 0.08,
            "reason": "",
            "vpin": 0.28,
            "portfolio_dd": 0.02,
            "strategy_dd": 0.005,
            "total_exposure": 0.53,
        },
        strategy_id="s1_vrp",
        symbol="NIFTY",
        ts=f"{TODAY}T09:35:00.100000Z",
    )

    writer.emit(
        event_type=EventType.ORDER.value,
        source="executor",
        payload={
            "order_id": "ord-s1-001",
            "action": "submit",
            "side": "sell",
            "order_type": "limit",
            "quantity": 25,
            "price": 85.0,
        },
        strategy_id="s1_vrp",
        symbol="NIFTY",
        ts=f"{TODAY}T09:35:00.200000Z",
    )

    # ── S4 IV Mean Revert: signal → gate (BLOCKED) ──
    s4_sig = writer.emit(
        event_type=EventType.SIGNAL.value,
        source="s4_iv_mr",
        payload={
            "direction": "long",
            "conviction": 0.62,
            "instrument_type": "OPT",
            "strike": 24000.0,
            "expiry": "2025-10-30",
            "ttl_bars": 3,
            "regime": "high_vol",
            "components": {
                "atm_iv": 18.5,
                "iv_pctile": 0.85,
                "spot": 24050.0,
                "iv_z_score": 1.8,
                "mean_iv_60d": 15.2,
            },
            "reasoning": "IV at 85th percentile, expecting mean reversion lower",
        },
        strategy_id="s4_iv_mr",
        symbol="NIFTY",
        ts=f"{TODAY}T10:00:00.000000Z",
    )

    writer.emit(
        event_type=EventType.GATE_DECISION.value,
        source="risk_engine",
        payload={
            "signal_seq": s4_sig.seq,
            "gate": "risk_check",
            "approved": False,
            "adjusted_weight": 0.0,
            "reason": "Portfolio drawdown limit exceeded (3.2% > 3.0%)",
            "vpin": 0.52,
            "portfolio_dd": 0.032,
            "strategy_dd": 0.018,
            "total_exposure": 0.72,
        },
        strategy_id="s4_iv_mr",
        symbol="NIFTY",
        ts=f"{TODAY}T10:00:00.100000Z",
    )

    # ── S7 Regime Switch: signal on BANKNIFTY → gate (pass) → order → fill ──
    s7_sig = writer.emit(
        event_type=EventType.SIGNAL.value,
        source="s7_regime",
        payload={
            "direction": "long",
            "conviction": 0.74,
            "instrument_type": "FUT",
            "strike": 0.0,
            "expiry": "",
            "ttl_bars": 8,
            "regime": "trending_up",
            "components": {
                "sub_strategy": "supertrend",
                "regime": "trending_up",
                "entropy": 0.42,
                "mi": 0.18,
                "z_score": 1.2,
                "pct_b": 0.78,
                "confidence": 0.81,
                "supertrend_signal": 1,
            },
            "reasoning": "Regime classified as trending_up; supertrend sub-strategy active with z_score=1.2",
        },
        strategy_id="s7_regime",
        symbol="BANKNIFTY",
        ts=f"{TODAY}T10:15:00.000000Z",
    )

    writer.emit(
        event_type=EventType.GATE_DECISION.value,
        source="risk_engine",
        payload={
            "signal_seq": s7_sig.seq,
            "gate": "risk_check",
            "approved": True,
            "adjusted_weight": 0.10,
            "reason": "",
            "vpin": 0.30,
            "portfolio_dd": 0.025,
            "strategy_dd": 0.012,
            "total_exposure": 0.55,
        },
        strategy_id="s7_regime",
        symbol="BANKNIFTY",
        ts=f"{TODAY}T10:15:00.100000Z",
    )

    writer.emit(
        event_type=EventType.ORDER.value,
        source="executor",
        payload={
            "order_id": "ord-s7-001",
            "action": "submit",
            "side": "buy",
            "order_type": "market",
            "quantity": 15,
            "price": 51250.0,
        },
        strategy_id="s7_regime",
        symbol="BANKNIFTY",
        ts=f"{TODAY}T10:15:00.200000Z",
    )

    writer.emit(
        event_type=EventType.FILL.value,
        source="executor",
        payload={
            "order_id": "ord-s7-001",
            "fill_price": 51255.0,
            "quantity": 15,
            "slippage_pts": 5.0,
            "commission": 20.0,
        },
        strategy_id="s7_regime",
        symbol="BANKNIFTY",
        ts=f"{TODAY}T10:15:00.300000Z",
    )

    # ── Portfolio snapshot ──
    writer.emit(
        event_type=EventType.SNAPSHOT.value,
        source="risk_engine",
        payload={
            "equity": 1.052,
            "peak_equity": 1.08,
            "portfolio_dd": 0.026,
            "total_exposure": 0.55,
            "vpin": 0.32,
            "position_count": 4,
            "regime": "low_vol",
            "strategies_active": ["s5_hawkes", "s1_vrp", "s7_regime"],
            "margin_used_pct": 0.45,
        },
        ts=f"{TODAY}T15:30:00.000000Z",
    )

    writer.close()

    # Verify
    from engine.replay.reader import WalReader
    reader = WalReader(base_dir=WAL_DIR)
    events = reader.read_date(TODAY)
    print(f"Seeded {len(events)} events for {TODAY}")
    for e in events:
        print(f"  seq={e.seq:3d}  {e.event_type:<16s}  {e.strategy_id or '-':12s}  {e.symbol or '-':12s}")
    print(f"\nAvailable dates: {reader.available_dates()}")
    print(f"\nWAL dir: {WAL_DIR.resolve()}")


if __name__ == "__main__":
    main()
