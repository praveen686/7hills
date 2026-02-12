"""Deterministic backtest engine for DTRN.

Replays micro-bars in chronological order:
1. Feature engine computes x_t, m_t
2. Topology learner updates A_t
3. DTRN model produces regime + position target
4. Risk manager clips position
5. Execution model simulates fills + costs
6. PnL tracking and metrics computation
"""
from __future__ import annotations

import gc
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ..config import DTRNConfig
from ..data.loader import load_day, list_available_dates
from ..data.features import FeatureEngine
from ..model.topology import DynamicTopology
from ..model.graph_net import DTRN as DTRNModel
from ..model.dtrn import create_dtrn
from .risk import RiskManager
from .execution import ExecutionModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_backtest(
    config: DTRNConfig = None,
    start_date: date = None,
    end_date: date = None,
    instrument: str = "NIFTY",
    model: Optional[DTRNModel] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Run deterministic backtest.

    If model is None, creates an untrained model (for sanity checking the pipeline).
    For actual backtesting, pass a trained model.

    Returns comprehensive results dict.
    """
    if config is None:
        config = DTRNConfig()

    # Initialize components
    feature_engine = FeatureEngine(config)
    n_features = feature_engine.n_features

    topology, dtrn_model = create_dtrn(config, n_features)
    if model is not None:
        dtrn_model = model
    dtrn_model = dtrn_model.to(device)
    dtrn_model.eval()

    risk_mgr = RiskManager(config)
    exec_model = ExecutionModel(config)

    # Get available dates
    all_dates = list_available_dates()
    if not all_dates:
        raise ValueError("No data available")

    if start_date is None:
        start_date = all_dates[0]
    if end_date is None:
        end_date = all_dates[-1]

    dates = [d for d in all_dates if start_date <= d <= end_date]
    if not dates:
        raise ValueError(f"No dates in range {start_date} to {end_date}")

    if verbose:
        print(f"Backtesting {instrument} from {dates[0]} to {dates[-1]} ({len(dates)} days)")
        print(f"  Model params: {sum(p.numel() for p in dtrn_model.parameters()):,}")
        print(f"  Features: {n_features}")
        print(f"  Initial capital: Rs.{config.initial_capital:,.0f}")

    # Track results
    daily_results = []
    all_positions = []
    all_regimes = []
    all_topology_stats = []

    equity = config.initial_capital
    peak_equity = equity

    # Persistent state across days (position + entry price carry overnight)
    position = 0
    entry_price = 0.0

    for day_idx, trading_date in enumerate(dates):
        df = load_day(trading_date, instrument)
        if df is None or len(df) < 10:
            continue

        # Reset daily state — topology, features, and GRU all reset together
        feature_engine.reset()
        topology.reset()
        h_state = None  # GRU hidden state resets with topology
        risk_mgr.reset_day()
        risk_mgr.state.equity = equity
        risk_mgr.state.peak_equity = peak_equity

        day_pnl = 0.0
        day_costs = 0.0
        day_trades = 0

        day_positions = []
        day_regimes = []

        # Pending order from previous bar's signal — filled at current bar's open
        pending_target_contracts = None
        pending_vol = None

        for bar_idx, (dt, row) in enumerate(df.iterrows()):
            bar = {
                "datetime": dt,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volume", 0),
                "oi": row.get("oi", 0),
            }
            price = float(row["close"])
            open_price = float(row["open"])

            # ── Execute pending order at THIS bar's open (signal from PREVIOUS bar) ──
            if pending_target_contracts is not None and pending_target_contracts != position:
                fill = exec_model.simulate_fill(
                    position, pending_target_contracts, open_price, instrument,
                    pending_vol if pending_vol is not None else 0.001,
                )

                # PnL from closed portion
                if position != 0:
                    closed_delta = min(abs(fill["delta"]), abs(position))
                    if fill["delta"] * position < 0:  # reducing position
                        pnl_per_contract = fill["fill_price"] - entry_price
                        if position < 0:
                            pnl_per_contract = -pnl_per_contract
                        day_pnl += closed_delta * pnl_per_contract

                day_costs += fill["cost"]
                day_trades += 1

                # VWAP entry price for additions (H4 fix)
                old_pos = position
                position = fill["new_position"]
                if abs(position) > 0:
                    if old_pos != 0 and np.sign(old_pos) == np.sign(position):
                        # Adding to position — VWAP average
                        total = abs(position)
                        old_part = abs(old_pos)
                        new_part = total - old_part
                        if total > 0:
                            entry_price = (entry_price * old_part + fill["fill_price"] * new_part) / total
                    else:
                        entry_price = fill["fill_price"]

                risk_mgr.update_position(position, entry_price, instrument)
                # Update day_pnl in risk state (H3 fix — kill switch)
                risk_mgr.state.day_pnl = day_pnl - day_costs

            pending_target_contracts = None
            pending_vol = None

            # ── Mark-to-market for risk checks ──
            if position != 0:
                unrealized_now = position * (price - entry_price)
                if position < 0:
                    unrealized_now = -position * (entry_price - price)
                risk_mgr.state.day_pnl = day_pnl - day_costs + unrealized_now

            # 1. Features
            feat, mask = feature_engine.update(bar)

            # 2. Topology
            adj = topology.update(feat, mask)
            weights = topology.get_weights()

            # 3. DTRN forward (single step)
            x_t = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)  # (1, d)
            m_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            a_t = torch.tensor(adj, dtype=torch.float32, device=device)
            w_t = torch.tensor(weights, dtype=torch.float32, device=device)

            outputs, h_state = dtrn_model.forward_step(x_t, m_t, a_t, w_t, h_state)

            target_pos = outputs["position"].item()
            regime_probs = outputs["regime_probs"].squeeze(0).cpu().numpy()

            # 4. Risk management
            target_contracts = risk_mgr.check_and_clip(
                target_pos, price, regime_probs, instrument
            )

            # 5. Queue order for NEXT bar execution (no look-ahead)
            pending_target_contracts = target_contracts
            pending_vol = feat[5] if mask[5] > 0 else 0.001  # rvol_10

            day_positions.append(position)
            day_regimes.append(regime_probs.copy())

        # End of day: mark-to-market unrealized PnL
        last_price = float(df.iloc[-1]["close"])
        if position != 0:
            unrealized = position * (last_price - entry_price)
            if position < 0:
                unrealized = -position * (entry_price - last_price)
        else:
            unrealized = 0.0

        net_pnl = day_pnl + unrealized - day_costs
        equity_start_of_day = equity  # before today's PnL
        equity += net_pnl
        if equity > peak_equity:
            peak_equity = equity

        daily_return = net_pnl / max(equity_start_of_day, 1e-10)

        daily_results.append({
            "date": trading_date,
            "pnl": net_pnl,
            "costs": day_costs,
            "trades": day_trades,
            "return": daily_return,
            "equity": equity,
            "position_eod": position,
            "bars": len(df),
        })

        all_positions.extend(day_positions)
        all_regimes.extend(day_regimes)
        all_topology_stats.append(topology.get_stats())

        if verbose and (day_idx % 5 == 0 or day_idx == len(dates) - 1):
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            print(f"  [{trading_date}] PnL: Rs.{net_pnl:+,.0f}  "
                  f"Costs: Rs.{day_costs:,.0f}  Trades: {day_trades}  "
                  f"Equity: Rs.{equity:,.0f}  DD: {dd:.2f}%")

    # Compute aggregate metrics
    if not daily_results:
        return {"error": "No trading days processed"}

    returns = np.array([d["return"] for d in daily_results])
    equities = np.array([d["equity"] for d in daily_results])
    total_costs = sum(d["costs"] for d in daily_results)
    total_trades = sum(d["trades"] for d in daily_results)

    # Sharpe
    if len(returns) > 1 and np.std(returns, ddof=1) > 1e-10:
        sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 1 and np.std(downside, ddof=1) > 1e-10:
        sortino = np.mean(returns) / np.std(downside, ddof=1) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equities)
    dd = (peak - equities) / np.maximum(peak, 1e-10)
    max_dd = float(dd.max())

    # Win rate
    profitable_days = sum(1 for r in returns if r > 0)
    win_rate = profitable_days / len(returns) if len(returns) > 0 else 0.0

    # Total return
    total_return = (equity - config.initial_capital) / config.initial_capital

    # Regime distribution (average across all bars)
    if all_regimes:
        regime_arr = np.array(all_regimes)
        avg_regime = regime_arr.mean(axis=0)
    else:
        avg_regime = np.zeros(config.n_regimes)

    # Topology churn
    if all_topology_stats:
        avg_edges = np.mean([s["n_edges"] for s in all_topology_stats])
        avg_density = np.mean([s["edge_density"] for s in all_topology_stats])
    else:
        avg_edges = 0
        avg_density = 0

    results = {
        "instrument": instrument,
        "start_date": str(dates[0]),
        "end_date": str(dates[-1]),
        "n_days": len(daily_results),
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd * 100,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "total_costs": total_costs,
        "avg_trades_per_day": total_trades / len(daily_results),
        "final_equity": equity,
        "avg_regime_probs": avg_regime.tolist(),
        "regime_names": config.regime_names,
        "avg_topology_edges": avg_edges,
        "avg_topology_density": avg_density,
        "daily_results": daily_results,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Backtest Results: {instrument}")
        print(f"{'='*60}")
        print(f"  Period: {dates[0]} to {dates[-1]} ({len(daily_results)} days)")
        print(f"  Total Return: {total_return*100:+.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Sortino Ratio: {sortino:.2f}")
        print(f"  Max Drawdown: {max_dd*100:.2f}%")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Total Trades: {total_trades}")
        print(f"  Total Costs: Rs.{total_costs:,.0f}")
        print(f"  Final Equity: Rs.{equity:,.0f}")
        print(f"  Avg Regime: {dict(zip(config.regime_names, [f'{p:.2f}' for p in avg_regime]))}")
        print(f"  Avg Topology Edges: {avg_edges:.1f} (density: {avg_density:.3f})")

    return results
