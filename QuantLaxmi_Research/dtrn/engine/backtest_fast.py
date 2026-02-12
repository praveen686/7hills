"""Vectorized backtest engine for DTRN.

Key optimizations over backtest.py:
1. Pre-compute ALL features + topology per day in numpy (no per-bar tensor creation)
2. Single GPU forward pass per day: (1, T, d) → (1, T, outputs)
3. Trade simulation stays sequential but is pure Python arithmetic (microseconds)

Expected speedup: 50-100x (30 min → 20-30 seconds).
"""
from __future__ import annotations

import gc
import logging
import time
from datetime import date
from typing import Optional

import numpy as np
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


def _precompute_day(
    df,
    feature_engine: FeatureEngine,
    topology: DynamicTopology,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute all features, masks, adjacency, and weights for one day.

    Returns:
        features: (T, d) float32
        masks: (T, d) float32
        adjs: (T, d, d) float32
        weights_arr: (T, d, d) float32
    """
    feature_engine.reset()
    topology.reset()

    T = len(df)
    d = feature_engine.n_features

    features = np.zeros((T, d), dtype=np.float32)
    masks = np.zeros((T, d), dtype=np.float32)
    adjs = np.zeros((T, d, d), dtype=np.float32)
    weights_arr = np.zeros((T, d, d), dtype=np.float32)

    # Extract numpy arrays from DataFrame once (avoid iterrows)
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64) if "volume" in df.columns else np.zeros(T)
    ois = df["oi"].values.astype(np.float64) if "oi" in df.columns else np.zeros(T)
    datetimes = df.index

    for i in range(T):
        bar = {
            "datetime": datetimes[i],
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "volume": volumes[i],
            "oi": ois[i],
        }
        feat, mask = feature_engine.update(bar)
        adj = topology.update(feat, mask)
        w = topology.get_weights()

        features[i] = feat
        masks[i] = mask
        adjs[i] = adj
        weights_arr[i] = w

    return features, masks, adjs, weights_arr


@torch.no_grad()
def run_backtest_fast(
    config: DTRNConfig = None,
    start_date: date = None,
    end_date: date = None,
    instrument: str = "NIFTY",
    model: Optional[DTRNModel] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Vectorized DTRN backtest — same results as run_backtest, ~50-100x faster.

    The key insight: features/topology are sequential (EWMA state) but cheap in numpy.
    The bottleneck was per-bar GPU transfer + model calls. We batch those per day.
    """
    if config is None:
        config = DTRNConfig()

    feature_engine = FeatureEngine(config)
    n_features = feature_engine.n_features

    topology_template, dtrn_model = create_dtrn(config, n_features)
    if model is not None:
        dtrn_model = model
    dtrn_model = dtrn_model.to(device)
    dtrn_model.eval()

    risk_mgr = RiskManager(config)
    exec_model = ExecutionModel(config)

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
        print(f"  Mode: VECTORIZED (1 GPU call per day)")

    # Tracking
    daily_results = []
    all_positions = []
    all_regimes = []
    all_topology_stats = []

    equity = config.initial_capital
    peak_equity = equity

    # Persistent state across days
    position = 0
    entry_price = 0.0

    t_feat = 0.0
    t_gpu = 0.0
    t_sim = 0.0

    for day_idx, trading_date in enumerate(dates):
        df = load_day(trading_date, instrument)
        if df is None or len(df) < 10:
            continue

        T = len(df)

        # ── Phase 1: Pre-compute features + topology (numpy, ~5ms per day) ──
        t0 = time.perf_counter()
        features, masks, adjs, weights_day = _precompute_day(
            df, feature_engine, topology_template,
        )
        t_feat += time.perf_counter() - t0

        # ── Phase 2: Single GPU forward pass (1, T, d) → (1, T, outputs) ──
        t0 = time.perf_counter()
        x = torch.from_numpy(features).unsqueeze(0).to(device)   # (1, T, d)
        m = torch.from_numpy(masks).unsqueeze(0).to(device)      # (1, T, d)
        a = torch.from_numpy(adjs).to(device)                    # (T, d, d)
        w = torch.from_numpy(weights_day).to(device)             # (T, d, d)

        outputs = dtrn_model.forward(x, m, a, w, h0=None)

        # Extract all T positions and regimes at once
        positions_all = outputs["position"].squeeze(0).squeeze(-1).cpu().numpy()   # (T,)
        regimes_all = outputs["regime_probs"].squeeze(0).cpu().numpy()              # (T, K)
        t_gpu += time.perf_counter() - t0

        # ── Phase 3: Sequential trade simulation (pure Python, ~0.1ms per day) ──
        t0 = time.perf_counter()
        opens = df["open"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        rvol_col = features[:, 5]  # rvol_10
        rvol_mask = masks[:, 5]

        risk_mgr.reset_day()
        risk_mgr.state.equity = equity
        risk_mgr.state.peak_equity = peak_equity

        day_pnl = 0.0
        day_costs = 0.0
        day_trades = 0
        pending_target_contracts = None
        pending_vol = None

        for bar_idx in range(T):
            price = closes[bar_idx]
            open_price = opens[bar_idx]
            regime_probs = regimes_all[bar_idx]

            # Execute pending order at THIS bar's open
            if pending_target_contracts is not None and pending_target_contracts != position:
                fill = exec_model.simulate_fill(
                    position, pending_target_contracts, open_price, instrument,
                    pending_vol if pending_vol is not None else 0.001,
                )

                if position != 0:
                    closed_delta = min(abs(fill["delta"]), abs(position))
                    if fill["delta"] * position < 0:
                        pnl_per_contract = fill["fill_price"] - entry_price
                        if position < 0:
                            pnl_per_contract = -pnl_per_contract
                        day_pnl += closed_delta * pnl_per_contract

                day_costs += fill["cost"]
                day_trades += 1

                old_pos = position
                position = fill["new_position"]
                if abs(position) > 0:
                    if old_pos != 0 and np.sign(old_pos) == np.sign(position):
                        total = abs(position)
                        old_part = abs(old_pos)
                        new_part = total - old_part
                        if total > 0:
                            entry_price = (entry_price * old_part + fill["fill_price"] * new_part) / total
                    else:
                        entry_price = fill["fill_price"]

                risk_mgr.update_position(position, entry_price, instrument)
                risk_mgr.state.day_pnl = day_pnl - day_costs

            pending_target_contracts = None
            pending_vol = None

            # Mark-to-market
            if position != 0:
                unrealized_now = position * (price - entry_price)
                if position < 0:
                    unrealized_now = -position * (entry_price - price)
                risk_mgr.state.day_pnl = day_pnl - day_costs + unrealized_now

            # Risk-clip the model's position signal
            target_pos = positions_all[bar_idx]
            target_contracts = risk_mgr.check_and_clip(
                target_pos, price, regime_probs, instrument
            )

            # Queue order for NEXT bar
            pending_target_contracts = target_contracts
            pending_vol = rvol_col[bar_idx] if rvol_mask[bar_idx] > 0 else 0.001

        # End of day
        last_price = closes[-1]
        if position != 0:
            unrealized = position * (last_price - entry_price)
            if position < 0:
                unrealized = -position * (entry_price - last_price)
        else:
            unrealized = 0.0

        net_pnl = day_pnl + unrealized - day_costs
        equity_start = equity
        equity += net_pnl
        if equity > peak_equity:
            peak_equity = equity

        daily_return = net_pnl / max(equity_start, 1e-10)
        t_sim += time.perf_counter() - t0

        daily_results.append({
            "date": trading_date,
            "pnl": net_pnl,
            "costs": day_costs,
            "trades": day_trades,
            "return": daily_return,
            "equity": equity,
            "position_eod": position,
            "bars": T,
        })

        all_positions.extend(positions_all.tolist())
        all_regimes.extend(regimes_all.tolist())
        all_topology_stats.append(topology_template.get_stats())

        if verbose and (day_idx % 20 == 0 or day_idx == len(dates) - 1):
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            print(f"  [{trading_date}] d{day_idx+1}/{len(dates)}  "
                  f"PnL: Rs.{net_pnl:+,.0f}  Trades: {day_trades}  "
                  f"Equity: Rs.{equity:,.0f}  DD: {dd:.2f}%")

    # Timing report
    total_time = t_feat + t_gpu + t_sim
    if verbose and total_time > 0:
        print(f"\n  Timing: features={t_feat:.1f}s  GPU={t_gpu:.1f}s  "
              f"sim={t_sim:.1f}s  total={total_time:.1f}s")

    # Aggregate metrics (identical to backtest.py)
    if not daily_results:
        return {"error": "No trading days processed"}

    returns = np.array([d["return"] for d in daily_results])
    equities = np.array([d["equity"] for d in daily_results])
    total_costs = sum(d["costs"] for d in daily_results)
    total_trades = sum(d["trades"] for d in daily_results)

    if len(returns) > 1 and np.std(returns, ddof=1) > 1e-10:
        sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
    else:
        sharpe = 0.0

    downside = returns[returns < 0]
    if len(downside) > 1 and np.std(downside, ddof=1) > 1e-10:
        sortino = np.mean(returns) / np.std(downside, ddof=1) * np.sqrt(252)
    else:
        sortino = 0.0

    peak = np.maximum.accumulate(equities)
    dd = (peak - equities) / np.maximum(peak, 1e-10)
    max_dd = float(dd.max())

    profitable_days = sum(1 for r in returns if r > 0)
    win_rate = profitable_days / len(returns) if len(returns) > 0 else 0.0

    total_return = (equity - config.initial_capital) / config.initial_capital

    if all_regimes:
        regime_arr = np.array(all_regimes)
        avg_regime = regime_arr.mean(axis=0)
    else:
        avg_regime = np.zeros(config.n_regimes)

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
