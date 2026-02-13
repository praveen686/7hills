"""Phase 1: Verification Script.

Compares baseline strategy performance vs RL/MDP-enhanced versions.
Standardises backtesting for S1, S7, and S10 using the production StrategyProtocol.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.strategies.protocol import Signal, StrategyProtocol
from quantlaxmi.strategies.s1_vrp.strategy import S1VRPStrategy
from quantlaxmi.strategies.s7_regime.strategy import S7RegimeSwitchStrategy
from quantlaxmi.strategies.s10_gamma_scalp.strategy import S10GammaScalpStrategy

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class VerifierBacktest:
    def __init__(self, store: MarketDataStore, strategy: StrategyProtocol):
        self.store = store
        self.strategy = strategy
        self.strategy_id = strategy.strategy_id
        
    def _get_spot(self, d: date, symbol: str) -> float | None:
        """Fetch spot price for price-based return computation."""
        d_str = d.isoformat()
        _idx_name = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank"}.get(
            symbol.upper(), f"Nifty {symbol}"
        )
        try:
            df = self.store.sql(
                'SELECT "Closing Index Value" as close FROM nse_index_close '
                'WHERE date = ? AND "Index Name" = ? LIMIT 1',
                [d_str, _idx_name],
            )
            if df is not None and not df.empty:
                return float(df["close"].iloc[0])
        except Exception:
            pass
        return None

    def run(self, start_date: date, end_date: date) -> dict:
        """Run event-driven backtest loop."""
        logger.info(f"Running backtest for {self.strategy_id} from {start_date} to {end_date}")
        
        current_date = start_date
        dates = []
        all_signals = []
        positions = {} # symbol -> {direction, entry_price, conviction}
        daily_pnl = []
        
        # Warmup handles lookbacks
        warmup = self.strategy.warmup_days()
        current_date = start_date - timedelta(days=warmup * 1.5) # rough calendar days
        
        pnl_series = []
        
        while current_date <= end_date:
            # Only scan and calculate for trading days
            spot_nifty = self._get_spot(current_date, "NIFTY")
            if spot_nifty is None:
                current_date += timedelta(days=1)
                continue
            
            # 1. Update existing positions (mark-to-market)
            # This is simplified: we use spot price change for attribution
            # Real options strategies would need MTM of the spread.
            # We follow the simplification in ensemble_backtest.py: attribution by spot return.
            day_pnl = 0.0
            for symbol, pos in list(positions.items()):
                current_spot = self._get_spot(current_date, symbol)
                if current_spot and pos.get("last_spot"):
                    ret = (current_spot / pos["last_spot"] - 1.0)
                    if pos["direction"] == "short":
                        ret = -ret
                    day_pnl += ret * pos["conviction"]
                    pos["last_spot"] = current_spot
            
            # 2. Strategy Scan
            if current_date >= start_date:
                signals = self.strategy.scan(current_date, self.store)
                for sig in signals:
                    if sig.direction == "flat":
                        if sig.symbol in positions:
                            del positions[sig.symbol]
                    else:
                        spot = self._get_spot(current_date, sig.symbol)
                        if spot:
                            positions[sig.symbol] = {
                                "direction": sig.direction,
                                "last_spot": spot,
                                "conviction": sig.conviction
                            }
                
                pnl_series.append({"date": current_date, "pnl": day_pnl})

            current_date += timedelta(days=1)
            
        return self._compute_metrics(pnl_series)

    def _compute_metrics(self, pnl_series: list[dict]) -> dict:
        if not pnl_series:
            return {"sharpe": 0, "total_return": 0, "max_dd": 0}
        
        df = pd.DataFrame(pnl_series)
        rets = df["pnl"].values
        equity = np.cumprod(1 + rets)
        
        total_ret = (equity[-1] - 1) if len(equity) > 0 else 0
        std = np.std(rets, ddof=1)
        sharpe = float(np.mean(rets) / std * np.sqrt(252)) if std > 0 else 0
        
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0
        
        return {
            "sharpe": round(sharpe, 3),
            "total_return_pct": round(total_ret * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "n_days": len(df)
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["s1", "s7", "s10"], required=True)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-02-06")
    args = parser.parse_args()
    
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    
    store = MarketDataStore()
    
    strategies_to_test = []
    if args.strategy == "s1":
        strategies_to_test = [
            ("S1 Baseline", S1VRPStrategy(use_kelly=False)),
            ("S1 RL/Kelly", S1VRPStrategy(use_kelly=True))
        ]
    elif args.strategy == "s7":
        strategies_to_test = [
            ("S7 Baseline", S7RegimeSwitchStrategy(use_mdp=False)),
            ("S7 RL/MDP", S7RegimeSwitchStrategy(use_mdp=True))
        ]
    elif args.strategy == "s10":
        strategies_to_test = [
            ("S10 Baseline", S10GammaScalpStrategy(use_deep_hedge=False)),
            ("S10 RL/DeepHedge", S10GammaScalpStrategy(use_deep_hedge=True))
        ]
        
    print(f"\nPhase 1 Verification: {args.strategy.upper()}")
    print("-" * 40)
    
    for name, strategy in strategies_to_test:
        verifier = VerifierBacktest(store, strategy)
        results = verifier.run(start, end)
        print(f"{name}:")
        print(f"  Sharpe Ratio:  {results['sharpe']}")
        print(f"  Total Return:  {results['total_return_pct']}%")
        print(f"  Max Drawdown:  {results['max_drawdown_pct']}%")
        print(f"  Observations:  {results['n_days']} days")
        print()

    store.close()

if __name__ == "__main__":
    main()
