"""Crypto Liquidity Regime Strategy (CLRS) — S26.

A funding rate carry strategy targeting high-funding altcoin perpetuals.

Proven (Signal A -- Carry):
  Enter when smoothed annualized funding > 20%, exit when < 3%.
  Volume floor $10M (not $50M -- captures 3x more symbols).
  90-day backtest: +4.18% avg, Sharpe 9.15, 9/12 profitable, max DD 1.34%.

Infrastructure (computed but not used for trading yet):
  - VPIN  -- Volume-synchronized Probability of Informed Trading
  - OFI   -- Order Flow Imbalance (Cont-Kukanov-Stoikov)
  - Hawkes -- Self-exciting point process for cascade detection
  - Kyle's Lambda -- Price impact regression (market fragility)

Killed:
  - Funding PCA (Signal B) -- z-scores ~ 0, no predictive value at 8h scale.
  - VPIN filter -- hourly VPIN is noise, causes 4x turnover, destroys carry returns.
    Needs tick-level data (SBE stream) to be meaningful.
"""
