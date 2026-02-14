# 8. VECTORBTPRO TEARSHEET & TRADE ANALYSIS
import vectorbtpro as vbt

print("=" * 70)
print("  VECTORBTPRO TEARSHEET & TRADE ANALYSIS")
print("=" * 70)

# Use results from Cell 7 (lab_output variable)
oos_returns = lab_output['results']['oos_returns']
oos_signals = lab_output['results']['oos_signals']
cfg = lab_output['cfg']

for ticker in oos_returns.columns:
    print(f"\n{'=' * 60}")
    print(f"  {ticker} TEARSHEET")
    print(f"{'=' * 60}")
    
    sig = oos_signals[ticker].dropna()
    ret = oos_returns[ticker].dropna()
    
    if len(sig) < 10:
        print(f"  Insufficient OOS data for {ticker} ({len(sig)} days), skipping.")
        continue
    
    # Get close prices for the OOS period from featured_data
    feat_df = lab_output['featured_data'][ticker]
    close = feat_df['close'].reindex(sig.index).dropna()
    
    # Align all series on common dates
    common_idx = sig.index.intersection(close.index)
    if len(common_idx) < 10:
        print(f"  Insufficient aligned data for {ticker} ({len(common_idx)} days), skipping.")
        continue
    
    sig = sig.loc[common_idx]
    close = close.loc[common_idx]
    
    # Build position series with T+1 lag (signal at t -> trade at t+1)
    position = np.sign(sig).shift(1).fillna(0)
    
    # Derive entry/exit signals from position changes
    prev_pos = position.shift(1).fillna(0)
    long_entries  = (position > 0) & (prev_pos <= 0)
    long_exits    = (position <= 0) & (prev_pos > 0)
    short_entries = (position < 0) & (prev_pos >= 0)
    short_exits   = (position >= 0) & (prev_pos < 0)
    
    # VBT Portfolio: long/short from signals
    pf = vbt.Portfolio.from_signals(
        close=close,
        long_entries=long_entries,
        long_exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        fees=cfg.bps_cost / 2,       # per-side fee
        slippage=cfg.bps_cost / 2,    # per-side slippage
        freq='1D',
        init_cash=1_000_000,
    )
    
    # Print full stats
    print(pf.stats())
    
    # Trade-level analysis
    if hasattr(pf, 'trades') and pf.trades.count > 0:
        trades = pf.trades
        print(f"\n  --- Trade Summary ---")
        print(f"  Total Trades:   {trades.count}")
        print(f"  Win Rate:       {trades.win_rate:.2%}")
        print(f"  Profit Factor:  {trades.profit_factor:.2f}")
        print(f"  Avg P&L:        {trades.pnl.mean():.2f}")
        print(f"  Max Win:        {trades.pnl.max():.2f}")
        print(f"  Max Loss:       {trades.pnl.min():.2f}")
        print(f"  Expectancy:     {trades.expectancy:.2f}")
        print(f"  Avg Duration:   {trades.duration.mean()}")
        
        # Show recent trades
        readable = trades.records_readable
        if len(readable) > 0:
            print(f"\n  --- Last 10 Trades ---")
            print(readable.tail(10).to_string())
    else:
        print("  No trades executed.")
    
    # Plot equity curve
    try:
        fig = pf.plot()
        fig.update_layout(
            title=f'{ticker} -- VectorBTPro Equity Curve',
            height=400,
            template='plotly_white'
        )
        fig.show()
    except Exception as e:
        print(f"  Plot failed: {e}")
        # Fallback: matplotlib equity curve
        equity = pf.value()
        plt.figure(figsize=(12, 4))
        plt.plot(equity.index, equity.values, linewidth=1.5)
        plt.title(f'{ticker} -- VectorBTPro Equity Curve')
        plt.ylabel('Portfolio Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

print("\n" + "=" * 70)
print("  VectorBTPro analysis complete")
print("=" * 70)