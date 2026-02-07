"use client";

import type { TradeAnalytics } from "@/lib/types";

interface Props {
  trade: TradeAnalytics;
  onClick?: () => void;
}

export function OpportunityCard({ trade, onClick }: Props) {
  const pnlColor = trade.pnl_pct >= 0 ? "text-profit" : "text-loss";
  const dirBadge = trade.direction === "long"
    ? "bg-profit/20 text-profit"
    : "bg-loss/20 text-loss";

  return (
    <div
      onClick={onClick}
      className="bg-gray-900 rounded-xl p-4 hover:bg-gray-800 transition-colors cursor-pointer border border-gray-800"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-white">{trade.symbol}</span>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${dirBadge}`}>
            {trade.direction.toUpperCase()}
          </span>
        </div>
        <div className="text-right">
          <span className="text-xs text-gray-500">{trade.strategy_id}</span>
          <p className="text-xs text-gray-600">{trade.duration_days}d</p>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-4 gap-3">
        <div>
          <p className="text-[10px] text-gray-500 uppercase">MFM</p>
          <p className="text-sm font-mono text-profit">{(trade.mfm * 100).toFixed(2)}%</p>
        </div>
        <div>
          <p className="text-[10px] text-gray-500 uppercase">MDA</p>
          <p className="text-sm font-mono text-loss">{(trade.mda * 100).toFixed(2)}%</p>
        </div>
        <div>
          <p className="text-[10px] text-gray-500 uppercase">Efficiency</p>
          <p className="text-sm font-mono text-accent">{(trade.efficiency * 100).toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-[10px] text-gray-500 uppercase">P&L</p>
          <p className={`text-sm font-mono ${pnlColor}`}>{(trade.pnl_pct * 100).toFixed(2)}%</p>
        </div>
      </div>

      {/* Efficiency Bar */}
      <div className="mt-3">
        <div className="w-full bg-gray-800 rounded-full h-1.5">
          <div
            className="bg-accent rounded-full h-1.5 transition-all"
            style={{ width: `${Math.min(100, Math.max(0, trade.efficiency * 100))}%` }}
          />
        </div>
      </div>

      {/* Price info */}
      {trade.price_path_available && (
        <div className="mt-2 flex justify-between text-[10px] text-gray-600">
          <span>Entry: {trade.entry_price.toFixed(2)}</span>
          <span>Exit: {trade.exit_price.toFixed(2)}</span>
          <span>Optimal: {trade.optimal_exit_price.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
}
