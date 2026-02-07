"use client";

import type { TradeAnalyticsSummary } from "@/lib/types";

interface Props {
  summary: TradeAnalyticsSummary | undefined;
  isLoading: boolean;
}

export function DiagnosticsSummary({ summary, isLoading }: Props) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-gray-900 rounded-xl p-4 animate-pulse h-24" />
        ))}
      </div>
    );
  }

  if (!summary || Object.keys(summary).length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 text-center text-gray-500">
        No trade analytics data available
      </div>
    );
  }

  // Aggregate across all strategies
  const strategies = Object.values(summary);
  const totalTrades = strategies.reduce((s, v) => s + v.n_trades, 0);
  const avgEfficiency = strategies.length > 0
    ? strategies.reduce((s, v) => s + v.avg_efficiency, 0) / strategies.length
    : 0;
  const avgMfm = strategies.length > 0
    ? strategies.reduce((s, v) => s + v.avg_mfm, 0) / strategies.length
    : 0;
  const avgExitQuality = strategies.length > 0
    ? strategies.reduce((s, v) => s + v.avg_exit_quality, 0) / strategies.length
    : 0;

  const cards = [
    { label: "Total Trades", value: totalTrades.toString(), color: "text-white" },
    { label: "Avg Efficiency", value: `${(avgEfficiency * 100).toFixed(1)}%`, color: avgEfficiency > 0 ? "text-profit" : "text-loss" },
    { label: "Avg MFM", value: `${(avgMfm * 100).toFixed(2)}%`, color: "text-accent" },
    { label: "Avg Exit Quality", value: `${(avgExitQuality * 100).toFixed(1)}%`, color: avgExitQuality > 0.5 ? "text-profit" : "text-yellow-400" },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card) => (
        <div key={card.label} className="bg-gray-900 rounded-xl p-4">
          <p className="text-xs text-gray-500 uppercase tracking-wider">{card.label}</p>
          <p className={`text-2xl font-bold mt-1 ${card.color}`}>{card.value}</p>
        </div>
      ))}
    </div>
  );
}
