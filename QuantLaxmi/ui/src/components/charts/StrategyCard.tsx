"use client";

import Link from "next/link";
import type { StrategySummary, StrategyStatus } from "@/lib/types";
import {
  formatCompact,
  formatPct,
  formatSharpe,
  formatPnl,
  pnlColor,
} from "@/lib/formatters";
import { clsx } from "clsx";

// ============================================================
// Strategy Summary Card
// ============================================================

interface StrategyCardProps {
  strategy: StrategySummary;
  compact?: boolean;
}

function StatusBadge({ status }: { status: StrategyStatus }) {
  const styles: Record<StrategyStatus, string> = {
    live: "badge-live",
    paused: "badge-paused",
    stopped: "badge-stopped",
    backtest: "badge-stopped",
    running: "badge-live",
    stale: "badge-paused",
  };

  const dots: Record<StrategyStatus, string> = {
    live: "bg-profit",
    paused: "bg-yellow-400",
    stopped: "bg-gray-500",
    backtest: "bg-gray-500",
    running: "bg-profit",
    stale: "bg-yellow-400",
  };

  return (
    <span className={styles[status]}>
      <span className={`w-1.5 h-1.5 rounded-full ${dots[status]} ${status === "live" ? "animate-pulse" : ""}`} />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

export function StrategyCard({ strategy, compact = false }: StrategyCardProps) {
  const sid = strategy.strategy_id ?? strategy.id ?? "";

  return (
    <Link href={`/strategies/${sid}`} className="block">
      <div
        className={clsx(
          "card hover:border-gray-700 transition-all duration-200 cursor-pointer group",
          compact ? "p-3" : "p-4"
        )}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className="text-sm font-semibold text-white group-hover:text-accent transition-colors">
              {strategy.name}
            </h3>
            <p className="text-xs text-gray-500 font-mono mt-0.5">
              {sid}
            </p>
          </div>
          <StatusBadge status={strategy.status as StrategyStatus} />
        </div>

        {/* P&L */}
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div>
            <p className="text-[10px] text-gray-500 uppercase tracking-wider">
              Return
            </p>
            <p className={`text-sm font-mono font-semibold ${pnlColor(strategy.return_pct)}`}>
              {formatPct(strategy.return_pct)}
            </p>
          </div>
          <div>
            <p className="text-[10px] text-gray-500 uppercase tracking-wider">
              Equity
            </p>
            <p className="text-sm font-mono font-semibold text-gray-300">
              {strategy.equity?.toFixed(4) ?? "--"}
            </p>
          </div>
        </div>

        {/* Metrics */}
        {!compact && (
          <div className="grid grid-cols-4 gap-2 pt-3 border-t border-gray-800">
            <MetricItem label="Sharpe" value={formatSharpe(strategy.sharpe)} />
            <MetricItem label="Win Rate" value={formatPct(strategy.win_rate, 1)} />
            <MetricItem label="Max DD" value={formatPct(strategy.max_drawdown)} negative />
            <MetricItem label="CAGR" value={formatPct(strategy.cagr)} />
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between mt-3 pt-2 border-t border-gray-800/50">
          <span className="text-xs text-gray-600">
            {strategy.n_open ?? 0} positions
          </span>
          <span className="text-xs text-gray-600">
            {strategy.n_closed ?? 0} trades
          </span>
        </div>
      </div>
    </Link>
  );
}

function MetricItem({
  label,
  value,
  negative = false,
}: {
  label: string;
  value: string;
  negative?: boolean;
}) {
  return (
    <div className="text-center">
      <p className="text-[10px] text-gray-600 uppercase">{label}</p>
      <p
        className={clsx(
          "text-xs font-mono font-medium mt-0.5",
          negative ? "text-loss" : "text-gray-300"
        )}
      >
        {value}
      </p>
    </div>
  );
}
