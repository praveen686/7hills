"use client";

import { useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { fetchStrategy, updateStrategyStatus } from "@/lib/api";
import { EquityChart } from "@/components/charts/EquityChart";
import { DrawdownChart } from "@/components/charts/DrawdownChart";
import { SignalTable } from "@/components/charts/SignalTable";
import {
  formatCurrency,
  formatCompact,
  formatPct,
  formatSharpe,
  formatRatio,
  formatPnl,
  formatCurrencyPrecise,
  formatDateTime,
  pnlColor,
} from "@/lib/formatters";
import { clsx } from "clsx";
import type { StrategyDetail, StrategyStatus } from "@/lib/types";

// ============================================================
// Strategy Detail Page
// ============================================================

export default function StrategyDetailPage() {
  const params = useParams<{ id: string }>();
  const strategyId = params.id;

  const {
    data: strategy,
    isLoading,
    error,
    refetch,
  } = useQuery<StrategyDetail>({
    queryKey: ["strategy", strategyId],
    queryFn: () => fetchStrategy(strategyId),
    refetchInterval: 10000,
  });

  const handleStatusChange = async (newStatus: "live" | "paused" | "stopped") => {
    try {
      await updateStrategyStatus(strategyId, newStatus);
      refetch();
    } catch (err) {
      console.error("Failed to update strategy status:", err);
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6 animate-in">
        <div className="h-8 w-48 bg-gray-800 rounded animate-pulse" />
        <div className="grid grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="card animate-pulse h-24" />
          ))}
        </div>
        <div className="card animate-pulse h-80" />
      </div>
    );
  }

  if (error || !strategy) {
    return (
      <div className="card text-center py-16">
        <p className="text-sm text-gray-500">
          Failed to load strategy: {strategyId}
        </p>
        <button onClick={() => refetch()} className="btn-primary mt-4">
          Retry
        </button>
      </div>
    );
  }

  const drawdownData = (strategy.equity_curve ?? [])
    .filter((p) => p.drawdown !== undefined)
    .map((p) => ({
      date: p.date,
      drawdown: p.drawdown ?? 0,
    }));

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-semibold text-white">
              {strategy.name}
            </h1>
            <StatusBadge status={strategy.status} />
          </div>
          <p className="text-sm text-gray-500 mt-1">{strategy.description}</p>
          <div className="flex items-center gap-4 mt-2">
            <span className="text-xs text-gray-600 font-mono">
              ID: {strategy.strategy_id ?? strategy.id}
            </span>
            {strategy.timeframe && (
              <span className="text-xs text-gray-600">
                {strategy.timeframe}
              </span>
            )}
            {strategy.instruments && (
              <span className="text-xs text-gray-600">
                {strategy.instruments.join(", ")}
              </span>
            )}
          </div>
        </div>
        <div className="flex gap-2">
          {strategy.status !== "live" && (
            <button
              onClick={() => handleStatusChange("live")}
              className="px-3 py-1.5 text-xs font-medium bg-profit/10 text-profit border border-profit/20 rounded-lg hover:bg-profit/20 transition-colors"
            >
              Start
            </button>
          )}
          {strategy.status === "live" && (
            <button
              onClick={() => handleStatusChange("paused")}
              className="px-3 py-1.5 text-xs font-medium bg-yellow-500/10 text-yellow-400 border border-yellow-500/20 rounded-lg hover:bg-yellow-500/20 transition-colors"
            >
              Pause
            </button>
          )}
          {strategy.status !== "stopped" && (
            <button
              onClick={() => handleStatusChange("stopped")}
              className="px-3 py-1.5 text-xs font-medium bg-loss/10 text-loss border border-loss/20 rounded-lg hover:bg-loss/20 transition-colors"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <MetricCard label="Return" value={formatPct(strategy.return_pct)} color={pnlColor(strategy.return_pct)} />
        <MetricCard label="Equity" value={formatCurrency(strategy.equity)} />
        <MetricCard label="Sharpe" value={formatSharpe(strategy.sharpe)} />
        <MetricCard label="Win Rate" value={formatPct(strategy.win_rate, 1)} />
        <MetricCard label="Open" value={String(strategy.n_open ?? 0)} />
        <MetricCard label="Closed" value={String(strategy.n_closed ?? 0)} />
      </div>

      {/* Equity Curve */}
      {strategy.equity_curve && strategy.equity_curve.length > 0 && (
        <div className="card">
          <p className="card-header mb-4">Equity Curve</p>
          <EquityChart data={strategy.equity_curve} height={300} showBenchmark />
        </div>
      )}

      {/* Drawdown */}
      {drawdownData.length > 0 && (
        <div className="card">
          <p className="card-header mb-4">Drawdown</p>
          <DrawdownChart data={drawdownData} height={180} />
        </div>
      )}

      {/* Positions & Trades */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Open Positions */}
        <div className="card p-0 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800">
            <p className="text-xs font-medium uppercase tracking-wider text-gray-400">
              Open Positions ({strategy.positions.length})
            </p>
          </div>
          {strategy.positions.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="table-header">Symbol</th>
                    <th className="table-header">Direction</th>
                    <th className="table-header">Weight</th>
                    <th className="table-header">Entry</th>
                    <th className="table-header">Type</th>
                  </tr>
                </thead>
                <tbody>
                  {strategy.positions.map((pos: Record<string, unknown>, i: number) => (
                    <tr key={i} className="border-b border-gray-800/50">
                      <td className="table-cell font-medium text-white text-sm">
                        {String(pos.symbol ?? "")}
                      </td>
                      <td className="table-cell">
                        <span
                          className={clsx(
                            "text-xs font-medium",
                            String(pos.direction) === "long" ? "text-profit" : "text-loss"
                          )}
                        >
                          {String(pos.direction ?? "").toUpperCase()}
                        </span>
                      </td>
                      <td className="table-cell font-mono text-sm">
                        {formatPct(Number(pos.weight ?? 0))}
                      </td>
                      <td className="table-cell font-mono text-sm">
                        {formatCurrencyPrecise(Number(pos.entry_price ?? 0))}
                      </td>
                      <td className="table-cell text-xs text-gray-500">
                        {String(pos.instrument_type ?? "")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex items-center justify-center h-24 text-sm text-gray-600">
              No open positions
            </div>
          )}
        </div>

        {/* Recent Trades */}
        <div className="card p-0 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800">
            <p className="text-xs font-medium uppercase tracking-wider text-gray-400">
              Recent Trades ({strategy.n_closed ?? 0} total)
            </p>
          </div>
          {strategy.recent_trades.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="table-header">Symbol</th>
                    <th className="table-header">Direction</th>
                    <th className="table-header">Entry</th>
                    <th className="table-header">Exit</th>
                    <th className="table-header">P&L</th>
                    <th className="table-header">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {strategy.recent_trades.slice(0, 15).map((trade, i) => (
                    <tr key={i} className="border-b border-gray-800/50">
                      <td className="table-cell text-sm text-white">
                        {trade.symbol}
                      </td>
                      <td className="table-cell">
                        <span
                          className={clsx(
                            "text-xs font-medium",
                            trade.direction === "long" ? "text-profit" : "text-loss"
                          )}
                        >
                          {trade.direction.toUpperCase()}
                        </span>
                      </td>
                      <td className="table-cell font-mono text-xs text-gray-400">
                        {trade.entry_date}
                      </td>
                      <td className="table-cell font-mono text-xs text-gray-400">
                        {trade.exit_date}
                      </td>
                      <td className={`table-cell font-mono text-sm font-medium ${pnlColor(trade.pnl_pct)}`}>
                        {formatPct(trade.pnl_pct)}
                      </td>
                      <td className="table-cell text-xs text-gray-500">
                        {trade.exit_reason}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex items-center justify-center h-24 text-sm text-gray-600">
              No trades yet
            </div>
          )}
        </div>
      </div>

      {/* Signals */}
      {strategy.signals && strategy.signals.length > 0 && (
        <div className="card p-0 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800">
            <p className="text-xs font-medium uppercase tracking-wider text-gray-400">
              Strategy Signals
            </p>
          </div>
          <SignalTable signals={strategy.signals} />
        </div>
      )}

      {/* Parameters */}
      {strategy.params && Object.keys(strategy.params).length > 0 && (
        <div className="card">
          <p className="card-header mb-3">Parameters</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(strategy.params).map(([key, value]) => (
              <div key={key} className="bg-gray-950 rounded-lg px-3 py-2">
                <p className="text-[10px] text-gray-600 uppercase">{key}</p>
                <p className="text-sm font-mono text-gray-300">
                  {String(value)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------- Sub-components ----------

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    live: "badge-live",
    running: "badge-live",
    paused: "badge-paused",
    stopped: "badge-stopped",
    backtest: "badge-stopped",
    stale: "badge-stopped",
  };

  return (
    <span className={styles[status] ?? "badge-stopped"}>
      {(status === "live" || status === "running") && (
        <span className="w-1.5 h-1.5 rounded-full bg-profit animate-pulse" />
      )}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

function MetricCard({
  label,
  value,
  color = "text-white",
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="card">
      <p className="text-[10px] text-gray-500 uppercase tracking-wider">
        {label}
      </p>
      <p className={`text-lg font-mono font-semibold mt-1 ${color}`}>
        {value}
      </p>
    </div>
  );
}
