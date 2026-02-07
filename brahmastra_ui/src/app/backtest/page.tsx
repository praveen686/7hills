"use client";

import { useState, useEffect } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { runBacktest, fetchBacktestStrategies } from "@/lib/api";
import { EquityChart } from "@/components/charts/EquityChart";
import { DrawdownChart } from "@/components/charts/DrawdownChart";
import {
  formatCurrency,
  formatPct,
  formatSharpe,
  formatRatio,
  formatNumber,
  formatCompact,
  pnlColor,
} from "@/lib/formatters";
import { clsx } from "clsx";
import type { BacktestParams, BacktestResult } from "@/lib/types";

// ============================================================
// Backtest Page — Strategy-Specific Parameter Form + Results
// ============================================================

export default function BacktestPage() {
  const { data: backtestStrategies } = useQuery({
    queryKey: ["backtest-strategies"],
    queryFn: fetchBacktestStrategies,
  });

  const [params, setParams] = useState<BacktestParams>({
    strategy_id: "",
    start_date: "2025-03-01",
    end_date: "2026-01-31",
    initial_capital: 10000000,
    params: {},
  });

  const [customParams, setCustomParams] = useState<string>("{}");

  // Auto-fill default params when strategy is selected
  useEffect(() => {
    if (!params.strategy_id || !backtestStrategies) return;
    const info = backtestStrategies.find((s) => s.strategy_id === params.strategy_id);
    if (info) {
      setCustomParams(JSON.stringify(info.default_params, null, 2));
    }
  }, [params.strategy_id, backtestStrategies]);

  const mutation = useMutation<BacktestResult, Error, BacktestParams>({
    mutationFn: runBacktest,
  });

  const handleRun = () => {
    if (!params.strategy_id) return;

    let parsedParams: Record<string, number | string | boolean> = {};
    try {
      parsedParams = JSON.parse(customParams);
    } catch {
      // Use empty params if JSON is invalid
    }

    mutation.mutate({
      ...params,
      params: parsedParams,
    });
  };

  const result = mutation.data;

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-white">Backtest Engine</h1>
        <p className="text-sm text-gray-500 mt-1">
          Run strategy-specific historical backtests with full causal integrity
        </p>
      </div>

      {/* Parameter Form */}
      <div className="card">
        <p className="card-header mb-4">Configuration</p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Strategy Select */}
          <div>
            <label className="block text-xs text-gray-500 mb-1.5">
              Strategy
            </label>
            <select
              value={params.strategy_id}
              onChange={(e) =>
                setParams((p) => ({ ...p, strategy_id: e.target.value }))
              }
              className="input-field"
            >
              <option value="">Select strategy...</option>
              {backtestStrategies?.map((s) => (
                <option key={s.strategy_id} value={s.strategy_id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>

          {/* Start Date */}
          <div>
            <label className="block text-xs text-gray-500 mb-1.5">
              Start Date
            </label>
            <input
              type="date"
              value={params.start_date}
              onChange={(e) =>
                setParams((p) => ({ ...p, start_date: e.target.value }))
              }
              className="input-field"
            />
          </div>

          {/* End Date */}
          <div>
            <label className="block text-xs text-gray-500 mb-1.5">
              End Date
            </label>
            <input
              type="date"
              value={params.end_date}
              onChange={(e) =>
                setParams((p) => ({ ...p, end_date: e.target.value }))
              }
              className="input-field"
            />
          </div>

          {/* Initial Capital */}
          <div>
            <label className="block text-xs text-gray-500 mb-1.5">
              Initial Capital
            </label>
            <input
              type="number"
              value={params.initial_capital}
              onChange={(e) =>
                setParams((p) => ({
                  ...p,
                  initial_capital: Number(e.target.value),
                }))
              }
              className="input-field"
              step={100000}
            />
          </div>
        </div>

        {/* Custom Parameters */}
        <div className="mt-4">
          <label className="block text-xs text-gray-500 mb-1.5">
            Strategy Parameters (JSON)
            {params.strategy_id && (
              <span className="text-gray-600 ml-2">
                — defaults loaded for {params.strategy_id}
              </span>
            )}
          </label>
          <textarea
            value={customParams}
            onChange={(e) => setCustomParams(e.target.value)}
            className="input-field font-mono text-xs h-24 resize-none"
            placeholder='{"lookback": 20, "threshold": 0.5}'
          />
        </div>

        {/* Run Button */}
        <div className="mt-4 flex items-center gap-4">
          <button
            onClick={handleRun}
            disabled={!params.strategy_id || mutation.isPending}
            className={clsx(
              "btn-primary",
              (!params.strategy_id || mutation.isPending) &&
                "opacity-50 cursor-not-allowed"
            )}
          >
            {mutation.isPending ? (
              <span className="flex items-center gap-2">
                <svg
                  className="animate-spin h-4 w-4"
                  viewBox="0 0 24 24"
                  fill="none"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Running Backtest...
              </span>
            ) : (
              "Run Backtest"
            )}
          </button>

          {mutation.isError && (
            <p className="text-xs text-loss">
              Error: {mutation.error.message}
            </p>
          )}
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-6 animate-in">
          {/* Error state */}
          {result.error && (
            <div className="card border-loss/30">
              <p className="text-loss text-sm font-medium">Backtest Failed</p>
              <p className="text-xs text-gray-400 mt-1">{result.error}</p>
            </div>
          )}

          {/* Summary Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <ResultMetric
              label="Total Return"
              value={formatPct(result.total_return)}
              color={pnlColor(result.total_return)}
            />
            <ResultMetric label="CAGR" value={formatPct(result.cagr)} />
            <ResultMetric
              label="Sharpe"
              value={formatSharpe(result.sharpe)}
              highlight={(result.sharpe ?? 0) > 1}
            />
            <ResultMetric
              label="Sortino"
              value={formatRatio(result.sortino)}
            />
            <ResultMetric
              label="Max DD"
              value={formatPct(result.max_drawdown)}
              color="text-loss"
            />
            <ResultMetric
              label="Win Rate"
              value={formatPct(result.win_rate, 1)}
            />
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <ResultMetric
              label="Final Equity"
              value={result.final_equity != null ? formatCurrency(result.final_equity) : "--"}
            />
            <ResultMetric
              label="Profit Factor"
              value={formatRatio(result.profit_factor)}
            />
            <ResultMetric
              label="Total Trades"
              value={result.total_trades != null ? formatNumber(result.total_trades) : "--"}
            />
            <ResultMetric
              label="Avg Trade P&L"
              value={formatCompact(result.avg_trade_pnl)}
              color={pnlColor(result.avg_trade_pnl)}
            />
          </div>

          {/* Equity Curve */}
          {result.equity_curve && result.equity_curve.length > 0 && (
          <div className="card">
            <p className="card-header mb-4">Equity Curve</p>
            <EquityChart
              data={result.equity_curve}
              height={320}
              showBenchmark
            />
          </div>
          )}

          {/* Drawdown */}
          {result.drawdown_curve && result.drawdown_curve.length > 0 && (
          <div className="card">
            <p className="card-header mb-4">Drawdown</p>
            <DrawdownChart data={result.drawdown_curve} height={200} />
          </div>
          )}

          {/* Monthly Returns Heatmap */}
          {result.monthly_returns && result.monthly_returns.length > 0 && (
            <div className="card">
              <p className="card-header mb-4">Monthly Returns</p>
              <MonthlyHeatmap returns={result.monthly_returns} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------- Sub-components ----------

function ResultMetric({
  label,
  value,
  color = "text-white",
  highlight = false,
}: {
  label: string;
  value: string;
  color?: string;
  highlight?: boolean;
}) {
  return (
    <div className={clsx("card", highlight && "glow-accent border-accent/30")}>
      <p className="text-[10px] text-gray-500 uppercase tracking-wider">
        {label}
      </p>
      <p className={`text-lg font-mono font-semibold mt-1 ${color}`}>
        {value}
      </p>
    </div>
  );
}

function MonthlyHeatmap({
  returns,
}: {
  returns: { year: number; month: number; return_pct: number }[];
}) {
  const MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];

  // Group by year
  const years = [...new Set(returns.map((r) => r.year))].sort();
  const grid: Record<number, Record<number, number>> = {};
  for (const r of returns) {
    if (!grid[r.year]) grid[r.year] = {};
    grid[r.year][r.month] = r.return_pct;
  }

  const cellColor = (val: number | undefined) => {
    if (val === undefined) return "bg-gray-900 text-gray-700";
    if (val > 5) return "bg-profit/30 text-profit";
    if (val > 0) return "bg-profit/15 text-profit/80";
    if (val > -5) return "bg-loss/15 text-loss/80";
    return "bg-loss/30 text-loss";
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr>
            <th className="table-header">Year</th>
            {MONTHS.map((m) => (
              <th key={m} className="table-header text-center">
                {m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {years.map((year) => (
            <tr key={year}>
              <td className="table-cell font-mono font-medium text-gray-400">
                {year}
              </td>
              {Array.from({ length: 12 }, (_, i) => i + 1).map((month) => {
                const val = grid[year]?.[month];
                return (
                  <td key={month} className="p-1 text-center">
                    <span
                      className={clsx(
                        "inline-block w-full px-2 py-1.5 rounded font-mono",
                        cellColor(val)
                      )}
                    >
                      {val !== undefined ? formatPct(val, 1) : "--"}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
