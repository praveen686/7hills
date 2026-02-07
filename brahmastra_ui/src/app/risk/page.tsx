"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchRiskMetrics } from "@/lib/api";
import { DrawdownChart } from "@/components/charts/DrawdownChart";
import {
  formatCurrency,
  formatCompact,
  formatPct,
  formatGreek,
  pnlColor,
} from "@/lib/formatters";
import { clsx } from "clsx";
import type { RiskMetrics } from "@/lib/types";

// ============================================================
// Risk Management Page
// ============================================================

export default function RiskPage() {
  const { data: risk, isLoading } = useQuery<RiskMetrics>({
    queryKey: ["risk"],
    queryFn: fetchRiskMetrics,
    refetchInterval: 5000,
  });

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-white">Risk Management</h1>
        <p className="text-sm text-gray-500 mt-1">
          Portfolio Greeks, drawdown tracking, and circuit breakers
        </p>
      </div>

      {/* Circuit Breaker Alert */}
      {risk?.circuit_breaker_active && (
        <div className="bg-loss/10 border border-loss/30 rounded-lg p-4 flex items-center gap-3">
          <svg
            className="w-6 h-6 text-loss flex-shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
            />
          </svg>
          <div>
            <p className="text-sm font-medium text-loss">
              Circuit Breaker Active
            </p>
            <p className="text-xs text-loss/70 mt-0.5">
              {risk.circuit_breaker_reason ?? "Risk limits breached — trading halted"}
            </p>
          </div>
        </div>
      )}

      {/* Portfolio Risk Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <RiskMetricCard
          label="Portfolio Delta"
          value={risk?.portfolio_delta != null ? formatGreek(risk.portfolio_delta) : "--"}
          loading={isLoading}
        />
        <RiskMetricCard
          label="Portfolio Gamma"
          value={risk?.portfolio_gamma != null ? formatGreek(risk.portfolio_gamma) : "--"}
          loading={isLoading}
        />
        <RiskMetricCard
          label="Portfolio Theta"
          value={risk?.portfolio_theta != null ? formatCompact(risk.portfolio_theta) : "--"}
          color={risk?.portfolio_theta != null ? pnlColor(risk.portfolio_theta) : undefined}
          loading={isLoading}
        />
        <RiskMetricCard
          label="Portfolio Vega"
          value={risk?.portfolio_vega != null ? formatGreek(risk.portfolio_vega) : "--"}
          loading={isLoading}
        />
        <RiskMetricCard
          label="VaR (95%)"
          value={risk?.var_95 != null ? formatCompact(risk.var_95) : "--"}
          color="text-loss"
          loading={isLoading}
        />
        <RiskMetricCard
          label="VaR (99%)"
          value={risk?.var_99 != null ? formatCompact(risk.var_99) : "--"}
          color="text-loss"
          loading={isLoading}
        />
      </div>

      {/* Margin Gauge + Drawdown Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Margin Gauge */}
        <div className="card">
          <p className="card-header mb-4">Margin Utilization</p>
          <MarginGauge utilization={risk?.margin_utilization ?? 0} />
        </div>

        {/* Drawdown Summary */}
        <div className="card">
          <p className="card-header mb-4">Drawdown</p>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-[10px] text-gray-600 uppercase">Current DD</p>
              <p className="text-xl font-mono font-semibold text-loss">
                {risk ? formatPct(risk.current_drawdown) : "--"}
              </p>
            </div>
            <div>
              <p className="text-[10px] text-gray-600 uppercase">Max DD</p>
              <p className="text-xl font-mono font-semibold text-loss">
                {risk ? formatPct(risk.max_drawdown) : "--"}
              </p>
            </div>
          </div>
          {/* Mini drawdown chart */}
          {risk?.drawdown_history && risk.drawdown_history.length > 0 && (
            <DrawdownChart data={risk.drawdown_history} height={120} />
          )}
        </div>
      </div>

      {/* Greeks Table */}
      <div className="card p-0 overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
          <p className="text-xs font-medium uppercase tracking-wider text-gray-400">
            Position Greeks
          </p>
          <span className="text-xs text-gray-600">
            {risk?.greeks?.length ?? 0} positions
          </span>
        </div>

        {isLoading ? (
          <div className="animate-pulse space-y-2 p-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-8 bg-gray-800 rounded" />
            ))}
          </div>
        ) : risk?.greeks && risk.greeks.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="table-header">Instrument</th>
                  <th className="table-header">Symbol</th>
                  <th className="table-header text-right">Delta</th>
                  <th className="table-header text-right">Gamma</th>
                  <th className="table-header text-right">Theta</th>
                  <th className="table-header text-right">Vega</th>
                  <th className="table-header text-right">IV</th>
                </tr>
              </thead>
              <tbody>
                {risk.greeks.map((row, i) => (
                  <tr
                    key={i}
                    className="border-b border-gray-800/50 hover:bg-gray-900/50"
                  >
                    <td className="table-cell text-sm text-gray-400">
                      {row.instrument}
                    </td>
                    <td className="table-cell text-sm font-medium text-white">
                      {row.symbol}
                    </td>
                    <td className={clsx("table-cell font-mono text-sm text-right", pnlColor(row.delta))}>
                      {formatGreek(row.delta)}
                    </td>
                    <td className="table-cell font-mono text-sm text-right text-gray-400">
                      {formatGreek(row.gamma)}
                    </td>
                    <td className={clsx("table-cell font-mono text-sm text-right", pnlColor(row.theta))}>
                      {formatGreek(row.theta)}
                    </td>
                    <td className="table-cell font-mono text-sm text-right text-gray-400">
                      {formatGreek(row.vega)}
                    </td>
                    <td className="table-cell font-mono text-sm text-right text-accent">
                      {formatPct(row.iv, 1)}
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="border-t border-gray-700 bg-gray-900/50">
                  <td className="table-cell text-sm font-medium text-white" colSpan={2}>
                    Total
                  </td>
                  <td className={clsx("table-cell font-mono text-sm text-right font-medium", pnlColor(risk.portfolio_delta))}>
                    {risk.portfolio_delta != null ? formatGreek(risk.portfolio_delta) : "--"}
                  </td>
                  <td className="table-cell font-mono text-sm text-right font-medium text-gray-300">
                    {risk.portfolio_gamma != null ? formatGreek(risk.portfolio_gamma) : "--"}
                  </td>
                  <td className={clsx("table-cell font-mono text-sm text-right font-medium", pnlColor(risk.portfolio_theta))}>
                    {risk.portfolio_theta != null ? formatGreek(risk.portfolio_theta) : "--"}
                  </td>
                  <td className="table-cell font-mono text-sm text-right font-medium text-gray-300">
                    {risk.portfolio_vega != null ? formatGreek(risk.portfolio_vega) : "--"}
                  </td>
                  <td className="table-cell" />
                </tr>
              </tfoot>
            </table>
          </div>
        ) : (
          <div className="flex items-center justify-center h-32 text-sm text-gray-600">
            No position Greeks data available
          </div>
        )}
      </div>

      {/* Circuit Breaker Status */}
      <div className="card">
        <p className="card-header mb-4">Circuit Breaker Status</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <CircuitBreaker
            label="Max Drawdown Limit"
            threshold="5%"
            current={risk?.current_drawdown != null ? `${Math.abs(risk.current_drawdown).toFixed(2)}%` : "--"}
            active={risk?.current_drawdown != null ? Math.abs(risk.current_drawdown) > 5 : false}
          />
          <CircuitBreaker
            label="VaR Limit"
            threshold="2% of equity"
            current={risk?.var_95 != null ? formatCompact(Math.abs(risk.var_95)) : "--"}
            active={false}
          />
          <CircuitBreaker
            label="Margin Utilization"
            threshold="80%"
            current={risk?.margin_utilization != null ? `${(risk.margin_utilization * 100).toFixed(1)}%` : "--"}
            active={risk?.margin_utilization != null ? risk.margin_utilization > 0.8 : false}
          />
        </div>
      </div>
    </div>
  );
}

// ---------- Sub-components ----------

function RiskMetricCard({
  label,
  value,
  color = "text-white",
  loading = false,
}: {
  label: string;
  value: string;
  color?: string;
  loading?: boolean;
}) {
  return (
    <div className={clsx("card", loading && "animate-pulse")}>
      <p className="text-[10px] text-gray-500 uppercase tracking-wider">
        {label}
      </p>
      <p className={`text-lg font-mono font-semibold mt-1 ${color}`}>
        {value}
      </p>
    </div>
  );
}

function MarginGauge({ utilization }: { utilization: number }) {
  const pct = Math.min(utilization * 100, 100);
  const color =
    pct > 80 ? "#ef4444" : pct > 50 ? "#eab308" : "#22c55e";

  return (
    <div className="flex flex-col items-center">
      {/* Circular gauge */}
      <div className="relative w-40 h-40">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          {/* Background circle */}
          <circle
            cx="60"
            cy="60"
            r="50"
            fill="none"
            stroke="#1e293b"
            strokeWidth="10"
          />
          {/* Value arc */}
          <circle
            cx="60"
            cy="60"
            r="50"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={`${pct * 3.14} 314`}
            className="transition-all duration-700"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <p className="text-2xl font-mono font-bold" style={{ color }}>
            {pct.toFixed(1)}%
          </p>
          <p className="text-[10px] text-gray-500 uppercase">Utilized</p>
        </div>
      </div>
    </div>
  );
}

function CircuitBreaker({
  label,
  threshold,
  current,
  active,
}: {
  label: string;
  threshold: string;
  current: string;
  active: boolean;
}) {
  return (
    <div
      className={clsx(
        "rounded-lg border p-3",
        active
          ? "bg-loss/10 border-loss/30"
          : "bg-gray-900 border-gray-800"
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-medium text-gray-400">{label}</p>
        <span
          className={clsx(
            "w-2.5 h-2.5 rounded-full",
            active ? "bg-loss animate-pulse" : "bg-profit"
          )}
        />
      </div>
      <div className="flex items-baseline justify-between">
        <div>
          <p className="text-[10px] text-gray-600">Current</p>
          <p className={clsx("text-sm font-mono font-medium", active ? "text-loss" : "text-white")}>
            {current}
          </p>
        </div>
        <div className="text-right">
          <p className="text-[10px] text-gray-600">Threshold</p>
          <p className="text-sm font-mono text-gray-500">{threshold}</p>
        </div>
      </div>
    </div>
  );
}
