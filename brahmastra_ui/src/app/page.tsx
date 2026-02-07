"use client";

import { usePortfolio, useStrategies, useTodaySignals, useVIX } from "@/hooks/usePortfolio";
import { useWhyPanel } from "@/hooks/useWhyPanel";
import { EquityChart } from "@/components/charts/EquityChart";
import { SignalTable } from "@/components/charts/SignalTable";
import { StrategyCard } from "@/components/charts/StrategyCard";
import { WhyPanel } from "@/components/WhyPanel";
import {
  formatCurrency,
  formatCompact,
  formatPct,
  formatPnl,
  pnlColor,
  pnlBg,
} from "@/lib/formatters";
import { clsx } from "clsx";

// ============================================================
// Dashboard Page
// ============================================================

export default function DashboardPage() {
  const { data: portfolio, isLoading: portfolioLoading } = usePortfolio();
  const { data: strategies } = useStrategies();
  const { data: signals } = useTodaySignals();
  const { data: vix } = useVIX();
  const whyPanel = useWhyPanel();

  return (
    <div className="space-y-6 animate-in">
      {/* Page Title */}
      <div>
        <h1 className="text-xl font-semibold text-white">Dashboard</h1>
        <p className="text-sm text-gray-500 mt-1">
          Portfolio overview and real-time signals
        </p>
      </div>

      {/* Top Cards Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Portfolio Equity */}
        <div className={clsx("card", portfolioLoading && "animate-pulse")}>
          <p className="card-header">Total Equity</p>
          <p className="card-value text-white">
            {portfolio ? formatCurrency(portfolio.total_equity ?? 0) : "--"}
          </p>
          {portfolio && (
            <p className={`text-sm font-mono mt-1 ${pnlColor(portfolio.total_pnl)}`}>
              {formatPnl(portfolio.total_pnl)} ({formatPct(portfolio.total_pnl_pct)})
            </p>
          )}
        </div>

        {/* Day P&L */}
        <div
          className={clsx(
            "card",
            portfolio && pnlBg(portfolio.day_pnl),
            portfolio && ((portfolio.day_pnl ?? 0) > 0 ? "glow-profit" : (portfolio.day_pnl ?? 0) < 0 ? "glow-loss" : "")
          )}
        >
          <p className="card-header">Day P&L</p>
          <p className={`card-value ${portfolio ? pnlColor(portfolio.day_pnl) : ""}`}>
            {portfolio ? formatPnl(portfolio.day_pnl) : "--"}
          </p>
          {portfolio && (
            <p className={`text-sm font-mono mt-1 ${pnlColor(portfolio.day_pnl_pct)}`}>
              {formatPct(portfolio.day_pnl_pct)}
            </p>
          )}
        </div>

        {/* Margin */}
        <div className="card">
          <p className="card-header">Margin Used</p>
          <p className="card-value text-white">
            {portfolio ? formatCompact(portfolio.margin_used) : "--"}
          </p>
          {portfolio && (
            <div className="mt-2">
              <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className={clsx(
                    "h-full rounded-full transition-all",
                    (portfolio.margin_used ?? 0) / ((portfolio.margin_used ?? 0) + (portfolio.margin_available ?? 1)) > 0.8
                      ? "bg-loss"
                      : (portfolio.margin_used ?? 0) / ((portfolio.margin_used ?? 0) + (portfolio.margin_available ?? 1)) > 0.5
                        ? "bg-yellow-500"
                        : "bg-accent"
                  )}
                  style={{
                    width: `${Math.min(
                      ((portfolio.margin_used ?? 0) /
                        ((portfolio.margin_used ?? 0) + (portfolio.margin_available ?? 1))) *
                        100,
                      100
                    )}%`,
                  }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1 font-mono">
                {formatCompact(portfolio.margin_available)} available
              </p>
            </div>
          )}
        </div>

        {/* VIX Gauge */}
        <div className="card">
          <p className="card-header">India VIX</p>
          <div className="flex items-end gap-3">
            <VIXGauge value={vix?.value ?? 0} />
            <div>
              <p className="card-value text-white">
                {vix ? vix.value.toFixed(2) : "--"}
              </p>
              {vix && (
                <p className={`text-xs font-mono ${pnlColor(-vix.change)}`}>
                  {formatPct(vix.change_pct)}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Equity Chart */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="card-header mb-0">Portfolio Equity Curve</p>
          </div>
          <div className="flex gap-2">
            <button className="px-3 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded text-gray-400 transition-colors">
              1W
            </button>
            <button className="px-3 py-1 text-xs bg-accent/20 text-accent rounded">
              1M
            </button>
            <button className="px-3 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded text-gray-400 transition-colors">
              3M
            </button>
            <button className="px-3 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded text-gray-400 transition-colors">
              ALL
            </button>
          </div>
        </div>
        <EquityChart data={portfolio?.equity_curve ?? []} height={280} />
      </div>

      {/* Strategy Cards + Signals Table */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Strategy P&L Cards */}
        <div className="lg:col-span-1 space-y-3">
          <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
            Active Strategies
          </h2>
          {strategies && strategies.length > 0 ? (
            strategies
              .filter((s) => ["live", "running", "active"].includes(s.status ?? ""))
              .slice(0, 4)
              .map((strategy) => (
                <StrategyCard key={strategy.strategy_id ?? strategy.id} strategy={strategy} compact />
              ))
          ) : (
            <div className="card text-center py-8">
              <p className="text-sm text-gray-600">No active strategies</p>
              <p className="text-xs text-gray-700 mt-1">
                Deploy a strategy to see it here
              </p>
            </div>
          )}
        </div>

        {/* Today's Signals */}
        <div className="lg:col-span-2">
          <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">
            Today&apos;s Signals
          </h2>
          <div className="card p-0 overflow-hidden">
            <SignalTable
              signals={signals ?? []}
              maxRows={10}
              onSignalClick={whyPanel.open}
            />
          </div>
        </div>
      </div>

      {/* Why Panel (slide-out from right) */}
      <WhyPanel
        isOpen={whyPanel.isOpen}
        chain={whyPanel.chain}
        isLoading={whyPanel.isLoading}
        error={whyPanel.error}
        onClose={whyPanel.close}
      />
    </div>
  );
}

// ---------- VIX Gauge Component ----------

function VIXGauge({ value }: { value: number }) {
  // VIX typically 10-40; normalize to 0-100
  const normalized = Math.min(Math.max((value - 10) / 30, 0), 1);
  const angle = normalized * 180 - 90; // -90 to 90 degrees
  const color =
    value > 25
      ? "#ef4444"
      : value > 18
        ? "#eab308"
        : "#22c55e";

  return (
    <div className="relative w-16 h-10">
      <svg viewBox="0 0 100 60" className="w-full h-full">
        {/* Background arc */}
        <path
          d="M 10 55 A 40 40 0 0 1 90 55"
          fill="none"
          stroke="#1e293b"
          strokeWidth="6"
          strokeLinecap="round"
        />
        {/* Value arc */}
        <path
          d="M 10 55 A 40 40 0 0 1 90 55"
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={`${normalized * 126} 126`}
          className="transition-all duration-500"
        />
        {/* Needle */}
        <line
          x1="50"
          y1="55"
          x2={50 + 30 * Math.cos((angle * Math.PI) / 180)}
          y2={55 - 30 * Math.sin((-angle * Math.PI) / 180)}
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          className="transition-all duration-500"
        />
        <circle cx="50" cy="55" r="3" fill={color} />
      </svg>
    </div>
  );
}
