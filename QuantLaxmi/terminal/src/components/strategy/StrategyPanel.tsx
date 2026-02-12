import { useEffect, useState, useCallback } from "react";
import { useTauriCommand } from "@/hooks/useTauriCommand";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type StrategyStatus = "live" | "paused" | "stopped";

export interface StrategyInfo {
  id: string;
  name: string;
  status: StrategyStatus;
  equity: number;
  returnPct: number;
  sharpe: number;
  maxDD: number;
  winRate: number;
  /** Equity curve data points for sparkline (normalized 0-1) */
  equityCurve: number[];
}

interface StrategyPanelProps {
  onSelect?: (strategyId: string) => void;
}

// ---------------------------------------------------------------------------
// Default data
// ---------------------------------------------------------------------------

function makeSparkline(trend: number, volatility: number): number[] {
  const pts: number[] = [];
  let v = 0.3;
  for (let i = 0; i < 30; i++) {
    v += trend * 0.01 + (Math.random() - 0.5) * volatility;
    v = Math.max(0, Math.min(1, v));
    pts.push(v);
  }
  // Normalize to 0-1
  const min = Math.min(...pts);
  const max = Math.max(...pts);
  const range = max - min || 1;
  return pts.map((p) => (p - min) / range);
}

const DEFAULT_STRATEGIES: StrategyInfo[] = [
  { id: "s1", name: "S1 VRP Options", status: "live", equity: 10559000, returnPct: 5.59, sharpe: 2.14, maxDD: 1.82, winRate: 0.68, equityCurve: makeSparkline(1.5, 0.06) },
  { id: "s4", name: "S4 IV Mean-Rev", status: "live", equity: 10307000, returnPct: 3.07, sharpe: 1.85, maxDD: 2.14, winRate: 0.62, equityCurve: makeSparkline(1.2, 0.05) },
  { id: "s5", name: "S5 Hawkes Jump", status: "paused", equity: 10429000, returnPct: 4.29, sharpe: 3.07, maxDD: 0.98, winRate: 0.71, equityCurve: makeSparkline(2.0, 0.04) },
  { id: "s8", name: "S8 Expiry Theta", status: "stopped", equity: 9985000, returnPct: -0.15, sharpe: 0.12, maxDD: 3.47, winRate: 0.51, equityCurve: makeSparkline(-0.2, 0.08) },
  { id: "s11", name: "S11 Stat Pairs", status: "live", equity: 10182000, returnPct: 1.82, sharpe: 1.43, maxDD: 1.56, winRate: 0.59, equityCurve: makeSparkline(0.8, 0.05) },
  { id: "s25", name: "S25 DFF", status: "live", equity: 10609000, returnPct: 6.09, sharpe: 1.87, maxDD: 2.05, winRate: 0.64, equityCurve: makeSparkline(1.8, 0.06) },
  { id: "s26", name: "S26 Crypto Flow", status: "paused", equity: 10215000, returnPct: 2.15, sharpe: 1.52, maxDD: 2.88, winRate: 0.57, equityCurve: makeSparkline(0.9, 0.07) },
  { id: "s6", name: "S6 Multi-Factor", status: "live", equity: 10340000, returnPct: 3.40, sharpe: 1.68, maxDD: 1.72, winRate: 0.61, equityCurve: makeSparkline(1.3, 0.05) },
];

// ---------------------------------------------------------------------------
// Sparkline SVG
// ---------------------------------------------------------------------------

function Sparkline({ data, positive }: { data: number[]; positive: boolean }) {
  const w = 80;
  const h = 24;
  const n = data.length;
  if (n < 2) return null;

  const points = data
    .map((v, i) => `${(i / (n - 1)) * w},${h - v * h}`)
    .join(" ");

  const fillPoints = `0,${h} ${points} ${w},${h}`;
  const strokeColor = positive ? "#00d4aa" : "#ff4d6a";
  const fillColor = positive ? "rgba(0,212,170,0.15)" : "rgba(255,77,106,0.15)";

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="flex-shrink-0">
      <polygon points={fillPoints} fill={fillColor} />
      <polyline
        points={points}
        fill="none"
        stroke={strokeColor}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: StrategyStatus }) {
  const config = {
    live: { label: "LIVE", bg: "bg-terminal-profit/20", text: "text-terminal-profit", dot: "bg-terminal-profit" },
    paused: { label: "PAUSED", bg: "bg-terminal-warning/20", text: "text-terminal-warning", dot: "bg-terminal-warning" },
    stopped: { label: "STOPPED", bg: "bg-terminal-loss/20", text: "text-terminal-loss", dot: "bg-terminal-loss" },
  }[status];

  return (
    <span className={cn("inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-2xs font-semibold", config.bg, config.text)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", config.dot, status === "live" && "animate-pulse")} />
      {config.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Strategy Card
// ---------------------------------------------------------------------------

function StrategyCard({
  strategy,
  onClick,
}: {
  strategy: StrategyInfo;
  onClick: () => void;
}) {
  const positive = strategy.returnPct >= 0;
  return (
    <button
      onClick={onClick}
      className="flex flex-col gap-2 rounded-lg border border-terminal-border bg-terminal-panel p-3 text-left transition-all hover:border-terminal-accent/40 hover:bg-terminal-surface focus:outline-none focus:ring-1 focus:ring-terminal-accent/50"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-100 truncate">{strategy.name}</span>
        <StatusBadge status={strategy.status} />
      </div>

      {/* Sparkline + Return */}
      <div className="flex items-center justify-between gap-2">
        <Sparkline data={strategy.equityCurve} positive={positive} />
        <span
          className={cn(
            "font-mono text-sm font-bold",
            positive ? "text-terminal-profit" : "text-terminal-loss",
          )}
        >
          {positive ? "+" : ""}
          {strategy.returnPct.toFixed(2)}%
        </span>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-x-3 gap-y-0.5 text-2xs">
        <div className="flex justify-between">
          <span className="text-terminal-muted">Sharpe</span>
          <span className="font-mono text-gray-200">{strategy.sharpe.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-muted">MaxDD</span>
          <span className="font-mono text-terminal-loss">{strategy.maxDD.toFixed(1)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-muted">WinR</span>
          <span className="font-mono text-gray-200">{(strategy.winRate * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Equity */}
      <div className="text-2xs text-terminal-muted">
        Equity: <span className="font-mono text-gray-300">{`₹${(strategy.equity / 100000).toFixed(1)}L`}</span>
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function StrategyPanel({ onSelect }: StrategyPanelProps) {
  const { data, execute } = useTauriCommand<StrategyInfo[]>("get_strategies");
  const [strategies, setStrategies] = useState<StrategyInfo[]>(DEFAULT_STRATEGIES);

  useEffect(() => {
    execute().catch(() => {});
    const interval = setInterval(() => execute().catch(() => {}), 5000);
    return () => clearInterval(interval);
  }, [execute]);

  useEffect(() => {
    if (data) setStrategies(data);
  }, [data]);

  const handleSelect = useCallback(
    (id: string) => {
      onSelect?.(id);
    },
    [onSelect],
  );

  return (
    <div className="flex flex-col gap-3 p-4 h-full overflow-y-auto">
      <h2 className="text-sm font-bold uppercase tracking-widest text-gray-100">
        Strategies
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {strategies.map((s) => (
          <StrategyCard key={s.id} strategy={s} onClick={() => handleSelect(s.id)} />
        ))}
      </div>
    </div>
  );
}
