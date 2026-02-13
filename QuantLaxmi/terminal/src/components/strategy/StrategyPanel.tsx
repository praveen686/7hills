import { useEffect, useState, useCallback } from "react";
import { apiFetch } from "@/lib/api";
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
  const strokeColor = positive ? "rgb(var(--terminal-profit))" : "rgb(var(--terminal-loss))";
  const fillColor = positive ? "rgb(var(--terminal-profit) / 0.15)" : "rgb(var(--terminal-loss) / 0.15)";

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

function StatusBadge({ status }: { status: string }) {
  const configs: Record<string, { label: string; bg: string; text: string; dot: string }> = {
    live: { label: "LIVE", bg: "bg-terminal-profit/20", text: "text-terminal-profit", dot: "bg-terminal-profit" },
    active: { label: "ACTIVE", bg: "bg-terminal-profit/20", text: "text-terminal-profit", dot: "bg-terminal-profit" },
    paused: { label: "PAUSED", bg: "bg-terminal-warning/20", text: "text-terminal-warning", dot: "bg-terminal-warning" },
    marginal: { label: "MARGINAL", bg: "bg-terminal-warning/20", text: "text-terminal-warning", dot: "bg-terminal-warning" },
    research: { label: "RESEARCH", bg: "bg-blue-500/20", text: "text-blue-400", dot: "bg-blue-400" },
    negative: { label: "NEGATIVE", bg: "bg-terminal-loss/20", text: "text-terminal-loss", dot: "bg-terminal-loss" },
    inactive: { label: "INACTIVE", bg: "bg-terminal-muted/20", text: "text-terminal-muted", dot: "bg-terminal-muted" },
    stopped: { label: "STOPPED", bg: "bg-terminal-loss/20", text: "text-terminal-loss", dot: "bg-terminal-loss" },
  };
  const config = configs[status] ?? configs.stopped;

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
  const ret = strategy.returnPct ?? (strategy as any).return_pct ?? 0;
  const sharpe = strategy.sharpe ?? 0;
  const maxDD = strategy.maxDD ?? (strategy as any).max_dd ?? 0;
  const winRate = strategy.winRate ?? (strategy as any).win_rate ?? 0;
  const equity = strategy.equity ?? 0;
  const curve = Array.isArray(strategy.equityCurve) ? strategy.equityCurve : [];
  const positive = ret >= 0;
  return (
    <button
      onClick={onClick}
      className="flex flex-col gap-2 rounded-lg border border-terminal-border bg-terminal-panel p-3 text-left transition-all hover:border-terminal-accent/40 hover:bg-terminal-surface focus:outline-none focus:ring-1 focus:ring-terminal-accent/50"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-terminal-text truncate font-mono">{strategy.id}</span>
        <StatusBadge status={strategy.status ?? "stopped"} />
      </div>

      {/* Sparkline + Return */}
      <div className="flex items-center justify-between gap-2">
        <Sparkline data={curve} positive={positive} />
        <span
          className={cn(
            "font-mono text-sm font-bold",
            positive ? "text-terminal-profit" : "text-terminal-loss",
          )}
        >
          {positive ? "+" : ""}
          {ret.toFixed(2)}%
        </span>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-x-3 gap-y-0.5 text-2xs">
        <div className="flex justify-between">
          <span className="text-terminal-muted">Sharpe</span>
          <span className="font-mono text-terminal-text-secondary">{sharpe.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-muted">MaxDD</span>
          <span className="font-mono text-terminal-loss">{maxDD.toFixed(1)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-muted">WinR</span>
          <span className="font-mono text-terminal-text-secondary">{(winRate * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Equity */}
      <div className="text-2xs text-terminal-muted">
        Equity: <span className="font-mono text-terminal-text-secondary">{equity > 1000 ? `₹${(equity / 100000).toFixed(1)}L` : `${equity.toFixed(4)}`}</span>
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function StrategyPanel({ onSelect }: StrategyPanelProps) {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStrategies = useCallback(() => {
    setLoading(true);
    setError(null);
    apiFetch<{ count: number; strategies: any[] }>("/api/strategies")
      .then((data) => {
        const raw = data.strategies ?? [];
        const list: StrategyInfo[] = raw.map((s: any) => ({
          id: s.strategy_id ?? s.id ?? "",
          name: s.name ?? "",
          status: s.status ?? "stopped",
          equity: s.equity ?? 0,
          returnPct: s.return_pct ?? s.returnPct ?? 0,
          sharpe: s.sharpe ?? 0,
          maxDD: s.max_dd ?? s.maxDD ?? 0,
          winRate: (s.win_rate ?? s.winRate ?? 0) / 100,
          equityCurve: [],
        }));
        setStrategies(list);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load strategies");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchStrategies();
    const interval = setInterval(fetchStrategies, 30000);
    return () => clearInterval(interval);
  }, [fetchStrategies]);

  const handleSelect = useCallback(
    (id: string) => {
      onSelect?.(id);
    },
    [onSelect],
  );

  return (
    <div className="flex flex-col gap-3 p-4 h-full overflow-y-auto">
      <h2 className="text-sm font-bold uppercase tracking-widest text-terminal-text">
        Strategies
      </h2>
      {loading && strategies.length === 0 ? (
        <div className="flex items-center justify-center h-32 text-terminal-muted text-xs font-mono animate-pulse">
          Loading strategies...
        </div>
      ) : error && strategies.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-32 gap-3">
          <p className="text-xs text-terminal-loss">{error}</p>
          <button
            onClick={fetchStrategies}
            className="text-xs text-terminal-accent hover:text-terminal-accent-dim underline"
          >
            Retry
          </button>
        </div>
      ) : strategies.length === 0 ? (
        <div className="flex items-center justify-center h-32 text-terminal-muted text-xs font-mono">
          No strategies configured
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {strategies.map((s) => (
            <StrategyCard key={s.id} strategy={s} onClick={() => handleSelect(s.id)} />
          ))}
        </div>
      )}
    </div>
  );
}
