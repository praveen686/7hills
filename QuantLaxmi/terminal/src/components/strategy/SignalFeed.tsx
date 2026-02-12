import { useCallback, useRef, useState } from "react";
import { useTauriStream } from "@/hooks/useTauriStream";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Signal {
  id: string;
  timestamp: number; // unix seconds
  strategy: string;
  symbol: string;
  direction: "BUY" | "SELL";
  conviction: number; // 0-1
  regime?: string;
}

interface SignalFeedProps {
  onSignalClick?: (signal: Signal) => void;
  maxItems?: number;
}

// ---------------------------------------------------------------------------
// Demo data
// ---------------------------------------------------------------------------

const DEMO_SIGNALS: Signal[] = [
  { id: "sig-1", timestamp: Date.now() / 1000 - 10, strategy: "S25 DFF", symbol: "NIFTY", direction: "BUY", conviction: 0.82, regime: "trending" },
  { id: "sig-2", timestamp: Date.now() / 1000 - 45, strategy: "S4 IV-MR", symbol: "BANKNIFTY", direction: "SELL", conviction: 0.67, regime: "mean-rev" },
  { id: "sig-3", timestamp: Date.now() / 1000 - 120, strategy: "S5 Hawkes", symbol: "NIFTY", direction: "BUY", conviction: 0.91, regime: "jump" },
  { id: "sig-4", timestamp: Date.now() / 1000 - 300, strategy: "S1 VRP", symbol: "BANKNIFTY", direction: "SELL", conviction: 0.54, regime: "vol-crush" },
  { id: "sig-5", timestamp: Date.now() / 1000 - 600, strategy: "S11 Pairs", symbol: "RELIANCE", direction: "BUY", conviction: 0.73 },
  { id: "sig-6", timestamp: Date.now() / 1000 - 900, strategy: "S25 DFF", symbol: "BANKNIFTY", direction: "BUY", conviction: 0.78, regime: "trending" },
  { id: "sig-7", timestamp: Date.now() / 1000 - 1800, strategy: "S6 Multi-Factor", symbol: "NIFTY", direction: "SELL", conviction: 0.61 },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString("en-IN", {
    timeZone: "Asia/Kolkata",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function timeAgo(ts: number): string {
  const diff = Math.floor(Date.now() / 1000 - ts);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function DirectionBadge({ direction }: { direction: "BUY" | "SELL" }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded px-1.5 py-0.5 text-2xs font-bold tracking-wider",
        direction === "BUY"
          ? "bg-terminal-profit/20 text-terminal-profit"
          : "bg-terminal-loss/20 text-terminal-loss",
      )}
    >
      {direction}
    </span>
  );
}

function ConvictionBar({ conviction }: { conviction: number }) {
  const pct = Math.round(conviction * 100);
  const color =
    conviction >= 0.8
      ? "bg-terminal-profit"
      : conviction >= 0.6
        ? "bg-terminal-accent"
        : "bg-terminal-warning";

  return (
    <div className="flex items-center gap-1.5 min-w-[80px]">
      <div className="flex-1 h-1.5 rounded-full bg-terminal-surface overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all duration-300", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-2xs font-mono text-terminal-muted w-7 text-right">{pct}%</span>
    </div>
  );
}

function SignalRow({
  signal,
  isNew,
  onClick,
}: {
  signal: Signal;
  isNew: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-3 w-full rounded px-3 py-2 text-left transition-all hover:bg-terminal-surface focus:outline-none focus:ring-1 focus:ring-terminal-accent/40",
        isNew && "animate-flash-green",
      )}
    >
      {/* Time */}
      <div className="flex flex-col items-end min-w-[52px]">
        <span className="text-2xs font-mono text-gray-300">{formatTime(signal.timestamp)}</span>
        <span className="text-2xs font-mono text-terminal-muted">{timeAgo(signal.timestamp)}</span>
      </div>

      {/* Strategy */}
      <span className="text-xs text-terminal-accent font-medium min-w-[80px] truncate">
        {signal.strategy}
      </span>

      {/* Symbol */}
      <span className="text-xs font-mono text-gray-100 min-w-[72px]">{signal.symbol}</span>

      {/* Direction badge */}
      <DirectionBadge direction={signal.direction} />

      {/* Conviction bar */}
      <ConvictionBar conviction={signal.conviction} />

      {/* Regime */}
      {signal.regime && (
        <span className="text-2xs text-terminal-muted bg-terminal-surface rounded px-1.5 py-0.5">
          {signal.regime}
        </span>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Filter controls
// ---------------------------------------------------------------------------

type DirectionFilter = "ALL" | "BUY" | "SELL";

function FilterBar({
  strategies,
  selectedStrategy,
  onStrategyChange,
  directionFilter,
  onDirectionChange,
}: {
  strategies: string[];
  selectedStrategy: string;
  onStrategyChange: (s: string) => void;
  directionFilter: DirectionFilter;
  onDirectionChange: (d: DirectionFilter) => void;
}) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 border-b border-terminal-border">
      <select
        value={selectedStrategy}
        onChange={(e) => onStrategyChange(e.target.value)}
        className="rounded bg-terminal-surface border border-terminal-border px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-terminal-accent"
      >
        <option value="ALL">All Strategies</option>
        {strategies.map((s) => (
          <option key={s} value={s}>{s}</option>
        ))}
      </select>

      <div className="flex rounded overflow-hidden border border-terminal-border">
        {(["ALL", "BUY", "SELL"] as DirectionFilter[]).map((d) => (
          <button
            key={d}
            onClick={() => onDirectionChange(d)}
            className={cn(
              "px-2 py-1 text-2xs font-semibold transition-colors",
              directionFilter === d
                ? "bg-terminal-accent text-white"
                : "bg-terminal-surface text-terminal-muted hover:text-gray-200",
            )}
          >
            {d}
          </button>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function SignalFeed({ onSignalClick, maxItems = 50 }: SignalFeedProps) {
  const [signals, setSignals] = useState<Signal[]>(DEMO_SIGNALS);
  const [strategyFilter, setStrategyFilter] = useState("ALL");
  const [directionFilter, setDirectionFilter] = useState<DirectionFilter>("ALL");
  const newIdsRef = useRef<Set<string>>(new Set());

  // Listen for real-time signals
  useTauriStream<Signal>("signal", (signal) => {
    newIdsRef.current.add(signal.id);
    setSignals((prev) => [signal, ...prev].slice(0, maxItems));
    // Clear "new" status after animation
    setTimeout(() => {
      newIdsRef.current.delete(signal.id);
    }, 500);
  });

  const handleClick = useCallback(
    (signal: Signal) => {
      onSignalClick?.(signal);
    },
    [onSignalClick],
  );

  // Unique strategies for filter
  const strategies = [...new Set(signals.map((s) => s.strategy))];

  // Filtered signals
  const filtered = signals.filter((s) => {
    if (strategyFilter !== "ALL" && s.strategy !== strategyFilter) return false;
    if (directionFilter !== "ALL" && s.direction !== directionFilter) return false;
    return true;
  });

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 pt-3 pb-1">
        <h2 className="text-sm font-bold uppercase tracking-widest text-gray-100">
          Signal Feed
        </h2>
        <span className="text-2xs font-mono text-terminal-muted">{filtered.length} signals</span>
      </div>

      <FilterBar
        strategies={strategies}
        selectedStrategy={strategyFilter}
        onStrategyChange={setStrategyFilter}
        directionFilter={directionFilter}
        onDirectionChange={setDirectionFilter}
      />

      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-terminal-muted text-xs">
            No signals matching filters
          </div>
        ) : (
          <div className="divide-y divide-terminal-border/50">
            {filtered.map((signal) => (
              <SignalRow
                key={signal.id}
                signal={signal}
                isNew={newIdsRef.current.has(signal.id)}
                onClick={() => handleClick(signal)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
