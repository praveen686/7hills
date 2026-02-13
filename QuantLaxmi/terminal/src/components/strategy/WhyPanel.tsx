import { useCallback, useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { apiFetch } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface GateDecision {
  gate: string;
  passed: boolean;
  reason: string;
}

interface RiskSnapshot {
  drawdownPct: number;
  grossExposure: number;
  var95: number;
  positionCount: number;
}

interface TradeChainEvent {
  type: "signal" | "gate" | "order" | "fill";
  timestamp: string;
  label: string;
  detail: string;
}

export interface WhyPanelData {
  signalId: string;
  strategy: string;
  symbol: string;
  direction: "BUY" | "SELL";
  conviction: number;
  regime: string;
  timestamp: string;
  gates: GateDecision[];
  riskSnapshot: RiskSnapshot;
  tradeChain: TradeChainEvent[];
}

interface WhyPanelProps {
  data: WhyPanelData | null;
  /** If provided and no data, fetch from API */
  signalRef?: { strategyId: string; symbol: string; date: string } | null;
  open: boolean;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Demo data
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SignalHeader({ data }: { data: WhyPanelData }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "rounded px-2 py-0.5 text-xs font-bold",
            data.direction === "BUY"
              ? "bg-terminal-profit/20 text-terminal-profit"
              : "bg-terminal-loss/20 text-terminal-loss",
          )}
        >
          {data.direction}
        </span>
        <span className="font-mono text-sm font-semibold text-terminal-text">{data.symbol}</span>
        <span className="text-xs text-terminal-accent">{data.strategy}</span>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">Conviction</div>
          <div className="font-mono text-sm text-terminal-text">{(data.conviction * 100).toFixed(0)}%</div>
        </div>
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">Regime</div>
          <div className="font-mono text-sm text-terminal-text">{data.regime}</div>
        </div>
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">Time</div>
          <div className="font-mono text-sm text-terminal-text">{data.timestamp.split(" ")[1]}</div>
        </div>
      </div>
    </div>
  );
}

function GateDecisions({ gates }: { gates: GateDecision[] }) {
  return (
    <section>
      <h4 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
        Gate Decisions
      </h4>
      <div className="space-y-1">
        {gates.map((g) => (
          <div
            key={g.gate}
            className={cn(
              "flex items-center gap-2 rounded px-3 py-1.5 text-xs border",
              g.passed
                ? "border-terminal-border bg-terminal-surface"
                : "border-terminal-loss/30 bg-terminal-loss/5",
            )}
          >
            <span
              className={cn(
                "flex-shrink-0 h-4 w-4 rounded-full flex items-center justify-center text-2xs font-bold",
                g.passed
                  ? "bg-terminal-profit/20 text-terminal-profit"
                  : "bg-terminal-loss/20 text-terminal-loss",
              )}
            >
              {g.passed ? "\u2713" : "\u2717"}
            </span>
            <span className="font-medium text-terminal-text-secondary min-w-[120px]">{g.gate}</span>
            <span className="text-terminal-muted flex-1 truncate">{g.reason}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function RiskAtTime({ snapshot }: { snapshot: RiskSnapshot }) {
  return (
    <section>
      <h4 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
        Risk State at Signal Time
      </h4>
      <div className="grid grid-cols-2 gap-2">
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">Drawdown</div>
          <div className="font-mono text-xs text-terminal-loss">{snapshot.drawdownPct.toFixed(2)}%</div>
        </div>
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">Gross Exposure</div>
          <div className="font-mono text-xs text-terminal-text">{`₹${(snapshot.grossExposure / 100000).toFixed(1)}L`}</div>
        </div>
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">VaR 95%</div>
          <div className="font-mono text-xs text-terminal-warning">{`₹${(snapshot.var95 / 1000).toFixed(0)}K`}</div>
        </div>
        <div className="rounded bg-terminal-surface border border-terminal-border px-2 py-1.5">
          <div className="text-2xs text-terminal-muted">Positions</div>
          <div className="font-mono text-xs text-terminal-text">{snapshot.positionCount}</div>
        </div>
      </div>
    </section>
  );
}

function TradeChainTimeline({ events }: { events: TradeChainEvent[] }) {
  const typeConfig: Record<string, { color: string; bg: string }> = {
    signal: { color: "text-terminal-accent", bg: "bg-terminal-accent" },
    gate: { color: "text-terminal-warning", bg: "bg-terminal-warning" },
    order: { color: "text-terminal-info", bg: "bg-terminal-info" },
    fill: { color: "text-terminal-profit", bg: "bg-terminal-profit" },
  };

  return (
    <section>
      <h4 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
        Trade Chain
      </h4>
      <div className="relative space-y-0 pl-4">
        {/* Vertical line */}
        <div className="absolute left-[7px] top-2 bottom-2 w-px bg-terminal-border" />

        {events.map((ev, i) => {
          const cfg = typeConfig[ev.type] ?? typeConfig.signal;
          return (
            <div key={i} className="relative flex gap-3 py-2">
              {/* Dot */}
              <div className={cn("absolute left-[-13px] top-3 h-2.5 w-2.5 rounded-full border-2 border-terminal-bg", cfg.bg)} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-2xs text-terminal-muted">{ev.timestamp}</span>
                  <span className={cn("text-xs font-semibold", cfg.color)}>{ev.label}</span>
                </div>
                <p className="text-xs text-terminal-text-secondary mt-0.5 leading-relaxed">{ev.detail}</p>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function WhyPanel({ data, signalRef, open, onClose }: WhyPanelProps) {
  // Close on Escape
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose],
  );

  useEffect(() => {
    if (open) {
      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }
  }, [open, handleKeyDown]);

  const [fetchedData, setFetchedData] = useState<WhyPanelData | null>(null);

  useEffect(() => {
    if (data || !signalRef || !open) {
      setFetchedData(null);
      return;
    }
    apiFetch<{
      strategy_id: string;
      symbol: string;
      date: string;
      signals: Array<{ seq: number; ts: string; event_type: string; strategy_id: string; symbol: string; payload: any }>;
      gates: Array<{ seq: number; ts: string; event_type: string; strategy_id: string; symbol: string; payload: any }>;
      orders: Array<{ seq: number; ts: string; event_type: string; strategy_id: string; symbol: string; payload: any }>;
      fills: Array<{ seq: number; ts: string; event_type: string; strategy_id: string; symbol: string; payload: any }>;
      snapshot: { seq: number; ts: string; event_type: string; strategy_id: string; symbol: string; payload: any } | null;
    }>(`/api/why/trades/${signalRef.strategyId}/${signalRef.symbol}/${signalRef.date}`).then((chain) => {
      // Map to WhyPanelData
      const firstSignal = chain.signals[0]?.payload ?? {};
      const mapped: WhyPanelData = {
        signalId: String(chain.signals[0]?.seq ?? ""),
        strategy: chain.strategy_id,
        symbol: chain.symbol,
        direction: firstSignal.direction === "SELL" ? "SELL" : "BUY",
        conviction: firstSignal.conviction ?? 0,
        regime: firstSignal.regime ?? "",
        timestamp: chain.signals[0]?.ts ?? chain.date,
        gates: chain.gates.map((g) => ({
          gate: g.payload?.gate ?? g.event_type,
          passed: g.payload?.approved ?? true,
          reason: g.payload?.reason ?? "",
        })),
        riskSnapshot: {
          drawdownPct: chain.snapshot?.payload?.portfolio_dd ?? 0,
          grossExposure: chain.snapshot?.payload?.total_exposure ?? 0,
          var95: chain.snapshot?.payload?.var_95 ?? 0,
          positionCount: chain.snapshot?.payload?.position_count ?? 0,
        },
        tradeChain: [
          ...chain.signals.map((e) => ({ type: "signal" as const, timestamp: e.ts, label: "Signal Generated", detail: `${e.payload?.direction ?? ""} conviction=${(e.payload?.conviction ?? 0).toFixed(2)}` })),
          ...chain.gates.map((e) => ({ type: "gate" as const, timestamp: e.ts, label: e.payload?.gate ?? "Gate", detail: e.payload?.reason ?? (e.payload?.approved ? "Passed" : "Blocked") })),
          ...chain.orders.map((e) => ({ type: "order" as const, timestamp: e.ts, label: "Order Placed", detail: `${e.payload?.side ?? ""} ${e.payload?.quantity ?? ""} @ ${e.payload?.price ?? "market"}` })),
          ...chain.fills.map((e) => ({ type: "fill" as const, timestamp: e.ts, label: "Fill", detail: `${e.payload?.quantity ?? ""} @ ${e.payload?.fill_price ?? ""}` })),
        ].sort((a, b) => a.timestamp.localeCompare(b.timestamp)),
      };
      setFetchedData(mapped);
    }).catch(() => {});
  }, [data, signalRef, open]);

  if (!data && !fetchedData) {
    return (
      <>
        {open && <div className="fixed inset-0 z-40 bg-black/40 transition-opacity" onClick={onClose} />}
        <div className={cn(
          "fixed top-0 right-0 z-50 h-full w-[420px] max-w-[90vw] bg-terminal-panel border-l border-terminal-border shadow-2xl transition-transform duration-300",
          open ? "translate-x-0" : "translate-x-full",
        )}>
          <div className="flex items-center justify-between px-4 py-3 border-b border-terminal-border">
            <h3 className="text-xs font-bold uppercase tracking-widest text-terminal-text">Signal Explanation</h3>
            <button onClick={onClose} className="rounded p-1 hover:bg-terminal-surface text-terminal-muted hover:text-terminal-text transition-colors">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M12 4L4 12M4 4l8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>
            </button>
          </div>
          <div className="flex items-center justify-center h-[calc(100%-48px)] text-terminal-muted text-xs font-mono">
            Select a signal to see explanation
          </div>
        </div>
      </>
    );
  }

  const panelData = (data ?? fetchedData)!;

  return (
    <>
      {/* Backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/40 transition-opacity"
          onClick={onClose}
        />
      )}

      {/* Sliding panel */}
      <div
        className={cn(
          "fixed top-0 right-0 z-50 h-full w-[420px] max-w-[90vw] bg-terminal-panel border-l border-terminal-border shadow-2xl transition-transform duration-300",
          open ? "translate-x-0" : "translate-x-full",
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-terminal-border">
          <h3 className="text-xs font-bold uppercase tracking-widest text-terminal-text">
            Signal Explanation
          </h3>
          <button
            onClick={onClose}
            className="rounded p-1 hover:bg-terminal-surface text-terminal-muted hover:text-terminal-text transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M12 4L4 12M4 4l8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex flex-col gap-5 p-4 h-[calc(100%-48px)] overflow-y-auto">
          <SignalHeader data={panelData} />
          <GateDecisions gates={panelData.gates} />
          <RiskAtTime snapshot={panelData.riskSnapshot} />
          <TradeChainTimeline events={panelData.tradeChain} />

          {/* Signal ID footer */}
          <div className="mt-auto pt-4 border-t border-terminal-border">
            <span className="text-2xs text-terminal-muted font-mono">
              ID: {panelData.signalId}
            </span>
          </div>
        </div>
      </div>
    </>
  );
}
